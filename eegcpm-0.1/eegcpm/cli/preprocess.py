"""Preprocess command for EEGCPM CLI."""

from pathlib import Path
from typing import List
from datetime import datetime
import yaml
import mne
from pydantic import ValidationError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from ..workflow.state import (
    WorkflowStateManager,
    WorkflowState,
    ProcessingStatus,
    StepRecord,
    _make_json_serializable,
)
from ..core.config import load_preprocessing_spec
from ..core.config_hash import resolved_config_hash
from ..core.paths import EEGCPMPaths, clear_run_outputs, fail_run_outputs
from ..modules.preprocessing import PreprocessingPipeline
from ..modules.qc.preprocessed_qc import PreprocessedQC
from ..modules.qc.hbn_recording_qc import recording_qc_metrics
from ..data.event_mapping import get_event_mapping_for_run, translate_event_codes
from ..core.task_config import TaskConfig
from ..modules.qc.quality_assessment import extract_metrics_from_qc_result
from ..modules.qc.metrics_io import save_qc_metrics_json
from ..data.bids_utils import supported_formats, find_eeg_run_files, read_raw_eeg


def _write_run_qc_metrics(
    *,
    run_output: Path,
    subject_id: str,
    session: str,
    task: str,
    run_num: str,
    status: str,
    reason: str,
    extra: dict | None = None,
) -> Path:
    """Write ``..._qc_metrics.json`` for a FAILED/EXCLUDED run (P1-F10).

    A recording that never produced a FIF must still leave a QC record: without
    it a downstream QC table cannot distinguish "excluded for contamination"
    from "not processed yet". The payload is deliberately minimal (status +
    reason + run identity, no HTML) and goes through the sidecar encoder so
    numpy scalars and non-finite values are safe.
    """
    payload = {
        "subject_id": subject_id,
        "session": session,
        "task": task,
        "run": run_num,
        "status": status,
        "reason": reason,
        "created_utc": datetime.now().isoformat(),
    }
    if extra:
        payload.update(extra)
    json_path = (
        Path(run_output)
        / f"{subject_id}_ses-{session}_task-{task}_run-{run_num}_qc_metrics.json"
    )
    save_qc_metrics_json(_make_json_serializable(payload), json_path)
    return json_path


def _subject_status(run_statuses: List[ProcessingStatus]) -> ProcessingStatus:
    """Aggregate per-run statuses (rule: failed > excluded > completed).

    Raises
    ------
    ValueError
        If no run status is supplied — an empty subject has no status,
        and silently returning COMPLETED would fake a successful batch.
    """
    if not run_statuses:
        raise ValueError("_subject_status() requires at least one run status")
    if all(s == ProcessingStatus.COMPLETED for s in run_statuses):
        return ProcessingStatus.COMPLETED
    if any(
        s not in (ProcessingStatus.COMPLETED, ProcessingStatus.EXCLUDED) for s in run_statuses
    ):
        return ProcessingStatus.FAILED
    return ProcessingStatus.EXCLUDED


def preprocess_command(args):
    """
    Run preprocessing pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    console = Console()

    # Load config
    if not args.config.exists():
        console.print(
            f"[red]Error: Config file not found: {args.config}[/red]")
        return 1

    with open(args.config, 'r') as f:
        raw_config = yaml.safe_load(f)

    # One schema, `extra='forbid'`: the legacy flat and keyed dialects
    # are hard errors, never silent no-ops.
    try:
        spec = load_preprocessing_spec(raw_config)
    except ValidationError as e:
        console.print(
            f"[red]Error: invalid preprocessing config {args.config}[/red]")
        console.print(str(e))
        return 1

    # Setup paths using new centralized system
    paths = EEGCPMPaths(
        project_root=args.project,
        eegcpm_root=args.eegcpm_root if hasattr(
            args, 'eegcpm_root') and args.eegcpm_root else None
    )

    # Setup state manager using new paths
    db_path = paths.get_state_db()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    manager = WorkflowStateManager(db_path)

    # Get subject list
    subjects = _get_subject_list(args, console, paths)
    if not subjects:
        console.print("[yellow]No subjects to process.[/yellow]")
        return

    # Track batch results
    batch_completed = 0
    batch_failed = 0
    batch_skipped = 0
    batch_excluded = 0

    # Get task
    task = args.task or spec.task or 'unknown'

    # NEW: Output uses stage-first architecture
    # derivatives/preprocessing/{pipeline}/{subject}/ses-{session}/task-{task}/run-{run}/
    preprocessing_root = paths.get_preprocessing_root(args.pipeline)
    preprocessing_root.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]EEGCPM Preprocessing[/bold]")
    console.print(f"Config: {args.config}")
    console.print(f"Project: {args.project}")
    console.print(f"Pipeline: {args.pipeline}")
    console.print(f"Task: {task}")
    console.print(f"Subjects: {len(subjects)}")
    console.print(f"Output: {preprocessing_root}")
    console.print(f"State DB: {db_path}\n")

    # R3 (ASR scoped audit, 2026-10-16): the resolved-config hash of the
    # pipeline this invocation would run. A stored COMPLETED/EXCLUDED row is
    # only a resume hit when its recorded ``config_hash`` equals this value —
    # otherwise the result was produced by a different pipeline configuration
    # (e.g. different ``chain_params``) and is stale. Built once: ``get_config``
    # does not depend on the output directory, so the value is identical for
    # every run in the batch and equals the hash recorded per run below. A
    # construction failure (e.g. dependency preflight) is a global
    # configuration/environment error, not a per-run data failure, so the
    # command fails loudly with a non-zero exit code instead of marking every
    # run FAILED.
    try:
        current_config_hash = resolved_config_hash(
            PreprocessingPipeline(spec.pipeline_steps()))
    except Exception as e:
        console.print(
            f"[red]Error: cannot build the preprocessing pipeline from "
            f"{args.config}: {e}[/red]")
        return 1

    # Process subjects
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:

        overall_task = progress.add_task(
            f"[cyan]Processing {len(subjects)} subjects...",
            total=len(subjects)
        )

        for i, subject_id in enumerate(subjects, 1):
            progress.update(
                overall_task,
                description=f"[cyan]Subject {i}/{len(subjects)}: {subject_id}"
            )

            # Per-run final statuses (one WorkflowState per run).
            run_statuses = []
            workflow = None

            try:
                # Find input files (may be multiple runs)
                input_files_response = _find_input_files(
                    args.project, subject_id, task)
                print(input_files_response)
                input_files = input_files_response.get("files", None)
                if not input_files:
                    console.print(
                        f"  [yellow]⚠ {subject_id}: No input files found for task {task}[/yellow]")
                    workflow = WorkflowState(
                        subject_id=subject_id,
                        task=task,
                        pipeline=args.pipeline,
                        status=ProcessingStatus.FAILED,
                        created_at=datetime.now()
                    )
                    workflow.add_step(StepRecord(
                        step_name="load_data",
                        status=ProcessingStatus.FAILED,
                        error_message=f"No input files found for task {task}"
                    ))
                    manager.save_state(workflow)
                    batch_failed += 1
                    progress.update(overall_task, advance=1)
                    continue

                console.print(
                    f"  Found {len(input_files)} run(s) for {subject_id}")

                # Process each run
                for run_idx, input_file in enumerate(input_files, 1):
                    # Extract run number and session from filename
                    run_num = input_file.stem.split(
                        '_run-')[1].split('_')[0] if '_run-' in input_file.stem else str(run_idx)
                    # Extract session (default to "01")
                    session = input_file.stem.split(
                        '_ses-')[1].split('_')[0] if '_ses-' in input_file.stem else "01"

                    # Skip runs already completed or excluded for this
                    # exact identity (subject/session/task/run) — but only when
                    # the stored result was produced by the *same* resolved
                    # pipeline config (R3, ASR scoped audit 2026-10-16). A
                    # config change (e.g. different ``chain_params``) must
                    # reprocess: silently reusing a stale result would label
                    # outputs with the new config's hash while the data came
                    # from the old pipeline.
                    existing_state = manager.load_state(
                        subject_id, task, args.pipeline,
                        session=session, run=run_num)
                    if (existing_state
                            and existing_state.status in (ProcessingStatus.COMPLETED,
                                                          ProcessingStatus.EXCLUDED)
                            and not args.force):
                        stored_hash = existing_state.config_hash
                        if stored_hash == current_config_hash:
                            console.print(
                                f"    [dim]✓ run {run_num} already "
                                f"{existing_state.status.value} "
                                f"(use --force to reprocess)[/dim]")
                            batch_skipped += 1
                            continue
                        console.print(
                            f"    [yellow]↻ run {run_num} was "
                            f"{existing_state.status.value} under a different "
                            f"pipeline config (stored "
                            f"{stored_hash or 'no config_hash'}, current "
                            f"{current_config_hash[:12]}…); reprocessing"
                            f"[/yellow]")

                    # Fresh handle per run: an exception below must mark
                    # *this* run FAILED, never re-save the previous row.
                    workflow = None
                    result = None
                    run_output = None
                    try:
                        # One workflow row per run.
                        workflow = WorkflowState(
                            subject_id=subject_id,
                            task=task,
                            pipeline=args.pipeline,
                            status=ProcessingStatus.IN_PROGRESS,
                            session=session,
                            run=run_num,
                            created_at=datetime.now()
                        )
                        manager.save_state(workflow)

                        # This run is definitely being (re)processed:
                        # drop stale outputs *before* loading the data, so
                        # even a load failure cannot leave the previous
                        # FIF + success sidecar next to a new FAILED row.
                        run_output = paths.get_preprocessing_dir(
                            pipeline=args.pipeline,
                            subject=subject_id,
                            session=session,
                            task=task,
                            run=run_num
                        )
                        run_output.mkdir(parents=True, exist_ok=True)
                        clear_run_outputs(run_output)

                        console.print(f"    Processing run {run_num}...")

                        # Load data step
                        load_step = StepRecord(
                            step_name=f"load_data_run{run_num}",
                            status=ProcessingStatus.IN_PROGRESS,
                            started_at=datetime.now()
                        )
                        workflow.add_step(load_step)
                        manager.save_state(workflow)
                        raw = None
                        print("read raw eeg enter...")
                        raw = read_raw_eeg(input_file)
                        # match extension:
                        #     case "vhdr":
                        #         raw = mne.io.read_raw_brainvision(
                        #             input_file, preload=True, verbose=False)
                        #         # Auto-set common auxiliary channel types for BrainVision
                        #         misc_channels = {
                        #             'photosensor': 'stim',
                        #             'optical': 'stim',
                        #             'ecg': 'ecg',
                        #             'resp': 'misc'
                        #         }
                        #         channels_to_set = {
                        #             ch: ch_type for ch, ch_type in misc_channels.items()
                        #             if ch in raw.ch_names
                        #         }
                        #         if channels_to_set:
                        #             raw.set_channel_types(channels_to_set)
                        #     case "fif":
                        #         raw = mne.io.read_raw_fif(
                        #             input_file, preload=True, verbose=False)
                        #     case "edf":
                        #         raw = mne.io.read_raw_edf(
                        #             input_file, preload=True, verbose=False)
                        #     case "bdf":
                        #         raw = mne.io.read_raw_bdf(
                        #             input_file, preload=True, verbose=False)
                        #     case "set":
                        #         raw = mne.io.read_raw_eeglab(
                        #             input_file, preload=True, verbose=False)
                        #     case "cnt":
                        #         raw = mne.io.read_raw_cnt(
                        #             input_file, preload=True, verbose=False)
                        #     case "cdt":
                        #         raw = mne.io.read_raw_curry(
                        #             input_file, preload=True, verbose=False)

                        # raw = mne.io.read_raw_fif(input_file, preload=True, verbose=False)

                        load_step.status = ProcessingStatus.COMPLETED
                        load_step.completed_at = datetime.now()
                        workflow.add_step(load_step)
                        manager.save_state(workflow)

                        # Preprocessing step
                        preproc_step = StepRecord(
                            step_name=f"preprocessing_run{run_num}",
                            status=ProcessingStatus.IN_PROGRESS,
                            started_at=datetime.now()
                        )
                        workflow.add_step(preproc_step)
                        manager.save_state(workflow)

                        # Run preprocessing from the validated spec (named
                        # chain or explicit step list — `steps` is the only
                        # selector; the schema already rejected anything
                        # else).
                        module = PreprocessingPipeline(
                            spec.pipeline_steps(), run_output)
                        # P1-F7: record the resolved-config hash on the
                        # workflow row before any save_state call, so the
                        # DB identifies the exact pipeline that ran.
                        workflow.config_hash = resolved_config_hash(module)
                        result = module.process(
                            raw,
                            subject_id=subject_id,
                            session_id=session,
                            run_id=run_num,
                        )

                        if result.success:
                            # The run is COMPLETED as soon as the pipeline
                            # succeeds: QC is diagnostic and must never
                            # flip the persisted status.
                            preproc_step.status = ProcessingStatus.COMPLETED
                            preproc_step.completed_at = datetime.now()
                            preproc_step.output_path = str(run_output)
                            preproc_step.metadata = result.metadata

                            # Extract bad channel information from metadata.
                            # P1-F5/F10: the canonical chain reports the
                            # detector's findings under
                            # `metadata['bad_channel_detection']`
                            # (`detected` + per-channel `reasons`); the legacy
                            # `bad_channels` dialect is kept as a fallback.
                            removed_channels = {}
                            detector_meta = result.metadata.get(
                                'bad_channel_detection') or {}
                            if detector_meta:
                                reasons_map = detector_meta.get('reasons') or {}
                                for ch in (detector_meta.get('detected') or []):
                                    criteria = reasons_map.get(ch) or ['detected']
                                    if not isinstance(criteria, (list, tuple)):
                                        criteria = [criteria]
                                    removed_channels[str(ch)] = ', '.join(
                                        str(c) for c in criteria)
                            elif 'bad_channels' in result.metadata:
                                bad_ch_meta = result.metadata['bad_channels']
                                bad_channel_list = bad_ch_meta.get(
                                    'bad_channels', [])
                                method = bad_ch_meta.get('method', 'unknown')
                                # Convert list to dict with method as reason
                                method_str = ', '.join(method) if isinstance(
                                    method, list) else method
                                removed_channels = {
                                    ch: method_str for ch in bad_channel_list}

                            # Try to load task config and extract events for ERP QC
                            events = None
                            event_id = None

                            # Use explicit task config if provided, otherwise try to auto-detect
                            task_config_name = getattr(
                                args, 'task_config', None) or task
                            task_config_path = paths.get_configs_dir(
                                "tasks") / f"{task_config_name}.yaml"

                            if task_config_path.exists():
                                try:
                                    # Load task config
                                    task_config = TaskConfig.from_yaml(
                                        task_config_path)

                                    # Extract events from preprocessed data
                                    try:
                                        events, event_dict = mne.events_from_annotations(
                                            result.outputs['data'], verbose=False
                                        )

                                        # Get event codes from task config
                                        event_codes_to_use = task_config.get_event_codes_to_epoch()

                                        # Translate semantic names to numeric codes if needed
                                        event_mapping = get_event_mapping_for_run(
                                            bids_root=paths.bids_root,
                                            subject=subject_id,
                                            session=session,
                                            task=task,
                                            run=run_num
                                        )

                                        if event_mapping:
                                            translated_codes = translate_event_codes(
                                                event_codes_to_use, event_mapping)
                                        else:
                                            translated_codes = event_codes_to_use

                                        # Filter event_dict to only include task-specified events
                                        event_id = {
                                            name: code for name, code in event_dict.items()
                                            if name in translated_codes
                                        }

                                    except Exception as e:
                                        console.print(
                                            f"      [yellow]⚠ Could not extract events for ERP QC: {e}[/yellow]")

                                except Exception as e:
                                    console.print(
                                        f"      [yellow]⚠ Could not load task config for ERP QC: {e}[/yellow]")

                            # QC is best-effort: a QC failure is recorded
                            # as `qc_error` in the step metadata and warns,
                            # but the run stays COMPLETED (its FIF exists
                            # and is valid).
                            try:
                                # NEW: QC output in same directory as preprocessed data
                                # No longer in separate derivatives folder
                                qc = PreprocessedQC(run_output)
                                qc_result = qc.compute(
                                    data=result.outputs['data'],
                                    subject_id=subject_id,
                                    ica=result.outputs.get('ica'),
                                    raw_before=raw,
                                    metadata=result.metadata,
                                    removed_channels=removed_channels,
                                    session_id=session,
                                    task_name=task,  # Changed from task= to task_name=
                                    run=run_num,
                                    events=events,  # Pass events for ERP QC
                                    event_id=event_id  # Pass filtered event_id for ERP QC
                                )

                                # Save HTML report
                                html_filename = f"{subject_id}_ses-{session}_task-{task}_run-{run_num}_preprocessed_qc.html"
                                qc.generate_html_report(
                                    qc_result, save_path=run_output / html_filename)

                                # Save QC metrics as JSON (same directory)
                                metrics_data = extract_metrics_from_qc_result(
                                    qc_result,
                                    subject_id=subject_id,
                                    session=session,
                                    task=task,
                                    run=run_num,
                                    pipeline=args.pipeline,
                                    qc_report_path=html_filename
                                )
                                # Per-recording chain QC row (P1-F10): duration,
                                # line-noise/EMG proxies, blink-residual gate,
                                # ASR burden, block-rejection fraction. Stored in
                                # the same metrics JSON so the HBN QC table can be
                                # built without re-reading the FIFs.
                                metrics_data['recording_qc'] = recording_qc_metrics(
                                    result.outputs['data'],
                                    task,
                                    subject_id,
                                    chain_metadata=result.metadata,
                                    raw_pre=raw,
                                )
                                json_path = run_output / \
                                    f"{subject_id}_ses-{session}_task-{task}_run-{run_num}_qc_metrics.json"
                                save_qc_metrics_json(metrics_data, json_path)

                                if qc_result.metrics:
                                    preproc_step.metadata['qc_metrics'] = {
                                        m.name: m.value for m in qc_result.metrics}
                                console.print(
                                    f"      [green]✓ Run {run_num} completed (QC generated)[/green]")
                            except Exception as qc_error:
                                import traceback
                                preproc_step.metadata['qc_error'] = str(qc_error)
                                console.print(
                                    f"      [red]⚠ Run {run_num} QC failed: "
                                    f"{qc_error}[/red]")
                                console.print(
                                    f"      [dim]{traceback.format_exc()}[/dim]")
                                console.print(
                                    f"      [green]✓ Run {run_num} completed "
                                    f"(preprocessing OK, QC failed)[/green]")
                        elif result.metadata.get('excluded'):
                            excluded = result.metadata['excluded']
                            preproc_step.status = ProcessingStatus.EXCLUDED
                            preproc_step.completed_at = datetime.now()
                            preproc_step.output_path = str(run_output)
                            preproc_step.metadata = result.metadata
                            preproc_step.error_message = (
                                f"excluded: {excluded.get('reason')}")
                            # A failed/excluded run still leaves a QC metrics
                            # record (status + reason, no HTML): downstream QC
                            # tables must be able to tell an excluded recording
                            # from a missing one (P1-F10).
                            _write_run_qc_metrics(
                                run_output=run_output,
                                subject_id=subject_id,
                                session=session,
                                task=task,
                                run_num=run_num,
                                status=ProcessingStatus.EXCLUDED.value,
                                reason=f"excluded: {excluded.get('reason')}",
                                extra={'excluded': excluded},
                            )
                            console.print(
                                f"      [yellow]⊘ Run {run_num} excluded "
                                f"({excluded.get('reason')})[/yellow]")
                        else:
                            preproc_step.status = ProcessingStatus.FAILED
                            preproc_step.completed_at = datetime.now()
                            preproc_step.error_message = str(result.errors)
                            _write_run_qc_metrics(
                                run_output=run_output,
                                subject_id=subject_id,
                                session=session,
                                task=task,
                                run_num=run_num,
                                status=ProcessingStatus.FAILED.value,
                                reason="; ".join(str(e) for e in result.errors),
                                extra={'errors': [str(e) for e in result.errors]},
                            )
                            console.print(
                                f"      [red]✗ Run {run_num} failed: {result.errors}[/red]")

                        workflow.add_step(preproc_step)
                        workflow.status = preproc_step.status
                        manager.save_state(workflow)
                        run_statuses.append(preproc_step.status)

                    except Exception as run_error:
                        import traceback
                        console.print(
                            f"    [red]✗ run {run_num} error: {run_error}[/red]")
                        console.print(
                            f"    [dim]{traceback.format_exc()}[/dim]")
                        # A run that already produced outputs and then
                        # failed must not keep them: delete the FIF and
                        # rewrite its success sidecar as FAILED.
                        if result is not None and result.success:
                            fail_run_outputs(
                                run_output=run_output,
                                subject_id=subject_id,
                                session=session,
                                run_num=run_num,
                                output_files=result.output_files,
                                error=run_error,
                            )
                        # Crashed run == FAILED run: leave the same minimal QC
                        # record (status + reason) when the run directory exists.
                        if run_output is not None:
                            _write_run_qc_metrics(
                                run_output=run_output,
                                subject_id=subject_id,
                                session=session,
                                task=task,
                                run_num=run_num,
                                status=ProcessingStatus.FAILED.value,
                                reason=f"run error: {run_error}",
                            )
                        if workflow is None:
                            workflow = WorkflowState(
                                subject_id=subject_id,
                                task=task,
                                pipeline=args.pipeline,
                                status=ProcessingStatus.FAILED,
                                session=session,
                                run=run_num,
                                created_at=datetime.now(),
                            )
                        workflow.status = ProcessingStatus.FAILED
                        workflow.add_step(StepRecord(
                            step_name=f"error_run{run_num}",
                            status=ProcessingStatus.FAILED,
                            error_message=str(run_error),
                        ))
                        manager.save_state(workflow)
                        run_statuses.append(ProcessingStatus.FAILED)
                        continue

                # Subject level counters (rule: failed > excluded > completed).
                if run_statuses:
                    subject_status = _subject_status(run_statuses)
                    if subject_status is ProcessingStatus.COMPLETED:
                        batch_completed += 1
                        console.print(
                            f"  [green]✓ {subject_id} completed "
                            f"({len(run_statuses)} runs)[/green]")
                    elif subject_status is ProcessingStatus.FAILED:
                        batch_failed += 1
                        console.print(
                            f"  [red]✗ {subject_id} failed[/red]")
                    else:
                        batch_excluded += 1
                        console.print(
                            f"  [yellow]⊘ {subject_id} excluded[/yellow]")

            except Exception as e:
                import traceback
                console.print(f"  [red]✗ {subject_id} error: {e}[/red]")
                console.print(f"  [dim]{traceback.format_exc()}[/dim]")
                batch_failed += 1
                if workflow is not None:
                    workflow.status = ProcessingStatus.FAILED
                    workflow.add_step(StepRecord(
                        step_name="error",
                        status=ProcessingStatus.FAILED,
                        error_message=str(e)
                    ))
                    manager.save_state(workflow)

            progress.update(overall_task, advance=1)

    # Summary
    console.print("\n[bold]Batch Processing Complete[/bold]")
    console.print(f"  Completed: {batch_completed}")
    console.print(f"  Failed: {batch_failed}")
    console.print(f"  Excluded: {batch_excluded}")
    console.print(f"  Skipped: {batch_skipped}")

    # Overall database stats
    summary = manager.get_summary()
    console.print(f"\n[bold]Overall Database Stats[/bold]")
    console.print(
        f"  Total Completed: {summary['status_counts'].get('completed', 0)}")
    console.print(
        f"  Total Failed: {summary['status_counts'].get('failed', 0)}")
    console.print(
        f"  Total Excluded: {summary['status_counts'].get('excluded', 0)}")
    console.print(f"\nState saved to: {db_path}")

    # Exit code: any failed run (or subject-level error) is a non-zero
    # batch result, so scripts and CI do not treat a broken batch as
    # success. Exclusions are a valid scientific outcome and do not
    # affect the exit code; config errors already returned 1 above.
    if batch_failed > 0:
        return 2
    return None


def _get_subject_list(args, console, paths: EEGCPMPaths) -> List[str]:
    """Get list of subjects to process."""
    if args.subject:
        return [args.subject]
    elif args.subjects:
        if not args.subjects.exists():
            console.print(
                f"[red]Error: Subjects file not found: {args.subjects}[/red]")
            return []
        with open(args.subjects, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        # Auto-detect from BIDS using new paths
        if not paths.bids_root.exists():
            console.print(
                f"[yellow]Warning: BIDS directory not found: {paths.bids_root}[/yellow]")
            return []

        subjects = []
        for subject_dir in sorted(paths.bids_root.glob("sub-*")):
            if subject_dir.is_dir():
                subjects.append(subject_dir.name.replace('sub-', ''))

        return subjects


def _find_input_files(project_root: Path, subject_id: str, task: str) -> dict["files": List[Path], "file_type": str]:
    """
    Find all input EEG files for subject/task.

    Returns list of files (multiple if runs exist, single file otherwise).
    """
    files = []

    # Try project_root directly as BIDS (common case)
    eeg_dir = project_root / f"sub-{subject_id}" / "ses-01" / "eeg"
    subject_dir = project_root / f"sub-{subject_id}"

    if eeg_dir.exists():
        print("task:\t", task)
        print("subject_id:\t", subject_id)
        print("eeg_dir:\t", eeg_dir)

        # Search for files in order of format preference (legacy)
        # for ext, file_type in supported_formats:
        #     # Look for run-based files first (multiple runs)
        #     run_files = sorted(eeg_dir.glob(f"sub-{subject_id}_ses-01_task-{task}_run-*_eeg.{ext}"))
        #     if run_files:
        #         print(f"Found {file_type} type with {len(run_files)} run(s)!")
        #         return {"files": run_files, "file_type": file_type}

        #     # Look for single file without run number
        #     single_file = eeg_dir / f"sub-{subject_id}_ses-01_task-{task}_eeg.{ext}"
        #     if single_file.exists():
        #         print(f"Found {file_type} type (single file)!")
        #         return {"files": [single_file], "file_type": file_type}

        # Search for files in order of format preference (Support multiple sessions)
        result = find_eeg_run_files(subject_dir=subject_dir,
                           subject_id=subject_id, task=task)
        if result is not None:
            return result

    # Try without session
    eeg_dir_no_ses = project_root / f"sub-{subject_id}" / "eeg"
    if eeg_dir_no_ses.exists():
        for ext, file_type in supported_formats:
            # Look for run-based files
            run_files = sorted(eeg_dir_no_ses.glob(
                f"sub-{subject_id}_task-{task}_run-*_eeg.{ext}"))
            if run_files:
                return {"files": run_files, "file_type": file_type}

            # Look for single file
            single_file = eeg_dir_no_ses / \
                f"sub-{subject_id}_task-{task}_eeg.{ext}"
            if single_file.exists():
                return {"files": [single_file], "file_type": file_type}

    # Try project_root/bids subdirectory (legacy)
    eeg_dir_legacy = project_root / "bids" / \
        f"sub-{subject_id}" / "ses-01" / "eeg"
    if eeg_dir_legacy.exists():
        for ext, file_type in supported_formats:
            # Look for run-based files
            run_files = sorted(eeg_dir_legacy.glob(
                f"sub-{subject_id}_ses-01_task-{task}_run-*_eeg.{ext}"))
            if run_files:
                return {"files": run_files, "file_type": file_type}

            # Look for single file
            single_file = eeg_dir_legacy / \
                f"sub-{subject_id}_ses-01_task-{task}_eeg.{ext}"
            if single_file.exists():
                return {"files": [single_file], "file_type": file_type}

    # No files found
    return {"files": files, "file_type": "NOT_FOUND"}
