"""Preprocess command for EEGCPM CLI."""

from pathlib import Path
from typing import List, Optional
from datetime import datetime
import yaml
import mne
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from ..workflow.state import WorkflowStateManager, WorkflowState, ProcessingStatus, StepRecord
from ..modules.preprocessing import PreprocessingPipeline
from ..modules.qc.preprocessed_qc import PreprocessedQC
from ..data.event_mapping import get_event_mapping_for_run, translate_event_codes
from ..core.task_config import TaskConfig
from ..modules.qc.quality_assessment import extract_metrics_from_qc_result
from ..modules.qc.metrics_io import save_qc_metrics_json
from ..core.paths import EEGCPMPaths


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
        console.print(f"[red]Error: Config file not found: {args.config}[/red]")
        return

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract preprocessing config if nested
    if 'preprocessing' in config:
        config = config['preprocessing']

    # Setup paths using new centralized system
    paths = EEGCPMPaths(
        project_root=args.project,
        eegcpm_root=args.eegcpm_root if hasattr(args, 'eegcpm_root') and args.eegcpm_root else None
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

    # Get task
    task = args.task or config.get('task', 'unknown')

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

            # Check if already completed
            existing_state = manager.load_state(subject_id, task, args.pipeline)
            if existing_state and existing_state.status == ProcessingStatus.COMPLETED and not args.force:
                console.print(f"  [dim]✓ {subject_id} already completed (use --force to reprocess)[/dim]")
                batch_skipped += 1
                progress.update(overall_task, advance=1)
                continue

            # Initialize workflow state
            workflow = WorkflowState(
                subject_id=subject_id,
                task=task,
                pipeline=args.pipeline,
                status=ProcessingStatus.IN_PROGRESS,
                created_at=datetime.now()
            )

            try:
                # Find input files (may be multiple runs)
                input_files_response = _find_input_files(args.project, subject_id, task)
                print(input_files_response)
                input_files = input_files_response.get("files", None)
                if not input_files:
                    console.print(f"  [yellow]⚠ {subject_id}: No input files found for task {task}[/yellow]")
                    workflow.status = ProcessingStatus.FAILED
                    workflow.add_step(StepRecord(
                        step_name="load_data",
                        status=ProcessingStatus.FAILED,
                        error_message=f"No input files found for task {task}"
                    ))
                    manager.save_state(workflow)
                    progress.update(overall_task, advance=1)
                    continue

                console.print(f"  Found {len(input_files)} run(s) for {subject_id}")

                # Process each run
                for run_idx, input_file in enumerate(input_files, 1):
                    # Extract run number and session from filename
                    run_num = input_file.stem.split('_run-')[1].split('_')[0] if '_run-' in input_file.stem else str(run_idx)
                    # Extract session (default to "01")
                    session = input_file.stem.split('_ses-')[1].split('_')[0] if '_ses-' in input_file.stem else "01"

                    console.print(f"    Processing run {run_num}...")

                    # Load data step
                    load_step = StepRecord(
                        step_name=f"load_data_run{run_num}",
                        status=ProcessingStatus.IN_PROGRESS,
                        started_at=datetime.now()
                    )
                    workflow.add_step(load_step)
                    workflow.session = session  # Update workflow with actual session
                    workflow.run = run_num      # Update workflow with actual run
                    manager.save_state(workflow)
                    _, _, after = str(input_file).rpartition(".")
                    extension = after
                    print("extension:\t", extension)
                    raw = None
                    match extension:
                        case "vhdr":
                            raw = mne.io.read_raw_brainvision(input_file, preload=True, verbose=False)
                            # Auto-set common auxiliary channel types for BrainVision
                            misc_channels = {
                                'photosensor': 'stim',
                                'optical': 'stim',
                                'ecg': 'ecg',
                                'resp': 'misc'
                            }
                            channels_to_set = {
                                ch: ch_type for ch, ch_type in misc_channels.items()
                                if ch in raw.ch_names
                            }
                            if channels_to_set:
                                raw.set_channel_types(channels_to_set)
                        case "fif":
                            raw = mne.io.read_raw_fif(input_file, preload=True, verbose=False)
                        case "edf":
                            raw = mne.io.read_raw_edf(input_file, preload=True, verbose=False)
                        case "bdf":
                            raw = mne.io.read_raw_bdf(input_file, preload=True, verbose=False)
                        case "set":
                            raw = mne.io.read_raw_eeglab(input_file, preload=True, verbose=False)
                        case "cnt":
                            raw = mne.io.read_raw_cnt(input_file, preload=True, verbose=False)
                        case "cdt":
                            raw = mne.io.read_raw_curry(input_file, preload=True, verbose=False)

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

                    # NEW: Use centralized path management
                    run_output = paths.get_preprocessing_dir(
                        pipeline=args.pipeline,
                        subject=subject_id,
                        session=session,
                        task=task,
                        run=run_num
                    )
                    run_output.mkdir(parents=True, exist_ok=True)

                    # Run preprocessing
                    module = PreprocessingPipeline(config['steps'], run_output)
                    result = module.process(raw, subject_id=subject_id)

                    if result.success:
                        # Extract bad channel information from metadata
                        removed_channels = {}
                        if 'bad_channels' in result.metadata:
                            bad_ch_meta = result.metadata['bad_channels']
                            bad_channel_list = bad_ch_meta.get('bad_channels', [])
                            method = bad_ch_meta.get('method', 'unknown')
                            # Convert list to dict with method as reason
                            method_str = ', '.join(method) if isinstance(method, list) else method
                            removed_channels = {ch: method_str for ch in bad_channel_list}

                        # Try to load task config and extract events for ERP QC
                        events = None
                        event_id = None

                        # Use explicit task config if provided, otherwise try to auto-detect
                        task_config_name = getattr(args, 'task_config', None) or task
                        task_config_path = paths.get_configs_dir("tasks") / f"{task_config_name}.yaml"

                        if task_config_path.exists():
                            try:
                                # Load task config
                                task_config = TaskConfig.from_yaml(task_config_path)

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
                                        translated_codes = translate_event_codes(event_codes_to_use, event_mapping)
                                    else:
                                        translated_codes = event_codes_to_use

                                    # Filter event_dict to only include task-specified events
                                    event_id = {
                                        name: code for name, code in event_dict.items()
                                        if name in translated_codes
                                    }

                                except Exception as e:
                                    console.print(f"      [yellow]⚠ Could not extract events for ERP QC: {e}[/yellow]")

                            except Exception as e:
                                console.print(f"      [yellow]⚠ Could not load task config for ERP QC: {e}[/yellow]")

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
                        qc.generate_html_report(qc_result, save_path=run_output / html_filename)

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
                        json_path = run_output / f"{subject_id}_ses-{session}_task-{task}_run-{run_num}_qc_metrics.json"
                        save_qc_metrics_json(metrics_data, json_path)

                        preproc_step.status = ProcessingStatus.COMPLETED
                        preproc_step.completed_at = datetime.now()
                        preproc_step.output_path = str(run_output)
                        preproc_step.metadata = result.metadata
                        if qc_result.metrics:
                            preproc_step.metadata['qc_metrics'] = {m.name: m.value for m in qc_result.metrics}
                        console.print(f"      [green]✓ Run {run_num} completed (QC generated)[/green]")
                    else:
                        preproc_step.status = ProcessingStatus.FAILED
                        preproc_step.completed_at = datetime.now()
                        preproc_step.error_message = str(result.errors)
                        console.print(f"      [red]✗ Run {run_num} failed: {result.errors}[/red]")

                    workflow.add_step(preproc_step)
                    manager.save_state(workflow)

                # Mark overall workflow complete
                workflow.status = ProcessingStatus.COMPLETED
                batch_completed += 1
                console.print(f"  [green]✓ {subject_id} completed ({len(input_files)} runs)[/green]")

            except Exception as e:
                import traceback
                console.print(f"  [red]✗ {subject_id} error: {e}[/red]")
                console.print(f"  [dim]{traceback.format_exc()}[/dim]")
                workflow.status = ProcessingStatus.FAILED
                batch_failed += 1
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
    console.print(f"  Skipped: {batch_skipped}")

    # Overall database stats
    summary = manager.get_summary()
    console.print(f"\n[bold]Overall Database Stats[/bold]")
    console.print(f"  Total Completed: {summary['status_counts'].get('completed', 0)}")
    console.print(f"  Total Failed: {summary['status_counts'].get('failed', 0)}")
    console.print(f"\nState saved to: {db_path}")


def _get_subject_list(args, console, paths: EEGCPMPaths) -> List[str]:
    """Get list of subjects to process."""
    if args.subject:
        return [args.subject]
    elif args.subjects:
        if not args.subjects.exists():
            console.print(f"[red]Error: Subjects file not found: {args.subjects}[/red]")
            return []
        with open(args.subjects, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        # Auto-detect from BIDS using new paths
        if not paths.bids_root.exists():
            console.print(f"[yellow]Warning: BIDS directory not found: {paths.bids_root}[/yellow]")
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

    # Define supported formats: (extension, file_type_name)
    supported_formats = [
        ('fif', 'FIF'),           # MNE native format
        ('vhdr', 'BrainVision'),  # BrainVision
        ('edf', 'EDF'),           # European Data Format
        ('bdf', 'BDF'),           # BioSemi
        ('set', 'EEGLAB'),        # EEGLAB
        ('cnt', 'NeuroScan'),     # NeuroScan
        ('cdt', 'CURRY'),         # CURRY
    ]

    # Try project_root directly as BIDS (common case)
    eeg_dir = project_root / f"sub-{subject_id}" / "ses-01" / "eeg"

    if eeg_dir.exists():
        print("task:\t", task)
        print("subject_id:\t", subject_id)
        print("eeg_dir:\t", eeg_dir)

        # Search for files in order of format preference
        for ext, file_type in supported_formats:
            # Look for run-based files first (multiple runs)
            run_files = sorted(eeg_dir.glob(f"sub-{subject_id}_ses-01_task-{task}_run-*_eeg.{ext}"))
            if run_files:
                print(f"Found {file_type} type with {len(run_files)} run(s)!")
                return {"files": run_files, "file_type": file_type}

            # Look for single file without run number
            single_file = eeg_dir / f"sub-{subject_id}_ses-01_task-{task}_eeg.{ext}"
            if single_file.exists():
                print(f"Found {file_type} type (single file)!")
                return {"files": [single_file], "file_type": file_type}

    # Try without session
    eeg_dir_no_ses = project_root / f"sub-{subject_id}" / "eeg"
    if eeg_dir_no_ses.exists():
        for ext, file_type in supported_formats:
            # Look for run-based files
            run_files = sorted(eeg_dir_no_ses.glob(f"sub-{subject_id}_task-{task}_run-*_eeg.{ext}"))
            if run_files:
                return {"files": run_files, "file_type": file_type}

            # Look for single file
            single_file = eeg_dir_no_ses / f"sub-{subject_id}_task-{task}_eeg.{ext}"
            if single_file.exists():
                return {"files": [single_file], "file_type": file_type}

    # Try project_root/bids subdirectory (legacy)
    eeg_dir_legacy = project_root / "bids" / f"sub-{subject_id}" / "ses-01" / "eeg"
    if eeg_dir_legacy.exists():
        for ext, file_type in supported_formats:
            # Look for run-based files
            run_files = sorted(eeg_dir_legacy.glob(f"sub-{subject_id}_ses-01_task-{task}_run-*_eeg.{ext}"))
            if run_files:
                return {"files": run_files, "file_type": file_type}

            # Look for single file
            single_file = eeg_dir_legacy / f"sub-{subject_id}_ses-01_task-{task}_eeg.{ext}"
            if single_file.exists():
                return {"files": [single_file], "file_type": file_type}

    # No files found
    return {"files": files, "file_type": "NOT_FOUND"}
