"""CLI command for source reconstruction."""

import sys
from pathlib import Path
from typing import List, Optional
import mne

from pydantic import ValidationError

from eegcpm.core.config import SourceConfig
from eegcpm.core.task_config import TaskConfig
from eegcpm.core.paths import EEGCPMPaths
from eegcpm.modules.source import SourceReconstructionModule
from eegcpm.data.event_mapping import get_event_mapping_for_run, translate_event_codes


def _select_epoch_file(session_dir: Path) -> Path:
    """Return the single ``*_epo.fif`` of a session directory (B5 task 6).

    Raises ``FileNotFoundError`` when the session has no epoch file and
    ``ValueError`` when it has more than one: silently taking the first of several
    epoch files would reconstruct a different dataset than requested (and mixing
    ``_epo.fif``/``_epo_trials.fif`` would go unnoticed).
    """
    epoch_files = sorted(session_dir.glob("*_epo.fif"))
    if not epoch_files:
        raise FileNotFoundError(f"no *_epo.fif file in {session_dir}")
    if len(epoch_files) > 1:
        raise ValueError(
            f"{len(epoch_files)} *_epo.fif files in {session_dir}: "
            f"{[path.name for path in epoch_files]}; exactly one is required"
        )
    return epoch_files[0]


def source_reconstruct_command(args):
    """
    Run source reconstruction on epoched data.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments with:
        - project: Path - Project root directory
        - config: Path - Source reconstruction config file
        - preprocessing: str - Preprocessing pipeline variant
        - task: str - Task name
        - subjects: List[str], optional - Specific subjects to process
        - sessions: List[str], optional - Specific sessions to process
    """
    project_root = Path(args.project)
    config_path = Path(args.config)

    print(f"\n{'='*60}")
    print("EEGCPM Source Reconstruction")
    print(f"{'='*60}\n")

    # Load configuration
    print(f"Loading config: {config_path}")
    try:
        source_config = SourceConfig.from_yaml(config_path)
    except ValidationError as exc:
        # B4/P2-F5: the source config is a single extra='forbid' schema; an
        # invalid/flat config must exit non-zero instead of silently no-op'ing.
        print(f"❌ Invalid source configuration: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(f"   Method: {source_config.method}")
    print(f"   Variant: {source_config.variant}")
    print(f"   Parcellation: {source_config.parcellation}")
    print(f"   Mode: {source_config.mode}")

    # Initialize paths
    paths = EEGCPMPaths(project_root)

    # Get dependencies - CLI args take precedence over config
    preprocessing = getattr(args, 'preprocessing', None)
    task = getattr(args, 'task', None)

    # Fall back to config if not specified on CLI
    if not preprocessing and source_config.depends_on:
        preprocessing = source_config.depends_on.get('preprocessing')
    if not task and source_config.depends_on:
        task = source_config.depends_on.get('task')

    epochs_variant = source_config.depends_on.get('epochs') if source_config.depends_on else None

    if not all([preprocessing, task]):
        raise ValueError("Must specify --preprocessing and --task (or include in config)")

    print(f"\n📊 Dependencies:")
    print(f"   Preprocessing: {preprocessing}")
    print(f"   Task: {task}")
    print(f"   Epochs: {epochs_variant or 'N/A (will use raw data)'}")

    # Load task config if available (for condition grouping)
    task_config = None
    task_config_name = getattr(args, 'task_config', None)

    # Determine which task config to use
    if task_config_name == "none":
        print(f"   ⚠️  Task config disabled (--task-config none)")
        print(f"   Will process each event separately (no grouping by condition)")
    else:
        # Use specified config or auto-detect by matching task_name field
        if task_config_name is None:
            # Auto-detect: find configs where task_name matches
            print(f"   📋 Auto-detecting task configs for task: {task}")
            task_configs_dir = paths.get_configs_dir("tasks")
            matching_configs = []

            if task_configs_dir.exists():
                for config_file in sorted(task_configs_dir.glob("*.yaml")):
                    try:
                        import yaml
                        with open(config_file) as f:
                            cfg = yaml.safe_load(f)
                        if not isinstance(cfg, dict):
                            raise ValueError(
                                "task config is not a YAML mapping "
                                f"({type(cfg).__name__})"
                            )
                    except Exception as exc:
                        # B5 task 6: an unreadable task config must not be skipped
                        # silently — it may be the config that matches this task.
                        print(
                            f"❌ Unreadable task config {config_file}: {exc}",
                            file=sys.stderr,
                        )
                        raise SystemExit(1)

                    # Check if task_name field matches
                    if cfg.get('task_name') == task:
                        matching_configs.append(config_file)

            if matching_configs:
                # Try exact filename match first
                exact_match = [c for c in matching_configs if c.stem == task]
                if exact_match:
                    task_config_path = exact_match[0]
                    print(f"   ✓ Found exact match: {task_config_path.name}")
                else:
                    # Use first alphabetically
                    matching_configs.sort(key=lambda x: x.stem)
                    task_config_path = matching_configs[0]
                    print(f"   ✓ Using first match: {task_config_path.name}")

                if len(matching_configs) > 1:
                    print(f"   ℹ️  Found {len(matching_configs)} matching configs for task '{task}'")
                    for cfg in matching_configs:
                        print(f"      - {cfg.name}")
            else:
                print(f"   ⚠️  No task configs found with task_name='{task}'")
                print(f"   Will process each event separately (no grouping by condition)")
                task_config_path = None
        else:
            # Specific config requested
            print(f"   📋 Using specified task config: {task_config_name}.yaml")
            task_config_path = paths.get_configs_dir("tasks") / f"{task_config_name}.yaml"

        # Load the selected config
        if task_config_path and task_config_path.exists():
            try:
                task_config = TaskConfig.from_yaml(task_config_path)

                # Validate task_name matches
                if task_config.task_name != task:
                    print(f"   ⚠️  Warning: Config task_name '{task_config.task_name}' doesn't match dependency task '{task}'")
                    print(f"   Proceeding anyway (manual override)")

                print(f"   ✓ Loaded task config: {task_config_path.name}")
                print(f"   Conditions: {[c.name for c in task_config.conditions]}")
            except Exception as e:
                print(f"   ⚠️  Could not load task config: {e}")
                print(f"   Will process each event separately")
                task_config = None
        elif task_config_path:
            print(f"   ⚠️  Task config not found: {task_config_path}")
            print(f"   Will process each event separately (no grouping by condition)")

    # Get epochs directory (root, without subject/session filters)
    epochs_dir = paths.derivatives_root / "epochs" / preprocessing / task

    if not epochs_dir.exists():
        raise FileNotFoundError(f"Epochs directory not found: {epochs_dir}")

    # Find subjects to process (sorted: deterministic order, B5 task 6)
    subject_dirs = sorted(epochs_dir.glob("sub-*"))
    if not subject_dirs:
        raise FileNotFoundError(f"No subjects found in: {epochs_dir}")

    # Filter subjects if specified
    if args.subjects:
        subject_dirs = [
            d for d in subject_dirs
            if d.name.replace("sub-", "") in args.subjects
        ]

    print(f"\n👥 Found {len(subject_dirs)} subjects to process")

    # Process each subject
    n_success = 0
    n_failed = 0
    failed_subjects = []

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name.replace("sub-", "")

        # Find sessions (sorted: deterministic order, B5 task 6)
        session_dirs = sorted(subject_dir.glob("ses-*"))
        if not session_dirs:
            # No session subdirectory, look for epoch files directly
            session_dirs = [subject_dir]

        for session_dir in session_dirs:
            if session_dir.name.startswith("ses-"):
                session_id = session_dir.name.replace("ses-", "")
            else:
                session_id = "01"

            # Filter sessions if specified
            if args.sessions and session_id not in args.sessions:
                continue

            print(f"\n🔄 Processing: sub-{subject_id}, ses-{session_id}")

            # Exactly one epoch file per session (B5 task 6): 0 or >1 counts the
            # session as failed with the reason, never a silent pick/skip.
            try:
                epoch_file = _select_epoch_file(session_dir)
            except (FileNotFoundError, ValueError) as exc:
                print(f"   ❌ {exc}")
                n_failed += 1
                failed_subjects.append(f"{subject_id}/ses-{session_id}: {exc}")
                continue

            print(f"   📂 Loading epochs: {epoch_file.name}")

            try:
                # Load epochs
                epochs = mne.read_epochs(epoch_file, preload=True, verbose=False)
                print(f"   ✓ Loaded {len(epochs)} epochs")

                # Translate task config event codes to match epochs
                task_config_translated = None
                if task_config:
                    # Get event mapping from BIDS
                    # Try to find events.tsv - check different run numbers
                    event_mapping = {}
                    for run_num in ["1", "01", "2", "02", "3", "03"]:
                        event_mapping = get_event_mapping_for_run(
                            bids_root=paths.bids_root,
                            subject=subject_id,
                            session=session_id,
                            task=task,
                            run=run_num
                        )
                        if event_mapping:
                            print(f"   📋 Using event mapping from run-{run_num}")
                            break

                    if event_mapping:
                        # Create translated version of task config
                        task_config_dict = task_config.model_dump()
                        for condition in task_config_dict['conditions']:
                            original_codes = condition['event_codes']
                            translated_codes = translate_event_codes(original_codes, event_mapping)
                            condition['event_codes'] = translated_codes
                            print(f"   🔄 Condition '{condition['name']}': {original_codes} → {translated_codes}")

                        task_config_translated = task_config_dict
                    else:
                        print(f"   ⚠️  No event mapping found - using task config as-is")
                        task_config_translated = task_config.model_dump() if task_config else None
                else:
                    task_config_translated = None

                # Get output directory using stage-first paths
                output_dir = paths.get_source_dir(
                    preprocessing=preprocessing,
                    task=task,
                    variant=source_config.variant,
                    subject=subject_id,
                    session=session_id
                )

                # Create subject object (simple namespace)
                from types import SimpleNamespace
                subject = SimpleNamespace(id=subject_id, session=session_id)

                # Run source reconstruction
                module = SourceReconstructionModule(
                    config=source_config.model_dump(),
                    output_dir=output_dir,
                    task_config=task_config_translated
                )
                result = module.process(
                    epochs, subject=subject, epochs_path=epoch_file
                )

                if result.success:
                    print(f"   ✅ Success!")
                    print(f"   📁 Output: {output_dir}")
                    if result.output_files:
                        print(f"   📄 Files: {len(result.output_files)}")
                        for f in result.output_files:
                            print(f"      - {f.name}")
                    n_success += 1
                else:
                    print(f"   ❌ Failed: {result.errors}")
                    n_failed += 1
                    failed_subjects.append(f"{subject_id}/ses-{session_id}")

                if result.warnings:
                    for warning in result.warnings:
                        print(f"   ⚠️  {warning}")

            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                n_failed += 1
                failed_subjects.append(f"{subject_id}/ses-{session_id}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"✅ Successful: {n_success}")
    print(f"❌ Failed: {n_failed}")

    if failed_subjects:
        print(f"\nFailed subjects:")
        for sub in failed_subjects:
            print(f"  - {sub}")

    print()
