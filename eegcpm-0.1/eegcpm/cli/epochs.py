"""CLI command for epoch extraction."""

from pathlib import Path
from typing import List, Optional
import mne
import numpy as np

from eegcpm.core.config import EpochsConfig
from eegcpm.core.task_config import TaskConfig
from eegcpm.core.paths import EEGCPMPaths
from eegcpm.modules.epochs import EpochExtractionModule
from eegcpm.modules.qc.epochs_qc import generate_epochs_qc_report
from eegcpm.data.event_mapping import translate_event_codes, get_event_mapping_for_run


def epochs_command(args):
    """
    Run epoch extraction on preprocessed data.

    Combines multiple runs per subject/session/task into a single epoch file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments with:
        - project: Path - Project root directory
        - config: Path - Epochs config file
        - preprocessing: str - Preprocessing pipeline variant
        - task: str, optional - Task name (can also be in config)
        - subjects: List[str], optional - Specific subjects to process
        - sessions: List[str], optional - Specific sessions to process
        - runs: List[str], optional - Specific runs to combine
    """
    project_root = Path(args.project)
    config_path = Path(args.config)

    print(f"\n{'='*60}")
    print("EEGCPM Epoch Extraction")
    print(f"{'='*60}\n")

    # Initialize paths
    paths = EEGCPMPaths(project_root)

    # Determine if config is an epochs config or task config
    import yaml
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    is_epochs_config = config_data.get('stage') == 'epochs'

    if is_epochs_config:
        # New style: epochs config (loads task config separately)
        print(f"📄 Loading epochs config: {config_path}")
        print(f"   Variant: {config_data.get('variant', 'N/A')}")

        # Get dependencies
        depends_on = config_data.get('depends_on', {})
        preprocessing = getattr(args, 'preprocessing', None) or depends_on.get('preprocessing')
        task = getattr(args, 'task', None) or depends_on.get('task')

        if not all([preprocessing, task]):
            raise ValueError("Epochs config must specify preprocessing and task in depends_on")

        # Load task config
        task_config_path = paths.eegcpm_root / "configs" / "tasks" / f"{task}.yaml"
        if not task_config_path.exists():
            raise FileNotFoundError(f"Task config not found: {task_config_path}")

        print(f"📄 Loading task config: {task_config_path.name}")
        task_config = TaskConfig.from_yaml(task_config_path)
        print(f"   Conditions: {[c.name for c in task_config.conditions]}")

        # Merge configs: Start with task config, override with epochs config params
        processing_config = task_config.model_dump()

        # Override processing parameters from epochs config
        for key in ['tmin', 'tmax', 'baseline', 'decim', 'detrend', 'rejection']:
            if key in config_data:
                processing_config[key] = config_data[key]

    else:
        # Old style: task config contains everything (backward compatibility)
        print(f"📄 Loading task config: {config_path}")
        task_config = TaskConfig.from_yaml(config_path)
        print(f"   Conditions: {[c.name for c in task_config.conditions]}")

        # Get dependencies from CLI or config
        preprocessing = getattr(args, 'preprocessing', None)
        task = getattr(args, 'task', None)

        if not preprocessing and task_config.depends_on:
            preprocessing = task_config.depends_on.get('preprocessing')
        if not task:
            task = getattr(task_config, 'task_name', None)
            if not task and task_config.depends_on:
                task = task_config.depends_on.get('task')

        if not all([preprocessing, task]):
            raise ValueError("Must specify --preprocessing and --task (or include in config)")

        # Use task config as processing config
        processing_config = task_config.model_dump()

    # Display processing parameters
    print(f"\n⚙️  Processing Parameters:")
    print(f"   Time window: {processing_config.get('tmin')} to {processing_config.get('tmax')} s")
    print(f"   Baseline: {processing_config.get('baseline')}")
    if processing_config.get('decim', 1) != 1:
        print(f"   Decimation: {processing_config.get('decim')}")
    if processing_config.get('detrend') is not None:
        print(f"   Detrend: {processing_config.get('detrend')}")
    rejection = processing_config.get('rejection')
    if rejection:
        strategy = rejection.get('strategy', 'none') if isinstance(rejection, dict) else 'none'
        print(f"   Rejection strategy: {strategy}")
        if strategy == 'threshold' and isinstance(rejection, dict):
            if rejection.get('reject'):
                print(f"   Rejection thresholds: {rejection.get('reject')}")
        elif strategy == 'autoreject':
            print(f"   AutoReject enabled (data-driven thresholds)")

    print(f"\n📊 Dependencies:")
    print(f"   Preprocessing: {preprocessing}")
    print(f"   Task: {task}")

    # Get preprocessing directory (root, without subject/session filters)
    preprocessing_dir = paths.derivatives_root / "preprocessing" / preprocessing

    if not preprocessing_dir.exists():
        raise FileNotFoundError(f"Preprocessing directory not found: {preprocessing_dir}")

    # Find subjects to process
    subject_dirs = list(preprocessing_dir.glob("sub-*"))
    if not subject_dirs:
        raise FileNotFoundError(f"No subjects found in: {preprocessing_dir}")

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

        # Find sessions
        session_dirs = list(subject_dir.glob("ses-*"))
        if not session_dirs:
            print(f"\n⚠️  No sessions found for sub-{subject_id}, skipping")
            continue

        for session_dir in session_dirs:
            session_id = session_dir.name.replace("ses-", "")

            # Filter sessions if specified
            if args.sessions and session_id not in args.sessions:
                continue

            print(f"\n🔄 Processing: sub-{subject_id}, ses-{session_id}")

            # Find task directory
            task_dirs = list(session_dir.glob(f"task-{task}*"))
            if not task_dirs:
                print(f"   ⚠️  Task '{task}' not found, skipping")
                continue

            task_dir = task_dirs[0]

            # Find all run directories
            run_dirs = sorted(task_dir.glob("run-*"))
            if not run_dirs:
                # No run subdirectories, check if files are directly in task dir
                preprocessed_files = list(task_dir.glob("*_preprocessed_raw.fif"))
                if preprocessed_files:
                    run_dirs = [task_dir]
                else:
                    print(f"   ⚠️  No runs found, skipping")
                    continue

            # Filter runs if specified
            if hasattr(args, 'runs') and args.runs:
                # Keep only specified runs
                filtered_run_dirs = []
                for run_dir in run_dirs:
                    run_id = run_dir.name.replace("run-", "")
                    if run_id in args.runs:
                        filtered_run_dirs.append(run_dir)
                run_dirs = filtered_run_dirs

                if not run_dirs:
                    print(f"   ⚠️  No matching runs found (requested: {', '.join(args.runs)}), skipping")
                    continue

                print(f"   📂 Found {len(run_dirs)} runs (filtered from --runs: {', '.join(args.runs)})")
            else:
                print(f"   📂 Found {len(run_dirs)} runs (combining all)")

            try:
                # Load and concatenate all runs
                raw_list = []
                for run_dir in run_dirs:
                    # Find preprocessed raw file
                    raw_files = list(run_dir.glob("*_preprocessed_raw.fif"))
                    if not raw_files:
                        print(f"   ⚠️  No preprocessed file in {run_dir.name}, skipping run")
                        continue

                    raw_file = raw_files[0]
                    print(f"   📖 Loading: {raw_file.name}")
                    raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
                    raw_list.append(raw)

                if not raw_list:
                    print(f"   ❌ No valid runs found")
                    n_failed += 1
                    failed_subjects.append(f"{subject_id}/ses-{session_id}")
                    continue

                # Equalize channel sets across runs before concatenating
                if len(raw_list) > 1:
                    print(f"   🔗 Concatenating {len(raw_list)} runs")

                    # Find common channels across all runs
                    common_channels = set(raw_list[0].ch_names)
                    for raw in raw_list[1:]:
                        common_channels = common_channels.intersection(set(raw.ch_names))
                    common_channels = sorted(list(common_channels))

                    print(f"   🔧 Using {len(common_channels)} common channels across runs")

                    # Pick only common channels and remove projectors
                    raw_list_picked = []
                    for raw in raw_list:
                        raw_pick = raw.copy().pick(common_channels)
                        # Remove SSP projectors to avoid mismatch issues
                        raw_pick.del_proj()
                        raw_list_picked.append(raw_pick)

                    raw_combined = mne.concatenate_raws(raw_list_picked, verbose=False)
                else:
                    raw_combined = raw_list[0]

                print(f"   ✓ Combined data: {raw_combined.n_times} samples, {len(raw_combined.ch_names)} channels")

                # Get events (try annotations first, then stim channel)
                try:
                    events, event_dict = mne.events_from_annotations(raw_combined, verbose=False)
                    print(f"   ✓ Found {len(events)} events from annotations")
                except ValueError:
                    # No annotations, try stim channel
                    events = mne.find_events(raw_combined, verbose=False)
                    event_dict = None
                    print(f"   ✓ Found {len(events)} events from stim channel")

                # Build event_id from task config to filter which events to epoch
                # Get event codes defined in task config conditions
                event_codes_to_epoch = task_config.get_event_codes_to_epoch()

                # Load event mapping from BIDS to translate semantic names to numeric codes
                event_mapping = get_event_mapping_for_run(
                    bids_root=paths.bids_root,
                    subject=subject_id,
                    session=session_id,
                    task=task,
                    run=run_dirs[0].name.replace("run-", "")  # Use first run for mapping
                )

                # Translate event codes if needed (semantic names -> numeric codes)
                if event_mapping:
                    translated_codes = translate_event_codes(event_codes_to_epoch, event_mapping)
                    if translated_codes != event_codes_to_epoch:
                        print(f"   🔄 Translated event codes: {dict(zip(event_codes_to_epoch, translated_codes))}")
                        event_codes_to_epoch = translated_codes

                # Build event_id dict filtering only the events we want
                if event_dict:
                    # Filter event_dict to only include events in task config
                    filtered_event_id = {
                        name: code for name, code in event_dict.items()
                        if name in event_codes_to_epoch
                    }
                    print(f"   🎯 Filtering to {len(filtered_event_id)} event types: {list(filtered_event_id.keys())}")
                else:
                    filtered_event_id = None

                # Get output directory using stage-first paths
                output_dir = paths.get_epochs_dir(
                    preprocessing=preprocessing,
                    task=task,
                    subject=subject_id,
                    session=session_id
                )

                # Create subject object (simple namespace)
                from types import SimpleNamespace
                subject = SimpleNamespace(id=subject_id, session=session_id)

                # Run epoch extraction using merged processing config
                module = EpochExtractionModule(
                    config=processing_config,
                    output_dir=output_dir
                )
                result = module.process(
                    raw_combined,
                    subject=subject,
                    events=events,
                    event_id=filtered_event_id  # Use filtered event_id
                )

                if result.success:
                    print(f"   ✅ Success!")
                    print(f"   📁 Output: {output_dir}")

                    # Show epoch info
                    if 'data' in result.outputs:
                        epochs = result.outputs['data']
                        print(f"   📊 Epochs: {len(epochs)} trials")
                        if hasattr(epochs, 'event_id'):
                            print(f"   🏷️  Conditions: {list(epochs.event_id.keys())}")

                        # Show rejection statistics
                        if result.metadata and 'rejection' in result.metadata:
                            rej_meta = result.metadata['rejection']
                            if 'stats' in rej_meta:
                                stats = rej_meta['stats']
                                print(f"   🔍 Rejection Statistics:")
                                print(f"      Original events: {stats.get('n_original', 'N/A')}")

                                if rej_meta['strategy'] in ['threshold', 'both']:
                                    n_after_thresh = stats.get('n_after_threshold', 'N/A')
                                    n_rej_thresh = stats.get('n_rejected_threshold', 'N/A')
                                    print(f"      After threshold rejection: {n_after_thresh} ({n_rej_thresh} rejected)")

                                if rej_meta['strategy'] in ['autoreject', 'both']:
                                    if 'n_after_autoreject' in stats:
                                        n_after_ar = stats.get('n_after_autoreject', 'N/A')
                                        n_rej_ar = stats.get('n_rejected_autoreject', 'N/A')
                                        print(f"      After AutoReject: {n_after_ar} ({n_rej_ar} additional rejected)")
                                        if 'autoreject_log' in stats and stats['autoreject_log']:
                                            ar_log = stats['autoreject_log']
                                            if 'n_bad_epochs' in ar_log:
                                                print(f"      AutoReject bad epochs: {ar_log['n_bad_epochs']}")

                                print(f"      Final: {len(epochs)} epochs kept")

                        # Generate QC report
                        try:
                            qc_report_path = output_dir / f"{subject_id}_ses-{session_id}_task-{task}_epochs_qc.html"
                            generate_epochs_qc_report(
                                epochs=epochs,
                                task_config=processing_config,  # Use merged processing config
                                output_path=qc_report_path,
                                runs_included=None,  # TODO: Track which runs were included
                                runs_excluded=None,
                                subject_id=subject_id,
                                session=session_id,
                                task=task,
                            )
                            print(f"   📊 QC Report: {qc_report_path.name}")
                        except Exception as qc_error:
                            print(f"   ⚠️  QC report generation failed: {qc_error}")

                    if result.output_files:
                        print(f"   📄 Files: {len(result.output_files)}")
                        for f in result.output_files:
                            print(f"      - {f.name}")
                    n_success += 1
                else:
                    print(f"   ❌ Failed: {result.errors}")
                    n_failed += 1
                    failed_subjects.append(f"{subject_id}/ses-{session_id}")

            except Exception as e:
                print(f"   ❌ Error: {e}")
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
        for subj in failed_subjects:
            print(f"  - {subj}")

    print()
