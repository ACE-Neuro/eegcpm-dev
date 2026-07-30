"""CLI command for connectivity analysis."""

from pathlib import Path
from typing import List, Optional
import numpy as np

from eegcpm.core.paths import EEGCPMPaths
from eegcpm.core.task_config import TaskConfig
from eegcpm.modules.connectivity import ConnectivityModule


def connectivity_command(args):
    """
    Run connectivity analysis on source data.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments with:
        - project: Path - Project root directory
        - config: Path - Connectivity config file
        - preprocessing: str - Preprocessing pipeline variant
        - task: str - Task name
        - source: str - Source variant (e.g., eLORETA-CONN32)
        - subjects: List[str], optional - Specific subjects to process
        - sessions: List[str], optional - Specific sessions to process
    """
    project_root = Path(args.project)
    config_path = Path(args.config)

    print(f"\n{'='*60}")
    print("EEGCPM Connectivity Analysis")
    print(f"{'='*60}\n")

    # Load configuration
    print(f"Loading config: {config_path}")
    import yaml
    with open(config_path) as f:
        conn_config = yaml.safe_load(f)

    # Get dependencies - CLI args take precedence over config
    preprocessing = getattr(args, 'preprocessing', None)
    task = getattr(args, 'task', None)
    source_variant = getattr(args, 'source', None)
    # ENG-008: --space {sensor, source} is the new discriminator
    # for sensor-input mode. Default to 'source' for backwards
    # compatibility with the existing CLI; 'sensor' is the new
    # mode for the EEG d-factor arm.
    space = getattr(args, 'space', 'source')

    # Fall back to config if not specified on CLI
    depends_on = conn_config.get('depends_on', {})
    if not preprocessing:
        preprocessing = depends_on.get('preprocessing')
    if not task:
        task = depends_on.get('task')
    if not source_variant:
        source_variant = depends_on.get('source')

    # In sensor mode, --source is NOT required (we compute
    # connectivity directly on the preprocessed EEG).
    if space == "source":
        if not all([preprocessing, task, source_variant]):
            raise ValueError(
                "Must specify --preprocessing, --task, and --source "
                "(or include in config) in source space.")
    else:
        if not all([preprocessing, task]):
            raise ValueError(
                "Must specify --preprocessing and --task in sensor space.")

    print(f"   Methods: {', '.join(conn_config.get('methods', []))}")
    print(f"   Frequency Bands: {', '.join(conn_config.get('frequency_bands', {}).keys())}")
    print(f"   Time Windows: {len(conn_config.get('time_windows', []))}")
    print(f"   Space: {space}")

    print(f"\n📊 Dependencies:")
    print(f"   Preprocessing: {preprocessing}")
    print(f"   Task: {task}")
    if space == "source":
        print(f"   Source Variant: {source_variant}")
    else:
        print(f"   Space: sensor (no source variant needed)")

    # Initialize paths
    paths = EEGCPMPaths(project_root)

    # Load task config if available
    task_config = None
    task_config_name = getattr(args, 'task_config', None)

    if task_config_name != "none":
        # Auto-detect or use specified
        if task_config_name is None:
            print(f"   📋 Auto-detecting task configs for task: {task}")
            task_configs_dir = paths.get_configs_dir("tasks")
            matching_configs = []

            if task_configs_dir.exists():
                for config_file in task_configs_dir.glob("*.yaml"):
                    try:
                        with open(config_file) as f:
                            cfg = yaml.safe_load(f)
                        if cfg.get('task_name') == task:
                            matching_configs.append(config_file)
                    except Exception:
                        continue

            if matching_configs:
                exact_match = [c for c in matching_configs if c.stem == task]
                task_config_path = exact_match[0] if exact_match else matching_configs[0]
                print(f"   ✓ Using: {task_config_path.name}")
            else:
                print(f"   ⚠️  No task configs found with task_name='{task}'")
                task_config_path = None
        else:
            print(f"   📋 Using specified task config: {task_config_name}.yaml")
            task_config_path = paths.get_configs_dir("tasks") / f"{task_config_name}.yaml"

        if task_config_path and task_config_path.exists():
            try:
                task_config = TaskConfig.from_yaml(task_config_path)
                print(f"   ✓ Loaded task config")
                print(f"   Conditions: {[c.name for c in task_config.conditions]}")
            except Exception as e:
                print(f"   ⚠️  Could not load task config: {e}")

    # Get source directory (add variant- prefix if not present)
    source_variant_dir = source_variant if source_variant.startswith("variant-") else f"variant-{source_variant}"
    source_base = paths.derivatives_root / "source" / preprocessing / task / source_variant_dir

    if not source_base.exists():
        raise FileNotFoundError(f"Source directory not found: {source_base}")

    # Find subjects
    subject_dirs = list(source_base.glob("sub-*"))
    if not subject_dirs:
        raise FileNotFoundError(f"No subjects found in: {source_base}")

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

            # Find ROI timecourse file
            roi_files = list(session_dir.glob("*_roi_tc.npz"))
            if not roi_files:
                print(f"   ⚠️  No ROI file found, skipping")
                continue

            roi_file = roi_files[0]
            print(f"   📂 Loading ROI data: {roi_file.name}")

            try:
                # Load ROI data into a plain dict. Passing NpzFile directly makes the
                # connectivity module treat the whole archive as one condition
                roi_npz = np.load(roi_file, allow_pickle=True)
                roi_data = {key: roi_npz[key] for key in roi_npz.files}
                print(f"   ✓ Loaded ROI timecourses")

                # Get output directory. EEGCPMPaths has no connectivity helper yet
                output_dir = (
                    paths.derivatives_root
                    / "connectivity"
                    / preprocessing
                    / task
                    / source_variant_dir
                    / f"sub-{subject_id}"
                )
                output_dir.mkdir(parents=True, exist_ok=True)

                # Create subject object
                from types import SimpleNamespace
                subject = SimpleNamespace(id=f"sub-{subject_id}", session=session_id)

                # Run connectivity analysis
                module = ConnectivityModule(
                    config=conn_config,
                    output_dir=output_dir,
                )

                sfreq = float(roi_data.get("sfreq", 500.0))
                result = module.process(roi_data, subject=subject, sfreq=sfreq)

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
                import traceback
                traceback.print_exc()
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
