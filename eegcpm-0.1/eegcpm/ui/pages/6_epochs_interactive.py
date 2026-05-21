"""Epochs Extraction - Interactive Mode.

Single-subject epoch extraction with manual run selection and QC review.
Use this page to test task configurations and handle individual subjects
that need custom run selection.
"""

import streamlit as st
from pathlib import Path
import sys
import yaml
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eegcpm.pipeline.run_processor import RunProcessor, RunProcessingResult
from eegcpm.pipeline.epoch_combiner import EpochCombiner
from eegcpm.workflow.state import WorkflowStateManager
from eegcpm.modules.qc.metrics_io import load_qc_metrics_from_directory
from eegcpm.modules.qc.models import RunQualityMetricsFromJSON
from eegcpm.core.paths import EEGCPMPaths
from eegcpm.ui.utils import (
    scan_subjects,
    scan_sessions,
    scan_tasks,
    scan_pipelines,
    get_processed_subjects
)
import json


# REMOVED: scan_pipeline_folders() - now using derivatives/preprocessing/ via EEGCPMPaths


def _get_event_codes_to_epoch(task_config: dict) -> list:
    """Get list of event codes to extract epochs for from task config.

    Extracts all event codes from all conditions.
    This is used to filter which events to epoch (ignoring trial_start, triggers, etc.).

    Parameters
    ----------
    task_config : dict
        Task configuration dictionary

    Returns
    -------
    list
        List of event codes (strings or ints) to epoch
    """
    event_codes = []
    if 'conditions' in task_config:
        for condition in task_config['conditions']:
            if 'event_codes' in condition:
                event_codes.extend(condition['event_codes'])
    return event_codes


def main():
    """Run selection interface main function."""

    st.set_page_config(
        page_title="Epochs: Interactive - EEGCPM",
        page_icon="🔬",
        layout="wide"
    )

    st.title("🔬 Epochs Extraction - Interactive")
    st.markdown("Single-subject mode: Review run quality, manually select runs, and extract epochs")

    # Get paths from main app project selection
    from eegcpm.ui.project_manager import ProjectManager
    from eegcpm.ui.session_persistence import restore_project_from_storage

    restore_project_from_storage()

    if 'project_manager' not in st.session_state:
        st.session_state.project_manager = ProjectManager()

    if 'current_project_name' not in st.session_state or st.session_state.current_project_name is None:
        st.error("⚠️ No project selected. Please select a project on the Home page first.")
        st.stop()

    pm = st.session_state.project_manager
    project = pm.get_project(st.session_state.current_project_name)

    if not project:
        st.error("⚠️ Project not found. Please select a project on the Home page first.")
        st.stop()

    # Display current project (read-only)
    st.sidebar.header("📂 Current Project")
    st.sidebar.info(f"**{project.name}**")
    st.sidebar.caption(f"BIDS: `{project.bids_root}`")
    st.sidebar.caption(f"EEGCPM: `{project.eegcpm_root}`")

    # Derive project root from bids_root (assumes bids/ is subfolder)
    bids_path = Path(project.bids_root)
    project_root = bids_path.parent if bids_path.name == "bids" else bids_path

    # Create path manager from project root
    paths = EEGCPMPaths(project_root)

    # Load task configs (NOT preprocessing configs)
    config_dir = paths.get_configs_dir("tasks")
    available_configs = []

    if config_dir.exists():
        available_configs = sorted([str(f) for f in config_dir.glob("*.yaml")])

    if not available_configs:
        st.sidebar.warning("⚠️ No task configs found. Create one in Task Config page.")
        st.stop()

    # Subject/task selection
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Selection")

    # NEW: Scan preprocessing variants in derivatives/preprocessing/
    preprocessing_dir = paths.derivatives_root / "preprocessing"
    pipeline_options = []

    if preprocessing_dir.exists():
        # Exclude special directories that aren't pipelines
        excluded_dirs = {'logs', 'qc', 'reports', '__pycache__'}
        pipeline_options = sorted([
            d.name for d in preprocessing_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.') and d.name not in excluded_dirs
        ])

    if not pipeline_options:
        st.sidebar.warning("⚠️ No preprocessing pipelines found in derivatives/preprocessing/")
        st.sidebar.info("Run preprocessing first, then return here to select runs.")
        st.stop()

    # Pipeline dropdown
    selected_pipeline = st.sidebar.selectbox(
        "Preprocessing Pipeline",
        options=pipeline_options,
        help="Select preprocessing variant (e.g., standard, minimal, robust)"
    )

    pipeline = selected_pipeline
    pipeline_path = preprocessing_dir / selected_pipeline

    # Get subjects from this pipeline
    subjects = sorted([
        d.name.replace('sub-', '') for d in pipeline_path.iterdir()
        if d.is_dir() and d.name.startswith('sub-')
    ])

    # Show pipeline info
    n_subjects = len(subjects)
    st.sidebar.caption(f"✓ {n_subjects} subject{'s' if n_subjects != 1 else ''} in {pipeline}")

    # Subject dropdown
    if not subjects:
        st.warning("No subjects found")
        return

    subject_id = st.sidebar.selectbox(
        "Subject",
        options=subjects,
        help="Select subject"
    )

    # Session dropdown
    sessions = scan_sessions(paths.bids_root, subject_id)
    if not sessions:
        st.warning(f"No sessions found for subject {subject_id}")
        return

    session = st.sidebar.selectbox(
        "Session",
        options=sessions,
        help="Select session"
    )

    # Task dropdown
    tasks = scan_tasks(paths.bids_root, subject_id)
    if not tasks:
        st.warning(f"No tasks found for subject {subject_id}")
        return

    task = st.sidebar.selectbox(
        "Task",
        options=tasks,
        help="Select task"
    )

    # Task Config Selection - filter by task_name matching selected task
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Task Configuration")

    matching_configs = []  # [(filepath, config_name, description)]
    for config_path in available_configs:
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)

            # Check if task_name matches selected task
            if cfg.get('task_name') == task:
                config_name = Path(config_path).stem
                description = cfg.get('description', 'No description')
                matching_configs.append((config_path, config_name, description))
        except Exception:
            continue

    if not matching_configs:
        st.sidebar.warning(f"⚠️ No task configs found for task '{task}'")
        st.sidebar.info("💡 Go to **Task Config** page to create one")
        st.stop()

    # Show dropdown with config name and description
    config_options = [f"{name} - {desc}" for _, name, desc in matching_configs]
    config_paths = [path for path, _, _ in matching_configs]

    selected_idx = st.sidebar.selectbox(
        "Config Variant",
        options=range(len(config_options)),
        format_func=lambda i: config_options[i],
        help="Select which task configuration to use for epoching"
    )

    config_file = config_paths[selected_idx]
    st.sidebar.success(f"✓ Task config: {Path(config_file).stem}")

    # Epochs Config Selection
    st.sidebar.markdown("---")
    st.sidebar.header("⚡ Epochs Configuration")

    epochs_config_dir = paths.get_configs_dir("epochs")
    epochs_configs = []  # [(filepath, variant, description)]

    if epochs_config_dir.exists():
        for epochs_config_path in epochs_config_dir.glob("*.yaml"):
            try:
                with open(epochs_config_path, 'r') as f:
                    epochs_cfg = yaml.safe_load(f)

                # Check it's an epochs config
                if epochs_cfg.get('stage') == 'epochs':
                    variant = epochs_cfg.get('variant', epochs_config_path.stem)
                    notes = epochs_cfg.get('notes', 'No description')
                    # Get first line of notes for display
                    description = notes.split('\n')[0][:50] if notes else 'No description'
                    epochs_configs.append((str(epochs_config_path), variant, description))
            except Exception:
                continue

    if not epochs_configs:
        st.sidebar.warning("⚠️ No epochs configs found. Using task config settings.")
        epochs_config_file = None
    else:
        # Show dropdown with variant and description
        epochs_config_options = [f"{variant} - {desc}" for _, variant, desc in epochs_configs]
        epochs_config_paths = [path for path, _, _ in epochs_configs]

        epochs_selected_idx = st.sidebar.selectbox(
            "Rejection Strategy",
            options=range(len(epochs_config_options)),
            format_func=lambda i: epochs_config_options[i],
            help="Select epoch rejection strategy (threshold vs autoreject)"
        )

        epochs_config_file = epochs_config_paths[epochs_selected_idx]
        st.sidebar.success(f"✓ Epochs config: {epochs_configs[epochs_selected_idx][1]}")

    # Process button
    st.sidebar.markdown("---")
    process_runs = st.sidebar.button("🔍 Load Runs", type="primary")

    # Import from HPC section
    st.sidebar.markdown("---")
    st.sidebar.header("🔄 Import from HPC")

    if st.sidebar.button("Import QC JSON", type="secondary", width="stretch"):
        with st.spinner("Importing QC metrics from JSON files..."):
            try:
                from eegcpm.workflow.import_qc import import_qc_metrics_to_state
                from eegcpm.workflow.state import WorkflowStateManager

                # Setup state manager using new paths
                state_db = paths.get_state_db()
                state_db.parent.mkdir(parents=True, exist_ok=True)
                manager = WorkflowStateManager(state_db)

                # Import
                result = import_qc_metrics_to_state(
                    derivatives_path=paths.derivatives_root,
                    state_manager=manager,
                    pipeline=pipeline if pipeline else "standard",
                    task=task if task else None,
                    force=False  # Don't re-import existing
                )

                st.sidebar.success(f"Imported {result['imported']} runs")
                if result['skipped'] > 0:
                    st.sidebar.info(f"Skipped {result['skipped']} existing")
                if result['failed'] > 0:
                    st.sidebar.warning(f"Failed {result['failed']} imports")
            except Exception as e:
                st.sidebar.error(f"Import error: {e}")

    # Initialize session state
    if 'run_results' not in st.session_state:
        st.session_state.run_results = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'manual_selection' not in st.session_state:
        st.session_state.manual_selection = {}

    # Load and process runs
    if process_runs:
        with st.spinner("Loading runs..."):
            try:
                # Load task config
                with open(config_file, 'r') as f:
                    task_config = yaml.safe_load(f)

                # Store task config in session state for later use
                st.session_state.task_config = task_config

                # Load and store epochs config if selected
                if epochs_config_file:
                    with open(epochs_config_file, 'r') as f:
                        st.session_state.epochs_config = yaml.safe_load(f)
                else:
                    st.session_state.epochs_config = None

                # Get task name from config
                task_name_from_config = task_config.get('task_name', task)

                # NEW: Load QC metrics from new structure
                # derivatives/preprocessing/{pipeline}/{subject}/ses-{session}/task-{task}/run-*/
                qc_metrics_list = []

                # Build path to subject's preprocessing output
                subject_path = paths.get_preprocessing_root(pipeline) / f"sub-{subject_id}" / f"ses-{session}" / f"task-{task}"

                if subject_path.exists():
                    # Scan for run directories
                    run_dirs = sorted([d for d in subject_path.iterdir() if d.is_dir() and d.name.startswith('run-')])

                    for run_dir in run_dirs:
                        # Look for QC metrics JSON
                        json_files = list(run_dir.glob('*_qc_metrics.json'))
                        for json_file in json_files:
                            try:
                                from eegcpm.modules.qc.metrics_io import load_qc_metrics_json
                                metrics = load_qc_metrics_json(json_file)
                                metrics['_json_path'] = str(json_file)
                                qc_metrics_list.append(metrics)
                            except Exception as e:
                                st.warning(f"Could not load {json_file}: {e}")
                                continue

                # Convert to RunProcessingResult objects for compatibility
                results = []
                for qc_data in qc_metrics_list:
                    try:
                        # Parse quality metrics from JSON
                        metrics = RunQualityMetricsFromJSON.from_json(qc_data)

                        # Build output path using new structure
                        run_id = metrics.run
                        output_dir = paths.get_preprocessing_dir(
                            pipeline=pipeline,
                            subject=subject_id,
                            session=session,
                            task=task,
                            run=run_id
                        )

                        # Convert to internal RunQualityMetrics format
                        from eegcpm.pipeline.run_processor import RunQualityMetrics

                        # Calculate pct_clustered from n_clustered_bad and n_bad_channels
                        pct_clustered = 0.0
                        if metrics.n_bad_channels > 0 and metrics.n_clustered_bad is not None:
                            pct_clustered = (metrics.n_clustered_bad / metrics.n_bad_channels) * 100.0

                        quality_metrics = RunQualityMetrics(
                            run=run_id,
                            n_original_channels=metrics.n_eeg_channels,
                            n_bad_channels=metrics.n_bad_channels,
                            pct_bad_channels=metrics.pct_bad_channels,
                            n_clustered_bad=metrics.n_clustered_bad if metrics.n_clustered_bad is not None else 0,
                            pct_clustered=pct_clustered,
                            clustering_severity=metrics.clustering_severity,
                            ica_success=metrics.ica_success,
                            n_ica_components_rejected=metrics.n_components_rejected,
                            quality_status=metrics.quality_status,
                            recommended_action=metrics.recommended_action,
                            warnings=[]
                        )

                        # Create RunProcessingResult
                        results.append(RunProcessingResult(
                            run=run_id,
                            success=True,
                            raw_preprocessed=None,  # Don't load into memory
                            output_path=output_dir,
                            qc_path=output_dir / metrics.qc_report_path if metrics.qc_report_path else None,
                            quality_metrics=quality_metrics
                        ))
                    except Exception as e:
                        st.warning(f"Could not parse QC metrics: {e}")
                        continue

                # Get recommendations (use internal function)
                recommendations = {}
                for result in results:
                    if result.quality_metrics:
                        recommendations[result.run] = result.quality_metrics.recommended_action == "accept"
                    else:
                        recommendations[result.run] = False

                # Store in session state
                st.session_state.run_results = results
                st.session_state.recommendations = recommendations
                st.session_state.task_config = task_config
                st.session_state.state_manager = None  # Not used when loading from JSON

                # Initialize manual selection from recommendations
                st.session_state.manual_selection = recommendations.copy()

                st.success(f"✓ Loaded {len(results)} runs from QC metrics JSON")

            except Exception as e:
                st.error(f"Error loading runs: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Display run selection interface
    if st.session_state.run_results:
        results = st.session_state.run_results
        recommendations = st.session_state.recommendations

        st.header(f"📊 Run Quality Assessment: {subject_id}")
        st.markdown(f"**Task:** {task} | **Session:** {session}")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Runs", len(results))

        with col2:
            n_successful = sum(1 for r in results if r.success)
            st.metric("Processed", n_successful)

        with col3:
            n_recommended = sum(1 for accept in recommendations.values() if accept)
            st.metric("Auto-Accepted", n_recommended)

        with col4:
            n_selected = sum(1 for accept in st.session_state.manual_selection.values() if accept)
            st.metric("Selected", n_selected, delta=n_selected - n_recommended)

        st.markdown("---")

        # Run selection table
        st.subheader("🎯 Run Selection")

        # Build dataframe for display
        run_data = []
        for result in results:
            if result.quality_metrics:
                m = result.quality_metrics

                # Construct path to QC report using new structure
                # QC reports are in same directory as preprocessed data
                run_dir = paths.get_preprocessing_dir(
                    pipeline=pipeline,
                    subject=subject_id,
                    session=session,
                    task=task,
                    run=result.run
                )

                qc_file = run_dir / f"{subject_id}_ses-{session}_task-{task}_run-{result.run}_preprocessed_qc.html"
                # Handle sub- prefix variations
                if not qc_file.exists():
                    qc_file = run_dir / f"sub-{subject_id}_ses-{session}_task-{task}_run-{result.run}_preprocessed_qc.html"

                qc_report_path = str(qc_file) if qc_file.exists() else None

                run_data.append({
                    'Run': result.run,
                    'Quality': m.quality_status,
                    'Bad Channels (%)': f"{m.pct_bad_channels:.1f}%",
                    'Clustering': m.clustering_severity,
                    'ICA': '✓' if m.ica_success else '✗',
                    'Recommendation': m.recommended_action,
                    'Auto': '✓' if recommendations.get(result.run) else '✗',
                    'QC Report': qc_report_path,
                    '_qc_exists': qc_report_path is not None
                })

        df = pd.DataFrame(run_data)

        # Display dataframe with styling only if data exists
        if len(df) > 0 and 'Quality' in df.columns and 'Recommendation' in df.columns:
            # Separate QC Report column for button display
            df_display = df.drop(columns=['QC Report', '_qc_exists']).copy()

            # Color code quality
            def color_quality(val):
                colors = {
                    'excellent': 'background-color: #d4edda',
                    'good': 'background-color: #d1ecf1',
                    'acceptable': 'background-color: #fff3cd',
                    'poor': 'background-color: #f8d7da'
                }
                return colors.get(val.lower(), '')

            def color_recommendation(val):
                colors = {
                    'accept': 'background-color: #d4edda',
                    'review': 'background-color: #fff3cd',
                    'reject': 'background-color: #f8d7da'
                }
                return colors.get(val.lower(), '')

            # Use 'map' instead of deprecated 'applymap'
            styled_df = df_display.style.map(color_quality, subset=['Quality'])\
                                        .map(color_recommendation, subset=['Recommendation'])

            # Display table
            st.dataframe(styled_df, width="stretch")

            # Add QC report buttons below table
            st.markdown("**📊 QC Reports**")
            cols = st.columns(min(len(df), 5))
            for idx, row in df.iterrows():
                with cols[idx % 5]:
                    if row['_qc_exists']:
                        # Read QC report for download/display
                        qc_path = Path(row['QC Report'])
                        with open(qc_path, 'rb') as f:
                            qc_content = f.read()

                        # Download button
                        st.download_button(
                            label=f"📊 Run {row['Run']}",
                            data=qc_content,
                            file_name=qc_path.name,
                            mime='text/html',
                            key=f"qc_download_{row['Run']}",
                            width="stretch"
                        )

                        # View button (expander)
                        if st.button(f"👁️ View Run {row['Run']}", key=f"qc_view_{row['Run']}", width="stretch"):
                            st.session_state[f'show_qc_{row["Run"]}'] = not st.session_state.get(f'show_qc_{row["Run"]}', False)

                        # Show QC report in expander if toggled
                        if st.session_state.get(f'show_qc_{row["Run"]}', False):
                            with st.expander(f"QC Report - Run {row['Run']}", expanded=True):
                                with open(qc_path, 'r') as f:
                                    html_content = f.read()
                                st.components.v1.html(html_content, height=600, scrolling=True)
                    else:
                        st.caption(f"Run {row['Run']}: No QC")

        elif len(df) > 0:
            # Display without styling if columns are missing
            st.dataframe(df.drop(columns=['QC Report', '_qc_exists'], errors='ignore'), width="stretch")
        else:
            st.warning("⚠️ No quality metrics available for runs. The runs may not have been processed yet.")

        # Manual selection checkboxes
        st.subheader("✅ Manual Selection")
        st.markdown("Override automatic recommendations by selecting runs to include:")

        cols = st.columns(min(len(results), 4))

        for i, result in enumerate(results):
            with cols[i % 4]:
                if result.quality_metrics:
                    m = result.quality_metrics

                    # Determine checkbox label color
                    if m.quality_status == 'excellent':
                        emoji = "🟢"
                    elif m.quality_status == 'good':
                        emoji = "🔵"
                    elif m.quality_status == 'acceptable':
                        emoji = "🟡"
                    else:
                        emoji = "🔴"

                    # Checkbox
                    default_value = st.session_state.manual_selection.get(result.run, False)
                    selected = st.checkbox(
                        f"{emoji} Run {result.run}",
                        value=default_value,
                        key=f"run_{result.run}",
                        help=f"Quality: {m.quality_status}\nBad channels: {m.pct_bad_channels:.1f}%\nClustering: {m.clustering_severity}"
                    )

                    # Update manual selection
                    st.session_state.manual_selection[result.run] = selected

                    # Show quality summary
                    st.caption(f"{m.quality_status.capitalize()}")
                    st.caption(f"{m.pct_bad_channels:.1f}% bad")

        st.markdown("---")

        # Combine epochs button
        st.subheader("🔬 Epoch Combination")

        n_selected = sum(1 for accept in st.session_state.manual_selection.values() if accept)

        if n_selected == 0:
            st.warning("⚠️ No runs selected. Please select at least one run to combine.")
        elif n_selected == 1:
            st.info("ℹ️ Only 1 run selected. Combination will proceed but works best with multiple runs.")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"**Ready to combine {n_selected} selected runs**")

        with col2:
            combine_button = st.button(
                "🚀 Combine Epochs",
                type="primary",
                disabled=(n_selected == 0),
                width="stretch"
            )

        # Perform combination
        if combine_button:
            with st.spinner("Combining epochs..."):
                try:
                    # Create epoch combiner using NEW epochs stage structure
                    # Output to: derivatives/epochs/{preprocessing}/{task}/{subject}/ses-{session}/
                    combined_output = paths.get_epochs_dir(
                        preprocessing=pipeline,
                        task=task,
                        subject=subject_id,
                        session=session
                    )
                    # Convert task config to epoch combiner format
                    # Pass event codes to filter which events to epoch
                    event_codes_to_epoch = _get_event_codes_to_epoch(st.session_state.task_config)

                    # Load event mapping from BIDS to translate semantic names to numeric codes
                    from eegcpm.data.event_mapping import get_event_mapping_for_run
                    event_mapping = {}
                    if results:
                        # Get mapping from first run (assumes all runs have same mapping)
                        first_run = results[0].run
                        event_mapping = get_event_mapping_for_run(
                            bids_root=paths.bids_root,
                            subject=subject_id,
                            session=session,
                            task=task,
                            run=first_run
                        )

                    # Build epoch config from task config, with epochs config overrides
                    task_cfg = st.session_state.task_config
                    epochs_cfg = st.session_state.get('epochs_config')

                    # Start with task config defaults
                    tmin = task_cfg.get('tmin', -0.5)
                    tmax = task_cfg.get('tmax', 1.0)
                    baseline = task_cfg.get('baseline', [-0.2, 0.0])
                    reject = task_cfg.get('reject')
                    decim = 1
                    detrend = None

                    # Override with epochs config if present
                    if epochs_cfg:
                        tmin = epochs_cfg.get('tmin', tmin)
                        tmax = epochs_cfg.get('tmax', tmax)
                        baseline = epochs_cfg.get('baseline', baseline)
                        decim = epochs_cfg.get('decim', decim)
                        detrend = epochs_cfg.get('detrend', detrend)

                        # Handle rejection settings from epochs config
                        rejection = epochs_cfg.get('rejection', {})
                        if rejection:
                            strategy = rejection.get('strategy', 'none')
                            if strategy == 'threshold':
                                reject = rejection.get('reject')
                            elif strategy == 'autoreject':
                                # Pass autoreject settings
                                reject = {
                                    'use_autoreject': True,
                                    'autoreject_n_interpolate': rejection.get('autoreject_n_interpolate', [1, 4, 8])
                                }
                            elif strategy == 'none':
                                reject = None

                    epoch_config = {
                        'epochs': {
                            'tmin': tmin,
                            'tmax': tmax,
                            'baseline': tuple(baseline) if baseline else None,
                            'reject': reject,
                            'decim': decim,
                            'detrend': detrend,
                            'event_codes_filter': event_codes_to_epoch,  # Filter events based on task config
                            'event_mapping': event_mapping  # For translating semantic names to numeric codes
                        }
                    }

                    combiner = EpochCombiner(
                        config=epoch_config,
                        output_dir=combined_output,
                        state_manager=st.session_state.state_manager,
                        verbose=False,  # Disable print statements in Streamlit
                        bids_root=paths.bids_root  # Pass BIDS root for event mapping
                    )

                    # Combine with manual selection
                    combination_result = combiner.combine_with_selection(
                        run_results=results,
                        selection=st.session_state.manual_selection,
                        subject_id=subject_id,
                        session=session,
                        task=task,
                        pipeline=pipeline,
                        generate_qc=True
                    )

                    if combination_result.success:
                        st.success("✅ Epoch combination successful!")

                        # Display results
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Runs Combined", combination_result.n_runs_combined)

                        with col2:
                            st.metric("Total Epochs", combination_result.n_total_epochs)

                        with col3:
                            st.metric("Runs Included", ", ".join(combination_result.runs_included))

                        # Show output path
                        st.info(f"📁 Output file: `{combination_result.output_path}`")

                        # QC report viewer
                        if combined_output:
                            qc_file = combined_output / f"{subject_id}_ses-{session}_task-{task}_combined_qc.html"
                            if qc_file.exists():
                                st.markdown("### 📊 Quality Control Report")

                                # Download button
                                with open(qc_file, 'rb') as f:
                                    st.download_button(
                                        label="⬇️ Download QC Report",
                                        data=f,
                                        file_name=qc_file.name,
                                        mime='text/html'
                                    )

                                # Embedded viewer
                                with st.expander("🔍 View QC Report (Embedded)", expanded=False):
                                    with open(qc_file, 'r') as f:
                                        html_content = f.read()
                                    st.components.v1.html(html_content, height=800, scrolling=True)

                    else:
                        st.error(f"❌ Combination failed: {combination_result.error}")

                except Exception as e:
                    st.error(f"Error during combination: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    else:
        # Show instructions
        st.info("👈 Configure settings in the sidebar and click 'Load Runs' to begin")

        st.markdown("""
        ### How to Use

        1. **Configure Paths**: Set BIDS root, output directory, and config file
        2. **Select Subject**: Enter subject ID, session, task, and pipeline name
        3. **Load Runs**: Click 'Load Runs' to process and analyze all runs
        4. **Review Quality**: Examine automatic quality assessments
        5. **Manual Selection**: Override recommendations by checking/unchecking runs
        6. **Combine Epochs**: Click 'Combine Epochs' to merge selected runs

        ### Quality Indicators

        - 🟢 **Excellent**: <10% bad channels, no clustering
        - 🔵 **Good**: 10-20% bad channels, mild clustering
        - 🟡 **Acceptable**: 20-30% bad channels, moderate clustering
        - 🔴 **Poor**: >30% bad channels or severe clustering

        ### Recommendations

        - **Accept**: Automatically include in combination
        - **Review**: Manual decision recommended
        - **Reject**: Automatically exclude from combination
        """)


if __name__ == "__main__":
    main()
