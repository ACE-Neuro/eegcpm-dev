"""Preprocessing interface for running the preprocessing pipeline."""

import streamlit as st
from pathlib import Path
import sys
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eegcpm.ui.utils import scan_subjects, scan_sessions, scan_tasks, get_bids_info
from eegcpm.ui.session_persistence import restore_project_from_storage
from eegcpm.pipeline.run_processor import RunProcessor
from eegcpm.workflow.state import WorkflowStateManager
from eegcpm.core.paths import EEGCPMPaths


def main():
    """Preprocessing interface main function."""

    st.set_page_config(
        page_title="Processing: Single Subject - EEGCPM",
        page_icon="⚙️",
        layout="wide"
    )

    st.title("⚙️ Processing: Single Subject Preprocessing")
    st.markdown("Run preprocessing pipeline on individual subject/session/run")

    # Restore project from localStorage if session state was wiped by a refresh
    restore_project_from_storage()

    # Check if project is configured
    if 'bids_root' not in st.session_state or 'eegcpm_root' not in st.session_state:
        st.warning("⚠️ No project configured. Please go to the Home page to set up a project.")
        st.page_link("app.py", label="→ Go to Home", icon="🏠")
        return

    # Get project paths from session state
    bids_root = st.session_state.bids_root
    eegcpm_root = st.session_state.eegcpm_root

    # Initialize path manager
    bids_path = Path(bids_root)
    project_root = bids_path.parent if bids_path.name == "bids" else bids_path
    paths = EEGCPMPaths(project_root)

    # Sidebar - show current project
    st.sidebar.header("📂 Current Project")
    if 'current_project_name' in st.session_state:
        st.sidebar.info(f"**{st.session_state.current_project_name}**")
    st.sidebar.caption(f"BIDS: `{bids_root}`")
    st.sidebar.caption(f"EEGCPM: `{eegcpm_root}`")

    # Config selection from project (not package templates)
    eegcpm_path = Path(eegcpm_root)
    project_config_dir = eegcpm_path / "configs" / "preprocessing"
    template_config_dir = Path(__file__).parent.parent.parent / "config" / "preprocessing"

    # Scan project configs
    project_configs = []
    if project_config_dir.exists():
        project_configs = sorted([f.stem for f in project_config_dir.glob("*.yaml")])

    # Show available configs (project + templates)
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Config Selection")

    if project_configs:
        config_selection = st.sidebar.radio(
            "Project Configs",
            options=project_configs,
            help="Your custom configs in project folder"
        )
        config_file = str(project_config_dir / f"{config_selection}.yaml")
        st.sidebar.success(f"📁 Using project config")
    else:
        # No project configs yet - show templates and prompt to copy
        st.sidebar.warning("⚠️ No project configs found")
        st.sidebar.info("💡 Go to **Pipeline Config** page to copy a template to your project")

        # Fall back to templates for now
        template_configs = sorted([f.stem for f in template_config_dir.glob("*.yaml")])
        if template_configs:
            config_selection = st.sidebar.radio(
                "Templates (Read-Only)",
                options=template_configs,
                help="Package templates - copy to project to customize"
            )
            config_file = str(template_config_dir / f"{config_selection}.yaml")
        else:
            st.sidebar.error("❌ No configs found!")
            config_file = None

    # Scan BIDS directory
    bids_root_path = Path(bids_root)

    if not bids_root_path.exists():
        st.error(f"❌ BIDS root not found: {bids_root}")
        return

    # Get dataset info
    bids_info = get_bids_info(bids_root_path)

    st.sidebar.markdown("---")
    st.sidebar.header("📊 Dataset Info")
    st.sidebar.metric("Subjects", bids_info['n_subjects'])
    st.sidebar.metric("Sessions", bids_info['n_sessions'])
    st.sidebar.metric("Tasks", len(bids_info['tasks']))

    # Subject selection
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Selection")

    if not bids_info['subjects']:
        st.warning("No subjects found in BIDS directory")
        return

    # Subject dropdown
    subject_id = st.sidebar.selectbox(
        "Subject",
        options=bids_info['subjects'],
        help="Select subject to preprocess"
    )

    # Session dropdown
    sessions = scan_sessions(bids_root_path, subject_id)
    if not sessions:
        st.warning(f"No sessions found for subject {subject_id}")
        return

    session = st.sidebar.selectbox(
        "Session",
        options=sessions,
        help="Select session to preprocess"
    )

    # Task dropdown
    tasks = scan_tasks(bids_root_path, subject_id)
    if not tasks:
        st.warning(f"No tasks found for subject {subject_id}")
        return

    task = st.sidebar.selectbox(
        "Task",
        options=tasks,
        help="Select task to preprocess"
    )

    # Task config selection (optional - for ERP QC)
    # Filter by task_name field to show only configs for selected task
    task_config_dir = paths.get_configs_dir("tasks")
    matching_configs = []  # [(filename_stem, description)]

    if task_config_dir.exists():
        for task_config_path in task_config_dir.glob("*.yaml"):
            try:
                import yaml
                with open(task_config_path) as f:
                    task_cfg = yaml.safe_load(f)

                # Check if task_name matches selected task
                if task_cfg.get('task_name') == task:
                    description = task_cfg.get('description', 'No description')
                    matching_configs.append((task_config_path.stem, description))
            except Exception:
                continue

    # Sort by filename
    matching_configs.sort(key=lambda x: x[0])

    # Build dropdown options
    task_config_options = ["None (skip ERP QC)"]
    task_config_display = task_config_options.copy()

    for filename, description in matching_configs:
        task_config_options.append(filename)
        task_config_display.append(f"{filename} - {description}")

    # Default: try exact match first, then first available
    default_idx = 0
    if matching_configs:
        exact_match = [cfg for cfg in matching_configs if cfg[0] == task]
        if exact_match:
            default_idx = task_config_options.index(exact_match[0][0])
        else:
            default_idx = 1  # First matching config (skip "None")

    selected_display = st.sidebar.selectbox(
        "Task Config (for ERP QC)",
        options=range(len(task_config_display)),
        format_func=lambda i: task_config_display[i],
        index=default_idx,
        help="Optional: Select task config to generate ERP plots in QC report\n"
             "Shows only configs matching the selected task"
    )

    selected_task_config = task_config_options[selected_display]

    # Store in session state for CLI to use
    if selected_task_config == "None (skip ERP QC)":
        st.session_state.task_config_for_qc = None
    else:
        st.session_state.task_config_for_qc = selected_task_config

    # Processing options
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Processing Options")

    # Pipeline name - scan existing pipelines in derivatives/preprocessing/
    preprocessing_root = paths.derivatives_root / "preprocessing"

    existing_pipelines = []
    if preprocessing_root.exists():
        # Exclude special directories that aren't pipelines
        excluded_dirs = {'logs', 'qc', 'reports', '__pycache__'}
        existing_pipelines = sorted([
            d.name for d in preprocessing_root.iterdir()
            if d.is_dir() and not d.name.startswith('.') and d.name not in excluded_dirs
        ])

    # Add option to create new pipeline
    pipeline_options = ['Create New...'] + existing_pipelines

    selected_pipeline_option = st.sidebar.selectbox(
        "Output Pipeline",
        options=pipeline_options,
        help="Select existing pipeline variant or create new one"
    )

    # If "Create New..." is selected, show text input
    if selected_pipeline_option == 'Create New...':
        # Auto-suggest pipeline name based on selected config
        default_name = config_selection if 'config_selection' in locals() else "preprocessing"

        pipeline = st.sidebar.text_input(
            "New Pipeline Name",
            value=default_name,
            help="Name for output folder (e.g., 'standard', 'optimal', 'minimal')"
        )
    else:
        pipeline = selected_pipeline_option

    force_reprocess = st.sidebar.checkbox(
        "Force Reprocess",
        value=False,
        help="Reprocess even if output files already exist"
    )

    # Main content
    st.header(f"📋 Preprocessing Configuration")

    # Check if config_file was set
    if config_file is None:
        st.error("❌ No preprocessing config selected or found")
        st.info("💡 Create preprocessing configs in the **Pipeline Config** page")
        return

    # Load and display config
    config_path = Path(config_file)
    if not config_path.exists():
        st.error(f"❌ Config file not found: {config_file}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Display config file info
    st.caption(f"📄 Config: `{config_path.name}`")

    # Display config summary (new modular format)
    preproc_config = config.get('preprocessing', {})
    steps = preproc_config.get('steps', [])

    if steps:
        st.subheader("Pipeline Steps")

        # Display steps in order
        for i, step in enumerate(steps, 1):
            step_name = step.get('name', 'unknown')
            params = step.get('params', {})

            # Format step display
            if step_name == 'montage':
                st.markdown(f"{i}. **📍 Montage**: {params.get('type', 'standard_1020')}")

            elif step_name == 'filter':
                l_freq = params.get('l_freq', 'None')
                h_freq = params.get('h_freq', 'None')
                st.markdown(f"{i}. **🌊 Filter**: {l_freq}-{h_freq} Hz ({params.get('method', 'fir')})")

            elif step_name == 'drop_flat':
                threshold = params.get('variance_threshold', 1e-15)
                st.markdown(f"{i}. **🔻 Drop Flat**: variance < {threshold:.0e}")

            elif step_name == 'bad_channels':
                mark_only = " (mark only)" if params.get('mark_only') else ""
                drop_str = " [drop]" if params.get('drop') else ""
                st.markdown(f"{i}. **🔴 Bad Channels**: {params.get('method', 'ransac')}{mark_only}{drop_str}")

            elif step_name == 'zapline':
                fline = params.get('fline', 'auto')
                fline_str = 'auto-detect' if fline is None else f"{fline} Hz"
                adaptive_str = " (adaptive)" if params.get('adaptive', True) else ""
                st.markdown(f"{i}. **⚡ Zapline**: {fline_str}{adaptive_str}")

            elif step_name == 'notch':
                freqs = params.get('freqs', [])
                if isinstance(freqs, list) and len(freqs) > 0:
                    freq_str = f"{freqs[0]} Hz" if len(freqs) == 1 else f"{freqs[0]} Hz + {len(freqs)-1} harmonics"
                else:
                    freq_str = "60 Hz"
                st.markdown(f"{i}. **🔕 Notch**: {freq_str}")

            elif step_name == 'lowpass':
                h_freq = params.get('h_freq', 40)
                method = params.get('method', 'fir')
                st.markdown(f"{i}. **⬇️ Lowpass**: {h_freq} Hz ({method})")

            elif step_name == 'artifacts':
                threshold = params.get('amplitude_threshold', 500e-6) * 1e6  # Convert to µV
                st.markdown(f"{i}. **⚠️ Artifacts**: threshold={threshold:.0f}µV")

            elif step_name == 'asr':
                cutoff = params.get('cutoff', 20)
                mode = params.get('mode', 'single')
                method = params.get('method', 'eegprep')
                st.markdown(f"{i}. **⚡ ASR ({mode})**: cutoff={cutoff}, {method}")

            elif step_name == 'ica':
                n_comp = params.get('n_components', 'rank-1')
                method = params.get('method', 'picard')
                st.markdown(f"{i}. **🧠 ICA**: {n_comp} components, {method}")

            elif step_name == 'iclabel':
                threshold = params.get('threshold', 0.8)
                labels = params.get('reject_labels', [])
                st.markdown(f"{i}. **🏷️ ICLabel**: threshold={threshold}, reject {labels}")

            elif step_name == 'interpolate':
                max_pct = params.get('max_bad_percent', 50)
                st.markdown(f"{i}. **🔧 Interpolate**: max {max_pct}% bad channels")

            elif step_name == 'reference':
                ref_type = params.get('type', 'average')
                st.markdown(f"{i}. **📌 Reference**: {ref_type}")

            elif step_name == 'resample':
                sfreq = params.get('sfreq', 256)
                st.markdown(f"{i}. **⏱️ Resample**: {sfreq} Hz")

            else:
                st.markdown(f"{i}. **{step_name}**: {params}")
    else:
        st.warning("⚠️ No pipeline steps configured")

    # Output path - use new stage-first structure
    output_root = paths.get_preprocessing_root(pipeline)
    st.info(f"📁 Output: `{output_root}/`")

    # Run preprocessing button
    st.markdown("---")

    # Initialize session state for results
    if 'preprocessing_results' not in st.session_state:
        st.session_state.preprocessing_results = None
    if 'preprocessing_subject' not in st.session_state:
        st.session_state.preprocessing_subject = None

    if st.button("🚀 Run Preprocessing", type="primary", width="stretch"):
        # Setup state manager using new paths
        state_db = paths.get_state_db()
        state_db.parent.mkdir(parents=True, exist_ok=True)
        state_manager = WorkflowStateManager(state_db)

        # Handle force reprocess - clear old outputs
        if force_reprocess:
            from eegcpm.data.bids_utils import find_subject_runs
            import shutil

            st.info("🔄 Force reprocess enabled - clearing old outputs...")

            # Find runs for this subject/task
            runs = find_subject_runs(bids_root_path, subject_id, task, session=session)

            # Clear output directories for each run
            for bids_file in runs:
                run_id = bids_file.run or "01"
                run_output = paths.get_preprocessing_dir(
                    pipeline=pipeline,
                    subject=subject_id,
                    session=session,
                    task=task,
                    run=run_id
                )

                if run_output.exists():
                    st.write(f"  Clearing: {run_output}")
                    shutil.rmtree(run_output)

                # Clear workflow state for this run
                if state_manager:
                    state_manager.delete_state(
                        subject_id=subject_id,
                        task=task,
                        pipeline=pipeline,
                        session=session,
                        run=run_id
                    )

        # Create processor
        processor = RunProcessor(
            bids_root=bids_root_path,
            output_root=output_root,
            config=config,
            state_manager=state_manager,
            verbose=False  # Disable print in Streamlit
        )

        # Run preprocessing
        with st.spinner(f"Processing {subject_id} task {task} session {session}..."):
            try:
                results = processor.process_subject_task(
                    subject_id=subject_id,
                    task=task,
                    session=session,
                    pipeline=pipeline
                )

                # Store results in session state
                st.session_state.preprocessing_results = results
                st.session_state.preprocessing_subject = subject_id

                # Display results
                st.success(f"✅ Processed {len(results)} runs")

            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                with st.expander("Show traceback"):
                    st.code(traceback.format_exc())

    # Display results (either from just-run or from session state)
    results = st.session_state.preprocessing_results
    if results:
        st.header("📊 Processing Results")

        result_data = []
        for r in results:
            if r.quality_metrics:
                m = r.quality_metrics

                # Format clustering with details
                if m.clustering_severity == 'none':
                    clustering_str = 'none'
                else:
                    # Calculate percentage of bad channels that are clustered
                    pct_clustered = (m.n_clustered_bad / m.n_bad_channels * 100) if m.n_bad_channels > 0 else 0
                    clustering_str = f"{m.clustering_severity} ({m.n_clustered_bad}/{m.n_bad_channels}, {pct_clustered:.0f}%)"

                result_data.append({
                    'Run': r.run,
                    'Status': '✓' if r.success else '✗',
                    'Quality': m.quality_status,
                    'Bad Channels': f"{m.pct_bad_channels:.1f}%",
                    'Clustering': clustering_str,
                    'ICA': '✓' if m.ica_success else '✗',
                    'Recommendation': m.recommended_action
                })

        if result_data:
            df = pd.DataFrame(result_data)
            st.dataframe(df, width='stretch')

            # Summary
            n_accept = sum(1 for r in results if r.quality_metrics and r.quality_metrics.recommended_action == 'accept')
            n_reject = sum(1 for r in results if r.quality_metrics and r.quality_metrics.recommended_action == 'reject')
            n_review = sum(1 for r in results if r.quality_metrics and r.quality_metrics.recommended_action == 'review')

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Auto-Accept", n_accept)
            with col2:
                st.metric("Review", n_review)
            with col3:
                st.metric("Auto-Reject", n_reject)

            if n_accept + n_review >= 2:
                st.info("✅ Ready for run selection and epoch combination")
            else:
                st.warning("⚠️ Fewer than 2 acceptable runs - epoch combination may not be possible")

        # Per-run logs and QC reports
        st.markdown("---")
        st.header("📄 Logs & QC Reports")

        for r in results:
            run_label = f"Run {r.run}"
            status_emoji = "✅" if r.success else "❌"

            with st.expander(f"{status_emoji} {run_label} - Details", expanded=False):
                # Run info
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"**Status:** {'Success' if r.success else 'Failed'}")
                    if r.quality_metrics:
                        st.markdown(f"**Quality:** {r.quality_metrics.quality_status}")

                with col2:
                    if r.quality_metrics:
                        st.markdown(f"**Bad Channels:** {r.quality_metrics.pct_bad_channels:.1f}%")
                        st.markdown(f"**Clustering:** {r.quality_metrics.clustering_severity}")

                with col3:
                    if r.quality_metrics:
                        st.markdown(f"**ICA:** {'Success' if r.quality_metrics.ica_success else 'Failed'}")
                        st.markdown(f"**Action:** {r.quality_metrics.recommended_action}")

                # Processing log (if available)
                if hasattr(r, 'logs') and r.logs:
                    st.markdown("### 📋 Processing Log")
                    log_text = "\n".join(r.logs)
                    st.text_area(
                        f"Log for run {r.run}",
                        value=log_text,
                        height=200,
                        key=f"log_{r.run}",
                        label_visibility="collapsed"
                    )
                else:
                    st.info("ℹ️ Processing logs not available (run was processed before logging was implemented)")

                # Error message if failed
                if not r.success and r.error:
                    st.markdown("### ❌ Error")
                    st.error(r.error)

                # QC Report link (use EEGCPMPaths for correct structure)
                qc_dir = paths.get_preprocessing_dir(
                    pipeline=pipeline,
                    subject=subject_id,
                    session=session,
                    task=task,
                    run=r.run
                )
                qc_file = qc_dir / f"{subject_id}_ses-{session}_task-{task}_run-{r.run}_preprocessed_qc.html"

                if qc_file.exists():
                    st.markdown("### 📊 QC Report")

                    col1, col2 = st.columns([1, 3])

                    with col1:
                        with open(qc_file, 'rb') as f:
                            st.download_button(
                                label="⬇️ Download QC",
                                data=f,
                                file_name=qc_file.name,
                                mime='text/html',
                                key=f"qc_download_{r.run}",
                                width="stretch"
                            )

                    with col2:
                        st.code(str(qc_file), language=None)

                    # Option to view inline (default: True)
                    if st.checkbox(f"View QC Report Inline", value=True, key=f"view_qc_{r.run}"):
                        with st.spinner("Loading QC report..."):
                            try:
                                with open(qc_file, 'r') as f:
                                    html_content = f.read()
                                st.components.v1.html(html_content, height=800, scrolling=True)
                            except Exception as e:
                                st.error(f"Error loading QC report: {e}")
                else:
                    st.warning("⚠️ QC report not found")


if __name__ == "__main__":
    import pandas as pd
    main()
