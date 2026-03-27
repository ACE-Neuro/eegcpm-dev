"""Epochs Extraction - Batch Mode.

Multi-subject batch epoch extraction with QC report generation.
Use this page for production processing of many subjects with consistent settings.
"""

import streamlit as st
from pathlib import Path
import yaml
from typing import Dict, Any

from eegcpm.core.paths import EEGCPMPaths
from eegcpm.ui.utils.executor import run_eegcpm_command
from eegcpm.ui.session_persistence import restore_project_from_storage


def main():
    """Epochs batch processing page."""
    st.set_page_config(
        page_title="Epochs: Batch - EEGCPM",
        page_icon="⚡",
        layout="wide"
    )

    st.title("⚡ Epochs Extraction - Batch")
    st.markdown("""
    Multi-subject mode: Extract epochs from all subjects with consistent task configuration.
    Generates QC reports for each subject.
    """)

    # Restore project from localStorage if session state was wiped by a refresh
    restore_project_from_storage()

    # Check for project root
    if 'eegcpm_root' not in st.session_state:
        st.warning("⚠️ No project configured. Please select a project on the home page.")
        return

    project_root = Path(st.session_state.eegcpm_root).parent
    paths = EEGCPMPaths(project_root)

    # Create tabs
    tab_run, tab_results = st.tabs(["▶️ Run", "📊 Results"])

    # Tab 1: Run
    with tab_run:
        st.header("Run Epochs Extraction")

        # Preprocessing pipeline selection (needed to find available tasks)
        st.subheader("Preprocessing Selection")

        preprocessing_dir = paths.derivatives_root / "preprocessing"
        if not preprocessing_dir.exists():
            st.warning("⚠️ No preprocessing data found. Run preprocessing first.")
            return

        # Get all preprocessing pipelines
        preprocessing_options = [d.name for d in preprocessing_dir.iterdir()
                                if d.is_dir() and not d.name.startswith('.')]

        if not preprocessing_options:
            st.warning("⚠️ No preprocessing pipelines found. Run preprocessing first.")
            return

        preprocessing = st.selectbox(
            "Preprocessing Pipeline",
            options=preprocessing_options,
            help="Which preprocessing pipeline to use"
        )

        # Task selection - scan preprocessing directory for available tasks
        st.subheader("Task Selection")

        # Scan preprocessing directory for task names
        all_tasks = set()
        if preprocessing:
            pipeline_path = preprocessing_dir / preprocessing
            # Scan all subjects for task directories
            for subject_dir in pipeline_path.glob("sub-*"):
                for session_dir in subject_dir.glob("ses-*"):
                    for task_dir in session_dir.glob("task-*"):
                        # Extract task name from directory
                        task_name = task_dir.name.replace("task-", "")
                        # Verify it has preprocessed data
                        if list(task_dir.rglob("*preprocessed_raw.fif")):
                            all_tasks.add(task_name)

        if not all_tasks:
            st.warning(f"⚠️ No tasks found in preprocessing pipeline '{preprocessing}'.")
            return

        selected_task = st.selectbox(
            "Select Task",
            options=sorted(all_tasks),
            help="Which task to extract epochs for"
        )

        # Config selection - filter by task_name
        config_dir = project_root / "eegcpm" / "configs" / "tasks"
        matching_configs = []  # [(filepath, name, description)]

        if config_dir.exists():
            for config_file in config_dir.glob("*.yaml"):
                try:
                    with open(config_file) as f:
                        cfg = yaml.safe_load(f)

                    # Filter by task_name
                    if cfg.get('task_name') == selected_task:
                        matching_configs.append((
                            config_file,
                            config_file.stem,
                            cfg.get('description', 'No description')
                        ))
                except Exception:
                    continue

        if not matching_configs:
            st.warning(f"⚠️ No task configs found for task '{selected_task}'. Create one in the Task Config page.")
            return

        # Show config dropdown with descriptions
        config_options = [f"{name} - {desc}" for _, name, desc in matching_configs]
        config_paths = [path for path, _, _ in matching_configs]

        selected_config_idx = st.selectbox(
            "Select Task Configuration Variant",
            options=range(len(config_options)),
            format_func=lambda i: config_options[i],
            help="Select which variant of the task config to use (e.g., stimulus-locked vs response-locked)"
        )

        config_path = config_paths[selected_config_idx]

        # Load and display task config
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        with st.expander("📄 View Task Configuration"):
            st.code(yaml.dump(config_data, default_flow_style=False, sort_keys=False), language="yaml")

        # Get task name from config
        task_name = config_data.get('task_name', selected_task)

        # Epochs Config Selection
        st.subheader("Epochs Configuration")

        epochs_config_dir = paths.get_configs_dir("epochs")
        epochs_configs = []  # [(filepath, variant, description)]

        if epochs_config_dir.exists():
            for epochs_config_path in epochs_config_dir.glob("*.yaml"):
                try:
                    with open(epochs_config_path) as f:
                        epochs_cfg = yaml.safe_load(f)

                    # Check it's an epochs config
                    if epochs_cfg.get('stage') == 'epochs':
                        variant = epochs_cfg.get('variant', epochs_config_path.stem)
                        notes = epochs_cfg.get('notes', 'No description')
                        # Get first line of notes for display
                        description = notes.split('\n')[0][:60] if notes else 'No description'
                        epochs_configs.append((epochs_config_path, variant, description))
                except Exception:
                    continue

        if not epochs_configs:
            st.warning("⚠️ No epochs configs found. Using task config settings.")
            epochs_config_path = None
        else:
            # Show dropdown with variant and description
            epochs_config_options = [f"{variant} - {desc}" for _, variant, desc in epochs_configs]
            epochs_config_paths_list = [path for path, _, _ in epochs_configs]

            epochs_selected_idx = st.selectbox(
                "Rejection Strategy",
                options=range(len(epochs_config_options)),
                format_func=lambda i: epochs_config_options[i],
                help="Select epoch rejection strategy (threshold vs autoreject)"
            )

            epochs_config_path = epochs_config_paths_list[epochs_selected_idx]

            # Load and display epochs config
            with open(epochs_config_path) as f:
                epochs_config_data = yaml.safe_load(f)

            with st.expander("📄 View Epochs Configuration"):
                st.code(yaml.dump(epochs_config_data, default_flow_style=False, sort_keys=False), language="yaml")

        # Subject selection
        st.subheader("Subject Selection")

        # Find subjects with this preprocessing and task
        if preprocessing:
            subject_dirs = []
            for prep_subject_dir in (preprocessing_dir / preprocessing).glob("sub-*"):
                if prep_subject_dir.is_dir():
                    # Check if this subject has the task (look for task directory OR preprocessed files)
                    task_dirs = list(prep_subject_dir.rglob(f"task-{task_name}"))
                    if task_dirs:
                        # Check for preprocessed files in task directory
                        task_files = list(task_dirs[0].rglob("*preprocessed_raw.fif"))
                        if task_files:
                            subject_dirs.append(prep_subject_dir.name.replace("sub-", ""))

            if subject_dirs:
                st.info(f"📊 Found {len(subject_dirs)} subjects with {task_name} data")

                process_all = st.checkbox("Process All Subjects", value=True)

                if not process_all:
                    selected_subjects = st.multiselect(
                        "Select Subjects",
                        options=subject_dirs,
                        default=subject_dirs[:5] if len(subject_dirs) >= 5 else subject_dirs
                    )
                else:
                    selected_subjects = subject_dirs

                st.write(f"Will process: {len(selected_subjects)} subjects")

                # Run button
                if st.button("▶️ Run Epochs Extraction", type="primary", width='stretch'):
                    # Prepare command arguments
                    # Use epochs config if available, otherwise use task config
                    config_to_use = str(epochs_config_path) if epochs_config_path else str(config_path)

                    args = {
                        'project': str(project_root),
                        'config': config_to_use,
                        'preprocessing': preprocessing,
                        'task': task_name
                    }

                    if not process_all and selected_subjects:
                        args['subjects'] = selected_subjects

                    # Create containers for output
                    status_container = st.empty()
                    log_container = st.container()

                    # Execute command
                    success = run_eegcpm_command(
                        command='epochs',
                        args=args,
                        log_container=log_container,
                        status_container=status_container
                    )

                    if success:
                        st.balloons()
                        st.success("✅ Epochs extraction completed! Check the Results tab.")

            else:
                st.warning(f"⚠️ No subjects found with {task_name} data in {preprocessing} pipeline.")

    # Tab 2: Results
    with tab_results:
        st.header("Epochs Results")

        # Get available epochs
        epochs_dir = paths.derivatives_root / "epochs"
        if not epochs_dir.exists():
            st.info("ℹ️ No epochs results yet. Run epochs extraction first.")
            return

        # Let user select preprocessing/task
        col_res1, col_res2 = st.columns(2)

        with col_res1:
            preprocessing_dirs = [d.name for d in epochs_dir.iterdir() if d.is_dir()]
            if not preprocessing_dirs:
                st.info("No preprocessing pipelines found")
                return
            res_preprocessing = st.selectbox("Preprocessing", preprocessing_dirs, key="res_prep")

        with col_res2:
            if res_preprocessing:
                task_dirs = [d.name for d in (epochs_dir / res_preprocessing).iterdir() if d.is_dir()]
                res_task = st.selectbox("Task", task_dirs, key="res_task")
            else:
                res_task = None

        if res_preprocessing and res_task:
            task_dir = epochs_dir / res_preprocessing / res_task

            # Get subjects
            subject_dirs = sorted([d for d in task_dir.glob("sub-*") if d.is_dir()])
            subject_names = [d.name for d in subject_dirs]

            # Generate summary report
            st.subheader("📊 Summary Report")

            summary_data = []
            for subject_dir in subject_dirs:
                subject_id = subject_dir.name.replace("sub-", "")

                # Find session dirs
                session_dirs = list(subject_dir.glob("ses-*"))
                if not session_dirs:
                    continue

                for session_dir in session_dirs:
                    session_id = session_dir.name.replace("ses-", "")

                    # Look for epochs file
                    epochs_files = list(session_dir.glob("*_epo.fif"))
                    qc_files = list(session_dir.glob("*_epochs_qc.html"))

                    if epochs_files:
                        # Try to load epoch count from filename or file
                        import mne
                        try:
                            epochs = mne.read_epochs(epochs_files[0], preload=False, verbose=False)
                            n_epochs = len(epochs)
                            n_conditions = len(epochs.event_id) if hasattr(epochs, 'event_id') else 0
                            conditions = list(epochs.event_id.keys()) if hasattr(epochs, 'event_id') else []
                        except:
                            n_epochs = "N/A"
                            n_conditions = "N/A"
                            conditions = []

                        summary_data.append({
                            'Subject': subject_id,
                            'Session': session_id,
                            'Epochs': n_epochs,
                            'Conditions': n_conditions,
                            'Condition Names': ', '.join(conditions) if conditions else 'N/A',
                            'QC Report': '✓' if qc_files else '✗',
                            '_subject_dir': subject_dir.name
                        })

            if summary_data:
                import pandas as pd
                df_summary = pd.DataFrame(summary_data)

                # Display summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Subjects", len(df_summary))
                with col2:
                    avg_epochs = df_summary['Epochs'].apply(lambda x: x if isinstance(x, int) else 0).mean()
                    st.metric("Avg Epochs", f"{avg_epochs:.0f}")
                with col3:
                    n_with_qc = (df_summary['QC Report'] == '✓').sum()
                    st.metric("With QC Reports", n_with_qc)
                with col4:
                    unique_conditions = df_summary['Conditions'].apply(lambda x: x if isinstance(x, int) else 0).max()
                    st.metric("Max Conditions", unique_conditions)

                # Display summary table
                st.dataframe(
                    df_summary.drop(columns=['_subject_dir']),
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("---")

            st.subheader("📁 Individual Subject Results")

            if subject_dirs:
                # Initialize session state for subject index
                if 'epochs_subject_idx' not in st.session_state:
                    st.session_state.epochs_subject_idx = 0

                # Navigation buttons
                col_nav1, col_nav2, col_nav3 = st.columns([1, 3, 1])

                with col_nav1:
                    if st.button("⬅️ Previous", disabled=(st.session_state.epochs_subject_idx == 0)):
                        st.session_state.epochs_subject_idx = max(0, st.session_state.epochs_subject_idx - 1)
                        st.rerun()

                with col_nav2:
                    # Subject dropdown (can also be used for navigation)
                    current_subject = subject_names[st.session_state.epochs_subject_idx]
                    selected_subject = st.selectbox(
                        "Select Subject",
                        options=subject_names,
                        index=st.session_state.epochs_subject_idx,
                        key="subject_selector"
                    )

                    # Update index if dropdown changed
                    if selected_subject != current_subject:
                        st.session_state.epochs_subject_idx = subject_names.index(selected_subject)
                        st.rerun()

                with col_nav3:
                    if st.button("Next ➡️", disabled=(st.session_state.epochs_subject_idx >= len(subject_names) - 1)):
                        st.session_state.epochs_subject_idx = min(len(subject_names) - 1, st.session_state.epochs_subject_idx + 1)
                        st.rerun()

                # Display current subject info
                selected_subject = subject_names[st.session_state.epochs_subject_idx]
                subject_dir = task_dir / selected_subject

                st.caption(f"Showing subject {st.session_state.epochs_subject_idx + 1} of {len(subject_names)}")

                # Find session
                session_dirs = [d for d in subject_dir.glob("ses-*") if d.is_dir()]
                if session_dirs:
                    session_dir = session_dirs[0]
                else:
                    session_dir = subject_dir

                # Load epochs data for ERP visualization
                epochs_files = list(session_dir.glob("*_epo.fif"))

                if epochs_files:
                    try:
                        import mne
                        import matplotlib.pyplot as plt
                        import numpy as np

                        epochs = mne.read_epochs(epochs_files[0], preload=True, verbose=False)

                        # Display epoch info
                        col_info1, col_info2, col_info3 = st.columns(3)
                        with col_info1:
                            st.metric("Total Epochs", len(epochs))
                        with col_info2:
                            st.metric("Conditions", len(epochs.event_id) if hasattr(epochs, 'event_id') else 0)
                        with col_info3:
                            st.metric("Channels", len(epochs.ch_names))

                        # Plot ERPs
                        st.markdown("**📈 Event-Related Potentials (ERPs)**")

                        if hasattr(epochs, 'event_id') and len(epochs.event_id) > 0:
                            # Create ERP plots for each condition
                            for condition in epochs.event_id.keys():
                                with st.expander(f"Condition: {condition}", expanded=True):
                                    try:
                                        evoked = epochs[condition].average()

                                        # Plot evoked response
                                        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

                                        # Butterfly plot (all channels)
                                        evoked.plot(axes=axes[0], show=False, spatial_colors=True,
                                                   gfp=True, time_unit='s')
                                        axes[0].set_title(f"{condition} - All Channels (Butterfly)")

                                        # Topographic plot at peak
                                        # Find global field power peak
                                        gfp = np.std(evoked.data, axis=0)
                                        peak_time = evoked.times[np.argmax(gfp)]

                                        evoked.plot_topomap(times=[peak_time], axes=axes[1],
                                                           show=False, time_unit='s',
                                                           colorbar=True)
                                        axes[1].set_title(f"{condition} - Topography at Peak ({peak_time*1000:.0f} ms)")

                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close(fig)

                                        # Show trial count
                                        st.caption(f"Based on {len(epochs[condition])} trials")

                                    except Exception as e:
                                        st.warning(f"Could not plot ERP for {condition}: {e}")
                        else:
                            st.info("No conditions found in epochs")

                    except Exception as e:
                        st.warning(f"Could not load epochs data: {e}")

                # List output files
                st.markdown("**📄 Output Files**")

                output_files = list(session_dir.iterdir())
                for f in output_files:
                    if f.is_file():
                        st.write(f"📄 {f.name} ({f.stat().st_size / 1024:.1f} KB)")

                # Show QC report if exists
                qc_html = list(session_dir.glob("*_epochs_qc.html"))
                if qc_html:
                    st.markdown("**📋 QC Report**")

                    # Add button to open in browser
                    if st.button("🌐 Open QC Report in Browser"):
                        import webbrowser
                        webbrowser.open(f"file://{qc_html[0]}")

                    # Embed in page
                    with open(qc_html[0]) as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=800, scrolling=True)
            else:
                st.info("No subjects found with epochs")


if __name__ == "__main__":
    main()
