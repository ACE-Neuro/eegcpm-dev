"""Trial Sorting - Sort epochs by condition, response, and bins.

This page allows you to:
1. Load combined epochs from run selection
2. Sort trials by experimental conditions
3. Filter by behavioral responses (correct/incorrect/etc.)
4. Bin trials by variables (RT quartiles, difficulty, etc.)
5. Preview trial counts and distributions
6. Save sorted epochs for analysis
"""

import streamlit as st
from pathlib import Path
import sys
import yaml
import pandas as pd
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eegcpm.ui.utils import scan_subjects, scan_sessions, scan_tasks
from eegcpm.core.paths import EEGCPMPaths
from eegcpm.core.task_config import TaskConfig, TrialSorter
import mne


def load_combined_epochs(epochs_dir: Path) -> tuple:
    """Load combined epochs and metadata."""
    # Look for combined epochs file
    epoch_files = list(epochs_dir.glob("*_epo.fif"))

    if not epoch_files:
        return None, None, "No epoch files found"

    # Load first epoch file
    epochs = mne.read_epochs(epoch_files[0], preload=True, verbose=False)

    # Load metadata if available
    metadata_file = epochs_dir / "metadata.csv"
    metadata = None
    if metadata_file.exists():
        metadata = pd.read_csv(metadata_file)

    return epochs, metadata, None


def main():
    """Trial sorting interface."""

    st.set_page_config(
        page_title="Processing: Trial Sorting - EEGCPM",
        page_icon="⚙️",
        layout="wide"
    )

    st.title("⚙️ Processing: Trial Sorting")
    st.markdown("Sort and filter epochs by experimental conditions, responses, and bins")

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

    # Derive project root from bids_root
    bids_path = Path(project.bids_root)
    project_root = bids_path.parent if bids_path.name == "bids" else bids_path
    paths = EEGCPMPaths(project_root)

    # Task config selection
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Task Configuration")

    task_config_dir = paths.eegcpm_root / "configs" / "tasks"
    task_config_dir.mkdir(parents=True, exist_ok=True)

    available_task_configs = list(task_config_dir.glob("*.yaml"))

    if not available_task_configs:
        st.sidebar.warning("⚠️ No task configs found")
        st.sidebar.info("📝 Create a task config in `eegcpm/configs/tasks/`")
        st.sidebar.code("""
# Example: contdet.yaml
task_name: contdet
conditions:
  - name: target
    event_codes: [1]
  - name: non_target
    event_codes: [2]
        """)
        st.stop()

    selected_task_config = st.sidebar.selectbox(
        "Task Config",
        options=available_task_configs,
        format_func=lambda x: x.stem,
        help="Select task configuration"
    )

    # Load task config
    try:
        task_config = TaskConfig.from_yaml(selected_task_config)
        st.sidebar.success(f"✓ Loaded: {task_config.task_name}")
    except Exception as e:
        st.sidebar.error(f"Error loading config: {e}")
        st.stop()

    # Pipeline/subject/session/task selection
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Data Selection")

    # Scan available preprocessing pipelines
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
        st.warning("⚠️ No preprocessing pipelines found. Run preprocessing first.")
        st.stop()

    selected_pipeline = st.sidebar.selectbox(
        "Preprocessing Pipeline",
        options=pipeline_options,
        help="Select preprocessing variant"
    )

    # Scan available epochs (from run selection output)
    epochs_root = paths.derivatives_root / "epochs" / selected_pipeline / task_config.task_name

    if not epochs_root.exists():
        st.info(f"📁 No epochs found for {task_config.task_name}")
        st.info("👈 Run **Run Selection** first to combine epochs")
        st.stop()

    # Get subjects with epochs
    subjects_with_epochs = []
    if epochs_root.exists():
        subjects_with_epochs = sorted([
            d.name.replace('sub-', '') for d in epochs_root.iterdir()
            if d.is_dir() and d.name.startswith('sub-')
        ])

    if not subjects_with_epochs:
        st.info("No combined epochs found. Run **Run Selection** to combine runs first.")
        st.stop()

    subject_id = st.sidebar.selectbox(
        "Subject",
        options=subjects_with_epochs,
        help="Select subject"
    )

    # Get sessions for this subject
    subject_epochs_dir = epochs_root / f"sub-{subject_id}"
    sessions = sorted([
        d.name.replace('ses-', '') for d in subject_epochs_dir.iterdir()
        if d.is_dir() and d.name.startswith('ses-')
    ])

    session = st.sidebar.selectbox(
        "Session",
        options=sessions if sessions else ["01"],
        help="Select session"
    )

    epochs_dir = subject_epochs_dir / f"ses-{session}"

    # Load epochs
    if st.sidebar.button("🔍 Load Epochs", type="primary"):
        with st.spinner("Loading epochs..."):
            epochs, metadata, error = load_combined_epochs(epochs_dir)

            if error:
                st.error(f"Error: {error}")
                st.stop()

            st.session_state.epochs = epochs
            st.session_state.metadata = metadata
            st.session_state.task_config = task_config
            st.success(f"✓ Loaded {len(epochs)} epochs")

    # Main interface
    if 'epochs' not in st.session_state:
        st.info("👈 Configure settings and click 'Load Epochs' to begin")

        # Show task config preview
        st.markdown("### 📋 Task Configuration Preview")
        st.markdown(f"**Task**: {task_config.task_name}")
        st.markdown(f"**Description**: {task_config.description}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Conditions:**")
            for cond in task_config.conditions:
                st.markdown(f"- **{cond.name}**: codes {cond.event_codes}")

        with col2:
            if task_config.response_mapping:
                st.markdown("**Response Mapping:**")
                for cat, val in task_config.response_mapping.categories.items():
                    st.markdown(f"- {cat}: {val}")

        return

    epochs = st.session_state.epochs
    metadata = st.session_state.metadata
    task_config = st.session_state.task_config

    # Display epoch info
    st.header("📊 Epoch Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Epochs", len(epochs))
    with col2:
        st.metric("Conditions", len(task_config.conditions))
    with col3:
        st.metric("Channels", len(epochs.ch_names))
    with col4:
        st.metric("Time Points", epochs.get_data().shape[2])

    st.markdown("---")

    # Sorting tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏷️ By Condition",
        "✅ By Response",
        "📊 By Bins",
        "💾 Export Sorted"
    ])

    with tab1:
        st.subheader("Sort by Experimental Condition")

        sorter = TrialSorter(task_config)

        try:
            sorted_by_condition = sorter.sort_by_condition(epochs)

            # Display condition counts
            condition_data = []
            for cond_name, cond_info in sorted_by_condition.items():
                condition_data.append({
                    'Condition': cond_name,
                    'N Trials': cond_info['n_trials'],
                    'Description': cond_info.get('description', '')
                })

            df_conditions = pd.DataFrame(condition_data)
            st.dataframe(df_conditions, width='stretch')

            # Store in session state
            st.session_state.sorted_by_condition = sorted_by_condition

        except Exception as e:
            st.error(f"Error sorting by condition: {e}")
            st.code(str(e))

    with tab2:
        st.subheader("Sort by Behavioral Response")

        if not task_config.response_mapping:
            st.warning("⚠️ No response mapping defined in task config")
        elif metadata is None:
            st.warning("⚠️ No metadata available. Metadata file not found.")
            st.info("Metadata should be saved as `metadata.csv` alongside epochs")
        else:
            sorter = TrialSorter(task_config)

            try:
                sorted_by_response = sorter.sort_by_response(epochs, metadata)

                # Display response counts
                response_data = []
                for resp_name, resp_info in sorted_by_response.items():
                    response_data.append({
                        'Response': resp_name,
                        'N Trials': resp_info['n_trials'],
                        'Percentage': f"{100 * resp_info['n_trials'] / len(epochs):.1f}%"
                    })

                df_responses = pd.DataFrame(response_data)
                st.dataframe(df_responses, width='stretch')

                # Visualize distribution
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(df_responses['Response'], df_responses['N Trials'])
                ax.set_xlabel('Response Category')
                ax.set_ylabel('Number of Trials')
                ax.set_title('Trial Distribution by Response')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close()

                # Store in session state
                st.session_state.sorted_by_response = sorted_by_response

            except Exception as e:
                st.error(f"Error sorting by response: {e}")
                st.code(str(e))

    with tab3:
        st.subheader("Bin Trials by Variable")

        if not task_config.binning:
            st.warning("⚠️ No binning specifications defined in task config")
        elif metadata is None:
            st.warning("⚠️ No metadata available for binning")
        else:
            # Select binning specification
            bin_spec_names = [spec.name for spec in task_config.binning]
            selected_bin = st.selectbox(
                "Binning Variable",
                options=bin_spec_names,
                help="Select how to bin trials"
            )

            bin_spec = next(b for b in task_config.binning if b.name == selected_bin)

            sorter = TrialSorter(task_config)

            try:
                binned_epochs = sorter.bin_trials(epochs, metadata, bin_spec)

                # Display bin counts
                bin_data = []
                for bin_name, bin_info in binned_epochs.items():
                    bin_data.append({
                        'Bin': bin_name,
                        'N Trials': bin_info['n_trials'],
                        'Percentage': f"{100 * bin_info['n_trials'] / len(epochs):.1f}%"
                    })

                df_bins = pd.DataFrame(bin_data)
                st.dataframe(df_bins, width='stretch')

                # Visualize distribution
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(df_bins['Bin'], df_bins['N Trials'])
                ax.set_xlabel(f'{bin_spec.name} Bin')
                ax.set_ylabel('Number of Trials')
                ax.set_title(f'Trial Distribution by {bin_spec.name}')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close()

                # Store in session state
                st.session_state.binned_epochs = binned_epochs

            except Exception as e:
                st.error(f"Error binning trials: {e}")
                st.code(str(e))

    with tab4:
        st.subheader("Export Sorted Epochs")

        st.markdown("Save sorted epochs to derivatives for further analysis")

        # Check what's been sorted
        has_condition = 'sorted_by_condition' in st.session_state
        has_response = 'sorted_by_response' in st.session_state
        has_bins = 'binned_epochs' in st.session_state

        if not any([has_condition, has_response, has_bins]):
            st.info("⚠️ Sort trials first in the tabs above, then return here to export")
        else:
            export_option = st.radio(
                "Export Type",
                options=["By Condition", "By Response", "By Bins"],
                horizontal=True
            )

            output_base = paths.derivatives_root / "epochs" / "sorted" / task_config.task_name / f"sub-{subject_id}" / f"ses-{session}"
            output_base.mkdir(parents=True, exist_ok=True)

            if st.button("💾 Export Sorted Epochs", type="primary"):
                with st.spinner("Exporting..."):
                    try:
                        if export_option == "By Condition" and has_condition:
                            sorted_data = st.session_state.sorted_by_condition
                            for cond_name, cond_info in sorted_data.items():
                                output_file = output_base / f"sub-{subject_id}_ses-{session}_task-{task_config.task_name}_condition-{cond_name}_epo.fif"
                                cond_info['epochs'].save(output_file, overwrite=True)
                                st.success(f"✓ Saved {cond_name}: {output_file.name}")

                        elif export_option == "By Response" and has_response:
                            sorted_data = st.session_state.sorted_by_response
                            for resp_name, resp_info in sorted_data.items():
                                output_file = output_base / f"sub-{subject_id}_ses-{session}_task-{task_config.task_name}_response-{resp_name}_epo.fif"
                                resp_info['epochs'].save(output_file, overwrite=True)
                                st.success(f"✓ Saved {resp_name}: {output_file.name}")

                        elif export_option == "By Bins" and has_bins:
                            sorted_data = st.session_state.binned_epochs
                            for bin_name, bin_info in sorted_data.items():
                                safe_bin_name = str(bin_name).replace(' ', '_').replace('/', '_')
                                output_file = output_base / f"sub-{subject_id}_ses-{session}_task-{task_config.task_name}_bin-{safe_bin_name}_epo.fif"
                                bin_info['epochs'].save(output_file, overwrite=True)
                                st.success(f"✓ Saved {bin_name}: {output_file.name}")

                        st.balloons()
                        st.info(f"📁 All files saved to: `{output_base}`")

                    except Exception as e:
                        st.error(f"Export error: {e}")
                        import traceback
                        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
