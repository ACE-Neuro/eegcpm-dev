"""Group Summary - Subject Inclusion/Exclusion for Analysis.

This page allows you to review all processed epochs and select which subjects
to include in downstream analysis (source reconstruction, connectivity, prediction).
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import json
from datetime import datetime
import mne

from eegcpm.core.paths import EEGCPMPaths
from eegcpm.ui.session_persistence import restore_project_from_storage


def load_inclusion_state(state_file: Path) -> dict:
    """Load saved inclusion/exclusion state."""
    if state_file.exists():
        with open(state_file, 'r') as f:
            return json.load(f)
    return {}


def save_inclusion_state(state_file: Path, state: dict):
    """Save inclusion/exclusion state."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, 'w') as f:
        json.dump(state, indent=2, fp=f)


def main():
    """Group summary main function."""

    st.set_page_config(
        page_title="Group Summary - EEGCPM",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Group Summary: Subject Inclusion/Exclusion")
    st.markdown("""
    Review epoch extraction results across all subjects and select which to include in downstream analysis.
    """)

    # Restore project from localStorage if session state was wiped by a refresh
    restore_project_from_storage()

    # Check for project root
    if 'eegcpm_root' not in st.session_state:
        st.warning("⚠️ No project configured. Please select a project on the home page.")
        return

    project_root = Path(st.session_state.eegcpm_root).parent
    paths = EEGCPMPaths(project_root)

    # Selection filters
    st.sidebar.header("📋 Dataset Selection")

    # Get available epochs
    epochs_dir = paths.derivatives_root / "epochs"
    if not epochs_dir.exists():
        st.info("ℹ️ No epochs results found. Run epochs extraction first.")
        return

    # Preprocessing pipeline selection
    preprocessing_dirs = sorted([d.name for d in epochs_dir.iterdir() if d.is_dir()])
    if not preprocessing_dirs:
        st.info("No preprocessing pipelines found")
        return

    preprocessing = st.sidebar.selectbox(
        "Preprocessing Pipeline",
        options=preprocessing_dirs,
        help="Which preprocessing variant"
    )

    # Task selection
    task_dirs = sorted([d.name for d in (epochs_dir / preprocessing).iterdir() if d.is_dir()])
    if not task_dirs:
        st.info(f"No tasks found for preprocessing '{preprocessing}'")
        return

    task = st.sidebar.selectbox(
        "Task",
        options=task_dirs,
        help="Which task"
    )

    # Build state key
    state_key = f"{preprocessing}_{task}"
    state_file = paths.eegcpm_root / "analysis" / "inclusion_state.json"

    # Load existing inclusion state
    all_states = load_inclusion_state(state_file)
    current_state = all_states.get(state_key, {})

    # Get subjects
    task_dir = epochs_dir / preprocessing / task
    subject_dirs = sorted([d for d in task_dir.glob("sub-*") if d.is_dir()])

    if not subject_dirs:
        st.info(f"No subjects found for {preprocessing}/{task}")
        return

    # Collect subject data
    st.header("📊 Dataset Overview")

    subject_data = []
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

            if epochs_files:
                try:
                    epochs = mne.read_epochs(epochs_files[0], preload=False, verbose=False)
                    n_epochs = len(epochs)
                    n_channels = len(epochs.ch_names)
                    conditions = list(epochs.event_id.keys()) if hasattr(epochs, 'event_id') else []
                    n_conditions = len(conditions)

                    # Get file size
                    file_size_mb = epochs_files[0].stat().st_size / (1024 * 1024)

                    # Check inclusion state (default: included)
                    subj_key = f"{subject_id}_{session_id}"
                    included = current_state.get(subj_key, True)

                    subject_data.append({
                        'Subject': subject_id,
                        'Session': session_id,
                        'Epochs': n_epochs,
                        'Channels': n_channels,
                        'Conditions': n_conditions,
                        'Condition Names': ', '.join(conditions),
                        'File Size (MB)': f"{file_size_mb:.1f}",
                        'Included': included,
                        '_key': subj_key,
                        '_epochs_file': str(epochs_files[0])
                    })

                except Exception as e:
                    st.warning(f"Could not load {subject_id}/{session_id}: {e}")

    if not subject_data:
        st.info("No epochs data found")
        return

    df = pd.DataFrame(subject_data)

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Subjects", len(df))

    with col2:
        n_included = df['Included'].sum()
        st.metric("Included", n_included, delta=f"{n_included/len(df)*100:.1f}%")

    with col3:
        avg_epochs = df['Epochs'].mean()
        st.metric("Avg Epochs", f"{avg_epochs:.0f}")

    with col4:
        total_epochs = df['Epochs'].sum()
        st.metric("Total Epochs", total_epochs)

    st.markdown("---")

    # Quick selection presets
    st.subheader("⚡ Quick Selection")

    col_preset1, col_preset2, col_preset3, col_preset4 = st.columns(4)

    with col_preset1:
        if st.button("✅ Include All", use_container_width=True):
            for item in subject_data:
                current_state[item['_key']] = True
            all_states[state_key] = current_state
            save_inclusion_state(state_file, all_states)
            st.rerun()

    with col_preset2:
        if st.button("❌ Exclude All", use_container_width=True):
            for item in subject_data:
                current_state[item['_key']] = False
            all_states[state_key] = current_state
            save_inclusion_state(state_file, all_states)
            st.rerun()

    with col_preset3:
        min_epochs = st.number_input("Min Epochs", min_value=1, value=50, step=10)
        if st.button(f"Include ≥{min_epochs} epochs", use_container_width=True):
            for item in subject_data:
                current_state[item['_key']] = item['Epochs'] >= min_epochs
            all_states[state_key] = current_state
            save_inclusion_state(state_file, all_states)
            st.rerun()

    with col_preset4:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            # Reset to all included
            current_state = {item['_key']: True for item in subject_data}
            all_states[state_key] = current_state
            save_inclusion_state(state_file, all_states)
            st.rerun()

    st.markdown("---")

    # Subject selection table
    st.subheader("📋 Subject Selection")

    # Filter controls
    col_filter1, col_filter2 = st.columns(2)

    with col_filter1:
        show_filter = st.selectbox(
            "Show",
            options=["All", "Included Only", "Excluded Only"],
            index=0
        )

    with col_filter2:
        search_query = st.text_input("Search Subject ID", "")

    # Apply filters
    df_display = df.copy()

    if show_filter == "Included Only":
        df_display = df_display[df_display['Included'] == True]
    elif show_filter == "Excluded Only":
        df_display = df_display[df_display['Included'] == False]

    if search_query:
        df_display = df_display[df_display['Subject'].str.contains(search_query, case=False)]

    st.caption(f"Showing {len(df_display)} of {len(df)} subjects")

    # Display table with checkboxes
    for idx, row in df_display.iterrows():
        col_check, col_info = st.columns([1, 11])

        with col_check:
            # Checkbox for inclusion
            included = st.checkbox(
                "✓",
                value=row['Included'],
                key=f"include_{row['_key']}",
                label_visibility="collapsed"
            )

            # Update state if changed
            if included != row['Included']:
                current_state[row['_key']] = included
                all_states[state_key] = current_state
                save_inclusion_state(state_file, all_states)

        with col_info:
            # Subject info
            status_icon = "✅" if included else "❌"
            st.markdown(
                f"{status_icon} **{row['Subject']}** (ses-{row['Session']}) | "
                f"{row['Epochs']} epochs | "
                f"{row['Channels']} channels | "
                f"{row['Conditions']} conditions: {row['Condition Names']}"
            )

    st.markdown("---")

    # Export section
    st.subheader("📥 Export Selection")

    col_export1, col_export2 = st.columns(2)

    with col_export1:
        # Generate inclusion list
        included_subjects = [
            {
                "subject": item['Subject'],
                "session": item['Session'],
                "epochs_file": item['_epochs_file'],
                "n_epochs": item['Epochs'],
                "n_channels": item['Channels']
            }
            for item in subject_data
            if current_state.get(item['_key'], True)
        ]

        inclusion_json = {
            "dataset": {
                "preprocessing": preprocessing,
                "task": task
            },
            "selection_date": datetime.now().isoformat(),
            "n_total": len(df),
            "n_included": len(included_subjects),
            "subjects": included_subjects
        }

        st.download_button(
            label="📄 Download Inclusion List (JSON)",
            data=json.dumps(inclusion_json, indent=2),
            file_name=f"inclusion_list_{preprocessing}_{task}_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True
        )

    with col_export2:
        # Generate subject ID list (for CLI)
        included_ids = [
            item['Subject']
            for item in subject_data
            if current_state.get(item['_key'], True)
        ]

        st.download_button(
            label="📄 Download Subject IDs (TXT)",
            data="\n".join(included_ids),
            file_name=f"subjects_{preprocessing}_{task}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    # Display export preview
    with st.expander("👁️ Preview Inclusion List"):
        st.json(inclusion_json)

    # Info about downstream use
    st.info("""
    **Next Steps:**

    1. **Source Reconstruction**: Use this selection for source localization
    2. **Connectivity Analysis**: Compute connectivity on included subjects
    3. **Prediction Pipeline**: Export for CPM or other ML pipelines

    The inclusion state is automatically saved and will persist across sessions.
    """)


if __name__ == "__main__":
    main()
