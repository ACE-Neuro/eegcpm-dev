"""EEGCPM Streamlit UI - Main Application

Multi-page Streamlit application for EEG preprocessing and analysis.
Run with: streamlit run eegcpm/ui/app.py
"""

import streamlit as st
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eegcpm.ui.project_manager import ProjectManager, Project
from eegcpm.ui.utils import get_bids_info, get_subject_task_matrix
from eegcpm.ui.session_persistence import save_project_to_storage, clear_project_from_storage
from eegcpm.core.paths import EEGCPMPaths
import pandas as pd


def get_pipeline_progress(project_root: Path) -> dict:
    """Scan derivatives folder to compute pipeline progress."""
    paths = EEGCPMPaths(project_root)

    # Get all subjects from BIDS
    bids_subjects = list(paths.bids_root.glob("sub-*"))
    n_total = len(bids_subjects)

    if n_total == 0:
        return {
            'preprocessing': {'completed': 0, 'total': 0, 'percent': 0},
            'epochs': {'completed': 0, 'total': 0, 'percent': 0},
            'source': {'completed': 0, 'total': 0, 'percent': 0},
            'features': {'completed': 0, 'total': 0, 'percent': 0},
            'prediction': {'completed': 0, 'total': 0, 'percent': 0},
        }

    # Count completed subjects at each stage
    preprocessing_dir = paths.derivatives_root / "preprocessing"
    epochs_dir = paths.derivatives_root / "epochs"
    source_dir = paths.derivatives_root / "source"
    features_dir = paths.derivatives_root / "features"
    prediction_dir = paths.derivatives_root / "prediction"

    def count_subjects_in_stage(stage_dir: Path) -> int:
        """Count unique subjects with data in this stage."""
        if not stage_dir.exists():
            return 0
        subjects = set()
        for path in stage_dir.rglob("sub-*"):
            if path.is_dir():
                subjects.add(path.name)
        return len(subjects)

    n_preprocessing = count_subjects_in_stage(preprocessing_dir)
    n_epochs = count_subjects_in_stage(epochs_dir)
    n_source = count_subjects_in_stage(source_dir)
    n_features = count_subjects_in_stage(features_dir)
    n_prediction = 1 if prediction_dir.exists() and any(prediction_dir.iterdir()) else 0

    return {
        'preprocessing': {
            'completed': n_preprocessing,
            'total': n_total,
            'percent': int(100 * n_preprocessing / n_total) if n_total > 0 else 0
        },
        'epochs': {
            'completed': n_epochs,
            'total': n_total,
            'percent': int(100 * n_epochs / n_total) if n_total > 0 else 0
        },
        'source': {
            'completed': n_source,
            'total': n_total,
            'percent': int(100 * n_source / n_total) if n_total > 0 else 0
        },
        'features': {
            'completed': n_features,
            'total': n_total,
            'percent': int(100 * n_features / n_total) if n_total > 0 else 0
        },
        'prediction': {
            'completed': n_prediction,
            'total': 1,
            'percent': 100 if n_prediction > 0 else 0
        },
    }


def get_recent_activity(project_root: Path, limit: int = 5) -> list:
    """Get recent processing activity from derivatives folder."""
    paths = EEGCPMPaths(project_root)

    activities = []

    # Scan QC reports (they have timestamps)
    for stage in ['preprocessing', 'epochs', 'source']:
        stage_dir = paths.derivatives_root / stage
        if not stage_dir.exists():
            continue

        for qc_file in stage_dir.rglob("*_qc.html"):
            try:
                mtime = qc_file.stat().st_mtime

                # Extract subject from path
                subject = None
                for part in qc_file.parts:
                    if part.startswith('sub-'):
                        subject = part
                        break

                activities.append({
                    'timestamp': mtime,
                    'stage': stage.capitalize(),
                    'subject': subject or 'unknown',
                    'time_ago': format_time_ago(datetime.fromtimestamp(mtime).isoformat())
                })
            except:
                continue

    # Sort by timestamp (most recent first) and limit
    activities.sort(key=lambda x: x['timestamp'], reverse=True)
    return activities[:limit]


def format_time_ago(iso_timestamp: str) -> str:
    """Format timestamp as relative time (e.g., '2 hours ago')."""
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        now = datetime.now()
        delta = now - dt

        if delta.days > 0:
            if delta.days == 1:
                return "yesterday"
            elif delta.days < 7:
                return f"{delta.days} days ago"
            else:
                return dt.strftime("%Y-%m-%d")

        hours = delta.seconds // 3600
        if hours > 0:
            return f"{hours} hour{'s' if hours > 1 else ''} ago"

        minutes = delta.seconds // 60
        if minutes > 0:
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"

        return "just now"
    except:
        return "unknown"


def main():
    """Main application page."""

    st.set_page_config(
        page_title="EEGCPM - Home",
        page_icon="🧠",
        layout="wide"
    )

    st.title("🧠 EEGCPM - EEG Connectome Predictive Modeling")
    st.markdown("### Web Interface for EEG Preprocessing and Analysis")

    st.markdown("---")

    # Initialize project manager
    if 'project_manager' not in st.session_state:
        st.session_state.project_manager = ProjectManager()

    pm = st.session_state.project_manager

    # Project selection section
    st.header("📂 Project Setup")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Get list of projects
        projects = pm.get_projects_sorted()
        project_names = [p.name for p in projects]
        project_options = project_names + ["➕ Create New Project..."]

        # Current project selection
        if 'current_project_name' not in st.session_state:
            st.session_state.current_project_name = project_names[0] if project_names else None

        selected_option = st.selectbox(
            "Select Project",
            options=project_options,
            index=project_options.index(st.session_state.current_project_name) if st.session_state.current_project_name in project_options else 0,
            help="Choose a recent project or create a new one"
        )

        # Show recent access time
        if selected_option != "➕ Create New Project..." and selected_option in project_names:
            project = pm.get_project(selected_option)
            if project:
                time_ago = format_time_ago(project.last_accessed)
                st.caption(f"Last accessed: {time_ago}")

    with col2:
        # Project actions
        if selected_option != "➕ Create New Project..." and selected_option in project_names:
            col_a, col_b = st.columns(2)

            with col_a:
                if st.button("✏️ Rename", width="stretch"):
                    st.session_state.show_rename_dialog = True

            with col_b:
                if st.button("🗑️ Delete", width="stretch"):
                    if pm.delete_project(selected_option):
                        st.success(f"Deleted project: {selected_option}")
                        st.session_state.current_project_name = None
                        clear_project_from_storage()
                        st.rerun()

    # Handle rename dialog
    if st.session_state.get('show_rename_dialog', False):
        with st.container():
            new_name = st.text_input(
                "New project name",
                value=selected_option,
                key="rename_input"
            )
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✓ Confirm"):
                    if pm.rename_project(selected_option, new_name):
                        st.success(f"Renamed to: {new_name}")
                        st.session_state.current_project_name = new_name
                        st.session_state.show_rename_dialog = False
                        st.rerun()
                    else:
                        st.error(f"Project '{new_name}' already exists")
            with col_b:
                if st.button("✗ Cancel"):
                    st.session_state.show_rename_dialog = False
                    st.rerun()

    st.markdown("---")

    # Project configuration
    if selected_option == "➕ Create New Project...":
        st.subheader("Create New Project")

        col1, col2 = st.columns(2)

        with col1:
            project_name = st.text_input(
                "Project Name",
                value="My Project",
                help="Descriptive name for this project"
            )

        with col2:
            st.write("")  # Spacer

        bids_root = st.text_input(
            "BIDS Root Directory",
            value="/Volumes/Work/data/hbn/bids",
            help="Path to your BIDS dataset root directory"
        )

        eegcpm_root = st.text_input(
            "EEGCPM Working Directory",
            value="/Volumes/Work/data/hbn/eegcpm",
            help="EEGCPM workspace (configs & outputs will be stored here)"
        )

        if st.button("💾 Create Project", type="primary", width="stretch"):
            # Validate project doesn't exist
            if pm.get_project(project_name):
                st.error(f"❌ Project '{project_name}' already exists")
            else:
                # Create and validate project
                project = Project(
                    name=project_name,
                    bids_root=bids_root,
                    eegcpm_root=eegcpm_root
                )

                is_valid, error_msg = project.validate()
                if not is_valid:
                    st.error(f"❌ {error_msg}")
                else:
                    pm.add_project(project)
                    st.session_state.current_project_name = project_name
                    st.session_state.bids_root = bids_root
                    st.session_state.eegcpm_root = eegcpm_root
                    save_project_to_storage()
                    st.success(f"✅ Created project: {project_name}")
                    st.rerun()

    else:
        # Load existing project
        if selected_option in project_names:
            project = pm.get_project(selected_option)

            if project:
                # Update last accessed
                pm.update_last_accessed(selected_option)

                # Store in session state for other pages
                st.session_state.current_project_name = project.name
                st.session_state.bids_root = project.bids_root
                st.session_state.eegcpm_root = project.eegcpm_root
                save_project_to_storage()

                # Show project paths
                st.subheader(f"Project: {project.name}")

                col1, col2 = st.columns(2)

                with col1:
                    st.text_input(
                        "BIDS Root",
                        value=project.bids_root,
                        disabled=True,
                        key="display_bids"
                    )

                with col2:
                    st.text_input(
                        "EEGCPM Root",
                        value=project.eegcpm_root,
                        disabled=True,
                        key="display_eegcpm"
                    )

                # Validate paths
                is_valid, error_msg = project.validate()
                if not is_valid:
                    st.error(f"❌ {error_msg}")
                else:
                    # Show pipeline progress
                    st.markdown("---")
                    st.subheader("📊 Pipeline Progress")

                    bids_path = Path(project.bids_root)
                    project_root_path = bids_path.parent  # Project root contains bids/ and derivatives/

                    try:
                        progress = get_pipeline_progress(project_root_path)

                        # Progress bars for each stage
                        for stage_name, stage_data in progress.items():
                            if stage_name == 'prediction':
                                # Prediction is binary (done or not)
                                label = f"**Prediction**: {'Complete' if stage_data['completed'] > 0 else 'Not started'}"
                                st.progress(stage_data['percent'] / 100, text=label)
                            else:
                                # Other stages track per-subject progress
                                label = f"**{stage_name.capitalize()}**: {stage_data['completed']}/{stage_data['total']} ({stage_data['percent']}%)"
                                st.progress(stage_data['percent'] / 100, text=label)

                    except Exception as e:
                        st.warning(f"Could not compute pipeline progress: {e}")

                    # Recent activity
                    st.markdown("---")
                    st.subheader("📋 Recent Activity")

                    try:
                        activities = get_recent_activity(project_root_path, limit=5)

                        if activities:
                            for activity in activities:
                                emoji_map = {
                                    'Preprocessing': '🔧',
                                    'Epochs': '📊',
                                    'Source': '🧠'
                                }
                                emoji = emoji_map.get(activity['stage'], '⚙️')
                                st.markdown(f"{emoji} **{activity['stage']}**: {activity['subject']} ({activity['time_ago']})")
                        else:
                            st.info("No recent processing activity")

                    except Exception as e:
                        st.warning(f"Could not load recent activity: {e}")

                    # Dataset summary
                    st.markdown("---")
                    st.subheader("📁 Dataset Summary")

                    try:
                        bids_info = get_bids_info(bids_path)

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Subjects", bids_info['n_subjects'])

                        with col2:
                            st.metric("Sessions", bids_info['n_sessions'])

                        with col3:
                            if bids_info['tasks']:
                                st.metric("Tasks", len(bids_info['tasks']))
                                st.caption(f"{', '.join(bids_info['tasks'])}")

                    except Exception as e:
                        st.warning(f"Could not scan BIDS dataset: {e}")

                    # Subject/Task/Run Summary Table
                    st.markdown("---")
                    st.subheader("📋 Subject × Task Matrix")

                    try:
                        matrix_data = get_subject_task_matrix(bids_path)

                        if matrix_data['subjects'] and matrix_data['tasks']:
                            # Create DataFrame
                            df_data = []
                            for subject in matrix_data['subjects']:
                                row = {'Subject': subject}
                                for task in matrix_data['tasks']:
                                    n_runs = matrix_data['matrix'][subject][task]
                                    row[task] = str(n_runs) if n_runs > 0 else '-'
                                df_data.append(row)

                            df = pd.DataFrame(df_data)

                            # Display with styling
                            st.dataframe(
                                df,
                                width="stretch",
                                height=min(400, (len(df) + 1) * 35 + 3)
                            )

                            # Summary stats
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                total_runs = sum(
                                    matrix_data['matrix'][s][t]
                                    for s in matrix_data['subjects']
                                    for t in matrix_data['tasks']
                                )
                                st.caption(f"**Total runs:** {total_runs}")
                            with col2:
                                st.caption(f"**Tasks:** {', '.join(matrix_data['tasks'])}")
                            with col3:
                                avg_runs_per_subj = total_runs / len(matrix_data['subjects']) if matrix_data['subjects'] else 0
                                st.caption(f"**Avg runs/subject:** {avg_runs_per_subj:.1f}")
                        else:
                            st.info("No data found in BIDS directory")

                    except Exception as e:
                        st.warning(f"Could not generate summary table: {e}")

    # Quick Start Guide
    st.markdown("---")
    st.header("🚀 Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("▶ Continue Last Stage", type="primary", width='stretch'):
            st.info("Navigate to the processing pipeline to continue")

    with col2:
        if st.button("🔄 Reprocess Failed Subjects", width='stretch'):
            st.info("Navigate to batch preprocessing to reprocess")

    with col3:
        if st.button("📊 View QC Reports", width='stretch'):
            st.switch_page("pages/6_qc_browser.py")

    # Workflow guide
    st.markdown("---")
    st.header("📚 Analysis Workflow")

    st.markdown("""
    **Stage-based pipeline** - Complete each stage before moving to the next:

    1. **🔧 Configuration** - Set up preprocessing and task parameters
    2. **⚙️ Processing** - Run preprocessing, epochs, source reconstruction
    3. **📊 Quality Control** - Review QC reports and select good runs
    4. **📈 Features & Prediction** - Extract features and build predictive models

    Use the sidebar to navigate between stages.
    """)

    # Footer
    st.markdown("---")
    st.markdown("**EEGCPM v0.1** | [Documentation](../docs/)")


if __name__ == "__main__":
    main()
