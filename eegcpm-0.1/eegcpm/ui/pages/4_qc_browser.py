"""QC Report Browser - View and download quality control reports."""

import streamlit as st
from pathlib import Path
import sys
import json
import pandas as pd
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eegcpm.ui.utils import scan_subjects, scan_tasks, scan_pipelines
from eegcpm.ui.session_persistence import restore_project_from_storage
from eegcpm.core.paths import EEGCPMPaths


def find_qc_reports(
    derivatives_root: Path,
    pipeline: Optional[str] = None,
    subject: Optional[str] = None,
    task: Optional[str] = None
) -> List[Dict]:
    """
    Find all QC reports in derivatives directory.

    Parameters
    ----------
    derivatives_root : Path
        Derivatives directory
    pipeline : str, optional
        Filter by pipeline name
    subject : str, optional
        Filter by subject ID
    task : str, optional
        Filter by task name

    Returns
    -------
    list of dict
        QC report metadata
    """
    reports = []
    # Search pattern based on filters
    if pipeline:
        search_root = derivatives_root / f'pipeline-{pipeline}'
    else:
        search_root = derivatives_root

    if not search_root.exists():
        return reports

    # Find all QC HTML files
    for qc_file in search_root.glob('**/*_qc.html'):
        # Parse filename to extract metadata
        filename = qc_file.name

        # Skip if doesn't match subject filter
        if subject and subject not in filename:
            continue

        # Skip if doesn't match task filter
        if task and f'task-{task}' not in filename:
            continue

        # Extract metadata from path
        parts = qc_file.parts

        # Extract subject from path (looks for sub-XXX)
        subject_id = None
        for part in parts:
            if part.startswith('sub-'):
                subject_id = part[4:]  # Remove 'sub-' prefix
                break

        # Extract report type
        if 'preprocessed_qc' in filename:
            report_type = 'Preprocessed'
        elif 'combined_qc' in filename:
            report_type = 'Combined'
        elif 'raw_qc' in filename:
            report_type = 'Raw Data'
        else:
            report_type = 'Other'

        # Extract task and run from path
        task_name = None
        run_id = None
        for part in parts:
            if part.startswith('task-'):
                task_name = part[5:]  # Remove 'task-' prefix
            if part.startswith('run-'):
                run_id = part[4:]  # Remove 'run-' prefix

        # Get pipeline from path (look for stage/variant pattern in derivatives)
        # Path structure: derivatives/{stage}/{variant}/sub-XXX/...
        # e.g., derivatives/preprocessing/standard/sub-XXX/...
        pipeline_name = None
        try:
            derivatives_idx = parts.index('derivatives')
            if derivatives_idx + 2 < len(parts):
                stage = parts[derivatives_idx + 1]
                variant = parts[derivatives_idx + 2]
                # Skip if it's a subject folder
                if not variant.startswith('sub-'):
                    pipeline_name = f"{stage}/{variant}"
        except (ValueError, IndexError):
            pass

        # Get file size and modification time
        file_size = qc_file.stat().st_size / 1024  # KB
        mod_time = qc_file.stat().st_mtime

        reports.append({
            'path': qc_file,
            'filename': filename,
            'subject': subject_id or 'Unknown',
            'task': task_name or 'Unknown',
            'run': run_id or 'N/A',
            'pipeline': pipeline_name or 'Unknown',
            'type': report_type,
            'size_kb': file_size,
            'modified': mod_time
        })

    # Sort by modification time (newest first)
    reports.sort(key=lambda x: x['modified'], reverse=True)

    return reports


def load_quality_metrics(qc_file: Path) -> Optional[Dict]:
    """
    Load quality metrics from QC report's accompanying JSON file.

    Parameters
    ----------
    qc_file : Path
        Path to HTML QC report

    Returns
    -------
    dict or None
        Quality metrics if available
    """
    # Look for accompanying JSON file
    json_file = qc_file.with_suffix('.json')

    if json_file.exists():
        try:
            with open(json_file, 'r') as f:
                return json.load(f)
        except:
            return None

    return None


def main():
    """QC Browser main function."""

    st.set_page_config(
        page_title="Quality Control: QC Browser - EEGCPM",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Quality Control: QC Browser")
    st.markdown("Browse and view quality control reports for all processing stages")

    # Sidebar filters
    if "qc_derivatives" in st.session_state:
        st.sidebar.header("🔍 Filters")

    # Restore project from localStorage if session state was wiped by a refresh
    restore_project_from_storage()

    # Auto-detect derivatives path from active project (bids_root is set on the home page)
    auto_derivatives = ""

    if "bids_root" in st.session_state and st.session_state.bids_root:
        auto_derivatives = str(Path(st.session_state.bids_root).parent / "derivatives")

    # Initialise session state on first load (or after browser reload wipes state)
    if "qc_derivatives_root" not in st.session_state or len(st.session_state.get("qc_derivatives_root", "")) == 0:
        st.session_state.qc_derivatives_root = auto_derivatives  # may be "" if no project set
    
    # tracks whether user clicked Reset Path
    if "qc_path_reset" not in st.session_state:
        st.session_state.qc_path_reset = False  

    # If a path is already confirmed, show it as a read-only label with a Reset button
    if st.session_state.qc_derivatives_root and not st.session_state.qc_path_reset:
        st.sidebar.markdown("**Derivatives Root**")
        st.sidebar.caption(f"📁 {st.session_state.qc_derivatives_root}")
        if st.sidebar.button("🔄 Reset Path", use_container_width=True):
            # Allow user to enter a different path
            st.session_state.qc_path_reset = True
            st.rerun()
        derivatives_root = st.session_state.qc_derivatives_root
    else:
        # No path yet (first visit / after reset) — show editable input
        entered = st.sidebar.text_input(
            "Derivatives Root",
            value=st.session_state.qc_derivatives_root,
            help="Path to derivatives directory"
        ) if st.session_state.qc_derivatives_root != "" else ""

        if entered != "":
            if st.sidebar.button("✅ Set Path", use_container_width=True):
                st.session_state.qc_derivatives_root = entered
                st.session_state.qc_path_reset = False
                st.rerun()

        derivatives_root = entered
        
    derivatives_path = Path(derivatives_root)

    if not derivatives_path.exists() or derivatives_root == "":
        if derivatives_root:
            st.error(f"❌ Derivatives directory not found: {derivatives_root}")
        else:
            st.warning("⚠️ No project configured. Please go to the Home page to set up a project.")
            st.page_link("app.py", label="→ Go to Home", icon="🏠")
        return

    # Pipeline filter
    pipelines = scan_pipelines(derivatives_path)
    pipeline_options = ['All'] + pipelines

    selected_pipeline = st.sidebar.selectbox(
        "Pipeline",
        options=pipeline_options,
        help="Filter by pipeline"
    )

    pipeline_filter = None if selected_pipeline == 'All' else selected_pipeline

    # Subject filter
    subject_filter = st.sidebar.text_input(
        "Subject ID (partial match)",
        value="",
        help="Filter by subject ID (leave empty for all)"
    )

    # Task filter
    task_filter = st.sidebar.text_input(
        "Task (partial match)",
        value="",
        help="Filter by task name (leave empty for all)"
    )

    # Report type filter
    report_type_filter = st.sidebar.multiselect(
        "Report Type",
        options=['Raw Data', 'Preprocessed', 'Combined', 'Other'],
        default=['Preprocessed', 'Combined'],
        help="Filter by report type"
    )

    st.sidebar.markdown("---")
    refresh_button = st.sidebar.button("🔄 Refresh Reports", width="stretch")

    # Find reports
    with st.spinner("Scanning for QC reports..."):
        reports = find_qc_reports(
            derivatives_path,
            pipeline=pipeline_filter,
            subject=subject_filter if subject_filter else None,
            task=task_filter if task_filter else None
        )

    # Filter by report type
    if report_type_filter:
        reports = [r for r in reports if r['type'] in report_type_filter]

    # Display summary
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Reports", len(reports))

    with col2:
        unique_subjects = len(set(r['subject'] for r in reports))
        st.metric("Subjects", unique_subjects)

    with col3:
        unique_tasks = len(set(r['task'] for r in reports))
        st.metric("Tasks", unique_tasks)

    with col4:
        total_size = sum(r['size_kb'] for r in reports) / 1024  # MB
        st.metric("Total Size", f"{total_size:.1f} MB")

    st.markdown("---")

    # Display reports
    if not reports:
        st.info("ℹ️ No QC reports found matching filters. Run preprocessing to generate reports.")

        st.markdown("""
        ### How to Generate QC Reports

        1. Go to **Preprocessing** page
        2. Select subject, task, and session
        3. Run preprocessing (QC reports are generated automatically)
        4. Come back here to view the reports
        """)
        return

    # Reports table
    st.subheader(f"📋 QC Reports ({len(reports)} found)")

    # Build dataframe
    report_data = []
    for r in reports:
        import datetime
        mod_time = datetime.datetime.fromtimestamp(r['modified']).strftime('%Y-%m-%d %H:%M')

        report_data.append({
            'Subject': r['subject'],
            'Task': r['task'],
            'Run': r['run'],
            'Pipeline': r['pipeline'],
            'Type': r['type'],
            'Size (KB)': f"{r['size_kb']:.1f}",
            'Modified': mod_time,
            'Filename': r['filename']
        })

    df = pd.DataFrame(report_data)

    # Display table with selection using data_editor (interactive)
    st.markdown("**Select a report from the table below:**")

    # Use selectbox for now (more reliable than row selection)
    # Create readable options
    report_options = []
    for i, r in enumerate(reports):
        option = f"{r['subject']} | {r['task']} | Run {r['run']} | {r['type']}"
        report_options.append(option)

    selected_idx = st.selectbox(
        "Select Report",
        options=range(len(reports)),
        format_func=lambda i: report_options[i],
        label_visibility="collapsed"
    )

    # Display the full table for reference
    st.dataframe(
        df,
        width='stretch',
        height=300
    )

    # Report viewer
    st.markdown("---")
    st.subheader("👁️ View Report")

    selected_report = reports[selected_idx]

    if selected_report:

        # Display report metadata
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"**Subject:** {selected_report['subject']}")
            st.markdown(f"**Task:** {selected_report['task']}")

        with col2:
            st.markdown(f"**Run:** {selected_report['run']}")
            st.markdown(f"**Type:** {selected_report['type']}")

        with col3:
            st.markdown(f"**Pipeline:** {selected_report['pipeline']}")
            st.markdown(f"**Size:** {selected_report['size_kb']:.1f} KB")

        # Action buttons
        col1, col2 = st.columns([1, 4])

        with col1:
            # Download button
            with open(selected_report['path'], 'rb') as f:
                st.download_button(
                    label="⬇️ Download Report",
                    data=f,
                    file_name=selected_report['filename'],
                    mime='text/html',
                    width="stretch"
                )

        with col2:
            # Copy path button
            st.code(str(selected_report['path']), language=None)

        # Quality metrics (if available)
        metrics = load_quality_metrics(selected_report['path'])
        if metrics:
            st.markdown("### 📈 Quality Metrics")

            # Display key metrics
            metric_cols = st.columns(4)

            with metric_cols[0]:
                if 'pct_bad_channels' in metrics:
                    st.metric("Bad Channels", f"{metrics['pct_bad_channels']:.1f}%")

            with metric_cols[1]:
                if 'quality_status' in metrics:
                    st.metric("Quality Status", metrics['quality_status'].capitalize())

            with metric_cols[2]:
                if 'clustering_severity' in metrics:
                    st.metric("Clustering", metrics['clustering_severity'].capitalize())

            with metric_cols[3]:
                if 'ica_success' in metrics:
                    st.metric("ICA", "✓ Success" if metrics['ica_success'] else "✗ Failed")

        # Embedded viewer
        st.markdown("---")

        view_option = st.radio(
            "Display Option",
            options=['Embedded Viewer', 'None'],
            horizontal=True,
            help="Embedded viewer shows report inline (may be slow for large reports)"
        )

        if view_option == 'Embedded Viewer':
            with st.spinner("Loading report..."):
                try:
                    with open(selected_report['path'], 'r') as f:
                        html_content = f.read()

                    # Display in iframe
                    st.components.v1.html(html_content, height=800, scrolling=True)

                except Exception as e:
                    st.error(f"Error loading report: {e}")
                    st.info("💡 Use the download button to view the report in your browser")


if __name__ == "__main__":
    main()
