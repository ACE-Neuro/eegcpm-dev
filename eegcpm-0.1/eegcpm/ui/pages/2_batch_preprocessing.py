"""Batch Preprocessing - Cleanup, Configure, and Generate Scripts

This page allows you to:
1. Clean up old preprocessing files and state
2. Configure batch preprocessing jobs
3. Generate scripts for local or HPC execution
"""

import streamlit as st
from pathlib import Path
import sys
import yaml
import shutil
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eegcpm.ui.utils import scan_subjects, scan_tasks


def cleanup_section(derivatives_path: Path, eegcpm_path: Path):
    """UI section for cleaning up old preprocessing data."""

    st.header("🧹 Cleanup Old Data")

    st.markdown("""
    Remove old preprocessing outputs and state to start fresh.

    **⚠️ Warning**: This will permanently delete data. Make sure you have backups.
    """)

    # Scan what exists
    derivatives_subjects = []
    if derivatives_path.exists():
        derivatives_subjects = [d.name for d in derivatives_path.iterdir()
                               if d.is_dir() and not d.name.startswith('.')]

    pipelines = []
    pipelines_dir = derivatives_path / "preprocessing"
    if pipelines_dir.exists():
        # Exclude special directories that aren't pipelines
        excluded_dirs = {'logs', 'qc', 'reports', '__pycache__'}
        pipelines = [d.name for d in pipelines_dir.iterdir()
                    if d.is_dir() and not d.name.startswith('.') and d.name not in excluded_dirs]

    state_db = derivatives_path / ".eegcpm" / "state.db"
    state_exists = state_db.exists()

    # Show what will be deleted
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Derivatives (processing_state.json)**")
        if derivatives_subjects:
            st.info(f"📁 {len(derivatives_subjects)} subjects found")
            with st.expander("View subjects"):
                for subj in derivatives_subjects[:20]:
                    st.text(f"  • {subj}")
                if len(derivatives_subjects) > 20:
                    st.text(f"  ... and {len(derivatives_subjects) - 20} more")
        else:
            st.success("✓ Already clean")

    with col2:
        st.markdown("**Pipeline Outputs** (preprocessed data)")
        st.caption(f"Location: `{pipelines_dir}/`")
        if pipelines:
            st.info(f"📁 {len(pipelines)} pipelines found")
            with st.expander("View pipelines"):
                for pipeline in pipelines:
                    pipeline_path = pipelines_dir / pipeline
                    # Count subjects in this pipeline
                    n_subjects = len([d for d in pipeline_path.iterdir() if d.is_dir() and not d.name.startswith('.')])
                    st.text(f"  • {pipeline}/ ({n_subjects} subjects)")
        else:
            st.success("✓ Already clean")

    st.markdown("**Workflow State Database**")
    if state_exists:
        st.info(f"📊 State DB: {state_db}")
    else:
        st.success("✓ No state DB")

    # Cleanup options
    st.markdown("---")
    st.subheader("Select what to clean")

    col1, col2, col3 = st.columns(3)

    with col1:
        clean_derivatives = st.checkbox(
            "Derivatives folder",
            value=False,
            help="Remove all subject folders from derivatives"
        )

    with col2:
        clean_pipelines = st.checkbox(
            "Pipeline outputs",
            value=False,
            help="Remove preprocessed data in derivatives/preprocessing/ (does NOT delete config files)"
        )

    with col3:
        clean_state = st.checkbox(
            "Workflow state DB",
            value=False,
            help="Remove workflow state database"
        )

    # Confirmation
    if any([clean_derivatives, clean_pipelines, clean_state]):
        st.warning("⚠️ **Confirm cleanup**")

        items_to_delete = []
        if clean_derivatives:
            items_to_delete.append(f"• {len(derivatives_subjects)} subjects in derivatives/")
        if clean_pipelines:
            items_to_delete.append(f"• {len(pipelines)} pipelines in derivatives/preprocessing/")
        if clean_state:
            items_to_delete.append(f"• Workflow state database")

        st.markdown("Will delete:\n" + "\n".join(items_to_delete))

        confirm = st.checkbox("I understand this is permanent and cannot be undone")

        col1, col2 = st.columns([3, 1])

        with col2:
            if st.button("🗑️ Delete Now", type="primary", disabled=not confirm, width="stretch"):
                with st.spinner("Cleaning up..."):
                    deleted_count = 0

                    # Clean derivatives
                    if clean_derivatives and derivatives_path.exists():
                        for subj_dir in derivatives_path.iterdir():
                            if subj_dir.is_dir() and not subj_dir.name.startswith('.'):
                                shutil.rmtree(subj_dir)
                                deleted_count += 1
                        st.success(f"✓ Deleted {deleted_count} subjects from derivatives/")
                    elif clean_derivatives:
                        st.info("ℹ️ No derivatives directory to clean")

                    # Clean pipelines
                    if clean_pipelines and pipelines_dir.exists():
                        pipe_count = 0
                        for pipeline_dir in pipelines_dir.iterdir():
                            if pipeline_dir.is_dir() and not pipeline_dir.name.startswith('.'):
                                shutil.rmtree(pipeline_dir)
                                pipe_count += 1
                        st.success(f"✓ Deleted {pipe_count} pipeline folders")
                    elif clean_pipelines:
                        st.info("ℹ️ No preprocessing directory to clean")

                    # Clean state
                    if clean_state and state_db.exists():
                        state_db.unlink()
                        st.success(f"✓ Deleted workflow state database")

                    st.balloons()
                    st.rerun()


def batch_config_section(bids_root: Path, eegcpm_path: Path):
    """UI section for configuring batch preprocessing."""

    st.header("⚙️ Configure Batch Preprocessing")

    # Config selection
    config_dir = eegcpm_path / "configs" / "preprocessing"
    available_configs = []

    if config_dir.exists():
        available_configs = sorted([f for f in config_dir.glob("*.yaml")])

    if not available_configs:
        st.error("No config files found. Create a config first in Pipeline Config page.")
        return None

    config_file = st.selectbox(
        "Config File",
        options=available_configs,
        format_func=lambda x: x.name,
        help="Select preprocessing configuration"
    )

    # Pipeline name
    pipeline_name = st.text_input(
        "Pipeline Name",
        value="standard",
        help="Name for this pipeline (will create derivatives/preprocessing/{name}/)"
    )

    # Subject selection
    st.subheader("Subject Selection")

    subjects = scan_subjects(bids_root)
   
    if not subjects:
        print("subjects scanned:\t", subjects)
        st.error("No subjects found in BIDS directory")
        return None

    selection_mode = st.radio(
        "Selection mode",
        options=["All subjects", "Specific subjects", "Range"],
        horizontal=True
    )

    selected_subjects = []

    if selection_mode == "All subjects":
        selected_subjects = subjects
        st.info(f"✓ Selected all {len(subjects)} subjects")

    elif selection_mode == "Specific subjects":
        selected_subjects = st.multiselect(
            "Select subjects",
            options=subjects,
            default=subjects[:5] if len(subjects) >= 5 else subjects
        )

    else:  # Range
        col1, col2 = st.columns(2)
        with col1:
            start_idx = st.number_input("Start index", min_value=0, max_value=len(subjects)-1, value=0)
        with col2:
            end_idx = st.number_input("End index", min_value=0, max_value=len(subjects)-1, value=min(9, len(subjects)-1))

        selected_subjects = subjects[start_idx:end_idx+1]
        st.info(f"✓ Selected {len(selected_subjects)} subjects (index {start_idx} to {end_idx})")

    # Task selection
    tasks = scan_tasks(bids_root, selected_subjects[0] if selected_subjects else None)

    task = st.selectbox(
        "Task",
        options=tasks if tasks else ["rest"],
        help="Select task to process"
    )

    # Advanced options
    with st.expander("Advanced Options"):
        force_reprocess = st.checkbox(
            "Force reprocess",
            value=False,
            help="Reprocess subjects even if already completed"
        )

        parallel_jobs = st.number_input(
            "Parallel jobs (for SLURM)",
            min_value=1,
            max_value=100,
            value=10,
            help="Number of subjects to process in parallel on HPC"
        )

    return {
        'config_file': config_file,
        'pipeline_name': pipeline_name,
        'subjects': selected_subjects,
        'task': task,
        'force': force_reprocess,
        'parallel_jobs': parallel_jobs
    }


def generate_local_script(config: dict, bids_root: Path, eegcpm_root: Path) -> str:
    """Generate bash script for local batch preprocessing."""

    script = f"""#!/bin/bash
# EEGCPM Batch Preprocessing Script
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Pipeline: {config['pipeline_name']}
# Subjects: {len(config['subjects'])}

set -e  # Exit on error

# Paths
BIDS_ROOT="{bids_root}"
EEGCPM_ROOT="{eegcpm_root}"
CONFIG_FILE="{config['config_file']}"
PIPELINE="{config['pipeline_name']}"
TASK="{config['task']}"

# Create output directories (handled by CLI automatically)

# Process subjects
TOTAL={len(config['subjects'])}
CURRENT=0

"""

    for subject in config['subjects']:
        script += f"""
# Subject: {subject}
CURRENT=$((CURRENT + 1))
echo "[$CURRENT/$TOTAL] Processing {subject}..."

eegcpm preprocess \\
    --project "$BIDS_ROOT" \\
    --config "$CONFIG_FILE" \\
    --pipeline "$PIPELINE" \\
    --subject {subject} \\
    --task "$TASK" \\
"""
        if config['force']:
            script += "    --force \\\n"

        script += """    2>&1 | tee -a "$BIDS_ROOT/derivatives/preprocessing/$PIPELINE/preprocessing.log"

if [ $? -eq 0 ]; then
    echo "  ✓ {subject} completed"
else
    echo "  ✗ {subject} failed"
fi

""".format(subject=subject)

    script += """
echo "Batch preprocessing complete!"
"""

    return script


def generate_slurm_script(config: dict, bids_root: Path, eegcpm_root: Path) -> str:
    """Generate SLURM script for HPC batch preprocessing."""

    script = f"""#!/bin/bash
#SBATCH --job-name=eegcpm_{config['pipeline_name']}
#SBATCH --output={bids_root}/derivatives/preprocessing/{config['pipeline_name']}/slurm_%A_%a.out
#SBATCH --error={bids_root}/derivatives/preprocessing/{config['pipeline_name']}/slurm_%A_%a.err
#SBATCH --array=0-{len(config['subjects'])-1}%{config['parallel_jobs']}
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# EEGCPM SLURM Batch Preprocessing
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Pipeline: {config['pipeline_name']}

set -e

# Paths
BIDS_ROOT="{bids_root}"
EEGCPM_ROOT="{eegcpm_root}"
CONFIG_FILE="{config['config_file']}"
PIPELINE="{config['pipeline_name']}"
TASK="{config['task']}"

# Subject list
SUBJECTS=(
"""

    for subject in config['subjects']:
        script += f'    "{subject}"\n'

    script += f""")

# Get subject for this array task
SUBJECT="${{SUBJECTS[$SLURM_ARRAY_TASK_ID]}}"

echo "SLURM Array Task: $SLURM_ARRAY_TASK_ID / {len(config['subjects'])}"
echo "Processing subject: $SUBJECT"

# Load modules (adjust for your HPC environment)
# module load python/3.10
# source /path/to/venv/bin/activate

# Run preprocessing
eegcpm preprocess \\
    --project "$BIDS_ROOT" \\
    --config "$CONFIG_FILE" \\
    --pipeline "$PIPELINE" \\
    --subject "$SUBJECT" \\
    --task "$TASK" \\
"""

    if config['force']:
        script += "    --force \\\n"

    script += """    2>&1

echo "Subject $SUBJECT complete"
"""

    return script


def script_generation_section(batch_config: dict, bids_root: Path, eegcpm_root: Path):
    """UI section for generating preprocessing scripts."""

    if not batch_config:
        return

    st.header("📝 Generate Scripts")

    st.markdown(f"""
    **Configuration Summary**:
    - Pipeline: `{batch_config['pipeline_name']}`
    - Config: `{batch_config['config_file'].name}`
    - Subjects: {len(batch_config['subjects'])}
    - Task: `{batch_config['task']}`
    - Force reprocess: {'Yes' if batch_config['force'] else 'No'}
    """)

    tab1, tab2 = st.tabs(["🖥️ Local Script", "🏛️ HPC/SLURM Script"])

    with tab1:
        st.markdown("### Bash script for local execution")

        local_script = generate_local_script(batch_config, bids_root, eegcpm_root)

        st.code(local_script, language='bash')

        col1, col2 = st.columns([3, 1])

        with col2:
            st.download_button(
                label="⬇️ Download",
                data=local_script,
                file_name=f"batch_preprocess_{batch_config['pipeline_name']}.sh",
                mime="text/plain",
                width="stretch"
            )

        st.markdown("""
        **Usage:**
        ```bash
        chmod +x batch_preprocess_{pipeline}.sh
        ./batch_preprocess_{pipeline}.sh
        ```
        """.format(pipeline=batch_config['pipeline_name']))

    with tab2:
        st.markdown("### SLURM script for HPC clusters")

        slurm_script = generate_slurm_script(batch_config, bids_root, eegcpm_root)

        st.code(slurm_script, language='bash')

        col1, col2 = st.columns([3, 1])

        with col2:
            st.download_button(
                label="⬇️ Download",
                data=slurm_script,
                file_name=f"batch_preprocess_{batch_config['pipeline_name']}_slurm.sh",
                mime="text/plain",
                width="stretch"
            )

        st.markdown(f"""
        **Usage:**
        ```bash
        sbatch batch_preprocess_{batch_config['pipeline_name']}_slurm.sh

        # Check job status
        squeue -u $USER

        # Check array job progress
        sacct -j <JOB_ID>
        ```

        **Notes:**
        - Runs {len(batch_config['subjects'])} subjects with max {batch_config['parallel_jobs']} parallel jobs
        - Adjust `#SBATCH` directives for your cluster
        - Load appropriate modules/environment before running
        """)


def main():
    """Main batch preprocessing page."""

    st.set_page_config(
        page_title="Processing: Batch Preprocessing - EEGCPM",
        page_icon="⚙️",
        layout="wide"
    )

    st.title("⚙️ Processing: Batch Preprocessing")
    st.markdown("Clean up old data, configure batch jobs, and generate scripts")

    # Get paths from main app project selection
    from eegcpm.ui.project_manager import ProjectManager
    from eegcpm.core.paths import EEGCPMPaths

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

    # Use EEGCPMPaths for consistent path management
    paths = EEGCPMPaths(project_root)

    bids_path = paths.bids_root
    eegcpm_path = paths.eegcpm_root
    derivatives_path = paths.derivatives_root

    # Main sections
    tab1, tab2 = st.tabs(["🧹 Cleanup", "⚙️ Batch Configure & Generate"])

    with tab1:
        cleanup_section(derivatives_path, eegcpm_path)

    with tab2:
        batch_config = batch_config_section(bids_path, eegcpm_path)

        if batch_config and len(batch_config['subjects']) > 0:
            st.markdown("---")
            script_generation_section(batch_config, bids_path, eegcpm_path)
        elif batch_config:
            st.warning("⚠️ No subjects selected. Select at least one subject to generate scripts.")


if __name__ == "__main__":
    main()
