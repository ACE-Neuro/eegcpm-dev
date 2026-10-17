"""Batch Preprocessing - Cleanup, Configure, and Generate Scripts

This page allows you to:
1. Clean up old preprocessing files and state
2. Configure batch preprocessing jobs
3. Generate scripts for local or HPC execution
"""

import streamlit as st
from pathlib import Path
import sys
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eegcpm.pipeline.slurm import generate_local_script, generate_slurm_script
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
    st.subheader("Task Selection")
    # Scan across all subjects to detect all available tasks
    tasks = scan_tasks(bids_root, None)
    available_tasks = tasks if tasks else ["rest"]

    task_mode = st.radio(
        "Selection mode",
        options=["All tasks", "Specific tasks"],
        horizontal=True,
        key="task_selection_mode"
    )

    if task_mode == "All tasks":
        selected_tasks = available_tasks
        st.info(f"✓ Selected all {len(selected_tasks)} tasks")
    else:
        selected_tasks = st.multiselect(
            "Select tasks",
            options=available_tasks,
            default=[available_tasks[0]] if available_tasks else []
        )

        if not selected_tasks:
            st.error("Select at least one task to generate scripts.")
            return None

    # Legacy: single task selection (replaced by multi-task above)
    # task = st.selectbox(
    #     "Task",
    #     options=tasks if tasks else ["rest"],
    #     help="Select task to process"
    # )

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

    with st.expander("🖥️ HPC Settings (for SLURM script)"):
        st.markdown("Configure paths and environment for your HPC cluster.")

        hpc_bids_root = st.text_input(
            "BIDS root on HPC",
            value=st.session_state.get('hpc_bids_root', ''),
            placeholder="/share/ps_clivewong/25SusAttn/bids",
            help="Path to BIDS data directory on HPC (where sub-XXX folders are)"
        )
        if hpc_bids_root:
            st.session_state['hpc_bids_root'] = hpc_bids_root

        hpc_eegcpm_root = st.text_input(
            "EEGCPM root on HPC",
            value=st.session_state.get('hpc_eegcpm_root', ''),
            placeholder="/share/ps_clivewong/eegcpm-dev/eegcpm-0.1",
            help="Path to eegcpm-0.1 directory on HPC"
        )
        if hpc_eegcpm_root:
            st.session_state['hpc_eegcpm_root'] = hpc_eegcpm_root

        hpc_venv_path = st.text_input(
            "Venv activate script on HPC",
            value=st.session_state.get('hpc_venv_path', ''),
            placeholder="/share/ps_clivewong/eegcpm-dev/eegcpm-0.1/venv/bin/activate",
            help="Path to the Python venv activate script on HPC"
        )
        if hpc_venv_path:
            st.session_state['hpc_venv_path'] = hpc_venv_path

        hpc_email = st.text_input(
            "Email for notifications (optional)",
            value=st.session_state.get('hpc_email', ''),
            placeholder="",
            help="Receive email when jobs finish or fail"
        )
        if hpc_email:
            st.session_state['hpc_email'] = hpc_email

        col1, col2, col3 = st.columns(3)
        with col1:
            hpc_partition = st.selectbox(
                "Partition",
                options=["shared_cpu", "shared_gpu_l40", "shared_gpu_h20"],
                index=0,
                help="HPC partition to submit jobs to"
            )
        with col2:
            hpc_time = st.text_input(
                "Time limit",
                value="02:00:00",
                help="Max wall time per subject (HH:MM:SS)"
            )
        with col3:
            hpc_mem = st.text_input(
                "Memory",
                value="16G",
                help="Memory allocation per subject"
            )

        hpc_cpus = st.number_input(
            "CPUs per subject",
            min_value=1,
            max_value=15,
            value=4,
            help="Number of CPU cores per subject (max 15 for shared_cpu)"
        )

    return {
        'config_file': config_file,
        'pipeline_name': pipeline_name,
        'subjects': selected_subjects,
        'tasks': selected_tasks,
        # 'task': task,  # Legacy: single task
        'force': force_reprocess,
        'parallel_jobs': parallel_jobs,
        'hpc': {
            'bids_root': hpc_bids_root,
            'eegcpm_root': hpc_eegcpm_root,
            'venv_path': hpc_venv_path,
            'email': hpc_email,
            'partition': hpc_partition,
            'time': hpc_time,
            'mem': hpc_mem,
            'cpus': hpc_cpus,
        }
    }


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
    - Tasks: `{', '.join(batch_config['tasks'])}` ({len(batch_config['tasks'])} tasks)
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

        # Always show script preview
        slurm_script = generate_slurm_script(batch_config, bids_root, eegcpm_root)
        st.code(slurm_script, language='bash')

        col1, col2 = st.columns([3, 1])

        with col2:
            st.download_button(
                label="⬇️ Download",
                data=slurm_script,
                file_name=f"batch_preprocess_{batch_config['pipeline_name']}_slurm.sh",
                mime="text/plain",
                use_container_width=True,
            )

        st.markdown(f"""
        **Usage:**
        ```bash
        # 1. Upload script to HPC (via Cyberduck/scp)

        # 2. SSH to HPC and submit
        cd /path/to/your/project
        sbatch batch_preprocess_{batch_config['pipeline_name']}_slurm.sh

        # 3. Monitor job status
        squeue -u $USER

        # 4. Check completed/failed summary
        sacct -j <JOB_ID> --brief

        # 5. Cancel if needed
        scancel <JOB_ID>
        ```

        **Notes:**
        - Runs {len(batch_config['subjects'])} subjects with max {batch_config['parallel_jobs']} parallel jobs
        - Each subject processes {len(batch_config['tasks'])} tasks sequentially
        - Completed subjects are skipped on resubmit (no --force)
        - Email notification on job completion or failure
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
    from eegcpm.ui.session_persistence import restore_project_from_storage
    from eegcpm.core.paths import EEGCPMPaths

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
