"""Source Reconstruction Configuration and Processing."""

import streamlit as st
from pathlib import Path
import yaml
from typing import Dict, Any, List

from eegcpm.core.paths import EEGCPMPaths
from eegcpm.core.config import SourceConfig
from eegcpm.ui.utils.executor import run_eegcpm_command, create_source_config
from eegcpm.ui.session_persistence import restore_project_from_storage


def main():
    """Source reconstruction page."""
    st.set_page_config(
        page_title="Processing: Source Reconstruction - EEGCPM",
        page_icon="⚙️",
        layout="wide"
    )

    st.title("⚙️ Processing: Source Reconstruction")
    st.markdown("""
    Configure and run source reconstruction on epoched data.
    Creates source estimates and ROI time courses using inverse methods (dSPM, sLORETA, etc.).
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
    tab_config, tab_run, tab_results = st.tabs(["📝 Configuration", "▶️ Run", "📊 Results"])

    # Tab 1: Configuration
    with tab_config:
        st.header("Source Reconstruction Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Dependencies")

            # Get available preprocessing pipelines
            preprocessing_dir = paths.derivatives_root / "preprocessing"
            if preprocessing_dir.exists():
                preprocessing_options = [d.name for d in preprocessing_dir.iterdir() if d.is_dir()]
            else:
                preprocessing_options = []

            preprocessing = st.selectbox(
                "Preprocessing Pipeline",
                options=preprocessing_options,
                help="Which preprocessing pipeline to use"
            )

            # Get available tasks
            if preprocessing:
                epochs_base = paths.derivatives_root / "epochs" / preprocessing
                if epochs_base.exists():
                    task_options = [d.name for d in epochs_base.iterdir() if d.is_dir()]
                else:
                    task_options = []
            else:
                task_options = []

            task = st.selectbox(
                "Task",
                options=task_options,
                help="Which task to use"
            )

            epochs_variant = st.text_input(
                "Epochs Variant (optional)",
                value="standard",
                help="Epochs configuration variant name"
            )

        with col2:
            st.subheader("Method Configuration")

            variant_name = st.text_input(
                "Variant Name",
                value="eLORETA-CONN32",
                help="Descriptive name for this source reconstruction variant"
            )

            method = st.selectbox(
                "Inverse Method",
                options=["dSPM", "sLORETA", "eLORETA", "MNE"],
                index=2,  # eLORETA default
                help="Source reconstruction method"
            )

            parcellation = st.selectbox(
                "Parcellation",
                options=["conn_networks", "aparc", "schaefer100"],
                help="ROI parcellation scheme"
            )

        st.subheader("Advanced Parameters")

        col3, col4, col5 = st.columns(3)

        with col3:
            st.markdown("**Forward Model**")
            template = st.text_input("Template", value="fsaverage")
            spacing = st.selectbox(
                "Spacing",
                options=["oct5", "oct6", "ico4", "ico5"],
                index=1
            )

        with col4:
            st.markdown("**Inverse Operator**")
            snr = st.number_input("SNR", value=3.0, min_value=1.0, max_value=10.0, step=0.5)
            loose = st.slider("Loose", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
            depth = st.slider("Depth", min_value=0.0, max_value=1.0, value=0.8, step=0.1)

        with col5:
            st.markdown("**ROI Extraction**")
            roi_radius = st.number_input(
                "ROI Radius (mm)",
                value=10.0,
                min_value=5.0,
                max_value=20.0,
                step=1.0
            )

        st.subheader("Output Options")

        col6, col7, col8 = st.columns(3)
        with col6:
            save_stc = st.checkbox("Save Source Estimates", value=True)
        with col7:
            save_roi_tc = st.checkbox("Save ROI Time Courses", value=True)
        with col8:
            generate_qc = st.checkbox("Generate QC Report", value=True)

        # Build config
        config_dict = {
            "stage": "source",
            "variant": variant_name,
            "depends_on": {
                "preprocessing": preprocessing,
                "task": task,
                "epochs": epochs_variant if epochs_variant else None
            },
            "method": method,
            "forward": {
                "template": template,
                "spacing": spacing
            },
            "inverse": {
                "snr": snr,
                "loose": loose,
                "depth": depth
            },
            "parcellation": parcellation,
            "roi_radius": roi_radius,
            "subjects": "all",
            "output": {
                "save_stc": save_stc,
                "save_roi_tc": save_roi_tc,
                "generate_qc": generate_qc
            }
        }

        # Preview config
        st.subheader("Configuration Preview")
        st.code(yaml.dump(config_dict, default_flow_style=False, sort_keys=False), language="yaml")

        # Save config button
        col_save1, col_save2 = st.columns([3, 1])
        with col_save1:
            config_name = st.text_input(
                "Config Filename",
                value=f"{variant_name.lower().replace(' ', '_')}.yaml"
            )
        with col_save2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("💾 Save Config", width='stretch'):
                config_dir = project_root / "eegcpm" / "configs" / "source"
                config_dir.mkdir(parents=True, exist_ok=True)
                config_path = config_dir / config_name

                with open(config_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

                st.success(f"✅ Config saved to: {config_path}")
                st.session_state['source_config_path'] = config_path

    # Tab 2: Run
    with tab_run:
        st.header("Run Source Reconstruction")

        # Config selection
        config_dir = project_root / "eegcpm" / "configs" / "source"
        if config_dir.exists():
            config_files = [f.name for f in config_dir.glob("*.yaml")]
        else:
            config_files = []

        if not config_files:
            st.warning("⚠️ No source reconstruction configs found. Create one in the Configuration tab.")
            return

        selected_config = st.selectbox(
            "Select Configuration",
            options=config_files,
            index=0 if config_files else None
        )

        if selected_config:
            config_path = config_dir / selected_config

            # Load and display config
            with open(config_path) as f:
                config_data = yaml.safe_load(f)

            with st.expander("📄 View Configuration"):
                st.code(yaml.dump(config_data, default_flow_style=False, sort_keys=False), language="yaml")

            # Task config selection
            st.subheader("Task Configuration")

            # Extract dependencies from source config
            task = config_data.get('depends_on', {}).get('task')
            preprocessing = config_data.get('depends_on', {}).get('preprocessing')
            epochs_variant = config_data.get('depends_on', {}).get('epochs')

            # Display task information
            col_task1, col_task2 = st.columns(2)
            with col_task1:
                st.info(f"**Task**: `{task}`")
            with col_task2:
                st.info(f"**Preprocessing**: `{preprocessing}`")

            # Get available task configs and filter by task_name field
            task_config_dir = paths.get_configs_dir("tasks")
            matching_configs = []  # [(filename_stem, description)]

            if task_config_dir.exists():
                for config_file in task_config_dir.glob("*.yaml"):
                    try:
                        with open(config_file) as f:
                            task_cfg = yaml.safe_load(f)

                        # Check if task_name matches
                        if task_cfg.get('task_name') == task:
                            description = task_cfg.get('description', 'No description')
                            matching_configs.append((config_file.stem, description))
                    except Exception as e:
                        # Skip invalid configs
                        continue

            # Sort by filename
            matching_configs.sort(key=lambda x: x[0])

            # Build options for dropdown
            task_config_options = ["(Auto-detect from task)", "(No grouping - individual events)"]
            task_config_display = task_config_options.copy()

            # Add matching configs with descriptions
            for filename, description in matching_configs:
                task_config_options.append(filename)
                task_config_display.append(f"{filename} - {description}")

            # Auto-detect logic: try exact match first, then first alphabetically
            default_idx = 0
            if task and matching_configs:
                # Try exact filename match first
                exact_match = [cfg for cfg in matching_configs if cfg[0] == task]
                if exact_match:
                    default_idx = task_config_options.index(exact_match[0][0])
                else:
                    # Use first matching config
                    default_idx = task_config_options.index(matching_configs[0][0])

            # Display dropdown
            selected_display = st.selectbox(
                "Task Config for Condition Grouping",
                options=range(len(task_config_display)),
                format_func=lambda i: task_config_display[i],
                index=default_idx,
                help="Select how to group events into conditions:\n"
                     "- Auto-detect: Uses config matching task name (by task_name field)\n"
                     "- No grouping: Creates separate files for each event (8, 9, etc.)\n"
                     "- Custom: Uses selected task config to group events by condition"
            )

            selected_task_config = task_config_options[selected_display]

            # Determine task_config argument
            if selected_task_config == "(Auto-detect from task)":
                task_config_arg = None  # CLI will auto-detect
                if matching_configs:
                    # Show which config will be used
                    exact_match = [cfg for cfg in matching_configs if cfg[0] == task]
                    if exact_match:
                        st.success(f"✓ Will auto-detect: `{exact_match[0][0]}.yaml` - {exact_match[0][1]}")
                    else:
                        st.success(f"✓ Will use first match: `{matching_configs[0][0]}.yaml` - {matching_configs[0][1]}")
                    st.info(f"📋 Found {len(matching_configs)} config(s) for task `{task}`")
                else:
                    st.warning(f"⚠️ No task configs found with task_name='{task}' - will process events individually")
            elif selected_task_config == "(No grouping - individual events)":
                task_config_arg = "none"
                st.info("ℹ️ Will process each event separately (no condition grouping)")
            else:
                task_config_arg = selected_task_config
                # Find description for selected config
                selected_desc = [desc for fname, desc in matching_configs if fname == selected_task_config]
                if selected_desc:
                    st.success(f"✓ Will use: `{selected_task_config}.yaml` - {selected_desc[0]}")

            # Subject selection
            st.subheader("Subject Selection")

            # Get available subjects
            preprocessing = config_data.get('depends_on', {}).get('preprocessing')

            if preprocessing and task:
                epochs_dir = paths.derivatives_root / "epochs" / preprocessing / task
                if epochs_dir.exists():
                    subject_dirs = [d.name.replace("sub-", "") for d in epochs_dir.glob("sub-*")]
                else:
                    subject_dirs = []
            else:
                subject_dirs = []

            if subject_dirs:
                st.info(f"📊 Found {len(subject_dirs)} subjects with epochs")

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
                if st.button("▶️ Run Source Reconstruction", type="primary", width='stretch'):
                    # Prepare command arguments
                    args = {
                        'project': str(project_root),
                        'config': str(config_path)
                    }

                    if not process_all and selected_subjects:
                        args['subjects'] = selected_subjects

                    # Add task config argument
                    if task_config_arg:
                        args['task-config'] = task_config_arg

                    # Create containers for output
                    status_container = st.empty()
                    log_container = st.container()

                    # Execute command
                    success = run_eegcpm_command(
                        command='source-reconstruct',
                        args=args,
                        log_container=log_container,
                        status_container=status_container
                    )

                    if success:
                        st.balloons()
                        st.success("✅ Source reconstruction completed! Check the Results tab.")

            else:
                st.warning("⚠️ No subjects found with epochs. Check dependencies in configuration.")

    # Tab 3: Results
    with tab_results:
        st.header("Source Reconstruction Results")

        # Get available source variants
        source_dir = paths.derivatives_root / "source"
        if not source_dir.exists():
            st.info("ℹ️ No source reconstruction results yet. Run source reconstruction first.")
            return

        # Let user select preprocessing/task/variant
        col_res1, col_res2, col_res3 = st.columns(3)

        with col_res1:
            preprocessing_dirs = [d.name for d in source_dir.iterdir() if d.is_dir()]
            if not preprocessing_dirs:
                st.info("No preprocessing pipelines found")
                return
            res_preprocessing = st.selectbox("Preprocessing", preprocessing_dirs, key="res_prep")

        with col_res2:
            if res_preprocessing:
                task_dirs = [d.name for d in (source_dir / res_preprocessing).iterdir() if d.is_dir()]
                res_task = st.selectbox("Task", task_dirs, key="res_task")
            else:
                res_task = None

        with col_res3:
            if res_task:
                variant_dirs = [
                    d.name.replace("variant-", "")
                    for d in (source_dir / res_preprocessing / res_task).iterdir()
                    if d.is_dir() and d.name.startswith("variant-")
                ]
                res_variant = st.selectbox("Variant", variant_dirs, key="res_variant")
            else:
                res_variant = None

        if res_preprocessing and res_task and res_variant:
            variant_dir = source_dir / res_preprocessing / res_task / f"variant-{res_variant}"

            # Get subjects
            subject_dirs = [d for d in variant_dir.glob("sub-*") if d.is_dir()]

            st.info(f"📊 Found {len(subject_dirs)} subjects with source reconstruction")

            # Display results
            if subject_dirs:
                selected_subject = st.selectbox(
                    "Select Subject",
                    options=[d.name for d in subject_dirs]
                )

                if selected_subject:
                    subject_dir = variant_dir / selected_subject

                    # Find session
                    session_dirs = [d for d in subject_dir.glob("ses-*") if d.is_dir()]
                    if session_dirs:
                        session_dir = session_dirs[0]
                    else:
                        session_dir = subject_dir

                    # List output files
                    st.subheader("Output Files")

                    output_files = list(session_dir.iterdir())
                    for f in output_files:
                        if f.is_file():
                            st.write(f"📄 {f.name}")

                    # Show QC report if exists
                    qc_html = list(session_dir.glob("*_source_qc.html"))
                    if qc_html:
                        st.subheader("QC Report")
                        with open(qc_html[0]) as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=800, scrolling=True)


if __name__ == "__main__":
    main()
