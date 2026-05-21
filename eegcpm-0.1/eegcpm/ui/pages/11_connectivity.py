"""Connectivity Analysis - Compute functional connectivity from source data.

This page allows you to:
1. Configure connectivity methods (PLV, wPLI, dwPLI, icoh, AEC, PDC, DTF, etc.)
2. Set frequency bands (theta, alpha, beta, gamma)
3. Define time windows for analysis
4. Run connectivity analysis on ROI time courses
5. Visualize connectivity matrices
6. Export results for downstream analysis
"""

import streamlit as st
from pathlib import Path
import sys
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eegcpm.ui.utils import scan_subjects, scan_sessions, scan_tasks
from eegcpm.core.paths import EEGCPMPaths
from eegcpm.modules.connectivity import ConnectivityModule


def create_default_connectivity_config() -> Dict[str, Any]:
    """Create default connectivity configuration."""
    return {
        'methods': ['plv', 'wpli', 'dwpli'],
        'frequency_bands': {
            'theta': [4, 8],
            'alpha': [8, 13],
            'beta': [13, 30],
            'gamma': [30, 50],
        },
        'time_windows': [
            {'name': 'baseline', 'tmin': -0.3, 'tmax': 0.0},
            {'name': 'early', 'tmin': 0.0, 'tmax': 0.2},
            {'name': 'late', 'tmin': 0.2, 'tmax': 0.5},
        ],
        'mvar_order': 10,
    }


def render_method_selector() -> List[str]:
    """Render connectivity method selection UI."""
    st.subheader("Connectivity Methods")

    # Organize methods by category
    method_categories = {
        'Phase-based': {
            'plv': 'Phase Locking Value (Lachaux et al. 1999)',
            'pli': 'Phase Lag Index (Stam et al. 2007)',
            'wpli': 'Weighted Phase Lag Index (Vinck et al. 2011)',
            'dwpli': 'Debiased wPLI (Vinck et al. 2011)',
        },
        'Spectral': {
            'coherence': 'Magnitude-Squared Coherence',
            'icoh': 'Imaginary Coherence (Nolte et al. 2004)',
        },
        'Amplitude': {
            'correlation': 'Pearson Correlation',
            'spearman': 'Spearman Correlation',
            'partial_correlation': 'Partial Correlation',
            'aec': 'Amplitude Envelope Correlation (Brookes et al. 2011)',
            'aec_orth': 'Orthogonalized AEC (leakage correction)',
        },
        'Directional (MVAR-based)': {
            'pdc': 'Partial Directed Coherence (Baccala & Sameshima 2001)',
            'dtf': 'Directed Transfer Function (Kaminski & Blinowska 1991)',
        }
    }

    selected_methods = []

    for category, methods in method_categories.items():
        with st.expander(f"📊 {category}", expanded=(category == 'Phase-based')):
            for method, description in methods.items():
                if st.checkbox(description, key=f"method_{method}", value=(method in ['plv', 'wpli', 'dwpli'])):
                    selected_methods.append(method)

    return selected_methods


def render_frequency_bands() -> Dict[str, List[float]]:
    """Render frequency band configuration UI."""
    st.subheader("Frequency Bands")

    default_bands = {
        'delta': [1, 4],
        'theta': [4, 8],
        'alpha': [8, 13],
        'beta': [13, 30],
        'gamma': [30, 50],
    }

    bands = {}

    cols = st.columns(2)

    for idx, (band_name, (fmin, fmax)) in enumerate(default_bands.items()):
        with cols[idx % 2]:
            use_band = st.checkbox(
                f"{band_name.capitalize()} ({fmin}-{fmax} Hz)",
                key=f"band_{band_name}",
                value=(band_name in ['theta', 'alpha', 'beta'])
            )

            if use_band:
                col1, col2 = st.columns(2)
                with col1:
                    fmin_val = st.number_input(
                        "Min (Hz)",
                        value=float(fmin),
                        min_value=0.5,
                        max_value=100.0,
                        step=0.5,
                        key=f"fmin_{band_name}"
                    )
                with col2:
                    fmax_val = st.number_input(
                        "Max (Hz)",
                        value=float(fmax),
                        min_value=0.5,
                        max_value=100.0,
                        step=0.5,
                        key=f"fmax_{band_name}"
                    )

                bands[band_name] = [fmin_val, fmax_val]

    # Custom band
    with st.expander("➕ Add Custom Band"):
        custom_name = st.text_input("Band Name", key="custom_band_name")
        col1, col2 = st.columns(2)
        with col1:
            custom_fmin = st.number_input("Min (Hz)", value=8.0, key="custom_fmin")
        with col2:
            custom_fmax = st.number_input("Max (Hz)", value=13.0, key="custom_fmax")

        if st.button("Add Band") and custom_name:
            bands[custom_name] = [custom_fmin, custom_fmax]
            st.success(f"Added {custom_name}: {custom_fmin}-{custom_fmax} Hz")

    return bands


def render_time_windows() -> List[Dict[str, Any]]:
    """Render time window configuration UI."""
    st.subheader("Time Windows")

    st.markdown("Define time windows for connectivity analysis (in seconds relative to event):")

    # Initialize session state for time windows
    if 'time_windows' not in st.session_state:
        st.session_state.time_windows = [
            {'name': 'baseline', 'tmin': -0.3, 'tmax': 0.0},
            {'name': 'early', 'tmin': 0.0, 'tmax': 0.2},
            {'name': 'late', 'tmin': 0.2, 'tmax': 0.5},
        ]

    windows = []

    for idx, window in enumerate(st.session_state.time_windows):
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            name = st.text_input(
                "Window Name",
                value=window['name'],
                key=f"window_name_{idx}",
                label_visibility="collapsed"
            )

        with col2:
            tmin = st.number_input(
                "Start (s)",
                value=float(window['tmin']),
                min_value=-2.0,
                max_value=2.0,
                step=0.05,
                format="%.2f",
                key=f"tmin_{idx}"
            )

        with col3:
            tmax = st.number_input(
                "End (s)",
                value=float(window['tmax']),
                min_value=-2.0,
                max_value=2.0,
                step=0.05,
                format="%.2f",
                key=f"tmax_{idx}"
            )

        with col4:
            if st.button("🗑️", key=f"delete_{idx}", help="Delete window"):
                st.session_state.time_windows.pop(idx)
                st.rerun()

        if tmin < tmax:
            windows.append({'name': name, 'tmin': tmin, 'tmax': tmax})

    # Add new window
    if st.button("➕ Add Time Window"):
        st.session_state.time_windows.append({
            'name': f'window_{len(st.session_state.time_windows)+1}',
            'tmin': 0.0,
            'tmax': 0.5
        })
        st.rerun()

    return windows


def main():
    """Connectivity analysis interface."""

    st.set_page_config(
        page_title="Analysis: Connectivity - EEGCPM",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Analysis: Functional Connectivity")
    st.markdown("Compute connectivity matrices from ROI time courses using multiple methods")

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

    # Create tabs
    tab_config, tab_run, tab_results = st.tabs(["⚙️ Configuration", "▶️ Run Analysis", "📈 Results"])

    # Tab 1: Configuration
    with tab_config:
        st.header("Connectivity Configuration")

        col1, col2 = st.columns([1, 1])

        with col1:
            # Method selection
            selected_methods = render_method_selector()

            if not selected_methods:
                st.warning("⚠️ Please select at least one connectivity method")

            # MVAR order for directional methods
            if any(m in selected_methods for m in ['pdc', 'dtf']):
                st.markdown("---")
                st.subheader("Directional Connectivity Settings")
                mvar_order = st.slider(
                    "MVAR Model Order",
                    min_value=1,
                    max_value=30,
                    value=10,
                    help="Model order for PDC/DTF (higher = more temporal lag)"
                )
            else:
                mvar_order = 10

        with col2:
            # Frequency bands
            frequency_bands = render_frequency_bands()

            if not frequency_bands:
                st.warning("⚠️ Please select at least one frequency band")

        st.markdown("---")

        # Time windows
        time_windows = render_time_windows()

        if not time_windows:
            st.warning("⚠️ Please add at least one time window")

        # Save configuration
        st.markdown("---")
        st.subheader("💾 Save Configuration")

        config_name = st.text_input(
            "Configuration Name",
            value="connectivity_config",
            help="Name for this connectivity configuration"
        )

        if st.button("💾 Save Configuration", type="primary"):
            config = {
                'methods': selected_methods,
                'frequency_bands': frequency_bands,
                'time_windows': time_windows,
                'mvar_order': mvar_order,
            }

            # Save to project configs
            config_dir = paths.eegcpm_root / "configs" / "connectivity"
            config_dir.mkdir(parents=True, exist_ok=True)

            config_path = config_dir / f"{config_name}.yaml"

            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            st.success(f"✓ Saved configuration to: {config_path}")

            # Show preview
            with st.expander("📄 Configuration Preview"):
                st.code(yaml.dump(config, default_flow_style=False), language='yaml')

    # Tab 2: Run Analysis
    with tab_run:
        st.header("Run Connectivity Analysis")

        # Load configuration
        config_dir = paths.eegcpm_root / "configs" / "connectivity"
        config_dir.mkdir(parents=True, exist_ok=True)

        config_files = list(config_dir.glob("*.yaml"))

        if not config_files:
            st.info("📝 No connectivity configurations found. Create one in the Configuration tab.")
            st.stop()

        selected_config_file = st.selectbox(
            "Select Configuration",
            options=config_files,
            format_func=lambda x: x.stem
        )

        # Load config
        with open(selected_config_file) as f:
            config = yaml.safe_load(f)

        st.success(f"✓ Loaded configuration: {selected_config_file.stem}")

        with st.expander("📄 View Configuration"):
            st.code(yaml.dump(config, default_flow_style=False), language='yaml')

        st.markdown("---")

        # Data selection
        st.subheader("📂 Data Selection")

        # Source reconstruction variant selection
        source_dir = paths.derivatives_root / "source"

        if not source_dir.exists():
            st.warning("⚠️ No source reconstruction data found. Run source reconstruction first.")
            st.stop()

        # Get preprocessing pipelines
        preprocessing_options = [d.name for d in source_dir.iterdir() if d.is_dir()]

        if not preprocessing_options:
            st.warning("⚠️ No preprocessing pipelines found.")
            st.stop()

        preprocessing = st.selectbox(
            "Preprocessing Pipeline",
            options=preprocessing_options
        )

        # Get tasks
        task_dir = source_dir / preprocessing
        task_options = [d.name for d in task_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

        task = st.selectbox(
            "Task",
            options=task_options if task_options else []
        )

        # Task config selection - filter by task_name
        task_config_file = None
        if task:
            task_config_dir = paths.eegcpm_root / "configs" / "tasks"
            matching_task_configs = []  # [(filepath, name, description)]

            if task_config_dir.exists():
                for cfg_file in task_config_dir.glob("*.yaml"):
                    try:
                        with open(cfg_file) as f:
                            cfg = yaml.safe_load(f)

                        # Filter by task_name
                        if cfg.get('task_name') == task:
                            matching_task_configs.append((
                                cfg_file,
                                cfg_file.stem,
                                cfg.get('description', 'No description')
                            ))
                    except Exception:
                        continue

            if matching_task_configs:
                # Show config dropdown with descriptions
                config_options = [f"{name} - {desc}" for _, name, desc in matching_task_configs]
                config_paths = [path for path, _, _ in matching_task_configs]

                selected_task_config_idx = st.selectbox(
                    "Task Configuration",
                    options=range(len(config_options)),
                    format_func=lambda i: config_options[i],
                    help="Select task configuration (defines conditions for connectivity analysis)"
                )

                task_config_file = config_paths[selected_task_config_idx]
                st.caption(f"✓ Using: {task_config_file.stem}")

        # Get source variants
        if task:
            variant_dir = task_dir / task
            variant_options = [d.name for d in variant_dir.iterdir() if d.is_dir() and d.name.startswith('variant-')]

            source_variant = st.selectbox(
                "Source Variant",
                options=variant_options if variant_options else []
            )

        st.markdown("---")

        # Subject selection
        st.subheader("👥 Subject Selection")

        process_all = st.checkbox("Process All Subjects", value=False)

        if not process_all:
            # Get available subjects
            if task and source_variant:
                subject_dir = variant_dir / source_variant
                subjects = sorted([
                    d.name.replace('sub-', '')
                    for d in subject_dir.glob("sub-*")
                    if d.is_dir()
                ])

                selected_subjects = st.multiselect(
                    "Select Subjects",
                    options=subjects,
                    default=subjects[:1] if subjects else []
                )
        else:
            selected_subjects = "all"

        # Run button
        st.markdown("---")

        if st.button("▶️ Run Connectivity Analysis", type="primary"):
            # Validate inputs
            if not task or not source_variant:
                st.error("⚠️ Please select preprocessing, task, and source variant")
                st.stop()

            if selected_subjects != "all" and not selected_subjects:
                st.error("⚠️ Please select at least one subject")
                st.stop()

            # Get subject list
            if selected_subjects == "all":
                subject_dir = variant_dir / source_variant
                subjects_to_process = sorted([
                    d.name.replace('sub-', '')
                    for d in subject_dir.glob("sub-*")
                    if d.is_dir()
                ])
            else:
                subjects_to_process = selected_subjects

            st.info(f"📊 Processing {len(subjects_to_process)} subject(s)")

            # Create output directory
            output_root = paths.derivatives_root / "connectivity" / preprocessing / task / source_variant
            output_root.mkdir(parents=True, exist_ok=True)

            # Process each subject
            progress_bar = st.progress(0)
            status_text = st.empty()

            results_summary = []

            for idx, subject in enumerate(subjects_to_process):
                status_text.text(f"Processing {subject} ({idx+1}/{len(subjects_to_process)})...")

                try:
                    # Load ROI time courses
                    source_subject_dir = variant_dir / source_variant / f"sub-{subject}"

                    # Find source files (roi time courses)
                    source_files = list(source_subject_dir.glob("**/*_roi_tc.npz"))

                    # Fallback to other patterns
                    if not source_files:
                        source_files = list(source_subject_dir.glob("**/*_roi.npz"))
                    if not source_files:
                        source_files = list(source_subject_dir.glob("**/*roi*.npz"))

                    if not source_files:
                        st.warning(f"⚠️ No ROI files found for {subject}")
                        results_summary.append({
                            'subject': subject,
                            'status': 'failed',
                            'error': 'No ROI files found'
                        })
                        continue

                    # Load ROI data
                    roi_file = source_files[0]
                    roi_data_npz = np.load(roi_file, allow_pickle=True)

                    # Extract ROI names
                    roi_names = roi_data_npz.get('roi_names', None)
                    if roi_names is not None:
                        roi_names = roi_names.tolist() if hasattr(roi_names, 'tolist') else list(roi_names)

                    # Prepare data for ConnectivityModule
                    # Format: {'condition1': data, 'condition1_times': times, 'roi_names': names}
                    connectivity_input = {}

                    # Check for different data formats
                    if 'roi_data' in roi_data_npz:
                        # Format 1: Single roi_data array
                        roi_tc = roi_data_npz['roi_data']
                        if 'sfreq' not in roi_data_npz:
                            raise ValueError(f"Missing 'sfreq' in {roi_file.name} - source reconstruction output is incomplete")
                        sfreq = float(roi_data_npz['sfreq'])
                        times = np.arange(roi_tc.shape[1]) / sfreq
                        connectivity_input['default'] = roi_tc
                        connectivity_input['default_times'] = times

                    elif 'data' in roi_data_npz:
                        # Format 2: Single data array
                        roi_tc = roi_data_npz['data']
                        if 'sfreq' not in roi_data_npz:
                            raise ValueError(f"Missing 'sfreq' in {roi_file.name} - source reconstruction output is incomplete")
                        sfreq = float(roi_data_npz['sfreq'])
                        times = np.arange(roi_tc.shape[1]) / sfreq
                        connectivity_input['default'] = roi_tc
                        connectivity_input['default_times'] = times

                    else:
                        # Format 3: Condition-based data (from task config)
                        # Keys are condition names like 'target', 'distractor', etc.
                        # Metadata keys to exclude: roi_names, sfreq, and anything ending with _times
                        metadata_keys = {'roi_names', 'sfreq'}
                        condition_keys = [
                            k for k in roi_data_npz.files
                            if k not in metadata_keys and not k.endswith('_times')
                        ]

                        if condition_keys:
                            # Format 3a: Condition-based (e.g., 'target', 'distractor')
                            st.info(f"📊 Found {len(condition_keys)} conditions: {condition_keys}")

                            # Get sampling frequency
                            if 'sfreq' not in roi_data_npz:
                                raise ValueError(f"Missing 'sfreq' in {roi_file.name} - source reconstruction output is incomplete")
                            sfreq = float(roi_data_npz['sfreq'])

                            # Check if data is trial-level (3D)
                            first_condition = condition_keys[0]
                            first_data = roi_data_npz[first_condition]
                            if first_data.ndim == 3:
                                n_trials = first_data.shape[0]
                                n_rois = first_data.shape[1]
                                n_times = first_data.shape[2]
                                st.success(f"✓ Trial-level data detected: {n_trials} trials × {n_rois} ROIs × {n_times} timepoints")
                                st.info("💡 Connectivity will be computed per-trial and aggregated (mean/std/variance)")
                            else:
                                st.info("ℹ️ Evoked data detected (averaged across trials)")

                            # Add all conditions to connectivity input
                            for condition in condition_keys:
                                roi_tc = roi_data_npz[condition]
                                time_key = f"{condition}_times"

                                if time_key in roi_data_npz:
                                    times = roi_data_npz[time_key]
                                else:
                                    # Handle both 2D (n_rois, n_times) and 3D (n_trials, n_rois, n_times)
                                    n_times = roi_tc.shape[-1]  # Last dimension is always time
                                    times = np.arange(n_times) / sfreq

                                connectivity_input[condition] = roi_tc
                                connectivity_input[f"{condition}_times"] = times

                        else:
                            # Format 3b: Old trial-based format (event_1, event_2, etc.)
                            event_keys = [k for k in roi_data_npz.files if k.startswith('event_') and not k.endswith('_times')]

                            if not event_keys:
                                st.warning(f"⚠️ No ROI data found in {roi_file.name}")
                                st.info(f"Available keys: {list(roi_data_npz.files)}")
                                results_summary.append({
                                    'subject': subject,
                                    'status': 'failed',
                                    'error': 'No ROI data keys found'
                                })
                                continue

                            st.info(f"📊 Computing connectivity for {len(event_keys)} trials for {subject}")

                            # Create subject output directory BEFORE processing trials
                            subject_output_dir = output_root / f"sub-{subject}"
                            subject_output_dir.mkdir(parents=True, exist_ok=True)

                            # We'll process each trial through ConnectivityModule separately
                            # and then aggregate the results
                            trial_connectivity_matrices = {}  # {method_band_window: [trial1_matrix, trial2_matrix, ...]}

                            # Get sampling frequency from file (REQUIRED - our source reconstruction should have saved this)
                            if 'sfreq' not in roi_data_npz:
                                raise ValueError(f"Missing 'sfreq' in {roi_file.name} - source reconstruction output is incomplete")
                            sfreq = float(roi_data_npz['sfreq'])

                            for trial_idx, event_key in enumerate(event_keys):
                                roi_tc = roi_data_npz[event_key]
                                time_key = f"{event_key}_times"

                                if time_key in roi_data_npz:
                                    times = roi_data_npz[time_key]
                                else:
                                    times = np.arange(roi_tc.shape[1]) / sfreq

                                # Create input for this single trial
                                trial_input = {
                                    'trial': roi_tc,
                                    'trial_times': times,
                                    'roi_names': roi_names if roi_names else [f'ROI_{i}' for i in range(roi_tc.shape[0])]
                                }

                                # Compute connectivity for this trial
                                trial_conn_module = ConnectivityModule(config, subject_output_dir)
                                trial_result = trial_conn_module.process(trial_input, sfreq=sfreq, subject=None)

                                if trial_result.success:
                                    # Store trial-level matrices
                                    for matrix_name, matrix in trial_result.outputs['connectivity'].items():
                                        if matrix_name not in trial_connectivity_matrices:
                                            trial_connectivity_matrices[matrix_name] = []
                                        trial_connectivity_matrices[matrix_name].append(matrix)

                            # Now aggregate across trials
                            connectivity_input = {
                                'roi_names': roi_names if roi_names else [f'ROI_{i}' for i in range(roi_tc.shape[0])]
                            }

                            # Compute mean, variance, std across trials
                            aggregated_matrices = {}

                            for matrix_name, trial_matrices in trial_connectivity_matrices.items():
                                # Stack matrices: shape (n_trials, n_rois, n_rois)
                                stacked = np.stack(trial_matrices, axis=0)

                                # Compute statistics across trials (axis=0)
                                mean_matrix = np.mean(stacked, axis=0)
                                var_matrix = np.var(stacked, axis=0)
                                std_matrix = np.std(stacked, axis=0)

                                # Store aggregated results
                                # Remove 'trial_' prefix from matrix names
                                clean_name = matrix_name.replace('trial_', '')
                                aggregated_matrices[f"{clean_name}_mean"] = mean_matrix
                                aggregated_matrices[f"{clean_name}_variance"] = var_matrix
                                aggregated_matrices[f"{clean_name}_std"] = std_matrix

                            # Skip the normal ConnectivityModule processing
                            # Save aggregated results directly
                            conn_path = subject_output_dir / f"sub-{subject}_connectivity.npz"
                            np.savez(conn_path, **aggregated_matrices)

                            # Also save trial-level data separately for advanced analyses
                            trial_level_data = {}
                            for matrix_name, trial_matrices in trial_connectivity_matrices.items():
                                clean_name = matrix_name.replace('trial_', '')
                                # Stack as (n_trials, n_rois, n_rois)
                                trial_level_data[f"{clean_name}_trials"] = np.stack(trial_matrices, axis=0)

                            trial_conn_path = subject_output_dir / f"sub-{subject}_connectivity_trials.npz"
                            np.savez(trial_conn_path, **trial_level_data)

                            st.success(f"✓ {subject}: Computed {len(aggregated_matrices)//3} connectivity measures (mean/var/std) from {len(event_keys)} trials")
                            results_summary.append({
                                'subject': subject,
                                'status': 'success',
                                'n_matrices': len(aggregated_matrices),
                                'n_trials': len(event_keys),
                                'output_dir': str(subject_output_dir)
                            })

                            # Skip normal processing path for this subject
                            progress_bar.progress((idx + 1) / len(subjects_to_process))
                            continue  # Go to next subject

                    # Add ROI names
                    if roi_names:
                        connectivity_input['roi_names'] = roi_names
                    else:
                        # Auto-generate names
                        first_data_key = [k for k in connectivity_input.keys() if not k.endswith('_times') and k != 'roi_names'][0]
                        n_rois = connectivity_input[first_data_key].shape[0]
                        connectivity_input['roi_names'] = [f'ROI_{i}' for i in range(n_rois)]

                    # Get sampling frequency (REQUIRED)
                    if 'sfreq' not in roi_data_npz:
                        raise ValueError(f"Missing 'sfreq' in {roi_file.name} - source reconstruction output is incomplete")
                    sfreq = float(roi_data_npz['sfreq'])

                    # Create subject output directory
                    subject_output_dir = output_root / f"sub-{subject}"
                    subject_output_dir.mkdir(parents=True, exist_ok=True)

                    # Initialize ConnectivityModule
                    conn_module = ConnectivityModule(config, subject_output_dir)

                    # Create a simple subject object for naming
                    class SubjectInfo:
                        def __init__(self, subject_id):
                            self.id = subject_id

                    subject_info = SubjectInfo(f"sub-{subject}")

                    # Run connectivity analysis
                    with st.spinner(f"Computing connectivity for {subject}..."):
                        result = conn_module.process(connectivity_input, sfreq=sfreq, subject=subject_info)

                    if result.success:
                        st.success(f"✓ {subject}: Computed {len(result.outputs.get('connectivity', {}))} matrices")
                        results_summary.append({
                            'subject': subject,
                            'status': 'success',
                            'n_matrices': len(result.outputs.get('connectivity', {})),
                            'output_dir': str(subject_output_dir)
                        })
                    else:
                        st.warning(f"⚠️ {subject}: Failed - {result.errors}")
                        results_summary.append({
                            'subject': subject,
                            'status': 'failed',
                            'error': str(result.errors)
                        })

                except Exception as e:
                    st.error(f"❌ {subject}: Error - {str(e)}")
                    results_summary.append({
                        'subject': subject,
                        'status': 'error',
                        'error': str(e)
                    })

                # Update progress
                progress_bar.progress((idx + 1) / len(subjects_to_process))

            # Display summary
            status_text.text("✓ Processing complete!")
            st.markdown("---")
            st.subheader("📊 Processing Summary")

            df_summary = pd.DataFrame(results_summary)
            st.dataframe(df_summary, width='stretch')

            # Show success/failure counts
            n_success = len([r for r in results_summary if r['status'] == 'success'])
            n_failed = len(results_summary) - n_success

            col1, col2 = st.columns(2)
            with col1:
                st.metric("✓ Successful", n_success)
            with col2:
                st.metric("✗ Failed", n_failed)

            st.success(f"🎉 Connectivity analysis complete! Results saved to: {output_root}")

            # Show sample results
            if n_success > 0:
                with st.expander("📄 View Sample Results"):
                    success_subjects = [r for r in results_summary if r['status'] == 'success']
                    if success_subjects:
                        sample = success_subjects[0]
                        st.code(f"""
Output directory: {sample['output_dir']}
Number of connectivity matrices: {sample['n_matrices']}

Files:
- sub-{sample['subject']}_connectivity.npz  # Connectivity matrices
- sub-{sample['subject']}_connectivity_info.json  # Metadata

Connectivity matrix naming:
- Format: condition_window_method_band
- Example: default_early_plv_alpha, default_baseline_dwpli_beta
                        """)

                        # Show config used
                        st.json({
                            'methods': config['methods'],
                            'frequency_bands': config['frequency_bands'],
                            'time_windows': config['time_windows'],
                        })

    # Tab 3: Results
    with tab_results:
        st.header("📈 Connectivity Results")

        # Data selection
        st.subheader("📂 Load Results")

        # Find connectivity results
        connectivity_root = paths.derivatives_root / "connectivity"

        if not connectivity_root.exists():
            st.info("ℹ️ No connectivity results found. Run connectivity analysis first.")
            st.stop()

        # Get available preprocessing pipelines
        preprocessing_opts = [d.name for d in connectivity_root.iterdir() if d.is_dir()]

        if not preprocessing_opts:
            st.info("ℹ️ No connectivity results found.")
            st.stop()

        col1, col2, col3 = st.columns(3)

        with col1:
            prep_select = st.selectbox("Preprocessing", preprocessing_opts, key="results_prep")

        # Get tasks
        task_dir = connectivity_root / prep_select
        task_opts = [d.name for d in task_dir.iterdir() if d.is_dir()]

        with col2:
            task_select = st.selectbox("Task", task_opts if task_opts else [], key="results_task")

        # Get source variants
        if task_select:
            variant_dir = task_dir / task_select
            variant_opts = [d.name for d in variant_dir.iterdir() if d.is_dir()]

            with col3:
                variant_select = st.selectbox("Source Variant", variant_opts if variant_opts else [], key="results_variant")

        # Get subjects
        if task_select and variant_select:
            subject_dir = variant_dir / variant_select
            subject_opts = sorted([d.name for d in subject_dir.glob("sub-*") if d.is_dir()])

            subject_select = st.selectbox("Subject", subject_opts if subject_opts else [])

            if subject_select and st.button("📊 Load Results"):
                subject_path = subject_dir / subject_select

                # Load connectivity matrices
                conn_file = subject_path / f"{subject_select}_connectivity.npz"

                # Fallback to legacy naming
                if not conn_file.exists():
                    conn_file = subject_path / "unknown_connectivity.npz"

                if not conn_file.exists():
                    st.error(f"❌ Connectivity file not found in: {subject_path}")
                    st.error(f"Looking for: {subject_select}_connectivity.npz or unknown_connectivity.npz")
                    st.stop()

                # Check for trial-level data
                trial_file = subject_path / f"{subject_select}_connectivity_trials.npz"
                has_trial_data = trial_file.exists()

                with st.spinner("Loading connectivity matrices..."):
                    try:
                        data = np.load(conn_file)
                        if has_trial_data:
                            trial_data = np.load(trial_file)
                        else:
                            trial_data = None
                    except Exception as e:
                        st.error(f"❌ Failed to load connectivity file: {e}")
                        st.warning(f"⚠️ The file appears to be corrupted.")
                        st.code(str(conn_file))

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🗑️ Delete Corrupted File", type="primary"):
                                try:
                                    conn_file.unlink()
                                    if has_trial_data and trial_file.exists():
                                        trial_file.unlink()
                                    st.success("✓ Deleted corrupted file(s). Please re-run connectivity analysis.")
                                    st.rerun()
                                except Exception as del_error:
                                    st.error(f"Failed to delete file: {del_error}")
                        with col2:
                            st.info("After deleting, re-run connectivity analysis to generate a fresh file.")
                        st.stop()

                    # Load ROI names from source reconstruction output
                    # Look for ROI time courses file in source directory
                    roi_names = None
                    source_base = paths.derivatives_root / "source" / prep_select / task_select
                    if source_base.exists():
                        # Find variant directory
                        variant_dirs = [d for d in source_base.iterdir() if d.is_dir() and d.name.startswith('variant-')]
                        if variant_dirs:
                            # Try to find ROI file for this subject
                            for variant_dir in variant_dirs:
                                roi_file = variant_dir / subject_select / "ses-01" / f"{subject_select.replace('sub-', '')}_roi_tc.npz"
                                if roi_file.exists():
                                    roi_data = np.load(roi_file)
                                    if 'roi_names' in roi_data:
                                        roi_names = roi_data['roi_names'].tolist() if hasattr(roi_data['roi_names'], 'tolist') else list(roi_data['roi_names'])
                                    break

                # Store in session state to persist across interactions
                st.session_state.connectivity_data = data
                st.session_state.connectivity_trial_data = trial_data
                st.session_state.connectivity_has_trial_data = has_trial_data
                st.session_state.connectivity_roi_names = roi_names
                st.session_state.connectivity_subject = subject_select

                st.success(f"✓ Loaded {len(data.files)} connectivity matrices")
                if has_trial_data:
                    st.info(f"📊 Trial-level data available: {len(trial_data.files)} matrices")
                if roi_names:
                    st.info(f"🏷️ Loaded {len(roi_names)} ROI names")

            # Check if data is loaded in session state
            if 'connectivity_data' in st.session_state:
                data = st.session_state.connectivity_data
                trial_data = st.session_state.connectivity_trial_data
                has_trial_data = st.session_state.connectivity_has_trial_data
                roi_names = st.session_state.connectivity_roi_names
                subject_select = st.session_state.connectivity_subject

                # Display available matrices
                st.markdown("---")
                st.subheader("📋 Available Matrices")

                matrix_names = sorted(data.files)
                st.write(f"**Total matrices**: {len(matrix_names)}")

                # Parse matrix names to extract metadata
                matrix_info = []
                for name in matrix_names:
                    # Format: condition_window_method_band or condition_window_method_band_statistic
                    parts = name.split('_')

                    # Check if this is aggregated data (ends with _mean, _variance, _std)
                    if name.endswith('_mean') or name.endswith('_variance') or name.endswith('_std'):
                        statistic = parts[-1]
                        # Remove statistic to parse the rest
                        name_without_stat = '_'.join(parts[:-1])
                        parts = name_without_stat.split('_')

                        if len(parts) >= 4:
                            matrix_info.append({
                                'name': name,
                                'condition': parts[0],
                                'window': parts[1],
                                'method': parts[2],
                                'band': parts[3] if len(parts) > 3 else 'N/A',
                                'statistic': statistic
                            })
                        else:
                            matrix_info.append({
                                'name': name,
                                'condition': parts[0] if len(parts) > 0 else 'N/A',
                                'window': parts[1] if len(parts) > 1 else 'N/A',
                                'method': parts[2] if len(parts) > 2 else 'N/A',
                                'band': 'N/A',
                                'statistic': statistic
                            })
                    elif len(parts) >= 4:
                        # Regular connectivity matrix (not aggregated)
                        matrix_info.append({
                            'name': name,
                            'condition': parts[0],
                            'window': parts[1],
                            'method': parts[2],
                            'band': parts[3] if len(parts) > 3 else 'N/A',
                            'statistic': '-'  # Changed from 'N/A' to '-' for clarity
                        })
                    else:
                        # Fallback for other formats
                        matrix_info.append({
                            'name': name,
                            'condition': parts[0] if len(parts) > 0 else 'N/A',
                            'window': parts[1] if len(parts) > 1 else 'N/A',
                            'method': parts[2] if len(parts) > 2 else 'N/A',
                            'band': 'N/A',
                            'statistic': '-'
                        })

                if matrix_info:
                    df_matrices = pd.DataFrame(matrix_info)
                    st.dataframe(df_matrices, width='stretch')

                    # Matrix selector
                    st.markdown("---")
                    st.subheader("🔍 Visualize Matrix")

                    selected_matrix = st.selectbox(
                        "Select Matrix to Visualize",
                        options=matrix_names,
                        format_func=lambda x: x.replace('_', ' ').title()
                    )

                    if selected_matrix:
                        matrix = data[selected_matrix]

                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.write(f"**Matrix**: {selected_matrix}")
                            st.write(f"**Shape**: {matrix.shape}")
                            st.write(f"**Range**: [{matrix.min():.3f}, {matrix.max():.3f}]")
                            st.write(f"**Mean**: {matrix.mean():.3f}")

                            # Simple heatmap using streamlit
                            try:
                                import plotly.graph_objects as go

                                # Prepare axis labels
                                if roi_names and len(roi_names) == matrix.shape[0]:
                                    # Shorten ROI names for display
                                    short_names = []
                                    for name in roi_names:
                                        if '.' in name:
                                            # Extract just the ROI part after the network
                                            short_names.append(name.split('.', 1)[1])
                                        else:
                                            short_names.append(name)
                                    x_labels = short_names
                                    y_labels = short_names
                                else:
                                    # Fallback to numbers
                                    x_labels = list(range(matrix.shape[1]))
                                    y_labels = list(range(matrix.shape[0]))

                                fig = go.Figure(data=go.Heatmap(
                                    z=matrix,
                                    x=x_labels,
                                    y=y_labels,
                                    colorscale='RdBu_r',
                                    zmid=0,
                                    colorbar=dict(title="Connectivity")
                                ))

                                fig.update_layout(
                                    title=selected_matrix.replace('_', ' ').title(),
                                    xaxis_title="ROI",
                                    yaxis_title="ROI",
                                    yaxis=dict(autorange='reversed'),  # Flip y-axis so diagonal goes from top-left to bottom-right
                                    width=600,
                                    height=600
                                )

                                # Rotate x-axis labels if using ROI names
                                if roi_names:
                                    fig.update_xaxes(tickangle=-45)

                                st.plotly_chart(fig, width='stretch')

                            except ImportError:
                                # Fallback: display as dataframe
                                st.dataframe(matrix, width='stretch')
                                st.info("💡 Install plotly for interactive visualization: `pip install plotly`")

                        with col2:
                            st.write("**Statistics**")
                            st.metric("Min", f"{matrix.min():.3f}")
                            st.metric("Max", f"{matrix.max():.3f}")
                            st.metric("Mean", f"{matrix.mean():.3f}")
                            st.metric("Std", f"{matrix.std():.3f}")

                            # Threshold controls
                            st.markdown("---")
                            st.write("**Threshold**")
                            threshold = st.slider(
                                "Connectivity Threshold",
                                min_value=float(matrix.min()),
                                max_value=float(matrix.max()),
                                value=float(matrix.mean()),
                                step=0.01
                            )

                            # Apply threshold
                            thresholded = matrix.copy()
                            thresholded[np.abs(thresholded) < threshold] = 0

                            n_edges = np.sum(np.abs(thresholded) > 0)
                            total_edges = matrix.shape[0] * (matrix.shape[0] - 1) / 2

                            st.metric("Edges (above threshold)", f"{n_edges} / {int(total_edges)}")
                            st.metric("Density", f"{n_edges/total_edges*100:.1f}%")

                        # Trial-level analysis
                        if has_trial_data and selected_matrix.endswith('_mean'):
                            st.markdown("---")
                            st.subheader("📊 Trial-Level Variability")

                            # Get base name without _mean
                            base_name = selected_matrix.replace('_mean', '')

                            # Check for variance and std
                            var_name = f"{base_name}_variance"
                            std_name = f"{base_name}_std"
                            trial_name = f"{base_name}_trials"

                            if var_name in data.files and std_name in data.files:
                                var_matrix = data[var_name]
                                std_matrix = data[std_name]

                                col1, col2 = st.columns(2)

                                with col1:
                                    st.write("**Variance Across Trials**")
                                    st.metric("Mean Variance", f"{var_matrix.mean():.4f}")
                                    st.metric("Max Variance", f"{var_matrix.max():.4f}")

                                with col2:
                                    st.write("**Standard Deviation Across Trials**")
                                    st.metric("Mean Std", f"{std_matrix.mean():.4f}")
                                    st.metric("Max Std", f"{std_matrix.max():.4f}")

                                # Show which connections have highest variability
                                st.write("**Top 5 Most Variable Connections**")
                                # Get upper triangle (unique connections)
                                triu_indices = np.triu_indices_from(var_matrix, k=1)
                                var_values = var_matrix[triu_indices]
                                top_5_idx = np.argsort(var_values)[-5:][::-1]

                                top_5_data = []
                                for idx in top_5_idx:
                                    i, j = triu_indices[0][idx], triu_indices[1][idx]
                                    roi_i = connectivity_input.get('roi_names', [f'ROI_{i}'])[i]
                                    roi_j = connectivity_input.get('roi_names', [f'ROI_{j}'])[j]
                                    top_5_data.append({
                                        'Connection': f"{roi_i} ↔ {roi_j}",
                                        'Mean': f"{matrix[i,j]:.3f}",
                                        'Std': f"{std_matrix[i,j]:.3f}",
                                        'Variance': f"{var_matrix[i,j]:.4f}"
                                    })

                                st.dataframe(pd.DataFrame(top_5_data), width='stretch')

                            # Show trial-level data if available
                            if trial_data and trial_name in trial_data.files:
                                st.markdown("---")
                                trials_matrix = trial_data[trial_name]  # Shape: (n_trials, n_rois, n_rois)
                                n_trials = trials_matrix.shape[0]

                                st.write(f"**Trial-by-Trial Data**: {n_trials} trials")

                                # Allow selection of specific connection
                                st.write("**View Single Connection Across Trials**")
                                col1, col2 = st.columns(2)

                                with col1:
                                    roi_1_idx = st.selectbox(
                                        "ROI 1",
                                        options=range(trials_matrix.shape[1]),
                                        format_func=lambda x: connectivity_input.get('roi_names', [f'ROI_{x}'])[x] if 'roi_names' in connectivity_input else f'ROI_{x}'
                                    )

                                with col2:
                                    roi_2_idx = st.selectbox(
                                        "ROI 2",
                                        options=range(trials_matrix.shape[1]),
                                        index=1,
                                        format_func=lambda x: connectivity_input.get('roi_names', [f'ROI_{x}'])[x] if 'roi_names' in connectivity_input else f'ROI_{x}'
                                    )

                                # Extract connectivity values across trials for this connection
                                connection_values = trials_matrix[:, roi_1_idx, roi_2_idx]

                                # Plot
                                try:
                                    import plotly.graph_objects as go

                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        y=connection_values,
                                        mode='lines+markers',
                                        name='Connectivity',
                                        line=dict(color='blue')
                                    ))

                                    # Add mean line
                                    fig.add_hline(
                                        y=connection_values.mean(),
                                        line_dash="dash",
                                        line_color="red",
                                        annotation_text=f"Mean: {connection_values.mean():.3f}"
                                    )

                                    fig.update_layout(
                                        title=f"Trial-by-Trial Connectivity",
                                        xaxis_title="Trial",
                                        yaxis_title="Connectivity",
                                        height=400
                                    )

                                    st.plotly_chart(fig, width='stretch')

                                except ImportError:
                                    st.write(connection_values)

                                # Stats
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Mean", f"{connection_values.mean():.3f}")
                                with col2:
                                    st.metric("Std", f"{connection_values.std():.3f}")
                                with col3:
                                    st.metric("Range", f"{connection_values.max() - connection_values.min():.3f}")

                        # Export
                        st.markdown("---")
                        st.subheader("💾 Export")

                        col1, col2 = st.columns(2)

                        with col1:
                            # Export as CSV
                            csv_data = pd.DataFrame(matrix).to_csv(index=False)
                            st.download_button(
                                "📥 Download as CSV",
                                csv_data,
                                f"{selected_matrix}.csv",
                                "text/csv"
                            )

                        with col2:
                            # Export thresholded as CSV
                            thresh_csv = pd.DataFrame(thresholded).to_csv(index=False)
                            st.download_button(
                                "📥 Download Thresholded CSV",
                                thresh_csv,
                                f"{selected_matrix}_thresholded.csv",
                                "text/csv"
                            )

        else:
            st.info("ℹ️ Select preprocessing, task, and source variant to view results.")


if __name__ == "__main__":
    main()
