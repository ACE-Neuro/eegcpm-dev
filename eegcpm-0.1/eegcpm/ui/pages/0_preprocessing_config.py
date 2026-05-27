"""Pipeline configuration interface.

Modern UI for step-based preprocessing pipeline configuration.
Matches the YAML format: preprocessing.steps[{name, params}]
"""

import streamlit as st
from pathlib import Path
import sys
import yaml
import shutil
from typing import Dict, Any, List
from eegcpm.ui.session_persistence import restore_project_from_storage

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def get_step_from_config(config: Dict, step_name: str) -> Dict:
    """Extract step parameters from config by step name."""
    steps = config.get('preprocessing', {}).get('steps', [])
    for step in steps:
        if step.get('name') == step_name:
            return step.get('params', {})
    return {}


# Define canonical pipeline order (based on research-backed optimal pipeline)
CANONICAL_STEP_ORDER = [
    'montage',
    'filter',
    'drop_flat',
    'bad_channels',
    'zapline',
    'notch',
    'lowpass',
    'asr',
    'ica',
    'iclabel',
    'interpolate',
    'reference',
    'resample',
    'artifacts',
]


def update_step_in_config(config: Dict, step_name: str, params: Dict) -> None:
    """Update or add step parameters in config."""
    steps = config.get('preprocessing', {}).get('steps', [])

    # Find existing step
    for i, step in enumerate(steps):
        if step.get('name') == step_name:
            steps[i]['params'] = params
            return

    # Add new step if not found
    steps.append({'name': step_name, 'params': params})


def sort_steps_by_canonical_order(steps: list) -> list:
    """Sort steps according to canonical pipeline order."""
    def get_order_index(step):
        name = step.get('name')
        try:
            return CANONICAL_STEP_ORDER.index(name)
        except ValueError:
            # Unknown steps go to the end
            return len(CANONICAL_STEP_ORDER)

    return sorted(steps, key=get_order_index)


def main():
    """Pipeline configuration interface main function."""

    st.set_page_config(
        page_title="Configuration: Preprocessing - EEGCPM",
        page_icon="🔧",
        layout="wide"
    )

    st.title("🔧 Configuration: Preprocessing")
    st.markdown("Configure preprocessing pipeline parameters (filtering, ICA, ASR, bad channels)")

    # Restore project from localStorage if session state was wiped by a refresh
    restore_project_from_storage()

    # Check if project is configured
    if 'eegcpm_root' not in st.session_state:
        st.warning("⚠️ No project configured. Please go to the Home page to set up a project.")
        st.page_link("app.py", label="→ Go to Home", icon="🏠")
        return

    # Get project paths from session state
    eegcpm_root = st.session_state.eegcpm_root
    eegcpm_path = Path(eegcpm_root)

    # Sidebar - show current project
    st.sidebar.header("📂 Current Project")
    if 'current_project_name' in st.session_state:
        st.sidebar.info(f"**{st.session_state.current_project_name}**")
    st.sidebar.caption(f"EEGCPM: `{eegcpm_root}`")

    # Define paths
    template_config_dir = Path(__file__).parent.parent.parent / "config" / "preprocessing"
    project_config_dir = eegcpm_path / "configs" / "preprocessing"

    # Ensure project config directory exists
    project_config_dir.mkdir(parents=True, exist_ok=True)

    # Scan available configs
    template_configs = sorted([f.stem for f in template_config_dir.glob("*.yaml")]) if template_config_dir.exists() else []
    project_configs = sorted([f.stem for f in project_config_dir.glob("*.yaml")]) if project_config_dir.exists() else []

    # Template descriptions
    template_descriptions = {
        'optimal': '🏆 Optimal - Research-backed pipeline (Highpass → Zapline → Lowpass)',
        'standard': '⭐ Standard - Full pipeline with ASR + ICA',
        'minimal': '🔹 Minimal - Basic filtering + ICA only',
        'robust': '💪 Robust - Three-stage ASR for noisy data'
    }

    # Sidebar - Config management
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Templates")

    if template_configs:
        for template in template_configs:
            desc = template_descriptions.get(template, template)
            st.sidebar.markdown(f"• {desc}")
    else:
        st.sidebar.warning("No templates found")

    # Main content - Two column layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("📦 Package Templates")
        st.markdown("*Read-only - copy to project to customize*")

        if template_configs:
            selected_template = st.radio(
                "Select Template",
                options=template_configs,
                format_func=lambda x: template_descriptions.get(x, x),
                help="Choose a template to copy or preview"
            )

            template_path = template_config_dir / f"{selected_template}.yaml"

            # Show template preview
            with open(template_path, 'r') as f:
                template_content = f.read()

            with st.expander("Preview Template", expanded=False):
                st.code(template_content, language='yaml')

            # Copy to project button
            st.markdown("---")

            new_config_name = st.text_input(
                "New Config Name",
                value=selected_template,
                help="Name for your project config (without .yaml)"
            )

            target_path = project_config_dir / f"{new_config_name}.yaml"

            if target_path.exists():
                st.warning(f"⚠️ Config '{new_config_name}' exists")
                overwrite = st.checkbox("Overwrite existing")
            else:
                overwrite = True

            if st.button("📋 Copy to Project", type="primary", width="stretch", disabled=not overwrite):
                try:
                    shutil.copy2(template_path, target_path)
                    st.success(f"✅ Copied to project")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        else:
            st.warning("No templates available")

    with col2:
        st.header("📁 Project Configs")
        st.markdown(f"*Your configs in `{project_config_dir.name}/`*")

        if not project_configs:
            st.info("👈 No project configs. Copy a template to start!")
        else:
            selected_config = st.selectbox(
                "Select Config to Edit",
                options=project_configs,
                help="Choose a config to edit"
            )

            config_path = project_config_dir / f"{selected_config}.yaml"

            # Load config
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Configuration Editor
            st.subheader("⚙️ Pipeline Steps")

            st.info("💡 **Tip**: Steps are executed in order. Enable/disable and configure each step below.")

            # Step 1: Montage
            with st.expander("1️⃣ Montage - Channel Locations", expanded=False):
                montage_params = get_step_from_config(config, 'montage')

                montage_type = st.selectbox(
                    "Montage Type",
                    options=['standard_1020', 'standard_1005', 'GSN-HydroCel-129', 'GSN-HydroCel-128'],
                    index=0 if montage_params.get('type') == 'standard_1020' else 0,
                    help="Channel position template"
                )

                on_missing = st.selectbox(
                    "If Channels Missing",
                    options=['warn', 'raise', 'ignore'],
                    index=0,
                    help="What to do if channels don't match montage"
                )

                update_step_in_config(config, 'montage', {
                    'type': montage_type,
                    'on_missing': on_missing
                })

            # Step 2: Filter
            with st.expander("2️⃣ Filter - Temporal Filtering", expanded=True):
                filter_params = get_step_from_config(config, 'filter')

                col_a, col_b = st.columns(2)

                with col_a:
                    enable_highpass = st.checkbox(
                        "Enable Highpass",
                        value=filter_params.get('l_freq') is not None,
                        help="Remove slow drifts and DC offset"
                    )
                    if enable_highpass:
                        l_freq = st.number_input(
                            "Highpass (Hz)",
                            min_value=0.0,
                            max_value=10.0,
                            value=float(filter_params.get('l_freq', 0.5)),
                            step=0.1,
                            help="Remove slow drifts (0.5-1.0 typical)"
                        )
                    else:
                        l_freq = None

                with col_b:
                    enable_lowpass = st.checkbox(
                        "Enable Lowpass",
                        value=filter_params.get('h_freq') is not None,
                        help="Remove high-frequency noise"
                    )
                    if enable_lowpass:
                        # Handle None case when loading config with h_freq: null
                        h_freq_value = filter_params.get('h_freq')
                        h_freq = st.number_input(
                            "Lowpass (Hz)",
                            min_value=10.0,
                            max_value=200.0,
                            value=float(h_freq_value) if h_freq_value is not None else 40.0,
                            step=1.0,
                            help="Remove high-freq noise (30-50 typical)"
                        )
                    else:
                        h_freq = None

                method = st.selectbox(
                    "Filter Method",
                    options=['fir', 'iir'],
                    index=0,
                    help="FIR is recommended (zero-phase, linear phase)"
                )

                if method == 'fir':
                    fir_window = st.selectbox(
                        "FIR Window",
                        options=['hamming', 'hann', 'blackman'],
                        index=0,
                        help="Window function for FIR filter"
                    )
                    phase = st.selectbox(
                        "Phase",
                        options=['zero', 'zero-double', 'minimum'],
                        index=0,
                        help="zero = zero-phase (non-causal, delay compensated)"
                    )

                    filter_config = {
                        'l_freq': l_freq,
                        'h_freq': h_freq,
                        'method': method,
                        'fir_window': fir_window,
                        'phase': phase
                    }
                else:
                    # IIR options
                    iir_order = st.slider(
                        "IIR Filter Order",
                        min_value=2,
                        max_value=8,
                        value=filter_params.get('iir_params', {}).get('order', 4),
                        help="Butterworth filter order (4 typical)"
                    )

                    filter_config = {
                        'l_freq': l_freq,
                        'h_freq': h_freq,
                        'method': method,
                        'iir_params': {'order': iir_order, 'ftype': 'butter'}
                    }

                st.info(f"💡 **{method.upper()} Filter**: " +
                       ("Zero-phase FIR (no phase distortion)" if method == 'fir'
                        else f"Butterworth IIR order {iir_order} (forward-backward, zero-phase)"))

                update_step_in_config(config, 'filter', filter_config)

            # Step 3: Drop Flat
            with st.expander("3️⃣ Drop Flat - Remove Dead Channels", expanded=False):
                flat_params = get_step_from_config(config, 'drop_flat')

                variance_threshold = st.number_input(
                    "Variance Threshold",
                    min_value=1e-20,
                    max_value=1e-10,
                    value=float(flat_params.get('variance_threshold', 1.0e-15)),
                    format="%.2e",
                    help="Channels below this variance are dropped"
                )

                update_step_in_config(config, 'drop_flat', {
                    'variance_threshold': variance_threshold
                })

            # Step 4: Bad Channels (before Zapline)
            with st.expander("4️⃣ Bad Channels Detection", expanded=True):
                bad_params = get_step_from_config(config, 'bad_channels')

                all_methods = ['prep', 'variance', 'ransac', 'correlation', 'deviation']
                current_method = bad_params.get('method', 'prep')
                if current_method not in all_methods:
                    current_method = 'prep'

                method = st.selectbox(
                    "Detection Method",
                    options=all_methods,
                    index=all_methods.index(current_method),
                    help=(
                        "PREP = comprehensive (4 criteria). "
                        "Variance = flag channels with abnormal signal variance."
                    )
                )

                bad_config = {'method': method}

                if method == 'variance':
                    variance_thresh = st.slider(
                        "Variance Threshold (SD from median)",
                        min_value=2.0, max_value=10.0,
                        value=bad_params.get('variance_threshold', 5.0),
                        step=0.5,
                        help="Channels with variance this many SDs from median are flagged"
                    )
                    bad_config['variance_threshold'] = variance_thresh

                drop = st.checkbox(
                    "Drop Bad Channels (vs Interpolate)",
                    value=bad_params.get('drop', False),
                    help="Drop removes channels; unchecked = interpolate (recommended)"
                )
                bad_config['drop'] = drop

                if method == 'prep':
                    st.info("💡 **PREP**: Uses deviation + correlation + HF noise + RANSAC")
                elif method == 'variance':
                    st.info("💡 **Variance**: Flags channels with abnormally high/low signal variance")

                update_step_in_config(config, 'bad_channels', bad_config)

            # Step 5: Zapline (after Bad Channels)
            with st.expander("5️⃣ Zapline - Line Noise Removal", expanded=False):
                zapline_params = get_step_from_config(config, 'zapline')

                enable_zapline = st.checkbox(
                    "Enable Zapline",
                    value=bool(zapline_params),
                    help="Adaptive removal of line noise (50/60 Hz) via subspace decomposition"
                )

                if enable_zapline:
                    col_a, col_b = st.columns(2)

                    with col_a:
                        auto_detect = st.checkbox(
                            "Auto-detect Line Frequency (Recommended)",
                            value=zapline_params.get('fline') is None,
                            help="Let pyzaplineplus automatically detect 50/60 Hz from data (recommended for multi-site studies)"
                        )

                        if not auto_detect:
                            fline = st.selectbox(
                                "Line Frequency (Hz)",
                                options=[60, 50],
                                index=0 if zapline_params.get('fline', 60) == 60 else 1,
                                help="Manual: 60 Hz (US/Americas), 50 Hz (EU/Asia/Africa)"
                            )
                        else:
                            fline = None

                    with col_b:
                        adaptive = st.checkbox(
                            "Adaptive Mode",
                            value=zapline_params.get('adaptive', True),
                            help="Automatically determine number of components to remove"
                        )

                        if not adaptive:
                            nremove = st.slider(
                                "Components to Remove",
                                min_value=1,
                                max_value=5,
                                value=int(zapline_params.get('nremove', 1)),
                                help="Number of line noise components (1-2 typical)"
                            )
                        else:
                            nremove = None

                    # Advanced parameters (collapsible)
                    st.markdown("**Advanced Settings**")

                    col_c, col_d = st.columns(2)

                    with col_c:
                        noiseCompDetectSigma = st.slider(
                            "Detection Threshold (σ)",
                            min_value=2.0,
                            max_value=5.0,
                            value=float(zapline_params.get('noiseCompDetectSigma', 3.0)),
                            step=0.1,
                            help="Lower = more aggressive cleaning (2.5 recommended for strong noise)"
                        )

                    with col_d:
                        fixedNremove = st.slider(
                            "Fallback Components",
                            min_value=1,
                            max_value=5,
                            value=int(zapline_params.get('fixedNremove', 1)),
                            help="Components removed if adaptive fails (2-3 for noisy data)"
                        )

                    if auto_detect:
                        st.success("✅ Zapline-Plus - Auto-detection enabled (searches for 50/60 Hz)")
                    else:
                        st.success(f"✅ Zapline-Plus - Targeting {fline} Hz line noise")

                    update_step_in_config(config, 'zapline', {
                        'fline': fline,
                        'nremove': nremove,
                        'adaptive': adaptive,
                        'noiseCompDetectSigma': noiseCompDetectSigma,
                        'fixedNremove': fixedNremove,
                    })
                else:
                    # Remove zapline step if disabled
                    steps = config.get('preprocessing', {}).get('steps', [])
                    config['preprocessing']['steps'] = [s for s in steps if s.get('name') != 'zapline']

            # Step 5b: Notch Filter (Alternative to Zapline)
            with st.expander("5️⃣b Notch - Line Noise Removal (Alternative)", expanded=False):
                notch_params = get_step_from_config(config, 'notch')

                enable_notch = st.checkbox(
                    "Enable Notch Filter",
                    value=bool(notch_params),
                    help="Traditional notch filtering (simpler than Zapline, but creates spectral notches)"
                )

                if enable_notch:
                    st.warning("⚠️ Using both Zapline and Notch is redundant. Choose one.")

                    # Frequency selection
                    region = st.radio(
                        "Line Frequency Region",
                        options=["US (60 Hz)", "EU (50 Hz)", "Custom"],
                        index=0,
                        help="Select your region's power line frequency"
                    )

                    if region == "US (60 Hz)":
                        default_freqs = [60, 120, 180]
                    elif region == "EU (50 Hz)":
                        default_freqs = [50, 100, 150]
                    else:
                        default_freqs = []

                    # Harmonics
                    n_harmonics = st.slider(
                        "Number of Harmonics",
                        min_value=0,
                        max_value=5,
                        value=2 if len(notch_params.get('freqs', [])) > 1 else 0,
                        help="0 = fundamental only, 2 = include 2 harmonics (recommended)"
                    )

                    if region != "Custom":
                        base_freq = 60 if region == "US (60 Hz)" else 50
                        freqs = [base_freq * (i + 1) for i in range(n_harmonics + 1)]
                    else:
                        freqs_str = st.text_input(
                            "Frequencies (comma-separated)",
                            value=",".join(map(str, notch_params.get('freqs', [60]))),
                            help="E.g., 60,120,180"
                        )
                        try:
                            freqs = [float(f.strip()) for f in freqs_str.split(',') if f.strip()]
                        except:
                            freqs = [60]

                    st.info(f"💡 Notch frequencies: {freqs} Hz")

                    update_step_in_config(config, 'notch', {
                        'freqs': freqs,
                        'method': 'fir',
                        'phase': 'zero',
                    })
                else:
                    # Remove notch step if disabled
                    steps = config.get('preprocessing', {}).get('steps', [])
                    config['preprocessing']['steps'] = [s for s in steps if s.get('name') != 'notch']

            # Step 5c: Lowpass (After Line Noise Removal)
            with st.expander("5️⃣c Lowpass - Final Frequency Cutoff (Optional)", expanded=False):
                lowpass_params = get_step_from_config(config, 'lowpass')

                enable_lowpass_step = st.checkbox(
                    "Enable Separate Lowpass Step",
                    value=bool(lowpass_params),
                    help="Apply lowpass AFTER line noise removal (prevents spectral leakage)"
                )

                if enable_lowpass_step:
                    st.info("💡 **Recommended pipeline**: Highpass → Bad Channels → Zapline → Lowpass → ASR → ICA")

                    h_freq_lowpass = st.number_input(
                        "Lowpass Frequency (Hz)",
                        min_value=10.0,
                        max_value=200.0,
                        value=float(lowpass_params.get('h_freq', 40.0)) if lowpass_params.get('h_freq') else 40.0,
                        step=1.0,
                        help="Typical: 30-50 Hz for cognitive tasks, 100 Hz for broad-spectrum"
                    )

                    method_lowpass = st.selectbox(
                        "Filter Method",
                        options=['fir', 'iir'],
                        index=0,
                        help="FIR recommended (zero-phase)"
                    )

                    update_step_in_config(config, 'lowpass', {
                        'h_freq': h_freq_lowpass,
                        'method': method_lowpass,
                        'phase': 'zero',
                    })
                else:
                    # Remove lowpass step if disabled
                    steps = config.get('preprocessing', {}).get('steps', [])
                    config['preprocessing']['steps'] = [s for s in steps if s.get('name') != 'lowpass']

            # Step 6: ASR
            with st.expander("6️⃣ ASR - Artifact Subspace Reconstruction", expanded=True):
                asr_params = get_step_from_config(config, 'asr')

                st.markdown("**ASR removes high-amplitude artifacts** (eye movements, muscle, etc.)")

                col_a, col_b = st.columns(2)

                with col_a:
                    asr_method = st.selectbox(
                        "ASR Implementation",
                        options=['eegprep', 'asrpy'],
                        index=0 if asr_params.get('method') == 'eegprep' else 0,
                        help="eegprep = EEGLAB port (recommended), asrpy = pure Python"
                    )

                with col_b:
                    cutoff = st.slider(
                        "ASR Cutoff (σ)",
                        min_value=5,
                        max_value=40,
                        value=int(asr_params.get('cutoff', 20)),
                        help="Lower = more aggressive (15-25 typical)"
                    )

                train_duration = st.number_input(
                    "Calibration Duration (s)",
                    min_value=10.0,
                    max_value=300.0,
                    value=float(asr_params.get('train_duration', 60.0)),
                    help="Duration of clean data for ASR calibration"
                )

                if asr_method == 'eegprep':
                    st.success("✅ eegprep (EEGLAB/SCCN) - Production ready")
                else:
                    st.warning("⚠️ asrpy - May have stability issues with poor data")

                update_step_in_config(config, 'asr', {
                    'cutoff': cutoff,
                    'mode': 'single',
                    'method': asr_method,
                    'train_duration': train_duration
                })

            # Step 7: ICA
            with st.expander("7️⃣ ICA - Independent Component Analysis", expanded=True):
                ica_params = get_step_from_config(config, 'ica')

                col_a, col_b = st.columns(2)

                with col_a:
                    ica_method = st.selectbox(
                        "ICA Algorithm",
                        options=['fastica', 'infomax'],
                        index=0 if ica_params.get('method') == 'fastica' else 1,
                        help="fastica = MNE FastICA (recommended, stable), infomax = MNE Infomax. Note: PICARD disabled due to NumPy 2.x incompatibility"
                    )

                    # Show note about PICARD
                    st.caption("⚠️ PICARD temporarily unavailable (NumPy 2.x incompatibility)")

                with col_b:
                    n_components = st.selectbox(
                        "Number of Components",
                        options=['rank-1', 'rank', 0.99, 0.999],
                        index=0,
                        format_func=lambda x: {
                            'rank-1': 'Auto (rank-1)',
                            'rank': 'Auto (rank)',
                            0.99: 'Variance 99%',
                            0.999: 'Variance 99.9%'
                        }.get(x, str(x)),
                        help="rank-1 accounts for average reference"
                    )

                # l_freq_fit for numerical stability
                l_freq_fit = st.number_input(
                    "ICA Fit Highpass (Hz)",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(ica_params.get('l_freq_fit', 1.0)),
                    step=0.1,
                    help="Fit ICA on highpassed copy (improves numerical stability, recommended: 1.0 Hz)"
                )

                auto_detect = st.checkbox(
                    "Auto-detect Artifacts (EOG/ECG)",
                    value=ica_params.get('auto_detect_artifacts', True),
                    help="Automatically find EOG/ECG components"
                )

                if ica_method == 'fastica' and l_freq_fit > 0:
                    st.success("✅ MNE FastICA + l_freq_fit - Numerically stable (recommended)")

                update_step_in_config(config, 'ica', {
                    'method': ica_method,
                    'n_components': n_components,
                    'random_state': 42,
                    'l_freq_fit': l_freq_fit if l_freq_fit > 0 else None,
                    'auto_detect_artifacts': auto_detect,
                    'reject_by_annotation': True
                })

            # Step 8: ICLabel
            with st.expander("8️⃣ ICLabel - Component Classification", expanded=True):
                iclabel_params = get_step_from_config(config, 'iclabel')

                threshold = st.slider(
                    "Classification Threshold",
                    min_value=0.5,
                    max_value=0.95,
                    value=float(iclabel_params.get('threshold', 0.8)),
                    step=0.05,
                    help="Probability threshold for labeling (0.7-0.8 typical)"
                )

                reject_labels = st.multiselect(
                    "Reject Artifact Types",
                    options=['eye', 'muscle', 'heart', 'line_noise', 'channel_noise'],
                    default=iclabel_params.get('reject_labels', ['eye', 'muscle']),
                    help="Component types to remove"
                )

                remove_components = st.checkbox(
                    "Remove Rejected Components",
                    value=iclabel_params.get('remove_components', True),
                    help="Apply ICA.exclude to remove components"
                )

                st.info("💡 ICLabel uses deep learning to classify ICA components")

                update_step_in_config(config, 'iclabel', {
                    'threshold': threshold,
                    'reject_labels': reject_labels,
                    'remove_components': remove_components
                })

            # Step 9: Reference
            with st.expander("9️⃣ Reference - Re-referencing", expanded=False):
                ref_params = get_step_from_config(config, 'reference')

                ref_type = st.selectbox(
                    "Reference Type",
                    options=['average', 'REST', 'custom'],
                    index=0,
                    help="Average reference is standard"
                )

                projection = st.checkbox(
                    "Use Projection",
                    value=ref_params.get('projection', True),
                    help="Apply reference as projection (reversible)"
                )

                update_step_in_config(config, 'reference', {
                    'type': ref_type,
                    'projection': projection
                })

            # Preview and save
            st.markdown("---")

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                if st.button("👁️ Preview YAML", width="stretch"):
                    # Sort steps for preview
                    preview_config = config.copy()
                    steps = preview_config.get('preprocessing', {}).get('steps', [])
                    preview_config['preprocessing']['steps'] = sort_steps_by_canonical_order(steps)
                    st.code(yaml.dump(preview_config, default_flow_style=False, sort_keys=False), language='yaml')

            with col_b:
                if st.button("💾 Save Config", type="primary", width="stretch"):
                    try:
                        # Sort steps by canonical order before saving
                        steps = config.get('preprocessing', {}).get('steps', [])
                        config['preprocessing']['steps'] = sort_steps_by_canonical_order(steps)

                        with open(config_path, 'w') as f:
                            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                        st.success(f"✅ Saved!")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            with col_c:
                if st.button("🗑️ Delete Config", width="stretch"):
                    try:
                        config_path.unlink()
                        st.success(f"✅ Deleted")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            # Show save location
            st.caption(f"📁 `{config_path}`")


if __name__ == "__main__":
    main()
