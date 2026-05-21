"""Task Configuration - Interactive task config builder.

This page allows you to:
1. Scan BIDS events to discover available triggers
2. Interactively configure conditions, timing, and responses
3. Preview and save task configuration YAML files
"""

import streamlit as st
from pathlib import Path
import sys
import yaml
import pandas as pd
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eegcpm.ui.utils import scan_subjects, scan_tasks
from eegcpm.core.paths import EEGCPMPaths
from eegcpm.modules.epochs import sanitize_name


def scan_events_from_bids(bids_root: Path, task: str, max_subjects: int | None = 50) -> dict:
    """Scan BIDS events.tsv files to find unique event types and counts.

    Args:
        bids_root: Path to BIDS root directory
        task: Task name to scan
        max_subjects: Maximum subjects to scan. If None, scans all subjects.

    Returns:
        Dictionary with event statistics:
        {
            'event_names': Counter({'target_left': 450, 'target_right': 438, ...}),
            'columns': ['onset', 'duration', 'trial_type', 'response_time', 'correct'],
            'sampling_info': {'total_files': 10, 'total_events': 888}
        }
    """
    event_names = Counter()
    columns_set = set()
    total_files = 0
    total_events = 0

    subject_dirs = sorted(bids_root.glob("sub-*"))
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]

    # Scan subject directories
    for subject_dir in subject_dirs:
        if not subject_dir.is_dir():
            continue

        # Find events.tsv files for this task
        events_files = list(subject_dir.rglob(f"*task-{task}*_events.tsv"))

        for events_file in events_files:
            try:
                df = pd.read_csv(events_file, sep='\t')
                total_files += 1
                total_events += len(df)

                # Track column names
                columns_set.update(df.columns.tolist())

                # Count event types (usually in 'trial_type' column)
                if 'trial_type' in df.columns:
                    event_names.update(df['trial_type'].dropna().astype(str).tolist())

            except Exception as e:
                st.warning(f"Could not read {events_file.name}: {e}")
                continue

    return {
        'event_names': event_names,
        'columns': sorted(list(columns_set)),
        'sampling_info': {
            'total_files': total_files,
            'total_events': total_events,
        }
    }


def preview_yaml_config(config_dict: dict) -> str:
    """Generate YAML preview from config dictionary."""
    return yaml.dump(config_dict, default_flow_style=False, sort_keys=False)


def main():
    """Task configuration interface."""

    st.set_page_config(
        page_title="Configuration: Task/Epochs - EEGCPM",
        page_icon="🔧",
        layout="wide"
    )

    st.title("🔧 Configuration: Task & Epochs")
    st.markdown("Configure epoch timing, event conditions, and trial filtering parameters")

    # Get paths from main app project selection
    from eegcpm.ui.project_manager import ProjectManager

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

    # Task selection
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Task Selection")

    # Get available tasks from BIDS
    tasks = scan_tasks(paths.bids_root)

    if not tasks:
        st.error("No tasks found in BIDS directory")
        st.stop()

    selected_task = st.sidebar.selectbox(
        "Task to Configure",
        options=tasks,
        help="Select which task to configure"
    )

    # Load existing config section
    st.sidebar.markdown("---")
    st.sidebar.header("📂 Load Existing Config")

    config_dir = paths.eegcpm_root / "configs" / "tasks"
    config_dir.mkdir(parents=True, exist_ok=True)

    existing_configs = sorted([f.stem for f in config_dir.glob("*.yaml")])

    if existing_configs:
        # Default to config matching task name if it exists
        default_index = 0  # '(Create New)'
        if selected_task in existing_configs:
            default_index = existing_configs.index(selected_task) + 1  # +1 for '(Create New)'

        selected_config = st.sidebar.selectbox(
            "Existing Configs",
            options=['(Create New)'] + existing_configs,
            index=default_index,
            help="Load an existing config to edit"
        )

        if selected_config != '(Create New)':
            load_button = st.sidebar.button("📥 Load Config", type="secondary")
            if load_button:
                config_path = config_dir / f"{selected_config}.yaml"
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)

                # Store in session state
                st.session_state.loaded_config = loaded_config
                st.session_state.config_name = selected_config
                st.success(f"✓ Loaded config: {selected_config}")
                st.rerun()
    else:
        st.sidebar.info("No existing configs found")

    # Scan events button
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Scan Settings")
    scan_all = st.sidebar.checkbox("Scan all subjects", value=False)
    if scan_all:
        max_subjects = None
        st.sidebar.caption("Scanning all subjects (may be slow for large datasets)")
    else:
        max_subjects = st.sidebar.number_input(
            "Max subjects to scan",
            min_value=1,
            value=50,
            step=10,
            help="Limit how many subjects are scanned for events"
        )
    scan_button = st.sidebar.button("🔍 Scan Events", type="primary")

    # Clear config button
    if st.sidebar.button("🆕 New Config", help="Clear loaded config and start fresh"):
        st.session_state.loaded_config = None
        st.session_state.conditions = []
        st.session_state.response_categories = {'correct': 1, 'incorrect': 0}
        st.session_state.config_name = selected_task
        st.rerun()

    # Initialize session state
    if 'event_data' not in st.session_state:
        st.session_state.event_data = None
    if 'config_name' not in st.session_state:
        st.session_state.config_name = selected_task
    if 'loaded_config' not in st.session_state:
        st.session_state.loaded_config = None

    # Auto-load matching config if exists (only on first load)
    if 'auto_loaded' not in st.session_state:
        st.session_state.auto_loaded = False
        # Try to find existing config with same name as selected task
        matching_config = config_dir / f"{selected_task}.yaml"
        if matching_config.exists():
            with open(matching_config, 'r') as f:
                st.session_state.loaded_config = yaml.safe_load(f)
                st.session_state.config_name = selected_task
                st.info(f"ℹ️ Auto-loaded existing config: {selected_task}.yaml")

    # Apply loaded config to session state if present
    loaded_config = st.session_state.loaded_config
    if loaded_config:
        # Extract values from loaded config
        if 'conditions' not in st.session_state or not st.session_state.conditions:
            st.session_state.conditions = loaded_config.get('conditions', [])

        # Response mapping
        response_mapping = loaded_config.get('response_mapping', {})
        if response_mapping and 'response_categories' not in st.session_state:
            st.session_state.response_categories = response_mapping.get('categories', {})
    else:
        # Default initialization
        if 'conditions' not in st.session_state:
            st.session_state.conditions = []
        if 'response_categories' not in st.session_state:
            st.session_state.response_categories = {'correct': 1, 'incorrect': 0}

    # Scan events from BIDS
    if scan_button:
        with st.spinner(f"Scanning events for task '{selected_task}'..."):
            event_data = scan_events_from_bids(paths.bids_root, selected_task, max_subjects=max_subjects)
            st.session_state.event_data = event_data
            st.success(f"✓ Scanned {event_data['sampling_info']['total_files']} files, found {event_data['sampling_info']['total_events']} events")

    # Main interface
    if st.session_state.event_data is None:
        st.info("👈 Select a task and click 'Scan Events' to begin")

        st.markdown("""
        ### How to Use

        1. **Select Task**: Choose which task to configure from the sidebar
        2. **Scan Events**: Click 'Scan Events' to discover all event types in your data
        3. **Configure Conditions**: Group events into experimental conditions
        4. **Set Timing**: Configure epoch timing (tmin, tmax, baseline)
        5. **Add Response Mapping**: (Optional) Configure behavioral response categories
        6. **Preview & Save**: Review the YAML and save to configs directory

        ### What Are Task Configs?

        Task configurations define:
        - **Conditions**: Which events belong to which experimental conditions
        - **Epoch Timing**: Time windows around events for analysis
        - **Response Mapping**: How to categorize behavioral responses (correct/incorrect, etc.)
        - **Binning**: How to group trials by continuous variables (RT, difficulty, etc.)

        These configs are used by:
        - ERP QC generation (preprocessing reports)
        - Trial sorting (grouping epochs by condition/response)
        - Feature extraction (condition-specific analysis)
        """)

        return

    event_data = st.session_state.event_data

    # Display discovered events
    st.header("📊 Discovered Events")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Files Scanned", event_data['sampling_info']['total_files'])
    with col2:
        st.metric("Total Events", event_data['sampling_info']['total_events'])
    with col3:
        st.metric("Unique Event Types", len(event_data['event_names']))

    # Show event counts
    st.subheader("Event Type Counts")
    event_df = pd.DataFrame([
        {'Event Type': event, 'Count': count}
        for event, count in event_data['event_names'].most_common()
    ])
    st.dataframe(event_df, width='stretch', hide_index=True)

    # Show available columns
    with st.expander("Available Columns in events.tsv"):
        st.code(", ".join(event_data['columns']))

    st.markdown("---")

    # Configuration builder
    st.header("⚙️ Build Configuration")

    # Configuration name
    config_name = st.text_input(
        "Configuration Name",
        value=st.session_state.config_name,
        help="Name for this configuration (will be saved as {name}.yaml)"
    )

    # Task description - load from config or use default
    default_description = loaded_config.get('description', f"{selected_task} task configuration") if loaded_config else f"{selected_task} task configuration"

    task_description = st.text_area(
        "Task Description",
        value=default_description,
        help="Human-readable description of this task"
    )

    # Task type - determines the analysis workflow
    default_task_type = loaded_config.get('task_type', 'event-related') if loaded_config else 'event-related'
    task_type = st.radio(
        "Task Type",
        options=["event-related", "continuous"],
        index=0 if default_task_type == "event-related" else 1,
        horizontal=True,
        help="event-related: time-locked to stimulus events | continuous: resting state or movie"
    )

    # Tab interface for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⏱️ Epoch Timing",
        "🏷️ Conditions",
        "✅ Response Mapping",
        "🚫 Rejection & Binning",
        "💾 Preview & Save"
    ])

    with tab1:
        st.subheader("Epoch Timing Configuration")

        st.markdown("""
        **Define the time window around each event to extract**

        Epochs are time-locked segments of EEG centered on specific events (e.g., stimulus onset, button press).
        These parameters determine:
        - How much pre-stimulus baseline to include
        - How much post-stimulus response to capture
        - What baseline period to use for normalization
        """)

        # Get defaults from loaded config or use hardcoded defaults
        default_tmin = loaded_config.get('tmin', -0.3) if loaded_config else -0.3
        default_tmax = loaded_config.get('tmax', 0.8) if loaded_config else 0.8
        default_baseline = loaded_config.get('baseline', [-0.2, 0.0]) if loaded_config else [-0.2, 0.0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**⏪ Pre-stimulus**")
            tmin = st.number_input(
                "tmin (seconds)",
                value=float(default_tmin),
                min_value=-5.0,
                max_value=0.0,
                step=0.1,
                help="Start time before event (negative value). Typical: -0.2 to -0.5 seconds for baseline."
            )
            st.caption("💡 Event occurs at t=0")

        with col2:
            st.markdown("**⏩ Post-stimulus**")
            tmax = st.number_input(
                "tmax (seconds)",
                value=float(default_tmax),
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                help="End time after event. Typical: 0.5-1.0s for ERPs, 2-3s for motor responses."
            )
            st.caption("💡 Longer windows capture later components (P300, LPP)")

        with col3:
            st.markdown("**📊 Baseline Correction**")
            baseline_start = st.number_input(
                "Baseline start",
                value=float(default_baseline[0]),
                min_value=-5.0,
                max_value=0.0,
                step=0.1,
                help="Baseline start time. Should be within [tmin, 0]. Typical: -200ms before stimulus."
            )
            baseline_end = st.number_input(
                "Baseline end",
                value=float(default_baseline[1]),
                min_value=-5.0,
                max_value=0.0,
                step=0.1,
                help="Baseline end time. Usually 0 (stimulus onset). Can use pre-stimulus period."
            )
            st.caption("💡 Removes DC offset and slow drifts")

        st.info(f"📏 **Epoch window**: [{tmin}, {tmax}] seconds ({(tmax-tmin)*1000:.0f}ms total) | **Baseline**: [{baseline_start}, {baseline_end}]")

    with tab2:
        st.subheader("Experimental Conditions")
        st.markdown("""
        **Define which events belong to which experimental conditions**

        Conditions represent your independent variables (what you manipulated in the experiment):
        - **Stimulus type**: target vs distractor, left vs right, easy vs hard
        - **Task condition**: Go vs NoGo, congruent vs incongruent
        - **Cognitive load**: 1-back vs 2-back vs 3-back

        Each condition can include multiple event codes. For example:
        - Condition "target" might include event codes: `target_left` + `target_right`
        - Condition "correct_response" might include: `correct_left` + `correct_right`
        """)

        # Add condition button
        if st.button("➕ Add Condition"):
            st.session_state.conditions.append({
                'name': sanitize_name(f'condition_{len(st.session_state.conditions) + 1}'),
                'event_codes': [],
                'description': ''
            })

        # Display and edit conditions
        available_events = list(event_data['event_names'].keys())

        for i, cond in enumerate(st.session_state.conditions):
            with st.expander(f"Condition {i+1}: {cond['name']}", expanded=True):
                col1, col2 = st.columns([1, 3])

                with col1:
                    raw_name = st.text_input(
                        "Condition Name",
                        value=cond['name'],
                        key=f"cond_name_{i}"
                    )
                    sanitized = sanitize_name(raw_name)
                    if sanitized != raw_name and raw_name:
                        st.caption(f"Saved as: `{sanitized}`")
                    cond['name'] = sanitized

                with col2:
                    cond['description'] = st.text_input(
                        "Description",
                        value=cond['description'],
                        key=f"cond_desc_{i}"
                    )

                # Filter event codes to only include valid options from available_events
                # This handles cases where config has numeric codes (8, 9) but UI expects strings
                valid_defaults = [
                    ec for ec in cond['event_codes']
                    if ec in available_events
                ]

                # Show warning if some codes were invalid
                if len(valid_defaults) != len(cond['event_codes']):
                    invalid_codes = [ec for ec in cond['event_codes'] if ec not in available_events]
                    st.warning(f"⚠️ Config contains event codes not found in scanned events: {invalid_codes}")
                    st.info(f"Available events: {available_events}")

                cond['event_codes'] = st.multiselect(
                    "Event Types",
                    options=available_events,
                    default=valid_defaults,
                    key=f"cond_events_{i}",
                    help="Select which event types belong to this condition"
                )

                if st.button(f"🗑️ Remove", key=f"remove_cond_{i}"):
                    st.session_state.conditions.pop(i)
                    st.rerun()

        if not st.session_state.conditions:
            st.warning("⚠️ No conditions defined. Click 'Add Condition' to start.")

    with tab3:
        st.subheader("Behavioral Response Mapping (Optional)")
        st.markdown("""
        **Link brain activity to behavioral performance**

        Response mapping allows you to:
        - Label trials as correct/incorrect based on accuracy
        - Track reaction times for each trial
        - Filter trials by performance (e.g., only analyze correct trials)
        - Bin trials by RT for speed-accuracy analyses

        This information comes from the `events.tsv` file in your BIDS data, which contains:
        - Trial-by-trial response accuracy
        - Reaction times
        - Other behavioral metadata
        """)

        st.info("💡 **When to use response mapping:**\n"
                "- You have behavioral responses recorded in events.tsv\n"
                "- You want to analyze correct vs incorrect trials separately\n"
                "- You're interested in RT effects on brain activity\n"
                "- You need to filter out trials with missing/invalid responses")

        # Check if response mapping exists in loaded config
        loaded_response = loaded_config.get('response_mapping', {}) if loaded_config else {}
        has_response = bool(loaded_response)

        enable_response = st.checkbox("Enable Response Mapping", value=has_response)

        if enable_response:
            # Get default response column from loaded config
            default_response_col = loaded_response.get('response_column', event_data['columns'][0] if event_data and event_data['columns'] else 'correct')

            response_column = st.selectbox(
                "Response Column",
                options=event_data['columns'] if event_data else [default_response_col],
                index=event_data['columns'].index(default_response_col) if event_data and default_response_col in event_data['columns'] else 0,
                help="Column in events.tsv containing response data"
            )

            st.markdown("**Response Categories**")
            st.markdown("Define how values in the response column map to categories:")

            # Edit response categories
            for key in list(st.session_state.response_categories.keys()):
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.text_input("Category", value=key, disabled=True, key=f"resp_cat_{key}")
                with col2:
                    st.session_state.response_categories[key] = st.number_input(
                        "Value",
                        value=st.session_state.response_categories[key],
                        key=f"resp_val_{key}"
                    )
                with col3:
                    if st.button("🗑️", key=f"remove_resp_{key}"):
                        del st.session_state.response_categories[key]
                        st.rerun()

            # Add category
            col1, col2 = st.columns([2, 1])
            with col1:
                new_category = st.text_input("New Category Name", key="new_resp_cat")
            with col2:
                if st.button("➕ Add Category"):
                    if new_category and new_category not in st.session_state.response_categories:
                        st.session_state.response_categories[new_category] = 0
                        st.rerun()

            # RT columns (optional)
            st.markdown("**Reaction Time (Optional)**")

            # Get RT defaults from loaded config
            default_rt_col = loaded_response.get('rt_column', 'None')
            default_rt_min = loaded_response.get('rt_min', 0.15)
            default_rt_max = loaded_response.get('rt_max', 2.0)

            rt_options = ['None'] + (event_data['columns'] if event_data else [])
            rt_column = st.selectbox(
                "RT Column",
                options=rt_options,
                index=rt_options.index(default_rt_col) if default_rt_col in rt_options else 0,
                help="Column containing reaction times"
            )

            if rt_column != 'None':
                col1, col2 = st.columns(2)
                with col1:
                    rt_min = st.number_input("RT Min (seconds)", value=float(default_rt_min), step=0.01)
                with col2:
                    rt_max = st.number_input("RT Max (seconds)", value=float(default_rt_max), step=0.1)
        else:
            response_column = None
            rt_column = None

    with tab4:
        st.subheader("Artifact Rejection & Trial Binning")

        st.info("""
        **What is epoch rejection?**

        After creating epochs (time-locked segments around events), individual trials can be
        automatically rejected if they contain artifacts (eye blinks, muscle activity, sensor issues).
        This happens AFTER preprocessing, during epoch extraction.
        """)

        # Artifact rejection section
        st.markdown("### 🚫 Artifact Rejection")
        st.markdown("""
        **Automatic trial rejection based on signal amplitude**

        These thresholds help remove individual trials contaminated by artifacts:
        - **Peak-to-peak amplitude**: Rejects trials with excessive voltage swings (eye blinks, muscle artifacts)
        - **Flat signal**: Rejects trials where channels show no activity (poor sensor contact, broken electrode)
        """)

        # Get defaults from loaded config
        loaded_reject = loaded_config.get('reject', None) if loaded_config else None
        loaded_flat = loaded_config.get('flat', None) if loaded_config else None

        enable_reject = st.checkbox(
            "Enable Amplitude Rejection",
            value=bool(loaded_reject),
            help="Reject epochs exceeding amplitude thresholds"
        )

        if enable_reject:
            st.markdown("""
            **Recommended thresholds:**
            - EEG: 100-200 µV (stricter for clean data, more lenient for noisy data)
            - EOG: 200-300 µV (eye channels have larger signals)
            - Flat: 1-10 µV (detects dead/disconnected channels)
            """)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Peak-to-peak amplitude (reject if exceeded)**")
                st.caption("⚠️ Rejects trials where voltage difference exceeds threshold")

                eeg_reject = st.number_input(
                    "EEG threshold (µV)",
                    value=float(loaded_reject.get('eeg', 150e-6) * 1e6) if loaded_reject and 'eeg' in loaded_reject else 150.0,
                    min_value=0.0,
                    step=10.0,
                    help="Reject epochs with EEG amplitude > threshold. Common artifacts: eye blinks (100-300µV), muscle (50-200µV)"
                )

                eog_reject = st.number_input(
                    "EOG threshold (µV)",
                    value=float(loaded_reject.get('eog', 250e-6) * 1e6) if loaded_reject and 'eog' in loaded_reject else 250.0,
                    min_value=0.0,
                    step=10.0,
                    help="Reject epochs with EOG amplitude > threshold. Eye channels naturally have larger signals than EEG."
                )

            with col2:
                st.markdown("**Flat signal detection (reject if below)**")
                st.caption("⚠️ Rejects trials where signal variance is too low (sensor issues)")

                eeg_flat = st.number_input(
                    "EEG flat threshold (µV)",
                    value=float(loaded_flat.get('eeg', 1e-6) * 1e6) if loaded_flat and 'eeg' in loaded_flat else 1.0,
                    min_value=0.0,
                    step=0.1,
                    help="Reject epochs with signal variance < threshold. Indicates broken electrode, poor contact, or amplifier issues."
                )

                st.caption("💡 Typical healthy EEG variance: 10-50 µV")


        else:
            eeg_reject = None
            eog_reject = None
            eeg_flat = None

        # Trial binning section
        st.markdown("---")
        st.markdown("### 📊 Trial Binning (Optional)")
        st.markdown("""
        **Group trials by behavioral or stimulus variables**

        Trial binning allows you to organize epochs into sub-groups for more detailed analysis:
        - **RT quartiles**: Compare fast vs slow responses
        - **Stimulus × Accuracy**: Analyze target_correct vs target_incorrect separately
        - **Difficulty levels**: Compare easy vs medium vs hard trials
        - **Learning effects**: Compare early vs late trials in session

        This is useful for:
        - Examining behavioral modulation of brain activity
        - Stratified analysis (e.g., ERP differences by RT)
        - Feature extraction for predictive modeling
        """)

        # Get defaults from loaded config
        loaded_binning = loaded_config.get('binning', None) if loaded_config else None

        enable_binning = st.checkbox(
            "Enable Trial Binning",
            value=bool(loaded_binning),
            help="Bin trials by RT quartiles, difficulty, etc."
        )

        if enable_binning:
            st.warning("🚧 **Trial binning UI is under development**")
            st.markdown("""
            For now, you can manually edit the YAML config to add binning specifications.

            **Common binning strategies:**

            **1. RT Quartiles** - Split trials into 4 speed groups:
            ```yaml
            binning:
              - name: rt_quartiles
                method: quantile
                n_bins: 4
                column: response_time
                labels: [fast, medium_fast, medium_slow, slow]
            ```

            **2. Stimulus × Accuracy** - Cross conditions with performance:
            ```yaml
            binning:
              - name: condition_by_accuracy
                method: custom
                # Automatically creates: target_left_correct, target_left_incorrect, etc.
            ```

            **3. Custom bins** - Define your own ranges:
            ```yaml
            binning:
              - name: difficulty
                method: custom
                bin_edges: [0, 0.5, 0.8, 1.0]
                column: difficulty_score
                labels: [easy, medium, hard]
            ```

            📖 See `planning/TASK_SYSTEM.md` for complete binning documentation.
            """)

    with tab5:
        st.subheader("Configuration Preview")

        # Build config dictionary
        config_dict = {
            'task_name': config_name,
            'description': task_description,
            'task_type': task_type,
            'tmin': float(tmin),
            'tmax': float(tmax),
            'baseline': [float(baseline_start), float(baseline_end)],
            'conditions': [
                {
                    'name': cond['name'],
                    'event_codes': cond['event_codes'],
                    'description': cond['description']
                }
                for cond in st.session_state.conditions
            ] if st.session_state.conditions else [],
        }

        # Add response mapping if enabled
        if enable_response:
            config_dict['response_mapping'] = {
                'response_column': response_column,
                'categories': st.session_state.response_categories
            }

            if rt_column != 'None':
                config_dict['response_mapping']['rt_column'] = rt_column
                config_dict['response_mapping']['rt_min'] = float(rt_min)
                config_dict['response_mapping']['rt_max'] = float(rt_max)
        else:
            config_dict['response_mapping'] = None

        # Add artifact rejection if enabled
        if enable_reject:
            reject_dict = {}
            flat_dict = {}

            if eeg_reject is not None:
                reject_dict['eeg'] = float(eeg_reject * 1e-6)  # Convert µV to V
            if eog_reject is not None:
                reject_dict['eog'] = float(eog_reject * 1e-6)  # Convert µV to V
            if eeg_flat is not None:
                flat_dict['eeg'] = float(eeg_flat * 1e-6)  # Convert µV to V

            config_dict['reject'] = reject_dict if reject_dict else None
            config_dict['flat'] = flat_dict if flat_dict else None
        else:
            config_dict['reject'] = None
            config_dict['flat'] = None

        # Add binning if enabled
        if enable_binning and loaded_binning:
            # Preserve existing binning config from loaded file
            config_dict['binning'] = loaded_binning
        else:
            config_dict['binning'] = None

        # Preview YAML
        yaml_preview = preview_yaml_config(config_dict)
        st.code(yaml_preview, language='yaml')

        # Validation
        if not config_dict['conditions']:
            st.warning("⚠️ No conditions defined. Add at least one condition before saving.")

        # Save button
        col1, col2 = st.columns([3, 1])

        with col2:
            if st.button("💾 Save Config", type="primary", disabled=not config_dict['conditions']):
                # Save to configs directory
                config_dir = paths.eegcpm_root / "configs" / "tasks"
                config_dir.mkdir(parents=True, exist_ok=True)

                output_file = config_dir / f"{config_name}.yaml"

                with open(output_file, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

                st.success(f"✓ Configuration saved to: {output_file}")
                st.balloons()


if __name__ == "__main__":
    main()
