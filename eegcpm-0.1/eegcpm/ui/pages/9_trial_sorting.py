"""Trial Sorting - Filter, sort, split and export epochs.

This page allows you to:
1. Load combined epochs from the epochs stage
2. Inspect trials by experimental condition
3. Filter by behavioral response (correct/incorrect/etc.)
4. Bin trials by variables (RT quartiles, difficulty, etc.)
5. Apply arbitrary filters (conditions / response / accuracy / RT / pandas query)
6. Sort trials by a metadata column and split them into halves
7. Export the result as ``selections/<name>/`` (epochs + ``selection.json``)

Trial metadata is taken from ``epochs.metadata`` (written into the epochs FIF by
the epochs stage) and only falls back to a CSV beside the FIF, so the Response /
Bins / Filter / Sort tabs work on freshly extracted epochs without any manual
metadata handling. When neither source is available the page warns and offers a
CSV upload; the Condition tab keeps working from the epochs' event ids.
"""

import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mne

from eegcpm.core.paths import EEGCPMPaths
from eegcpm.core.task_config import TaskConfig, TrialSorter

#: Selection names become directory names (same rule as `eegcpm epochs-select`).
SELECTION_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

#: Characters replaced in derived file names (condition/bin labels).
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


def _safe_label(label) -> str:
    """Make a label safe for use inside a file name."""
    safe = _UNSAFE_NAME.sub("_", str(label)).strip("_")
    return safe or "unnamed"


def _valid_selection_name(name: str) -> bool:
    return bool(name) and SELECTION_NAME_PATTERN.fullmatch(name) is not None


def _metadata_csv_paths(epochs_dir: Path) -> list:
    """Metadata CSV candidates beside the epochs FIF (legacy first, then Pass 1)."""
    candidates = []
    legacy = epochs_dir / "metadata.csv"
    if legacy.is_file():
        candidates.append(legacy)
    for candidate in sorted(epochs_dir.glob("*_epochs_metadata.csv")):
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def load_combined_epochs(epochs_dir: Path) -> tuple:
    """Load combined epochs and their trial metadata.

    Metadata resolution order:
    1. ``epochs.metadata`` from the FIF (preferred — written by the epochs
       stage, so it cannot drift from the saved trials);
    2. ``metadata.csv`` then ``*_epochs_metadata.csv`` beside the FIF, attached
       to the in-memory epochs after a row-count check.

    Returns
    -------
    (epochs, metadata, source, error)
        ``epochs`` is None only when no usable epochs file exists (fatal).
        ``error`` is a non-fatal message (e.g. a mismatched CSV) that the caller
        must show; metadata is then None and the metadata tabs stay disabled.
    """
    epochs_dir = Path(epochs_dir)
    epoch_files = sorted(epochs_dir.glob("*_epo.fif"))

    if not epoch_files:
        return None, None, None, "No epoch files found"
    if len(epoch_files) > 1:
        return None, None, None, (
            "Multiple epoch files found "
            f"({[p.name for p in epoch_files]}); the epochs stage writes "
            "exactly one '*_epo.fif' per session directory."
        )

    epochs = mne.read_epochs(epoch_files[0], preload=True, verbose=False)

    if epochs.metadata is not None:
        return epochs, epochs.metadata, f"epochs FIF ({epoch_files[0].name})", None

    candidates = _metadata_csv_paths(epochs_dir)
    if not candidates:
        return epochs, None, None, None

    mismatches = []
    for csv_path in candidates:
        try:
            metadata = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError) as exc:
            mismatches.append(f"{csv_path.name}: unreadable ({exc})")
            continue
        if len(metadata) != len(epochs):
            mismatches.append(
                f"{csv_path.name}: {len(metadata)} rows vs {len(epochs)} epochs"
            )
            continue
        try:
            epochs.metadata = metadata
        except ValueError as exc:
            return epochs, None, None, f"{csv_path.name}: {exc}"
        return epochs, epochs.metadata, f"CSV ({csv_path.name})", None

    return epochs, None, None, (
        "Metadata CSV does not match the epochs; the Response/Bins/Filter/Sort "
        "tabs are disabled. " + "; ".join(mismatches)
    )


def attach_metadata(epochs, metadata: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Attach an uploaded metadata frame to the in-memory epochs.

    Raises
    ------
    ValueError
        Row count mismatch (a silent misalignment would corrupt every
        downstream selection).
    """
    if len(metadata) != len(epochs):
        raise ValueError(
            f"{source_label} has {len(metadata)} rows but the epochs have "
            f"{len(epochs)} trials; metadata must have exactly one row per trial."
        )
    epochs.metadata = metadata
    return epochs.metadata


def _parse_optional_float(text: str, field: str):
    """Parse a text field into a float, or None when blank."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be a number or blank, got {text!r}") from exc


def export_selection(
    output_dir: Path,
    *,
    name: str,
    subject_id: str,
    session: str,
    task: str,
    entries: list,
    source_epochs_file,
    overwrite: bool = False,
) -> dict:
    """Write epochs (one file per entry) and ``selection.json`` to ``output_dir``.

    ``entries`` is a list of ``(label, mne.Epochs, [SelectionRecord, ...])``.
    Returns ``{"output_dir": str, "files": [...], "record": str}``.
    """
    output_dir = Path(output_dir)
    existing = sorted(output_dir.glob("*_epo.fif")) if output_dir.is_dir() else []
    if existing and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains epochs output "
            f"({[p.name for p in existing]}); tick 'Overwrite' to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    multiple = len(entries) > 1
    files = []
    records = []
    for label, label_epochs, label_records in entries:
        suffix = f"_{_safe_label(label)}" if multiple else ""
        filename = f"{subject_id}{suffix}_epo.fif"
        label_epochs.save(output_dir / filename, overwrite=True, verbose=False)
        files.append(filename)
        for record in label_records or []:
            entry = {"label": label, "n_output_trials": int(len(label_epochs))}
            if record is not None:
                entry.update(asdict(record))
            records.append(entry)

    payload = {
        "name": name,
        "task": task,
        "subject": subject_id,
        "session": session,
        "source_epochs_file": str(source_epochs_file),
        "created_by": "ui:9_trial_sorting",
        "n_outputs": len(entries),
        "records": records,
        "outputs": files,
    }
    record_path = output_dir / "selection.json"
    record_path.write_text(json.dumps(payload, indent=2, default=str))
    return {
        "output_dir": str(output_dir),
        "files": files,
        "record": str(record_path),
    }


def _legacy_entries(sorted_dict: dict) -> list:
    """Convert a legacy ``{label: {'epochs', 'selection_record'}}`` dict."""
    return [
        (label, info["epochs"], [info.get("selection_record")])
        for label, info in sorted_dict.items()
    ]


def main():
    """Trial sorting interface."""
    st.set_page_config(
        page_title="Processing: Trial Sorting - EEGCPM",
        page_icon="⚙️",
        layout="wide",
    )

    st.title("⚙️ Processing: Trial Sorting")
    st.markdown(
        "Filter, sort and split epochs by experimental conditions, responses, "
        "RT, and arbitrary metadata queries"
    )

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
        help="Select task configuration",
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
        help="Select preprocessing variant",
    )

    # Scan available epochs (from the epochs stage)
    epochs_root = paths.derivatives_root / "epochs" / selected_pipeline / task_config.task_name

    if not epochs_root.exists():
        st.info(f"📁 No epochs found for {task_config.task_name}")
        st.info("👈 Run the **Epochs** stage first to extract epochs")
        st.stop()

    # Get subjects with epochs
    subjects_with_epochs = sorted([
        d.name.replace('sub-', '') for d in epochs_root.iterdir()
        if d.is_dir() and d.name.startswith('sub-')
    ])

    if not subjects_with_epochs:
        st.info("No epochs found. Run the **Epochs** stage first.")
        st.stop()

    subject_id = st.sidebar.selectbox(
        "Subject",
        options=subjects_with_epochs,
        help="Select subject",
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
        help="Select session",
    )

    epochs_dir = subject_epochs_dir / f"ses-{session}"

    # Load epochs
    if st.sidebar.button("🔍 Load Epochs", type="primary"):
        with st.spinner("Loading epochs..."):
            epoch_files = sorted(epochs_dir.glob("*_epo.fif"))
            epochs, metadata, metadata_source, error = load_combined_epochs(epochs_dir)

            if epochs is None:
                st.error(f"Error: {error}")
                st.stop()

            if error:
                st.error(error)

            st.session_state.epochs = epochs
            st.session_state.metadata = metadata
            st.session_state.metadata_source = metadata_source
            st.session_state.epochs_file = epoch_files[0] if epoch_files else None
            st.session_state.task_config = task_config
            source_note = f" (metadata: {metadata_source})" if metadata_source else ""
            st.success(f"✓ Loaded {len(epochs)} epochs{source_note}")

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
    metadata = st.session_state.get('metadata')
    metadata_source = st.session_state.get('metadata_source')
    epochs_file = st.session_state.get('epochs_file')
    # Fall back to the config loaded from the sidebar (the same object stored on
    # load) so a stale/partial session state can never KeyError the page.
    task_config = st.session_state.get('task_config', task_config)

    # A metadata frame loaded in an earlier rerun stays attached to the epochs.
    if metadata is None and epochs.metadata is not None:
        metadata = epochs.metadata
        st.session_state.metadata = metadata

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

    if metadata_source:
        st.caption(f"Trial metadata: {metadata_source}")

    # Metadata fallback: offer a CSV upload rather than failing silently.
    if metadata is None:
        st.warning(
            "⚠️ No trial metadata found: the epochs FIF has none and no "
            "metadata CSV sits beside it. The Response, Bins, Filter and Sort "
            "tabs are disabled until metadata is available."
        )
        uploaded = st.file_uploader(
            "Upload trial metadata CSV (one row per trial)", type=["csv"]
        )
        if uploaded is not None:
            try:
                candidate = pd.read_csv(uploaded)
                metadata = attach_metadata(epochs, candidate, uploaded.name)
            except (ValueError, pd.errors.ParserError) as exc:
                st.error(f"Cannot use {uploaded.name}: {exc}")
                metadata = None
            if metadata is not None:
                st.session_state.metadata = metadata
                st.session_state.metadata_source = f"CSV upload ({uploaded.name})"
                st.success(f"✓ Attached metadata from {uploaded.name}")

    st.markdown("---")

    # Sorting tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏷️ By Condition",
        "✅ By Response",
        "📊 By Bins",
        "🎯 Filter",
        "↕️ Sort / Split",
        "💾 Export",
    ])

    # ------------------------------------------------------------------
    # By condition (works without metadata: uses epochs.event_id)
    # ------------------------------------------------------------------
    with tab1:
        st.subheader("Sort by Experimental Condition")

        sorter = TrialSorter(task_config)

        try:
            sorted_by_condition = sorter.sort_by_condition(epochs)

            condition_data = []
            for cond_name, cond_info in sorted_by_condition.items():
                condition_data.append({
                    'Condition': cond_name,
                    'N Trials': cond_info['n_trials'],
                    'Description': cond_info.get('description', ''),
                })

            st.dataframe(pd.DataFrame(condition_data), width='stretch')

            st.session_state.sorted_by_condition = sorted_by_condition

        except Exception as e:
            st.error(f"Error sorting by condition: {e}")
            st.code(str(e))

    # ------------------------------------------------------------------
    # By response (metadata required)
    # ------------------------------------------------------------------
    with tab2:
        st.subheader("Sort by Behavioral Response")

        if not task_config.response_mapping:
            st.warning("⚠️ No response mapping defined in task config")
        elif metadata is None:
            st.warning(
                "⚠️ No trial metadata available. Attach metadata above (upload "
                "a CSV) to use this tab."
            )
        else:
            sorter = TrialSorter(task_config)

            try:
                sorted_by_response = sorter.sort_by_response(epochs, metadata)

                response_data = []
                for resp_name, resp_info in sorted_by_response.items():
                    response_data.append({
                        'Response': resp_name,
                        'N Trials': resp_info['n_trials'],
                        'Percentage': f"{100 * resp_info['n_trials'] / len(epochs):.1f}%",
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

                st.session_state.sorted_by_response = sorted_by_response

            except Exception as e:
                st.error(f"Error sorting by response: {e}")
                st.code(str(e))

    # ------------------------------------------------------------------
    # By bins (metadata required)
    # ------------------------------------------------------------------
    with tab3:
        st.subheader("Bin Trials by Variable")

        if not task_config.binning:
            st.warning("⚠️ No binning specifications defined in task config")
        elif metadata is None:
            st.warning(
                "⚠️ No trial metadata available for binning. Attach metadata "
                "above (upload a CSV) to use this tab."
            )
        else:
            bin_spec_names = [spec.name for spec in task_config.binning]
            selected_bin = st.selectbox(
                "Binning Variable",
                options=bin_spec_names,
                help="Select how to bin trials",
            )

            bin_spec = next(b for b in task_config.binning if b.name == selected_bin)

            sorter = TrialSorter(task_config)

            try:
                binned_epochs = sorter.bin_trials(epochs, metadata, bin_spec)

                bin_data = []
                for bin_name, bin_info in binned_epochs.items():
                    bin_data.append({
                        'Bin': bin_name,
                        'N Trials': bin_info['n_trials'],
                        'Percentage': f"{100 * bin_info['n_trials'] / len(epochs):.1f}%",
                    })

                df_bins = pd.DataFrame(bin_data)
                st.dataframe(df_bins, width='stretch')

                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(df_bins['Bin'], df_bins['N Trials'])
                ax.set_xlabel(f'{bin_spec.name} Bin')
                ax.set_ylabel('Number of Trials')
                ax.set_title(f'Trial Distribution by {bin_spec.name}')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close()

                st.session_state.binned_epochs = binned_epochs

            except Exception as e:
                st.error(f"Error binning trials: {e}")
                st.code(str(e))

    # ------------------------------------------------------------------
    # Filter (new): TrialSorter.select
    # ------------------------------------------------------------------
    with tab4:
        st.subheader("Filter Trials")
        st.caption(
            "Filters combine with AND. Leave a field empty to ignore it. "
            "Condition names come from the task config; the pandas query uses "
            "metadata column names (backticks for names with dots)."
        )

        if metadata is None:
            st.warning(
                "⚠️ No trial metadata available. Filtering needs per-trial "
                "metadata; attach it above (upload a CSV)."
            )
        else:
            sorter = TrialSorter(task_config)

            condition_names = [cond.name for cond in task_config.conditions]
            selected_conditions = st.multiselect(
                "Conditions", options=condition_names,
                help="Keep only these conditions (empty = all)",
            )

            response_choice = "(any)"
            accuracy_choice = "(any)"
            if task_config.response_mapping:
                response_choice = st.selectbox(
                    "Response category",
                    options=["(any)"] + list(task_config.response_mapping.categories),
                )
                accuracy_choice = st.selectbox(
                    "Accuracy",
                    options=["(any)", "correct (1)", "incorrect (0)"],
                )
            else:
                st.caption("This task config has no response mapping: response/accuracy filters are unavailable.")

            rt_col1, rt_col2 = st.columns(2)
            rt_min_text = rt_col1.text_input("RT min (s, blank = none)")
            rt_max_text = rt_col2.text_input("RT max (s, blank = none)")

            where = st.text_input(
                "pandas query (optional)",
                placeholder="e.g. `response.rt` > 0.3 and actualSSD > 0.2",
            )

            if st.button("🎯 Apply filter", type="primary"):
                try:
                    rt = (
                        _parse_optional_float(rt_min_text, "RT min"),
                        _parse_optional_float(rt_max_text, "RT max"),
                    )
                    accuracy = None
                    if accuracy_choice.startswith("correct"):
                        accuracy = 1
                    elif accuracy_choice.startswith("incorrect"):
                        accuracy = 0
                    selected_epochs, record = sorter.select(
                        epochs,
                        metadata=metadata,
                        conditions=selected_conditions or None,
                        response=None if response_choice == "(any)" else response_choice,
                        accuracy=accuracy,
                        rt=rt,
                        where=where.strip() or None,
                    )
                    st.session_state.filtered = {
                        "epochs": selected_epochs,
                        "record": record,
                    }
                    st.success(
                        f"✓ {len(selected_epochs)} / {len(epochs)} trials kept "
                        f"(mask {record.mask_hash[:12]}…)"
                    )
                except Exception as e:
                    st.error(f"Filter failed: {e}")

        if 'filtered' in st.session_state:
            filtered = st.session_state.filtered
            st.markdown("#### Filtered selection")
            st.metric("Trials kept", len(filtered['epochs']))
            if filtered['epochs'].metadata is not None:
                counts = (
                    filtered['epochs'].metadata['trial_type'].value_counts()
                    if 'trial_type' in filtered['epochs'].metadata.columns
                    else None
                )
                if counts is not None:
                    st.dataframe(
                        counts.rename_axis('Condition').reset_index(name='N Trials'),
                        width='stretch',
                    )
            st.json(asdict(filtered['record']))

    # ------------------------------------------------------------------
    # Sort / split (new): TrialSorter.sort + TrialSorter.split
    # ------------------------------------------------------------------
    with tab5:
        st.subheader("Sort Trials by Metadata Column")

        if metadata is None:
            st.warning(
                "⚠️ No trial metadata available. Sorting needs per-trial "
                "metadata; attach it above (upload a CSV)."
            )
        else:
            sorter = TrialSorter(task_config)
            sort_columns = sorted(str(c) for c in metadata.columns)
            default_index = 0
            rt_column = (
                task_config.response_mapping.rt_column
                if task_config.response_mapping
                else None
            )
            if rt_column in sort_columns:
                default_index = sort_columns.index(rt_column)

            sort_column = st.selectbox(
                "Sort by", options=sort_columns, index=default_index
            )
            ascending = st.checkbox("Ascending", value=True)

            if st.button("↕️ Sort trials"):
                try:
                    sorted_epochs, record = sorter.sort(
                        epochs, metadata=metadata, by=sort_column,
                        ascending=ascending,
                    )
                    st.session_state.sorted_selection = {
                        "epochs": sorted_epochs,
                        "record": record,
                    }
                    st.success(
                        f"✓ Sorted {len(sorted_epochs)} trials by {sort_column}"
                    )
                except Exception as e:
                    st.error(f"Sort failed: {e}")

            if 'sorted_selection' in st.session_state:
                sorted_selection = st.session_state.sorted_selection
                st.caption(
                    "Sorted order (the epoch data is reordered with the "
                    "metadata, so they stay aligned):"
                )
                st.dataframe(
                    sorted_selection['epochs'].metadata[sort_column]
                    .head(20).rename('value').reset_index(),
                    width='stretch',
                )

            st.markdown("---")
            st.subheader("Split Trials")
            split_mode = st.radio(
                "Split mode",
                options=["Sequential halves", "By column quantiles"],
                horizontal=True,
            )
            split_parts = st.number_input(
                "Parts", min_value=2, max_value=10, value=2, step=1
            )
            split_column = None
            if split_mode == "By column quantiles":
                split_column = st.selectbox(
                    "Split column", options=sort_columns, key="split_column"
                )

            if st.button("✂️ Split trials"):
                try:
                    parts = sorter.split(
                        epochs,
                        metadata=metadata,
                        by="position" if split_column is None else split_column,
                        parts=int(split_parts),
                    )
                    st.session_state.split_selection = parts
                    st.success(f"✓ Split {len(epochs)} trials into {len(parts)} parts")
                except Exception as e:
                    st.error(f"Split failed: {e}")

            if 'split_selection' in st.session_state:
                split_rows = [
                    {"Part": label, "N Trials": len(part_epochs)}
                    for label, (part_epochs, _record) in
                    st.session_state.split_selection.items()
                ]
                st.dataframe(pd.DataFrame(split_rows), width='stretch')

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    with tab6:
        st.subheader("Export Selection")

        st.markdown(
            "Writes the selection to `derivatives/epochs/{preprocessing}/{task}/"
            "selections/{name}/` together with a `selection.json` provenance "
            "record (the same layout as `eegcpm epochs-select`)."
        )

        export_options = {}
        if 'sorted_by_condition' in st.session_state:
            export_options["By Condition"] = (
                "by-condition",
                _legacy_entries(st.session_state.sorted_by_condition),
            )
        if 'sorted_by_response' in st.session_state:
            export_options["By Response"] = (
                "by-response",
                _legacy_entries(st.session_state.sorted_by_response),
            )
        if 'binned_epochs' in st.session_state:
            export_options["By Bins"] = (
                "by-bins",
                _legacy_entries(st.session_state.binned_epochs),
            )
        if 'filtered' in st.session_state:
            filtered = st.session_state.filtered
            export_options["Filtered Selection"] = (
                "filtered",
                [("selection", filtered['epochs'], [filtered['record']])],
            )
        if 'sorted_selection' in st.session_state:
            sorted_selection = st.session_state.sorted_selection
            export_options["Sorted Selection"] = (
                "sorted",
                [("selection", sorted_selection['epochs'],
                  [sorted_selection['record']])],
            )
        if 'split_selection' in st.session_state:
            export_options["Split Parts"] = (
                "split",
                [
                    (label, part_epochs, [record])
                    for label, (part_epochs, record) in
                    st.session_state.split_selection.items()
                ],
            )

        if not export_options:
            st.info(
                "⚠️ Create a selection first (Condition / Response / Bins / "
                "Filter / Sort tabs), then return here to export"
            )
        else:
            export_choice = st.radio(
                "Export Type", options=list(export_options), horizontal=True
            )
            default_name, entries = export_options[export_choice]

            name = st.text_input("Selection name", value=default_name)
            overwrite = st.checkbox("Overwrite an existing selection", value=False)

            output_base = (
                paths.derivatives_root
                / "epochs"
                / selected_pipeline
                / task_config.task_name
                / "selections"
                / name
                / f"sub-{subject_id}"
                / f"ses-{session}"
            )
            st.caption(f"Output directory: `{output_base}`")

            if st.button("💾 Export Selection", type="primary"):
                if not _valid_selection_name(name):
                    st.error(
                        "Invalid selection name: use letters, digits, '-' and "
                        "'_' only (it becomes a directory name)."
                    )
                else:
                    with st.spinner("Exporting..."):
                        try:
                            result = export_selection(
                                output_base,
                                name=name,
                                subject_id=subject_id,
                                session=session,
                                task=task_config.task_name,
                                entries=entries,
                                source_epochs_file=epochs_file,
                                overwrite=overwrite,
                            )
                            for filename in result["files"]:
                                st.success(f"✓ Saved {filename}")
                            st.info(
                                f"📁 Output: `{result['output_dir']}`\n\n"
                                f"📄 Provenance: `{Path(result['record']).name}`"
                            )
                            st.balloons()
                        except Exception as e:
                            st.error(f"Export error: {e}")
                            import traceback
                            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
