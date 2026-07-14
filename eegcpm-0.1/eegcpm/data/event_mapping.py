"""Event mapping utilities for translating BIDS events.

Handles conversion between semantic event names (trial_type) and
numeric event codes (value) from BIDS events.tsv files.
"""

from pathlib import Path
from typing import Dict, Union, Optional
import pandas as pd


def load_event_mapping_from_bids(
    events_file: Path
) -> Dict[str, Union[int, str]]:
    """Load mapping from trial_type to value from BIDS events.tsv.

    BIDS events.tsv has columns:
    - trial_type: Semantic event name (e.g., 'target_left', 'target_right')
    - value: Numeric event code (e.g., 8, 9)

    When MNE loads BIDS .fif files, annotations use the numeric 'value' as
    description strings, not the semantic 'trial_type' names.

    Parameters
    ----------
    events_file : Path
        Path to BIDS events.tsv file

    Returns
    -------
    dict
        Mapping from trial_type (str) to value (int or str)
        Example: {'target_left': '8', 'target_right': '9'}
    """
    if not events_file.exists():
        return {}

    df = pd.read_csv(events_file, sep='\t')

    # Check for required columns
    if 'trial_type' not in df.columns or 'value' not in df.columns:
        return {}

    # Build mapping: trial_type -> value
    # Convert value to string to match how MNE stores annotations
    mapping = {}
    for _, row in df.iterrows():
        trial_type = row['trial_type']
        value = str(int(row['value']))  # Convert to string to match annotations
        if pd.notna(trial_type) and pd.notna(value):
            mapping[trial_type] = value

    return mapping


def translate_event_codes(
    event_codes: list,
    mapping: Dict[str, Union[int, str]]
) -> list:
    """Translate semantic event codes to numeric codes using mapping.

    Parameters
    ----------
    event_codes : list
        List of event codes (can be semantic names or numeric codes)
    mapping : dict
        Mapping from semantic names to numeric codes

    Returns
    -------
    list
        Translated event codes (semantic names replaced with numeric codes)
    """
    translated = []
    for code in event_codes:
        # If code is a semantic name and exists in mapping, translate it
        if isinstance(code, str) and code in mapping:
            translated.append(mapping[code])
        else:
            # Already numeric or not in mapping, keep as-is
            translated.append(str(code) if not isinstance(code, str) else code)

    return translated


def load_events_from_bids_tsv(
    events_tsv_path: Path,
    sfreq: float,
    trial_type_col: str = "trial_type",
    filter_col: Optional[str] = None,
    filter_val: Optional[str] = None,
    max_time: Optional[float] = None,
) -> tuple:
    """Load MNE-compatible events from a BIDS events.tsv file.

    Creates an events array and event_id dict from onset/trial_type columns,
    useful when the preprocessed FIF has no embedded annotations.

    Parameters
    ----------
    events_tsv_path : Path
        Path to BIDS events.tsv file.
    sfreq : float
        Sampling frequency (Hz) for converting onset seconds to samples.
    trial_type_col : str
        Column name for trial type/condition labels (default: 'trial_type').
    filter_col : str, optional
        Column to filter on (e.g., 'phase' to select 'main' trials only).
    filter_val : str, optional
        Value to match in filter_col (e.g., 'main').
    max_time : float, optional
        Maximum onset time in seconds; events beyond this are excluded.

    Returns
    -------
    tuple of (np.ndarray, dict) or (None, None)
        events: shape (n_events, 3), compatible with mne.Epochs
        event_id: dict mapping trial_type to integer code
    """
    import numpy as np
    import pandas as pd

    if not events_tsv_path.exists():
        return None, None

    df = pd.read_csv(events_tsv_path, sep='\t')

    if 'onset' not in df.columns or trial_type_col not in df.columns:
        return None, None

    # Apply optional filter (e.g., phase == "main")
    if filter_col and filter_val and filter_col in df.columns:
        df = df[df[filter_col] == filter_val]

    # Exclude rows with missing onset/trial_type and beyond max_time
    valid = df['onset'].notna() & df[trial_type_col].notna()
    if max_time is not None:
        valid = valid & (df['onset'] <= max_time)
    df_valid = df[valid]

    if len(df_valid) == 0:
        return None, None

    unique_types = sorted(df_valid[trial_type_col].unique())
    event_id = {tt: i + 1 for i, tt in enumerate(unique_types)}

    events = np.array([
        [int(onset * sfreq), 0, event_id[trial_type]]
        for onset, trial_type in zip(df_valid['onset'], df_valid[trial_type_col])
    ], dtype=int)

    return events, event_id


def find_events_tsv(
    bids_root: Path,
    subject: str,
    session: str,
    task: str,
    run: Optional[str] = None,
) -> Path:
    """Find the events.tsv for a BIDS run, trying both with and without run number.

    Parameters
    ----------
    bids_root : Path
        BIDS dataset root directory.
    subject : str
        Subject ID (without 'sub-' prefix).
    session : str
        Session ID (without 'ses-' prefix).
    task : str
        Task name (without 'task-' prefix).
    run : str, optional
        Run number (without 'run-' prefix).

    Returns
    -------
    Path to existing events.tsv, or empty Path if not found.
    """
    eeg_dir = bids_root / f"sub-{subject}" / f"ses-{session}" / "eeg"

    # Try with run number first
    if run:
        candidate = eeg_dir / f"sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv"
        if candidate.exists():
            return candidate

    # Fall back to without run number
    candidate = eeg_dir / f"sub-{subject}_ses-{session}_task-{task}_events.tsv"
    if candidate.exists():
        return candidate

    return Path()


def get_event_mapping_for_run(
    bids_root: Path,
    subject: str,
    session: str,
    task: str,
    run: str
) -> Dict[str, str]:
    """Get event mapping for a specific BIDS run.

    Parameters
    ----------
    bids_root : Path
        BIDS dataset root directory
    subject : str
        Subject ID (without 'sub-' prefix)
    session : str
        Session ID (without 'ses-' prefix)
    task : str
        Task name (without 'task-' prefix)
    run : str
        Run ID (without 'run-' prefix)

    Returns
    -------
    dict
        Event mapping for this run
    """
    # Build path to events.tsv
    events_file = (
        bids_root / f"sub-{subject}" / f"ses-{session}" / "eeg" /
        f"sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv"
    )

    return load_event_mapping_from_bids(events_file)
