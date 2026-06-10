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
