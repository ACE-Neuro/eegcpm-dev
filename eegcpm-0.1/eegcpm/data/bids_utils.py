"""BIDS utilities for file discovery and naming conventions.

This module handles both legacy and BIDS-compliant naming:
- Legacy: sub-ID_ses-01_task-saiit2afcblock1_eeg.fif
- BIDS:   sub-ID_ses-01_task-saiit_run-01_eeg.fif
"""

import mne
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import re

# Supported EEG formats: (extension, file_type_name)
supported_formats = [
    ('fif', 'FIF'),           # MNE native format
    ('vhdr', 'BrainVision'),  # BrainVision
    ('edf', 'EDF'),           # European Data Format
    ('bdf', 'BDF'),           # BioSemi
    ('set', 'EEGLAB'),        # EEGLAB
    ('cnt', 'NeuroScan'),     # NeuroScan
    ('cdt', 'CURRY'),         # CURRY
]


@dataclass
class BIDSFile:
    """BIDS file information."""
    path: Path
    subject: str
    session: str
    task: str
    run: Optional[str] = None
    suffix: str = "eeg"
    extension: str = ".fif"

    @property
    def is_legacy_naming(self) -> bool:
        """Check if this uses legacy naming (no run, task includes block/run info)."""
        return self.run is None and any(
            substr in self.task.lower()
            for substr in ['block', 'run', '2afc']
        )

    def to_bids_compliant(self) -> 'BIDSFile':
        """Convert legacy naming to BIDS-compliant naming."""
        if not self.is_legacy_naming:
            return self

        # Extract task and run from legacy task name
        task, run = parse_legacy_task_name(self.task)

        return BIDSFile(
            path=self.path,
            subject=self.subject,
            session=self.session,
            task=task,
            run=run,
            suffix=self.suffix,
            extension=self.extension
        )

    def get_bids_filename(self) -> str:
        """Get BIDS-compliant filename."""
        parts = [f"sub-{self.subject}"]

        if self.session:
            parts.append(f"ses-{self.session}")

        parts.append(f"task-{self.task}")

        if self.run:
            parts.append(f"run-{self.run}")

        parts.append(self.suffix)

        return "_".join(parts) + self.extension


def read_raw_eeg(input_file: Path):
    """Read a raw EEG file using the appropriate MNE reader based on extension."""
    _, _, after = str(input_file).rpartition(".")
    extension = after
    print("file extension:\t", extension, flush=True)
    match extension:
        case "vhdr":
            raw = mne.io.read_raw_brainvision(
                input_file, preload=True, verbose=False)
            misc_channels = {
                'photosensor': 'stim',
                'optical': 'stim',
                'ecg': 'ecg',
                'resp': 'misc'
            }
            channels_to_set = {ch: t for ch,
                               t in misc_channels.items() if ch in raw.ch_names}
            if channels_to_set:
                raw.set_channel_types(channels_to_set)
        case "fif":
            raw = mne.io.read_raw_fif(input_file, preload=True, verbose=False)
        case "edf":
            raw = mne.io.read_raw_edf(input_file, preload=True, verbose=False)
        case "bdf":
            raw = mne.io.read_raw_bdf(input_file, preload=True, verbose=False)
        case "set":
            raw = mne.io.read_raw_eeglab(
                input_file, preload=True, verbose=False)
        case "cnt":
            raw = mne.io.read_raw_cnt(input_file, preload=True, verbose=False)
        case "cdt":
            raw = mne.io.read_raw_curry(
                input_file, preload=True, verbose=False)
        case _:
            raise ValueError(f"Unsupported EEG file extension: .{extension}")
    return raw


def find_eeg_run_files(
    subject_dir: Path,
    subject_id: str,
    task: str,
    formats: list = supported_formats,
) -> Optional[dict]:
    """
    Find EEG run files for a given subject/task, auto-detecting BIDS sessions.

    Checks whether subject_dir contains ses-* subdirectories. If so, loops
    over all of them; otherwise looks directly in subject_dir/eeg/.

    Args:
        subject_dir: Path to the subject directory (e.g. project/sub-001/).
        subject_id:  Subject ID (without 'sub-' prefix).
        task:        Task name (without 'task-' prefix).
        formats:     List of (ext, file_type) tuples to search, in priority order.

    Returns:
        {"files": List[Path], "file_type": str} on the first format match,
        or None if nothing is found.
    """
    session_dirs = sorted(d for d in subject_dir.glob("ses-*") if d.is_dir())
    if session_dirs:
        for ext, file_type in formats:
            all_files = []
            for ses_dir in session_dirs:
                eeg_dir = ses_dir / "eeg"
                if not eeg_dir.exists():
                    continue
                ses_id = ses_dir.name  # e.g. "ses-01"
                run_files = sorted(eeg_dir.glob(
                    f"sub-{subject_id}_ses-{ses_id}_task-{task}_eeg.{ext}"))
                if not run_files:
                    # Try "_run-*" naming format
                    run_files = sorted(eeg_dir.glob(
                        f"sub-{subject_id}_{ses_id}_task-{task}_run-*_eeg.{ext}"))
                if run_files:
                    all_files.extend(run_files)
                else:
                    single_file = eeg_dir / \
                        f"sub-{subject_id}_{ses_id}_task-{task}_eeg.{ext}"
                    if single_file.exists():
                        all_files.append(single_file)
            if all_files:
                return {"files": all_files, "file_type": file_type}
    else:
        eeg_dir = subject_dir / "eeg"
        if eeg_dir.exists():
            for ext, file_type in formats:
                run_files = sorted(eeg_dir.glob(
                    f"sub-{subject_id}_task-{task}_eeg.{ext}"))
                if not run_files:
                    # Try "_run-*" naming format
                    run_files = sorted(eeg_dir.glob(
                        f"sub-{subject_id}_task-{task}_run-*_eeg.{ext}"))
                if run_files:
                    return {"files": run_files, "file_type": file_type}

                single_file = eeg_dir / \
                    f"sub-{subject_id}_task-{task}_eeg.{ext}"
                if single_file.exists():
                    return {"files": [single_file], "file_type": file_type}

    return None


def parse_legacy_task_name(task_name: str) -> Tuple[str, str]:
    """
    Parse legacy task name to extract task and run.

    Examples:
        saiit2afcblock1 -> (saiit, 01)
        saiit2afcblock2 -> (saiit, 02)
        surrsuppblockblock1 -> (surrsupp, 01)
        rest -> (rest, 01)

    Parameters
    ----------
    task_name : str
        Legacy task name

    Returns
    -------
    tuple
        (task, run) where run is zero-padded
    """
    # Pattern: task name + optional suffix + block/run number
    patterns = [
        # saiit2afcblock1 -> saiit, 1
        (r'^(saiit)2afcblock(\d+)$', r'\1', r'\2'),
        # surrsuppblockblock1 -> surrsupp, 1
        (r'^(surrsupp)blockblock(\d+)$', r'\1', r'\2'),
        # Generic: taskblock1 -> task, 1
        (r'^(.+)block(\d+)$', r'\1', r'\2'),
        # Generic: taskrun1 -> task, 1
        (r'^(.+)run(\d+)$', r'\1', r'\2'),
    ]

    for pattern, task_group, run_group in patterns:
        match = re.match(pattern, task_name, re.IGNORECASE)
        if match:
            task = match.group(1)
            run = match.group(2).zfill(2)  # Zero-pad to 2 digits
            return task, run

    # No pattern matched - assume single run
    return task_name, "01"


def parse_bids_filename(filename: str) -> Optional[BIDSFile]:
    """
    Parse BIDS filename to extract components.

    Handles both:
    - BIDS: sub-ID_ses-01_task-saiit_run-01_eeg.fif
    - Legacy: sub-ID_ses-01_task-saiit2afcblock1_eeg.fif

    Parameters
    ----------
    filename : str
        BIDS filename (with or without path)

    Returns
    -------
    BIDSFile or None
        Parsed file info, or None if not BIDS format
    """
    filename = Path(filename).name

    # Remove extension
    name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
    ext = '.' + ext if ext else ''

    # Parse entities using regex
    entities = {}

    # Required: subject
    match = re.search(r'sub-([a-zA-Z0-9]+)', name)
    if not match:
        return None
    entities['subject'] = match.group(1)

    # Optional: session
    match = re.search(r'ses-([a-zA-Z0-9]+)', name)
    entities['session'] = match.group(1) if match else None

    # Required: task
    match = re.search(r'task-([a-zA-Z0-9]+)', name)
    if not match:
        return None
    entities['task'] = match.group(1)

    # Optional: run
    match = re.search(r'run-([a-zA-Z0-9]+)', name)
    entities['run'] = match.group(1) if match else None

    # Suffix (last entity before extension)
    parts = name.split('_')
    suffix = parts[-1] if parts else 'eeg'

    return BIDSFile(
        path=Path(filename),
        subject=entities['subject'],
        session=entities['session'],
        task=entities['task'],
        run=entities['run'],
        suffix=suffix,
        extension=ext
    )


def find_bids_files(
    bids_root: Path,
    subject: Optional[str] = None,
    session: Optional[str] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
    suffix: str = "eeg",
    extension: str = ".fif"
) -> List[BIDSFile]:
    """
    Find BIDS files matching criteria.

    Handles both legacy and BIDS-compliant naming.

    Parameters
    ----------
    bids_root : Path
        BIDS dataset root
    subject : str, optional
        Subject ID (without 'sub-' prefix)
    session : str, optional
        Session ID (without 'ses-' prefix)
    task : str, optional
        Task name (without 'task-' prefix)
    run : str, optional
        Run number (without 'run-' prefix)
    suffix : str
        File suffix (default: 'eeg')
    extension : str
        File extension (default: '.fif')

    Returns
    -------
    List[BIDSFile]
        List of matching BIDS files
    """
    bids_root = Path(bids_root)
    results = []

    # Build search pattern
    pattern_parts = []

    if subject:
        pattern_parts.append(f"sub-{subject}")
        subject_pattern = f"sub-{subject}"
    else:
        subject_pattern = "sub-*"

    # Search in subject directories
    for subject_dir in sorted(bids_root.glob(subject_pattern)):
        print("searching in subject directories with pattern:\t",
              sorted(bids_root.glob(subject_pattern)))
        if not subject_dir.is_dir():
            continue

        # Look for session directories or directly in subject
        search_dirs = []

        if session:
            ses_dir = subject_dir / f"ses-{session}"
            print("session dir path:\t", ses_dir)
            if ses_dir.exists():
                search_dirs.append(ses_dir)
        else:
            # Look for all session directories
            ses_dirs = list(subject_dir.glob("ses-*"))
            if ses_dirs:
                search_dirs.extend(ses_dirs)
            else:
                # No session structure
                search_dirs.append(subject_dir)

        for search_dir in search_dirs:
            # Look in eeg subdirectory
            eeg_dir = search_dir / "eeg"
            if not eeg_dir.exists():
                continue

            # Find all EEG files
            for file_path in eeg_dir.glob(f"*{suffix}{extension}"):
                bids_file = parse_bids_filename(file_path.name)
                if not bids_file:
                    continue

                bids_file.path = file_path

                # Apply filters
                if task and bids_file.task != task:
                    # Check if legacy naming matches
                    bids_compliant = bids_file.to_bids_compliant()
                    if bids_compliant.task != task:
                        continue

                if run and bids_file.run != run:
                    continue

                results.append(bids_file)

    return sorted(results, key=lambda x: (x.subject, x.session or '', x.task, x.run or ''))


def find_subject_runs(
    bids_root: Path,
    subject: str,
    task: str,
    session: Optional[str] = None
) -> List[BIDSFile]:
    """
    Find all runs for a specific subject and task.

    Handles legacy naming by converting to BIDS-compliant.

    Parameters
    ----------
    bids_root : Path
        BIDS dataset root
    subject : str
        Subject ID
    task : str
        Task name (BIDS-compliant, e.g., 'saiit' not 'saiit2afcblock1')
    session : str, optional
        Session ID

    Returns
    -------
    List[BIDSFile]
        List of runs, sorted by run number
    """

    print("find subject runs's root path:\t", bids_root)
    # Find all files for this subject/task (legacy)
    # files = find_bids_files(
    #     bids_root,
    #     subject=subject,
    #     session=session,
    #     task=None  # Don't filter by task yet
    # )

    # Find all files for this subject/task
    subject_dir = bids_root / f"sub-{subject}"
    # print(f"print subject path", path)
    results = find_eeg_run_files(
        subject_dir,
        subject_id=subject,
        task=task,
    )

    files = results.get("files", None)
    print("find subject runs's find_bids_files:\t", results.get("files"))

    # Convert to BIDS-compliant and filter
    runs = []
    for f in files:
        # Parse BIDS entities from filename stem (e.g. sub-ID_ses-01_task-rest_run-01_eeg)
        entities = {}
        for part in f.stem.split("_"):
            if "-" in part:
                key, val = part.split("-", 1)
                entities[key] = val
            else:
                entities["suffix"] = part  # "eeg"

        bids_file = BIDSFile(
            path=f,
            subject=entities.get("sub", ""),
            session=entities.get("ses", ""),
            task=entities.get("task", ""),
            run=entities.get("run"),
            suffix=entities.get("suffix", "eeg"),
            extension=f.suffix,
        ).to_bids_compliant()

        print(f"bids_file: ", bids_file)

        if bids_file.task == task:
            runs.append(bids_file)

    # Sort by run number
    return sorted(runs, key=lambda x: x.run or '01')


def get_task_runs_summary(bids_root: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Get summary of all tasks and runs in dataset.

    Returns
    -------
    dict
        Nested dict: {subject: {task: [runs]}}
        For single-run tasks (no run entity), run list will be ['01']
    """
    summary = {}

    all_files = find_bids_files(bids_root)

    for f in all_files:
        bids_file = f.to_bids_compliant()

        if bids_file.subject not in summary:
            summary[bids_file.subject] = {}

        if bids_file.task not in summary[bids_file.subject]:
            summary[bids_file.subject][bids_file.task] = []

        # Default to 01 for single-run tasks
        run = bids_file.run if bids_file.run else '01'
        summary[bids_file.subject][bids_file.task].append(run)

    # Sort and deduplicate runs
    for subject in summary:
        for task in summary[subject]:
            summary[subject][task] = sorted(set(summary[subject][task]))

    return summary
