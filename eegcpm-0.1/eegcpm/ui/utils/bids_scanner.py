"""BIDS directory scanning utilities for Streamlit UI."""

from pathlib import Path
from typing import List, Dict, Set, Optional
import re


def scan_subjects(bids_root: Path) -> List[str]:
    """
    Scan BIDS directory for subject IDs.

    Parameters
    ----------
    bids_root : Path
        BIDS dataset root directory

    Returns
    -------
    list of str
        Subject IDs (without 'sub-' prefix), sorted
    """
    if not bids_root.exists():
        return []

    subjects = []
    for sub_dir in sorted(bids_root.glob('sub-*')):
        if sub_dir.is_dir():
            # Remove 'sub-' prefix
            subject_id = sub_dir.name[4:]
            subjects.append(subject_id)

    return subjects


def scan_sessions(bids_root: Path, subject_id: str) -> List[str]:
    """
    Scan subject directory for session IDs.

    Parameters
    ----------
    bids_root : Path
        BIDS dataset root directory
    subject_id : str
        Subject identifier (without 'sub-' prefix)

    Returns
    -------
    list of str
        Session IDs (without 'ses-' prefix), sorted
    """
    subject_dir = bids_root / f'sub-{subject_id}'
    if not subject_dir.exists():
        return []

    sessions = []
    for ses_dir in sorted(subject_dir.glob('ses-*')):
        if ses_dir.is_dir():
            # Remove 'ses-' prefix
            session_id = ses_dir.name[4:]
            sessions.append(session_id)

    # Sessionless BIDS layout: subject_dir/eeg/ exists directly (no ses-* subdirs)
    # Return a single empty-string session so downstream code can proceed; this
    # matches the behaviour of find_eeg_run_files() in eegcpm.data.bids_utils
    if not sessions and (subject_dir / 'eeg').is_dir():
        sessions = [""]

    return sessions


def scan_tasks(bids_root: Path, subject_id: Optional[str] = None) -> List[str]:
    """
    Scan for task names from EEG filenames.

    Parameters
    ----------
    bids_root : Path
        BIDS dataset root directory
    subject_id : str, optional
        Limit search to specific subject

    Returns
    -------
    list of str
        Task names (without 'task-' prefix), sorted and unique
    """
    if not bids_root.exists():
        return []

    tasks = set()

    # Pattern to match task name in BIDS filename
    # e.g., sub-001_ses-01_task-rest_eeg.fif -> 'rest'
    task_pattern = re.compile(r'task-([a-zA-Z0-9]+)')

    # Search pattern
    if subject_id:
        search_pattern = f'sub-{subject_id}/**/eeg/*_eeg.*'
    else:
        search_pattern = 'sub-*/**/eeg/*_eeg.*'

    for eeg_file in bids_root.glob(search_pattern):
        match = task_pattern.search(eeg_file.name)
        if match:
            tasks.add(match.group(1))

    return sorted(tasks)


def scan_pipelines(derivatives_root: Path) -> List[str]:
    """
    Scan derivatives directory for pipeline names.

    Parameters
    ----------
    derivatives_root : Path
        Derivatives directory containing pipeline-* folders

    Returns
    -------
    list of str
        Pipeline names (without 'pipeline-' prefix), sorted
    """
    if not derivatives_root.exists():
        return []

    pipelines = []
    for pipeline_dir in sorted(derivatives_root.glob('pipeline-*')):
        if pipeline_dir.is_dir():
            # Remove 'pipeline-' prefix
            pipeline_name = pipeline_dir.name[9:]
            pipelines.append(pipeline_name)

    return pipelines


def get_available_runs(
    bids_root: Path,
    subject_id: str,
    session: str,
    task: str
) -> List[str]:
    """
    Get available runs for a subject/session/task.

    Parameters
    ----------
    bids_root : Path
        BIDS dataset root directory
    subject_id : str
        Subject identifier (without 'sub-' prefix)
    session : str
        Session identifier (without 'ses-' prefix)
    task : str
        Task name (without 'task-' prefix)

    Returns
    -------
    list of str
        Run identifiers, sorted
    """
    eeg_dir = bids_root / f'sub-{subject_id}' / f'ses-{session}' / 'eeg'
    if not eeg_dir.exists():
        return []

    runs = set()
    run_pattern = re.compile(r'run-([0-9]+)')

    # Find all EEG files matching this task
    for eeg_file in eeg_dir.glob(f'*_task-{task}*.fif'):
        match = run_pattern.search(eeg_file.name)
        if match:
            runs.add(match.group(1))

    return sorted(runs)


def get_processed_subjects(
    derivatives_root: Path,
    pipeline: str
) -> List[str]:
    """
    Get subjects that have been processed by a pipeline.

    Parameters
    ----------
    derivatives_root : Path
        Derivatives directory
    pipeline : str
        Pipeline name (without 'pipeline-' prefix)

    Returns
    -------
    list of str
        Subject IDs that have been processed, sorted
    """
    pipeline_dir = derivatives_root / f'pipeline-{pipeline}'
    if not pipeline_dir.exists():
        return []

    subjects = []
    for subject_dir in sorted(pipeline_dir.glob('*')):
        if subject_dir.is_dir() and not subject_dir.name.startswith('.'):
            subjects.append(subject_dir.name)

    return subjects


def get_bids_info(bids_root: Path) -> Dict:
    """
    Get comprehensive BIDS dataset information.

    Parameters
    ----------
    bids_root : Path
        BIDS dataset root directory

    Returns
    -------
    dict
        Dictionary containing dataset statistics
    """
    if not bids_root.exists():
        return {
            'n_subjects': 0,
            'n_sessions': 0,
            'tasks': [],
            'subjects': []
        }

    subjects = scan_subjects(bids_root)
    all_tasks = scan_tasks(bids_root)

    # Count total sessions
    n_sessions = 0
    for subject_id in subjects:
        n_sessions += len(scan_sessions(bids_root, subject_id))

    return {
        'n_subjects': len(subjects),
        'n_sessions': n_sessions,
        'tasks': all_tasks,
        'subjects': subjects
    }


def get_subject_task_run_summary(bids_root: Path) -> List[Dict]:
    """
    Get detailed summary of subjects, tasks, and run counts.

    Parameters
    ----------
    bids_root : Path
        BIDS dataset root directory

    Returns
    -------
    list of dict
        Each dict has: subject, session, task, n_runs, runs (list)
    """
    if not bids_root.exists():
        return []

    results = []
    task_pattern = re.compile(r'task-([a-zA-Z0-9]+)')
    run_pattern = re.compile(r'run-([0-9]+)')

    subjects = scan_subjects(bids_root)

    for subject_id in subjects:
        sessions = scan_sessions(bids_root, subject_id)

        for session_id in sessions:
            eeg_dir = bids_root / f'sub-{subject_id}' / f'ses-{session_id}' / 'eeg'
            if not eeg_dir.exists():
                continue

            # Group files by task
            task_runs: Dict[str, Set[str]] = {}

            for eeg_file in eeg_dir.glob('*_eeg.fif'):
                task_match = task_pattern.search(eeg_file.name)
                if task_match:
                    task_name = task_match.group(1)
                    run_match = run_pattern.search(eeg_file.name)
                    run_id = run_match.group(1) if run_match else '1'

                    if task_name not in task_runs:
                        task_runs[task_name] = set()
                    task_runs[task_name].add(run_id)

            # Add to results
            for task_name, runs in sorted(task_runs.items()):
                results.append({
                    'subject': subject_id,
                    'session': session_id,
                    'task': task_name,
                    'n_runs': len(runs),
                    'runs': sorted(runs)
                })

    return results


def get_subject_task_matrix(bids_root: Path) -> Dict:
    """
    Get a matrix summary: subjects as rows, tasks as columns with run counts.

    Parameters
    ----------
    bids_root : Path
        BIDS dataset root directory

    Returns
    -------
    dict with keys:
        - 'subjects': list of subject IDs
        - 'tasks': list of task names
        - 'matrix': dict of {subject: {task: n_runs}}
    """
    summary = get_subject_task_run_summary(bids_root)

    subjects = sorted(set(item['subject'] for item in summary))
    tasks = sorted(set(item['task'] for item in summary))

    # Build matrix
    matrix = {subj: {task: 0 for task in tasks} for subj in subjects}

    for item in summary:
        matrix[item['subject']][item['task']] = item['n_runs']

    return {
        'subjects': subjects,
        'tasks': tasks,
        'matrix': matrix
    }
