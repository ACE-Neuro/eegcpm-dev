"""QC metrics JSON I/O for CLI/UI synchronization.

This module provides functions to save and load QC metrics as standalone JSON files,
enabling workflow state-independent sharing between CLI and UI.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from .quality_assessment import extract_metrics_from_qc_result

logger = logging.getLogger(__name__)


def save_qc_metrics_json(data: Dict[str, Any], path: Path) -> None:
    """
    Save QC metrics as JSON file.

    Parameters
    ----------
    data : dict
        QC metrics dictionary (see JSON schema in quality_assessment.py)
    path : Path
        Output JSON file path

    Returns
    -------
    None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_qc_metrics_json(path: Path) -> Dict[str, Any]:
    """
    Load QC metrics from JSON file.

    Parameters
    ----------
    path : Path
        JSON file path

    Returns
    -------
    dict
        QC metrics dictionary

    Raises
    ------
    FileNotFoundError
        If JSON file does not exist
    json.JSONDecodeError
        If JSON is malformed
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"QC metrics JSON not found: {path}")

    with open(path, 'r') as f:
        return json.load(f)


def load_qc_metrics_from_directory(
    derivatives_path: Path,
    subject_id: str,
    session: str = "01",
    task: Optional[str] = None,
    pipeline: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Scan derivatives directory for QC metrics JSON files.

    Parameters
    ----------
    derivatives_path : Path
        Derivatives root directory
    subject_id : str
        Subject ID
    session : str
        Session ID (default: "01")
    task : str, optional
        Filter by task name
    pipeline : str, optional
        Filter by pipeline name (not used in current structure)

    Returns
    -------
    List[dict]
        List of QC metrics dictionaries, one per run

    Notes
    -----
    Scans structure:
        derivatives/{subject}/ses-{session}/task-{task}/run-{run}/*_qc_metrics.json
    """
    derivatives_path = Path(derivatives_path)
    results = []

    # Build path to subject derivatives
    subject_dir = derivatives_path / subject_id / f"ses-{session}"

    if not subject_dir.exists():
        return results

    # Find all task directories
    task_dirs = []
    if task:
        task_dir = subject_dir / f"task-{task}"
        if task_dir.exists():
            task_dirs = [task_dir]
    else:
        task_dirs = sorted([d for d in subject_dir.iterdir()
                           if d.is_dir() and d.name.startswith('task-')])

    # Scan each task directory for run QC JSON files
    for task_dir in task_dirs:
        run_dirs = sorted([d for d in task_dir.iterdir()
                          if d.is_dir() and d.name.startswith('run-')])

        for run_dir in run_dirs:
            # Look for QC metrics JSON
            json_files = list(run_dir.glob('*_qc_metrics.json'))

            for json_file in json_files:
                try:
                    metrics = load_qc_metrics_json(json_file)
                except (OSError, ValueError) as exc:
                    # A malformed metrics file is skipped, but recorded: a
                    # silent skip hides a broken run from the QC table (P1-F11).
                    logger.warning(
                        "skipping unreadable QC metrics file %s: %s", json_file, exc)
                    continue
                # Add file path for reference
                metrics['_json_path'] = str(json_file)
                results.append(metrics)

    return results
