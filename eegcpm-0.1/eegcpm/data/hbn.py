"""
Healthy Brain Network (HBN) dataset utilities.

Dataset: https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import logging

from eegcpm.core.models import Project, Subject, Session, Run

logger = logging.getLogger(__name__)


def scan_hbn_directory(
    root_path: Path,
    include_derivatives: bool = False,
) -> Project:
    """
    Scan HBN dataset directory and create Project.

    Expected HBN structure varies. This handles common patterns.

    Args:
        root_path: Root of HBN data directory
        include_derivatives: Include derivative files

    Returns:
        Project with discovered subjects
    """
    root_path = Path(root_path)

    project = Project(
        name="HBN_Dataset",
        description="Healthy Brain Network EEG data",
        root_path=root_path,
    )

    # Find subject directories
    for sub_dir in sorted(root_path.glob("sub-*")):
        if not sub_dir.is_dir():
            continue

        subject = Subject(id=sub_dir.name)

        # Check for sessions or direct eeg folder
        ses_dirs = list(sub_dir.glob("ses-*"))
        if not ses_dirs:
            # No session level - treat subject dir as session
            ses_dirs = [sub_dir]

        for ses_dir in sorted(ses_dirs):
            session_id = ses_dir.name if "ses-" in ses_dir.name else "ses-01"
            session = Session(id=session_id, subject_id=subject.id)

            # Find EEG directory
            eeg_dir = ses_dir / "eeg"
            if not eeg_dir.exists():
                eeg_dir = ses_dir  # Files might be directly in session dir

            # Find EEG files
            eeg_patterns = ["*.set", "*.fdt", "*.edf", "*.bdf", "*.fif"]
            for pattern in eeg_patterns:
                for eeg_file in sorted(eeg_dir.glob(pattern)):
                    # Skip .fdt files (EEGLAB data, loaded with .set)
                    if eeg_file.suffix == ".fdt":
                        continue

                    # Parse task and run from filename
                    task_name = _extract_task(eeg_file.stem)
                    run_id = _extract_run(eeg_file.stem)

                    run = Run(
                        id=run_id,
                        task_name=task_name,
                        eeg_file=eeg_file,
                    )

                    # Check for events file
                    events_file = eeg_file.with_suffix(".tsv")
                    if not events_file.exists():
                        events_file = eeg_dir / f"{eeg_file.stem}_events.tsv"
                    if events_file.exists():
                        run.events_file = events_file

                    session.runs.append(run)

            if session.runs:
                subject.sessions.append(session)

        if subject.sessions:
            project.subjects.append(subject)

    return project


def _extract_task(filename: str) -> str:
    """Extract task name from filename."""
    parts = filename.split("_")
    for part in parts:
        if part.startswith("task-"):
            return part.replace("task-", "")
    return "unknown"


def _extract_run(filename: str) -> str:
    """Extract run ID from filename."""
    parts = filename.split("_")
    for part in parts:
        if part.startswith("run-"):
            return part
    return "run-01"


def load_hbn_phenotype(
    phenotype_file: Path,
) -> Dict[str, Dict[str, Any]]:
    """
    Load HBN phenotype/behavioral data.

    Args:
        phenotype_file: Path to phenotype CSV/TSV file

    Returns:
        Dict mapping subject_id to behavioral scores
    """
    import pandas as pd

    if phenotype_file.suffix == ".csv":
        df = pd.read_csv(phenotype_file)
    else:
        df = pd.read_csv(phenotype_file, sep="\t")

    # Assume first column is subject ID
    id_col = df.columns[0]

    phenotype = {}
    for _, row in df.iterrows():
        sub_id = row[id_col]
        if not str(sub_id).startswith("sub-"):
            sub_id = f"sub-{sub_id}"
        phenotype[sub_id] = row.to_dict()

    return phenotype


def add_behavioral_scores(
    project: Project,
    phenotype: Dict[str, Dict[str, Any]],
    score_columns: Optional[List[str]] = None,
) -> None:
    """
    Add behavioral scores from phenotype data to project subjects.

    Args:
        project: Project to update
        phenotype: Phenotype data from load_hbn_phenotype
        score_columns: Specific columns to include (None = all numeric)

    Notes:
        Values that are not numeric (``ValueError``/``TypeError`` from
        ``float``) are skipped; the skips are logged (debug per value, warning
        summary) instead of being swallowed by a bare ``pass`` (B7-A1).
    """
    skipped: Dict[str, List[str]] = {}

    for subject in project.subjects:
        if subject.id not in phenotype:
            continue
        pheno = phenotype[subject.id]
        if score_columns:
            columns = [col for col in score_columns if col in pheno]
            missing = [col for col in score_columns if col not in pheno]
            if missing:
                # A requested score column the phenotype table does not have is
                # a drop too — recorded, not silently ignored (B7-A1).
                logger.debug(
                    "HBN phenotype: %s has no column(s) %s; requested but absent",
                    subject.id, ", ".join(missing),
                )
        else:
            # Add all numeric values
            columns = list(pheno.keys())

        for col in columns:
            try:
                subject.behavioral_scores[col] = float(pheno[col])
            except (ValueError, TypeError) as exc:
                skipped.setdefault(subject.id, []).append(col)
                logger.debug(
                    "HBN phenotype: %s.%s is not numeric (%s); skipped",
                    subject.id, col, exc,
                )

    if skipped:
        n_values = sum(len(cols) for cols in skipped.values())
        detail = "; ".join(
            f"{sub}: {', '.join(cols)}" for sub, cols in sorted(skipped.items())
        )
        logger.warning(
            "HBN phenotype: skipped %d non-numeric score(s) for %d subject(s): %s",
            n_values, len(skipped), detail,
        )
