"""Centralized path management for EEGCPM.

This module provides a unified path management system for the stage-first
architecture, ensuring consistent directory structures across all components.

Stage-First Architecture:
- derivatives/preprocessing/{pipeline}/{subject}/ses-{session}/task-{task}/run-{run}/
- derivatives/epochs/{preprocessing}/{task}/{subject}/ses-{session}/
- derivatives/source/{preprocessing}/{task}/variant-{method}-{template}/{subject}/ses-{session}/
- derivatives/features/{preprocessing}/{task}/{source-variant}/{feature-type}/
- derivatives/prediction/{model-name}/
"""

from pathlib import Path
from typing import List, Optional
import json


class EEGCPMPaths:
    """Centralized path management for EEGCPM.

    This class provides a single source of truth for all path construction
    in the EEGCPM framework. It implements the stage-first architecture
    where outputs are organized by processing stage rather than pipeline.

    Parameters
    ----------
    project_root : Path
        Root directory of the project containing bids/, derivatives/, etc.
    eegcpm_root : Path, optional
        EEGCPM workspace directory. If not provided, defaults to
        {project_root}/eegcpm.

    Attributes
    ----------
    project_root : Path
        Project root directory
    bids_root : Path
        BIDS data directory ({project_root}/bids)
    eegcpm_root : Path
        EEGCPM workspace directory
    derivatives_root : Path
        Derivatives output directory ({project_root}/derivatives)

    Examples
    --------
    >>> paths = EEGCPMPaths(Path("/data/study"))
    >>> preproc_dir = paths.get_preprocessing_dir(
    ...     pipeline="standard",
    ...     subject="sub-001",
    ...     session="01",
    ...     task="rest",
    ...     run="01"
    ... )
    >>> print(preproc_dir)
    /data/study/derivatives/preprocessing/standard/sub-001/ses-01/task-rest/run-01
    """

    def __init__(
        self,
        project_root: Path,
        eegcpm_root: Optional[Path] = None
    ):
        self.project_root = Path(project_root)
        self.bids_root = self.project_root / "bids"
        self.eegcpm_root = Path(eegcpm_root) if eegcpm_root else self.project_root / "eegcpm"
        self.derivatives_root = self.project_root / "derivatives"

    # =========================================================================
    # State Database
    # =========================================================================

    def get_state_db(self) -> Path:
        """Get workflow state database path.

        Returns
        -------
        Path
            Path to state.db file at {eegcpm_root}/.eegcpm/state.db
        """
        return self.eegcpm_root / ".eegcpm" / "state.db"

    def get_configs_dir(self, stage: Optional[str] = None) -> Path:
        """Get configuration directory.

        Parameters
        ----------
        stage : str, optional
            Processing stage (preprocessing, source, features).
            If None, returns the root configs directory.

        Returns
        -------
        Path
            Path to configs directory
        """
        base = self.eegcpm_root / "configs"
        if stage:
            return base / stage
        return base

    # =========================================================================
    # Preprocessing Stage
    # =========================================================================

    def get_preprocessing_dir(
        self,
        pipeline: str,
        subject: str,
        session: str,
        task: str,
        run: str
    ) -> Path:
        """Get preprocessing output directory.

        Parameters
        ----------
        pipeline : str
            Preprocessing variant name (e.g., "standard", "minimal", "robust")
        subject : str
            Subject ID (with or without 'sub-' prefix)
        session : str
            Session ID (without 'ses-' prefix)
        task : str
            Task name (without 'task-' prefix)
        run : str
            Run ID (without 'run-' prefix)

        Returns
        -------
        Path
            Full path to preprocessing output directory:
            derivatives/preprocessing/{pipeline}/{subject}/ses-{session}/task-{task}/run-{run}/
        """
        # Normalize subject ID - ensure sub- prefix
        if not subject.startswith("sub-"):
            subject = f"sub-{subject}"

        return (
            self.derivatives_root
            / "preprocessing"
            / pipeline
            / subject
            / f"ses-{session}"
            / f"task-{task}"
            / f"run-{run}"
        )

    def get_preprocessing_root(self, pipeline: str) -> Path:
        """Get root preprocessing directory for a pipeline variant.

        Parameters
        ----------
        pipeline : str
            Preprocessing variant name

        Returns
        -------
        Path
            Path to derivatives/preprocessing/{pipeline}/
        """
        return self.derivatives_root / "preprocessing" / pipeline

    # =========================================================================
    # Epochs Stage
    # =========================================================================

    def get_epochs_dir(
        self,
        preprocessing: str,
        task: str,
        subject: str,
        session: str
    ) -> Path:
        """Get epochs output directory.

        Parameters
        ----------
        preprocessing : str
            Name of preprocessing variant this depends on
        task : str
            Task name (without 'task-' prefix)
        subject : str
            Subject ID (with or without 'sub-' prefix)
        session : str
            Session ID (without 'ses-' prefix)

        Returns
        -------
        Path
            Full path to epochs output directory:
            derivatives/epochs/{preprocessing}/{task}/{subject}/ses-{session}/
        """
        if not subject.startswith("sub-"):
            subject = f"sub-{subject}"

        return (
            self.derivatives_root
            / "epochs"
            / preprocessing
            / task
            / subject
            / f"ses-{session}"
        )

    def get_epochs_root(self, preprocessing: str, task: str) -> Path:
        """Get root epochs directory for a preprocessing/task combination.

        Parameters
        ----------
        preprocessing : str
            Preprocessing variant name
        task : str
            Task name

        Returns
        -------
        Path
            Path to derivatives/epochs/{preprocessing}/{task}/
        """
        return self.derivatives_root / "epochs" / preprocessing / task

    # =========================================================================
    # Source Stage
    # =========================================================================

    def get_source_dir(
        self,
        preprocessing: str,
        task: str,
        variant: str,
        subject: str,
        session: str
    ) -> Path:
        """Get source reconstruction output directory.

        Parameters
        ----------
        preprocessing : str
            Name of preprocessing variant
        task : str
            Task name (without 'task-' prefix)
        variant : str
            Source variant name (e.g., "dSPM-CONN32", "sLORETA-AAL")
        subject : str
            Subject ID (with or without 'sub-' prefix)
        session : str
            Session ID (without 'ses-' prefix)

        Returns
        -------
        Path
            Full path to source output directory:
            derivatives/source/{preprocessing}/{task}/variant-{variant}/{subject}/ses-{session}/
        """
        if not subject.startswith("sub-"):
            subject = f"sub-{subject}"

        # Normalize variant name
        if not variant.startswith("variant-"):
            variant = f"variant-{variant}"

        return (
            self.derivatives_root
            / "source"
            / preprocessing
            / task
            / variant
            / subject
            / f"ses-{session}"
        )

    def get_source_root(
        self,
        preprocessing: str,
        task: str,
        variant: Optional[str] = None
    ) -> Path:
        """Get root source directory.

        Parameters
        ----------
        preprocessing : str
            Preprocessing variant name
        task : str
            Task name
        variant : str, optional
            Source variant name. If None, returns task-level directory.

        Returns
        -------
        Path
            Path to source root directory
        """
        base = self.derivatives_root / "source" / preprocessing / task
        if variant:
            if not variant.startswith("variant-"):
                variant = f"variant-{variant}"
            return base / variant
        return base

    def list_source_variants(
        self,
        preprocessing: str,
        task: str
    ) -> list[str]:
        """List available source reconstruction variants.

        Parameters
        ----------
        preprocessing : str
            Preprocessing variant name
        task : str
            Task name

        Returns
        -------
        list[str]
            List of variant names (without 'variant-' prefix), sorted alphabetically
        """
        base_path = self.derivatives_root / "source" / preprocessing / task

        if not base_path.exists():
            return []

        # Extract variant names (remove "variant-" prefix)
        variants = []
        for d in base_path.iterdir():
            if d.is_dir() and d.name.startswith('variant-'):
                variants.append(d.name.replace('variant-', '', 1))

        return sorted(variants)

    # =========================================================================
    # Features Stage
    # =========================================================================

    def get_features_dir(
        self,
        preprocessing: str,
        task: str,
        source_variant: str,
        feature_type: str
    ) -> Path:
        """Get features output directory.

        Parameters
        ----------
        preprocessing : str
            Preprocessing variant name
        task : str
            Task name
        source_variant : str
            Source variant name
        feature_type : str
            Feature type (e.g., "bandpower", "connectivity", "erp")

        Returns
        -------
        Path
            Full path to features directory:
            derivatives/features/{preprocessing}/{task}/{source-variant}/{feature-type}/
        """
        if not source_variant.startswith("variant-"):
            source_variant = f"variant-{source_variant}"

        return (
            self.derivatives_root
            / "features"
            / preprocessing
            / task
            / source_variant
            / feature_type
        )

    def get_features_root(
        self,
        preprocessing: str,
        task: str,
        source_variant: Optional[str] = None
    ) -> Path:
        """Get root features directory.

        Parameters
        ----------
        preprocessing : str
            Preprocessing variant name
        task : str
            Task name
        source_variant : str, optional
            Source variant name. If None, returns task-level directory.

        Returns
        -------
        Path
            Path to features root directory
        """
        base = self.derivatives_root / "features" / preprocessing / task
        if source_variant:
            if not source_variant.startswith("variant-"):
                source_variant = f"variant-{source_variant}"
            return base / source_variant
        return base

    # =========================================================================
    # Prediction Stage
    # =========================================================================

    def get_prediction_dir(self, model_name: str) -> Path:
        """Get prediction model output directory.

        Parameters
        ----------
        model_name : str
            Model/analysis name

        Returns
        -------
        Path
            Path to derivatives/prediction/{model-name}/
        """
        return self.derivatives_root / "prediction" / model_name

    # =========================================================================
    # BIDS Input Paths
    # =========================================================================

    def get_bids_eeg_dir(
        self,
        subject: str,
        session: Optional[str] = None
    ) -> Path:
        """Get BIDS EEG directory for a subject.

        Parameters
        ----------
        subject : str
            Subject ID (with or without 'sub-' prefix)
        session : str, optional
            Session ID (without 'ses-' prefix)

        Returns
        -------
        Path
            Path to BIDS EEG directory
        """
        if not subject.startswith("sub-"):
            subject = f"sub-{subject}"

        base = self.bids_root / subject
        if session:
            base = base / f"ses-{session}"
        return base / "eeg"

    def get_bids_eeg_file(
        self,
        subject: str,
        session: str,
        task: str,
        run: Optional[str] = None,
        suffix: str = "eeg",
        extension: str = ".fif"
    ) -> Path:
        """Get path to a BIDS EEG file.

        Parameters
        ----------
        subject : str
            Subject ID (with or without 'sub-' prefix)
        session : str
            Session ID (without 'ses-' prefix)
        task : str
            Task name (without 'task-' prefix)
        run : str, optional
            Run ID (without 'run-' prefix)
        suffix : str
            BIDS suffix (default: "eeg")
        extension : str
            File extension (default: ".fif")

        Returns
        -------
        Path
            Full path to BIDS EEG file
        """
        if not subject.startswith("sub-"):
            subject = f"sub-{subject}"

        # Build filename parts
        parts = [subject, f"ses-{session}", f"task-{task}"]
        if run:
            parts.append(f"run-{run}")
        parts.append(suffix)

        filename = "_".join(parts) + extension

        return self.get_bids_eeg_dir(subject, session) / filename

    # =========================================================================
    # File Naming Conventions
    # =========================================================================

    @staticmethod
    def build_filename(
        subject: str,
        session: str,
        task: str,
        run: Optional[str] = None,
        desc: Optional[str] = None,
        suffix: str = "eeg",
        extension: str = ".fif"
    ) -> str:
        """Build a BIDS-compliant filename.

        Parameters
        ----------
        subject : str
            Subject ID (with or without 'sub-' prefix)
        session : str
            Session ID (without 'ses-' prefix)
        task : str
            Task name (without 'task-' prefix)
        run : str, optional
            Run ID (without 'run-' prefix)
        desc : str, optional
            Description label (e.g., "preproc")
        suffix : str
            BIDS suffix (default: "eeg")
        extension : str
            File extension (default: ".fif")

        Returns
        -------
        str
            BIDS-compliant filename
        """
        if not subject.startswith("sub-"):
            subject = f"sub-{subject}"

        parts = [subject, f"ses-{session}", f"task-{task}"]
        if run:
            parts.append(f"run-{run}")
        if desc:
            parts.append(f"desc-{desc}")
        parts.append(suffix)

        return "_".join(parts) + extension

    # =========================================================================
    # Legacy Compatibility
    # =========================================================================

    def get_legacy_pipeline_dir(self, pipeline: str) -> Path:
        """Get legacy pipeline directory path.

        For backward compatibility with old structure:
        eegcpm/pipelines/{pipeline}/

        Parameters
        ----------
        pipeline : str
            Pipeline name

        Returns
        -------
        Path
            Path to legacy pipeline directory
        """
        return self.eegcpm_root / "pipelines" / pipeline

    def get_legacy_derivatives_dir(self, pipeline: str) -> Path:
        """Get legacy derivatives directory path.

        For backward compatibility with old structure:
        derivatives/pipeline-{pipeline}/

        Parameters
        ----------
        pipeline : str
            Pipeline name

        Returns
        -------
        Path
            Path to legacy derivatives directory
        """
        return self.derivatives_root / f"pipeline-{pipeline}"

    # =========================================================================
    # Directory Creation and Validation
    # =========================================================================

    def ensure_dir(self, path: Path) -> Path:
        """Ensure a directory exists, creating if necessary.

        Parameters
        ----------
        path : Path
            Directory path

        Returns
        -------
        Path
            The same path (for chaining)
        """
        path.mkdir(parents=True, exist_ok=True)
        return path

    def validate_bids_structure(self) -> bool:
        """Validate that the expected BIDS structure exists.

        Returns
        -------
        bool
            True if BIDS root exists and contains subject directories
        """
        if not self.bids_root.exists():
            return False

        # Check for at least one subject directory
        subjects = list(self.bids_root.glob("sub-*"))
        return len(subjects) > 0

    def validate_project_structure(self) -> dict:
        """Validate the complete project structure.

        Returns
        -------
        dict
            Validation results with:
            - valid: bool
            - bids_exists: bool
            - derivatives_exists: bool
            - eegcpm_exists: bool
            - errors: list of error messages
        """
        result = {
            "valid": True,
            "bids_exists": self.bids_root.exists(),
            "derivatives_exists": self.derivatives_root.exists(),
            "eegcpm_exists": self.eegcpm_root.exists(),
            "errors": []
        }

        if not result["bids_exists"]:
            result["valid"] = False
            result["errors"].append(f"BIDS root not found: {self.bids_root}")

        return result


def find_preprocessed_raw(run_dir: Path) -> Optional[Path]:
    """Find the preprocessed raw FIF inside one per-run output directory.

    The preprocessed FIF lives in the run directory with a stem of
    ``{subject}{_ses-X}{_run-Y}`` (see
    :meth:`PreprocessingPipeline._output_stem`), so consumers must glob
    for it instead of hard-coding a subject-only filename.

    Parameters
    ----------
    run_dir : Path
        Per-run output directory (e.g.
        ``derivatives/preprocessing/{pipeline}/sub-001/ses-01/task-rest/run-1``).

    Returns
    -------
    Path or None
        The single ``*_preprocessed_raw.fif`` in ``run_dir``, or ``None``
        when the directory does not exist or holds no such file.

    Raises
    ------
    RuntimeError
        If more than one ``*_preprocessed_raw.fif`` is present — picking
        one arbitrarily would silently consume a stale output.
    """
    run_dir = Path(run_dir)
    matches = sorted(run_dir.glob('*_preprocessed_raw.fif'))
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple preprocessed raw FIF files in {run_dir}: "
            f"{[p.name for p in matches]}"
        )
    return matches[0]


def clear_run_outputs(run_dir: Path) -> None:
    """Delete stale preprocessing outputs before a run is reprocessed.

    A forced re-run that fails must not leave the previous run's FIF
    alongside the new ``*_preprocessing_FAILED.json`` sidecar. Deletes
    (non-recursively, missing files ignored):

    - ``*_preprocessed_raw.fif``
    - ``*_ica.fif``
    - ``*_preprocessing*.json`` (success / FAILED / EXCLUDED sidecars)
    - ``*_qc_metrics.json``
    - ``*_preprocessed_qc.html`` / ``*_preprocessed_qc.json``

    Parameters
    ----------
    run_dir : Path
        Per-run output directory.
    """
    run_dir = Path(run_dir)
    patterns = (
        '*_preprocessed_raw.fif',
        '*_ica.fif',
        '*_preprocessing*.json',
        '*_qc_metrics.json',
        '*_preprocessed_qc.html',
        '*_preprocessed_qc.json',
    )
    for pattern in patterns:
        for stale in run_dir.glob(pattern):
            if stale.is_file():
                stale.unlink()


# Convenience function for creating paths from project root
def create_paths(
    project_root: Path,
    eegcpm_root: Optional[Path] = None
) -> EEGCPMPaths:
    """Create an EEGCPMPaths instance.

    Parameters
    ----------
    project_root : Path
        Project root directory
    eegcpm_root : Path, optional
        EEGCPM workspace directory

    Returns
    -------
    EEGCPMPaths
        Path manager instance
    """
    return EEGCPMPaths(project_root, eegcpm_root)


def fail_run_outputs(
    run_output: Path,
    subject_id: str,
    session: str,
    run_num: str,
    output_files: List[Path],
    error: Exception,
) -> None:
    """Turn a succeeded-then-crashed run's outputs into a FAILED run.

    Deletes the run's saved outputs and rewrites its
    ``{stem}_preprocessing.json`` success sidecar as
    ``{stem}_preprocessing_FAILED.json`` with the error added, so the
    filesystem never claims a success the persisted status denies.
    Shared by the CLI and ``RunProcessor`` (audit condition N4/F2).
    """
    for out_file in output_files or []:
        try:
            Path(out_file).unlink()
        except FileNotFoundError:
            pass

    stem = f"{subject_id}_ses-{session}_run-{run_num}"
    success_sidecar = Path(run_output) / f"{stem}_preprocessing.json"
    failed_sidecar = Path(run_output) / f"{stem}_preprocessing_FAILED.json"

    payload = {}
    if success_sidecar.exists():
        try:
            with open(success_sidecar) as f:
                payload = json.load(f)
        except (OSError, ValueError):
            payload = {}
        success_sidecar.unlink()

    payload['success'] = False
    payload['error'] = str(error)
    payload['errors'] = list(payload.get('errors') or []) + [str(error)]
    failed_sidecar.parent.mkdir(parents=True, exist_ok=True)
    with open(failed_sidecar, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
