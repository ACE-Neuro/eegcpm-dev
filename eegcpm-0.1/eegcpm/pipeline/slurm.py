"""HPC/SLURM tooling for EEGCPM — the single script-generation module.

This module is the ONE place that builds sbatch scripts and submits them.
Both entry points go through it:

- the CLI (``eegcpm hpc manifest|prune|submit|run-unit|status|finalize``,
  :mod:`eegcpm.cli.hpc`) and
- the Streamlit UI pages (``ui/pages/2_batch_preprocessing.py`` for the
  per-subject preprocessing arrays and ``ui/pages/12_prediction.py`` for the
  prediction job).

No page may inline its own ``#SBATCH`` header (a unit test enforces this).

Script flavours
---------------
Three flavours are generated here; all share the ``_sbatch_header_lines``
helper so there is a single header implementation:

1. :func:`generate_array_script` — the batch workflow below: one array task per
   work unit, each running ``eegcpm hpc run-unit``.
2. :func:`generate_slurm_script` — the UI batch page's per-subject array
   (subjects looped over their tasks inside the task; kept byte-compatible for
   the existing download).
3. :func:`generate_command_slurm_script` — one arbitrary command (used by the
   prediction page); no array.

Legacy single-job API
---------------------
:class:`SlurmJob` and :class:`SlurmSubmitter` (submit one monolithic script per
job, ``squeue`` status, wait, cancel) are **superseded** by the per-work-unit
job-array workflow below: they have no consumers left in the codebase and are
retained only as a thin wrapper (``submit``/``submit_array``/status helpers).
``SlurmSubmitter.submit_subject_jobs`` used to call the removed
``python -m eegcpm.cli run`` surface; it now builds a manifest per subject and
submits through the current CLI (``eegcpm hpc run-unit``), so it produces the
same per-unit commands as the CLI.

Model
-----
A *work unit* is one ``(subject, session, task, run)`` BIDS recording. Every
unit runs the per-unit stages ``preprocess -> epochs -> source``. The
task-level ``features`` stage is an explicit finalize step because it writes
ONE parquet per task combining every subject — running it per unit would race.
The ``features`` stage is therefore deliberately excluded from the per-unit
stage list.

Idempotency and state isolation
-------------------------------
- ``run-unit`` skips any stage whose artifact already exists, so re-submitting
  the same array is retry-safe.
- ``preprocess`` is invoked with ``--state-db <state_db_dir>/<unit.key>.db`` so
  concurrent array tasks never share (and never lock) a SQLite file.
- ``preprocess`` is invoked with ``--sessions``/``--runs`` so a unit processes
  exactly its own recording. ``epochs``/``source`` are invoked with
  ``--sessions`` because their artifacts are per (subject, session, task) and
  combine the runs of that session — that is the artifact's own semantics.

Caveat (multi-run sessions)
---------------------------
The epochs/source artifacts are per (subject, session, task). When a task has
more than one run in a session, several units of that session would produce the
same artifact: submit those stages for one unit per session/task (the artifact
check makes the other units skip once the first one produced it), or serialize
the units. The 25SusAttn dataset has exactly one recording per (subject, task),
so this does not apply there.

Script safety
-------------
Generated sbatch scripts use absolute paths, ``set -eo pipefail`` and the same
interpreter that ran the submitter (``sys.executable``) unless a ``venv`` is
given. ``sbatch`` is never called from this module — the CLI submits (and only
with ``--dry-run`` in tests).
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from eegcpm.core.config import SlurmConfig
from eegcpm.core.models import Project
from eegcpm.core.paths import EEGCPMPaths
from eegcpm.data.bids_utils import (
    find_eeg_run_files,
    parse_run_identity,
    supported_formats,
)

logger = logging.getLogger(__name__)


#: Stages executed once per work unit (one array task each).
PER_UNIT_STAGES: Tuple[str, ...] = ("preprocess", "epochs", "source")
#: Task-level stages (finalize only — one invocation covers every subject).
TASK_LEVEL_STAGES: Tuple[str, ...] = ("features",)
ALL_STAGES: Tuple[str, ...] = PER_UNIT_STAGES + TASK_LEVEL_STAGES

#: Manifest TSV header, in column order.
MANIFEST_COLUMNS: Tuple[str, ...] = ("subject", "session", "task", "run")


# ============================================================ single-job API
# Superseded by the work-unit/job-array workflow below; kept for callers that
# need to drive one job script per submission (``sbatch``/``squeue``/``scancel``).


class SlurmJob:
    """Represents a SLURM job."""

    def __init__(
        self,
        job_id: str,
        name: str,
        script_path: Path,
        status: str = "PENDING",
    ):
        self.job_id = job_id
        self.name = name
        self.script_path = script_path
        self.status = status
        self.submitted_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

    def __repr__(self) -> str:
        return f"SlurmJob({self.job_id}, {self.name}, {self.status})"


class SlurmSubmitter:
    """
    Submit and manage SLURM jobs for EEGCPM pipelines.

    Superseded (for batch runs) by the work-unit job-array workflow further
    down this module and by ``eegcpm hpc submit``; this class is the small
    single-/array-job wrapper around ``sbatch``, ``squeue`` and ``scancel``.
    """

    def __init__(
        self,
        config: SlurmConfig,
        log_dir: Optional[Path] = None,
    ):
        """
        Initialize SLURM submitter.

        Args:
            config: SLURM configuration
            log_dir: Directory for job logs and generated scripts
        """
        self.config = config
        self.log_dir = log_dir or Path("./slurm_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, SlurmJob] = {}

    def generate_script(
        self,
        job_name: str,
        python_command: str,
        subject_id: Optional[str] = None,
        additional_modules: Optional[List[str]] = None,
    ) -> str:
        """
        Generate SLURM batch script.

        Args:
            job_name: Job name
            python_command: Python command to execute
            subject_id: Optional subject ID for job arrays
            additional_modules: Additional modules to load

        Returns:
            SLURM script content
        """
        modules = self.config.modules.copy()
        if additional_modules:
            modules.extend(additional_modules)

        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={self.config.partition}
#SBATCH --time={self.config.time}
#SBATCH --mem={self.config.mem}
#SBATCH --cpus-per-task={self.config.cpus_per_task}
#SBATCH --output={self.log_dir}/{job_name}_%j.out
#SBATCH --error={self.log_dir}/{job_name}_%j.err
"""

        if self.config.gpus:
            script += f"#SBATCH --gpus={self.config.gpus}\n"

        script += "\n# Load modules\n"
        for module in modules:
            script += f"module load {module}\n"

        script += f"""
# Activate conda environment if needed
# conda activate eegcpm

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Run command
{python_command}

echo "End time: $(date)"
"""

        return script

    def _sbatch_submit(self, script_path: Path, job_name: str) -> SlurmJob:
        """``sbatch`` one already-written script and track the resulting job."""
        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit job: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error("sbatch command not found - is SLURM installed?")
            raise

        # Parse job ID from output like "Submitted batch job 12345"
        job_id = result.stdout.strip().split()[-1]
        job = SlurmJob(job_id=job_id, name=job_name, script_path=script_path)
        job.submitted_at = datetime.now()
        self.jobs[job_id] = job
        logger.info(f"Submitted job {job_id}: {job_name}")
        return job

    def submit(
        self,
        job_name: str,
        python_command: str,
        dry_run: bool = False,
    ) -> Optional[SlurmJob]:
        """
        Submit a job to SLURM.

        Args:
            job_name: Job name
            python_command: Command to execute
            dry_run: If True, only generate script without submitting

        Returns:
            SlurmJob object or None if dry_run
        """
        script_content = self.generate_script(job_name, python_command)

        # Write script to log/temp dir
        script_path = self.log_dir / f"{job_name}.sh"
        with open(script_path, "w") as f:
            f.write(script_content)

        if dry_run:
            logger.info(f"Dry run - script saved to {script_path}")
            return None

        return self._sbatch_submit(script_path, job_name)

    def submit_array(
        self,
        script_path: Union[str, Path],
        dry_run: bool = False,
        job_name: Optional[str] = None,
    ) -> Optional[SlurmJob]:
        """
        Submit an already-generated script (e.g. a job array) to SLURM.

        This is the single submission path shared by the batch scripts of
        :func:`generate_array_script` / :func:`generate_slurm_script` and by
        :meth:`submit_subject_jobs`.

        Args:
            script_path: Path of the script written by one of the generators
            dry_run: If True, do not call ``sbatch`` (the script stays on disk)
            job_name: Job name for the tracked :class:`SlurmJob`
                (default: the script's stem)

        Returns:
            SlurmJob object or None if dry_run
        """
        script_path = Path(script_path)
        if not script_path.is_file():
            raise FileNotFoundError(f"sbatch script not found: {script_path}")
        job_name = job_name or script_path.stem

        if dry_run:
            logger.info(f"Dry run - script ready at {script_path}")
            return None

        return self._sbatch_submit(script_path, job_name)

    def submit_subject_jobs(
        self,
        project: Project,
        config_path: Path,
        subject_ids: Optional[List[str]] = None,
        *,
        pipeline: str = "standard",
        stages: Sequence[str] = ("preprocess",),
        bids_root: Optional[Path] = None,
        submission_dir: Optional[Path] = None,
        dry_run: bool = False,
    ) -> List[SlurmJob]:
        """
        Submit one job array per subject through the current CLI.

        Each subject's BIDS recordings (from :func:`build_manifest`) become one
        array task running ``python -m eegcpm.cli.main hpc run-unit``, i.e. the
        same per-unit commands as ``eegcpm hpc submit``. Chunk manifests and
        sbatch scripts are written under
        ``<project>/hpc/submissions/<timestamp>/``.

        Args:
            project: Project to process
            config_path: Preprocessing config YAML (for the ``preprocess`` stage)
            subject_ids: Specific subjects (None = all project subjects)
            pipeline: Preprocessing pipeline name baked into the commands
            stages: Per-unit stages to run (default: preprocess only)
            bids_root: BIDS root (default: ``<project.root_path>/bids``)
            submission_dir: Where to write chunk TSVs/scripts (default:
                ``<project.root_path>/hpc/submissions/<timestamp>``)
            dry_run: If True, write the scripts but do not call ``sbatch``

        Returns:
            List of submitted jobs (empty with ``dry_run=True``)
        """
        root = Path(project.root_path)
        bids_root = Path(bids_root) if bids_root is not None else root / "bids"
        config_path = Path(config_path)

        if subject_ids:
            wanted = {_strip_prefix(sid, "sub-") for sid in subject_ids}
            selected = [subject.id for subject in project.subjects
                        if _strip_prefix(subject.id, "sub-") in wanted]
        else:
            selected = [subject.id for subject in project.subjects]

        if submission_dir is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            submission_dir = root / "hpc" / "submissions" / stamp
        submission_dir = Path(submission_dir)
        submission_dir.mkdir(parents=True, exist_ok=True)

        jobs: List[SlurmJob] = []
        for subject_id in selected:
            units = build_manifest(bids_root, subjects=[subject_id])
            if not units:
                logger.warning(
                    "submit_subject_jobs: no BIDS input files for subject %s "
                    "under %s", subject_id, bids_root,
                )
                continue

            job_name = f"eegcpm_{_strip_prefix(subject_id, 'sub-')}"
            chunk_path = submission_dir / f"{job_name}.tsv"
            write_manifest(units, chunk_path)

            script = generate_array_script(
                job_name=job_name,
                partition=self.config.partition,
                units_chunk_path=chunk_path,
                n_units=len(units),
                concurrency=len(units),
                cpus=self.config.cpus_per_task,
                mem=self.config.mem,
                time=self.config.time,
                log_dir=self.log_dir,
                venv=self.config.venv,
                modules=self.config.modules,
                exports=self.config.exports,
                project=root,
                stages=stages,
                pipeline=pipeline,
                config_paths={"preprocessing": config_path},
            )
            script_path = submission_dir / f"{job_name}.sbatch"
            script_path.write_text(script, encoding="utf-8")

            job = self.submit_array(script_path, dry_run=dry_run, job_name=job_name)
            if job:
                jobs.append(job)

        return jobs

    def get_job_status(self, job_id: str) -> str:
        """
        Get status of a submitted job.

        Args:
            job_id: SLURM job ID

        Returns:
            Job status string
        """
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
            )
            status = result.stdout.strip()
            return status if status else "COMPLETED"
        except Exception:
            return "UNKNOWN"

    def update_all_statuses(self) -> None:
        """Update status of all tracked jobs."""
        for job_id, job in self.jobs.items():
            job.status = self.get_job_status(job_id)

    def wait_for_completion(
        self,
        job_ids: Optional[List[str]] = None,
        poll_interval: int = 60,
    ) -> Dict[str, str]:
        """
        Wait for jobs to complete.

        Args:
            job_ids: Jobs to wait for (None = all tracked)
            poll_interval: Seconds between status checks

        Returns:
            Dict mapping job_id to final status
        """
        if job_ids is None:
            job_ids = list(self.jobs.keys())

        final_statuses = {}

        while job_ids:
            for job_id in job_ids.copy():
                status = self.get_job_status(job_id)

                if status in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", ""]:
                    final_statuses[job_id] = status or "COMPLETED"
                    job_ids.remove(job_id)
                    if job_id in self.jobs:
                        self.jobs[job_id].status = status
                        self.jobs[job_id].completed_at = datetime.now()

            if job_ids:
                logger.info(f"Waiting for {len(job_ids)} jobs...")
                time.sleep(poll_interval)

        return final_statuses

    def cancel(self, job_id: str) -> bool:
        """Cancel a job."""
        try:
            subprocess.run(["scancel", job_id], check=True)
            return True
        except Exception:
            return False

    def cancel_all(self) -> None:
        """Cancel all tracked jobs."""
        for job_id in self.jobs:
            self.cancel(job_id)


# ================================================================= work units


def _strip_prefix(value: str, prefix: str) -> str:
    """Drop a BIDS entity prefix (``"sub-001"`` → ``"001"``) when present."""
    value = str(value).strip()
    return value[len(prefix):] if value.startswith(prefix) else value


def _normalize_set(values: Optional[Sequence[str]], prefix: str) -> Optional[set]:
    """Normalize a filter list (with or without entity prefixes) to a set."""
    if values is None:
        return None
    return {_strip_prefix(value, prefix) for value in values}


@dataclass(frozen=True, order=True)
class WorkUnit:
    """One ``(subject, session, task, run)`` recording to process.

    Entity values are stored **without** their BIDS prefixes (``"001"``,
    ``"01"``, ``"cptvis"``, ``"1"``); :attr:`key` rebuilds the prefixed form.
    """

    subject: str
    session: str
    task: str
    run: str

    def __post_init__(self) -> None:
        # Accept prefixed values from hand-written manifests.
        for field_name, prefix in (
            ("subject", "sub-"), ("session", "ses-"),
            ("task", "task-"), ("run", "run-"),
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"WorkUnit.{field_name} must be a non-empty string, got {value!r}")
            stripped = _strip_prefix(value, prefix)
            if stripped != value:
                object.__setattr__(self, field_name, stripped)

    @property
    def key(self) -> str:
        """Stable, filename-safe identity, e.g. ``sub-001_ses-01_task-cptvis_run-1``."""
        return (
            f"sub-{self.subject}_ses-{self.session}"
            f"_task-{self.task}_run-{self.run}"
        )

    def to_row(self) -> List[str]:
        """Manifest columns for this unit (no prefixes)."""
        return [self.subject, self.session, self.task, self.run]

    @classmethod
    def from_row(cls, row: Sequence[str]) -> "WorkUnit":
        """Build a unit from a manifest row; fails on malformed input."""
        fields = [str(field).strip() for field in row]
        if len(fields) != 4:
            raise ValueError(
                f"manifest row must have {len(MANIFEST_COLUMNS)} columns "
                f"({'/'.join(MANIFEST_COLUMNS)}), got {fields!r}"
            )
        return cls(*fields)


def _parse_entities(stem: str) -> Dict[str, str]:
    """Parse ``key-value`` entities from a BIDS file stem."""
    entities: Dict[str, str] = {}
    for part in stem.split("_"):
        if "-" in part:
            key, _, value = part.partition("-")
            entities[key] = value
    return entities


def _discover_tasks(subject_dir: Path) -> List[str]:
    """Task names present in a subject directory (supported file formats only)."""
    extensions = {ext for ext, _ in supported_formats}
    tasks = set()
    for path in sorted(subject_dir.rglob("*_eeg.*")):
        if path.suffix.lstrip(".") not in extensions:
            continue
        task = _parse_entities(path.stem).get("task")
        if task:
            tasks.add(task)
    return sorted(tasks)


def _indexed_input_files(
    subject_dir: Path, subject_id: str, task: str
) -> List[Tuple[int, Path]]:
    """Input files for ``(subject, task)`` with their 1-based run-list index.

    The index is the same one ``eegcpm.cli.preprocess`` uses as the fallback
    run number for files without a ``run-`` entity, so manifest units and
    preprocessing output paths always agree.
    """
    response = find_eeg_run_files(subject_dir, subject_id, task)
    if not response:
        return []
    return list(enumerate(response.get("files") or [], 1))


def build_manifest(
    bids_root: Union[str, Path],
    tasks: Optional[Sequence[str]] = None,
    subjects: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[str]] = None,
) -> List[WorkUnit]:
    """Scan a BIDS tree and return the sorted, deduped work units.

    Parameters
    ----------
    bids_root : path
        BIDS dataset root (contains ``sub-*`` directories).
    tasks : sequence of str, optional
        Restrict to these task names (default: every task found).
    subjects : sequence of str, optional
        Restrict to these subject IDs (with or without ``sub-``).
    sessions : sequence of str, optional
        Restrict to these session IDs (with or without ``ses-``).

    Returns
    -------
    list of WorkUnit
        Sorted by ``(subject, session, task, run)``, duplicates removed.

    Raises
    ------
    FileNotFoundError
        If ``bids_root`` is not a directory (an empty manifest from a wrong
        path would silently mark every unit as to-do).
    """
    bids_root = Path(bids_root)
    if not bids_root.is_dir():
        raise FileNotFoundError(f"BIDS root not found: {bids_root}")

    wanted_subjects = _normalize_set(subjects, "sub-")
    wanted_tasks = _normalize_set(tasks, "task-")
    wanted_sessions = _normalize_set(sessions, "ses-")

    units: List[WorkUnit] = []
    for subject_dir in sorted(p for p in bids_root.glob("sub-*") if p.is_dir()):
        subject_id = subject_dir.name[len("sub-"):]
        if wanted_subjects is not None and subject_id not in wanted_subjects:
            continue

        task_names = (
            sorted(wanted_tasks) if wanted_tasks is not None
            else _discover_tasks(subject_dir)
        )
        for task in task_names:
            for index, path in _indexed_input_files(subject_dir, subject_id, task):
                session, run = parse_run_identity(path.name, fallback_index=index)
                if wanted_sessions is not None and session not in wanted_sessions:
                    continue
                units.append(WorkUnit(subject_id, session, task, run))

    return sorted(set(units))


def write_manifest(units: Sequence[WorkUnit], path: Union[str, Path]) -> Path:
    """Write units to a TSV with header ``subject session task run``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(MANIFEST_COLUMNS)]
    lines.extend("\t".join(unit.to_row()) for unit in units)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_manifest(path: Union[str, Path]) -> List[WorkUnit]:
    """Read a manifest TSV written by :func:`write_manifest`.

    Blank lines and ``#`` comment lines are ignored. A missing header, a row
    with the wrong number of columns or a duplicated unit is a hard error:
    silently re-processing (or dropping) work units is worse than failing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")

    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"manifest is empty: {path}")

    header = tuple(field.strip() for field in lines[0].split("\t"))
    if header != MANIFEST_COLUMNS:
        raise ValueError(
            f"manifest header must be {'/'.join(MANIFEST_COLUMNS)} (tab-separated), "
            f"got {header!r}: {path}"
        )

    units: List[WorkUnit] = []
    seen = set()
    for lineno, line in enumerate(lines[1:], start=2):
        fields = [field.strip() for field in line.split("\t")]
        if len(fields) != len(MANIFEST_COLUMNS):
            raise ValueError(
                f"{path}:{lineno}: expected {len(MANIFEST_COLUMNS)} tab-separated "
                f"columns ('{'/'.join(MANIFEST_COLUMNS)}'), got {fields!r}"
            )
        unit = WorkUnit.from_row(fields)
        if unit in seen:
            raise ValueError(f"{path}:{lineno}: duplicate work unit {unit.key}")
        seen.add(unit)
        units.append(unit)
    return units


def chunk_units(
    units: Sequence[WorkUnit], partitions: Sequence[str]
) -> List[Tuple[str, List[WorkUnit]]]:
    """Split units across partitions by deterministic round-robin.

    Unit ``i`` (0-based, in the given order) goes to partition
    ``i % len(partitions)``, so every partition receives a balanced mix and
    every unit appears exactly once. The same inputs always produce the same
    chunks.
    """
    partition_names = list(partitions)
    if not partition_names:
        raise ValueError("at least one partition is required")
    if len(set(partition_names)) != len(partition_names):
        raise ValueError(f"duplicate partition names: {partition_names}")

    chunks: List[List[WorkUnit]] = [[] for _ in partition_names]
    for index, unit in enumerate(units):
        chunks[index % len(partition_names)].append(unit)
    return list(zip(partition_names, chunks))


# ============================================================ artifact checks


def features_output_dir(project: Union[str, Path], pipeline: str) -> Path:
    """Directory of the task-level source-ERP feature tables.

    ``derivatives/features/<pipeline>/source_erp/`` — the layout written by the
    project's ``features`` invocations (one parquet per task/condition).
    """
    return Path(project) / "derivatives" / "features" / pipeline / "source_erp"


def features_parquet_path(
    project: Union[str, Path], task: str, pipeline: str
) -> Path:
    """Task-level features parquet for ``(pipeline, task)`` (finalize output)."""
    return features_output_dir(project, pipeline) / f"source_erp_{task}.parquet"


def _read_json_object(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON object, logging (not raising) on unreadable content.

    A missing success sidecar simply means "not complete" — but a present yet
    unreadable one is recorded at WARNING so the repeated reprocessing it
    triggers is traceable.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning("hpc: unreadable JSON artifact %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning(
            "hpc: JSON artifact %s is a %s, expected an object",
            path, type(payload).__name__,
        )
        return None
    return payload


def _preprocess_sidecar(project: Union[str, Path], unit: WorkUnit, pipeline: str) -> Path:
    run_dir = EEGCPMPaths(Path(project)).get_preprocessing_dir(
        pipeline=pipeline,
        subject=unit.subject,
        session=unit.session,
        task=unit.task,
        run=unit.run,
    )
    return run_dir / f"{unit.subject}_ses-{unit.session}_run-{unit.run}_preprocessing.json"


def unit_complete(
    project: Union[str, Path],
    unit: WorkUnit,
    stage: str,
    *,
    pipeline: str,
    source_variant: Optional[str] = None,
    expected_config_hash: Optional[str] = None,
) -> bool:
    """Whether the artifact of ``stage`` exists for ``unit``.

    Per-stage checks:

    - ``preprocess`` — the run's ``*_preprocessing.json`` sidecar exists,
      reports ``success is True`` and (when ``expected_config_hash`` is given)
      carries that exact ``config_hash``. Excluded/failed runs have no success
      sidecar and therefore never count as complete.
    - ``epochs`` — ``<subject>_epo.fif`` under
      ``derivatives/epochs/<pipeline>/<task>/sub-<s>/ses-<ses>/``.
    - ``source`` — ``<subject>_roi_tc.npz`` under
      ``derivatives/source/<pipeline>/<task>/variant-<variant>/sub-<s>/ses-<ses>/``
      (requires ``source_variant``).
    - ``features`` — the task-level parquet
      ``derivatives/features/<pipeline>/source_erp/source_erp_<task>.parquet``
      (used by finalize only; it does not depend on the unit).
    """
    project = Path(project)
    paths = EEGCPMPaths(project)

    if stage == "preprocess":
        sidecar = _preprocess_sidecar(project, unit, pipeline)
        payload = _read_json_object(sidecar) if sidecar.is_file() else None
        if payload is None or payload.get("success") is not True:
            return False
        if expected_config_hash is not None:
            return payload.get("config_hash") == expected_config_hash
        return True

    if stage == "epochs":
        session_dir = paths.get_epochs_dir(
            preprocessing=pipeline,
            task=unit.task,
            subject=unit.subject,
            session=unit.session,
        )
        return (session_dir / f"{unit.subject}_epo.fif").is_file()

    if stage == "source":
        if not source_variant:
            raise ValueError("unit_complete(stage='source') requires source_variant")
        session_dir = paths.get_source_dir(
            preprocessing=pipeline,
            task=unit.task,
            variant=source_variant,
            subject=unit.subject,
            session=unit.session,
        )
        return (session_dir / f"{unit.subject}_roi_tc.npz").is_file()

    if stage == "features":
        return features_parquet_path(project, unit.task, pipeline).is_file()

    raise ValueError(
        f"unknown stage {stage!r}; expected one of {', '.join(ALL_STAGES)}"
    )


def prune_units(
    project: Union[str, Path],
    units: Sequence[WorkUnit],
    stages: Sequence[str],
    *,
    pipeline: str,
    source_variant: Optional[str] = None,
    expected_config_hash: Optional[str] = None,
) -> List[WorkUnit]:
    """Drop units whose **every** requested per-unit stage is complete.

    Returns the still-pending units (in input order). ``features`` is
    task-level and therefore rejected here.
    """
    stage_list = list(stages)
    if not stage_list:
        raise ValueError("prune_units requires at least one stage")
    for stage in stage_list:
        if stage in TASK_LEVEL_STAGES:
            raise ValueError(
                f"{stage!r} is a task-level stage and cannot be used per unit; "
                "use finalize_task() instead"
            )
        if stage not in PER_UNIT_STAGES:
            raise ValueError(
                f"unknown stage {stage!r}; expected one of {', '.join(PER_UNIT_STAGES)}"
            )

    pending: List[WorkUnit] = []
    for unit in units:
        if all(
            unit_complete(
                project,
                unit,
                stage,
                pipeline=pipeline,
                source_variant=source_variant,
                expected_config_hash=expected_config_hash,
            )
            for stage in stage_list
        ):
            continue
        pending.append(unit)
    return pending


# ==================================================== records and state paths


def unit_record_path(logs_dir: Union[str, Path], unit: WorkUnit) -> Path:
    """Per-unit JSON record path: ``<logs_dir>/../units/<unit.key>.json``."""
    return Path(logs_dir).parent / "units" / f"{unit.key}.json"


def unit_state_db_path(state_db_dir: Union[str, Path], unit: WorkUnit) -> Path:
    """Per-unit state database path: ``<state_db_dir>/<unit.key>.db``."""
    return Path(state_db_dir) / f"{unit.key}.db"


# ======================================================= per-unit execution


def _resolve_python(
    venv: Optional[Union[str, Path]], python: Optional[str] = None
) -> str:
    if python:
        return python
    if venv:
        return str(Path(venv) / "bin" / "python")
    return sys.executable


def _config_path(
    config_paths: Mapping[str, Any], key: str, stage: str
) -> Path:
    value = config_paths.get(key) if config_paths else None
    if not value:
        raise ValueError(
            f"config_paths[{key!r}] is required for the {stage!r} stage"
        )
    return Path(value)


def _stage_command(
    unit: WorkUnit,
    stage: str,
    *,
    project: Path,
    pipeline: str,
    source_variant: Optional[str],
    config_paths: Mapping[str, Any],
    state_db_dir: Path,
    python: str,
) -> List[str]:
    """Argv for one stage of one unit (exact CLI flags)."""
    base = [python, "-m", "eegcpm.cli.main"]

    if stage == "preprocess":
        command = base + [
            "preprocess",
            "--config", str(_config_path(config_paths, "preprocessing", stage)),
            "--project", str(project),
            "--subject", unit.subject,
            "--task", unit.task,
            "--pipeline", pipeline,
            "--task-config", unit.task,
            "--state-db", str(unit_state_db_path(state_db_dir, unit)),
            "--sessions", unit.session,
            "--runs", unit.run,
        ]
        return command

    if stage == "epochs":
        tasks_dir = _config_path(config_paths, "tasks_dir", stage)
        return base + [
            "epochs",
            "--project", str(project),
            "--config", str(tasks_dir / f"{unit.task}.yaml"),
            "--preprocessing", pipeline,
            "--task", unit.task,
            "--subjects", unit.subject,
            "--sessions", unit.session,
        ]

    if stage == "source":
        if not source_variant:
            raise ValueError("the 'source' stage requires source_variant")
        return base + [
            "source-reconstruct",
            "--project", str(project),
            "--config", str(_config_path(config_paths, "source", stage)),
            "--preprocessing", pipeline,
            "--task", unit.task,
            "--subjects", unit.subject,
            "--task-config", unit.task,
            "--sessions", unit.session,
        ]

    raise ValueError(
        f"unknown per-unit stage {stage!r}; expected one of {', '.join(PER_UNIT_STAGES)}"
    )


def run_unit(
    unit: WorkUnit,
    *,
    project: Union[str, Path],
    stages: Sequence[str],
    pipeline: str,
    source_variant: Optional[str],
    config_paths: Mapping[str, Any],
    state_db_dir: Union[str, Path],
    logs_dir: Union[str, Path],
    dry_run: bool = False,
    python: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the requested per-unit stages for one work unit.

    Completed stages are skipped (idempotent retries); the remaining stages run
    in the order given (canonically preprocess → epochs → source) via
    ``subprocess``. The first failing stage stops the chain — downstream stages
    depend on its artifact — and sets ``exit_code`` to a non-zero value.

    The per-stage record (``status``/``seconds``/``returncode``/``commands``)
    is written to :func:`unit_record_path` and returned. With ``dry_run=True``
    the commands are returned but nothing is executed and nothing is written.
    """
    project = Path(project)
    state_db_dir = Path(state_db_dir)
    logs_dir = Path(logs_dir)
    python = _resolve_python(None, python)

    stage_list = list(stages)
    for stage in stage_list:
        if stage in TASK_LEVEL_STAGES:
            raise ValueError(
                f"{stage!r} is a task-level stage and cannot run per unit; "
                "use finalize_task() instead"
            )
        if stage not in PER_UNIT_STAGES:
            raise ValueError(
                f"unknown stage {stage!r}; expected one of {', '.join(PER_UNIT_STAGES)}"
            )

    record: Dict[str, Any] = {
        "unit": unit.key,
        "subject": unit.subject,
        "session": unit.session,
        "task": unit.task,
        "run": unit.run,
        "project": str(project),
        "pipeline": pipeline,
        "source_variant": source_variant,
        "dry_run": bool(dry_run),
        "stages": {},
        "exit_code": 0,
    }

    for stage in stage_list:
        if unit_complete(
            project,
            unit,
            stage,
            pipeline=pipeline,
            source_variant=source_variant,
        ):
            record["stages"][stage] = {
                "status": "skipped",
                "seconds": 0.0,
                "returncode": None,
                "commands": [],
            }
            continue

        command = _stage_command(
            unit,
            stage,
            project=project,
            pipeline=pipeline,
            source_variant=source_variant,
            config_paths=config_paths,
            state_db_dir=state_db_dir,
            python=python,
        )
        if dry_run:
            record["stages"][stage] = {
                "status": "dry-run",
                "seconds": 0.0,
                "returncode": None,
                "commands": [command],
            }
            continue

        start = time.monotonic()
        try:
            completed = subprocess.run(command, cwd=str(project), check=False)
            returncode = completed.returncode
            error = None
        except OSError as exc:
            # A missing interpreter / working directory is a per-unit failure,
            # recorded like any other (never a silent crash of the array task).
            returncode = 1
            error = f"could not execute {command[0]}: {exc}"
            logger.error("hpc run-unit %s: %s", unit.key, error)
        seconds = time.monotonic() - start
        failed = returncode != 0
        stage_record: Dict[str, Any] = {
            "status": "failed" if failed else "completed",
            "seconds": round(seconds, 3),
            "returncode": returncode,
            "commands": [command],
        }
        if error:
            stage_record["error"] = error
        record["stages"][stage] = stage_record
        if failed:
            record["exit_code"] = returncode or 1
            break

    if not dry_run:
        record_path = unit_record_path(logs_dir, unit)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


# =========================================================== script building


def _q(value: Any) -> str:
    """Shell-quote a value for the generated script."""
    return shlex.quote(str(value))


def _mail_user_line(email: Optional[str]) -> str:
    """The ``--mail-user`` line: real address, or a commented placeholder.

    Callers that do not emit mail lines at all simply do not call this.
    """
    if email:
        return f"#SBATCH --mail-user={email}"
    return "# #SBATCH --mail-user=your_email@eduhk.hk"


def _sbatch_header_lines(
    *,
    job_name: str,
    partition: str,
    cpus: Union[int, str],
    mem: str,
    time: str,
    output: str,
    error: str,
    nodes: Optional[int] = 1,
    ntasks_per_node: Optional[int] = 1,
    array: Optional[str] = None,
    email: Optional[str] = None,
    mail_type: Optional[str] = None,
) -> List[str]:
    """One ``#SBATCH`` header implementation for every generated script.

    Field order: job name, partition, node/task layout (when given), per-task
    resources (cpus, memory, wall time), array indices (when given), log paths,
    then the mail lines (only when ``mail_type`` is given; an empty ``email``
    emits the commented placeholder the UI scripts ship with).
    """
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
    ]
    if nodes is not None:
        lines.append(f"#SBATCH --nodes={nodes}")
    if ntasks_per_node is not None:
        lines.append(f"#SBATCH --ntasks-per-node={ntasks_per_node}")
    lines.append(f"#SBATCH --cpus-per-task={cpus}")
    lines.append(f"#SBATCH --mem={mem}")
    lines.append(f"#SBATCH --time={time}")
    if array is not None:
        lines.append(f"#SBATCH --array={array}")
    lines.append(f"#SBATCH --output={output}")
    lines.append(f"#SBATCH --error={error}")
    if mail_type is not None:
        lines.append(f"#SBATCH --mail-type={mail_type}")
        lines.append(_mail_user_line(email))
    return lines


def generate_array_script(
    *,
    job_name: str,
    partition: str,
    units_chunk_path: Union[str, Path],
    n_units: int,
    concurrency: int,
    cpus: int,
    mem: str,
    time: str,
    log_dir: Union[str, Path],
    venv: Optional[Union[str, Path]] = None,
    modules: Sequence[str] = (),
    exports: Optional[Mapping[str, Any]] = None,
    project: Union[str, Path],
    stages: Sequence[str],
    pipeline: str,
    source_variant: Optional[str] = None,
    config_paths: Optional[Mapping[str, Any]] = None,
    state_db_dir: Optional[Union[str, Path]] = None,
) -> str:
    """Render the sbatch script for one partition's chunk of units.

    The array task ``i`` processes ``units_chunk_path`` row ``i`` through
    ``eegcpm hpc run-unit``. The script uses the same interpreter that ran the
    submitter (``sys.executable``) unless ``venv`` is given, in which case the
    venv's ``bin/python`` is used and its ``activate`` is sourced.

    ``state_db_dir`` is the per-unit SQLite directory baked into the command
    (default ``<log_dir>/../state``, the CLI's own default).

    Returns the script text; writing it (and calling ``sbatch``) is the CLI's
    job.
    """
    unit_stages = list(stages)
    if not unit_stages:
        raise ValueError("generate_array_script requires at least one stage")
    unknown = [stage for stage in unit_stages if stage not in PER_UNIT_STAGES]
    if unknown:
        raise ValueError(
            f"unknown per-unit stage(s) {', '.join(unknown)}; expected one of "
            f"{', '.join(PER_UNIT_STAGES)} (task-level features run via finalize)"
        )
    if n_units <= 0:
        raise ValueError(f"n_units must be positive, got {n_units}")
    if concurrency <= 0:
        raise ValueError(f"concurrency must be positive, got {concurrency}")

    log_dir = Path(log_dir)
    python = _resolve_python(venv)
    state_db_dir = (
        Path(state_db_dir) if state_db_dir is not None else log_dir.parent / "state"
    )

    lines = _sbatch_header_lines(
        job_name=job_name,
        partition=partition,
        cpus=cpus,
        mem=mem,
        time=time,
        output=f"{log_dir}/{job_name}_%A_%a.out",
        error=f"{log_dir}/{job_name}_%A_%a.err",
        nodes=None,
        ntasks_per_node=None,
        array=f"0-{n_units - 1}%{concurrency}",
    )
    lines.append("")
    lines.append("set -eo pipefail")
    lines.append("")

    if modules:
        lines.append("# Load required modules")
        for module in modules:
            lines.append(f"module load {module}")
        lines.append("")

    if venv:
        lines.append("# Activate the project virtual environment")
        lines.append(f"source {_q(Path(venv) / 'bin' / 'activate')}")
        lines.append("")

    lines.append("# Threading: one array task must not oversubscribe its cpus")
    lines.append("export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK")
    lines.append("export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK")
    if exports:
        for key, value in exports.items():
            lines.append(f"export {key}={_q(value)}")
    lines.append("")

    lines.append('echo "Job ID: $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID"')
    lines.append('echo "Node: $SLURM_NODELIST  Start: $(date)"')
    lines.append("")

    argv = [
        python, "-m", "eegcpm.cli.main", "hpc", "run-unit",
        "--project", str(project),
        "--manifest", str(units_chunk_path),
        "--index", "$SLURM_ARRAY_TASK_ID",
        "--stages", ",".join(unit_stages),
        "--pipeline", pipeline,
        "--logs", str(log_dir),
        "--state-db-dir", str(state_db_dir),
    ]
    if source_variant:
        argv += ["--source-variant", source_variant]
    if config_paths:
        for flag, key in (
            ("--preprocessing-config", "preprocessing"),
            ("--source-config", "source"),
            ("--tasks-config-dir", "tasks_dir"),
        ):
            if config_paths.get(key):
                argv += [flag, str(config_paths[key])]

    # Per-unit state database: <state_db_dir>/<unit.key>.db (isolation between
    # concurrent array tasks).
    lines.append(
        f"# Per-unit state DB: {state_db_dir}/<unit.key>.db "
        "(see eegcpm.pipeline.slurm.unit_state_db_path)"
    )
    lines.append(" ".join(_q(arg) if arg != "$SLURM_ARRAY_TASK_ID" else arg for arg in argv))
    lines.append("")
    # `set -e` makes a failing run-unit end the array task with its exit code,
    # so SLURM accounting (sacct) shows the failure too.
    lines.append('echo "End: $(date)"')
    return "\n".join(lines) + "\n"


def generate_command_slurm_script(
    job_name: str,
    command: str,
    *,
    hpc: Mapping[str, Any],
    project_root: Union[str, Path],
    log_dir: Union[str, Path],
    email: Optional[str] = None,
    partition: Optional[str] = None,
    time: Optional[str] = None,
    mem: Optional[str] = None,
    cpus: Optional[int] = None,
    venv: Optional[Union[str, Path]] = None,
    modules: Optional[Sequence[str]] = None,
    work_dir: Optional[Union[str, Path]] = None,
    output_pattern: str = "%A.out",
    error_pattern: str = "%A.err",
    mail_type: str = "END,FAIL",
) -> str:
    """Render a one-off SLURM script that runs ``command``.

    This is the generic flavour used by the UI's "run this command on HPC"
    scripts (prediction page): one job, no array, the command embedded verbatim
    after the environment setup. Cluster values default from the ``hpc``
    mapping (``partition``/``cpus``/``mem``/``time``/``email``/``modules``),
    and explicit keyword arguments win over it.

    ``log_dir`` receives the ``%A``-suffixed out/err files (there is no array,
    so no ``%a``) and is created by the script. ``venv`` is sourced when given;
    ``work_dir`` is ``cd``-ed into when given.

    The header comes from the same :func:`_sbatch_header_lines` helper as the
    other two flavours — never inline a ``#SBATCH`` header in a UI page.
    """
    if not command or not command.strip():
        raise ValueError("generate_command_slurm_script requires a command")

    partition = partition or hpc.get("partition") or "shared_cpu"
    cpus = cpus or hpc.get("cpus") or 4
    mem = mem or hpc.get("mem") or "16G"
    time = time or hpc.get("time") or "02:00:00"
    email = email if email is not None else hpc.get("email", "")
    if modules is None:
        modules = hpc.get("modules") or []

    log_dir = Path(log_dir)
    lines = _sbatch_header_lines(
        job_name=job_name,
        partition=partition,
        cpus=cpus,
        mem=mem,
        time=time,
        output=f"{log_dir}/{output_pattern}",
        error=f"{log_dir}/{error_pattern}",
        mail_type=mail_type,
        email=email,
    )

    body = ["", "set -e", f'mkdir -p "{log_dir}"']
    for module in modules:
        body.append(f"module load {module}")
    if venv:
        body.append(f'source "{venv}"')
    if work_dir:
        body.append(f'cd "{work_dir}"')

    return "\n".join(lines) + "\n" + "\n".join(body) + "\n\n" + command + "\n"


def generate_local_script(config: dict, bids_root: Path, eegcpm_root: Path) -> str:
    """Generate bash script for local batch preprocessing."""

    tasks_str = " ".join(f'"{t}"' for t in config['tasks'])

    script = f"""#!/bin/bash
# EEGCPM Batch Preprocessing Script
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Pipeline: {config['pipeline_name']}
# Subjects: {len(config['subjects'])}
# Tasks: {', '.join(config['tasks'])}

set -e  # Exit on error

# Paths
BIDS_ROOT="{bids_root}"
PROJECT_ROOT="$(dirname "$BIDS_ROOT")"
EEGCPM_ROOT="{eegcpm_root}"
CONFIG_FILE="{config['config_file']}"
PIPELINE="{config['pipeline_name']}"

# Tasks to process
TASKS=({tasks_str})

# Create output directories (handled by CLI automatically)

# Process subjects
TOTAL={len(config['subjects'])}

"""

    script += """for TASK in "${TASKS[@]}"; do
    echo "========================================"
    echo "Processing task: $TASK"
    echo "========================================"
    CURRENT=0

"""

    for subject in config['subjects']:
        script += f"""    # Subject: {subject}
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TOTAL] Processing {subject} (task: $TASK)..."

    eegcpm preprocess \\
        --project "$PROJECT_ROOT" \\
        --config "$CONFIG_FILE" \\
        --pipeline "$PIPELINE" \\
        --subject {subject} \\
        --task "$TASK" \\
"""
        if config['force']:
            script += "        --force \\\n"

        script += """        2>&1 | tee -a "$PROJECT_ROOT/derivatives/preprocessing/$PIPELINE/preprocessing.log"

    if [ $? -eq 0 ]; then
        echo "  ✓ {subject} completed"
    else
        echo "  ✗ {subject} failed"
    fi

""".format(subject=subject)

    script += """done

echo "Batch preprocessing complete!"
"""

    return script


def generate_slurm_script(config: dict, bids_root: Path, eegcpm_root: Path) -> str:
    """Generate SLURM script for HPC batch preprocessing (per-subject array).

    One array task per subject; the subject loops over all tasks. Kept
    byte-compatible with the historical UI output (same HPC paths, placeholders
    when unset, ``--array=0-N%parallel_jobs``, force flag, email line, venv
    activation, pycache clearing, log dirs); it is the batch page's generator.
    """

    tasks_str = " ".join(f'"{t}"' for t in config['tasks'])

    # HPC settings (use placeholders if not configured)
    hpc = config.get('hpc', {})
    hpc_bids = hpc.get('bids_root', '') or '/path/to/your/bids/on/hpc'
    hpc_eegcpm = hpc.get('eegcpm_root', '') or '/path/to/eegcpm-0.1/on/hpc'
    hpc_email = hpc.get('email', '')
    hpc_partition = hpc.get('partition', 'shared_cpu')
    hpc_time = hpc.get('time', '02:00:00')
    hpc_mem = hpc.get('mem', '16G')
    hpc_cpus = hpc.get('cpus', 4)

    # Derive project root (parent of bids/)
    hpc_project_root = hpc_bids.rstrip('/').removesuffix('/bids') if hpc_bids.rstrip('/').endswith('/bids') else hpc_bids

    # Determine config file path on HPC
    # Use project-level configs (eegcpm/configs/) not repo-level (eegcpm/config/)
    config_filename = Path(str(config['config_file'])).name
    hpc_config_file = f"{hpc_project_root}/eegcpm/configs/preprocessing/{config_filename}"

    # Venv path (use user-specified or derive from eegcpm root)
    hpc_venv = hpc.get('venv_path', '') or f'{hpc_eegcpm}/venv/bin/activate'

    header = "\n".join(_sbatch_header_lines(
        job_name=f"eegcpm_{config['pipeline_name']}",
        partition=hpc_partition,
        cpus=hpc_cpus,
        mem=hpc_mem,
        time=hpc_time,
        array=f"0-{len(config['subjects'])-1}%{config['parallel_jobs']}",
        output=f"{hpc_project_root}/derivatives/preprocessing/{config['pipeline_name']}/slurm_%A_%a.out",
        error=f"{hpc_project_root}/derivatives/preprocessing/{config['pipeline_name']}/slurm_%A_%a.err",
        mail_type="END,FAIL",
        email=hpc_email,
    ))

    script = header + f"""

# EEGCPM SLURM Batch Preprocessing
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Pipeline: {config['pipeline_name']}
# Tasks: {', '.join(config['tasks'])}
# Subjects: {len(config['subjects'])}

set -e

# ============================================================
# Paths
# ============================================================
BIDS_ROOT="{hpc_bids}"
PROJECT_ROOT="$(dirname "$BIDS_ROOT")"
EEGCPM_ROOT="{hpc_eegcpm}"
CONFIG_FILE="{hpc_config_file}"
PIPELINE="{config['pipeline_name']}"

# Tasks to process
TASKS=({tasks_str})

# ============================================================
# Environment Setup
# ============================================================
# Activate Python venv (not conda)
source "{hpc_venv}"

# ============================================================
# Pre-flight: Clear Python cache to ensure fresh code is loaded
# ============================================================
find "$EEGCPM_ROOT" -type d -name "__pycache__" -exec rm -rf {{}} + 2>/dev/null || true

# ============================================================
# Create output directory (SLURM fails if log dir doesn't exist)
# ============================================================
mkdir -p "$PROJECT_ROOT/derivatives/preprocessing/$PIPELINE"

# ============================================================
# Subject list
# ============================================================
SUBJECTS=(
"""

    for subject in config['subjects']:
        script += f'    "{subject}"\n'

    script += f""")

# Get subject for this array task
SUBJECT="${{SUBJECTS[$SLURM_ARRAY_TASK_ID]}}"

echo "========================================"
echo "SLURM Array Task: $SLURM_ARRAY_TASK_ID / {len(config['subjects'])}"
echo "Processing subject: $SUBJECT"
echo "Tasks: {', '.join(config['tasks'])}"
echo "Pipeline: {config['pipeline_name']}"
echo "Started: $(date)"
echo "========================================"

# ============================================================
# Run preprocessing (loop through all tasks)
# ============================================================
for TASK in "${{TASKS[@]}}"; do
    echo "--- Processing task: $TASK ---"

    eegcpm preprocess \\
        --project "$PROJECT_ROOT" \\
        --config "$CONFIG_FILE" \\
        --pipeline "$PIPELINE" \\
        --subject "$SUBJECT" \\
        --task "$TASK" \\
"""

    if config['force']:
        script += "        --force \\\n"

    script += """        2>&1

    echo "--- Task $TASK complete for $SUBJECT ---"
done

echo "========================================"
echo "Subject $SUBJECT complete (all tasks)"
echo "Finished: $(date)"
echo "========================================"
"""

    return script


# =================================================================== finalize


def finalize_task(
    project: Union[str, Path],
    task: str,
    *,
    pipeline: str,
    source_variant: str,
    subjects: Optional[Sequence[str]] = None,
    features_config: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    dry_run: bool = False,
    force: bool = False,
    python: Optional[str] = None,
) -> Dict[str, Any]:
    """Build (and optionally run) the ONE task-level features invocation.

    Every expected unit of ``task`` must have a complete **source** artifact
    (the input of the source_erp features stage); otherwise a ``RuntimeError``
    is raised unless ``force=True``. The features command is then built with
    every subject of the task in a single invocation — the features stage is
    task-level and must never run per unit (it would overwrite the same
    parquet from concurrent array tasks).

    Returns a record with the expected/complete unit keys, the command(s) and,
    when executed, ``returncode``.
    """
    project = Path(project)
    python = _resolve_python(None, python)
    features_config = Path(features_config)
    if not features_config.is_file():
        raise FileNotFoundError(f"features config not found: {features_config}")

    paths = EEGCPMPaths(project)
    output_dir = (
        Path(output_dir) if output_dir is not None
        else features_output_dir(project, pipeline)
    )

    units = build_manifest(paths.bids_root, tasks=[task], subjects=subjects)
    if not units:
        raise FileNotFoundError(
            f"no BIDS input files for task {task!r} under {paths.bids_root}"
            + (f" (subjects={list(subjects)})" if subjects else "")
        )

    unit_subjects = sorted({unit.subject for unit in units})
    if subjects:
        requested = _normalize_set(subjects, "sub-")
        missing = sorted(requested - set(unit_subjects))
        if missing:
            logger.warning(
                "hpc finalize: requested subject(s) have no BIDS input for task %s: %s",
                task, ", ".join(missing),
            )

    incomplete = prune_units(
        project, units, ("source",),
        pipeline=pipeline, source_variant=source_variant,
    )
    if incomplete and not force:
        raise RuntimeError(
            f"finalize refused: {len(incomplete)}/{len(units)} unit(s) for task "
            f"{task!r} have no complete source artifact — "
            + ", ".join(unit.key for unit in incomplete)
            + "; re-run the missing source stages or pass force=True"
        )

    command = [python, "-m", "eegcpm.cli.main", "features"]
    command += [
        "--project", str(project),
        "--config", str(features_config),
        "--feature-type", "source_erp",
        "--output-dir", str(output_dir),
        "--preprocessing", pipeline,
        "--task", task,
        "--source", source_variant,
        "--subjects", *unit_subjects,
        "--condition", task,
    ]

    result: Dict[str, Any] = {
        "task": task,
        "pipeline": pipeline,
        "source_variant": source_variant,
        "subjects": unit_subjects,
        "units": [unit.key for unit in units],
        "incomplete_units": [unit.key for unit in incomplete],
        "commands": [command],
        "output": str(output_dir / f"source_erp_{task}.parquet"),
        "dry_run": bool(dry_run),
    }
    if dry_run:
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(command, cwd=str(project), check=False)
        result["returncode"] = completed.returncode
    except OSError as exc:
        result["returncode"] = 1
        result["error"] = f"could not execute {command[0]}: {exc}"
        logger.error("hpc finalize %s: %s", task, result["error"])
    return result
