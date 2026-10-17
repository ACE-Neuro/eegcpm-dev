"""HPC batch tooling (``eegcpm hpc`` + the UI script flavours) — no cluster access.

Pins the work-unit manifest, the round-robin partition chunking, the
per-stage artifact checks (prune), the generated sbatch scripts (array,
per-subject and one-off flavours), the ``HpcBatchConfig`` schema behind
``hpc submit --config``, the ``run-unit`` dry-run/skip/state-isolation
behaviour, the CLI flow (``manifest`` → ``submit --dry-run``, ``finalize``),
the page-12 delegation to the unified module, and the preprocess
``--state-db`` override.

Everything here runs locally: ``submit`` is only exercised with ``--dry-run``
(``sbatch`` is monkeypatched to fail the test if it is ever called), and
``run-unit`` subprocesses are either dry-runs or a deliberately-failing
interpreter (``/bin/false``).
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from eegcpm.cli import hpc as hpc_cli
from eegcpm.cli.main import main as cli_main
from eegcpm.cli.preprocess import preprocess_command
from eegcpm.core.config import HpcBatchConfig, SlurmConfig
from eegcpm.core.models import Project, Subject
from eegcpm.data.bids_utils import parse_run_identity
from eegcpm.pipeline import slurm as pipeline_slurm
from eegcpm.pipeline.slurm import (
    SlurmSubmitter,
    WorkUnit,
    build_manifest,
    chunk_units,
    finalize_task,
    generate_array_script,
    generate_command_slurm_script,
    prune_units,
    read_manifest,
    run_unit,
    unit_complete,
    write_manifest,
)

PIPELINE = "pp"
VARIANT = "sv"


# --------------------------------------------------------------------- helpers

def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _preprocess_sidecar(
    project: Path, unit: WorkUnit, *, pipeline: str = PIPELINE,
    success: bool = True, config_hash: str = "hash-a",
) -> Path:
    path = (
        project / "derivatives" / "preprocessing" / pipeline
        / f"sub-{unit.subject}" / f"ses-{unit.session}"
        / f"task-{unit.task}" / f"run-{unit.run}"
        / f"{unit.subject}_ses-{unit.session}_run-{unit.run}_preprocessing.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"success": success, "config_hash": config_hash}))
    return path


def _epochs_artifact(project: Path, unit: WorkUnit, *, pipeline: str = PIPELINE) -> Path:
    path = (
        project / "derivatives" / "epochs" / pipeline / unit.task
        / f"sub-{unit.subject}" / f"ses-{unit.session}"
        / f"{unit.subject}_epo.fif"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _source_artifact(
    project: Path, unit: WorkUnit, *, pipeline: str = PIPELINE, variant: str = VARIANT
) -> Path:
    path = (
        project / "derivatives" / "source" / pipeline / unit.task
        / f"variant-{variant}" / f"sub-{unit.subject}" / f"ses-{unit.session}"
        / f"{unit.subject}_roi_tc.npz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _features_artifact(project: Path, task: str, *, pipeline: str = PIPELINE) -> Path:
    path = (
        project / "derivatives" / "features" / pipeline / "source_erp"
        / f"source_erp_{task}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _config_paths(tmp_path: Path) -> dict:
    return {
        "preprocessing": tmp_path / "configs" / "pre.yaml",
        "source": tmp_path / "configs" / "source.yaml",
        "tasks_dir": tmp_path / "configs" / "tasks",
    }


@pytest.fixture
def bids_tree(tmp_path: Path) -> Path:
    """Synthetic BIDS tree.

    sub-01: ses-01 with task-A (2 runs) + task-B (single file);
    sub-02: ses-01 with task-A + task-B;
    sub-03: no session directory, single task-A file.
    """
    root = tmp_path / "bids"
    for name in (
        "sub-01_ses-01_task-A_run-1_eeg.fif",
        "sub-01_ses-01_task-A_run-2_eeg.fif",
        "sub-01_ses-01_task-B_eeg.fif",
        "sub-02_ses-01_task-A_eeg.fif",
        "sub-02_ses-01_task-B_eeg.fif",
    ):
        _touch(root / name.split("_")[0] / "ses-01" / "eeg" / name)
    _touch(root / "sub-03" / "eeg" / "sub-03_task-A_eeg.fif")
    return root


EXPECTED_UNITS = [
    WorkUnit("01", "01", "A", "1"),
    WorkUnit("01", "01", "A", "2"),
    WorkUnit("01", "01", "B", "1"),
    WorkUnit("02", "01", "A", "1"),
    WorkUnit("02", "01", "B", "1"),
    WorkUnit("03", "01", "A", "1"),
]


# ------------------------------------------------------------------ manifest

class TestBuildManifest:
    def test_exact_unit_set(self, bids_tree):
        assert build_manifest(bids_tree) == EXPECTED_UNITS

    def test_task_and_subject_filters(self, bids_tree):
        assert build_manifest(bids_tree, tasks=["B"]) == [
            WorkUnit("01", "01", "B", "1"),
            WorkUnit("02", "01", "B", "1"),
        ]
        assert build_manifest(bids_tree, subjects=["sub-01", "03"]) == [
            WorkUnit("01", "01", "A", "1"),
            WorkUnit("01", "01", "A", "2"),
            WorkUnit("01", "01", "B", "1"),
            WorkUnit("03", "01", "A", "1"),
        ]

    def test_missing_bids_root_is_loud(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_manifest(tmp_path / "nope")

    def test_manifest_roundtrip_and_key(self, bids_tree, tmp_path):
        units = build_manifest(bids_tree)
        path = write_manifest(units, tmp_path / "hpc" / "manifest.tsv")
        assert read_manifest(path) == units
        assert path.read_text().splitlines()[0] == "subject\tsession\ttask\trun"
        assert units[0].key == "sub-01_ses-01_task-A_run-1"

    def test_read_manifest_rejects_duplicates_and_bad_header(self, tmp_path):
        dup = tmp_path / "dup.tsv"
        dup.write_text(
            "subject\tsession\ttask\trun\n"
            "01\t01\tA\t1\n"
            "01\t01\tA\t1\n"
        )
        with pytest.raises(ValueError, match="duplicate"):
            read_manifest(dup)

        no_header = tmp_path / "bad.tsv"
        no_header.write_text("01\t01\tA\t1\n")
        with pytest.raises(ValueError, match="header"):
            read_manifest(no_header)

    def test_parse_run_identity_shared_rule(self):
        assert parse_run_identity("sub-01_ses-02_task-A_run-3_eeg.vhdr") == ("02", "3")
        # No run entity: the fallback index is the run number (as in the CLI).
        assert parse_run_identity("sub-01_ses-01_task-A_eeg.fif", fallback_index=2) == ("01", "2")
        assert parse_run_identity("sub-01_task-A_eeg.fif", fallback_index=1) == ("01", "1")


class TestChunkUnits:
    def test_every_unit_exactly_once_round_robin(self):
        partitions = ["normal", "long", "gpu"]
        chunks = chunk_units(EXPECTED_UNITS, partitions)
        assert [name for name, _ in chunks] == partitions
        assert chunks[0][1] == [EXPECTED_UNITS[0], EXPECTED_UNITS[3]]
        assert chunks[1][1] == [EXPECTED_UNITS[1], EXPECTED_UNITS[4]]
        assert chunks[2][1] == [EXPECTED_UNITS[2], EXPECTED_UNITS[5]]
        flat = [unit for _, units in chunks for unit in units]
        assert sorted(flat) == EXPECTED_UNITS
        assert len(flat) == len(EXPECTED_UNITS) == len(set(flat))

    def test_deterministic_and_balanced(self):
        assert chunk_units(EXPECTED_UNITS, ["a", "b"]) == chunk_units(
            EXPECTED_UNITS, ["a", "b"])
        chunks = chunk_units(EXPECTED_UNITS, ["a", "b"])
        assert abs(len(chunks[0][1]) - len(chunks[1][1])) <= 1

    def test_requires_partitions(self):
        with pytest.raises(ValueError):
            chunk_units(EXPECTED_UNITS, [])


# --------------------------------------------------------- completeness/prune

class TestUnitCompleteAndPrune:
    def test_preprocess_requires_success_and_matching_hash(self, tmp_path):
        project = tmp_path / "proj"
        unit = WorkUnit("01", "01", "A", "1")
        assert unit_complete(project, unit, "preprocess", pipeline=PIPELINE) is False

        _preprocess_sidecar(project, unit, success=False)
        assert unit_complete(project, unit, "preprocess", pipeline=PIPELINE) is False

        _preprocess_sidecar(project, unit, success=True, config_hash="hash-a")
        assert unit_complete(project, unit, "preprocess", pipeline=PIPELINE) is True
        assert unit_complete(
            project, unit, "preprocess", pipeline=PIPELINE,
            expected_config_hash="hash-a",
        ) is True
        assert unit_complete(
            project, unit, "preprocess", pipeline=PIPELINE,
            expected_config_hash="other",
        ) is False

    def test_corrupt_preprocess_sidecar_is_not_complete(self, tmp_path, caplog):
        """An unreadable sidecar is incomplete AND logged (never silent)."""
        project = tmp_path / "proj"
        unit = WorkUnit("01", "01", "A", "1")
        sidecar = _preprocess_sidecar(project, unit)
        sidecar.write_text("{not json")
        with caplog.at_level("WARNING", logger="eegcpm.pipeline.slurm"):
            assert unit_complete(project, unit, "preprocess", pipeline=PIPELINE) is False
        assert any("unreadable JSON" in record.message for record in caplog.records)

    def test_missing_artifact_never_complete(self, tmp_path):
        project = tmp_path / "proj"
        unit = WorkUnit("01", "01", "A", "1")
        assert unit_complete(project, unit, "epochs", pipeline=PIPELINE) is False
        assert unit_complete(
            project, unit, "source", pipeline=PIPELINE, source_variant=VARIANT
        ) is False
        _epochs_artifact(project, unit)
        _source_artifact(project, unit)
        assert unit_complete(project, unit, "epochs", pipeline=PIPELINE) is True
        assert unit_complete(
            project, unit, "source", pipeline=PIPELINE, source_variant=VARIANT
        ) is True

    def test_source_stage_requires_variant(self, tmp_path):
        unit = WorkUnit("01", "01", "A", "1")
        with pytest.raises(ValueError, match="source_variant"):
            unit_complete(tmp_path, unit, "source", pipeline=PIPELINE)

    def test_unknown_stage_is_loud(self, tmp_path):
        unit = WorkUnit("01", "01", "A", "1")
        with pytest.raises(ValueError, match="unknown stage"):
            unit_complete(tmp_path, unit, "predict", pipeline=PIPELINE)

    def test_features_stage_is_task_level(self, tmp_path):
        project = tmp_path / "proj"
        unit = WorkUnit("01", "01", "A", "1")
        assert unit_complete(project, unit, "features", pipeline=PIPELINE) is False
        _features_artifact(project, "A")
        assert unit_complete(project, unit, "features", pipeline=PIPELINE) is True
        with pytest.raises(ValueError, match="task-level"):
            prune_units(project, [unit], ["features"], pipeline=PIPELINE)

    def test_prune_drops_only_fully_complete_units(self, tmp_path):
        project = tmp_path / "proj"
        complete = WorkUnit("01", "01", "A", "1")
        pre_only = WorkUnit("02", "01", "A", "1")
        wrong_hash = WorkUnit("03", "01", "A", "1")

        _preprocess_sidecar(project, complete)
        _epochs_artifact(project, complete)
        _source_artifact(project, complete)
        _preprocess_sidecar(project, pre_only)
        _preprocess_sidecar(project, wrong_hash, config_hash="stale")

        # preprocess-only: the wrong-hash unit stays pending
        assert prune_units(
            project, [complete, pre_only, wrong_hash], ["preprocess"],
            pipeline=PIPELINE, expected_config_hash="hash-a",
        ) == [wrong_hash]

        # a unit complete for preprocess but not source is pruned only when the
        # requested stages are preprocess-only
        assert prune_units(
            project, [complete, pre_only], ["preprocess"], pipeline=PIPELINE,
        ) == []
        assert prune_units(
            project, [complete, pre_only], ["preprocess", "source"],
            pipeline=PIPELINE, source_variant=VARIANT,
        ) == [pre_only]

    def test_epochs_and_source_artifacts_are_session_level(self, tmp_path):
        """The documented multi-run semantics: both runs of a session share the
        epochs/source artifact, so once one run produced it the other units
        report those stages complete (they then skip, never re-run them)."""
        project = tmp_path / "proj"
        run_1 = WorkUnit("01", "01", "A", "1")
        run_2 = WorkUnit("01", "01", "A", "2")
        _epochs_artifact(project, run_1)
        _source_artifact(project, run_1)
        assert unit_complete(project, run_2, "epochs", pipeline=PIPELINE) is True
        assert unit_complete(
            project, run_2, "source", pipeline=PIPELINE, source_variant=VARIANT
        ) is True
        # ... but run_2's own preprocess artifact is still independent.
        assert unit_complete(project, run_2, "preprocess", pipeline=PIPELINE) is False

    def test_prune_requires_stages(self, tmp_path):
        with pytest.raises(ValueError):
            prune_units(tmp_path, EXPECTED_UNITS, [], pipeline=PIPELINE)


# ------------------------------------------------------------ sbatch scripts

class TestGenerateArrayScript:
    def _script(self, tmp_path, **overrides):
        kwargs = dict(
            job_name="eegcpm_pp_normal",
            partition="normal",
            units_chunk_path="/proj/hpc/submissions/s/chunk.tsv",
            n_units=7,
            concurrency=3,
            cpus=4,
            mem="16G",
            time="02:00:00",
            log_dir="/proj/hpc/logs",
            venv="/opt/venv",
            modules=["python/3.11", "cuda/12.2"],
            exports={"EEGCPM_HPC": "1"},
            project="/proj",
            stages=["preprocess", "epochs", "source"],
            pipeline=PIPELINE,
            source_variant=VARIANT,
            config_paths={
                "preprocessing": "/proj/configs/pre.yaml",
                "source": "/proj/configs/source.yaml",
                "tasks_dir": "/proj/eegcpm/configs/tasks",
            },
        )
        kwargs.update(overrides)
        return generate_array_script(**kwargs)

    def test_headers_and_environment(self, tmp_path):
        script = self._script(tmp_path)
        assert script.startswith("#!/bin/bash\n")
        assert "#SBATCH --array=0-6%3" in script
        assert "#SBATCH --partition=normal" in script
        assert "#SBATCH --job-name=eegcpm_pp_normal" in script
        assert "#SBATCH --cpus-per-task=4" in script
        assert "#SBATCH --mem=16G" in script
        assert "#SBATCH --time=02:00:00" in script
        assert "#SBATCH --output=/proj/hpc/logs/eegcpm_pp_normal_%A_%a.out" in script
        assert "#SBATCH --error=/proj/hpc/logs/eegcpm_pp_normal_%A_%a.err" in script
        assert "module load python/3.11" in script
        assert "module load cuda/12.2" in script
        assert "source /opt/venv/bin/activate" in script
        assert "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK" in script
        assert "export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK" in script
        assert "export EEGCPM_HPC=1" in script
        assert "set -eo pipefail" in script

    def test_job_step(self, tmp_path):
        script = self._script(tmp_path)
        assert "/opt/venv/bin/python -m eegcpm.cli.main hpc run-unit" in script
        assert "--project /proj" in script
        assert "--manifest /proj/hpc/submissions/s/chunk.tsv" in script
        assert "--index $SLURM_ARRAY_TASK_ID" in script
        assert "--stages preprocess,epochs,source" in script
        assert f"--pipeline {PIPELINE}" in script
        assert f"--source-variant {VARIANT}" in script
        assert "--logs /proj/hpc/logs" in script
        assert "--preprocessing-config /proj/configs/pre.yaml" in script
        assert "--source-config /proj/configs/source.yaml" in script
        assert "--tasks-config-dir /proj/eegcpm/configs/tasks" in script
        # Per-unit state DB isolation (path pattern documented in the script).
        assert "--state-db-dir /proj/hpc/state" in script
        assert "/proj/hpc/state/<unit.key>.db" in script

    def test_default_interpreter_is_the_submitter(self):
        script = self._script(Path("/tmp"), venv=None)
        assert f"{sys.executable} -m eegcpm.cli.main hpc run-unit" in script
        assert "activate" not in script

    def test_single_unit_array_and_validation(self):
        script = self._script(Path("/tmp"), n_units=1, concurrency=1, venv=None)
        assert "#SBATCH --array=0-0%1" in script
        with pytest.raises(ValueError):
            self._script(Path("/tmp"), n_units=0)
        with pytest.raises(ValueError):
            self._script(Path("/tmp"), stages=[])


# ------------------------------------------------------------------ run_unit

class TestRunUnit:
    def test_dry_run_exact_commands(self, tmp_path):
        project = tmp_path / "proj"
        configs = _config_paths(tmp_path)
        logs_dir = tmp_path / "proj" / "hpc" / "logs"
        state_db_dir = tmp_path / "proj" / "hpc" / "state"
        unit = WorkUnit("01", "01", "A", "1")

        record = run_unit(
            unit,
            project=project,
            stages=["preprocess", "epochs", "source"],
            pipeline=PIPELINE,
            source_variant=VARIANT,
            config_paths=configs,
            state_db_dir=state_db_dir,
            logs_dir=logs_dir,
            dry_run=True,
            python="/usr/bin/python3",
        )

        assert record["unit"] == "sub-01_ses-01_task-A_run-1"
        assert record["exit_code"] == 0
        assert record["stages"]["preprocess"]["commands"] == [[
            "/usr/bin/python3", "-m", "eegcpm.cli.main", "preprocess",
            "--config", str(configs["preprocessing"]),
            "--project", str(project),
            "--subject", "01",
            "--task", "A",
            "--pipeline", PIPELINE,
            "--task-config", "A",
            "--state-db", str(state_db_dir / "sub-01_ses-01_task-A_run-1.db"),
            "--sessions", "01",
            "--runs", "1",
        ]]
        assert record["stages"]["epochs"]["commands"] == [[
            "/usr/bin/python3", "-m", "eegcpm.cli.main", "epochs",
            "--project", str(project),
            "--config", str(configs["tasks_dir"] / "A.yaml"),
            "--preprocessing", PIPELINE,
            "--task", "A",
            "--subjects", "01",
            "--sessions", "01",
        ]]
        assert record["stages"]["source"]["commands"] == [[
            "/usr/bin/python3", "-m", "eegcpm.cli.main", "source-reconstruct",
            "--project", str(project),
            "--config", str(configs["source"]),
            "--preprocessing", PIPELINE,
            "--task", "A",
            "--subjects", "01",
            "--task-config", "A",
            "--sessions", "01",
        ]]
        # Dry run: no executions, no writes at all.
        assert not logs_dir.exists()
        assert not state_db_dir.exists()

    def test_completed_stages_are_skipped(self, tmp_path):
        project = tmp_path / "proj"
        unit = WorkUnit("01", "01", "A", "1")
        _preprocess_sidecar(project, unit)
        _epochs_artifact(project, unit)

        record = run_unit(
            unit,
            project=project,
            stages=["preprocess", "epochs", "source"],
            pipeline=PIPELINE,
            source_variant=VARIANT,
            config_paths=_config_paths(tmp_path),
            state_db_dir=tmp_path / "hpc" / "state",
            logs_dir=tmp_path / "hpc" / "logs",
            dry_run=True,
            python="/usr/bin/python3",
        )

        assert record["stages"]["preprocess"]["status"] == "skipped"
        assert record["stages"]["preprocess"]["commands"] == []
        assert record["stages"]["epochs"]["status"] == "skipped"
        assert record["stages"]["epochs"]["commands"] == []
        assert record["stages"]["source"]["status"] == "dry-run"
        commands = [
            command
            for stage in record["stages"].values()
            for command in stage["commands"]
        ]
        assert len(commands) == 1
        assert commands[0][3] == "source-reconstruct"

    def test_failed_stage_records_and_exits_nonzero(self, tmp_path):
        """A real (non-dry) run whose stage fails writes the record + rc."""
        project = tmp_path / "proj"
        project.mkdir(parents=True)
        logs_dir = tmp_path / "proj" / "hpc" / "logs"
        unit = WorkUnit("01", "01", "A", "1")

        record = run_unit(
            unit,
            project=project,
            stages=["preprocess", "epochs"],
            pipeline=PIPELINE,
            source_variant=VARIANT,
            config_paths=_config_paths(tmp_path),
            state_db_dir=project / "hpc" / "state",
            logs_dir=logs_dir,
            dry_run=False,
            python="/bin/false",  # exits 1 without doing anything
        )

        assert record["exit_code"] == 1
        assert record["stages"]["preprocess"]["status"] == "failed"
        assert record["stages"]["preprocess"]["returncode"] == 1
        assert "epochs" not in record["stages"], "chain must stop at the failure"

        record_path = logs_dir.parent / "units" / f"{unit.key}.json"
        assert record_path.is_file(), "failed run wrote no per-unit record"
        payload = json.loads(record_path.read_text())
        assert payload["stages"]["preprocess"]["status"] == "failed"
        assert payload["exit_code"] == 1

    def test_task_level_stage_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="task-level"):
            run_unit(
                WorkUnit("01", "01", "A", "1"),
                project=tmp_path,
                stages=["features"],
                pipeline=PIPELINE,
                source_variant=VARIANT,
                config_paths={},
                state_db_dir=tmp_path,
                logs_dir=tmp_path,
                dry_run=True,
            )

    def test_missing_config_path_is_loud(self, tmp_path):
        with pytest.raises(ValueError, match="config_paths"):
            run_unit(
                WorkUnit("01", "01", "A", "1"),
                project=tmp_path,
                stages=["preprocess"],
                pipeline=PIPELINE,
                source_variant=None,
                config_paths={},
                state_db_dir=tmp_path,
                logs_dir=tmp_path,
                dry_run=True,
            )


# ---------------------------------------------------------------- CLI flow

def _run_cli(monkeypatch, *argv) -> None:
    monkeypatch.setattr(sys, "argv", ["eegcpm", *argv])
    cli_main()


def _boom(*args, **kwargs):
    raise AssertionError("the cluster must not be contacted in tests")


class TestHpcCli:
    def test_manifest_then_submit_dry_run(self, bids_tree, tmp_path, monkeypatch, capsys):
        project = bids_tree.parent  # bids_tree = <tmp>/bids
        pre = tmp_path / "pre.yaml"
        src = tmp_path / "source.yaml"

        # Any cluster call is a test failure: --dry-run must not even look for
        # sbatch.
        monkeypatch.setattr(hpc_cli.shutil, "which", _boom)
        monkeypatch.setattr(hpc_cli.subprocess, "run", _boom)

        _run_cli(
            monkeypatch, "hpc", "manifest",
            "--project", str(project), "--out", "hpc/manifest.tsv",
        )
        manifest = project / "hpc" / "manifest.tsv"
        assert manifest.is_file()
        assert read_manifest(manifest) == EXPECTED_UNITS
        assert "6 work unit(s)" in capsys.readouterr().out

        _run_cli(
            monkeypatch, "hpc", "submit",
            "--project", str(project),
            "--manifest", "hpc/manifest.tsv",
            "--stages", "preprocess,epochs,source",
            "--pipeline", PIPELINE,
            "--source-variant", VARIANT,
            "--partitions", "normal,long",
            "--concurrency-per-partition", "2",
            "--cpus", "4", "--mem", "16G", "--time", "02:00:00",
            "--log-dir", "hpc/logs",
            "--preprocessing-config", str(pre),
            "--source-config", str(src),
            "--venv", str(tmp_path / "venv"),
            "--module", "python/3.11",
            "--dry-run",
        )
        out = capsys.readouterr().out
        submission_dirs = list((project / "hpc" / "submissions").iterdir())
        assert len(submission_dirs) == 1
        scripts = {p.name: p for p in submission_dirs[0].glob("*.sbatch")}
        assert set(scripts) == {"eegcpm_normal.sbatch", "eegcpm_long.sbatch"}
        chunks = {p.name: p for p in submission_dirs[0].glob("*.tsv")}
        assert set(chunks) == {"eegcpm_normal.tsv", "eegcpm_long.tsv"}
        # round-robin: normal gets units 0, 2, 4; long gets 1, 3, 5
        assert read_manifest(chunks["eegcpm_normal.tsv"]) == [EXPECTED_UNITS[i] for i in (0, 2, 4)]
        assert read_manifest(chunks["eegcpm_long.tsv"]) == [EXPECTED_UNITS[i] for i in (1, 3, 5)]
        normal_script = scripts["eegcpm_normal.sbatch"].read_text()
        assert "#SBATCH --array=0-2%2" in normal_script
        assert "#SBATCH --partition=normal" in normal_script
        assert "module load python/3.11" in normal_script
        assert f"source {tmp_path / 'venv'}/bin/activate" in normal_script
        assert f"{tmp_path / 'venv'}/bin/python -m eegcpm.cli.main hpc run-unit" in normal_script
        assert "hpc run-unit" in normal_script
        assert "sbatch " in out, "dry run did not print the sbatch commands"

    def test_submit_without_dry_run_requires_sbatch(
        self, bids_tree, tmp_path, monkeypatch, capsys
    ):
        project = bids_tree.parent
        write_manifest(EXPECTED_UNITS, project / "hpc" / "manifest.tsv")
        monkeypatch.setattr(hpc_cli.shutil, "which", lambda name: None)
        monkeypatch.setattr(hpc_cli.subprocess, "run", _boom)

        with pytest.raises(SystemExit) as exc:
            _run_cli(
                monkeypatch, "hpc", "submit",
                "--project", str(project),
                "--manifest", "hpc/manifest.tsv",
                "--stages", "preprocess",
                "--pipeline", PIPELINE,
                "--partitions", "normal",
                "--concurrency-per-partition", "1",
                "--cpus", "1", "--mem", "1G", "--time", "00:10:00",
                "--log-dir", "hpc/logs",
                "--preprocessing-config", str(tmp_path / "pre.yaml"),
            )
        assert exc.value.code == 1
        assert "sbatch not found" in capsys.readouterr().err

    def test_submit_requires_stage_configs(self, bids_tree, tmp_path, monkeypatch, capsys):
        project = bids_tree.parent
        write_manifest(EXPECTED_UNITS, project / "hpc" / "manifest.tsv")
        monkeypatch.setattr(hpc_cli.shutil, "which", _boom)

        with pytest.raises(SystemExit) as exc:
            _run_cli(
                monkeypatch, "hpc", "submit",
                "--project", str(project),
                "--manifest", "hpc/manifest.tsv",
                "--stages", "preprocess,source",
                "--pipeline", PIPELINE,
                "--partitions", "normal",
                "--concurrency-per-partition", "1",
                "--cpus", "1", "--mem", "1G", "--time", "00:10:00",
                "--log-dir", "hpc/logs",
                "--dry-run",
            )
        assert exc.value.code == 2
        assert "--preprocessing-config is required" in capsys.readouterr().err

    def test_run_unit_dry_run_via_cli(self, bids_tree, tmp_path, monkeypatch, capsys):
        project = bids_tree.parent
        write_manifest(EXPECTED_UNITS, project / "hpc" / "manifest.tsv")
        monkeypatch.setattr(hpc_cli.subprocess, "run", _boom)

        _run_cli(
            monkeypatch, "hpc", "run-unit",
            "--project", str(project),
            "--manifest", "hpc/manifest.tsv",
            "--index", "0",
            "--stages", "preprocess,epochs,source",
            "--pipeline", PIPELINE,
            "--source-variant", VARIANT,
            "--logs", "hpc/logs",
            "--preprocessing-config", str(tmp_path / "pre.yaml"),
            "--source-config", str(tmp_path / "source.yaml"),
            "--dry-run",
        )
        out = capsys.readouterr().out
        assert "preprocess: dry-run" in out
        assert "--state-db" in out and "sub-01_ses-01_task-A_run-1.db" in out
        assert not (project / "hpc" / "units").exists()

    def test_run_unit_index_out_of_range(self, bids_tree, tmp_path, monkeypatch, capsys):
        project = bids_tree.parent
        write_manifest(EXPECTED_UNITS, project / "hpc" / "manifest.tsv")
        with pytest.raises(SystemExit) as exc:
            _run_cli(
                monkeypatch, "hpc", "run-unit",
                "--project", str(project),
                "--manifest", "hpc/manifest.tsv",
                "--index", "99",
                "--stages", "preprocess",
                "--pipeline", PIPELINE,
                "--logs", "hpc/logs",
                "--preprocessing-config", str(tmp_path / "pre.yaml"),
                "--dry-run",
            )
        assert exc.value.code == 2
        assert "out of range" in capsys.readouterr().err

    def test_status_reads_records(self, tmp_path, monkeypatch, capsys):
        project = tmp_path / "proj"
        units_dir = project / "hpc" / "units"
        units_dir.mkdir(parents=True)
        (units_dir / "sub-01_ses-01_task-A_run-1.json").write_text(json.dumps({
            "unit": "sub-01_ses-01_task-A_run-1",
            "stages": {
                "preprocess": {"status": "completed", "seconds": 12.0, "returncode": 0},
                "epochs": {"status": "completed", "seconds": 3.0, "returncode": 0},
                "source": {"status": "failed", "seconds": 1.0, "returncode": 1},
            },
        }))
        _run_cli(monkeypatch, "hpc", "status", "--project", str(project), "--logs", "hpc/logs")
        out = capsys.readouterr().out
        assert "sub-01_ses-01_task-A_run-1" in out
        assert "source: completed=0, failed=1, missing=0, skipped=0" in out

        # Empty records directory: read-only, no crash, no writes.
        empty = project / "hpc2"
        _run_cli(monkeypatch, "hpc", "status", "--project", str(project),
                 "--logs", str(empty / "logs"))
        assert "no per-unit records" in capsys.readouterr().out
        assert not (empty / "units").exists()

    def test_finalize_refuses_incomplete_then_lists_command(
        self, bids_tree, tmp_path, monkeypatch, capsys
    ):
        project = bids_tree.parent
        features_config = tmp_path / "features.yaml"
        features_config.write_text(
            "stage: features\nvariant: source_erp_test\n"
            "components: {P3: {search_window: [0.3, 0.5], polarity: positive}}\n"
            "metrics: [peak_amplitude]\n"
        )
        prepare = [
            "hpc", "finalize",
            "--project", str(project),
            "--task", "A",
            "--pipeline", PIPELINE,
            "--source-variant", VARIANT,
            "--features-config", str(features_config),
            "--subjects", "01", "02",
            "--dry-run",
        ]

        with pytest.raises(SystemExit) as exc:
            _run_cli(monkeypatch, *prepare)
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "finalize refused" in err
        assert "sub-01_ses-01_task-A_run-1" in err

        for subject in ("01", "02"):
            for run in ("1", "2") if subject == "01" else ("1",):
                _source_artifact(project, WorkUnit(subject, "01", "A", run))

        _run_cli(monkeypatch, *prepare)
        out = capsys.readouterr().out
        assert "--feature-type source_erp" in out
        assert f"--output-dir {project}/derivatives/features/{PIPELINE}/source_erp" in out
        assert "--subjects 01 02" in out
        assert "--condition A" in out
        # dry run wrote nothing
        assert not (project / "derivatives" / "features").exists()


class TestFinalizeTaskDirect:
    def test_force_lists_incomplete_units(self, bids_tree, tmp_path):
        project = bids_tree.parent
        features_config = tmp_path / "features.yaml"
        features_config.write_text("stage: features\n")

        with pytest.raises(RuntimeError, match="finalize refused"):
            finalize_task(
                project, "A", pipeline=PIPELINE, source_variant=VARIANT,
                subjects=["01"], features_config=features_config, dry_run=True,
            )

        record = finalize_task(
            project, "A", pipeline=PIPELINE, source_variant=VARIANT,
            subjects=["01"], features_config=features_config, dry_run=True,
            force=True,
        )
        assert record["subjects"] == ["01"]
        assert record["incomplete_units"] == [
            "sub-01_ses-01_task-A_run-1", "sub-01_ses-01_task-A_run-2",
        ]
        assert record["commands"][0][:5] == [
            sys.executable, "-m", "eegcpm.cli.main", "features", "--project",
        ]
        assert not (project / "derivatives" / "features").exists()

    def test_missing_features_config_is_loud(self, bids_tree, tmp_path):
        with pytest.raises(FileNotFoundError, match="features config"):
            finalize_task(
                bids_tree.parent, "A", pipeline=PIPELINE, source_variant=VARIANT,
                subjects=["01"], features_config=tmp_path / "nope.yaml",
                dry_run=True,
            )


# ------------------------------------------------- preprocess --state-db

def _state_db_args(config: Path, project: Path, state_db,
                   runs=None) -> argparse.Namespace:
    return argparse.Namespace(
        config=config,
        project=project,
        subjects=None,
        subject="01",
        task="T",
        task_config=None,
        pipeline="testpipe",
        eegcpm_root=None,
        force=False,
        parallel=None,
        verbose=False,
        sessions=None,
        runs=runs,
        state_db=state_db,
    )


def _preprocess_config(path: Path) -> Path:
    with open(path, "w") as f:
        yaml.safe_dump({
            "stage": "preprocessing",
            "name": "test",
            "steps": [{"name": "filter", "params": {"l_freq": 1.0, "h_freq": 40.0}}],
        }, f)
    return path


class TestPreprocessStateDb:
    def test_state_db_flag_overrides_default(self, tmp_path):
        project = tmp_path / "proj"
        (project / "bids" / "sub-01" / "ses-01" / "eeg").mkdir(parents=True)
        config = _preprocess_config(tmp_path / "config.yaml")
        custom_db = tmp_path / "hpc" / "state" / "sub-01_ses-01_task-T_run-1.db"

        # No input files: the run is marked FAILED (fast, no stage execution),
        # which is enough to prove the custom DB was the one written to.
        rc = preprocess_command(_state_db_args(config, project, custom_db))
        assert rc == 2
        assert custom_db.is_file(), "custom --state-db was not created"
        assert not (project / "eegcpm" / ".eegcpm" / "state.db").exists()

    def test_default_state_db_unchanged_without_flag(self, tmp_path):
        project = tmp_path / "proj"
        (project / "bids" / "sub-01" / "ses-01" / "eeg").mkdir(parents=True)
        config = _preprocess_config(tmp_path / "config.yaml")

        rc = preprocess_command(_state_db_args(config, project, None))
        assert rc == 2
        assert (project / "eegcpm" / ".eegcpm" / "state.db").is_file()
        assert not (tmp_path / "hpc").exists()

    def test_runs_filter_selects_only_the_unit_recording(self, tmp_path):
        project = tmp_path / "proj"
        eeg_dir = project / "bids" / "sub-01" / "ses-01" / "eeg"
        eeg_dir.mkdir(parents=True)
        rng = np.random.default_rng(0)
        info = mne.create_info([f"EEG{i:03d}" for i in range(1, 9)], 250.0, "eeg")
        for run in ("1", "2"):
            raw = mne.io.RawArray(rng.standard_normal((8, 2500)) * 1e-6, info)
            raw.save(eeg_dir / f"sub-01_ses-01_task-T_run-{run}_eeg.fif",
                     overwrite=True, verbose=False)
        config = _preprocess_config(tmp_path / "config.yaml")

        rc = preprocess_command(_state_db_args(
            config, project, tmp_path / "state.db", runs=["2"]))
        assert rc is None

        base = (project / "derivatives" / "preprocessing" / "testpipe"
                / "sub-01" / "ses-01" / "task-T")
        assert not (base / "run-1").exists(), "run-1 must not be processed"
        assert len(list((base / "run-2").glob("*_preprocessed_raw.fif"))) == 1

    def test_no_matching_run_fails_loudly(self, tmp_path):
        project = tmp_path / "proj"
        eeg_dir = project / "bids" / "sub-01" / "ses-01" / "eeg"
        eeg_dir.mkdir(parents=True)
        info = mne.create_info([f"EEG{i:03d}" for i in range(1, 9)], 250.0, "eeg")
        raw = mne.io.RawArray(np.zeros((8, 250)), info)
        raw.save(eeg_dir / "sub-01_ses-01_task-T_run-1_eeg.fif",
                 overwrite=True, verbose=False)
        config = _preprocess_config(tmp_path / "config.yaml")

        rc = preprocess_command(_state_db_args(
            config, project, tmp_path / "state.db", runs=["7"]))
        assert rc == 2
        assert list(project.rglob("*_preprocessed_raw.fif")) == []


# ------------------------------------------------------- batch config schema

class TestHpcBatchConfig:
    """``HpcBatchConfig`` is the ONE schema for the ``hpc submit --config`` YAML."""

    def _data(self, tmp_path: Path, **overrides) -> dict:
        data = {
            "slurm": {
                "partitions": [
                    {"name": "normal", "max_concurrent": 24, "cpus": 4,
                     "mem": "16G", "time": "02:00:00"},
                    {"name": "long", "max_concurrent": 8, "cpus": 4,
                     "mem": "24G", "time": "08:00:00"},
                ],
                "venv": "/opt/venv",
                "modules": ["python/3.11"],
                "exports": {"EEGCPM_HPC": "1"},
            },
            "project": tmp_path / "proj",
            "preprocessing_config": "configs/pre.yaml",
            "source_config": "configs/source.yaml",
            "features_config": "configs/features.yaml",
            "stages": ["preprocess", "epochs", "source"],
        }
        data.update(overrides)
        return data

    def test_valid_config_reuses_the_partition_schema(self, tmp_path):
        config = HpcBatchConfig(**self._data(tmp_path))
        project = tmp_path / "proj"

        assert [p.name for p in config.slurm.partitions] == ["normal", "long"]
        assert config.slurm.partitions[0].max_concurrent == 24
        assert config.slurm.partitions[1].mem == "24G"
        assert config.slurm.venv == "/opt/venv"
        assert config.slurm.modules == ["python/3.11"]
        assert config.slurm.exports == {"EEGCPM_HPC": "1"}
        assert config.stages == ["preprocess", "epochs", "source"]

        # Stage configs resolve against the project; the optional dirs default
        # relative to the project (absolute for the compute nodes).
        assert config.project == project
        assert config.preprocessing_config == project / "configs" / "pre.yaml"
        assert config.source_config == project / "configs" / "source.yaml"
        assert config.features_config == project / "configs" / "features.yaml"
        assert config.logs_dir == project / "hpc" / "logs"
        assert config.state_db_dir == project / "hpc" / "state"
        assert config.submissions_dir == project / "hpc" / "submissions"

    def test_explicit_dirs_are_kept(self, tmp_path):
        config = HpcBatchConfig(
            **self._data(tmp_path, logs_dir="/scratch/logs", state_db_dir="/scratch/state")
        )
        assert config.logs_dir == Path("/scratch/logs")
        assert config.state_db_dir == Path("/scratch/state")

    def test_unknown_key_is_a_hard_error(self, tmp_path):
        with pytest.raises(ValidationError):
            HpcBatchConfig(**self._data(tmp_path, partition="typo"))
        with pytest.raises(ValidationError):
            HpcBatchConfig(**self._data(
                tmp_path, slurm={"partitions": [], "partiton": "normal"}))
        with pytest.raises(ValidationError):
            HpcBatchConfig(**self._data(
                tmp_path,
                slurm={"partitions": [{"name": "normal", "max_coccurrent": 4}]},
            ))

    def test_stages_literal_enforced(self, tmp_path):
        with pytest.raises(ValidationError):
            HpcBatchConfig(**self._data(tmp_path, stages=["preprocess", "features"]))
        with pytest.raises(ValidationError):
            HpcBatchConfig(**self._data(tmp_path, stages=["source-reconstruct"]))
        config = HpcBatchConfig(**self._data(tmp_path, stages=["preprocess"]))
        assert config.stages == ["preprocess"]

    def test_partition_concurrency_must_be_positive(self, tmp_path):
        with pytest.raises(ValidationError):
            HpcBatchConfig(**self._data(
                tmp_path,
                slurm={"partitions": [{"name": "normal", "max_concurrent": 0}]},
            ))

    def test_from_yaml_resolves_relative_project_against_the_file(self, tmp_path):
        cfg_dir = tmp_path / "cluster"
        cfg_dir.mkdir()
        data = self._data(tmp_path)
        data["project"] = ".."  # relative to the YAML's directory
        data["preprocessing_config"] = "configs/pre.yaml"
        yaml_path = cfg_dir / "hpc.yaml"
        yaml_path.write_text(yaml.safe_dump(_strings(data)))

        config = HpcBatchConfig.from_yaml(yaml_path)
        assert config.project == tmp_path
        assert config.preprocessing_config == tmp_path / "configs" / "pre.yaml"

    def test_from_yaml_rejects_typos_and_bad_yaml(self, tmp_path):
        cfg = tmp_path / "hpc.yaml"
        cfg.write_text(yaml.safe_dump(_strings(self._data(tmp_path, typo_key=1))))
        with pytest.raises(ValidationError):
            HpcBatchConfig.from_yaml(cfg)

        cfg.write_text("slurm: [unclosed\n")
        with pytest.raises(ValueError):
            HpcBatchConfig.from_yaml(cfg)

        cfg.write_text("- just\n- a list\n")
        with pytest.raises(ValueError):
            HpcBatchConfig.from_yaml(cfg)

    def test_from_yaml_missing_file_is_loud(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            HpcBatchConfig.from_yaml(tmp_path / "nope.yaml")


def _strings(data: dict) -> dict:
    """YAML-safe copy of a config dict (``Path`` -> ``str``)."""
    return json.loads(json.dumps(data, default=str))


# ---------------------------------------------- generic command-script flavour

class TestGenerateCommandSlurmScript:
    """The one-off ``generate_command_slurm_script`` flavour (prediction page)."""

    def _script(self, **overrides) -> str:
        kwargs = dict(
            job_name="eegcpm_predict",
            command="python -m eegcpm.cli.main prediction\n",
            hpc={
                "partition": "shared_gpu_l40", "cpus": 8, "mem": "48G",
                "time": "04:00:00", "email": "user@eduhk.hk",
            },
            project_root="/proj",
            log_dir="/proj/logs",
            venv="/opt/venv/bin/activate",
            work_dir="/opt/eegcpm-0.1",
        )
        kwargs.update(overrides)
        return generate_command_slurm_script(**kwargs)

    def test_header_fields_and_env_setup(self):
        script = self._script()

        assert script.startswith("#!/bin/bash\n#SBATCH --job-name=eegcpm_predict\n")
        assert "#SBATCH --partition=shared_gpu_l40" in script
        assert "#SBATCH --nodes=1" in script
        assert "#SBATCH --ntasks-per-node=1" in script
        assert "#SBATCH --cpus-per-task=8" in script
        assert "#SBATCH --mem=48G" in script
        assert "#SBATCH --time=04:00:00" in script
        assert "#SBATCH --output=/proj/logs/%A.out" in script
        assert "#SBATCH --error=/proj/logs/%A.err" in script
        assert "#SBATCH --mail-type=END,FAIL" in script
        assert "#SBATCH --mail-user=user@eduhk.hk" in script
        assert 'set -e' in script
        assert 'mkdir -p "/proj/logs"' in script
        assert 'source "/opt/venv/bin/activate"' in script
        assert 'cd "/opt/eegcpm-0.1"' in script
        # the command is embedded verbatim, with the page-style trailing blank
        assert script.endswith("python -m eegcpm.cli.main prediction\n\n")

    def test_no_array_markers_without_an_array(self):
        script = self._script()
        assert "#SBATCH --array" not in script
        assert "%a" not in script
        assert "%A" in script

        array_script = generate_array_script(
            job_name="eegcpm_pp_normal", partition="normal",
            units_chunk_path="/proj/hpc/chunk.tsv", n_units=3, concurrency=2,
            cpus=4, mem="16G", time="02:00:00", log_dir="/proj/hpc/logs",
            project="/proj", stages=["preprocess"], pipeline=PIPELINE,
            config_paths={"preprocessing": "/proj/pre.yaml"},
        )
        assert "#SBATCH --array=0-2%2" in array_script
        assert "%A_%a.out" in array_script

    def test_hpc_defaults_and_email_placeholder(self):
        script = self._script(hpc={})
        assert "#SBATCH --partition=shared_cpu" in script
        assert "#SBATCH --cpus-per-task=4" in script
        assert "#SBATCH --mem=16G" in script
        assert "#SBATCH --time=02:00:00" in script
        assert "# #SBATCH --mail-user=your_email@eduhk.hk" in script
        assert "#SBATCH --mail-user=" not in script.split("#SBATCH --mail-type")[1].split("\n")[0]

    def test_explicit_arguments_override_the_hpc_mapping(self):
        script = self._script(
            hpc={"partition": "a", "cpus": 1, "mem": "1G", "time": "00:01:00"},
            partition="b", cpus=2, mem="2G", time="00:02:00",
        )
        assert "#SBATCH --partition=b" in script
        assert "#SBATCH --partition=a" not in script
        assert "#SBATCH --cpus-per-task=2" in script
        assert "#SBATCH --mem=2G" in script
        assert "#SBATCH --time=00:02:00" in script

    def test_modules_and_missing_venv_and_workdir(self):
        script = self._script(hpc={"modules": ["python/3.11"]}, venv=None, work_dir=None)
        assert "module load python/3.11" in script
        assert "source " not in script
        assert "cd " not in script

    def test_empty_command_is_loud(self):
        with pytest.raises(ValueError):
            self._script(command="   ")


# ------------------------------------------------- prediction page delegation

_PAGE12_PATH = (
    Path(__file__).parent.parent.parent
    / "eegcpm" / "ui" / "pages" / "12_prediction.py"
)


def _load_page12_functions() -> dict:
    """Extract the page's pure script functions (the page needs real streamlit)."""
    source = _PAGE12_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"_make_local_script", "_make_slurm_script"}
    segments = [
        ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert len(segments) == len(wanted)
    namespace = {"json": json, "generate_command_slurm_script": generate_command_slurm_script}
    exec("\n\n".join(segments), namespace)
    return namespace


class TestPredictionPageDelegation:
    """Page 12 must delegate to the unified generator (no inline #SBATCH)."""

    CONFIG = {
        "project_root": "/local/project",
        "behavior_path": "/local/behavior.csv",
        "subject_column": "subject",
        "target_name": "asrs_inattention",
        "pipeline": "optimized",
        "subjects": ["001", "002"],
        "tasks": ["cptvis"], "variants": ["a"], "conditions": ["c"],
        "windows": ["w"], "methods": ["m"], "bands": ["b"], "statistics": ["s"],
        "clip_negative_dwpli": True, "models": ["ridge"],
        "cv_method": "kfold", "n_splits": 5, "n_repeats": 1, "random_state": 0,
        "hpc_behavior_path": "/share/behavior.csv",
    }
    HPC = {
        "project_root": "/share/ps_clivewong/25SusAttn",
        "eegcpm_root": "/share/ps_clivewong/eegcpm-dev/eegcpm-0.1",
        "venv": "/share/ps_clivewong/eegcpm-dev/eegcpm-0.1/venv/bin/activate",
        "email": "user@eduhk.hk",
        "partition": "shared_cpu", "time": "02:00:00", "mem": "16G", "cpus": 4,
    }

    def test_output_is_pinned_to_the_recorded_golden(self):
        """Page-12's script is byte-pinned to the pre-refactor golden.

        ``tests/fixtures/prediction_slurm_golden.sh`` was captured from the
        pre-refactor inline ``_make_slurm_script`` (HEAD ``b979e7b``) and is
        byte-identical to the unified generator's output for this exact config;
        the header pin and the delegation test alone would not catch a
        body regression (2026-10-17 HPC consolidation validation).
        """
        functions = _load_page12_functions()
        script = functions["_make_slurm_script"](dict(self.CONFIG), dict(self.HPC))
        golden = (
            Path(__file__).parent.parent / "fixtures" / "prediction_slurm_golden.sh"
        ).read_text(encoding="utf-8")
        assert script == golden

    def test_no_sbatch_header_in_the_page_functions(self):
        source = _PAGE12_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "_make_slurm_script":
                assert "#SBATCH" not in ast.get_source_segment(source, node)

    def test_produced_script_equals_the_generator_output(self):
        functions = _load_page12_functions()
        script = functions["_make_slurm_script"](self.CONFIG, dict(self.HPC))

        local_config = dict(self.CONFIG)
        local_config["project_root"] = self.HPC["project_root"]
        local_config["behavior_path"] = self.CONFIG["hpc_behavior_path"]
        expected = generate_command_slurm_script(
            "eegcpm_predict",
            functions["_make_local_script"](local_config),
            hpc=self.HPC,
            project_root=self.HPC["project_root"],
            log_dir=f"{self.HPC['project_root']}/logs",
            venv=self.HPC["venv"],
            work_dir=self.HPC["eegcpm_root"],
            output_pattern="prediction_%A.out",
            error_pattern="prediction_%A.err",
        )
        assert script == expected

    def test_hpc_paths_partition_and_behavior_substitution(self):
        functions = _load_page12_functions()
        script = functions["_make_slurm_script"](self.CONFIG, dict(self.HPC))

        assert "#SBATCH --job-name=eegcpm_predict" in script
        assert "#SBATCH --partition=shared_cpu" in script
        assert "#SBATCH --output=/share/ps_clivewong/25SusAttn/logs/prediction_%A.out" in script
        assert "#SBATCH --mail-user=user@eduhk.hk" in script
        assert 'cd "/share/ps_clivewong/eegcpm-dev/eegcpm-0.1"' in script
        assert "#SBATCH --array" not in script
        # The embedded run uses the HPC behavior path, never the local one.
        assert '"/share/behavior.csv"' in script
        assert "/local/behavior.csv" not in script

    def test_placeholder_venv_when_hpc_root_unset(self):
        functions = _load_page12_functions()
        script = functions["_make_slurm_script"](self.CONFIG, {})
        assert "#SBATCH --partition=shared_cpu" in script
        assert "# #SBATCH --mail-user=your_email@eduhk.hk" in script
        assert 'cd "/path/to/eegcpm-0.1"' in script
        assert 'source "/path/to/eegcpm-0.1/venv/bin/activate"' in script


# ------------------------------------------------------ CLI via batch config

class TestHpcCliBatchConfig:
    """``hpc ... --config <hpc.yaml>``: validated defaults, flags override."""

    def _config_file(
        self, tmp_path: Path, project: Path, *, stages=None, extra=None,
        partitions=None,
    ) -> Path:
        data = {
            "slurm": {
                "partitions": partitions or [
                    {"name": "normal", "max_concurrent": 2, "cpus": 4,
                     "mem": "16G", "time": "02:00:00"},
                    {"name": "long", "max_concurrent": 1, "cpus": 8,
                     "mem": "48G", "time": "08:00:00"},
                ],
                "venv": str(tmp_path / "venv"),
                "modules": ["python/3.11"],
                "exports": {"EEGCPM_HPC": "1"},
            },
            "project": str(project),
            "preprocessing_config": str(tmp_path / "pre.yaml"),
            "source_config": str(tmp_path / "source.yaml"),
            "features_config": str(tmp_path / "features.yaml"),
            "stages": stages or ["preprocess", "epochs", "source"],
            "logs_dir": str(project / "hpc" / "logs"),
        }
        if extra:
            data.update(extra)
        path = tmp_path / "hpc.yaml"
        path.write_text(yaml.safe_dump(data))
        return path

    def test_submit_config_drives_partitions_and_resources(
        self, bids_tree, tmp_path, monkeypatch, capsys
    ):
        project = bids_tree.parent
        write_manifest(EXPECTED_UNITS, project / "hpc" / "manifest.tsv")
        config = self._config_file(tmp_path, project)
        monkeypatch.setattr(hpc_cli.shutil, "which", _boom)
        monkeypatch.setattr(hpc_cli.subprocess, "run", _boom)

        _run_cli(
            monkeypatch, "hpc", "submit",
            "--config", str(config),
            "--manifest", "hpc/manifest.tsv",
            "--pipeline", PIPELINE,
            "--source-variant", VARIANT,
            "--dry-run",
        )
        out = capsys.readouterr().out
        assert "normal: 3 unit(s)" in out
        assert "(array 0-2%2, cpus=4, mem=16G, time=02:00:00)" in out
        assert "(array 0-2%1, cpus=8, mem=48G, time=08:00:00)" in out
        assert "sbatch " in out

        submission_dir = next((project / "hpc" / "submissions").iterdir())
        normal = (submission_dir / "eegcpm_normal.sbatch").read_text()
        long = (submission_dir / "eegcpm_long.sbatch").read_text()

        assert "#SBATCH --array=0-2%2" in normal
        assert "#SBATCH --partition=normal" in normal
        assert "#SBATCH --cpus-per-task=4" in normal
        assert "#SBATCH --mem=16G" in normal
        assert "#SBATCH --time=02:00:00" in normal
        assert "#SBATCH --output=" + str(project / "hpc" / "logs") in normal
        assert f"source {tmp_path / 'venv'}/bin/activate" in normal
        assert "module load python/3.11" in normal
        assert "export EEGCPM_HPC=1" in normal
        assert f"--preprocessing-config {tmp_path / 'pre.yaml'}" in normal
        assert f"--source-config {tmp_path / 'source.yaml'}" in normal
        assert "--stages preprocess,epochs,source" in normal
        assert f"--state-db-dir {project}/hpc/state" in normal

        assert "#SBATCH --array=0-2%1" in long
        assert "#SBATCH --cpus-per-task=8" in long
        assert "#SBATCH --mem=48G" in long
        assert "#SBATCH --time=08:00:00" in long

    def test_cli_flags_override_the_config(self, bids_tree, tmp_path, monkeypatch, capsys):
        project = bids_tree.parent
        write_manifest(EXPECTED_UNITS, project / "hpc" / "manifest.tsv")
        config = self._config_file(tmp_path, project)
        monkeypatch.setattr(hpc_cli.shutil, "which", _boom)
        monkeypatch.setattr(hpc_cli.subprocess, "run", _boom)

        _run_cli(
            monkeypatch, "hpc", "submit",
            "--config", str(config),
            "--manifest", "hpc/manifest.tsv",
            "--pipeline", PIPELINE,
            "--source-variant", VARIANT,
            "--partitions", "normal",
            "--concurrency-per-partition", "5",
            "--cpus", "1", "--mem", "1G", "--time", "00:10:00",
            "--dry-run",
        )
        capsys.readouterr()
        submission_dir = next((project / "hpc" / "submissions").iterdir())
        script = (submission_dir / "eegcpm_normal.sbatch").read_text()
        assert "#SBATCH --array=0-5%5" in script
        assert "#SBATCH --cpus-per-task=1" in script
        assert "#SBATCH --mem=1G" in script
        assert "#SBATCH --time=00:10:00" in script

    def test_unknown_config_key_is_loud(self, bids_tree, tmp_path, monkeypatch, capsys):
        project = bids_tree.parent
        config = self._config_file(tmp_path, project, extra={"partitons": ["normal"]})

        with pytest.raises(SystemExit) as exc:
            _run_cli(
                monkeypatch, "hpc", "submit",
                "--config", str(config),
                "--manifest", "hpc/manifest.tsv",
                "--pipeline", PIPELINE,
                "--dry-run",
            )
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "hpc submit:" in err
        assert "partitons" in err

    def test_stage_config_and_stages_come_from_config(
        self, bids_tree, tmp_path, monkeypatch, capsys
    ):
        project = bids_tree.parent
        write_manifest(EXPECTED_UNITS, project / "hpc" / "manifest.tsv")
        config = self._config_file(tmp_path, project, stages=["preprocess"])

        # prune: stages (and project) from the config
        _run_cli(
            monkeypatch, "hpc", "prune",
            "--config", str(config),
            "--manifest", "hpc/manifest.tsv",
            "--pipeline", PIPELINE,
            "--out", "hpc/pending.tsv",
        )
        assert "for [preprocess]" in capsys.readouterr().out
        assert read_manifest(project / "hpc" / "pending.tsv") == EXPECTED_UNITS

        # submit: no --stages/--preprocessing-config flag needed
        monkeypatch.setattr(hpc_cli.shutil, "which", _boom)
        monkeypatch.setattr(hpc_cli.subprocess, "run", _boom)
        _run_cli(
            monkeypatch, "hpc", "submit",
            "--config", str(config),
            "--manifest", "hpc/manifest.tsv",
            "--pipeline", PIPELINE,
            "--dry-run",
        )
        capsys.readouterr()
        submission_dir = next((project / "hpc" / "submissions").iterdir())
        script = (submission_dir / "eegcpm_normal.sbatch").read_text()
        assert "--stages preprocess" in script
        assert "--preprocessing-config" in script
        assert "--source-config" not in script

    def test_missing_project_is_loud(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_cli(monkeypatch, "hpc", "status")
        assert exc.value.code == 2
        assert "--project is required" in capsys.readouterr().err

    def test_status_uses_config_logs_dir(self, tmp_path, monkeypatch, capsys):
        project = tmp_path / "proj"
        units_dir = project / "hpc" / "logs" / ".." / "units"
        units_dir.mkdir(parents=True)
        (units_dir / "sub-01_ses-01_task-A_run-1.json").write_text(json.dumps({
            "unit": "sub-01_ses-01_task-A_run-1",
            "stages": {"preprocess": {"status": "completed", "returncode": 0}},
        }))
        config = self._config_file(tmp_path, project)

        _run_cli(monkeypatch, "hpc", "status", "--config", str(config))
        out = capsys.readouterr().out
        assert "sub-01_ses-01_task-A_run-1" in out

    def test_finalize_uses_config_features_config(
        self, bids_tree, tmp_path, monkeypatch, capsys
    ):
        project = bids_tree.parent
        config = self._config_file(tmp_path, project)
        (tmp_path / "features.yaml").write_text("stage: features\n")

        with pytest.raises(SystemExit) as exc:
            _run_cli(
                monkeypatch, "hpc", "finalize",
                "--config", str(config),
                "--task", "A", "--pipeline", PIPELINE, "--source-variant", VARIANT,
                "--subjects", "01",
                "--dry-run",
            )
        assert exc.value.code == 1
        assert "finalize refused" in capsys.readouterr().err

        for run in ("1", "2"):
            _source_artifact(project, WorkUnit("01", "01", "A", run))

        _run_cli(
            monkeypatch, "hpc", "finalize",
            "--config", str(config),
            "--task", "A", "--pipeline", PIPELINE, "--source-variant", VARIANT,
            "--subjects", "01",
            "--dry-run",
        )
        out = capsys.readouterr().out
        assert f"--config {tmp_path / 'features.yaml'}" in out


    def test_project_flag_rebases_config_default_dirs(
        self, bids_tree, tmp_path, monkeypatch, capsys
    ):
        """--project overrides the config's project and its default dirs."""
        project = bids_tree.parent
        other = tmp_path / "other"
        other.mkdir()
        write_manifest(EXPECTED_UNITS, project / "hpc" / "manifest.tsv")
        config = self._config_file(tmp_path, other, stages=["preprocess"],
                                   extra={"logs_dir": str(other / "hpc" / "logs")})
        monkeypatch.setattr(hpc_cli.shutil, "which", _boom)
        monkeypatch.setattr(hpc_cli.subprocess, "run", _boom)

        _run_cli(
            monkeypatch, "hpc", "submit",
            "--config", str(config),
            "--project", str(project),
            "--manifest", "hpc/manifest.tsv",
            "--pipeline", PIPELINE,
            "--dry-run",
        )
        capsys.readouterr()
        submission_dir = next((project / "hpc" / "submissions").iterdir())
        script = (submission_dir / "eegcpm_normal.sbatch").read_text()
        assert f"--state-db-dir {project}/hpc/state" in script
        assert not (other / "hpc" / "submissions").exists()



# --------------------------------------------------- SlurmSubmitter (repaired)

class TestSlurmSubmitter:
    """The legacy single-job class now submits through the current CLI."""

    def _submitter(self, tmp_path: Path) -> SlurmSubmitter:
        return SlurmSubmitter(
            SlurmConfig(partition="normal", cpus_per_task=2, mem="8G", time="01:00:00"),
            log_dir=tmp_path / "logs",
        )

    def test_submit_subject_jobs_uses_the_current_cli(self, bids_tree, tmp_path):
        project_dir = bids_tree.parent
        pre = tmp_path / "pre.yaml"
        pre.write_text("stage: preprocessing\n")
        project = Project(
            name="t", root_path=project_dir,
            subjects=[Subject(id="01"), Subject(id="02"), Subject(id="03")],
        )
        submitter = self._submitter(tmp_path)

        jobs = submitter.submit_subject_jobs(
            project, pre, pipeline=PIPELINE, dry_run=True,
            submission_dir=tmp_path / "sub",
        )
        assert jobs == []
        assert sorted(p.name for p in (tmp_path / "sub").glob("*.sbatch")) == [
            "eegcpm_01.sbatch", "eegcpm_02.sbatch", "eegcpm_03.sbatch",
        ]
        assert read_manifest(tmp_path / "sub" / "eegcpm_01.tsv") == [
            unit for unit in EXPECTED_UNITS if unit.subject == "01"
        ]
        script = (tmp_path / "sub" / "eegcpm_01.sbatch").read_text()
        # 3 units for sub-01 -> array over them
        assert "#SBATCH --array=0-2%3" in script
        assert "#SBATCH --partition=normal" in script
        assert "#SBATCH --cpus-per-task=2" in script
        assert "-m eegcpm.cli.main hpc run-unit" in script
        assert f"--preprocessing-config {pre}" in script
        # the dead `eegcpm.cli run` surface must never come back
        assert "cli run" not in script
        assert "--state-db-dir" in script

    def test_submit_subject_jobs_skips_subjects_without_bids(self, bids_tree, tmp_path, caplog):
        project = Project(
            name="t", root_path=bids_tree.parent,
            subjects=[Subject(id="01"), Subject(id="99")],
        )
        pre = tmp_path / "pre.yaml"
        pre.write_text("stage: preprocessing\n")
        submitter = self._submitter(tmp_path)

        with caplog.at_level("WARNING", logger="eegcpm.pipeline.slurm"):
            submitter.submit_subject_jobs(
                project, pre, dry_run=True, submission_dir=tmp_path / "sub")
        assert "no BIDS input files for subject 99" in caplog.text
        assert [p.name for p in (tmp_path / "sub").glob("*.sbatch")] == [
            "eegcpm_01.sbatch"]

    def test_submit_subject_jobs_filters_by_subject_ids(self, bids_tree, tmp_path):
        project = Project(
            name="t", root_path=bids_tree.parent,
            subjects=[Subject(id="01"), Subject(id="02"), Subject(id="03")],
        )
        pre = tmp_path / "pre.yaml"
        pre.write_text("stage: preprocessing\n")
        submitter = self._submitter(tmp_path)

        submitter.submit_subject_jobs(
            project, pre, ["sub-03"],
            dry_run=True, submission_dir=tmp_path / "sub",
        )
        assert [p.name for p in (tmp_path / "sub").glob("*.sbatch")] == [
            "eegcpm_03.sbatch"]

    def test_submit_array_calls_sbatch_once(self, tmp_path, monkeypatch):
        calls = []

        class _Result:
            returncode = 0
            stdout = "Submitted batch job 4242\n"
            stderr = ""

        def fake_run(argv, **kwargs):
            calls.append(argv)
            return _Result()

        monkeypatch.setattr(pipeline_slurm.subprocess, "run", fake_run)
        submitter = self._submitter(tmp_path)
        script = tmp_path / "job.sbatch"
        script.write_text("#!/bin/bash\n")

        job = submitter.submit_array(script)
        assert calls == [["sbatch", str(script)]]
        assert job is not None and job.job_id == "4242"
        assert submitter.jobs["4242"].name == "job"

    def test_submit_array_dry_run_and_missing_file(self, tmp_path):
        submitter = self._submitter(tmp_path)
        script = tmp_path / "job.sbatch"
        script.write_text("#!/bin/bash\n")
        assert submitter.submit_array(script, dry_run=True) is None
        with pytest.raises(FileNotFoundError):
            submitter.submit_array(tmp_path / "nope.sbatch")


class TestHpcCliHelp:
    """Every ``hpc`` subcommand's --help must render (percent signs escaped)."""

    @pytest.mark.parametrize(
        "subcommand", ["manifest", "prune", "submit", "run-unit", "status", "finalize"]
    )
    def test_subcommand_help_renders(self, monkeypatch, capsys, subcommand):
        with pytest.raises(SystemExit) as exc:
            _run_cli(monkeypatch, "hpc", subcommand, "--help")
        assert exc.value.code == 0
        assert "usage:" in capsys.readouterr().out

    def test_hpc_help_renders(self, monkeypatch, capsys):
        with pytest.raises(SystemExit) as exc:
            _run_cli(monkeypatch, "hpc", "--help")
        assert exc.value.code == 0
        assert "manifest" in capsys.readouterr().out
