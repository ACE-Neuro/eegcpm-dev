"""Run-status persistence for failed/excluded recordings (B1 — C1).

Both preprocessing entry points (``preprocess_command`` and
``RunProcessor``) must persist ``FAILED`` / ``EXCLUDED`` for the exact
subject/session/task/run identity and must never leave a
``*_preprocessed_raw.fif`` for a run that failed or was excluded.

B1 rev1 additions: a successful run with an optional-step failure still
persists COMPLETED (N1); a QC failure never flips a completed run (N4);
an exception after ``process()`` marks that run FAILED, deletes its
outputs and rewrites its success sidecar (N4); a forced re-run that
fails leaves no stale FIF (N3); and ``processing_state.json`` records
the final status, error and ``output_files['preprocessed'] is None``
(K7).
"""

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pytest
import yaml

from eegcpm.cli.preprocess import preprocess_command
from eegcpm.modules.preprocessing.steps import STEP_REGISTRY, ProcessingStep
from eegcpm.modules.preprocessing.steps.base import RecordingExcluded
from eegcpm.pipeline.run_processor import RunProcessor
from eegcpm.workflow.state import ProcessingStatus, WorkflowStateManager

PIPELINE = "testpipe"


class InjectedFailureStep(ProcessingStep):
    name = "injected_failure"
    version = "0.1"

    def process(self, raw, metadata):
        raise RuntimeError("injected_failure: boom")


class InjectedExclusionStep(ProcessingStep):
    name = "injected_exclusion"
    version = "0.1"

    def process(self, raw, metadata):
        raise RecordingExcluded("injected_exclusion_reason", {"n_bad": 7})


class PoisonBadChannelsStep(ProcessingStep):
    """Succeeds, but leaves a non-iterable `bad_channels` entry.

    Registered under the real ``bad_channels`` step name (per-test
    monkeypatch), because the CLI iterates
    ``result.metadata['bad_channels'].get('bad_channels', [])`` after
    ``process()`` returns; the malformed value raises there (outside the
    QC block) and exercises the per-run exception safety net (N4).
    """
    name = "bad_channels"
    version = "0.1"

    def process(self, raw, metadata):
        return raw, {"applied": True, "bad_channels": 42}


@pytest.fixture
def registered_steps(monkeypatch):
    """Register throwaway failing/excluding/poison steps for this test only."""
    monkeypatch.setitem(STEP_REGISTRY, "injected_failure", InjectedFailureStep)
    monkeypatch.setitem(STEP_REGISTRY, "injected_exclusion", InjectedExclusionStep)
    monkeypatch.setitem(STEP_REGISTRY, "bad_channels", PoisonBadChannelsStep)


@pytest.fixture
def bids_tree(tmp_path):
    """Minimal BIDS tree with one saved synthetic run."""
    eeg_dir = tmp_path / "sub-01" / "ses-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    info = mne.create_info([f"EEG{i:03d}" for i in range(1, 9)], 250.0, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((8, 2500)) * 1e-6, info)
    raw.save(eeg_dir / "sub-01_ses-01_task-T_run-1_eeg.fif",
             overwrite=True, verbose=False)
    return tmp_path


def _write_config(path: Path, steps) -> Path:
    with open(path, "w") as f:
        yaml.safe_dump({
            "stage": "preprocessing",
            "name": "test",
            "steps": [
                {"name": "filter", "params": {"l_freq": 1.0, "h_freq": 40.0}},
                *steps,
            ],
        }, f)
    return path


def _config_dict(steps):
    return {
        "stage": "preprocessing",
        "name": "test",
        "steps": [
            {"name": "filter", "params": {"l_freq": 1.0, "h_freq": 40.0}},
            *steps,
        ],
    }


def _cli_args(config: Path, project: Path) -> argparse.Namespace:
    """Mirror the `preprocess` subparser fields from cli/main.py."""
    return argparse.Namespace(
        config=config,
        project=project,
        subjects=None,
        subject="01",
        task="T",
        task_config=None,
        pipeline=PIPELINE,
        eegcpm_root=None,
        force=False,
        parallel=None,
        verbose=False,
    )


def _run_dir(project: Path) -> Path:
    return (project / "derivatives" / "preprocessing" / PIPELINE
            / "sub-01" / "ses-01" / "task-T" / "run-1")


def _state(project: Path, db_root: Path):
    manager = WorkflowStateManager(db_root / ".eegcpm" / "state.db")
    return manager, manager.load_state("01", "T", PIPELINE,
                                       session="01", run="1")


def _no_fif(project: Path):
    assert list(project.rglob("*_preprocessed_raw.fif")) == []


def _processing_state(project: Path) -> dict:
    return json.loads((_run_dir(project) / "processing_state.json").read_text())


# --------------------------------------------------------------- CLI path

def test_cli_persists_failed_status(bids_tree, registered_steps, tmp_path):
    config = _write_config(tmp_path / "fail.yaml",
                           [{"name": "injected_failure", "params": {}}])
    rc = preprocess_command(_cli_args(config, bids_tree))
    assert rc == 2, "a batch with a failed run must exit non-zero"

    manager, state = _state(bids_tree, bids_tree / "eegcpm")
    assert state is not None, "no workflow row for the failed run"
    assert state.status == ProcessingStatus.FAILED
    assert state.status.value == "failed"
    _no_fif(bids_tree)
    assert list(_run_dir(bids_tree).glob("*_preprocessing_FAILED.json")), \
        "missing FAILED sidecar"


def test_cli_persists_excluded_status(bids_tree, registered_steps, tmp_path):
    config = _write_config(tmp_path / "exclude.yaml",
                           [{"name": "injected_exclusion", "params": {}}])
    rc = preprocess_command(_cli_args(config, bids_tree))
    assert rc is None, "exclusions are a valid outcome, not a batch failure"

    manager, state = _state(bids_tree, bids_tree / "eegcpm")
    assert state is not None, "no workflow row for the excluded run"
    assert state.status == ProcessingStatus.EXCLUDED
    assert state.status.value == "excluded"
    _no_fif(bids_tree)
    assert list(_run_dir(bids_tree).glob("*_preprocessing_EXCLUDED.json")), \
        "missing EXCLUDED sidecar"


def test_cli_invalid_config_returns_non_zero(bids_tree, tmp_path):
    """A legacy (flat) dialect is a hard CLI error, not a silent no-op."""
    bad = tmp_path / "flat.yaml"
    bad.write_text(yaml.safe_dump({
        "steps": [{"name": "filter", "l_freq": 1.0}],
    }))
    assert preprocess_command(_cli_args(bad, bids_tree)) not in (0, None)
    # No state row was written for the invalid config.
    manager = WorkflowStateManager(bids_tree / "eegcpm" / ".eegcpm" / "state.db")
    assert manager.get_all_states() == []


def test_cli_skips_completed_run_by_session_and_run(
        bids_tree, registered_steps, tmp_path):
    """The skip check must match the stored session/run identity.

    A COMPLETED row for (subject, session='01', run='1') is matched, so
    the run is never reprocessed (the old subject-level check compared
    session=None/run=None and never matched)."""
    from datetime import datetime
    from eegcpm.workflow.state import WorkflowState

    manager = WorkflowStateManager(bids_tree / "eegcpm" / ".eegcpm" / "state.db")
    manager.save_state(WorkflowState(
        subject_id="01", task="T", pipeline=PIPELINE,
        status=ProcessingStatus.COMPLETED,
        session="01", run="1", created_at=datetime.now(),
    ))

    config = _write_config(tmp_path / "fail.yaml",
                           [{"name": "injected_failure", "params": {}}])
    preprocess_command(_cli_args(config, bids_tree))

    _, state = _state(bids_tree, bids_tree / "eegcpm")
    assert state.status == ProcessingStatus.COMPLETED  # untouched
    assert not _run_dir(bids_tree).exists(), "the run was not skipped"


def test_cli_forced_rerun_failure_clears_stale_outputs(
        bids_tree, registered_steps, tmp_path):
    """A forced re-run that fails must leave no stale FIF (N3).

    The first run succeeds (FIF + success sidecar). The forced second
    run fails: its outputs are cleared *before* processing, so exactly
    one FAILED sidecar and no FIF remain — never both sidecars."""
    good = _write_config(tmp_path / "good.yaml", [])
    assert preprocess_command(_cli_args(good, bids_tree)) is None

    run_dir = _run_dir(bids_tree)
    assert len(list(run_dir.glob("*_preprocessed_raw.fif"))) == 1
    assert len(list(run_dir.glob("*_preprocessing.json"))) == 1

    bad = _write_config(tmp_path / "bad.yaml",
                        [{"name": "injected_failure", "params": {}}])
    args = _cli_args(bad, bids_tree)
    args.force = True
    assert preprocess_command(args) == 2

    assert list(run_dir.glob("*_preprocessed_raw.fif")) == []
    assert list(run_dir.glob("*_preprocessing.json")) == []
    failed = list(run_dir.glob("*_preprocessing_FAILED.json"))
    assert len(failed) == 1, f"expected exactly one FAILED sidecar, got {failed}"
    assert json.loads(failed[0].read_text())["success"] is False


def test_cli_qc_failure_keeps_completed(bids_tree, monkeypatch, tmp_path):
    """A QC exception is recorded as `qc_error`, warns, and the run
    stays COMPLETED with its FIF intact (N4)."""
    import eegcpm.cli.preprocess as cli_preprocess

    class _BoomQC:
        def __init__(self, *args, **kwargs):
            pass

        def compute(self, *args, **kwargs):
            raise RuntimeError("injected QC failure")

    monkeypatch.setattr(cli_preprocess, "PreprocessedQC", _BoomQC)

    config = _write_config(tmp_path / "filter.yaml", [])
    rc = preprocess_command(_cli_args(config, bids_tree))
    assert rc is None, "a QC failure is not a batch failure"

    _, state = _state(bids_tree, bids_tree / "eegcpm")
    assert state.status == ProcessingStatus.COMPLETED
    step = state.get_step("preprocessing_run1")
    assert step is not None and step.metadata, "preprocessing step not persisted"
    assert "injected QC failure" in step.metadata["qc_error"]
    assert len(list(_run_dir(bids_tree).glob("*_preprocessed_raw.fif"))) == 1


def test_cli_exception_after_process_fails_the_run(
        bids_tree, registered_steps, tmp_path):
    """An exception raised after `process()` succeeded must mark *that*
    run FAILED, delete the FIF and rewrite the success sidecar as
    FAILED (N4)."""
    config = _write_config(
        tmp_path / "poison.yaml",
        [{"name": "bad_channels", "params": {}}])
    rc = preprocess_command(_cli_args(config, bids_tree))
    assert rc == 2

    _, state = _state(bids_tree, bids_tree / "eegcpm")
    assert state.status == ProcessingStatus.FAILED

    run_dir = _run_dir(bids_tree)
    assert list(run_dir.glob("*_preprocessed_raw.fif")) == []
    assert list(run_dir.glob("*_preprocessing.json")) == []
    failed = list(run_dir.glob("*_preprocessing_FAILED.json"))
    assert len(failed) == 1, f"expected exactly one FAILED sidecar, got {failed}"

    payload = json.loads(failed[0].read_text())
    assert payload["success"] is False
    assert any("not iterable" in e or "missing" in e
               for e in payload["errors"]), payload["errors"]


# --------------------------------------------------- run_processor path

def test_run_processor_persists_failed_status(
        bids_tree, registered_steps, tmp_path):
    db_root = tmp_path / "rp_state"
    manager = WorkflowStateManager(db_root / ".eegcpm" / "state.db")
    processor = RunProcessor(
        bids_root=bids_tree,
        output_root=bids_tree / "derivatives" / "preprocessing" / PIPELINE,
        config=_config_dict([{"name": "injected_failure", "params": {}}]),
        state_manager=manager,
        verbose=False,
    )
    results = processor.process_subject_task(
        subject_id="01", task="T", session="01", pipeline=PIPELINE)

    assert len(results) == 1
    assert results[0].success is False
    assert manager.load_state("01", "T", PIPELINE,
                              session="01", run="1").status == ProcessingStatus.FAILED
    _no_fif(bids_tree)
    assert list(_run_dir(bids_tree).glob("*_preprocessing_FAILED.json")), \
        "missing FAILED sidecar"

    # K7: the filesystem state agrees with the DB and never claims an
    # output for a failed run.
    pstate = _processing_state(bids_tree)
    assert pstate["status"] == "failed"
    assert pstate["error"]
    assert pstate["completed_at"] is not None
    assert pstate["output_files"]["preprocessed"] is None


def test_run_processor_persists_excluded_status(
        bids_tree, registered_steps, tmp_path):
    db_root = tmp_path / "rp_state"
    manager = WorkflowStateManager(db_root / ".eegcpm" / "state.db")
    processor = RunProcessor(
        bids_root=bids_tree,
        output_root=bids_tree / "derivatives" / "preprocessing" / PIPELINE,
        config=_config_dict([{"name": "injected_exclusion", "params": {}}]),
        state_manager=manager,
        verbose=False,
    )
    results = processor.process_subject_task(
        subject_id="01", task="T", session="01", pipeline=PIPELINE)

    assert len(results) == 1
    assert results[0].success is False
    state = manager.load_state("01", "T", PIPELINE, session="01", run="1")
    assert state.status == ProcessingStatus.EXCLUDED
    assert state.status.value == "excluded"
    _no_fif(bids_tree)
    assert list(_run_dir(bids_tree).glob("*_preprocessing_EXCLUDED.json")), \
        "missing EXCLUDED sidecar"

    # K7: an exclusion is final too — status, error and completed_at are
    # all persisted, and no output file is claimed.
    pstate = _processing_state(bids_tree)
    assert pstate["status"] == "excluded"
    assert "excluded" in pstate["error"]
    assert pstate["completed_at"] is not None
    assert pstate["output_files"]["preprocessed"] is None


def test_run_processor_optional_failure_still_completes(
        bids_tree, registered_steps, tmp_path):
    """A successful run with an optional-step failure persists COMPLETED
    (N1): the metadata log loop must not crash on the top-level
    ``degraded`` flag and write FAILED next to a real FIF."""
    db_root = tmp_path / "rp_state"
    manager = WorkflowStateManager(db_root / ".eegcpm" / "state.db")
    processor = RunProcessor(
        bids_root=bids_tree,
        output_root=bids_tree / "derivatives" / "preprocessing" / PIPELINE,
        config=_config_dict([{"name": "injected_failure", "params": {},
                              "required": False}]),
        state_manager=manager,
        verbose=False,
    )
    results = processor.process_subject_task(
        subject_id="01", task="T", session="01", pipeline=PIPELINE)

    assert len(results) == 1
    assert results[0].success is True, results[0].logs

    state = manager.load_state("01", "T", PIPELINE, session="01", run="1")
    assert state.status == ProcessingStatus.COMPLETED
    step = state.get_step("preprocessing")
    assert step is not None and step.metadata["degraded"] is True
    assert step.metadata["step_status"]["injected_failure"] == "failed_tolerated"

    run_dir = _run_dir(bids_tree)
    fifs = list(run_dir.glob("*_preprocessed_raw.fif"))
    assert len(fifs) == 1, f"expected exactly one FIF, got {fifs}"
    assert list(run_dir.glob("*_preprocessing_FAILED.json")) == []
    assert len(list(run_dir.glob("*_preprocessing.json"))) == 1

    pstate = _processing_state(bids_tree)
    assert pstate["status"] == "completed"
    assert pstate["metadata"]["degraded"] is True
    assert pstate["output_files"]["preprocessed"] == str(fifs[0])


def test_subject_status_precedence():
    """Aggregation rule: failed dominates excluded dominates completed."""
    from eegcpm.cli.preprocess import _subject_status

    C = ProcessingStatus.COMPLETED
    E = ProcessingStatus.EXCLUDED
    F = ProcessingStatus.FAILED

    assert _subject_status([C, C]) is C
    assert _subject_status([C, E]) is E
    assert _subject_status([E, C]) is E
    assert _subject_status([E, F]) is F  # a failed run must surface over an exclusion
    assert _subject_status([C, E, F]) is F


def test_subject_status_empty_raises():
    """An empty run list has no subject status: silently returning a
    status would fake a successful batch."""
    from eegcpm.cli.preprocess import _subject_status

    with pytest.raises(ValueError):
        _subject_status([])


def test_cli_forced_rerun_load_failure_clears_stale_outputs(
        bids_tree, monkeypatch, tmp_path):
    """A forced re-run that fails *before* the pipeline (input load) must
    still clear the previous outputs (audit F1): no stale FIF and no
    success sidecar survive beside the FAILED row."""
    good = _write_config(tmp_path / "good.yaml", [])
    assert preprocess_command(_cli_args(good, bids_tree)) is None

    run_dir = _run_dir(bids_tree)
    assert len(list(run_dir.glob("*_preprocessed_raw.fif"))) == 1

    import eegcpm.cli.preprocess as cli_preprocess

    def _boom(input_file):
        raise RuntimeError("injected read_raw_eeg failure")

    monkeypatch.setattr(cli_preprocess, "read_raw_eeg", _boom)

    args = _cli_args(good, bids_tree)
    args.force = True
    assert preprocess_command(args) == 2

    _, state = _state(bids_tree, bids_tree / "eegcpm")
    assert state.status == ProcessingStatus.FAILED
    assert list(run_dir.glob("*_preprocessed_raw.fif")) == []
    assert list(run_dir.glob("*_preprocessing.json")) == []


def test_cli_load_failure_does_not_stop_later_runs(tmp_path, monkeypatch):
    """Run-1 failing at input load must not stop run-2; the batch exits
    with code 2 (N4 multi-run property)."""
    eeg_dir = tmp_path / "sub-01" / "ses-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    info = mne.create_info([f"EEG{i:03d}" for i in range(1, 9)], 250.0, "eeg")
    for run in ("1", "2"):
        raw = mne.io.RawArray(rng.standard_normal((8, 2500)) * 1e-6, info)
        raw.save(eeg_dir / f"sub-01_ses-01_task-T_run-{run}_eeg.fif",
                 overwrite=True, verbose=False)

    import eegcpm.cli.preprocess as cli_preprocess

    def _selective(input_file):
        if "run-1_" in str(input_file):
            raise RuntimeError("injected read failure for run-1")
        return mne.io.read_raw_fif(input_file, preload=True, verbose=False)

    monkeypatch.setattr(cli_preprocess, "read_raw_eeg", _selective)

    config = _write_config(tmp_path / "good.yaml", [])
    rc = preprocess_command(_cli_args(config, tmp_path))
    assert rc == 2

    manager = WorkflowStateManager(tmp_path / "eegcpm" / ".eegcpm" / "state.db")
    s1 = manager.load_state("01", "T", PIPELINE, session="01", run="1")
    s2 = manager.load_state("01", "T", PIPELINE, session="01", run="2")
    assert s1.status == ProcessingStatus.FAILED
    assert s2.status == ProcessingStatus.COMPLETED

    run2_dir = (tmp_path / "derivatives" / "preprocessing" / PIPELINE
                / "sub-01" / "ses-01" / "task-T" / "run-2")
    assert list(_run_dir(tmp_path).glob("*_preprocessed_raw.fif")) == []
    assert len(list(run2_dir.glob("*_preprocessed_raw.fif"))) == 1


def test_run_processor_exception_after_success_cleans_outputs(
        bids_tree, registered_steps, tmp_path, monkeypatch):
    """A run_processor run that succeeds and then crashes (post-process)
    must not keep its FIF/success sidecar (audit F2), and filesystem
    discovery must not report a successful run."""
    db_root = tmp_path / "rp_state"
    manager = WorkflowStateManager(db_root / ".eegcpm" / "state.db")
    processor = RunProcessor(
        bids_root=bids_tree,
        output_root=bids_tree / "derivatives" / "preprocessing" / PIPELINE,
        config=_config_dict([]),
        state_manager=manager,
        verbose=False,
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("injected quality-metrics failure")

    monkeypatch.setattr(RunProcessor, "_extract_quality_metrics", _boom)

    results = processor.process_subject_task(
        subject_id="01", task="T", session="01", pipeline=PIPELINE)
    assert len(results) == 1 and results[0].success is False

    run_dir = _run_dir(bids_tree)
    assert list(run_dir.glob("*_preprocessed_raw.fif")) == []
    assert list(run_dir.glob("*_preprocessing.json")) == []
    failed = list(run_dir.glob("*_preprocessing_FAILED.json"))
    assert len(failed) == 1, f"expected exactly one FAILED sidecar, got {failed}"

    ps = _processing_state(bids_tree)
    assert ps["status"] == "failed"
    assert "quality-metrics failure" in str(ps.get("error"))

    assert processor.load_processed_runs("01", "T", pipeline=PIPELINE) == []
