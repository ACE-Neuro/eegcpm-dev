"""Executor failure semantics (B1 — P1-F1 / F4, approval condition C1).

A **required** step that raises, returns ``skipped``/``applied: False``,
or fails input validation must fail the run:

- ``result.success is False`` and ``result.outputs['data'] is None``;
- no ``*_preprocessed_raw.fif`` under the output dir;
- a ``*_preprocessing_FAILED.json`` sidecar instead.

An **optional** step failure (``required: False``) is tolerated:
``failed_tolerated`` + ``metadata['degraded'] is True``, the run still
succeeds and writes its FIF.

A not-applied result whose reason is in ``ALLOWED_NOOP_REASONS`` is a
legitimate no-op (``noop``), not a failure. ``RecordingExcluded`` marks
the run ``excluded`` (no FIF, an ``*_EXCLUDED.json`` sidecar), and the
block-rejection gate appends its ``BAD_BLOCK`` annotations **before**
raising.
"""

import json

import numpy as np
import mne
import pytest

from eegcpm.modules.preprocessing.steps import STEP_REGISTRY, ProcessingStep
from eegcpm.modules.preprocessing.steps.base import RecordingExcluded
from eegcpm.modules.preprocessing.pipeline_builder import PreprocessingPipeline


# --- throwaway steps registered per-test via monkeypatch ---

class BoomStep(ProcessingStep):
    name = "boom"
    version = "0.1"

    def process(self, raw, metadata):
        raise RuntimeError("injected boom")


class AppliedFalseStep(ProcessingStep):
    name = "applied_false"
    version = "0.1"

    def process(self, raw, metadata):
        return raw, {"applied": False, "reason": "injected_no_op"}


class SkippedStep(ProcessingStep):
    name = "skipped"
    version = "0.1"

    def process(self, raw, metadata):
        return raw, {"skipped": True, "reason": "injected_skip"}


class ValidationFalseStep(ProcessingStep):
    name = "validation_false"
    version = "0.1"

    def validate_inputs(self, raw):
        return False

    def process(self, raw, metadata):
        raise AssertionError("process must not run when validation fails")


class ExclusionStep(ProcessingStep):
    name = "exclusion"
    version = "0.1"

    def process(self, raw, metadata):
        raise RecordingExcluded("too_bad", {"x": 1})


class NumpySkippedStep(ProcessingStep):
    """Returns `np.bool_(True)` for `skipped` (numpy truthiness)."""
    name = "numpy_skipped"
    version = "0.1"

    def process(self, raw, metadata):
        return raw, {"skipped": np.bool_(True), "reason": "injected_numpy_skip"}


class NumpyAppliedFalseStep(ProcessingStep):
    """Returns `np.bool_(False)` for `applied` (numpy truthiness)."""
    name = "numpy_applied_false"
    version = "0.1"

    def process(self, raw, metadata):
        return raw, {"applied": np.bool_(False), "reason": "injected_numpy_no_op"}


class NoneMetadataStep(ProcessingStep):
    """Violates the (raw, dict) return contract."""
    name = "none_metadata"
    version = "0.1"

    def process(self, raw, metadata):
        return raw, None


def _make_raw(n_channels=8, sfreq=250.0, duration_s=20.0, seed=0):
    """Synthetic low-noise RawArray (8 EEG ch, 250 Hz, 20 s)."""
    rng = np.random.default_rng(seed)
    ch_names = [f"EEG{i:03d}" for i in range(1, n_channels + 1)]
    info = mne.create_info(ch_names, sfreq, "eeg")
    data = rng.standard_normal((n_channels, int(sfreq * duration_s))) * 1e-6
    return mne.io.RawArray(data, info)


FILTER = {"name": "filter", "params": {"l_freq": 1.0, "h_freq": 40.0}}


# --- construction-time failures ---

@pytest.mark.parametrize("empty", [[], None, ()])
def test_empty_steps_raises(empty):
    """An empty pipeline must fail at construction, never no-op."""
    with pytest.raises(ValueError, match="no steps"):
        PreprocessingPipeline(empty)


def test_missing_dependency_fails_at_construction(monkeypatch):
    """A missing step backend fails the dependency preflight."""
    from eegcpm.modules.preprocessing.steps.zapline import ZaplineStep

    def _raise_import_error(self):
        raise ImportError("pyzaplineplus not installed")

    monkeypatch.setattr(ZaplineStep, "check_dependencies", _raise_import_error)

    with pytest.raises(ImportError):
        PreprocessingPipeline([{"name": "zapline", "params": {}}])


# --- required-step failure semantics ---

def test_required_exception_aborts(monkeypatch, tmp_path):
    """A required step raising aborts the run: no FIF, FAILED sidecar."""
    monkeypatch.setitem(STEP_REGISTRY, "boom", BoomStep)
    pipeline = PreprocessingPipeline(
        [FILTER, {"name": "boom", "params": {}}, FILTER],
        output_dir=tmp_path,
    )
    result = pipeline.process(_make_raw(), subject_id="sub-01",
                              session_id="01", run_id="1")

    assert not result.success
    assert result.outputs["data"] is None
    assert result.outputs["ica"] is None
    assert result.errors, "the required failure must be recorded in errors"
    assert result.metadata["step_status"] == {
        # The executor suffixes duplicated step names with their index.
        "filter_0": "ok",
        "boom": "failed",
        "filter_2": "not_run",
    }
    assert result.metadata["boom"]["error"] == "injected boom"
    assert "traceback" in result.metadata["boom"]

    assert list(tmp_path.glob("*_preprocessed_raw.fif")) == []
    failed = list(tmp_path.glob("*_preprocessing_FAILED.json"))
    assert len(failed) == 1, f"expected one FAILED sidecar, got {failed}"
    assert failed[0].name.startswith("sub-01_ses-01_run-1")


def test_required_applied_false_fails(monkeypatch, tmp_path):
    """`applied: False` on a required step is a failure, not a no-op."""
    monkeypatch.setitem(STEP_REGISTRY, "applied_false", AppliedFalseStep)
    pipeline = PreprocessingPipeline(
        [{"name": "applied_false", "params": {}}, FILTER],
        output_dir=tmp_path,
    )
    result = pipeline.process(_make_raw(), subject_id="sub-01")

    assert not result.success
    assert result.metadata["step_status"]["applied_false"] == "failed"
    assert result.metadata["step_status"]["filter"] == "not_run"
    assert list(tmp_path.glob("*_preprocessed_raw.fif")) == []
    assert result.outputs["data"] is None


def test_required_skipped_fails(monkeypatch, tmp_path):
    """`skipped: True` with a non-allow-listed reason is a failure."""
    monkeypatch.setitem(STEP_REGISTRY, "skipped", SkippedStep)
    pipeline = PreprocessingPipeline(
        [{"name": "skipped", "params": {}}], output_dir=tmp_path)
    result = pipeline.process(_make_raw(), subject_id="sub-01")

    assert not result.success
    assert result.metadata["step_status"]["skipped"] == "failed"
    assert "injected_skip" in result.errors[0]
    assert list(tmp_path.glob("*_preprocessed_raw.fif")) == []


def test_validate_inputs_false_fails(monkeypatch, tmp_path):
    """A False `validate_inputs` on a required step fails the run."""
    monkeypatch.setitem(STEP_REGISTRY, "validation_false", ValidationFalseStep)
    pipeline = PreprocessingPipeline(
        [{"name": "validation_false", "params": {}}], output_dir=tmp_path)
    result = pipeline.process(_make_raw(), subject_id="sub-01")

    assert not result.success
    assert result.metadata["step_status"]["validation_false"] == "failed"
    assert "validation_failed" in result.errors[0]
    assert list(tmp_path.glob("*_preprocessed_raw.fif")) == []


def test_required_conditional_skip_fails(tmp_path):
    """A required step skipped by `skip_step` (iclabel without an ICA
    step) is a failure with reason `conditional_skip`."""
    pipeline = PreprocessingPipeline(
        [{"name": "iclabel", "params": {}}], output_dir=tmp_path)
    result = pipeline.process(_make_raw(), subject_id="sub-01")

    assert not result.success
    assert result.metadata["step_status"]["iclabel"] == "failed"
    assert result.metadata["iclabel"]["applied"] is False
    assert result.metadata["iclabel"]["reason"] == "conditional_skip"
    assert "conditional_skip" in result.errors[0]
    assert list(tmp_path.glob("*_preprocessed_raw.fif")) == []
    assert len(list(tmp_path.glob("*_preprocessing_FAILED.json"))) == 1


# --- optional-step tolerance and allow-listed no-ops ---

def test_optional_failure_tolerated(monkeypatch, tmp_path):
    """An optional failure is recorded, degrades the run, and the run
    still succeeds (FIF written)."""
    monkeypatch.setitem(STEP_REGISTRY, "boom", BoomStep)
    pipeline = PreprocessingPipeline(
        [FILTER, {"name": "boom", "params": {}, "required": False}, FILTER],
        output_dir=tmp_path,
    )
    result = pipeline.process(_make_raw(), subject_id="sub-01",
                              session_id="01", run_id="1")

    assert result.success
    assert result.metadata["step_status"]["boom"] == "failed_tolerated"
    assert result.metadata["step_status"]["filter_2"] == "ok"
    assert result.metadata["degraded"] is True
    assert result.errors == []
    assert result.warnings
    assert result.outputs["data"] is not None

    fifs = list(tmp_path.glob("*_preprocessed_raw.fif"))
    assert len(fifs) == 1
    success_sidecar = list(tmp_path.glob("*_preprocessing.json"))
    assert len(success_sidecar) == 1
    assert not list(tmp_path.glob("*_preprocessing_FAILED.json"))


def test_allowlisted_noop_ok(tmp_path):
    """Interpolate with zero bad channels is an allow-listed no-op."""
    pipeline = PreprocessingPipeline(
        [{"name": "interpolate", "params": {}}], output_dir=tmp_path)
    result = pipeline.process(_make_raw(), subject_id="sub-01",
                              session_id="01", run_id="1")

    assert result.success
    assert result.metadata["step_status"]["interpolate"] == "noop"
    assert result.errors == []
    assert result.metadata["interpolate"]["reason"] == "no_bad_channels"
    assert len(list(tmp_path.glob("*_preprocessed_raw.fif"))) == 1


def test_disabled_step_recorded_as_disabled(tmp_path):
    """`enabled: False` is an explicit skip, not a failure."""
    pipeline = PreprocessingPipeline(
        [FILTER,
         {"name": "interpolate", "params": {"enabled": False}}],
        output_dir=tmp_path,
    )
    result = pipeline.process(_make_raw(), subject_id="sub-01")

    assert result.success
    assert result.metadata["step_status"]["interpolate"] == "disabled"


# --- numpy truthiness / return-contract robustness ---

def test_numpy_skipped_counts_as_not_applied(monkeypatch, tmp_path):
    """`{'skipped': np.bool_(True)}` must be treated as not applied."""
    monkeypatch.setitem(STEP_REGISTRY, "numpy_skipped", NumpySkippedStep)
    pipeline = PreprocessingPipeline(
        [{"name": "numpy_skipped", "params": {}}], output_dir=tmp_path)
    result = pipeline.process(_make_raw(), subject_id="sub-01")

    assert not result.success
    assert result.metadata["step_status"]["numpy_skipped"] == "failed"
    assert "injected_numpy_skip" in result.errors[0]
    assert list(tmp_path.glob("*_preprocessed_raw.fif")) == []


def test_numpy_applied_false_counts_as_not_applied(monkeypatch, tmp_path):
    """`{'applied': np.bool_(False)}` must be treated as not applied."""
    monkeypatch.setitem(STEP_REGISTRY, "numpy_applied_false",
                        NumpyAppliedFalseStep)
    pipeline = PreprocessingPipeline(
        [{"name": "numpy_applied_false", "params": {}}], output_dir=tmp_path)
    result = pipeline.process(_make_raw(), subject_id="sub-01")

    assert not result.success
    assert result.metadata["step_status"]["numpy_applied_false"] == "failed"
    assert "injected_numpy_no_op" in result.errors[0]
    assert list(tmp_path.glob("*_preprocessed_raw.fif")) == []


def test_none_step_metadata_fails(monkeypatch, tmp_path):
    """`(raw, None)` violates the return contract → failed run."""
    monkeypatch.setitem(STEP_REGISTRY, "none_metadata", NoneMetadataStep)
    pipeline = PreprocessingPipeline(
        [{"name": "none_metadata", "params": {}}], output_dir=tmp_path)
    result = pipeline.process(_make_raw(), subject_id="sub-01")

    assert not result.success
    assert result.metadata["step_status"]["none_metadata"] == "failed"
    assert "must return" in result.metadata["none_metadata"]["error"]
    assert result.outputs["data"] is None
    assert list(tmp_path.glob("*_preprocessed_raw.fif")) == []
    assert len(list(tmp_path.glob("*_preprocessing_FAILED.json"))) == 1


# --- allow-listed legitimate no-ops (K1) ---

def test_montage_already_set_is_noop(tmp_path):
    """A recording that already has a montage is a legitimate no-op."""
    raw = _make_raw()
    montage = mne.channels.make_standard_montage("biosemi64")
    names = montage.ch_names[:8]
    info = mne.create_info(names, 250.0, "eeg")
    raw = mne.io.RawArray(raw.get_data()[:8, :5000], info)
    raw.set_montage(montage)

    pipeline = PreprocessingPipeline(
        [{"name": "montage", "params": {"type": "biosemi64"}}],
        output_dir=tmp_path,
    )
    result = pipeline.process(raw, subject_id="sub-01")

    assert result.success
    assert result.metadata["step_status"]["montage"] == "noop"
    assert result.metadata["montage"]["reason"] == "montage_already_set"
    assert result.errors == []
    assert len(list(tmp_path.glob("*_preprocessed_raw.fif"))) == 1


def test_resample_same_sfreq_is_noop(tmp_path):
    """A recording already at the target sfreq is a legitimate no-op."""
    pipeline = PreprocessingPipeline(
        [{"name": "resample", "params": {"sfreq": 250.0}}], output_dir=tmp_path)
    result = pipeline.process(_make_raw(sfreq=250.0), subject_id="sub-01")

    assert result.success
    assert result.metadata["step_status"]["resample"] == "noop"
    assert result.metadata["resample"]["reason"] == "already_at_target_sfreq"
    assert result.errors == []
    assert len(list(tmp_path.glob("*_preprocessed_raw.fif"))) == 1


# --- exclusion ---

def test_recording_excluded(monkeypatch, tmp_path):
    """`RecordingExcluded` fails the run as EXCLUDED: no FIF, metrics
    preserved, EXCLUDED sidecar written."""
    monkeypatch.setitem(STEP_REGISTRY, "exclusion", ExclusionStep)
    pipeline = PreprocessingPipeline(
        [{"name": "exclusion", "params": {}}], output_dir=tmp_path)
    result = pipeline.process(_make_raw(), subject_id="sub-01",
                              session_id="01", run_id="1")

    assert not result.success
    assert result.metadata["excluded"]["reason"] == "too_bad"
    assert result.metadata["excluded"]["x"] == 1
    assert result.metadata["step_status"]["exclusion"] == "excluded"
    assert result.outputs["data"] is None

    assert list(tmp_path.glob("*_preprocessed_raw.fif")) == []
    excluded = list(tmp_path.glob("*_preprocessing_EXCLUDED.json"))
    assert len(excluded) == 1, f"expected one EXCLUDED sidecar, got {excluded}"


def test_block_rejection_annotates_then_excludes():
    """F4 + R-006: over-budget block contamination raises
    `RecordingExcluded` *after* appending BAD_BLOCK annotations, so the
    original (stim) annotations survive the hard failure."""
    from eegcpm.modules.preprocessing.hbn_langer_chain import (
        _BlockRejectionStepWrapper,
    )

    rng = np.random.RandomState(0)
    sfreq = 250.0
    n_channels = 8
    n_blocks = 10
    window_seconds = 2.0
    n_times = int(n_blocks * window_seconds * sfreq)  # 20 s
    info = mne.create_info([f"E{i}" for i in range(1, n_channels + 1)],
                           sfreq, "eeg")
    data = rng.randn(n_channels, n_times) * 1e-6
    # Contaminate 3/10 blocks = 30 % (> max_rejected_fraction 0.20)
    for b in (1, 4, 7):
        start = int(b * window_seconds * sfreq)
        stop = int((b + 1) * window_seconds * sfreq)
        data[:, start:stop] *= 200.0
    raw = mne.io.RawArray(data, info)
    raw.set_annotations(mne.Annotations([0.5, 3.25], [0.0, 0.0],
                                        ["stim_1", "stim_2"]))

    step = _BlockRejectionStepWrapper(max_rejected_fraction=0.20,
                                      window_seconds=window_seconds)
    with pytest.raises(RecordingExcluded) as excinfo:
        step.process(raw, {})

    assert excinfo.value.reason == "block_rejection_fraction_exceeded"
    assert excinfo.value.metrics["n_blocks_rejected"] == 3
    assert excinfo.value.metrics["n_blocks_total"] == n_blocks
    assert excinfo.value.metrics["rejected_fraction"] == pytest.approx(0.3)
    assert excinfo.value.metrics["bad_block_onsets_s"] == [2.0, 8.0, 14.0]

    descriptions = list(raw.annotations.description)
    # Pre-existing annotations were NOT wiped (append-only).
    assert "stim_1" in descriptions and "stim_2" in descriptions
    # BAD_BLOCK marks were appended before the exclusion was raised.
    assert descriptions.count("BAD_BLOCK") == 3


# --- save / sidecar ordering (K8) ---

def test_save_output_failure_fails_run_and_cleans_up(tmp_path, monkeypatch):
    """A raising `_save_outputs` must not leave a partial FIF: the run
    fails and only a FAILED sidecar remains."""
    pipeline = PreprocessingPipeline([FILTER], output_dir=tmp_path)

    def _partial_save(raw, ica, subject_id, **kwargs):
        stem = pipeline._output_stem(subject_id, **kwargs)
        (tmp_path / f"{stem}_preprocessed_raw.fif").write_bytes(b"partial")
        raise RuntimeError("injected save failure")

    monkeypatch.setattr(pipeline, "_save_outputs", _partial_save)
    result = pipeline.process(_make_raw(), subject_id="sub-01",
                              session_id="01", run_id="1")

    assert not result.success
    assert result.output_files == []
    assert result.outputs["data"] is None
    assert result.outputs["ica"] is None
    assert any("injected save failure" in e for e in result.errors)
    assert list(tmp_path.glob("*_preprocessed_raw.fif")) == []
    failed = list(tmp_path.glob("*_preprocessing_FAILED.json"))
    assert len(failed) == 1
    payload = json.loads(failed[0].read_text())
    assert payload["success"] is False
    assert payload["save_error"]["error"] == "injected save failure"


def test_sidecar_failure_after_successful_save_deletes_outputs(
        tmp_path, monkeypatch):
    """If the sidecar write raises after a successful save, the saved
    outputs are deleted and the error is re-raised."""
    pipeline = PreprocessingPipeline([FILTER], output_dir=tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("injected sidecar failure")

    monkeypatch.setattr(pipeline, "_write_sidecar", _boom)
    with pytest.raises(RuntimeError, match="injected sidecar failure"):
        pipeline.process(_make_raw(), subject_id="sub-01",
                         session_id="01", run_id="1")

    assert list(tmp_path.glob("*_preprocessed_raw.fif")) == []
