"""ASR timeline integrity (B3 — P1-F3 / F8 / F15; approval condition C2, unit level).

ASR must **repair at full length**, never cut:

- identical sample count and identical event samples before/after;
- ``info`` intact (``meas_date`` / ``highpass`` / ``lowpass`` / ``bads``);
- pre-existing annotations kept on their samples (incl. ``BAD_manual``);
- rejected windows marked with ``BAD_asr_window`` (from eegprep's
  ``clean_windows`` sample mask) instead of being removed;
- repaired (burst-corrected) intervals reported in metadata only;
- non-finite output → ``RuntimeError`` (no ``nan_to_num``, no
  ``BAD_asr_naninf``);
- metadata (not ``info['temp']``) is the provenance home and reaches the
  executor's JSON sidecar.

Frame note (probed 2026-10-14, mne 1.13.2): a stored annotation onset is
always ``raw.first_time + <seconds from the first data sample>`` — mne adds
``raw.first_time`` in ``set_annotations`` for ``orig_time is None`` too.
The data-relative position of an annotation is therefore
``onset - raw.first_time``, and an event's sample is ``onset * sfreq``.
"""

import json
from datetime import datetime, timezone

import mne
import numpy as np
import pytest

from eegcpm.modules.preprocessing.hbn_langer_chain import (
    _ASRStepWrapper,
    _BlockRejectionStepWrapper,
)
from eegcpm.modules.preprocessing.pipeline_builder import PreprocessingPipeline
from eegcpm.modules.preprocessing.steps.asr import ASRStep

SFREQ = 250.0
DURATION_S = 180.0
N_TIMES = int(DURATION_S * SFREQ)
MEAS_DATE = datetime(2020, 1, 1, tzinfo=timezone.utc)
ARTIFACT = (100.0, 103.0)          # data seconds
BAD_CHANNEL = "Pz"

# 32 standard_1020 names including Cz (the recording reference).
CH_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7",
    "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4",
    "P8", "O1", "Oz", "O2", "FT9", "FT10", "FC1", "FC2",
    "FC5", "FC6", "CP1", "CP2", "CP5", "CP6", "TP9", "TP10",
]

# Metadata keys ASRStep must report (plan §B3 task 2).
REQUIRED_METADATA_KEYS = [
    "applied", "implementation", "calibration", "n_calibration_samples",
    "cutoff", "window_criterion", "window_tolerances", "n_channels_asr",
    "excluded_from_asr", "window_rejected_fraction", "n_bad_windows",
    "bad_window_intervals_s", "repaired_fraction", "repaired_intervals_s",
    "asr_burden", "n_samples_before", "n_samples_after",
    "montage_present", "data_roundtrip",
]


def _make_raw(meas_date=MEAS_DATE, first_samp=1000, artifact=ARTIFACT, seed=0):
    """32-channel fixture: 180 s, 1/f-ish 10 µV noise, stim every 2 s.

    ``first_samp=1000`` (4 s at 250 Hz) makes every offset mistake visible.
    """
    rng = np.random.default_rng(seed)
    data = np.cumsum(rng.standard_normal((len(CH_NAMES), N_TIMES)), axis=1)
    data -= data.mean(axis=1, keepdims=True)
    data *= 10e-6 / data.std(axis=1, keepdims=True)

    info = mne.create_info(CH_NAMES, SFREQ, "eeg")
    raw = mne.io.RawArray(data, info, first_samp=first_samp, verbose=False)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                    match_case=False, on_missing="ignore")
    if meas_date is not None:
        raw.set_meas_date(meas_date)
    raw.filter(0.5, 45.0, verbose=False)      # sets highpass / lowpass

    if artifact is not None:
        t = np.arange(raw.n_times) / SFREQ
        sel = (t >= artifact[0]) & (t < artifact[1])
        raw._data[:, sel] += (
            rng.standard_normal((len(CH_NAMES), int(sel.sum()))) * 500e-6)

    raw.info["bads"] = [BAD_CHANNEL]

    onsets = np.append(np.arange(0.0, DURATION_S, 2.0), 50.0)
    descriptions = ["stim"] * int(DURATION_S / 2.0) + ["BAD_manual"]
    raw.set_annotations(mne.Annotations(
        onsets, np.zeros(len(onsets)), descriptions))
    return raw


def _artifact_samples():
    t = np.arange(N_TIMES) / SFREQ
    return (t >= ARTIFACT[0]) & (t < ARTIFACT[1])


def _annotation_keys(annotations):
    """Order-independent ``(onset, duration, description)`` triples.

    ``set_annotations`` sorts annotations by onset (mne 1.13.2), so array
    position is not a stable identity for pre-existing marks once a new
    ``BAD_*`` annotation is inserted between them.
    """
    return sorted(zip(np.asarray(annotations.onset, dtype=float),
                      np.asarray(annotations.duration, dtype=float),
                      list(annotations.description)))


def _assert_survived(before_keys, after_keys):
    """Every pre-existing annotation must still be there, on its sample."""
    for key in before_keys:
        assert key in after_keys, (
            f"pre-existing annotation moved or was lost: {key}")


def _assert_json_native(obj, path="metadata"):
    """Fail on numpy scalars/arrays or other non-JSON-native values."""
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            assert isinstance(key, str), f"{path} has a non-string key {key!r}"
            _assert_json_native(value, f"{path}[{key!r}]")
        return
    if isinstance(obj, list):
        for i, value in enumerate(obj):
            _assert_json_native(value, f"{path}[{i}]")
        return
    raise AssertionError(
        f"{path} is not JSON-native ({type(obj).__name__}); it would be "
        "corrupted by the run sidecar encoder")


# ----------------------------------------------------------------------
# timeline integrity
# ----------------------------------------------------------------------

@pytest.mark.parametrize("calibration", ["data_driven", "fixed_first_60s"])
def test_events_identical_after_asr(calibration):
    """Events, sample count and info are untouched by ASR."""
    raw = _make_raw()
    pre = raw.get_data().copy()
    events_before, _ = mne.events_from_annotations(raw)
    meas_date_before = raw.info["meas_date"]
    highpass_before = raw.info["highpass"]
    lowpass_before = raw.info["lowpass"]
    bads_before = list(raw.info["bads"])
    descriptions_before = list(raw.annotations.description)
    keys_before = _annotation_keys(raw.annotations)

    out, meta = ASRStep(calibration=calibration).process(raw, {})

    events_after, _ = mne.events_from_annotations(out)
    assert np.array_equal(events_before, events_after), (
        "event samples moved during ASR")
    assert out.n_times == N_TIMES
    assert out is raw, "ASR must write back into the original Raw"
    assert out.info["meas_date"] == meas_date_before
    assert out.info["highpass"] == highpass_before
    assert out.info["lowpass"] == lowpass_before
    assert list(out.info["bads"]) == bads_before

    # Append-only: the original annotations are still there, unmoved.
    _assert_survived(keys_before, _annotation_keys(out.annotations))
    assert "BAD_manual" in descriptions_before
    assert "BAD_manual" in list(out.annotations.description)

    # ASR actually repaired the artifact window (not a no-op).
    sel = _artifact_samples()
    assert np.std(out.get_data()[:, sel]) < 0.5 * np.std(pre[:, sel])

    assert meta["applied"] is True
    assert meta["n_samples_before"] == meta["n_samples_after"] == N_TIMES


def test_bad_asr_window_marks_artifact():
    """Rejected windows are annotated, never cut; repaired ones are not."""
    raw = _make_raw()
    out, meta = ASRStep().process(raw, {})

    descriptions = list(out.annotations.description)
    n_bad = descriptions.count("BAD_asr_window")
    assert n_bad >= 1, "the artifact window was not flagged as a bad ASR window"
    assert n_bad == meta["n_bad_windows"] == len(meta["bad_window_intervals_s"])
    assert meta["window_rejected_fraction"] > 0.0

    intervals = meta["bad_window_intervals_s"]
    assert any(o < ARTIFACT[1] and o + d > ARTIFACT[0] for o, d in intervals), (
        f"no BAD_asr_window overlaps the artifact window; got {intervals}")
    for _, duration in intervals:
        assert duration <= 0.2 * DURATION_S, (
            "a rejected window covers more than 20 % of the recording")

    # The annotations carry exactly the metadata intervals (data frame).
    asr_idx = [i for i, d in enumerate(descriptions) if d == "BAD_asr_window"]
    from_annotations = sorted(
        (float(out.annotations.onset[i] - out.first_time),
         float(out.annotations.duration[i])) for i in asr_idx)
    from_metadata = sorted((float(o), float(d)) for o, d in intervals)
    assert [x for iv in from_annotations for x in iv] == pytest.approx(
        [x for iv in from_metadata for x in iv])

    # Repaired intervals are metadata-only (design decision, no BAD_* mark).
    assert meta["repaired_fraction"] > 0.0
    assert len(meta["repaired_intervals_s"]) > 0
    assert "BAD_asr_repaired" not in descriptions
    assert meta["asr_burden"] > 0.0


def test_meas_date_none_also_works():
    """The meas_date=None branch of the offset recipe keeps the timeline.

    ``set_annotations`` re-adds ``raw.first_time`` when ``orig_time`` is
    None; without the correction in ``append_bad_annotations`` every
    pre-existing annotation would move by ``first_time`` (4 s here).
    """
    raw = _make_raw(meas_date=None)
    assert raw.info["meas_date"] is None
    assert raw.annotations.orig_time is None

    events_before, _ = mne.events_from_annotations(raw)
    keys_before = _annotation_keys(raw.annotations)

    out, meta = ASRStep().process(raw, {})

    events_after, _ = mne.events_from_annotations(out)
    assert np.array_equal(events_before, events_after), (
        "event samples moved during ASR with meas_date=None")
    assert out.n_times == N_TIMES
    assert out.info["meas_date"] is None
    _assert_survived(keys_before, _annotation_keys(out.annotations))
    assert "BAD_manual" in list(out.annotations.description)
    assert meta["n_bad_windows"] >= 1
    intervals = meta["bad_window_intervals_s"]
    assert any(o < ARTIFACT[1] and o + d > ARTIFACT[0] for o, d in intervals)


def test_nan_output_raises(monkeypatch):
    """Non-finite ASR output fails loudly (no nan_to_num, no BAD_asr_naninf)."""
    import eegprep

    raw = _make_raw()

    def _nan_clean_asr(eeg, **kwargs):
        out = dict(eeg)
        out["data"] = np.full(np.asarray(eeg["data"]).shape, np.nan)
        return out

    monkeypatch.setattr(eegprep, "clean_asr", _nan_clean_asr)
    with pytest.raises(RuntimeError, match="non-finite"):
        ASRStep().process(raw, {})
    assert not any(d.startswith("BAD_asr_naninf")
                   for d in raw.annotations.description)


def test_non_finite_input_raises():
    """Non-finite input fails loudly (eegprep would zero it silently)."""
    raw = _make_raw()
    raw._data[0, 100] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        ASRStep().process(raw, {})


def test_reference_channel_excluded():
    """A flat reference channel must not enter ASR (nor be repaired)."""
    raw = _make_raw()
    cz_idx = raw.ch_names.index("Cz")
    raw._data[cz_idx] = 0.0

    step = ASRStep()
    picks, excluded = step._asr_picks(
        raw, {"channel_roles": {"reference_channel": "Cz"}})
    assert excluded == ["Cz"]
    assert len(picks) == len(CH_NAMES) - 2          # minus Pz (bad) and Cz

    out, meta = step.process(raw, {"channel_roles": {"reference_channel": "Cz"}})
    assert "Cz" in meta["excluded_from_asr"]
    assert meta["n_channels_asr"] == len(CH_NAMES) - 2
    assert np.array_equal(out._data[cz_idx], np.zeros(out.n_times)), (
        "the excluded reference channel was modified by ASR")

    # Without the channel_roles metadata the flat channel is not excluded
    # (the exclusion is metadata-driven, not a hard-coded name).
    raw2 = _make_raw()
    raw2._data[raw2.ch_names.index("Cz")] = 0.0
    _, meta2 = ASRStep().process(raw2, {})
    assert "Cz" not in meta2["excluded_from_asr"]
    assert meta2["n_channels_asr"] == len(CH_NAMES) - 1

    # Explicit exclude_channels are honoured too (picks-level check);
    # the configured names are reported first, then the reference channel.
    _, excluded3 = ASRStep(exclude_channels=("Fz",))._asr_picks(
        raw, {"channel_roles": {"reference_channel": "Cz"}})
    assert excluded3 == ["Fz", "Cz"]


# ----------------------------------------------------------------------
# wrapper / metadata
# ----------------------------------------------------------------------

def test_wrapper_metadata_truthful():
    """P1-F8: the wrapper forwards the true ASR metadata, no phantom fields."""
    raw = _make_raw()
    wrapper = _ASRStepWrapper(cutoff=20.0, window_criterion=0.25,
                              retain_burst_masks=True)
    out, meta = wrapper.process(
        raw, {"channel_roles": {"reference_channel": "Cz"}})

    assert meta["applied"] is True
    assert meta["implementation"] == "eegprep"
    assert meta["calibration"] == "fixed_first_60s"
    assert meta["n_calibration_samples"] == int(60.0 * SFREQ) == 15000
    assert meta["cutoff"] == 20.0
    assert meta["window_criterion"] == 0.25
    assert meta["retain_burst_masks"] is True
    assert meta["n_samples_before"] == meta["n_samples_after"] == N_TIMES
    assert isinstance(meta["n_bad_windows"], int)       # int, not numpy
    assert "temp" not in out.info or not out.info["temp"], (
        "ASR wrote state into info['temp'] (it does not survive a FIF save)")

    # An unsupported calibration is refused instead of silently ignored.
    with pytest.raises(ValueError, match="fixed_first_60s"):
        _ASRStepWrapper(calibration_window="data_driven")


def test_metadata_is_json_native_and_complete():
    """The metadata replaces info['temp']: it must survive the sidecar."""
    raw = _make_raw()
    _, meta = ASRStep().process(raw, {})

    for key in REQUIRED_METADATA_KEYS:
        assert key in meta, f"missing ASR metadata key: {key}"
    assert meta["n_calibration_samples"] is None        # data_driven
    assert meta["window_tolerances"] == [-np.inf, 7.0]
    _assert_json_native(meta)
    json.dumps(meta)                                    # must not raise


def test_asr_metadata_reaches_sidecar(tmp_path):
    """The executor's JSON sidecar carries the ASR provenance (not info['temp'])."""
    raw = _make_raw()
    pipeline = PreprocessingPipeline(
        [{"name": "asr", "params": {"calibration": "fixed_first_60s"}}],
        tmp_path,
    )
    result = pipeline.process(raw, subject_id="sub-01")
    assert result.success, result.errors

    sidecar_path = tmp_path / "sub-01_preprocessing.json"
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["step_status"] == {"asr": "ok"}
    assert sidecar["asr"]["calibration"] == "fixed_first_60s"
    assert sidecar["asr"]["n_calibration_samples"] == 15000
    assert sidecar["asr"]["n_bad_windows"] >= 1
    assert sidecar["asr"]["window_rejected_fraction"] > 0.0
    saved = result.outputs["data"]
    assert "temp" not in saved.info or not saved.info["temp"]


def test_data_driven_sidecar_keeps_none_and_is_strict_json(tmp_path):
    """B3 rev1: the default (data-driven) calibration has no reported sample
    count — the sidecar must keep the ``None`` (not drop the key) and stay
    strict JSON (no ``Infinity`` tokens from ``window_tolerances``)."""
    raw = _make_raw()
    pipeline = PreprocessingPipeline(
        [{"name": "asr", "params": {}}],  # data-driven calibration
        tmp_path,
    )
    result = pipeline.process(raw, subject_id="sub-01")
    assert result.success, result.errors

    sidecar_path = tmp_path / "sub-01_preprocessing.json"
    text = sidecar_path.read_text()
    assert "-Infinity" not in text and "Infinity" not in text
    sidecar = json.loads(text)
    assert "n_calibration_samples" in sidecar["asr"]
    assert sidecar["asr"]["n_calibration_samples"] is None
    assert sidecar["asr"]["window_tolerances"] == ["-inf", 7.0]


def test_recording_qc_reads_asr_metadata():
    """The QC row sources the ASR fields from chain metadata (P1-F10 part)."""
    from eegcpm.modules.qc.hbn_recording_qc import recording_qc_metrics

    raw = _make_raw()
    row = recording_qc_metrics(
        raw, task="Rest", subject_id="sub-01",
        chain_metadata={"asr": {"asr_burden": 0.25,
                                "window_rejected_fraction": 0.1,
                                "repaired_fraction": 0.2}})
    assert row["asr_burden"] == 0.25
    assert row["asr_rejected_fraction"] == 0.1
    assert row["asr_repaired_fraction"] == 0.2

    # Without chain metadata the keys still exist (stable schema) but are None.
    row2 = recording_qc_metrics(raw, task="Rest")
    assert row2["asr_burden"] is None
    assert row2["asr_rejected_fraction"] is None
    assert row2["asr_repaired_fraction"] is None


def test_fixed_calibration_requires_enough_data(tmp_path):
    """validate_inputs enforces train_duration for the fixed branch only."""
    rng = np.random.default_rng(0)
    short = mne.io.RawArray(
        rng.standard_normal((len(CH_NAMES), int(10 * SFREQ))) * 10e-6,
        mne.create_info(CH_NAMES, SFREQ, "eeg"), verbose=False)

    assert ASRStep(calibration="data_driven").validate_inputs(short) is True
    with pytest.raises(ValueError, match="too short"):
        ASRStep(calibration="fixed_first_60s",
                train_duration=60.0).validate_inputs(short)
    assert ASRStep(calibration="fixed_first_60s",
                   train_duration=5.0).validate_inputs(short) is True

    # Through the executor: the fixed branch fails the run loudly (no FIF),
    # while the data-driven branch processes the same short recording.
    fixed = PreprocessingPipeline(
        [{"name": "asr", "params": {"calibration": "fixed_first_60s"}}],
        tmp_path)
    result = fixed.process(short.copy(), subject_id="short")
    assert result.success is False
    assert any("too short" in err for err in result.errors)
    assert list(tmp_path.glob("*_preprocessed_raw.fif")) == []
    assert (tmp_path / "short_preprocessing_FAILED.json").exists()

    data_driven = PreprocessingPipeline(
        [{"name": "asr", "params": {}}], tmp_path)
    assert data_driven.process(short.copy(), subject_id="short_dd").success


def test_golden_chain_wrapper_enforces_minimum_length(tmp_path, monkeypatch):
    """B3-K1: the hbn_langer ASR wrapper itself must reject short recordings
    (the base validate_inputs returns True, which silently capped the
    fixed-first-60 s calibration on the canonical chain)."""
    from eegcpm.modules.preprocessing.hbn_langer_chain import _ASRStepWrapper
    from eegcpm.modules.preprocessing.pipeline_builder import PreprocessingPipeline
    from eegcpm.modules.preprocessing.steps import STEP_REGISTRY

    rng = np.random.default_rng(0)
    short = mne.io.RawArray(
        rng.standard_normal((len(CH_NAMES), int(10 * SFREQ))) * 10e-6,
        mne.create_info(CH_NAMES, SFREQ, "eeg"), verbose=False)

    wrapper = _ASRStepWrapper()
    with pytest.raises(ValueError, match="too short"):
        wrapper.validate_inputs(short)
    # The long fixture passes.
    assert wrapper.validate_inputs(_make_raw()) is True

    # Executor path: the wrapper registered as a step fails the short run loudly.
    monkeypatch.setitem(STEP_REGISTRY, "asr_wrapper", _ASRStepWrapper)
    pipeline = PreprocessingPipeline(
        [{"name": "asr_wrapper", "params": {}}], tmp_path)
    result = pipeline.process(short.copy(), subject_id="short_chain")
    assert result.success is False
    assert list(tmp_path.glob("*_preprocessed_raw.fif")) == []


# ----------------------------------------------------------------------
# block rejection offsets (F4)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("with_meas_date", [True, False],
                         ids=["meas_date", "meas_date_none"])
def test_block_rejection_appends_with_offsets(with_meas_date):
    """F4: BAD_BLOCK is appended on the contaminated block, both frames.

    The stored onset frame always includes ``raw.first_time`` (mne adds it
    in ``set_annotations`` when ``orig_time`` is None as well), so the
    data-relative position is ``onset - raw.first_time`` in both branches.
    """
    raw = _make_raw(meas_date=MEAS_DATE if with_meas_date else None,
                    artifact=(10.0, 12.0))
    keys_before = _annotation_keys(raw.annotations)
    events_before, _ = mne.events_from_annotations(raw)

    step = _BlockRejectionStepWrapper(max_rejected_fraction=0.20,
                                      window_seconds=2.0)
    out, meta = step.process(raw, {})

    assert meta["applied"] is True
    assert meta["n_blocks_rejected"] == 1

    keys_after = _annotation_keys(out.annotations)
    _assert_survived(keys_before, keys_after)

    bad_keys = [key for key in keys_after if key[2] == "BAD_BLOCK"]
    assert len(bad_keys) == 1
    assert bad_keys[0][0] - out.first_time == pytest.approx(10.0)
    assert bad_keys[0][1] == pytest.approx(2.0)

    events_after, _ = mne.events_from_annotations(out)
    assert np.array_equal(events_before, events_after), (
        "existing event samples moved when BAD_BLOCK was appended")


@pytest.mark.parametrize("with_meas_date", [True, False],
                         ids=["meas_date", "meas_date_none"])
def test_artifact_annotation_step_keeps_event_samples(with_meas_date):
    """B3-K9: the four artifact append sites must not shift the timeline.

    With ``meas_date=None`` and ``first_samp=1000`` the uncorrected
    ``set_annotations(raw.annotations + annot)`` recipe moved every existing
    annotation by ``first_time`` (4 s here) per append site. The step now
    routes through ``append_bad_annotations``; the stim event samples must be
    identical and the new ``BAD_amplitude`` marks must land on the artifact.
    """
    from eegcpm.modules.preprocessing.steps.artifacts import (
        ArtifactAnnotationStep,
    )

    rng = np.random.default_rng(3)
    sfreq = 250.0
    duration = 30.0
    n_times = int(duration * sfreq)
    data = rng.standard_normal((len(CH_NAMES), n_times)) * 10e-6
    info = mne.create_info(CH_NAMES, sfreq, "eeg")
    raw = mne.io.RawArray(data, info, first_samp=1000, verbose=False)
    if with_meas_date:
        raw.set_meas_date(MEAS_DATE)

    # 1 s of +/-1 mV on every channel (ptp 2 mV >> 500 uV threshold)
    t = np.arange(n_times) / sfreq
    artifact_sel = (t >= 10.0) & (t < 11.0)
    raw._data[:, artifact_sel] += 1e-3 * np.sign(rng.standard_normal(
        (len(CH_NAMES), int(artifact_sel.sum()))))

    onsets = np.arange(0.0, duration, 2.0)
    raw.set_annotations(mne.Annotations(
        onsets, np.zeros(len(onsets)), ["stim"] * len(onsets)))

    keys_before = _annotation_keys(raw.annotations)
    events_before, _ = mne.events_from_annotations(raw)

    out, meta = ArtifactAnnotationStep().process(raw, {})

    assert meta["applied"] is True
    assert meta["n_annotations_added"] >= 1
    assert meta["annotation_counts"].get("BAD_amplitude", 0) >= 1

    keys_after = _annotation_keys(out.annotations)
    _assert_survived(keys_before, keys_after)

    events_after, _ = mne.events_from_annotations(out)
    assert np.array_equal(events_before, events_after), (
        "existing event samples moved when artifact annotations were appended")

    amplitude_keys = [key for key in keys_after if key[2] == "BAD_amplitude"]
    assert amplitude_keys
    # Data-relative frame: onset - first_time is seconds from the first
    # sample in both meas_date branches; the 100 ms sliding window starts at
    # or just before the artifact.
    first_onset = min(key[0] for key in amplitude_keys)
    assert 9.8 <= (first_onset - out.first_time) <= 10.1
    # ... and no other stim annotation moved.
    stim_after = sorted(key[0] - out.first_time
                        for key in keys_after if key[2] == "stim")
    assert stim_after == pytest.approx(sorted(onsets), abs=1e-9)


def test_append_bad_annotations_refuses_to_drop_timeline():
    """B3-K6: a desynced ``orig_time`` (private-state mutation that MNE's
    public API never produces) must fail loudly instead of silently dropping
    the whole annotation timeline."""
    from datetime import timedelta

    from eegcpm.modules.preprocessing.steps.asr import append_bad_annotations

    raw = _make_raw(meas_date=MEAS_DATE)
    desynced = mne.Annotations(
        onset=raw.annotations.onset,
        duration=raw.annotations.duration,
        description=raw.annotations.description,
        orig_time=MEAS_DATE + timedelta(hours=1),  # desync, private-state only
    )
    raw._annotations = desynced

    with pytest.raises(RuntimeError, match="refusing to rewrite"):
        append_bad_annotations(raw, [1.0], [0.0], ["BAD_test"])


def test_clean_windows_receives_repaired_data(monkeypatch):
    """B3-K7: the rejection mask is computed on the REPAIRED data (residual
    post-repair badness), not on the pre-ASR input. Without this assertion a
    regression to ``clean_windows(pre)`` would leave every test green (the
    fixture's pre/post masks are identical)."""
    import eegprep

    from eegcpm.modules.preprocessing.steps.asr import ASRStep

    seen = {}
    real = eegprep.clean_windows

    def spy(EEG, *args, **kwargs):
        seen["data"] = np.array(EEG["data"], dtype=float, copy=True)
        return real(EEG, *args, **kwargs)

    monkeypatch.setattr(eegprep, "clean_windows", spy)

    raw = _make_raw()
    step = ASRStep()  # data-driven
    out, meta = step.process(raw, {})
    assert "data" in seen, "clean_windows was not called"

    # The write-back is exactly the repaired data the spy saw (× 1e-6 V).
    # ASR runs on the EEG picks minus bads (the fixture has one bad channel).
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    post = raw.get_data(picks=picks)
    np.testing.assert_allclose(seen["data"] * 1e-6, post, rtol=1e-9, atol=1e-12)


def test_asr_wrapper_config_matches_fixed_path_kwargs(monkeypatch):
    """NEW-4 re-audit: the wrapper's get_config must describe the fixed-first-60 s
    path it actually runs (ndarray calibration reference; ref_tolerances not
    passed) — not the data-driven constants."""
    import eegprep

    from eegcpm.modules.preprocessing.hbn_langer_chain import _ASRStepWrapper

    calls = {}
    real = eegprep.clean_asr

    def spy(EEG, *args, **kwargs):
        calls.update(kwargs)
        return real(EEG, *args, **kwargs)

    monkeypatch.setattr(eegprep, "clean_asr", spy)

    raw = _make_raw()
    wrapper = _ASRStepWrapper()
    out, meta = wrapper.process(raw, {})

    assert calls["cutoff"] == 20.0
    ref = calls["ref_maxbadchannels"]
    assert isinstance(ref, np.ndarray) and ref.ndim == 2
    assert ref.shape[1] == int(60.0 * SFREQ)
    assert "ref_tolerances" not in calls

    cfg = wrapper.get_config()
    assert cfg["calibration_reference"] == "first_train_duration_s_of_data"
    assert cfg["ref_maxbadchannels"] == "calibration_array"
    assert cfg["ref_tolerances"] == "eegprep_default"
