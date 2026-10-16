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

import contextlib
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

# Metadata keys ASRStep must report (plan §B3 task 2; B7 rev2 additions;
# §12.6 EEGLAB-standard ASR additions; R1/R4 ASR scoped audit additions).
REQUIRED_METADATA_KEYS = [
    "applied", "implementation", "calibration", "n_calibration_samples",
    "cutoff", "window_criterion", "window_tolerances", "n_channels_asr",
    "excluded_from_asr", "window_rejected_fraction", "n_bad_windows",
    "bad_window_intervals_s", "repaired_fraction", "repaired_intervals_s",
    "asr_burden", "n_samples_before", "n_samples_after",
    "montage_present", "data_roundtrip",
    # B7 rev2 fix 7 (re-interpolation) + B7-N2 (magnitude guard provenance)
    "reinterpolated_after_asr", "reinterpolation_mode",
    "reinterpolation_relative_norm", "pre_max_uv", "post_max_uv",
    "output_magnitude_ratio", "calibration_cov_cond",
    "calibration_cov_min_eig",
    # Plan §12.6 (EEGLAB-standard ASR): input high-pass + max_dims provenance.
    "drift_transition_hz", "drift_filter_applied",
    "drift_filter_modified_fraction", "max_dims",
    # R1/R4 (ASR scoped audit): pinv tolerance + truthful info['highpass'].
    "info_highpass_hz", "pinv_tolerance", "eegprep_version",
    "eegprep_pinv_verified_version",
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
    # R4 (ASR scoped audit): the write-back is the clean_drifts-high-passed
    # data, so info['highpass'] is updated to the filter's passband edge
    # (0.75 Hz) — the fixture's own 0.5 Hz value describes the pre-ASR data.
    assert highpass_before == 0.5
    assert out.info["highpass"] == 0.75
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
    # (the exclusion is metadata-driven, not a hard-coded name) — and feeding a
    # flat, zero-variance channel into the ASR covariance is now caught by the
    # B7-N2 magnitude guard instead of silently returning garbage: measured
    # 2026-10-16, eegprep returns 2.06e16 µV from a 2006 µV input on this
    # fixture. The old test asserted that garbage path as a success.
    raw2 = _make_raw()
    raw2._data[raw2.ch_names.index("Cz")] = 0.0
    with pytest.raises(RuntimeError, match="asr_output_magnitude_blowup"):
        ASRStep().process(raw2, {})

    # Explicit exclude_channels are honoured too (picks-level check);
    # the configured names are reported first, then the reference channel.
    _, excluded3 = ASRStep(exclude_channels=("Fz",))._asr_picks(
        raw, {"channel_roles": {"reference_channel": "Cz"}})
    assert excluded3 == ["Fz", "Cz"]


# ----------------------------------------------------------------------
# B7-N1: interpolated channels must not enter the ASR covariance
# ----------------------------------------------------------------------

def test_interpolated_channels_excluded_from_asr():
    """B7-N1: a previous interpolate step's channels stay out of the ASR input.

    Spherical-spline interpolation reconstructs a bad channel as an exact linear
    combination of the remaining ones, so including it makes the ASR calibration
    covariance exactly singular; eegprep's ``clean_asr`` then returns ~1e15 µV
    (probed 2026-10-16) and the block gate rejects the recording. The step
    excludes them via ``metadata['interpolate']['interpolated_channels']``.
    """
    raw = _make_raw()
    # Interpolate a channel exactly as the chain does (bads -> interpolate).
    raw.info["bads"] = ["Pz"]
    raw.interpolate_bads(reset_bads=True, verbose=False)

    metadata = {
        "channel_roles": {"reference_channel": "Cz"},
        "interpolate": {
            "applied": True,
            "interpolated_channels": ["Pz"],
        },
    }
    step = ASRStep()
    picks, excluded = step._asr_picks(raw, metadata)
    assert excluded == ["Cz", "Pz"]
    assert len(picks) == len(CH_NAMES) - 2

    out, meta = step.process(raw, metadata)
    assert meta["interpolated_excluded_from_asr"] == ["Pz"]
    assert "Pz" in meta["excluded_from_asr"]
    assert meta["n_channels_asr"] == len(CH_NAMES) - 2


def test_interpolated_channel_does_not_break_asr():
    """B7-N1 regression: one interpolated channel must not degenerate ASR.

    Measured 2026-10-16 (probe under ``/tmp/opencode/``): with the interpolated
    channel in the ASR input, the calibration covariance has 1 exact zero
    eigenvalue (condition number 1.2e15) and the repaired data come back at
    ~1e15 µV (``LinAlgWarning: Matrix is singular``); excluding it restores a
    full-rank covariance and an output at the input's scale. This test pins the
    *sane* behaviour, so a regression to the degenerate call fails loudly.
    """
    raw = _make_raw()
    raw.info["bads"] = ["Pz"]
    raw.interpolate_bads(reset_bads=True, verbose=False)
    input_scale = float(np.std(raw.get_data(picks="eeg")))

    metadata = {
        "channel_roles": {"reference_channel": "Cz"},
        "interpolate": {"applied": True, "interpolated_channels": ["Pz"],
                        "mode": "accurate"},
    }
    out, meta = ASRStep().process(raw, metadata)
    out_scale = float(np.std(out.get_data(picks="eeg")))
    assert np.isfinite(out.get_data()).all()
    assert out_scale < 10 * input_scale, (
        f"ASR output scale {out_scale:.3g} is far above the input scale "
        f"{input_scale:.3g}: the interpolated channel entered the covariance"
    )

    # Control: without the metadata the channel enters ASR and the calibration
    # covariance is (near-)singular: the interpolated channel is an exact linear
    # combination of the others, so the smallest eigenvalue collapses towards
    # the float64 noise floor. On this 32-channel fixture the condition number
    # degrades from ~17 to ~2e4; on the EGI-129 chain fixture it reaches 1.2e15
    # and eegprep's `sqrtm` returns garbage (that is what B7-N1 fixed).
    control = _make_raw()
    control.info["bads"] = ["Pz"]
    control.interpolate_bads(reset_bads=True, verbose=False)
    import eegprep

    picks, _ = ASRStep()._asr_picks(control, {"channel_roles": {"reference_channel": "Cz"}})
    picks_without = [p for p in picks if control.ch_names[p] != "Pz"]

    def _cond(pick_list):
        eeg = eegprep.eeg_mne2eeg(control.copy().pick(pick_list))
        eigvals = np.linalg.eigvalsh(np.cov(eeg["data"][:, : int(60.0 * SFREQ)]))
        return float(eigvals[-1] / max(eigvals[0], 1e-300))

    cond_with = _cond(picks)
    cond_without = _cond(picks_without)
    assert cond_with > 100 * cond_without, (
        f"fixture no longer shows the rank deficiency: cond with the "
        f"interpolated channel {cond_with:.3g} vs without {cond_without:.3g}"
    )


def test_interpolated_channels_reinterpolated_after_asr():
    """B7 rev2 fix 7: the ASR-excluded interpolated channels are re-interpolated.

    The excluded channel must equal a **fresh** re-interpolation of the
    ASR-cleaned data (same mode), not its pre-ASR spline values. On the
    EGI-129 chain fixture the pre-ASR values diverge from a fresh
    re-interpolation by ~1.02/1.09 relative norm (probed 2026-10-16); on this
    32-channel fixture the divergence is smaller but must still be non-zero
    (ASR rewrites the neighbours).
    """
    raw = _make_raw()
    raw.info["bads"] = ["Pz"]
    raw.interpolate_bads(reset_bads=True, verbose=False)
    pz = raw.ch_names.index("Pz")
    pre_asr_pz = raw._data[pz].copy()

    metadata = {
        "channel_roles": {"reference_channel": "Cz"},
        "interpolate": {"applied": True, "interpolated_channels": ["Pz"],
                        "mode": "accurate"},
    }
    out, meta = ASRStep().process(raw, metadata)

    # The fresh re-interpolation of the ASR output must reproduce the stored
    # channel exactly (the step re-interpolates from the same repaired data).
    fresh = out.copy()
    fresh.info["bads"] = ["Pz"]
    fresh.interpolate_bads(reset_bads=False, mode="accurate", verbose=False)
    fresh_pz = fresh.get_data(picks=["Pz"])[0]
    np.testing.assert_allclose(out._data[pz], fresh_pz, rtol=1e-9, atol=1e-12)

    # ... and the pre-ASR values were NOT kept: the re-interpolation changed
    # the channel (non-vacuous), and the metadata records how much.
    assert meta["reinterpolated_after_asr"] == ["Pz"]
    assert meta["reinterpolation_mode"] == "accurate"
    rel = meta["reinterpolation_relative_norm"]["Pz"]
    assert rel > 0, "re-interpolation was a no-op (the pre-ASR values were kept)"
    assert rel == pytest.approx(
        float(np.linalg.norm(pre_asr_pz - out._data[pz])
              / np.linalg.norm(out._data[pz])), rel=1e-6)
    # The bads list is restored (the synthetic channel is never left marked).
    assert list(out.info["bads"]) == []


# ----------------------------------------------------------------------
# B7-N2: ASR output magnitude blow-up guard
# ----------------------------------------------------------------------

def _near_duplicate_hp_raw(seed=3, duration_s=120.0):
    """32-channel 0.1 Hz high-passed fixture with near-duplicate channels.

    Probed 2026-10-16: three near-duplicate pairs give a calibration covariance
    condition number of 2.2e8 (not exactly singular — the audit's blow-up
    regime) and ``eegprep.clean_asr`` returns **4.6e15 µV** from a ~13 µV
    input. Without the B7-N2 guard the garbage flows into the block-rejection
    gate, which then reports the recording as *excluded for block
    contamination* — a misattributed backend failure.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * SFREQ)
    data = np.cumsum(rng.standard_normal((len(CH_NAMES), n)), axis=1)
    data -= data.mean(axis=1, keepdims=True)
    data *= 10e-6 / data.std(axis=1, keepdims=True)
    for dup, src in (("Pz", "P3"), ("P4", "P8"), ("C4", "C3")):
        data[CH_NAMES.index(dup)] = (
            data[CH_NAMES.index(src)] + 1e-9 * rng.standard_normal(n))
    info = mne.create_info(CH_NAMES, SFREQ, "eeg")
    raw = mne.io.RawArray(data, info, first_samp=1000, verbose=False)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                    match_case=False, on_missing="ignore")
    raw.filter(0.1, 45.0, verbose=False)
    raw.info["bads"] = []
    return raw


def test_asr_blowup_guard_fires_on_real_eegprep_output(monkeypatch):
    """B7-N2: real eegprep garbage (4.6e15 µV) fails loudly in the ASR step.

    The guard must fire *inside* the step (reason ``asr_output_magnitude_blowup``)
    and must not have written the garbage back into the caller's recording, so
    the block-rejection gate can never misattribute the blow-up to the data.

    R1 (ASR scoped audit, 2026-10-16): the step now runs ``clean_asr`` under the
    MATLAB-equivalent pinv tolerance, which fixes *this very fixture* — so the
    default-tolerance regime is reproduced here by disabling the patch
    (``_MatlabPinvTolerancePatch`` -> a no-op context manager). The guard stays
    as defense-in-depth: a flat channel entering ASR still blows up even with
    the patch (see ``test_reference_channel_excluded``), and future backend
    regressions must still fail here rather than in the block gate.
    """
    from eegcpm.modules.preprocessing.steps import asr as asr_mod

    with monkeypatch.context() as m:
        m.setattr(asr_mod, "_MatlabPinvTolerancePatch", contextlib.nullcontext)

        raw = _near_duplicate_hp_raw()
        input_max_uv = float(np.abs(raw.get_data()).max()) * 1e6
        assert input_max_uv < 1e3  # sanity: a µV-scale fixture

        with pytest.raises(RuntimeError,
                           match="asr_output_magnitude_blowup") as excinfo:
            ASRStep(calibration="fixed_first_60s", train_duration=60.0,
                    drift_transition_hz=None).process(raw, {})

        message = str(excinfo.value)
        assert "input max" in message and "output max" in message
        # The failed run must not leave the caller's data corrupted (the guard
        # runs before the write-back).
        assert float(np.abs(raw.get_data()).max()) * 1e6 == pytest.approx(
            input_max_uv)
        # Attribution: the message names the eegprep failure mode, not the data.
        assert "clean_asr" in message and "sqrtm" in message

    # R1 non-vacuity: the same fixture is now clean even in the non-high-passed
    # (drift-off) regime — the fix, not the high-pass, is what tames it.
    raw2 = _near_duplicate_hp_raw()
    _, meta = ASRStep(calibration="fixed_first_60s", train_duration=60.0,
                      drift_transition_hz=None).process(raw2, {})
    assert np.isfinite(raw2.get_data()).all()
    assert meta["output_magnitude_ratio"] < 2.0, (
        "the MATLAB-tolerance pinv no longer tames the near-duplicate fixture "
        f"(ratio {meta['output_magnitude_ratio']:.3g})"
    )

    # ... and the EEGLAB-standard path (the chain default) is clean too.
    raw3 = _near_duplicate_hp_raw()
    _, meta3 = ASRStep(calibration="fixed_first_60s",
                       train_duration=60.0).process(raw3, {})
    assert meta3["drift_filter_applied"] is True
    assert np.isfinite(raw3.get_data()).all()
    assert meta3["output_magnitude_ratio"] < 2.0


def test_asr_blowup_guard_reports_pre_and_post_maxima(monkeypatch):
    """Deterministic guard test: a scaled-up output fails with the maxima.

    Monkeypatching ``clean_asr`` keeps this independent of eegprep's numeric
    behaviour, while the real-output test above pins the actual failure mode.
    """
    import eegprep

    raw = _make_raw()
    real = eegprep.clean_asr

    def _scaled(EEG, **kwargs):
        out = dict(real(EEG, **kwargs))
        out["data"] = np.asarray(out["data"], dtype=float) * 1e9
        return out

    monkeypatch.setattr(eegprep, "clean_asr", _scaled)
    with pytest.raises(RuntimeError, match="asr_output_magnitude_blowup") as excinfo:
        ASRStep().process(raw, {})

    message = str(excinfo.value)
    assert "1000 x" in message
    # Pre/post maxima are reported (µV), not just a boolean.
    assert "µV" in message


def test_asr_guard_does_not_fire_on_legitimate_repair():
    """Non-vacuity: a legitimate ASR repair stays far below the guard factor."""
    from eegcpm.modules.preprocessing.steps.asr import ASR_OUTPUT_BLOWUP_FACTOR

    raw = _make_raw()
    _, meta = ASRStep().process(raw, {})
    assert meta["output_magnitude_ratio"] is not None
    assert meta["output_magnitude_ratio"] < ASR_OUTPUT_BLOWUP_FACTOR
    assert meta["pre_max_uv"] > 0 and meta["post_max_uv"] > 0


def test_asr_blowup_is_never_reported_as_block_exclusion(tmp_path, monkeypatch):
    """B7-N2: a blow-up fails the ASR step, never the block-rejection gate.

    The garbage output would make every 2-s block exceed the block gate's
    100 µV threshold, so without the in-step guard the run would be reported as
    ``excluded`` for *block contamination* — a backend failure blamed on the
    data. Through the executor: ``asr`` is ``failed``, the run has no FIF and a
    FAILED sidecar, and there is no ``excluded`` record.

    R1: the step's MATLAB-tolerance pinv fixes this fixture's blow-up, so the
    default-tolerance regime is reproduced by disabling the patch — the point
    here is the *attribution* (step failure, never a data exclusion) if any
    backend ever returns garbage again.
    """
    from eegcpm.modules.preprocessing.steps import asr as asr_mod

    raw = _near_duplicate_hp_raw()
    pipeline = PreprocessingPipeline(
        [{"name": "asr",
          # The non-high-passed regime (see the guard test above).
          "params": {"calibration": "fixed_first_60s",
                     "drift_transition_hz": None}}],
        tmp_path,
    )
    with monkeypatch.context() as m:
        m.setattr(asr_mod, "_MatlabPinvTolerancePatch", contextlib.nullcontext)
        result = pipeline.process(raw, subject_id="blowup")

    assert result.success is False
    assert result.metadata["step_status"]["asr"] == "failed"
    assert "excluded" not in result.metadata
    assert "asr_output_magnitude_blowup" in str(
        result.metadata["asr"].get("error", ""))
    assert list(tmp_path.glob("*_preprocessed_raw.fif")) == []
    assert (tmp_path / "blowup_preprocessing_FAILED.json").exists()




# ----------------------------------------------------------------------
# §12.6 — EEGLAB-standard ASR: clean_drifts input high-pass + max_dims
# ----------------------------------------------------------------------

def _drift_fixture(drift_hz=0.2, drift_uv=300.0, duration_s=120.0, seed=5):
    """32-channel 120-s fixture with a strong sub-transition (0.2 Hz) drift.

    Filtered 0.1-45 Hz like the chain's step-3 output, so the 0.2 Hz drift is
    still in the data when ASR runs: the EEGLAB-standard ``clean_drifts``
    (0.25, 0.75) must remove it (an 80 dB stop-band high-pass), while ASR
    alone (the pre-§12.6 path) barely touches it.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * SFREQ)
    t = np.arange(n) / SFREQ
    data = rng.standard_normal((len(CH_NAMES), n)) * 10e-6
    data += (drift_uv * 1e-6) * np.sin(2 * np.pi * drift_hz * t)[None, :]
    info = mne.create_info(CH_NAMES, SFREQ, "eeg")
    raw = mne.io.RawArray(data, info, first_samp=1000, verbose=False)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                    match_case=False, on_missing="ignore")
    raw.filter(0.1, 45.0, verbose=False)
    raw.info["bads"] = []
    return raw


def _band_power(data, sfreq, lo, hi, nperseg_s=20.0):
    """Mean Welch PSD in [lo, hi) Hz over channels."""
    from scipy.signal import welch

    nperseg = min(int(nperseg_s * sfreq), data.shape[1])
    freqs, psd = welch(data, sfreq, nperseg=nperseg, axis=1)
    band = (freqs >= lo) & (freqs < hi)
    return float(psd[:, band].mean())


def test_drift_highpass_applied_to_asr_input(monkeypatch):
    """§12.6: clean_drifts runs on the ASR input, before clean_asr.

    The EEGLAB-standard path is ``clean_drifts(EEG, (0.25, 0.75))`` then ASR
    (``clean_artifacts`` does exactly this; ``clean_asr`` requires high-passed
    input). The data handed to ``clean_asr`` must be the drift-filtered data —
    asserted by object identity, because ``clean_drifts`` mutates and returns
    the same EEG dict.
    """
    import eegprep

    calls = []
    seen = {}
    real_drifts = eegprep.clean_drifts
    real_asr = eegprep.clean_asr

    def spy_drifts(EEG, transition=(0.5, 1.0), **kwargs):
        calls.append(("clean_drifts", tuple(transition)))
        out = real_drifts(EEG, transition, **kwargs)
        seen["drift_output"] = out
        return out

    def spy_asr(EEG, **kwargs):
        calls.append(("clean_asr", kwargs.get("cutoff")))
        seen["asr_input"] = EEG
        return real_asr(EEG, **kwargs)

    monkeypatch.setattr(eegprep, "clean_drifts", spy_drifts)
    monkeypatch.setattr(eegprep, "clean_asr", spy_asr)

    raw = _make_raw()
    out, meta = ASRStep().process(raw, {})

    assert [name for name, _ in calls] == ["clean_drifts", "clean_asr"], (
        f"expected clean_drifts before clean_asr; got {calls}")
    assert calls[0][1] == (0.25, 0.75)
    assert seen["asr_input"] is seen["drift_output"], (
        "clean_asr did not receive the clean_drifts output")
    assert meta["drift_filter_applied"] is True
    assert meta["drift_transition_hz"] == [0.25, 0.75]


def test_drift_highpass_removes_sub_transition_content():
    """§12.6: the write-back has the sub-transition band removed (band power).

    The EEGLAB-standard path keeps the high-passed data (no add-back), so the
    step output must have far less 0.1-0.4 Hz power than the input while the
    8-45 Hz band is preserved (ASR repairs, it does not low-pass).
    """
    raw = _drift_fixture()
    pre = raw.get_data(picks="eeg")
    pre_low = _band_power(pre, SFREQ, 0.1, 0.4)
    pre_high = _band_power(pre, SFREQ, 8.0, 45.0)
    assert pre_low > 0, "fixture has no sub-transition power to remove"

    out, meta = ASRStep().process(raw, {})
    post = out.get_data(picks="eeg")

    low_ratio = _band_power(post, SFREQ, 0.1, 0.4) / pre_low
    high_ratio = _band_power(post, SFREQ, 8.0, 45.0) / pre_high
    assert low_ratio < 0.1, (
        f"0.1-0.4 Hz power ratio {low_ratio:.3g}: the clean_drifts high-pass "
        "did not reach the write-back")
    assert 0.5 < high_ratio < 2.0, (
        f"8-45 Hz power ratio {high_ratio:.3g}: ASR changed the passband band "
        "power far beyond repair expectations")
    assert meta["drift_filter_modified_fraction"] > 0.0


def test_drift_off_skips_filter_with_warning(monkeypatch):
    """``None``/``'off'`` skips the high-pass, warns, and records it.

    The non-EEGLAB regime (ASR on non-high-passed data) is still expressible,
    but it must be loud: ``clean_drifts`` is not called, a RuntimeWarning is
    emitted, and the metadata states ``drift_filter_applied: False`` so a
    sidecar can never claim the EEGLAB path was run.
    """
    import eegprep

    called = []
    real_drifts = eegprep.clean_drifts
    monkeypatch.setattr(
        eegprep, "clean_drifts",
        lambda EEG, *a, **k: (called.append(True), real_drifts(EEG, *a, **k))[1])

    assert ASRStep(drift_transition_hz="off").drift_transition_hz is None
    assert ASRStep(drift_transition_hz=None).drift_transition_hz is None

    raw = _make_raw()
    with pytest.warns(RuntimeWarning, match="drift_transition_hz is off"):
        out, meta = ASRStep(drift_transition_hz=None).process(raw, {})

    assert called == [], "clean_drifts ran although drift_transition_hz=None"
    assert meta["drift_filter_applied"] is False
    assert meta["drift_transition_hz"] is None
    assert meta["drift_filter_modified_fraction"] is None
    assert meta["max_dims"] == 0.66

    # Malformed transition bands are refused at construction.
    for bad in (0.5, (1.0, 0.5), (0.0, 0.75), (0.25, 0.75, 1.0), "nope"):
        with pytest.raises(ValueError, match="drift_transition_hz"):
            ASRStep(drift_transition_hz=bad)
    for bad_dims in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="max_dims"):
            ASRStep(max_dims=bad_dims)


def test_max_dims_reaches_clean_asr(monkeypatch):
    """§12.6: the configured ``max_dims`` is the value clean_asr receives."""
    import eegprep

    seen = {}
    real_asr = eegprep.clean_asr

    def spy_asr(EEG, **kwargs):
        seen.update(kwargs)
        return real_asr(EEG, **kwargs)

    monkeypatch.setattr(eegprep, "clean_asr", spy_asr)

    raw = _make_raw()
    _, meta = ASRStep().process(raw, {})
    assert seen["max_dims"] == 0.66
    assert meta["max_dims"] == 0.66

    raw2 = _make_raw()
    _, meta2 = ASRStep(max_dims=0.5).process(raw2, {})
    assert seen["max_dims"] == 0.5
    assert meta2["max_dims"] == 0.5

    # Through the chain wrapper: the configured value is what the step runs.
    raw3 = _make_raw()
    _, meta3 = _ASRStepWrapper(max_dims=0.4).process(
        raw3, {"channel_roles": {"reference_channel": "Cz"}})
    assert seen["max_dims"] == 0.4
    assert meta3["max_dims"] == 0.4


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
    # Plan §12.6: the chain default runs the EEGLAB-standard path and the
    # wrapper's metadata reports it.
    assert meta["drift_filter_applied"] is True
    assert meta["drift_transition_hz"] == [0.25, 0.75]
    assert meta["max_dims"] == 0.66

    # An unsupported calibration is refused instead of silently ignored ...
    with pytest.raises(ValueError, match="fixed_first_60s"):
        _ASRStepWrapper(calibration_window="not_a_calibration")

    # ... while the per-dataset data-driven calibration (EEGLAB's automatic
    # clean-window selection) is now supported and reported truthfully.
    dd = _ASRStepWrapper(calibration_window="data_driven")
    assert dd.validate_inputs(_make_raw()) is True
    cfg = dd.get_config()
    assert cfg["calibration_window"] == "data_driven"
    assert cfg["calibration_reference"] == "eegprep_clean_windows_selection"
    assert cfg["ref_maxbadchannels"] == 0.075
    assert cfg["ref_tolerances"] == [-np.inf, 5.5]


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
    # §12.6: the EEGLAB-standard path is what the wrapper actually runs.
    assert calls["max_dims"] == 0.66

    cfg = wrapper.get_config()
    assert cfg["calibration_reference"] == "first_train_duration_s_of_data"
    assert cfg["ref_maxbadchannels"] == "calibration_array"
    assert cfg["ref_tolerances"] == "eegprep_default"
    assert cfg["drift_transition_hz"] == [0.25, 0.75]
    assert cfg["max_dims"] == 0.66


def test_asr_degenerate_calibration_covariance_raises(monkeypatch):
    """The deterministic 'degenerate' branch must fire for a truly singular
    calibration covariance.

    The B7 validator found this branch practically unreachable: float rounding
    of an exactly rank-deficient covariance yields a tiny positive eigenvalue,
    so the magnitude guard is the operative defense. This test forces the
    singular case (np.cov stubbed) so the error code itself stays pinned.
    """
    import numpy as _np

    def singular_cov(data, *args, **kwargs):
        return _np.array([[0.0, 0.0], [0.0, 1.0]])

    monkeypatch.setattr(_np, "cov", singular_cov)

    raw = _make_raw()
    step = ASRStep(calibration="fixed_first_60s", train_duration=60.0)
    with pytest.raises(RuntimeError, match="asr_calibration_covariance_degenerate"):
        step.process(raw, {})


# ----------------------------------------------------------------------
# R1 (ASR scoped audit, 2026-10-16): MATLAB-equivalent pinv tolerance
# ----------------------------------------------------------------------

def _pinv_reproducer_raw(sinusoid_uv=20.0, seed=11, duration_s=120.0):
    """32-channel fixture whose ASR blow-up depends on the pinv tolerance.

    The auditor's starting point (32 channels, calibration covariance
    condition number ~8.7e3, a 10 Hz common sinusoid) reproduced
    deterministically: cumulative-sum 1/f noise (10 µV) plus a 20 µV common
    10 Hz sinusoid, 0.1-45 Hz filtered — measured calibration covariance
    condition number 1.2e4 and a 38 µV input maximum. With eegprep's default
    ``np.linalg.pinv`` (``rcond=1e-15``) ``clean_asr`` returns garbage at
    ~1e13 µV (caught by the ``asr_output_magnitude_blowup`` guard); with
    the MATLAB tolerance ``max(size) * eps(norm(A))`` the same call is clean
    (ratio ~1). Probed 2026-10-16; this is the regression test for the R1 fix.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * SFREQ)
    data = np.cumsum(rng.standard_normal((len(CH_NAMES), n)), axis=1)
    data -= data.mean(axis=1, keepdims=True)
    data *= 10e-6 / data.std(axis=1, keepdims=True)
    t = np.arange(n) / SFREQ
    data += (sinusoid_uv * 1e-6) * np.sin(2 * np.pi * 10.0 * t)[None, :]
    info = mne.create_info(CH_NAMES, SFREQ, "eeg")
    raw = mne.io.RawArray(data, info, first_samp=1000, verbose=False)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                    match_case=False, on_missing="ignore")
    raw.filter(0.1, 45.0, verbose=False)
    raw.info["bads"] = []
    return raw


def test_matlab_pinv_tolerance_helper_matches_matlab_semantics():
    """R1: the patched pinv truncates exactly what MATLAB's pinv truncates.

    MATLAB's ``pinv`` zeroes singular values <= ``max(size(A)) * eps(norm(A))``
    (an ABSOLUTE cutoff); numpy's default ``rcond=1e-15`` is relative. On a
    64x64 matrix with a 3e-15 singular value (s_max = 1) numpy inverts it
    (|pinv| ~ 4e13) while the MATLAB tolerance (64 * spacing(1) ~ 1.4e-14)
    zeroes it. A well-conditioned matrix must come back identical.
    """
    from eegcpm.modules.preprocessing.steps.asr import (
        _matlab_pinv_tolerance,
        _pinv_with_matlab_tolerance,
    )

    rng = np.random.default_rng(0)
    q1, _ = np.linalg.qr(rng.standard_normal((64, 64)))
    q2, _ = np.linalg.qr(rng.standard_normal((64, 64)))
    a = q1 @ np.diag(np.array([1.0, 3e-15] + [0.5] * 62)) @ q2.T

    assert _matlab_pinv_tolerance(a) == pytest.approx(
        float(np.spacing(1.0)) * 64, rel=1e-3)
    assert np.abs(np.linalg.pinv(a)).max() > 1e12
    assert np.abs(_pinv_with_matlab_tolerance(a)).max() < 1e3

    b = rng.standard_normal((16, 16))
    np.testing.assert_allclose(
        _pinv_with_matlab_tolerance(b), np.linalg.pinv(b),
        rtol=1e-9, atol=1e-12)


def test_pinv_tolerance_is_the_blowup_fix(monkeypatch):
    """R1: both directions on the deterministic reproducer.

    Control (the real eegprep default ``np.linalg.pinv``): the step raises
    ``asr_output_magnitude_blowup`` and writes nothing back. Fixed (the step's
    MATLAB-tolerance patch): the same fixture is repaired cleanly.
    """
    from eegcpm.modules.preprocessing.steps import asr as asr_mod

    kwargs = dict(calibration="fixed_first_60s", train_duration=60.0,
                  drift_transition_hz=None)

    with monkeypatch.context() as m:
        m.setattr(asr_mod, "_MatlabPinvTolerancePatch", contextlib.nullcontext)
        raw = _pinv_reproducer_raw()
        input_max_uv = float(np.abs(raw.get_data()).max()) * 1e6
        with pytest.raises(RuntimeError,
                           match="asr_output_magnitude_blowup") as excinfo:
            ASRStep(**kwargs).process(raw, {})
        assert "ratio" in str(excinfo.value)
        # The guard runs before the write-back: no garbage in the caller's data.
        assert float(np.abs(raw.get_data()).max()) * 1e6 == pytest.approx(
            input_max_uv)

    raw2 = _pinv_reproducer_raw()
    out, meta = ASRStep(**kwargs).process(raw2, {})
    assert meta["pinv_tolerance"] == asr_mod.PINV_TOLERANCE_SPEC
    assert meta["pinv_tolerance"] == "matlab_max_size_eps(norm)"
    assert meta["output_magnitude_ratio"] < 2.0, (
        "the MATLAB-tolerance pinv did not tame the reproducer "
        f"(ratio {meta['output_magnitude_ratio']:.3g})")
    assert np.isfinite(out.get_data()).all()


def test_pinv_patch_is_scoped_and_guarded(monkeypatch):
    """R1: the patch is restored after the call and refuses a changed call site.

    The version guard is source-based: if the installed eegprep no longer calls
    ``finite_pinv(Q)`` (or the module global is gone), the step must raise
    ``eegprep_pinv_assumption_broken`` instead of silently reverting to the
    default numpy tolerance.
    """
    import importlib

    from eegcpm.modules.preprocessing.steps import asr as asr_mod

    asr_module = importlib.import_module(
        "eegprep.plugins.clean_rawdata.asr_process")
    original = asr_module.finite_pinv

    _, meta = ASRStep().process(_make_raw(), {})
    assert asr_module.finite_pinv is original, (
        "the pinv patch leaked out of the clean_asr call")
    assert meta["pinv_tolerance"] == asr_mod.PINV_TOLERANCE_SPEC
    assert meta["eegprep_version"]
    assert (meta["eegprep_pinv_verified_version"]
            == asr_mod.EEGPREP_PINV_VERIFIED_VERSION)

    # A changed call site (source no longer calls finite_pinv(Q)) is refused.
    def fake_asr_process(*args, **kwargs):  # noqa: ARG001 - never called
        raise AssertionError("the guard must fire before clean_asr runs")

    with monkeypatch.context() as m:
        m.setattr(asr_module, "asr_process", fake_asr_process)
        with pytest.raises(RuntimeError, match="eegprep_pinv_assumption_broken"):
            ASRStep().process(_make_raw(), {})

    # A missing module global is refused too.
    with monkeypatch.context() as m:
        m.delattr(asr_module, "finite_pinv")
        with pytest.raises(RuntimeError, match="eegprep_pinv_assumption_broken"):
            ASRStep().process(_make_raw(), {})


# ----------------------------------------------------------------------
# R4 (ASR scoped audit): truthful info['highpass']
# ----------------------------------------------------------------------

def test_info_highpass_updated_when_drift_filter_runs():
    """R4: the FIF's info['highpass'] describes the high-passed write-back.

    MNE's own ``filter`` records ``info['highpass'] = l_freq`` (the passband
    edge); the equivalent edge for ``clean_drifts(0.25, 0.75)`` is 0.75 Hz.
    Before R4 the FIF kept the chain's stale 0.1 Hz value.
    """
    raw = _make_raw()
    assert raw.info["highpass"] == 0.5          # the fixture's own filter
    out, meta = ASRStep().process(raw, {})
    assert meta["drift_filter_applied"] is True
    assert meta["info_highpass_hz"] == 0.75
    assert out.info["highpass"] == 0.75
    # The band stays fully recorded in the metadata.
    assert meta["drift_transition_hz"] == [0.25, 0.75]

    # A custom band records its own passband edge.
    out2, meta2 = ASRStep(drift_transition_hz=(0.5, 1.0)).process(
        _make_raw(), {})
    assert meta2["info_highpass_hz"] == 1.0
    assert out2.info["highpass"] == 1.0


def test_info_highpass_untouched_when_drift_filter_off():
    """R4: with the filter skipped the recording keeps its own value."""
    raw = _make_raw()
    with pytest.warns(RuntimeWarning, match="drift_transition_hz is off"):
        out, meta = ASRStep(drift_transition_hz=None).process(raw, {})
    assert meta["info_highpass_hz"] is None
    assert out.info["highpass"] == 0.5


# ----------------------------------------------------------------------
# R6 (ASR scoped audit): runtime Nyquist bound on the drift band
# ----------------------------------------------------------------------

def test_drift_transition_above_nyquist_refused():
    """R6: a drift transition band reaching Nyquist is refused at runtime.

    The schema bound (<= 5 Hz) cannot see the recording's sampling rate, so the
    step checks the upper edge against Nyquist in ``validate_inputs`` (the
    executor's pre-flight) **and** at the start of the eegprep path (a direct
    ``process()`` call must not bypass it).
    """
    rng = np.random.default_rng(0)
    info = mne.create_info(CH_NAMES, 4.0, "eeg")
    raw = mne.io.RawArray(
        rng.standard_normal((len(CH_NAMES), 80)) * 1e-6, info, verbose=False)

    step = ASRStep(drift_transition_hz=(0.5, 2.5), calibration="data_driven")
    with pytest.raises(ValueError, match="Nyquist"):
        step.validate_inputs(raw)
    with pytest.raises(ValueError, match="Nyquist"):
        step.process(raw, {})

    # A band below Nyquist passes the check (data-driven calibration works on
    # the short recording).
    ok = ASRStep(drift_transition_hz=(0.25, 0.75), calibration="data_driven")
    assert ok.validate_inputs(raw) is True


def test_drift_filter_never_lowers_an_existing_highpass():
    """N1 (scoped re-audit): explicit configs may already high-pass at 1 Hz;
    the drift filter must report the effective edge (max), never under-report."""
    raw = _make_raw()
    raw.filter(1.0, 40.0, verbose=False)
    assert raw.info["highpass"] == 1.0

    step = ASRStep()
    out, meta = step.process(raw, {})

    assert out.info["highpass"] == 1.0
    assert meta["info_highpass_hz"] == 1.0
