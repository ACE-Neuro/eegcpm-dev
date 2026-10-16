"""B6 tier-2 step fixes (plan §B6 tasks 9-15; P1-F9, F13, F16, F18, F20, F21, F22).

- zapline: `noisefreqs=[fline]` when a line frequency is set, removed-component
  counts from `analytics` (KeyError when the key is missing), detected
  frequencies from the zapline config (KeyError, never an invented default);
- EOG regression: one multi-target `lstsq` (equals the old per-channel
  normal-equations solution), condition number of the z-scored design (invariant
  to the EOG scale), recorded spec deviation;
- reference: `type='rest'` raises `NotImplementedError`, post-reference channel
  variance check;
- ICA `rank-1`: `min(max(5, r-1), r-1)` and `ValueError` when `r-1 < 1`;
- montage: a channel without a valid position after `set_montage` fails the step
  (never dropped silently);
- `BadChannelDetector`: correlation over montage neighbours with `np.nanmedian`,
  flat channel = bad; a failing RANSAC/PREP path fails instead of reporting zero
  bad channels.
"""

from __future__ import annotations

import sys
import types

import mne
import numpy as np
import pytest

from eegcpm.modules.preprocessing.steps.eog_regression import (
    MAX_HF_POWER_FRACTION,
    SPEC_DEVIATION,
    EOGRegressionStep,
)
from eegcpm.modules.preprocessing.steps.ica import ICAStep
from eegcpm.modules.preprocessing.steps.montage import MontageStep
from eegcpm.modules.preprocessing.steps.reference import ReferenceStep
from eegcpm.modules.preprocessing.steps.zapline import ZaplineStep


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _raw(ch_names, sfreq=250.0, n_times=2500, seed=0, ch_types="eeg"):
    rng = np.random.RandomState(seed)
    info = mne.create_info(list(ch_names), sfreq, ch_types)
    return mne.io.RawArray(rng.randn(len(ch_names), n_times) * 1e-6, info)


def _montaged_raw(ch_names=None, seed=0, n_times=2500):
    ch_names = ch_names or [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "C4",
        "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2", "Cz",
    ]
    raw = _raw(ch_names, n_times=n_times, seed=seed)
    raw.set_montage(
        mne.channels.make_standard_montage("standard_1020"),
        match_case=False, on_missing="ignore",
    )
    return raw


# ---------------------------------------------------------------------------
# zapline (P1-F9, plan task 9)
# ---------------------------------------------------------------------------

class _FakeZapline:
    """Records the kwargs and returns a configurable 4-tuple."""

    def __init__(self, noisefreqs=(60.0,), n_remove=(2.0,), analytics_keys=None,
                 config_has_noisefreqs=True):
        self.calls = []
        self.noisefreqs = list(noisefreqs)
        self.n_remove = np.asarray(n_remove, dtype=float)
        self.analytics_keys = analytics_keys
        self.config_has_noisefreqs = config_has_noisefreqs

    def __call__(self, data, sampling_rate, **kwargs):
        self.calls.append({"shape": np.asarray(data).shape, **kwargs})
        config = {"noisefreqs": list(self.noisefreqs)}
        if not self.config_has_noisefreqs:
            config = {}
        if self.analytics_keys is not None:
            analytics = {
                key: {"n_remove_final": self.n_remove}
                for key in self.analytics_keys
            }
        else:
            analytics = {
                f"noise_freq_{freq}": {"n_remove_final": self.n_remove}
                for freq in self.noisefreqs
            }
        # Deliberately different from the input so a premature write-back
        # (B6-K7) is detectable in the KeyError tests.
        return np.asarray(data).copy() + 1e-9, config, analytics, []


def test_zapline_passes_fline_as_noisefreqs(monkeypatch):
    """P1-F9: `noisefreqs=[fline]` when fline is set (never the auto list)."""
    import pyzaplineplus

    fake = _FakeZapline()
    monkeypatch.setattr(pyzaplineplus, "zapline_plus", fake)

    raw = _raw([f"E{i}" for i in range(1, 9)], n_times=2500)
    step = ZaplineStep(fline=60.0)
    out, meta = step.process(raw, {})

    assert fake.calls and fake.calls[0]["noisefreqs"] == [60.0]
    assert meta["noisefreqs_passed"] == [60.0]
    assert meta["detected_freqs"] == [60.0]
    assert meta["detected_freq"] == 60.0
    assert meta["n_components_removed"] == {"60.0": 2}
    assert meta["n_components_removed_by_chunk"] == {"60.0": [2]}
    assert meta["n_components_removed_total"] == 2
    assert meta["applied"] is True
    assert out is raw  # written in place


def test_zapline_auto_detection_uses_configured_noisefreqs(monkeypatch):
    """fline=None -> the step's own `noisefreqs` ([] = auto) is passed."""
    import pyzaplineplus

    fake = _FakeZapline(noisefreqs=(50.0,))
    monkeypatch.setattr(pyzaplineplus, "zapline_plus", fake)

    raw = _raw([f"E{i}" for i in range(1, 9)])
    _, meta = ZaplineStep(fline=None, noisefreqs=[50.0]).process(raw, {})
    assert fake.calls[0]["noisefreqs"] == [50.0]
    assert meta["noisefreqs_passed"] == [50.0]

    _, meta_auto = ZaplineStep(fline=None).process(raw, {})
    assert fake.calls[1]["noisefreqs"] == []
    assert meta_auto["noisefreqs_passed"] == []


def test_zapline_missing_analytics_key_raises(monkeypatch):
    """A missing `noise_freq_<f>` analytics entry raises KeyError (no default)."""
    import pyzaplineplus

    fake = _FakeZapline(analytics_keys=[])  # empty analytics
    monkeypatch.setattr(pyzaplineplus, "zapline_plus", fake)

    raw = _raw([f"E{i}" for i in range(1, 9)])
    before = raw.get_data().copy()
    with pytest.raises(KeyError):
        ZaplineStep(fline=60.0).process(raw, {})
    # B6-K7: the KeyError is raised while reading the library outputs, before
    # the write-back — the raw data must be untouched.
    assert np.array_equal(raw.get_data(), before)


def test_zapline_missing_config_noisefreqs_raises(monkeypatch):
    """A config without `noisefreqs` raises KeyError (no invented default)."""
    import pyzaplineplus

    fake = _FakeZapline(config_has_noisefreqs=False)
    monkeypatch.setattr(pyzaplineplus, "zapline_plus", fake)

    raw = _raw([f"E{i}" for i in range(1, 9)])
    before = raw.get_data().copy()
    with pytest.raises(KeyError):
        ZaplineStep(fline=60.0).process(raw, {})
    # B6-K7: the KeyError is raised while reading the library outputs, before
    # the write-back — the raw data must be untouched.
    assert np.array_equal(raw.get_data(), before)


def test_zapline_real_run_reports_library_counts():
    """Integration: the real backend's analytics drive the metadata."""
    rng = np.random.RandomState(0)
    sfreq = 500.0
    n_times = int(30 * sfreq)
    t = np.arange(n_times) / sfreq
    data = rng.randn(n_times, 8) * 1e-6
    data += 20e-6 * np.sin(2 * np.pi * 60.0 * t)[:, None]
    raw = mne.io.RawArray(data.T, mne.create_info(
        [f"E{i}" for i in range(1, 9)], sfreq, "eeg"))

    out, meta = ZaplineStep(fline=60.0).process(raw, {})
    assert meta["applied"] is True
    assert meta["noisefreqs_passed"] == [60.0]
    assert meta["detected_freqs"] == [60.0]
    assert meta["auto_detected"] is False
    assert meta["n_components_removed_total"] >= 1
    assert set(meta["n_components_removed"]) == {"60.0"}
    # The line noise power actually dropped.
    assert meta["line_reduction_db"] > 0


def test_zapline_validate_inputs():
    raw = _raw([f"E{i}" for i in range(1, 9)])
    assert ZaplineStep(fline=60.0).validate_inputs(raw) is True
    with pytest.raises(ValueError, match="Nyquist"):
        ZaplineStep(fline=200.0).validate_inputs(raw)
    # fline=None skips the Nyquist check entirely.
    assert ZaplineStep(fline=None).validate_inputs(raw) is True
    with pytest.raises(ValueError, match="at least 2 EEG channels"):
        ZaplineStep().validate_inputs(_raw(["E1"]))


# ---------------------------------------------------------------------------
# EOG regression (P1-F13, plan task 10)
# ---------------------------------------------------------------------------

def _eog_raw(n_eeg=4, n_eog=3, n_times=2000, seed=0, sfreq=250.0, filtered=True):
    """EEG + EOG fixture.

    ``filtered=True`` (default) applies the step's documented input contract:
    the regressors must be band-limited (the canonical chain filters the EEG
    **and EOG** picks). ``filtered=False`` is the unfiltered case the
    band-limited guard refuses (B7, carried B6 item).
    """
    rng = np.random.RandomState(seed)
    eeg_names = [f"E{i}" for i in range(1, n_eeg + 1)]
    eog_names = [f"VEOG{i}" for i in range(1, n_eog + 1)]
    eog_sources = rng.randn(n_eog, n_times) * 50e-6
    # EEG = mixture of the EOG sources + independent brain signal
    mixing = rng.randn(n_eeg, n_eog) * 0.3
    eeg = mixing @ eog_sources + rng.randn(n_eeg, n_times) * 10e-6
    data = np.vstack([eeg, eog_sources])
    info = mne.create_info(eeg_names + eog_names, sfreq,
                           ["eeg"] * n_eeg + ["eog"] * n_eog)
    raw = mne.io.RawArray(data, info)
    if filtered:
        # 0.5 Hz (not 0.1) keeps the FIR length inside this short fixture; the
        # guard only constrains the upper edge. Picks mirror the canonical
        # chain: mne's default picks exclude EOG channels (B6 rev2 NEW-1).
        raw.filter(0.5, 45.0, picks=("eeg", "eog"), verbose=False)
    return raw


def test_eog_regression_equals_per_channel_normal_equations():
    """The single multi-target lstsq matches the old per-channel solve."""
    raw = _eog_raw(seed=1)
    step = EOGRegressionStep(eog_channels=["VEOG1", "VEOG2", "VEOG3"])
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    eog_picks = mne.pick_types(raw.info, eog=True, exclude="bads")
    data = raw.get_data(picks=np.concatenate([eeg_picks, eog_picks]))
    eeg_data = data[:len(eeg_picks)]
    eog_data = data[len(eeg_picks):]
    X = np.vstack([eog_data, np.ones((1, eog_data.shape[1]))])

    # Reference: the pre-B6 per-channel normal-equations solve.
    n_coef = X.shape[0]
    B_ref = np.zeros((eeg_data.shape[0], n_coef))
    for i in range(eeg_data.shape[0]):
        A = X @ X.T
        b = eeg_data[i] @ X.T
        B_ref[i] = np.linalg.lstsq(A, b, rcond=None)[0]
    expected = eeg_data - B_ref @ X

    out, meta = step.process(raw, {})
    cleaned = out.get_data(picks=eeg_picks)

    assert meta["solve"] == "lstsq_multi_target"
    assert np.allclose(cleaned, expected, atol=1e-10), (
        f"max |diff| = {np.abs(cleaned - expected).max():.3e}")


def test_eog_regression_condition_number_invariant_to_eog_scale():
    """Z-scoring the design rows makes the reported conditioning scale-free."""
    channels = ["VEOG1", "VEOG2", "VEOG3"]
    raw_small = _eog_raw(seed=2)
    _, meta_small = EOGRegressionStep(eog_channels=channels).process(
        raw_small, {})

    raw_big = _eog_raw(seed=2)
    eog_picks = mne.pick_types(raw_big.info, eog=True, exclude="bads")
    raw_big._data[eog_picks] *= 1e6
    _, meta_big = EOGRegressionStep(eog_channels=channels).process(
        raw_big, {})

    assert meta_small["condition_number"] == pytest.approx(
        meta_big["condition_number"], rel=1e-9)
    assert meta_small["condition_number"] > 0
    assert meta_small["rank"] == 4  # 3 EOG + intercept
    assert meta_small["rank_deficient"] is False
    assert meta_small["spec_deviation"] == SPEC_DEVIATION


# ---------------------------------------------------------------------------
# NEW-1 (B6 rev2): the filter step band-passes the EOG regressors
# ---------------------------------------------------------------------------

def _shared_component_raw(sfreq=250.0, duration=180.0, seed=5, n_eeg=8):
    """EEG + EOG sharing one ocular/environmental component (NEW-1 fixture).

    The shared component carries a blink (the part EOG regression must remove),
    a slow 0.02 Hz drift and 60 Hz line noise. The EEG adds private white noise
    and a small private drift; the EOG adds a small private drift. The
    amplitudes model the real failure mode (audit NEW-1): the EOG's drift/line
    are much larger than the EEG's, and with unfiltered EOG regressors the
    regression re-injects them into the already-filtered EEG.
    """
    rng = np.random.RandomState(seed)
    n_times = int(sfreq * duration)
    t = np.arange(n_times) / sfreq
    blink = np.exp(-0.5 * (((t % 3.0) - 1.0) / 0.05) ** 2)
    shared = (
        blink
        + 0.5 * np.sin(2 * np.pi * 0.02 * t)
        + 0.7 * np.sin(2 * np.pi * 60.0 * t)
    )
    eeg_private = rng.randn(n_eeg, n_times) * 10e-6
    eeg_private += 5e-6 * np.sin(2 * np.pi * 0.02 * t + 1.3)
    eeg = eeg_private + np.outer(np.linspace(0.2, 0.6, n_eeg), shared) * 60e-6
    eog1 = (150e-6 * shared
            + 5e-6 * np.sin(2 * np.pi * 0.02 * t + 0.9)
            + rng.randn(n_times) * 5e-6)
    eog2 = 100e-6 * shared + rng.randn(n_times) * 5e-6
    eog3 = rng.randn(n_times) * 5e-6
    data = np.vstack([eeg, eog1, eog2, eog3])
    info = mne.create_info(
        [f"E{i}" for i in range(1, n_eeg + 1)] + ["VEOG", "HEOG", "EOG3"],
        sfreq, ["eeg"] * n_eeg + ["eog"] * 3,
    )
    return mne.io.RawArray(data, info), t, blink


def _band_power(x, sfreq=250.0, band="hi"):
    from scipy import signal

    f, psd = signal.welch(
        x, fs=sfreq, nperseg=min(x.shape[1], int(60 * sfreq))
    )
    mask = {
        "lo": f < 0.1,
        "hi": f > 45.0,
        "line": (f >= 59.0) & (f <= 61.0),
    }[band]
    return psd[:, mask].mean(axis=1)


def _blink_locked_ptp(x, t, blink, sfreq=250.0):
    """Baseline-corrected blink-locked average (the blink component)."""
    peaks = np.where((blink[1:-1] > blink[:-2]) & (blink[1:-1] > blink[2:]))[0] + 1
    pre = int(0.3 * sfreq)
    peaks = [p for p in peaks if pre < p < x.shape[1] - pre]
    epochs = np.stack([x[:, p - pre:p + pre] for p in peaks])
    epochs = epochs - epochs[:, :, :int(0.1 * sfreq)].mean(axis=2, keepdims=True)
    return np.ptp(epochs.mean(axis=0), axis=1)


def test_filter_step_bandpasses_eog_channels():
    """NEW-1: the chain's filter step processes the EOG picks, not only EEG.

    ``raw.filter(picks=None)`` resolves to the *data* channels, which exclude
    EOG (probed 2026-10-16, mne 1.13.2) — the pre-fix wrapper filtered the EEG
    only, leaving the EOG regressors with their full DC drift and 60 Hz line
    noise. The wrapper now pins ``picks=('eeg', 'eog')`` (reported in the step
    metadata and ``get_config``).
    """
    from eegcpm.modules.preprocessing.hbn_langer_chain import _FilterStepWrapper

    raw, _, _ = _shared_component_raw()
    eog_picks = mne.pick_types(raw.info, eog=True)
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    raw_eog = raw.get_data(picks=eog_picks)
    raw_eeg = raw.get_data(picks=eeg_picks)
    line_before = _band_power(raw_eog, band="line")
    lo_before = _band_power(raw_eog, band="lo")
    eeg_line_before = _band_power(raw_eeg, band="line")

    step = _FilterStepWrapper(l_freq=0.1, h_freq=45.0)
    out, meta = step.process(raw, {})

    assert meta["picks"] == ["eeg", "eog"]
    assert step.get_config()["picks"] == ["eeg", "eog"]

    filtered_eog = out.get_data(picks=eog_picks)
    # The 60 Hz line on the EOG is attenuated exactly as it is on the EEG
    # (the FIR stopband gives ~-58 dB at 60 Hz).
    assert np.all(_band_power(filtered_eog, band="line") / line_before < 1e-4)
    # The slow drift is removed (< 1 % on the drift-carrying channels; the
    # white-noise EOG channel sits in the 0-0.2 Hz transition band, ratio ~0.8,
    # which is still <= 1: filtering never *adds* sub-0.1 Hz power).
    lo_ratio = _band_power(filtered_eog, band="lo") / lo_before
    assert np.all(lo_ratio <= 1.0), lo_ratio
    assert lo_ratio[0] < 0.01 and lo_ratio[1] < 0.01, lo_ratio
    # ... and the EEG is still filtered as before.
    assert np.all(
        _band_power(out.get_data(picks=eeg_picks), band="line")
        / eeg_line_before
        < 1e-4
    )


def test_eog_regression_after_filter_does_not_inject_eog_noise():
    """NEW-1: filtered EOG regressors do not re-inject drift/line/blinks.

    Chain order (filter EEG+EOG -> ``EOGRegressionStep``): the regression
    must not increase the EEG's power above 45 Hz or below 0.1 Hz and must
    remove the blink component. Measured on this fixture: <0.1 Hz ratio
    mean 0.25 (max 0.72), >45 Hz ratio mean 1.005 (max 1.010), blink-locked
    amplitude mean 0.22 (max 0.35) of the pre-regression value. The pre-fix
    contrast (EEG-only filter, the old wrapper's effective picks) re-injects
    the EOG's line noise (59-61 Hz power x5000 here; the audit's real-data
    probe measured x~14000) and leaves the blink in place, which is the
    failure mode this test guards against.
    """
    from eegcpm.modules.preprocessing.hbn_langer_chain import _FilterStepWrapper

    raw, t, blink = _shared_component_raw()
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    _FilterStepWrapper(l_freq=0.1, h_freq=45.0).process(raw, {})

    x_before = raw.get_data(picks=eeg_picks)
    lo_before = _band_power(x_before, band="lo")
    hi_before = _band_power(x_before, band="hi")
    blink_before = _blink_locked_ptp(x_before, t, blink)

    _, meta = EOGRegressionStep().process(raw, {})
    assert meta["applied"] is True
    x_after = raw.get_data(picks=eeg_picks)

    lo_ratio = _band_power(x_after, band="lo") / lo_before
    hi_ratio = _band_power(x_after, band="hi") / hi_before
    assert np.all(lo_ratio <= 1.0), lo_ratio
    assert np.all(hi_ratio <= 1.05), hi_ratio

    blink_ratio = _blink_locked_ptp(x_after, t, blink) / blink_before
    assert np.all(blink_ratio <= 0.5), blink_ratio

    # Contrast: EEG-only filtering (the pre-fix behaviour) re-injects the
    # EOG's line noise and does not remove the blink. It is now refused by the
    # band-limited guard (B7, carried B6 item); the old behaviour is reachable
    # only through the explicit opt-out, which is what the contrast measures.
    raw_prefix, t_prefix, blink_prefix = _shared_component_raw()
    raw_prefix.filter(0.1, 45.0, picks="eeg", verbose=False)
    with pytest.raises(ValueError, match="not band-limited"):
        EOGRegressionStep().process(raw_prefix, {})
    line_before_prefix = _band_power(raw_prefix.get_data(picks=eeg_picks),
                                     band="line")
    blink_before_prefix = _blink_locked_ptp(
        raw_prefix.get_data(picks=eeg_picks), t_prefix, blink_prefix)
    EOGRegressionStep(allow_unfiltered_regressors=True).process(raw_prefix, {})
    line_ratio_prefix = (
        _band_power(raw_prefix.get_data(picks=eeg_picks), band="line")
        / line_before_prefix
    )
    blink_ratio_prefix = (
        _blink_locked_ptp(raw_prefix.get_data(picks=eeg_picks), t_prefix,
                          blink_prefix)
        / blink_before_prefix
    )
    assert line_ratio_prefix.mean() > 100.0, line_ratio_prefix
    assert blink_ratio_prefix.mean() > 0.9, blink_ratio_prefix


# ---------------------------------------------------------------------------
# reference (P1-F16, plan task 11; post-ref check: plan task 7)
# ---------------------------------------------------------------------------

def test_reference_rest_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="rest"):
        ReferenceStep(type="rest")


def test_reference_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown reference type"):
        ReferenceStep(type="magic")


def test_reference_get_config_has_no_dead_exclude_bads():
    config = ReferenceStep(type="average").get_config()
    assert "exclude_bads" not in config
    assert config["check_reference_channel_variance"] is None


def test_reference_post_ref_variance_check_passes():
    raw = _montaged_raw(seed=3)
    out, meta = ReferenceStep(
        type="average", check_reference_channel_variance="Cz").process(raw, {})
    assert meta["applied"] is True
    assert meta["reference_channel_post_ref_variance"] > 0
    assert np.isfinite(meta["reference_channel_post_ref_variance"])


def test_reference_post_ref_variance_zero_raises():
    """A zero post-reference column means the reference was not applied."""
    raw = _montaged_raw(seed=4)
    # Cz identical to the reference channel -> post-ref Cz is exactly zero.
    raw._data[raw.ch_names.index("Cz")] = raw._data[raw.ch_names.index("Pz")]
    step = ReferenceStep(type="channels", channels=["Pz"], projection=False,
                         check_reference_channel_variance="Cz")
    with pytest.raises(RuntimeError, match="post-reference variance"):
        step.process(raw, {})


def test_reference_post_ref_missing_channel_raises():
    raw = _montaged_raw(seed=5)
    step = ReferenceStep(type="average",
                         check_reference_channel_variance="NOPE")
    with pytest.raises(RuntimeError, match="not present"):
        step.process(raw, {})


def test_reference_without_check_reports_none():
    raw = _montaged_raw(seed=6)
    _, meta = ReferenceStep(type="average").process(raw, {})
    assert meta["reference_channel_post_ref_variance"] is None


# ---------------------------------------------------------------------------
# ICA rank-1 (P1-F20, plan task 13)
# ---------------------------------------------------------------------------

def test_ica_rank_minus_one_never_exceeds_rank(monkeypatch):
    import eegcpm.modules.preprocessing.steps.ica as ica_module

    raw = _raw([f"E{i}" for i in range(1, 21)])
    step = ICAStep(n_components="rank-1")

    for rank, expected in ((20, 19), (6, 5), (5, 4), (2, 1)):
        monkeypatch.setattr(
            ica_module.mne, "compute_rank", lambda *a, _r=rank, **k: {"eeg": _r}
        )
        assert step._resolve_n_components(raw) == expected
        assert step._resolve_n_components(raw) <= rank - 1

    for rank in (1, 0):
        monkeypatch.setattr(
            ica_module.mne, "compute_rank", lambda *a, _r=rank, **k: {"eeg": _r}
        )
        with pytest.raises(ValueError, match="rank >= 2"):
            step._resolve_n_components(raw)


def test_ica_rank_minus_one_on_real_rank():
    raw = _raw([f"E{i}" for i in range(1, 21)], n_times=2000)
    step = ICAStep(n_components="rank-1")
    n = step._resolve_n_components(raw)
    rank = mne.compute_rank(raw, rank="info")["eeg"]
    assert n == rank - 1


# ---------------------------------------------------------------------------
# montage guard (P1-F22, plan task 15)
# ---------------------------------------------------------------------------

def test_montage_channel_without_position_raises():
    raw = _raw(["Fp1", "Fp2", "Cz", "NOTINMONTAGE"])
    step = MontageStep(type="standard_1020", on_missing="warn")
    with pytest.raises(ValueError, match="NOTINMONTAGE"):
        step.process(raw, {})


def test_montage_zero_position_on_existing_montage_raises():
    raw = _montaged_raw(seed=7)
    idx = raw.ch_names.index("Pz")
    raw.info["chs"][idx]["loc"][:3] = 0.0
    step = MontageStep(type="standard_1020")
    with pytest.raises(ValueError, match="Pz"):
        step.process(raw, {})


def test_montage_valid_recording_applies_and_keeps_channels():
    raw = _raw(["Fp1", "Fp2", "F3", "F4", "Cz", "Pz", "O1", "O2"])
    step = MontageStep(type="standard_1020")
    out, meta = step.process(raw, {})
    assert meta["applied"] is True
    assert meta["n_dropped"] == 0
    assert meta["channels_dropped"] == []
    assert len(out.ch_names) == 8


def test_montage_already_set_is_a_noop_without_drops():
    raw = _montaged_raw(seed=8)
    out, meta = MontageStep(type="standard_1020").process(raw, {})
    assert meta["skipped"] is True
    assert meta["reason"] == "montage_already_set"
    assert meta["n_dropped"] == 0
    assert len(out.ch_names) == len(raw.ch_names)


# ---------------------------------------------------------------------------
# BadChannelDetector: correlation over montage neighbours, loud failures
# (P1-F18 + B3-K9, plan task 12)
# ---------------------------------------------------------------------------

def _correlated_raw(seed=0, n_times=5000, bad_channel=None, flat_channel=None):
    """Spatially correlated 20-channel 1020 recording."""
    raw = _montaged_raw(seed=seed, n_times=n_times)
    rng = np.random.RandomState(seed + 1)
    # A common source + independent noise: neighbours are correlated.
    common = rng.randn(n_times) * 10e-6
    raw._data[:] = 0.8 * common[None, :] + rng.randn(*raw._data.shape) * 2e-6
    if bad_channel is not None:
        raw._data[raw.ch_names.index(bad_channel)] = (
            rng.randn(n_times) * 10e-6)
    if flat_channel is not None:
        raw._data[raw.ch_names.index(flat_channel)] = 0.0
    return raw


def test_correlation_flat_channel_is_bad_and_scores_stay_finite():
    from eegcpm.modules.preprocessing.bad_channels import BadChannelDetector

    raw = _correlated_raw(seed=0, flat_channel="Pz")
    detector = BadChannelDetector(methods=["correlation"])
    bads, scores = detector.detect(raw, return_scores=True)

    assert "Pz" in bads, f"flat channel not flagged: {bads}"
    assert "Pz" in scores["correlation"]
    others = [v for ch, v in scores["correlation"].items() if ch != "Pz"]
    assert others and all(np.isfinite(v) for v in others), (
        "a flat channel NaN-ed the correlation criterion for other channels")


def test_correlation_flags_channel_uncorrelated_with_neighbours():
    from eegcpm.modules.preprocessing.bad_channels import BadChannelDetector

    raw = _correlated_raw(seed=1, bad_channel="Pz")
    detector = BadChannelDetector(methods=["correlation"],
                                  correlation_threshold=0.4)
    bads, scores = detector.detect(raw, return_scores=True)

    assert "Pz" in bads
    assert scores["correlation"]["Pz"] < 0.4
    # Its neighbours stay healthy.
    assert "P3" not in bads and "P4" not in bads


def test_correlation_neighbours_are_spatially_local():
    from eegcpm.modules.preprocessing.bad_channels import BadChannelDetector

    raw = _montaged_raw(seed=2)
    detector = BadChannelDetector(methods=["correlation"])
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    neighbours = detector._neighbour_indices(raw, picks)

    positions = np.array([raw.info["chs"][p]["loc"][:3] for p in picks])
    for i, near in enumerate(neighbours):
        assert len(near) >= detector.MIN_NEIGHBOURS
        assert i not in near
        distances = np.linalg.norm(positions[i] - positions[near], axis=1)
        # Either inside the radius or (fallback) among the closest channels.
        assert distances.max() <= max(
            detector.NEIGHBOUR_RADIUS_M,
            np.sort(np.linalg.norm(positions[i] - positions, axis=1))[
                detector.MIN_NEIGHBOURS],
        ) + 1e-9


def test_ransac_failure_raises_instead_of_zero_bads(monkeypatch):
    """B3-K9: a failing RANSAC path must fail the step."""
    from autoreject import Ransac

    from eegcpm.modules.preprocessing.bad_channels import BadChannelDetector
    from eegcpm.modules.preprocessing.steps.bad_channels import (
        BadChannelDetectionStep,
    )

    def boom(self, epochs, **kwargs):
        raise RuntimeError("synthetic RANSAC failure")

    monkeypatch.setattr(Ransac, "fit_transform", boom)

    raw = _correlated_raw(seed=3)
    with pytest.raises(RuntimeError, match="RANSAC bad-channel detection failed"):
        BadChannelDetector(methods=["ransac"]).detect(raw)

    step = BadChannelDetectionStep(method="ransac", mark_only=True)
    with pytest.raises(RuntimeError, match="RANSAC bad-channel detection failed"):
        step.process(raw, {})


def test_prep_failure_raises_instead_of_zero_bads(monkeypatch):
    """B3-K9: a failing PREP path must fail the step (fake pyprep backend)."""
    from eegcpm.modules.preprocessing.bad_channels import BadChannelDetector

    class _FailingNoisyChannels:
        def __init__(self, raw, do_detrend=False):
            self.bad_by_deviation = []

        def find_all_bads(self, **kwargs):
            raise RuntimeError("synthetic PREP failure")

        def get_bads(self):
            return []

    fake_find = types.ModuleType("pyprep.find_noisy_channels")
    fake_find.NoisyChannels = _FailingNoisyChannels
    fake_pyprep = types.ModuleType("pyprep")
    fake_pyprep.find_noisy_channels = fake_find
    monkeypatch.setitem(sys.modules, "pyprep", fake_pyprep)
    monkeypatch.setitem(sys.modules, "pyprep.find_noisy_channels", fake_find)

    raw = _correlated_raw(seed=4)
    with pytest.raises(RuntimeError, match="PREP bad-channel detection failed"):
        BadChannelDetector(methods=["prep"]).detect(raw)


# ---------------------------------------------------------------------------
# Legacy detect_and_interpolate_bad_channels: no silent channel drops
# (B6-K9 → plan §B7 task 15)
# ---------------------------------------------------------------------------

def test_legacy_interpolation_without_montage_raises_instead_of_dropping(monkeypatch):
    """A montage-less recording fails loudly; no channel is dropped silently.

    The legacy convenience function used to ``print`` the failure and return
    the raw with the bad channels *dropped* — a silent change of the sensor set
    (B6-K9/P1-F11). It now raises. ``set_montage(on_missing='warn')`` does not
    raise for unknown channel names (probed), so the failure is injected at the
    montage constructor, which is the only way that branch can fire.
    """
    import mne

    from eegcpm.modules.preprocessing.bad_channels import (
        detect_and_interpolate_bad_channels,
    )

    def _no_montage(*args, **kwargs):
        raise ValueError("standard_1020 montage unavailable")

    monkeypatch.setattr(mne.channels, "make_standard_montage", _no_montage)

    raw = _raw(["Fp1", "Fp2", "F3", "F4", "Cz", "Pz", "O1", "O2"])  # no montage
    with pytest.raises(RuntimeError, match="no montage"):
        detect_and_interpolate_bad_channels(
            raw,
            {"manual_bads": ["Pz"], "auto_detect": False, "interpolate": True},
            copy=True,
        )


def test_legacy_interpolation_failure_raises_instead_of_dropping(monkeypatch):
    """An interpolation error propagates; the channels stay in the recording."""
    from eegcpm.modules.preprocessing import bad_channels as legacy

    raw = _montaged_raw(seed=11)

    def _boom(*args, **kwargs):
        raise RuntimeError("interpolate_bads exploded")

    monkeypatch.setattr(type(raw), "interpolate_bads", _boom)
    with pytest.raises(RuntimeError, match="refusing to drop channels"):
        legacy.detect_and_interpolate_bad_channels(
            raw,
            {"manual_bads": ["Pz"], "auto_detect": False, "interpolate": True},
            copy=True,
        )


def test_legacy_zero_variance_drop_is_logged(caplog):
    """Zero-variance channels are dropped, but the drop is recorded (log)."""
    import logging

    from eegcpm.modules.preprocessing import bad_channels as legacy

    raw = _montaged_raw(seed=12)
    raw._data[raw.ch_names.index("Pz")] = 0.0
    with caplog.at_level(logging.WARNING, logger="eegcpm.modules.preprocessing.bad_channels"):
        out, bads = legacy.detect_and_interpolate_bad_channels(
            raw,
            {"manual_bads": ["Pz"], "auto_detect": False, "interpolate": True},
            copy=True,
        )
    assert "Pz" in bads
    assert "Pz" not in out.ch_names
    assert any("zero-variance" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# B7 (carried B6 item): EOGRegressionStep refuses unfiltered regressors
# ---------------------------------------------------------------------------

def test_eog_regression_refuses_unfiltered_regressors():
    """Unfiltered EOG regressors re-inject line noise/drift → loud refusal."""
    raw = _eog_raw(seed=5, filtered=False)
    # Give the unfiltered EOG a line component (the real failure mode).
    sfreq = raw.info["sfreq"]
    t = np.arange(raw.n_times) / sfreq
    eog_picks = mne.pick_types(raw.info, eog=True, exclude="bads")
    raw._data[eog_picks] += 20e-6 * np.sin(2 * np.pi * 60.0 * t)[None, :]

    # The metadata contract fires first for an unfiltered RawArray (its
    # declared lowpass is Nyquist); the spectral check is exercised below.
    with pytest.raises(ValueError, match="declares lowpass"):
        EOGRegressionStep(eog_channels=["VEOG1", "VEOG2", "VEOG3"]).process(raw, {})

    # EEG-only filtering (the pre-fix chain behaviour, B6 rev2 NEW-1): the
    # metadata declares 45 Hz while the EOG data stay unfiltered — the
    # spectral check catches it.
    raw_eeg_only = _eog_raw(seed=5, filtered=False)
    raw_eeg_only.filter(0.5, 45.0, picks="eeg", verbose=False)
    assert raw_eeg_only.info["lowpass"] == pytest.approx(45.0)
    with pytest.raises(ValueError, match="not band-limited"):
        EOGRegressionStep(eog_channels=["VEOG1", "VEOG2", "VEOG3"]).process(
            raw_eeg_only, {})


def test_eog_regression_accepts_filtered_regressors_and_records_the_fraction():
    raw = _eog_raw(seed=5)
    _, meta = EOGRegressionStep(eog_channels=["VEOG1", "VEOG2", "VEOG3"]).process(raw, {})
    assert meta["applied"] is True
    assert meta["eog_band_limited_check"] == "passed"
    assert meta["eog_hf_power_fraction"] < MAX_HF_POWER_FRACTION
    assert meta["eog_hf_cutoff_hz"] == pytest.approx(50.0)
    assert meta["eog_lowpass_hz"] == pytest.approx(45.0)


def test_eog_regression_bypass_is_explicit_and_recorded():
    raw = _eog_raw(seed=6, filtered=False)
    _, meta = EOGRegressionStep(
        eog_channels=["VEOG1", "VEOG2", "VEOG3"],
        allow_unfiltered_regressors=True,
    ).process(raw, {})
    assert meta["applied"] is True
    assert meta["eog_band_limited_check"] == "bypassed"
    assert meta["eog_hf_power_fraction"] > MAX_HF_POWER_FRACTION


def test_eog_guard_cutoff_follows_nyquist():
    """A low-rate recording cannot be required to be clean above Nyquist."""
    raw = _eog_raw(seed=7, sfreq=100.0, filtered=False)
    _, meta = EOGRegressionStep(
        eog_channels=["VEOG1", "VEOG2", "VEOG3"],
        allow_unfiltered_regressors=True,
    ).process(raw, {})
    assert meta["eog_hf_cutoff_hz"] == pytest.approx(0.45 * 50.0)
