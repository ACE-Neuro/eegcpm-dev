"""P1-F5: bad-channel detector — robust stats, explicit exemption, loud budget.

Covers the required cases:

- flat Cz + heavy-tailed channel: the heavy-tailed channel is detected, Cz is
  exempt, every non-exempt kurtosis z-score is finite (a single flat channel no
  longer NaNs the criterion for all channels);
- a flat non-exempt channel is bad outright (reason ``flat``);
- ``MAX_BAD_CHANNELS + 5`` detections raise :class:`RecordingExcluded` (no silent
  truncation);
- the chain wrapper takes the reference channel from
  ``metadata['channel_roles']['reference_channel']`` (no ``info`` flag);
- interpolate over budget raises and never drops channels;
- B4-D1 recalibration: a healthy Cz-referenced, spatially correlated
  GSN-HydroCel-129-like recording stays within the ``max_bad`` budget (the
  pre-recalibration raw-scale two-sided rule flags 23 channels on this fixture),
  the injected heavy-tailed channel is still detected, the lower tail is never
  flagged, and the relative-epsilon degeneracy rule (B4-K3) zeroes ulp-level
  spread instead of flagging it.
"""

import mne
import numpy as np
import pytest

from eegcpm.modules.preprocessing.hbn_langer_chain import (
    _BadChannelDetectorWrapper,
)
from eegcpm.modules.preprocessing.steps.bad_channels_pinned import (
    HF_BAND_HZ,
    MAX_BAD_CHANNELS,
    TAILS,
    _robust_z,
    apply_bad_channels,
    detect_bad_channels,
)
from eegcpm.modules.preprocessing.steps.base import RecordingExcluded
from eegcpm.modules.preprocessing.steps.channel_roles import (
    EGI_129_REFERENCE_CHANNEL,
    ChannelRolesStep,
)
from eegcpm.modules.preprocessing.steps.interpolate import InterpolateBadChannelsStep


def _egi_like_raw(n_channels: int = 110, duration_s: float = 10.0,
                  sfreq: float = 500.0, seed: int = 0):
    """Synthetic EGI-like raw: E1..E{n-1} + Cz with a GSN montage."""
    rng = np.random.RandomState(seed)
    ch_names = [f"E{i}" for i in range(1, n_channels)] + ["Cz"]
    info = mne.create_info(ch_names, sfreq, "eeg")
    data = rng.randn(n_channels, int(duration_s * sfreq)) * 1e-6
    raw = mne.io.RawArray(data, info)
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    raw.set_montage(montage, match_case=False, on_missing="ignore")
    return raw


def _healthy_cz_referenced_raw(
    duration_s: float = 20.0,
    sfreq: float = 250.0,
    seed: int = 42,
    corr_length_m: float = 0.03,
    sensor_noise_frac: float = 0.3,
    spike_channel: str = "E70",
    n_spikes: int = 60,
    spike_amp: float = 20.0,
):
    """GSN-HydroCel-129-like, spatially correlated, Cz-referenced healthy raw.

    Background = a Gaussian spatial process (correlation length 3 cm, i.e.
    ``<= 5 cm``) plus an independent per-electrode sensor-noise floor; one
    channel (``E70``) carries injected heavy-tailed spikes. Subtracting the Cz
    column models the physical Cz reference (Cz is exactly zero -> flat, and the
    detector receives it as ``exempt``), which gives the near-Cz electrodes the
    wide *low*-variance spread that the pre-B4-D1 raw-scale two-sided rule
    over-flagged (23 flags on this fixture, > ``MAX_BAD_CHANNELS``).
    """
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    ch_names = list(montage.ch_names)
    coords = np.array(
        [montage.get_positions()["ch_pos"][ch] for ch in ch_names]
    )
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    cov = np.exp(-d2 / (2 * corr_length_m ** 2))
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(len(ch_names)))

    rng = np.random.RandomState(seed)
    n_times = int(duration_s * sfreq)
    data = L @ rng.randn(len(ch_names), n_times)
    data += sensor_noise_frac * rng.randn(len(ch_names), n_times)
    data *= 1e-6  # volts
    if spike_channel is not None:
        idx = ch_names.index(spike_channel)
        spikes = rng.choice(n_times, n_spikes, replace=False)
        data[idx, spikes] += spike_amp * data[idx].std()
    data -= data[ch_names.index("Cz")][None, :]  # Cz-referenced

    info = mne.create_info(ch_names, sfreq, "eeg")
    raw = mne.io.RawArray(data, info)
    raw.set_montage(montage)
    return raw


def test_heavy_tailed_detected_cz_exempt_all_kurtosis_z_finite():
    """Flat Cz must not disable the kurtosis criterion for the other channels."""
    raw = _egi_like_raw(seed=1)
    n_times = raw.n_times
    rng = np.random.RandomState(7)

    cz_idx = raw.ch_names.index("Cz")
    raw._data[cz_idx] = 0.0  # identically flat reference

    ht_idx = raw.ch_names.index("E50")
    spikes = rng.choice(n_times, 50, replace=False)
    raw._data[ht_idx, spikes] += 20.0 * raw._data[ht_idx].std()

    bads, report = detect_bad_channels(raw, exempt=("Cz",))

    assert "E50" in bads
    assert "kurtosis" in report["reasons"]["E50"]
    assert "Cz" not in bads
    assert "Cz" not in report["z_scores"]  # exempt channels never enter the stats
    assert report["exempt"] == ["Cz"]

    kurt_z = [entry["kurtosis"] for entry in report["z_scores"].values()]
    assert kurt_z, "no z-scores reported"
    assert all(z is not None and np.isfinite(z) for z in kurt_z)
    assert report["hf_band_hz"] == list(HF_BAND_HZ)


def test_flat_non_exempt_channel_is_bad_with_reason():
    raw = _egi_like_raw(seed=2)
    flat_idx = raw.ch_names.index("E10")
    raw._data[flat_idx] = 0.0

    bads, report = detect_bad_channels(raw, exempt=("Cz",))

    assert "E10" in bads
    assert "flat" in report["reasons"]["E10"]
    assert report["flat"] == ["E10"]
    # flat channels are removed from the statistics
    assert report["z_scores"]["E10"]["var"] is None


def test_report_keys_present():
    raw = _egi_like_raw(seed=3)
    _, report = detect_bad_channels(raw, exempt=("Cz",))
    for key in (
        "detected", "reasons", "z_scores", "n_detected_precap", "flat",
        "exempt", "hf_band_hz", "tails", "degenerate_metrics",
    ):
        assert key in report
    assert report["tails"] == TAILS
    assert report["tails"] == {
        "variance": "upper", "hf_power": "upper", "kurtosis": "two-sided",
    }


def test_healthy_cz_referenced_recording_stays_within_budget():
    """B4-D1 regression: a healthy real-like recording must not be excluded.

    The spatial correlation + physical Cz reference give the near-Cz electrodes a
    wide *low*-variance spread; the pre-recalibration raw-scale two-sided rule
    flagged 23 channels here (``RecordingExcluded`` at ``max_bad=11``). The
    log-scale upper-tail rule must stay within budget while still detecting the
    injected heavy-tailed channel.
    """
    raw = _healthy_cz_referenced_raw()
    bads, report = detect_bad_channels(raw, exempt=("Cz",))

    assert len(bads) <= MAX_BAD_CHANNELS, (len(bads), report["reasons"])
    assert "E70" in bads
    assert "kurtosis" in report["reasons"]["E70"]
    assert "Cz" not in bads
    assert "Cz" not in report["z_scores"]  # exempt -> excluded from the stats
    assert report["exempt"] == ["Cz"]
    assert report["degenerate_metrics"] == []

    # B4-D1 upper-tail semantics: the one-sided criteria never flag the lower
    # tail (near-Cz low-variance electrodes are healthy, not bad).
    for ch in bads:
        if "variance" in report["reasons"][ch]:
            assert report["z_scores"][ch]["var"] > 0
        if "hf_power" in report["reasons"][ch]:
            assert report["z_scores"][ch]["hf"] > 0

    # ... and the lowest-variance electrodes carry negative variance z-scores
    # without being flagged by that criterion.
    var_z = {
        ch: entry["var"]
        for ch, entry in report["z_scores"].items()
        if entry["var"] is not None
    }
    lowest = sorted(var_z, key=var_z.get)[:10]
    assert min(var_z[ch] for ch in lowest) < 0
    assert all("variance" not in report["reasons"].get(ch, []) for ch in lowest)


def test_robust_z_relative_epsilon_degeneracy():
    """B4-K3: ulp-level spread is degenerate, not a 10-sigma detection.

    ``mad == 0`` with a tiny (but ulp-nonzero) std previously produced huge
    z-scores; the relative-epsilon rule (``scale <= 1e-12 * max(|median|, 1)``)
    reports the metric as degenerate with ``z = 0`` instead.
    """
    eligible = np.ones(100, dtype=bool)
    metric = np.full(100, -27.0)
    metric[0] += 1e-13  # ulp-level spread around log-variance scale
    z, degenerate = _robust_z(metric, eligible)
    assert degenerate is True
    assert np.all(z == 0.0)

    # A genuinely varying metric is not degenerate: MAD == 0 falls back to the
    # (well above epsilon) std, and the upper tail flags the outlier.
    spread = np.zeros(100)
    spread[-1] = 1.0
    z2, degenerate2 = _robust_z(spread, eligible, tail="upper")
    assert degenerate2 is False
    assert z2[-1] > 3.0
    assert np.all(z2[:-1] <= 0.0)  # signed z; only the upper tail is compared


def test_too_many_bad_channels_excludes():
    """MAX_BAD_CHANNELS + 5 high-amplitude channels -> RecordingExcluded."""
    raw = _egi_like_raw(seed=4)
    rng = np.random.RandomState(3)
    n_bad = MAX_BAD_CHANNELS + 5
    for i in range(n_bad):
        idx = i + 1  # skip Cz (last channel)
        raw._data[idx] = rng.randn(raw.n_times) * 50e-6

    with pytest.raises(RecordingExcluded) as excinfo:
        detect_bad_channels(raw, exempt=("Cz",))

    assert excinfo.value.reason == "too_many_bad_channels"
    metrics = excinfo.value.metrics
    assert metrics["n_detected"] >= MAX_BAD_CHANNELS + 1
    assert metrics["max_bad_channels"] == MAX_BAD_CHANNELS
    assert len(metrics["detected"]) == metrics["n_detected"]


def test_apply_never_adds_exempt_channels():
    raw = _egi_like_raw(seed=5)
    raw = apply_bad_channels(raw, ["Cz", "E7"], exempt=("Cz",))
    assert "Cz" not in raw.info["bads"]
    assert "E7" in raw.info["bads"]
    # pre-existing bads survive
    raw.info["bads"] = ["E9"]
    raw = apply_bad_channels(raw, ["E8"], exempt=("Cz",))
    assert set(raw.info["bads"]) == {"E9", "E8"}


# --- chain wrapper path -----------------------------------------------------


def test_chain_wrapper_exempts_reference_channel():
    """ChannelRolesStep -> wrapper: Cz flat, exempt via metadata (no info flag)."""
    raw = _egi_like_raw(seed=6)
    cz_idx = raw.ch_names.index(EGI_129_REFERENCE_CHANNEL)
    raw._data[cz_idx] = 0.0

    raw, roles_meta = ChannelRolesStep().process(raw, {})
    assert roles_meta["reference_channel"] == EGI_129_REFERENCE_CHANNEL
    assert not hasattr(raw, "_eegcpm_reference_channels")

    out, step_meta = _BadChannelDetectorWrapper().process(
        raw, {"channel_roles": roles_meta}
    )
    assert EGI_129_REFERENCE_CHANNEL not in out.info["bads"]
    assert EGI_129_REFERENCE_CHANNEL not in step_meta["detected"]
    assert step_meta["exempt"] == [EGI_129_REFERENCE_CHANNEL]
    assert step_meta["n_bads"] == len(step_meta["detected"])
    assert "hf_band_hz" in step_meta


def test_chain_wrapper_requires_channel_roles_metadata():
    """Missing channel_roles metadata -> KeyError (the run fails, correctly)."""
    raw = _egi_like_raw(seed=7)
    with pytest.raises(KeyError):
        _BadChannelDetectorWrapper().process(raw, {})


# --- interpolate budget -----------------------------------------------------


def test_interpolate_over_budget_excludes_and_keeps_channels():
    raw = _egi_like_raw(seed=8)
    n_channels = len(raw.ch_names)
    raw.info["bads"] = [raw.ch_names[i] for i in range(int(0.6 * n_channels))]

    step = InterpolateBadChannelsStep(max_bad_percent=50.0)
    with pytest.raises(RecordingExcluded) as excinfo:
        step.process(raw, {})

    assert excinfo.value.reason == "too_many_bad_channels_for_interpolation"
    assert excinfo.value.metrics["n_bad"] > 0
    assert excinfo.value.metrics["pct_bad"] > 50.0
    assert excinfo.value.metrics["max_bad_percent"] == 50.0
    # channels are never dropped
    assert len(raw.ch_names) == n_channels
    assert len(raw.info["bads"]) == int(0.6 * n_channels)


def test_interpolate_within_budget_interpolates():
    raw = _egi_like_raw(seed=9)
    raw.info["bads"] = ["E1", "E2"]
    out, meta = InterpolateBadChannelsStep(max_bad_percent=50.0).process(raw, {})
    assert meta["applied"] is True
    assert meta["n_interpolated"] == 2
    assert out.info["bads"] == []  # reset_bads default
