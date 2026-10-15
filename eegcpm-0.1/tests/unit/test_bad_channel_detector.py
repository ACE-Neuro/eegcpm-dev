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
    THRESHOLD_SD,
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
        # B4-D2 additions
        "near_flat", "blink_mask",
    ):
        assert key in report
    assert report["tails"] == TAILS
    assert report["tails"] == {
        "variance": "upper", "hf_power": "upper", "kurtosis": "upper",
    }
    # Blink-mask reporting (B4-D2): every key is present, whether or not a
    # mask was applied, so the fallback is never silent.
    for key in (
        "applied", "source", "channel", "n_blinks", "masked_fraction",
        "half_width_s", "reason",
    ):
        assert key in report["blink_mask"]


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


def test_blink_mask_uses_data_frame_with_first_samp_offset():
    """B4-D2 frame check: `find_eog_events` returns **raw-frame** samples.

    Probed 2026-10-16 on mne 1.13.2: the returned sample column includes
    ``raw.first_samp``. The mask indexes the data array, so a missing
    subtraction would shift the blink-free samples by ``first_samp`` — the
    kurtosis criterion would then keep the blink peaks and remove clean data.
    """
    from eegcpm.modules.preprocessing.steps.bad_channels_pinned import (
        BLINK_MASK_HALF_WIDTH_S,
        _find_blink_mask,
    )

    sfreq = 250.0
    duration = 30.0
    n_times = int(duration * sfreq)
    first_samp = 1000
    rng = np.random.RandomState(11)
    ch_names = [f"E{i}" for i in range(1, 129)] + ["Cz"]
    raw = mne.io.RawArray(
        rng.randn(len(ch_names), n_times) * 1e-6,
        mne.create_info(ch_names, sfreq, "eeg"),
        first_samp=first_samp,
    )
    raw.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-129"),
                    match_case=False, on_missing="ignore")
    raw.set_channel_types({"E8": "eog"})

    blink_data_samples = [int(t * sfreq) for t in (5.0, 15.0, 25.0)]
    e8 = raw.ch_names.index("E8")
    for sample in blink_data_samples:
        raw._data[e8, sample:sample + 50] += 300e-6

    picks = [p for p in mne.pick_types(raw.info, eeg=True, exclude="bads")
             if raw.ch_names[p] != "Cz"]
    mask, report = _find_blink_mask(raw, picks)

    assert report["source"] == "eog"
    assert report["channel"] == "E8"
    assert report["applied"] is True
    assert report["n_blinks"] >= 3

    half_width = int(BLINK_MASK_HALF_WIDTH_S * sfreq)
    for sample in blink_data_samples:
        window = mask[max(0, sample - half_width):sample + half_width]
        assert window.mean() > 0.5, (
            f"injected blink at data sample {sample} is not masked "
            f"(coverage {window.mean():.2f})")
        # ... and the first_samp-shifted position is NOT the masked one.
        shifted = sample + first_samp
        if shifted + half_width <= n_times:
            shifted_window = mask[
                shifted - half_width:shifted + half_width]
            assert shifted_window.mean() < 0.5, (
                "mask appears shifted by first_samp")


def test_blink_mask_dead_eog_retries_frontal_proxy():
    """B6-K5: a dead EOG electrode must not disable blink masking.

    ``find_eog_events`` raises ``TypeError`` on a flat channel (probed on
    mne 1.13.2). The old code fell straight back to the full signal; the
    detector must retry the synthetic frontal proxy, mask the blinks there
    and record both attempts.
    """
    from eegcpm.modules.preprocessing.steps.bad_channels_pinned import (
        BLINK_MASK_HALF_WIDTH_S,
        _find_blink_mask,
    )

    sfreq = 250.0
    duration = 30.0
    n_times = int(duration * sfreq)
    rng = np.random.RandomState(13)
    ch_names = [f"E{i}" for i in range(1, 129)] + ["Cz"]
    raw = mne.io.RawArray(
        rng.randn(len(ch_names), n_times) * 1e-6,
        mne.create_info(ch_names, sfreq, "eeg"),
    )
    raw.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-129"),
                    match_case=False, on_missing="ignore")
    raw.set_channel_types({"E8": "eog"})
    raw._data[raw.ch_names.index("E8")] = 0.0  # dead EOG electrode

    picks = [p for p in mne.pick_types(raw.info, eeg=True, exclude="bads")
             if raw.ch_names[p] != "Cz"]
    positions = np.array([raw.info["chs"][p]["loc"][:3] for p in picks])
    proxy = raw.ch_names[picks[int(np.argmax(positions[:, 1]))]]

    blink_samples = [int(t * sfreq) for t in (5.0, 15.0, 25.0)]
    proxy_idx = raw.ch_names.index(proxy)
    for sample in blink_samples:
        raw._data[proxy_idx, sample:sample + 50] += 300e-6

    mask, report = _find_blink_mask(raw, picks)

    assert report["source"] == "frontal_proxy"
    assert report["channel"] == proxy
    assert report["applied"] is True
    assert report["n_blinks"] >= 3
    # Both paths are recorded: the EOG attempt failed, the proxy succeeded.
    assert any(a["source"] == "eog" and a["error"] for a in report["attempts"])
    assert any(a["source"] == "frontal_proxy" and a["error"] is None
               for a in report["attempts"])

    half_width = int(BLINK_MASK_HALF_WIDTH_S * sfreq)
    for sample in blink_samples:
        window = mask[sample - half_width:sample + half_width]
        assert window.mean() > 0.5, (
            f"injected blink at data sample {sample} is not masked "
            f"(coverage {window.mean():.2f})")

    # The detector report records the same proxy path: the mask is actually
    # used for the kurtosis criterion, not silently dropped.
    _, det_report = detect_bad_channels(raw, exempt=("Cz",))
    assert det_report["blink_mask"]["source"] == "frontal_proxy"
    assert det_report["blink_mask"]["applied"] is True


def test_blink_channel_selected_on_bandpassed_variance():
    """NEW-2: the blink channel is ranked on 1-10 Hz band-passed variance.

    Raw-variance ranking picks a drift-dominated EOG channel (the highest raw
    variance is DC drift), and ``find_eog_events`` then finds almost no blinks
    on real pilot recordings. The band-passed ranking must pick the
    blink-carrying channel instead, and the mask must cover the injected
    blinks.
    """
    from eegcpm.modules.preprocessing.steps.bad_channels_pinned import (
        BLINK_BANDPASS_HZ,
        BLINK_MASK_HALF_WIDTH_S,
        _find_blink_mask,
    )

    sfreq = 250.0
    duration = 60.0
    n_times = int(duration * sfreq)
    t = np.arange(n_times) / sfreq
    rng = np.random.RandomState(23)
    ch_names = [f"E{i}" for i in range(1, 129)] + ["Cz"]
    data = rng.randn(len(ch_names), n_times) * 5e-6
    blink = np.exp(-0.5 * (((t % 3.0) - 1.0) / 0.05) ** 2)
    # E8: drifting, blink-free EOG (raw variance dominated by the drift).
    data[ch_names.index("E8")] += 1e-3 * np.sin(2 * np.pi * 0.05 * t)
    # E9: blink-carrying EOG.
    data[ch_names.index("E9")] += 300e-6 * blink
    raw = mne.io.RawArray(
        data, mne.create_info(ch_names, sfreq, "eeg")
    )
    raw.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-129"),
                    match_case=False, on_missing="ignore")
    raw.set_channel_types({"E8": "eog", "E9": "eog"})

    eog_picks = mne.pick_types(raw.info, eog=True, exclude="bads")
    raw_var = np.var(raw.get_data(picks=eog_picks), axis=1)
    band_var = np.var(
        mne.filter.filter_data(raw.get_data(picks=eog_picks), sfreq,
                               *BLINK_BANDPASS_HZ, verbose=False),
        axis=1,
    )
    # The pre-fix trap: raw variance picks the drift channel (E8), the
    # band-passed variance picks the blink channel (E9).
    assert raw_var.argmax() == 0, raw_var
    assert band_var.argmax() == 1, band_var

    picks = [p for p in mne.pick_types(raw.info, eeg=True, exclude="bads")
             if raw.ch_names[p] != "Cz"]
    mask, report = _find_blink_mask(raw, picks)

    assert report["channel"] == "E9", report["selection"]
    assert report["source"] == "eog"
    assert report["applied"] is True
    assert report["selection"]["band_hz"] == list(BLINK_BANDPASS_HZ)
    assert report["selection"]["ranking"][0]["channel"] == "E9"
    assert report["n_blinks"] >= 15, report  # 20 blinks injected

    # The mask covers the injected blinks.
    half_width = int(BLINK_MASK_HALF_WIDTH_S * sfreq)
    blink_samples = [int((1.0 + 3.0 * k) * sfreq) for k in range(20)]
    for sample in blink_samples:
        assert mask[sample - half_width:sample + half_width].mean() > 0.5, sample

    # Non-vacuity: the drift channel yields a flood of spurious events (their
    # +/-200 ms windows cover the whole recording -> the documented
    # full-signal fallback), not the injected blinks.
    events_drift = mne.preprocessing.find_eog_events(
        raw, ch_name="E8", verbose=False)
    assert events_drift.shape[0] > 50, events_drift.shape


def test_blink_mask_records_implausible_blink_rate(monkeypatch):
    """NEW-2: a blink rate below the floor is recorded, never silent.

    The mask is still applied (the note is provenance, not a fallback); the
    floor is monkeypatched so the check is deterministic.
    """
    from eegcpm.modules.preprocessing.steps import bad_channels_pinned as bcp
    from eegcpm.modules.preprocessing.steps.bad_channels_pinned import (
        _find_blink_mask,
    )

    monkeypatch.setattr(bcp, "MIN_BLINK_RATE_PER_MINUTE", 100.0)

    sfreq = 250.0
    duration = 30.0
    n_times = int(duration * sfreq)
    t = np.arange(n_times) / sfreq
    rng = np.random.RandomState(24)
    ch_names = [f"E{i}" for i in range(1, 129)] + ["Cz"]
    data = rng.randn(len(ch_names), n_times) * 5e-6
    blink = np.exp(-0.5 * (((t % 10.0) - 5.0) / 0.05) ** 2)
    data[ch_names.index("E8")] += 300e-6 * blink
    raw = mne.io.RawArray(data, mne.create_info(ch_names, sfreq, "eeg"))
    raw.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-129"),
                    match_case=False, on_missing="ignore")
    raw.set_channel_types({"E8": "eog"})

    picks = [p for p in mne.pick_types(raw.info, eeg=True, exclude="bads")
             if raw.ch_names[p] != "Cz"]
    mask, report = _find_blink_mask(raw, picks)

    assert report["n_blinks"] >= 1
    assert report["applied"] is True
    assert report["blinks_per_minute"] is not None
    assert report["plausibility_reason"] is not None
    assert report["plausibility_reason"].startswith("low_blink_rate:")
    assert report["reason"] is None  # the mask was applied


def test_nanovolt_channel_is_near_flat():
    """NEW-9(a): a 1 nV electrode is flagged ``flat`` and listed in near_flat.

    1 nV of noise has variance 1e-18 V**2 — far above the absolute
    ``FLAT_VARIANCE_FLOOR`` (1e-20) but ~1e-6 of a typical EEG channel, which
    is exactly the blind spot the relative floor closes.
    """
    from eegcpm.modules.preprocessing.steps.bad_channels_pinned import (
        FLAT_VARIANCE_FLOOR,
    )

    raw = _egi_like_raw(seed=10)
    idx = raw.ch_names.index("E30")
    rng = np.random.RandomState(11)
    raw._data[idx] = rng.randn(raw.n_times) * 1e-9
    assert np.var(raw._data[idx]) > FLAT_VARIANCE_FLOOR

    bads, report = detect_bad_channels(raw, exempt=("Cz",))

    assert "E30" in bads
    assert "flat" in report["reasons"]["E30"]
    assert "E30" in report["near_flat"]
    assert "E30" not in report["flat"]  # above the absolute floor
    # near-flat channels stay in the statistics (the ratio is robust).
    assert report["z_scores"]["E30"]["var"] is not None


def _blink_dominated_raw(seed=21, n_channels=129, duration=60.0, sfreq=250.0,
                         frontal=("E22", "E23"), blink_amp=32e-6,
                         eog_amp=300e-6):
    """Frontal EEG channels dominated by narrow blinks, plus an EOG channel.

    The blink is 10 ms wide and 3.2x the 10 uV background std: its kurtosis
    contribution is large while its variance contribution stays below the
    upper-tail variance threshold, so the fixture isolates the *kurtosis*
    criterion the blink mask modifies.
    """
    ch_names = [f"E{i}" for i in range(1, n_channels)] + ["Cz"]
    rng = np.random.RandomState(seed)
    n_times = int(duration * sfreq)
    t = np.arange(n_times) / sfreq
    data = rng.randn(len(ch_names), n_times) * 10e-6
    blink = np.exp(-0.5 * (((t % 5.0) - 2.5) / 0.01) ** 2)
    data[ch_names.index("E8")] += eog_amp * blink
    for ch in frontal:
        data[ch_names.index(ch)] += blink_amp * blink
    raw = mne.io.RawArray(data, mne.create_info(ch_names, sfreq, "eeg"))
    raw.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-129"),
                    match_case=False, on_missing="ignore")
    raw.set_channel_types({"E8": "eog"})
    return raw, list(frontal)


def test_blink_dominated_frontal_channels_masked_vs_full_signal(monkeypatch):
    """NEW-9(b): the blink mask removes the blink kurtosis signature.

    With the mask (blink-free samples) the frontal blink channels are not
    flagged; on the full signal they are flagged by kurtosis. The no-mask run
    monkeypatches ``_find_blink_mask`` to the documented
    ``applied: False`` fallback report (the full-signal path), so the two
    regimes are compared without relying on a detection failure.
    """
    from eegcpm.modules.preprocessing.steps import bad_channels_pinned as bcp

    raw, frontal = _blink_dominated_raw()

    bads, report = detect_bad_channels(raw, exempt=("Cz",))
    assert report["blink_mask"]["applied"] is True
    for ch in frontal:
        assert ch not in bads, (ch, report["reasons"].get(ch))
        assert "kurtosis" not in report["reasons"].get(ch, [])

    def _no_mask(raw_, eligible_picks):
        return np.zeros(raw_.n_times, dtype=bool), {
            "applied": False,
            "source": None,
            "channel": None,
            "n_blinks": 0,
            "blinks_per_minute": None,
            "masked_fraction": 0.0,
            "half_width_s": bcp.BLINK_MASK_HALF_WIDTH_S,
            "plausibility_reason": None,
            "reason": "test_forced_full_signal",
        }

    monkeypatch.setattr(bcp, "_find_blink_mask", _no_mask)
    bads_full, report_full = detect_bad_channels(raw, exempt=("Cz",))
    for ch in frontal:
        assert ch in bads_full, (ch, report_full["reasons"])
        assert "kurtosis" in report_full["reasons"][ch]
        assert report_full["z_scores"][ch]["kurtosis"] > bcp.THRESHOLD_SD


def test_sub_gaussian_sinusoid_not_flagged_by_kurtosis_upper_tail():
    """NEW-9(c): a large sinusoid (Pearson kurtosis ~1.5) is not kurtosis-bad.

    Its kurtosis z-score is strongly negative (sub-Gaussian); the upper-tail
    rule must not flag it. The channel may legitimately be flagged by the
    variance criterion (a 50 uV sinusoid is 50x the background std), which is
    a different, in-band measure — the assertion is about the kurtosis tail.
    """
    raw = _egi_like_raw(seed=12)
    idx = raw.ch_names.index("E40")
    t = np.arange(raw.n_times) / raw.info["sfreq"]
    raw._data[idx] = 50e-6 * np.sin(2 * np.pi * 7.0 * t)

    bads, report = detect_bad_channels(raw, exempt=("Cz",))

    z_kurt = report["z_scores"]["E40"]["kurtosis"]
    assert z_kurt < -THRESHOLD_SD, z_kurt  # strongly sub-Gaussian
    assert "kurtosis" not in report["reasons"].get("E40", [])
    # The old two-sided rule would have flagged |z| > 3; the upper tail does
    # not. (The variance criterion is expected to flag the channel.)
    assert "variance" in report["reasons"].get("E40", [])


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
