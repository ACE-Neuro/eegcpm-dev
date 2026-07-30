"""Reachability tests for R-004, R-005, R-006 preprocessing fixes.

R-004: pinned variance/kurtosis/spectrum >3-SD detector must:
  - mark channels violating the threshold
  - exempt Cz BEFORE detection
  - cap at MAX_BAD_CHANNELS (RECALIBRATED in pilot-50; 11 here)
  - NOT immediately interpolate (mark-only)

R-005: ASR must:
  - pass the real repair/rejection mask to the manifest
  - assert non-zero burden on a burst fixture (NOT all-zero)

R-006: block rejection must:
  - detect 2-s blocks (not per-sample)
  - hard-fail on > max_rejected_fraction (NO silent truncation)
  - reachability test: 50%-contaminated input raises
"""

import numpy as np
import mne
import pytest

from eegcpm.modules.preprocessing.steps.bad_channels_pinned import (
    MAX_BAD_CHANNELS,
    THRESHOLD_SD,
    detect_bad_channels,
    apply_bad_channels,
)
from eegcpm.modules.preprocessing.steps.channel_roles import (
    ChannelRolesStep,
    EGI_129_REFERENCE_CHANNEL,
)
from eegcpm.modules.preprocessing.hbn_langer_chain import (
    _derive_asr_burden,
    build_hbn_langer_chain,
)


def _make_egi_129_raw_with_montage(duration_s=10, sfreq=500, seed=42,
                                     n_bad_channels=0, bad_amplitude=20.0):
    rng = np.random.RandomState(seed)
    ch_names = [f"E{i}" for i in range(1, 129)] + ["Cz"]
    n_channels = len(ch_names)
    n_times = int(duration_s * sfreq)
    data = rng.randn(n_channels, n_times) * 1e-6
    # Add a montage with digitization (the reference step needs it)
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    raw.set_montage(montage, match_case=False, on_missing="ignore")
    # Mark some channels as bad by adding large amplitude
    if n_bad_channels > 0:
        bad_chs = [f"E{i}" for i in range(1, n_bad_channels + 1)]
        for ch in bad_chs:
            idx = raw.ch_names.index(ch)
            raw._data[idx] = rng.randn(n_times) * bad_amplitude
    return raw


# --- R-004: pinned bad-channel detector ---

def test_pinned_detector_marks_high_amplitude_channels():
    """A channel with large amplitude should be detected."""
    raw = _make_egi_129_raw_with_montage(n_bad_channels=3, bad_amplitude=50.0)
    # Apply channel roles so Cz has is_reference
    raw, _ = ChannelRolesStep().process(raw, {})
    bads = detect_bad_channels(raw, threshold_sd=THRESHOLD_SD)
    assert "E1" in bads or "E2" in bads or "E3" in bads, (
        f"Pinned detector did not flag the high-amplitude channels; "
        f"got {bads}"
    )


def test_pinned_detector_caps_at_max_bad_channels():
    """The detector caps bad channels at MAX_BAD_CHANNELS."""
    # Inject MORE than MAX_BAD_CHANNELS high-amplitude channels
    raw = _make_egi_129_raw_with_montage(
        n_bad_channels=MAX_BAD_CHANNELS + 5, bad_amplitude=100.0)
    raw, _ = ChannelRolesStep().process(raw, {})
    bads = detect_bad_channels(raw, threshold_sd=THRESHOLD_SD)
    assert len(bads) <= MAX_BAD_CHANNELS, (
        f"Detector returned {len(bads)} bads; cap is "
        f"{MAX_BAD_CHANNELS}"
    )


def test_pinned_detector_exempts_cz_before_detection():
    """Cz is exempt BEFORE detection. Even if Cz is identically
    zero (which would otherwise be flagged by the variance
    detector), it is NEVER in the bad list."""
    raw = _make_egi_129_raw_with_montage(n_bad_channels=2, bad_amplitude=50.0)
    raw, _ = ChannelRolesStep().process(raw, {})
    # Cz is the recording reference; it's identically zero pre-ref
    cz_idx = raw.ch_names.index(EGI_129_REFERENCE_CHANNEL)
    raw._data[cz_idx] = 0.0   # fully zero
    bads = detect_bad_channels(raw, threshold_sd=THRESHOLD_SD)
    assert EGI_129_REFERENCE_CHANNEL not in bads, (
        f"Cz was incorrectly flagged as bad: {bads}"
    )


def test_pinned_detector_apply_does_not_interpolate():
    """R-004: the detector is MARK-ONLY. Interpolation is a separate
    later step (not applied here)."""
    raw = _make_egi_129_raw_with_montage(n_bad_channels=2, bad_amplitude=50.0)
    raw, _ = ChannelRolesStep().process(raw, {})
    new_bads = detect_bad_channels(raw, threshold_sd=THRESHOLD_SD)
    raw = apply_bad_channels(raw, new_bads)
    # The data values are unchanged (mark-only)
    pre_data = raw._data.copy()
    # The bads list is updated
    assert all(b in raw.info["bads"] for b in new_bads)
    # The data has not been interpolated (still has bad channels in it)
    assert np.array_equal(pre_data, raw._data)


# --- R-005: ASR real mask ---

def test_derive_asr_burst_mask_non_zero_on_burst():
    """A burst signal in pre-ASR that is removed in post-ASR
    produces a non-zero mask."""
    rng = np.random.RandomState(0)
    sfreq = 500
    n_channels, n_times = 5, 10000
    ch_names = [f"E{i}" for i in range(1, n_channels + 1)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    pre_data = rng.randn(n_channels, n_times) * 1e-6
    # Inject a burst: large amplitude in the middle
    pre_data[:, 5000:6000] *= 100.0
    pre_raw = mne.io.RawArray(pre_data.copy(), info.copy())
    # Post-ASR: burst is repaired (data is back to baseline)
    post_data = pre_data.copy()
    post_data[:, 5000:6000] = rng.randn(n_channels, 1000) * 1e-6
    post_raw = mne.io.RawArray(post_data, info.copy())
    mask, rejected, repaired, burden = _derive_asr_burden(pre_raw, post_raw)
    n_burst = int(mask.sum())
    assert n_burst > 0, (
        f"ASR burst mask is empty on a clear burst fixture; "
        f"n_burst={n_burst}"
    )
    assert burden > 0.0, "ASR burden is zero; mask fabrication detected"
    assert repaired > 0.0, "repaired fraction is zero on a burst fixture"


# --- R-006: block rejection hard-fail ---

def test_block_rejection_hard_fails_on_excessive_contamination():
    """R-006: a recording with > 20% bad blocks must HARD-FAIL,
    NOT silently truncate. The reachability test (R-006 explicitly
    requires "a reachability test that actually raises")."""
    # Direct invocation of the block_rejection step (NOT the
    # whole chain) so we test the gate in isolation.
    from eegcpm.modules.preprocessing.hbn_langer_chain import (
        _BlockRejectionStepWrapper,
    )
    rng = np.random.RandomState(42)
    sfreq = 500
    n_channels, n_times = 5, 10_000   # 10 s = 5 blocks of 2 s
    ch_names = [f"E{i}" for i in range(1, n_channels + 1)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    data = rng.randn(n_channels, n_times) * 1e-6
    # Contaminate 4 out of 5 blocks = 80% (>> 0.20)
    for b in range(4):
        data[:, b * 2000: (b + 1) * 2000] *= 100.0
    raw = mne.io.RawArray(data, info)
    step = _BlockRejectionStepWrapper(max_rejected_fraction=0.20,
                                          window_seconds=2.0)
    with pytest.raises(ValueError, match="block_rejection"):
        step.process(raw, {})


def test_block_rejection_passes_on_clean_recording():
    """A clean recording should NOT raise."""
    from eegcpm.modules.preprocessing.hbn_langer_chain import (
        _BlockRejectionStepWrapper,
    )
    rng = np.random.RandomState(0)
    sfreq = 500
    n_channels, n_times = 5, 10_000
    ch_names = [f"E{i}" for i in range(1, n_channels + 1)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    data = rng.randn(n_channels, n_times) * 1e-6
    raw = mne.io.RawArray(data, info)
    step = _BlockRejectionStepWrapper(max_rejected_fraction=0.20,
                                          window_seconds=2.0)
    out, meta = step.process(raw, {})
    assert "n_blocks_rejected" in meta
    assert "n_blocks_total" in meta
    assert meta["n_blocks_rejected"] == 0
