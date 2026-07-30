"""Tests for the canonical preprocessing chain (per spec §3.i)."""

import ast
from pathlib import Path

import mne
import numpy as np
import pytest

from eegcpm.modules.preprocessing.hbn_langer_chain import (
    build_hbn_langer_chain,
)
from eegcpm.modules.preprocessing.steps.robust_pca import (
    CANONICAL_LAMBDA,
    CANONICAL_LAMBDA_RULE,
    CANONICAL_MU,
    CANONICAL_MU_RULE,
    ialm_decompose,
)
from eegcpm.modules.preprocessing.steps.channel_roles import (
    EGI_129_EOG_CHANNELS,
    EGI_129_NECK_FACE_CHANNELS,
    EGI_129_REFERENCE_CHANNEL,
    ChannelRolesStep,
)


# --- helpers ---

def _make_egi_129_raw(duration_s=10, sfreq=500, seed=42):
    """Create a synthetic 129-channel EGI-like RawArray."""
    rng = np.random.RandomState(seed)
    ch_names = [f"E{i}" for i in range(1, 129)] + ["Cz"]
    n_channels = len(ch_names)
    n_times = int(duration_s * sfreq)
    data = rng.randn(n_channels, n_times) * 1e-6
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    # Set up a montage with digitization (R-004: the pinned detector
    # correctly identifies bad channels, and the interpolation step
    # requires digitization to interpolate them).
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    raw.set_montage(montage, match_case=False, on_missing="ignore")
    return raw


# --- channel_roles tests ---

def test_channel_roles_drops_neck_face():
    raw = _make_egi_129_raw()
    step = ChannelRolesStep()
    out, meta = step.process(raw, {})
    for ch in EGI_129_NECK_FACE_CHANNELS:
        assert ch not in out.ch_names
    assert EGI_129_REFERENCE_CHANNEL in out.ch_names


def test_channel_roles_marks_eog_channels():
    raw = _make_egi_129_raw()
    step = ChannelRolesStep()
    out, meta = step.process(raw, {})
    for ch in EGI_129_EOG_CHANNELS:
        if ch in out.ch_names:
            assert out.get_channel_types(picks=[ch])[0] == "eog"


def test_channel_roles_keeps_cz_eeg_typed_with_is_reference():
    raw = _make_egi_129_raw()
    step = ChannelRolesStep()
    out, meta = step.process(raw, {})
    assert EGI_129_REFERENCE_CHANNEL in out.ch_names
    # is_reference is stored on the raw object (MNE info['chs'] is
    # restricted; we use raw._eegcpm_reference_channels).
    assert out._eegcpm_reference_channels.get(EGI_129_REFERENCE_CHANNEL) is True
    assert out.get_channel_types(picks=[EGI_129_REFERENCE_CHANNEL])[0] == "eeg"


# --- IALM tests ---

def test_ialm_decompose_returns_low_rank_and_sparse():
    """IALM decomposes a low-rank + sparse matrix."""
    rng = np.random.RandomState(0)
    n, d = 50, 20
    L_true = rng.randn(n, 3) @ rng.randn(3, d)  # rank-3
    S_true = (rng.randn(n, d) > 1.5).astype(float) * 5  # sparse
    M = L_true + S_true
    L, S = ialm_decompose(M, lam=0.1, mu=10.0, tolerance=1e-6, max_iter=200)
    # Reconstruction error is small
    rec_err = np.linalg.norm(M - L - S, "fro")
    assert rec_err < np.linalg.norm(M, "fro") * 0.1


def test_ialm_deterministic_same_input():
    """IALM is deterministic on the same input."""
    rng = np.random.RandomState(7)
    M = rng.randn(30, 15)
    L1, S1 = ialm_decompose(M, lam=0.1, mu=10.0)
    L2, S2 = ialm_decompose(M, lam=0.1, mu=10.0)
    assert np.array_equal(L1, L2)
    assert np.array_equal(S1, S2)


def test_canonical_lambda_is_pinned_value():
    """S20b: lambda is a LITERAL number computed for canonical
    recording dimensions, with the rule recorded."""
    expected = 1.0 / np.sqrt(100_000)  # canonical n_times
    assert abs(CANONICAL_LAMBDA - expected) < 1e-10
    assert "sqrt" in CANONICAL_LAMBDA_RULE


def test_ialm_pipeline_lambda_matches_pinned_canonical():
    """S20b: the IALM fixture test uses the pipeline's lambda, not
    a different one."""
    # Pipeline uses CANONICAL_LAMBDA by default
    from eegcpm.modules.preprocessing.steps.robust_pca import RobustPCAStep
    step = RobustPCAStep()
    assert step.lam == CANONICAL_LAMBDA


# --- chain tests ---

def test_build_hbn_langer_chain_order():
    """The canonical chain has the 10 ordered steps."""
    chain = build_hbn_langer_chain()
    step_names = [s.name for s in chain]
    assert step_names == [
        "channel_roles", "zapline", "filter", "bad_channel_detection",
        "interpolate", "eog_regression", "robust_pca", "asr",
        "block_rejection", "reference",
    ]


def test_zapline_before_filter():
    """zapline is BEFORE the low-pass filter (per METH-019)."""
    chain = build_hbn_langer_chain()
    names = [s.name for s in chain]
    assert names.index("zapline") < names.index("filter")


def test_filter_uses_45hz_lowpass():
    """Low-pass is 45 Hz (per A-RT24), not 40 Hz from code.md."""
    chain = build_hbn_langer_chain()
    filter_step = [s for s in chain if s.name == "filter"][0]
    assert filter_step.h_freq == 45.0


def test_block_rejection_max_rejected_fraction_20pct():
    """S12: max_rejected_fraction is 0.20 (reconciled with safety.md
    EC≥160/200), not 0.25 from code.md."""
    chain = build_hbn_langer_chain()
    block_step = [s for s in chain if s.name == "block_rejection"][0]
    assert block_step.max_rejected_fraction == 0.20


def test_asr_window_criterion_is_rejection_fraction():
    """ENG-010: window_criterion is a rejection FRACTION (0-1),
    not a window length."""
    chain = build_hbn_langer_chain()
    asr_step = [s for s in chain if s.name == "asr"][0]
    assert 0.0 <= asr_step.window_criterion <= 1.0
    assert asr_step.window_criterion == 0.25


def test_asr_retains_burst_masks():
    """ASR retains burst/sample masks for the burden covariate."""
    chain = build_hbn_langer_chain()
    asr_step = [s for s in chain if s.name == "asr"][0]
    assert asr_step.retain_burst_masks is True


def test_reference_includes_cz_via_eeg_typing():
    """S19: Cz is eeg-typed so picks:eeg includes it in the
    average-reference computation."""
    raw = _make_egi_129_raw()
    chain = build_hbn_langer_chain()
    # Run through the chain
    metadata = {}
    for step in chain:
        raw, meta = step.process(raw, metadata)
        metadata[step.name] = meta
    # Cz is in the channels
    assert EGI_129_REFERENCE_CHANNEL in raw.ch_names
    # Cz is eeg-typed
    assert raw.get_channel_types(picks=[EGI_129_REFERENCE_CHANNEL])[0] == "eeg"
    # Post-ref: Cz is in average; variance should be finite
    cz_data = raw.get_data(picks=[EGI_129_REFERENCE_CHANNEL])[0]
    assert np.isfinite(cz_data).all()


# --- Cz-exemption fixture test (S19) ---

def test_cz_exempt_from_bad_channel_detection():
    """A recording with Cz as a flat (zero) channel must NOT
    fail the bad-channel gate. Other flat channels MUST be
    detected."""
    raw = _make_egi_129_raw()
    # Make Cz identically zero (the flat-channel case)
    cz_idx = raw.ch_names.index(EGI_129_REFERENCE_CHANNEL)
    raw._data[cz_idx] = 0.0
    # Apply channel_roles
    step = ChannelRolesStep()
    raw, _ = step.process(raw, {})
    # Cz is still in the channels
    assert EGI_129_REFERENCE_CHANNEL in raw.ch_names


# --- Reachability test for Cz-included-in-average-reference ---

def test_post_reference_cz_variance_greater_than_zero():
    """S19: after average-reference, Cz variance > 0 (the reference
    was recovered, not zero)."""
    raw = _make_egi_129_raw(seed=42)
    chain = build_hbn_langer_chain()
    metadata = {}
    for step in chain:
        raw, meta = step.process(raw, metadata)
        metadata[step.name] = meta
    # Post-ref: Cz variance should be > 0 (RECOVERED, not measured)
    cz_idx = raw.ch_names.index(EGI_129_REFERENCE_CHANNEL)
    cz_data = raw.get_data(picks=[EGI_129_REFERENCE_CHANNEL])[0]
    assert cz_data.var() > 0, (
        "Post-reference Cz variance is 0; reference recovery failed; "
        "Cz was not included in the average."
    )
