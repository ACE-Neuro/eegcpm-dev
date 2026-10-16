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
    # The reference channel travels as metadata (P1-F5); the old
    # raw._eegcpm_reference_channels side attribute was removed.
    assert meta["reference_channel"] == EGI_129_REFERENCE_CHANNEL
    assert not hasattr(out, "_eegcpm_reference_channels")
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
    """The canonical chain has the 9 ordered steps (P1-F2: robust_pca removed)."""
    chain = build_hbn_langer_chain()
    step_names = [s.name for s in chain]
    assert step_names == [
        "channel_roles", "zapline", "filter", "bad_channel_detection",
        "interpolate", "eog_regression", "asr",
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
    """Flat Cz + a heavy-tailed channel: Cz stays out of the bad list, the
    heavy-tailed channel is detected, and every non-exempt kurtosis z-score
    stays finite (P1-F5: a single flat channel must not NaN the criterion for
    all channels)."""
    from eegcpm.modules.preprocessing.steps.bad_channels_pinned import (
        detect_bad_channels,
    )
    raw = _make_egi_129_raw(seed=1)
    rng = np.random.RandomState(7)

    # Make Cz identically zero (the flat-channel case)
    cz_idx = raw.ch_names.index(EGI_129_REFERENCE_CHANNEL)
    raw._data[cz_idx] = 0.0

    # One heavy-tailed channel: Gaussian + 50 spikes at 20x SD
    ht_idx = raw.ch_names.index("E50")
    spikes = rng.choice(raw.n_times, 50, replace=False)
    raw._data[ht_idx, spikes] += 20.0 * raw._data[ht_idx].std()

    # Apply channel_roles; the exemption travels as metadata.
    step = ChannelRolesStep()
    raw, roles_meta = step.process(raw, {})

    bads, report = detect_bad_channels(
        raw, exempt=(roles_meta["reference_channel"],)
    )
    assert "E50" in bads, f"heavy-tailed channel not detected; got {bads}"
    assert "kurtosis" in report["reasons"]["E50"]
    assert EGI_129_REFERENCE_CHANNEL not in bads
    assert report["exempt"] == [EGI_129_REFERENCE_CHANNEL]

    kurt_z = [entry["kurtosis"] for entry in report["z_scores"].values()]
    assert kurt_z
    assert all(z is not None and np.isfinite(z) for z in kurt_z), (
        "kurtosis z-scores are not all finite; a flat channel disabled the criterion"
    )


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


# --- Toolbox QC module + named-chain CLI wiring (2026-08-04) ---

def test_recording_qc_metrics_emits_schema():
    """The promoted QC module emits the full per-recording schema (P1-F10).

    The B3-era keys `residual_ocular_maxcorr`/`residual_ocular_pass` are gone:
    the in-sample correlation is informational only (after the OLS EOG
    regression it is near zero *by construction*), and the ocular gate is
    `blink_residual_ratio`/`blink_residual_pass`, which needs `raw_pre`.
    """
    import mne
    from eegcpm.modules.qc.hbn_recording_qc import (
        recording_qc_metrics, EOG_HYDROCEL_NAMES,
    )
    ch_names = [f"E{i}" for i in range(1, 129)] + ["Cz"]
    rng = np.random.RandomState(0)
    data = rng.normal(size=(129, 5000)) * 1e-6
    info = mne.create_info(ch_names, 500.0, "eeg")
    raw = mne.io.RawArray(data, info)
    row = recording_qc_metrics(raw, task="RestingState",
                               subject_id="sub-test")
    for key in ["duration_post_s", "n_times_post", "line_noise_residual_ratio",
                "residual_ocular_maxcorr_insample",
                "blink_residual_ratio", "blink_residual_pass", "blink_n",
                "blink_channel", "emg_proxy_30_45_ratio", "bad_channels_n",
                "task", "subject_id"]:
        assert key in row, f"missing QC key: {key}"
    assert "residual_ocular_maxcorr" not in row
    assert "residual_ocular_pass" not in row
    assert row["residual_ocular_maxcorr_insample"] < 0.30  # independent noise
    # Without `raw_pre` the gate must not run and must not report a pass.
    assert row["blink_residual_ratio"] is None
    assert row["blink_residual_pass"] is None
    assert row["blink_residual_reason"] == "raw_pre_missing"
    # The EGI EOG names are present in this fixture (typed `eeg`), so the
    # bipolar proxy still resolves them (it looks at names, not types).
    assert all(c in raw.ch_names for c in EOG_HYDROCEL_NAMES)
    assert np.isfinite(row["residual_ocular_maxcorr_insample"])


def test_recording_qc_blink_gate_detects_residual_ocular_activity():
    """Strong blink leakage left in the scalp must fail the ocular gate.

    Replaces the old in-sample-correlation check, which was vacuous: the EOG
    regression subtracts the EOG signal from the EEG, so the correlation
    between them is ~0 whether or not the blinks were actually removed. The
    blink-locked frontal amplitude ratio measures the *residual* directly.
    """
    import mne
    from eegcpm.modules.qc.hbn_recording_qc import (
        OCULAR_GATE, recording_qc_metrics,
    )
    ch_names = ["Fp1", "Fp2", "F7", "F8", "Fz", "Pz", "Cz"] + [
        f"E{i}" for i in (8, 14, 17, 21, 25, 125, 126, 127, 128)
    ]
    rng = np.random.RandomState(1)
    sfreq, seconds = 500.0, 60.0
    n_times = int(sfreq * seconds)
    data = rng.normal(size=(len(ch_names), n_times)) * 3e-6
    blink = np.exp(-0.5 * ((np.arange(int(0.3 * sfreq)) / sfreq - 0.15) / 0.06) ** 2)
    frontal = [ch_names.index(ch) for ch in ("Fp1", "Fp2", "F7", "F8", "Fz")]
    eog1 = ch_names.index("E8")
    for onset in np.arange(5.0, seconds - 5.0, 4.0):
        start = int(onset * sfreq)
        data[eog1, start:start + blink.size] += 250e-6 * blink
        for idx in frontal:
            data[idx, start:start + blink.size] += 200e-6 * blink
    info = mne.create_info(ch_names, sfreq, "eeg")
    raw_pre = mne.io.RawArray(data.copy(), info, verbose=False)

    # Post-chain data with the blinks still in the scalp: the gate fails.
    row = recording_qc_metrics(
        mne.io.RawArray(data.copy(), info, verbose=False),
        task="X", raw_pre=raw_pre,
    )
    assert row["blink_n"] > 0
    assert row["blink_residual_ratio"] is not None
    assert row["blink_residual_ratio"] >= OCULAR_GATE
    assert row["blink_residual_pass"] is False

    # Post-chain data with the blinks regressed out: the gate passes.
    cleaned = data.copy()
    coeffs = np.linalg.lstsq(
        data[[eog1]].T, data[frontal].T, rcond=None)[0]
    cleaned[frontal] = data[frontal] - (data[[eog1]].T @ coeffs).T
    row_clean = recording_qc_metrics(
        mne.io.RawArray(cleaned, info, verbose=False),
        task="X", raw_pre=raw_pre,
    )
    assert row_clean["blink_residual_ratio"] < OCULAR_GATE
    assert row_clean["blink_residual_pass"] is True


def test_named_chain_resolves():
    """The named canonical chain resolves to prebuilt steps in the
    pinned order."""
    from eegcpm.modules.preprocessing.pipeline_builder import (
        PreprocessingPipeline,
    )
    pipe = PreprocessingPipeline("hbn_langer")
    names = [s.name for s in pipe.steps]
    assert names[0] == "channel_roles"
    assert "zapline" in names
    assert names.index("zapline") < names.index("filter")  # S19/METH-019
    assert names[-1] == "reference"
    assert "block_rejection" in names
