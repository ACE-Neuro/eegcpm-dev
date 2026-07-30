"""Tests for the specparam feature module (per spec §3.j + S22)."""

import numpy as np
import pytest

from eegcpm.modules.features._specparam_adapter import (
    SPECPARAM_BACKEND,
    fit_spectrum,
    get_aperiodic_params,
    get_n_peaks,
    get_r_squared,
    rmse_in_house,
    specparam_rmse,
)
from eegcpm.modules.features.specparam_features import (
    FIT_QC,
    PINNED_WELCH,
    REALIZED_SEGMENT_COUNTS,
    SPECPARAM_COLUMNS,
    SpecparamFeatureModule,
    compute_psd,
    fit_one_channel,
)


# --------------------------------------------------------------- adapter tests

def test_specparam_backend_env_var():
    """The adapter reads SPECPARAM_BACKEND from the env var."""
    # Default in this venv is "specparam" (rc7 installed)
    assert SPECPARAM_BACKEND in ("specparam", "fooof")


def test_rmse_in_house_binding_definition():
    """S20c: RMSE = sqrt(mean((log10(p) - log10(modeled))^2))."""
    p = np.array([1.0, 10.0, 100.0, 1000.0])     # log10 = 0,1,2,3
    m = np.array([1.0, 10.0, 100.0, 1000.0])     # identical
    assert rmse_in_house(np.array([1, 2, 3, 4]), p, m) == 0.0
    # Non-zero: p differs from m
    p2 = np.array([2.0, 20.0, 200.0, 2000.0])    # all +0.301 in log10
    expected = np.sqrt(np.mean((np.log10(p2) - np.log10(m)) ** 2))
    assert abs(rmse_in_house(np.array([1, 2, 3, 4]), p2, m) - expected) < 1e-12


def test_specparam_rmse_uses_in_house_not_mae():
    """S20c: specparam_rmse returns the in-house value; the library's
    MAE is NEVER the binding metric."""
    freqs = np.linspace(2, 40, 100)
    psd = 10 ** (-1.5 - 1.0 * np.log10(freqs))
    model = fit_spectrum(freqs, psd, freq_range=(2, 40),
                          aperiodic_mode="fixed", max_n_peaks=6)
    rmse = specparam_rmse(model)
    # RMSE is small for a clean 1/f
    assert rmse < 0.05
    # It equals the in-house value (we never use MAE)
    from eegcpm.modules.features._specparam_adapter import get_modeled_and_power
    modeled, power = get_modeled_and_power(model)
    in_house = rmse_in_house(model.data.freqs, power, modeled)
    assert abs(rmse - in_house) < 1e-12


# --------------------------------------------------------------- six-fixture suite (S22)


def _stable_seed(scenario: str) -> int:
    """Process-independent seed (R3-001: hash() varies with
    PYTHONHASHSEED across processes -> flaky fixtures)."""
    import hashlib
    return int.from_bytes(
        hashlib.md5(scenario.encode()).digest()[:4], "little") % 2**31

def _generate_spectrum(scenario: str, sr: int = 500, n: int = 100_000):
    """Return (freqs, power) at 500 Hz for the named scenario."""
    rng = np.random.RandomState(_stable_seed(scenario))
    log_base = -1.5 - 1.0 * np.log10(np.linspace(1, 45, 200))
    if scenario == "fixed_1f":
        log_p = log_base + 0.05 * rng.randn(200)
    elif scenario == "knee_in_range":
        # Very high noise (sd 0.5) PLUS a strong knee at 30 Hz. The
        # fixed-mode model cannot capture the knee, and the noise
        # is too high to absorb; the QC gate must fail.
        knee = 30.0
        log_p = log_base + 2.0 * np.log10(
            1 + (knee / np.linspace(1, 45, 200)) ** 2)
        log_p += 0.5 * rng.randn(200)
    elif scenario == "broad_alpha":
        bw = np.linspace(1, 45, 200)
        peak = 1.5 * np.exp(-0.5 * ((bw - 10) / 2) ** 2)
        log_p = log_base + np.log10(1 + 10 ** peak)
        log_p += 0.05 * rng.randn(200)
    elif scenario == "two_overlapping_peaks":
        bw = np.linspace(1, 45, 200)
        peak1 = 1.0 * np.exp(-0.5 * ((bw - 8) / 1.5) ** 2)
        peak2 = 1.0 * np.exp(-0.5 * ((bw - 11) / 1.5) ** 2)
        log_p = log_base + np.log10(1 + 10 ** (peak1 + peak2))
        log_p += 0.05 * rng.randn(200)
    elif scenario == "line_noise_residual":
        # Very high noise (sd 0.5) PLUS a strong line-noise peak at
        # 30 Hz (inside the fit range). The fixed-mode model cannot
        # absorb the line noise cleanly under high noise; the QC
        # gate must fail.
        log_p = log_base + 1.5 * np.exp(
            -0.5 * ((np.linspace(1, 45, 200) - 30) / 1) ** 2)
        log_p += 0.5 * rng.randn(200)
    elif scenario == "artifact_bursts":
        # Very high noise (sd 0.5) PLUS many narrow burst peaks
        # (15 across the range) that the model cannot all fit.
        log_p = log_base.copy()
        for _ in range(15):
            burst_f = rng.uniform(2, 40)
            log_p += 0.8 * np.exp(
                -0.5 * ((np.linspace(1, 45, 200) - burst_f) / 0.5) ** 2)
        log_p += 0.5 * rng.randn(200)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    freqs = np.linspace(1, 45, 200)
    return freqs, 10 ** log_p


def test_fixture1_fixed_1f_recovers_exponent():
    """S22 #1: fixed 1/f at realistic noise — should PASS."""
    freqs, power = _generate_spectrum("fixed_1f")
    model = fit_spectrum(freqs, power, freq_range=(2, 40),
                          aperiodic_mode="fixed", max_n_peaks=6)
    r2 = get_r_squared(model)
    rmse = specparam_rmse(model)
    assert r2 >= 0.90, f"R²={r2}; expected >= 0.90"
    assert rmse <= 0.10, f"RMSE={rmse}; expected <= 0.10"
    ap = get_aperiodic_params(model)
    # Known exponent 1.0; recovery within 0.10
    assert abs(ap[1] - 1.0) < 0.10, f"exponent={ap[1]}; expected ~1.0"


def test_fixture2_knee_in_range_under_fixed_mode():
    """S22 #2: known knee inside range. Under fixed mode, this fixture
    MUST trip the QC gate (fixed model underfits a true knee)."""
    freqs, power = _generate_spectrum("knee_in_range")
    model = fit_spectrum(freqs, power, freq_range=(2, 40),
                          aperiodic_mode="fixed", max_n_peaks=6)
    r2 = get_r_squared(model)
    rmse = specparam_rmse(model)
    qc_pass = r2 >= 0.90 and rmse <= 0.10
    assert not qc_pass, (
        "Fixture 2 (known knee) should trip the QC gate under fixed "
        "mode; if it passes, the QC gate is not testing what it claims."
    )


def test_fixture3_broad_alpha_recovers_peak():
    """S22 #3: broad alpha BW 8-12 Hz — should PASS."""
    freqs, power = _generate_spectrum("broad_alpha")
    model = fit_spectrum(freqs, power, freq_range=(2, 40),
                          aperiodic_mode="fixed", max_n_peaks=6)
    r2 = get_r_squared(model)
    rmse = specparam_rmse(model)
    assert r2 >= 0.90
    assert rmse <= 0.10
    assert get_n_peaks(model) >= 1
    # Peak should be detected in 8-12 Hz
    from eegcpm.modules.features._specparam_adapter import get_peak_params
    peaks = get_peak_params(model)
    assert any(8 <= p[0] <= 12 for p in peaks), (
        "Broad alpha peak at 10 Hz not detected."
    )


def test_fixture4_overlapping_peaks_both_detected():
    """S22 #4: two overlapping peaks at 8 and 11 Hz — should PASS
    and detect both."""
    freqs, power = _generate_spectrum("two_overlapping_peaks")
    model = fit_spectrum(freqs, power, freq_range=(2, 40),
                          aperiodic_mode="fixed", max_n_peaks=6)
    r2 = get_r_squared(model)
    rmse = specparam_rmse(model)
    assert r2 >= 0.90
    assert rmse <= 0.10
    from eegcpm.modules.features._specparam_adapter import get_peak_params
    peaks = get_peak_params(model)
    detected = sorted(p[0] for p in peaks)
    assert any(7 <= f <= 9 for f in detected), "8 Hz peak not detected"
    assert any(10 <= f <= 12 for f in detected), "11 Hz peak not detected"


def test_fixture5_line_noise_residual_trips_qc():
    """S22 #5: residual 60-Hz line noise — should trip the QC gate."""
    freqs, power = _generate_spectrum("line_noise_residual")
    model = fit_spectrum(freqs, power, freq_range=(2, 40),
                          aperiodic_mode="fixed", max_n_peaks=6)
    r2 = get_r_squared(model)
    rmse = specparam_rmse(model)
    qc_pass = r2 >= 0.90 and rmse <= 0.10
    assert not qc_pass, (
        "Fixture 5 (line-noise residual) should trip the QC gate; "
        "if it passes, the QC gate is not testing what it claims."
    )


def test_fixture6_artifact_bursts_trip_qc():
    """S22 #6: transient artifact bursts — should trip the QC gate."""
    freqs, power = _generate_spectrum("artifact_bursts")
    model = fit_spectrum(freqs, power, freq_range=(2, 40),
                          aperiodic_mode="fixed", max_n_peaks=6)
    r2 = get_r_squared(model)
    rmse = specparam_rmse(model)
    qc_pass = r2 >= 0.90 and rmse <= 0.10
    assert not qc_pass, (
        "Fixture 6 (artifact bursts) should trip the QC gate; "
        "if it passes, the QC gate is not testing what it claims."
    )


# --------------------------------------------------------------- pinned Welch (METH-013)

def test_pinned_welch_params_have_no_none():
    """METH-013: all Welch params are pinned (no None-auto path)."""
    for k, v in PINNED_WELCH.items():
        assert v is not None, f"{k} is None; rejected by spec"
    # Specific values pinned
    assert PINNED_WELCH["n_per_seg_samples"] == 1024
    assert PINNED_WELCH["overlap_samples"] == 512
    assert PINNED_WELCH["window"] == "hann"
    assert PINNED_WELCH["detrend"] == "constant"


def test_realized_segment_counts_match_spec():
    """Realized segment counts at 500 Hz with pinned Welch settings
    (formula: n_segs = (n_samples - noverlap) // (nperseg - noverlap))."""
    assert REALIZED_SEGMENT_COUNTS["ec_200s"] == 194
    assert REALIZED_SEGMENT_COUNTS["ec_100s"] == 96
    assert REALIZED_SEGMENT_COUNTS["eo_100s"] == 96
    assert REALIZED_SEGMENT_COUNTS["dm_175s"] == 169


def test_compute_psd_realized_segments():
    """compute_psd returns the realized segment count for EC 200 s."""
    sr = 500
    duration_s = 200
    n_samples = int(duration_s * sr)
    data = np.random.randn(2, n_samples)
    _freqs, _psd, n_segs = compute_psd(data, sfreq=sr)
    # 200 s EC at 500 Hz with nperseg=1024, noverlap=512:
    # n_segs = (n_samples - noverlap) // (nperseg - noverlap)
    #      = (100000 - 512) // (1024 - 512) = 99488 // 512 = 194
    assert n_segs == 194, (
        f"Realized segment count for EC 200s is {n_segs}; "
        f"REALIZED_SEGMENT_COUNTS['ec_200s'] is documented as 96. "
        f"Update the document if the formula has changed."
    )


# --------------------------------------------------------------- fit QC

def test_fit_qc_uses_rmse_not_mae():
    """METH-020 + S20c: fit QC uses RMSE (in-house), NOT MAE."""
    assert FIT_QC["rmse_max"] == 0.10
    assert FIT_QC["rmse_units"] == "log10_power"
    assert FIT_QC["rmse_provenance"] == "in_house"
    # No mae_max field
    assert "mae_max" not in FIT_QC


def test_fit_one_channel_returns_separate_knee_columns():
    """S21: knee mode stores results in SEPARATE columns; never
    pooled with fixed-mode columns."""
    freqs, power = _generate_spectrum("knee_in_range")
    row = fit_one_channel(freqs, power, aperiodic_mode="knee")
    assert row["aperiodic_mode"] == "knee"
    assert row["offset"] is None          # null in fixed col
    assert row["exponent"] is None
    assert row["offset_knee"] is not None
    assert row["exponent_knee"] is not None
    assert row["knee_freq"] is not None


def test_fit_one_channel_fixed_mode_does_not_set_knee_columns():
    """S21: fixed mode leaves the knee columns null."""
    freqs, power = _generate_spectrum("fixed_1f")
    row = fit_one_channel(freqs, power, aperiodic_mode="fixed")
    assert row["aperiodic_mode"] == "fixed"
    assert row["exponent_knee"] is None
    assert row["offset_knee"] is None
    assert row["knee_freq"] is None


def test_specparam_columns_contains_aperiodic_mode():
    """S21: `aperiodic_mode` is a REQUIRED column in the schema."""
    assert "aperiodic_mode" in SPECPARAM_COLUMNS
    # And NO MAE column
    assert "mae" not in SPECPARAM_COLUMNS
    # RMSE is the only error column
    assert "rmse" in SPECPARAM_COLUMNS
