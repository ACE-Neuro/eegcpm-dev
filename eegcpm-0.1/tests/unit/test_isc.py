"""Tests for the ISC module (per spec §3.m + S13)."""

import numpy as np
import pandas as pd
import pytest

from eegcpm.modules.features.isc_cca import (
    ISC_BAND_HZ,
    ISC_N_COMPONENTS,
    ISC_REGULARIZATION,
    fit_template,
    isc_effective_sample_ratio,
    isc_regression_freedman_lane,
    ledoit_wolf_shrinkage,
    process,
    regularized_covariance,
    transform,
)


# --------------------------------------------------------------- pin tests (S13)

def test_isc_band_pinned_to_broadband():
    """S13c: ISC band is broadband 2-45 Hz (the only band with
    effective ratio > 1)."""
    assert ISC_BAND_HZ == (2.0, 45.0)


def test_isc_n_components_is_1():
    """S13d: n_components=1 under the pinned threshold."""
    assert ISC_N_COMPONENTS == 1


def test_isc_regularization_is_ledoit_wolf():
    assert ISC_REGULARIZATION == "ledoit_wolf"


# --------------------------------------------------------------- effective ratio

def test_effective_ratio_per_band():
    """Per-band effective ratio table at T=175 s, 109 ch:
    delta 0.12, theta 0.23, alpha 0.29, beta 0.99, gamma 0.88,
    broadband 2-45 2.51. (S13c)"""
    n_ch = 109
    T = 175
    assert isc_effective_sample_ratio(n_ch, T, (2, 4)) == pytest.approx(0.12, abs=0.01)
    assert isc_effective_sample_ratio(n_ch, T, (4, 8)) == pytest.approx(0.23, abs=0.01)
    assert isc_effective_sample_ratio(n_ch, T, (8, 13)) == pytest.approx(0.29, abs=0.01)
    assert isc_effective_sample_ratio(n_ch, T, (13, 30)) == pytest.approx(0.99, abs=0.05)
    assert isc_effective_sample_ratio(n_ch, T, (30, 45)) == pytest.approx(0.88, abs=0.05)
    assert isc_effective_sample_ratio(n_ch, T, (2, 45)) == pytest.approx(2.51, abs=0.05)


# --------------------------------------------------------------- Ledoit-Wolf

def test_ledoit_wolf_shrinkage_in_unit_interval():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 20)
    alpha = ledoit_wolf_shrinkage(X)
    assert 0 <= alpha <= 1


def test_regularized_covariance_psd():
    """Ledoit-Wolf regularized covariance is positive semidefinite."""
    rng = np.random.RandomState(0)
    X = rng.randn(50, 20)
    S = regularized_covariance(X, regularization="ledoit_wolf")
    eigvals = np.linalg.eigvalsh(S)
    assert np.all(eigvals >= -1e-10), (
        f"Regularized cov has negative eigenvalues: min={eigvals.min()}"
    )


# --------------------------------------------------------------- ISC pipeline

def test_fit_template_and_transform_produces_one_row_per_subject():
    """ISC pipeline produces one row per subject with isc and
    n_zeroed columns."""
    rng = np.random.RandomState(42)
    n_ch = 20  # small for speed
    n_subj = 5
    n_times = 1000
    sfreq = 100.0
    # Common sinusoidal signal across all subjects + per-subject noise
    t = np.arange(n_times) / sfreq
    common = np.sin(2 * np.pi * 10 * t)
    data = {}
    for i in range(n_subj):
        subj_signal = common + 0.5 * rng.randn(n_ch, n_times)
        data[f"S{i}"] = subj_signal
    template = fit_template(data, sfreq=sfreq)
    df = transform(data, template)
    assert len(df) == n_subj
    assert "isc" in df.columns
    assert "n_zeroed" in df.columns
    assert "subject_id" in df.columns


def test_isc_high_with_shared_signal():
    """With a strong shared signal, ISC should be high."""
    rng = np.random.RandomState(0)
    n_ch = 20
    n_subj = 5
    n_times = 1000
    sfreq = 100.0
    t = np.arange(n_times) / sfreq
    common = np.sin(2 * np.pi * 10 * t) * 3.0
    data = {}
    for i in range(n_subj):
        subj_signal = np.broadcast_to(common, (n_ch, n_times)).copy()
        subj_signal += 0.01 * rng.randn(n_ch, n_times)
        data[f"S{i}"] = subj_signal
    template = fit_template(data, sfreq=sfreq)
    df = transform(data, template)
    # ISC should be very high
    assert df["isc"].mean() > 0.8, (
        f"ISC mean={df['isc'].mean()}; expected > 0.8 with shared signal"
    )


def test_isc_low_with_independent_signals():
    """With independent signals, ISC should be near zero."""
    rng = np.random.RandomState(0)
    n_ch = 20
    n_subj = 5
    n_times = 1000
    sfreq = 100.0
    data = {f"S{i}": rng.randn(n_ch, n_times) for i in range(n_subj)}
    template = fit_template(data, sfreq=sfreq)
    df = transform(data, template)
    # ISC should be near zero (independent)
    assert abs(df["isc"].mean()) < 0.3, (
        f"ISC mean={df['isc'].mean()}; expected ~0 with independent signals"
    )


def test_n_zeroed_recorded_in_dataframe():
    """n_zeroed is recorded per subject."""
    rng = np.random.RandomState(0)
    n_ch = 20
    n_subj = 3
    n_times = 500
    sfreq = 100.0
    data = {f"S{i}": rng.randn(n_ch, n_times) for i in range(n_subj)}
    bad_channels = {"S0": [0, 1, 2], "S1": [5], "S2": []}
    template = fit_template(data, sfreq=sfreq)
    df = transform(data, template, bad_channels=bad_channels)
    assert df.set_index("subject_id").loc["S0", "n_zeroed"] == 3
    assert df.set_index("subject_id").loc["S1", "n_zeroed"] == 1
    assert df.set_index("subject_id").loc["S2", "n_zeroed"] == 0


# --------------------------------------------------------------- lower-tail (S13a)

def test_isc_regression_lower_tail_with_known_negative():
    """S13a: feed a known negative r(ISC, d); the lower-tail p
    should be < 0.05."""
    rng = np.random.RandomState(0)
    n = 200
    # Known negative: higher d -> lower ISC
    d = rng.randn(n)
    isc = -0.30 * d + 0.10 * rng.randn(n)
    cov = rng.randn(n, 3)
    isc_df = pd.DataFrame({
        "subject_id": range(n),
        "isc": isc,
    })
    d_df = pd.DataFrame({
        "subject_id": range(n),
        "d": d,
        "age_spline_1": cov[:, 0],
        "sex_M": cov[:, 1],
        "site_NA": cov[:, 2],
        "n_zeroed_channels": np.random.RandomState(0).randint(0, 10, n),
        "zeroed_topography_diversity": np.random.RandomState(0).uniform(0, 1, n),
    })
    cfg = type("Cfg", (), {})()
    cfg.n_permutations_isc = 200
    cfg.permutation_seed = 42
    r_obs, p, null = isc_regression_freedman_lane(isc_df, d_df, cfg)
    # r_obs is negative (known negative effect)
    assert r_obs < 0, f"r_obs should be negative; got {r_obs}"
    # The lower-tail p should be small
    assert p < 0.05, f"lower-tail p={p} for known negative r_obs"


# --------------------------------------------------------------- fit_template/transform separation (ENG-014)

def test_fit_template_transform_separation():
    """ENG-014: fit_template returns a template dict; transform uses
    that template to compute ISC scores."""
    rng = np.random.RandomState(0)
    n_ch = 10
    n_subj = 3
    n_times = 500
    sfreq = 100.0
    data = {f"S{i}": rng.randn(n_ch, n_times) for i in range(n_subj)}
    template = fit_template(data, sfreq=sfreq)
    # transform accepts a separate template argument
    df = transform(data, template)
    assert len(df) == n_subj


# --------------------------------------------------------------- mandatory covariates (METH-017)

def test_isc_regression_uses_n_zeroed_as_mandatory_covariate():
    """METH-017: n_zeroed_channels is a mandatory covariate; without
    it, the regression omits it from the covariates list (but the
    test just verifies the function runs with it present)."""
    rng = np.random.RandomState(0)
    n = 100
    d = rng.randn(n)
    isc = -0.20 * d + 0.30 * rng.randn(n)  # add noise so r is not extreme
    isc_df = pd.DataFrame({"subject_id": range(n), "isc": isc})
    d_df = pd.DataFrame({
        "subject_id": range(n),
        "d": d,
        "n_zeroed_channels": np.random.RandomState(0).randint(0, 5, n),
        "zeroed_topography_diversity": np.random.RandomState(0).uniform(0, 1, n),
    })
    cfg = type("Cfg", (), {})()
    cfg.n_permutations_isc = 50
    cfg.permutation_seed = 0
    # No demographics columns: the regression should still work
    r_obs, p, null = isc_regression_freedman_lane(isc_df, d_df, cfg)
    assert isinstance(r_obs, float)
    assert isinstance(p, float)
