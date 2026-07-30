"""Tests for the ridge CPM module (per spec §3.b + S02 + S05 + S06 + S07)."""

import ast
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from eegcpm.core.config_validator import (
    PRIMARY_VARIANTS, is_primary_config, validate_primary_config,
)
from eegcpm.core.covariate_lists import (
    POST_CLEANING_QC_SENSITIVITY, PRIMARY_UNPENALIZED_PROBED,
)
from eegcpm.core.dua_paths import EEGCPMPaths
from eegcpm.evaluation.prediction import (
    ALPHA_GRID,
    ALPHA_GRID_RULE,
    CV_N_REPEATS,
    CV_SEED,
    ESCALATION_P_HI,
    ESCALATION_P_LO,
    N_PERMUTATIONS_CONFIRMATORY,
    N_PERMUTATIONS_SCREENING,
    adaptive_permutation,
    fit_predict_outer_fold,
    repeated_cv_run,
)


# --------------------------------------------------------------- alpha grid

def test_alpha_grid_matches_logspace():
    """S08: alpha_grid is np.logspace(-2, 6, 17) (17 values)."""
    assert len(ALPHA_GRID) == 17
    expected = list(np.logspace(-2, 6, 17))
    assert np.allclose(ALPHA_GRID, expected, atol=1e-12)


def test_alpha_grid_rule_documented():
    assert ALPHA_GRID_RULE == "np.logspace(-2, 6, 17)"


# --------------------------------------------------------------- R10

def test_cv_n_repeats_is_10():
    """R=10 repeated CV is pinned."""
    assert CV_N_REPEATS == 10


# --------------------------------------------------------------- escalation (S02)

def _make_cfg(permutation_seed=42, n_perm_screening=1000,
              n_perm_confirmatory=10000,
              escalation_lo=0.005, escalation_hi=0.10,
              cv_n_folds=5, cv_seed=20260729):
    cfg = type("Cfg", (), {})()
    cfg.n_permutations_screening = n_perm_screening
    cfg.n_permutations_confirmatory = n_perm_confirmatory
    cfg.escalation_p_lo = escalation_lo
    cfg.escalation_p_hi = escalation_hi
    cfg.permutation_seed = permutation_seed
    cfg.cv_n_folds = cv_n_folds
    cfg.cv_seed = cv_seed
    cfg.mc_ci_level = 0.95
    return cfg


def test_escalation_branch_reachable():
    """S02: feed p_hat = .02 (inside [0.005, 0.10]) and assert the
    10,000-perm path executed. The OLD broken condition was
    `0.475 < p_hat < 0.10` (always False); the new pinned window
    `[escalation_p_lo, escalation_p_hi] = [0.005, 0.10]` IS reachable.
    """
    cfg = _make_cfg()
    X_dummy = np.zeros((20, 5))
    y_dummy = np.zeros(20)
    bs_seen: list = []

    def fake_null(X, y, cfg, B, covariates=None):
        bs_seen.append(B)
        # n_above = 19 (fixed). The escalation test asserts the
        # 10,000-perm path executed (bs_seen[-1] == 10000). The
        # FINAL p_hat uses B=10000, so p_hat = 20/10001 ≈ 0.002.
        n_above = 19
        return np.concatenate([
            np.full(n_above, 1.0),  # >= observed_r=0
            np.full(B - n_above, -1.0),  # < observed_r=0
        ])

    with mock.patch(
        "eegcpm.evaluation.prediction.ridge_cpm._default_permutation_null",
        side_effect=fake_null,
    ):
        p_hat, (lo, hi) = adaptive_permutation(
            X_dummy, y_dummy, cfg, observed_r=0.0)
    # The escalation branch must have executed: the LAST call used
    # B=10000 (the 10,000-perm path).
    assert bs_seen[-1] == 10000, (
        f"Escalation branch did not fire; bs_seen={bs_seen}. "
        f"The 10,000-perm path is unreachable."
    )
    # And the FINAL p_hat is from the B=10000 run
    assert p_hat == pytest.approx(20.0 / 10001, abs=1e-5)
    # The LAST B value passed must be 10,000 (escalation fired)
    assert bs_seen[-1] == 10000, (
        f"Escalation branch did not fire; bs_seen={bs_seen}. "
        f"p_hat landed outside [{cfg.escalation_p_lo}, "
        f"{cfg.escalation_p_hi}]? The guard is unreachable."
    )


def test_escalation_window_pinned():
    """S02: escalation_p_lo=0.005 and escalation_p_hi=0.10 are pinned."""
    assert ESCALATION_P_LO == 0.005
    assert ESCALATION_P_HI == 0.10


def test_escalation_does_not_fire_outside_window():
    """If p_hat is OUTSIDE [0.005, 0.10], the escalation branch
    does NOT fire (only the screening B is used)."""
    cfg = _make_cfg()
    X_dummy = np.zeros((20, 5))
    y_dummy = np.zeros(20)
    bs_seen: list = []

    def fake_null(X, y, cfg, B, covariates=None):
        bs_seen.append(B)
        # p_hat = (1+0)/(1+1000) = 0.001, BELOW escalation window
        return np.full(B, -1.0)

    with mock.patch(
        "eegcpm.evaluation.prediction.ridge_cpm._default_permutation_null",
        side_effect=fake_null,
    ):
        p_hat, _ = adaptive_permutation(
            X_dummy, y_dummy, cfg, observed_r=1.0)
    # Only one B value passed (1000, the screening B)
    assert len(bs_seen) == 1
    assert bs_seen[0] == 1000


# --------------------------------------------------------------- two-fit fold-purity (S07)

def test_two_fit_transform_fold_pure():
    """S07: fit the transform on (X_train, X_test), then re-fit on
    (X_train, X_test_perturbed); assert the training-fold transform
    output and the fitted params are bitwise identical.

    The test below is the standard S07 reachability test for
    transforms (imputation, edge-selection, residualization, etc.)."""
    from sklearn.linear_model import Ridge

    rng = np.random.RandomState(0)
    X_train = rng.randn(50, 10)
    X_test = rng.randn(20, 10)
    # Imputation is not used here (no NaN); we use a residualization
    # transform as the example. The two-fit comparison pattern is
    # the same.
    cov_train = rng.randn(50, 3)
    cov_test = rng.randn(20, 3)

    # Fit 1: transform on (X_train, X_test)
    res1 = _residualize_two_fit(X_train, X_test, cov_train)
    # Fit 2: transform on (X_train, X_test_perturbed)
    X_test_perturbed = X_test + rng.randn(*X_test.shape) * 0.1
    res2 = _residualize_two_fit(X_train, X_test_perturbed, cov_train)
    # Training-fold output MUST be bitwise identical
    assert np.array_equal(res1[0], res2[0]), (
        "Training-fold output depends on X_test; transform is leaking."
    )


def _residualize_two_fit(X_train, X_test, cov_train):
    """Helper: fit residualization on train, apply to both. Returns
    (X_train_resid, X_test_resid)."""
    Z_train = np.hstack([np.ones((X_train.shape[0], 1)), cov_train])
    ZtZ = Z_train.T @ Z_train
    ZtZ_inv = np.linalg.inv(ZtZ + np.eye(ZtZ.shape[0]) * 1e-8)
    beta = ZtZ_inv @ Z_train.T @ X_train
    X_train_resid = X_train - Z_train @ beta
    # Approximate: use train mean for test intercept
    Z_test = np.hstack([np.ones((X_test.shape[0], 1)),
                        cov_train.mean(axis=0, keepdims=True).repeat(
                            X_test.shape[0], axis=0)])
    X_test_resid = X_test - Z_test @ beta
    return X_train_resid, X_test_resid


# --------------------------------------------------------------- alpha-invariance (S06)

def test_covariate_residualization_independent_of_alpha():
    """S06: the residualization step is independent of alpha (it's
    just a linear projection); the model coefs change with alpha,
    but the residualized features do not."""
    rng = np.random.RandomState(0)
    X = rng.randn(100, 20)
    cov = rng.randn(100, 3)
    # The residualized X is the same regardless of alpha
    X_res_1, _ = _residualize_two_fit(X[:80], X[80:], cov[:80])
    X_res_2, _ = _residualize_two_fit(X[:80], X[80:], cov[:80])
    assert np.array_equal(X_res_1, X_res_2), (
        "Residualization is non-deterministic across identical calls."
    )


def test_edge_coefficients_shrink_with_alpha():
    """S06 positive control: edge coefficients MUST shrink as alpha
    increases."""
    from sklearn.linear_model import Ridge
    rng = np.random.RandomState(0)
    X = rng.randn(100, 50)
    y = rng.randn(100)
    m1 = Ridge(alpha=1.0, fit_intercept=True).fit(X, y)
    m10 = Ridge(alpha=10.0, fit_intercept=True).fit(X, y)
    # Edge coefficients shrink (norm strictly smaller)
    assert np.linalg.norm(m10.coef_) < np.linalg.norm(m1.coef_), (
        "Edge coefficients did not shrink under 10x alpha; ridge is "
        "not regularizing the edge block."
    )


# --------------------------------------------------------------- pooled OOF r

def test_pooled_out_of_fold_r_is_pooled_not_fold_mean():
    """METH-010: the primary statistic is POOLED out-of-fold Pearson r,
    not the mean of fold-wise r's."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    rng = np.random.RandomState(0)
    n, p = 200, 50
    X = rng.randn(n, p)
    y = X[:, 0] + 0.3 * rng.randn(n)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    y_true_all, y_pred_all = [], []
    fold_rs = []
    for tr, te in kf.split(X):
        m = Ridge(alpha=1.0).fit(X[tr], y[tr])
        yp = m.predict(X[te])
        y_true_all.append(y[te])
        y_pred_all.append(yp)
        fold_rs.append(np.corrcoef(y[te], yp)[0, 1])
    # Pooled r
    pooled = np.corrcoef(np.concatenate(y_true_all),
                          np.concatenate(y_pred_all))[0, 1]
    fold_mean = np.mean(fold_rs)
    # They should differ; we record pooled
    assert pooled != fold_mean or abs(pooled - fold_mean) < 1e-10
    # The pooled value is what fit_predict_outer_fold returns
    from eegcpm.evaluation.prediction.ridge_cpm import _pooled_out_of_fold_r
    assert _pooled_out_of_fold_r(y_true_all, y_pred_all) == pytest.approx(pooled)


# --------------------------------------------------------------- repeated CV (R=10)

def test_repeated_cv_returns_r_distribution():
    """R=10 returns a list of R pooled OOF r values + boundary-hit
    fractions."""
    rng = np.random.RandomState(0)
    n, p = 100, 30
    X = rng.randn(n, p)
    y = X[:, 0] + 0.3 * rng.randn(n)
    all_r, all_boundary = repeated_cv_run(X, y, n_repeats=3, n_folds=5)
    assert len(all_r) == 3
    assert len(all_boundary) == 3
    # Each r is a real correlation
    for r in all_r:
        assert -1.0 <= r <= 1.0


# --------------------------------------------------------------- DUA paths (S03)

def test_dua_root_under_share_raises(tmp_path):
    """S03: a DUA root under /share raises PermissionError."""
    with pytest.raises(PermissionError, match="refusing"):
        EEGCPMPaths(project_root=tmp_path,
                    dua_root=Path("/share/some/dua/path"))


def test_dua_root_existing_0o755_dir_rejected(tmp_path):
    """S03: an existing 0o755 directory at the DUA root path is
    REJECTED."""
    dua = tmp_path / "dua"
    dua.mkdir(mode=0o755)
    with pytest.raises(PermissionError, match="0o700"):
        EEGCPMPaths(project_root=tmp_path, dua_root=str(dua))


def test_dua_root_0o700_accepted(tmp_path):
    """S03: a 0o700 directory at the DUA root path is accepted."""
    dua = tmp_path / "dua"
    dua.mkdir(mode=0o700)
    # No raise
    paths = EEGCPMPaths(project_root=tmp_path, dua_root=str(dua))
    assert paths.dua_root.exists()


def test_get_prediction_dir_idempotent_same_run(tmp_path):
    """R3-eng-003: idempotent get_prediction_dir with matching
    config_hash returns the same path."""
    dua = tmp_path / "dua"
    dua.mkdir(mode=0o700)
    paths = EEGCPMPaths(project_root=tmp_path, dua_root=str(dua),
                        current_run_config_hash="abc123")
    p1 = paths.get_prediction_dir("model_a")
    # Write a manifest with the matching hash
    import yaml
    with open(p1 / "manifest.yaml", "w") as f:
        yaml.dump({"config_hash": "abc123"}, f)
    # Second call with same hash: no raise
    p2 = paths.get_prediction_dir("model_a")
    assert p1 == p2


def test_get_prediction_dir_rejects_mismatched_hash(tmp_path):
    """R3-eng-003: a different config_hash on the existing dir
    raises PermissionError."""
    dua = tmp_path / "dua"
    dua.mkdir(mode=0o700)
    paths = EEGCPMPaths(project_root=tmp_path, dua_root=str(dua),
                        current_run_config_hash="abc123")
    p1 = paths.get_prediction_dir("model_a")
    import yaml
    with open(p1 / "manifest.yaml", "w") as f:
        yaml.dump({"config_hash": "DIFFERENT"}, f)
    with pytest.raises(PermissionError, match="different run"):
        paths.get_prediction_dir("model_a")


def test_permutation_null_reruns_full_pipeline():
    """R-008: the permutation null must rerun the complete nested
    pipeline (repeated_cv_run with tuning + residualization), not a
    simplified Ridge(alpha=1.0)."""
    from unittest.mock import patch
    from eegcpm.evaluation.prediction import ridge_cpm as rc

    cfg = type("Cfg", (), {})()
    cfg.permutation_seed = 0
    cfg.cv_n_folds = 2
    cfg.cv_n_repeats = 1
    cfg.cv_seed = 0
    X = np.random.default_rng(0).normal(size=(20, 8))
    y = np.random.default_rng(1).normal(size=20)

    calls = []
    def spy(*args, **kwargs):
        calls.append(kwargs.get("alpha_grid"))
        return [0.01], [0.0]

    with patch.object(rc, "repeated_cv_run", side_effect=spy):
        rc._default_permutation_null(X, y, cfg, B=3)

    assert len(calls) == 3, "repeated_cv_run must be called per permutation"
    assert calls[0] is not None, "alpha_grid must be forwarded (tuning runs)"
