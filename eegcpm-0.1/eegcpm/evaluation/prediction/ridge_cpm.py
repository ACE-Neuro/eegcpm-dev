"""
Ridge CPM — Connectome Predictive Modeling with ridge regression on
ALL edges (per spec §3.b + S05 + S06 + S08 + S10).

The PRIMARY model is ridge regression on all edges (no pre-selection).
BH-FDR edge selection is REPLACED by a labelled sensitivity arm only.

The 17-literal logspace alpha grid (`np.logspace(-2, 6, 17)`) is the
ONLY regularisation control. The contradictory scalar `alpha: 1.0`
is DELETED.

Nested 5-fold CV with inner alpha tune; pooled out-of-fold Pearson r
is the binding statistic. R=10 repeated CV runs; cv_seed pinned;
report_repeated_cv_distribution=True.

Adaptive permutation schedule (per S02): escalation_p_lo=0.005,
escalation_p_hi=0.10. The 10,000-perm path is REACHABLE.

covariate_adjustment_mode=residualize_features_train_only is PRIMARY.
The unadjusted model is a labelled sensitivity.

Two-fit fold-purity guards (per S07): the transform pipeline fits
on (X_train, X_test) and (X_train, X_test_perturbed); training-fold
outputs and fitted params are bitwise identical.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# The 17-element logspace alpha grid (pinned; see §3.s golden hash).
# Both the named rule AND the explicit literals are pinned; the
# validator (validate_primary_config) asserts len(alpha_grid)==17
# and np.allclose(alpha_grid, np.logspace(-2, 6, 17), atol=1e-12).
ALPHA_GRID: List[float] = list(np.logspace(-2, 6, 17))
ALPHA_GRID_RULE: str = "np.logspace(-2, 6, 17)"

# Default R for repeated CV (pinned)
CV_N_REPEATS: int = 10

# Default inner CV folds (pinned)
CV_INNER_FOLDS: int = 3

# CV seed (pinned to the project's lineage seed)
CV_SEED: int = 20260729

# Adaptive permutation window (pinned; S02)
ESCALATION_P_LO: float = 0.005
ESCALATION_P_HI: float = 0.10
N_PERMUTATIONS_SCREENING: int = 1000
N_PERMUTATIONS_CONFIRMATORY: int = 10000
MC_CI_LEVEL: float = 0.95


def fit_predict_outer_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    covariate_train: Optional[np.ndarray] = None,
    covariate_test: Optional[np.ndarray] = None,
    alpha_grid: List[float] = None,
    n_inner_folds: int = CV_INNER_FOLDS,
    inner_cv_seed: int = CV_SEED,
    random_state: int = 0,
) -> Tuple[np.ndarray, float, bool]:
    """Single outer fold: train on (X_train, y_train), predict on X_test.

    Returns (y_pred, best_alpha, boundary_hit).
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    alpha_grid = alpha_grid if alpha_grid is not None else ALPHA_GRID

    # 1. Covariate adjustment: residualize features on covariates
    # (fitted on training only; applied to actual train AND test rows).
    if covariate_train is not None and covariate_test is not None:
        X_train, X_test = _residualize_train_apply(
            X_train, X_test, covariate_train, covariate_test)

    # 2. Inner CV: tune alpha
    kf = KFold(n_splits=n_inner_folds, shuffle=True,
               random_state=inner_cv_seed)
    best_alpha, best_r = None, -np.inf
    for alpha in alpha_grid:
        r = _pooled_out_of_fold_r_for_alpha(
            X_train, y_train, alpha, kf)
        if r > best_r:
            best_r, best_alpha = r, alpha

    # 3. Boundary-hit detection
    boundary_hit = (best_alpha == alpha_grid[0]
                    or best_alpha == alpha_grid[-1])

    # 4. Fit on full training fold
    # solver='lsqr': at p >> n (88k-103k edges) cholesky on X'X needs
    # O(p^2) memory (~86 GB) and would OOM; lsqr is iterative and
    # memory-safe (0.6 s/fit at n=916, p=103,545).
    model = Ridge(alpha=best_alpha, fit_intercept=True,
                  solver="lsqr", random_state=random_state)
    model.fit(X_train, y_train)

    # 5. Predict on test
    y_pred = model.predict(X_test)
    return y_pred, best_alpha, boundary_hit


def _pooled_out_of_fold_r_for_alpha(
    X: np.ndarray, y: np.ndarray, alpha: float, kf,
) -> float:
    """Compute pooled out-of-fold r for a single alpha value."""
    from sklearn.linear_model import Ridge
    y_true_all, y_pred_all = [], []
    for tr, te in kf.split(X):
        m = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
        m.fit(X[tr], y[tr])
        y_true_all.append(y[te])
        y_pred_all.append(m.predict(X[te]))
    return _pooled_out_of_fold_r(y_true_all, y_pred_all)


def _pooled_out_of_fold_r(
    y_true_all: List[np.ndarray], y_pred_all: List[np.ndarray],
) -> float:
    """THE primary statistic. Pool all outer folds first, then correlate."""
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    if y_true.std() == 0 or y_pred.std() == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _residualize_train_apply(
    X_train: np.ndarray, X_test: np.ndarray,
    covariate_train: np.ndarray, covariate_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Residualize X on covariates, fit on train only, applied to the
    ACTUAL train and test covariate rows (R2-001: test rows must use
    their own covariates, never the training mean).
    """
    Z_train = np.hstack([np.ones((X_train.shape[0], 1)), covariate_train])
    Z_test = np.hstack([np.ones((X_test.shape[0], 1)), covariate_test])
    if Z_test.shape[1] != Z_train.shape[1]:
        raise ValueError(
            f"covariate_test has {Z_test.shape[1] - 1} columns but "
            f"covariate_train has {Z_train.shape[1] - 1}; schemas must match."
        )
    # Beta per feature column, fitted on training only (lstsq: stable
    # for collinear covariates, no explicit inverse)
    beta, *_ = np.linalg.lstsq(Z_train, X_train, rcond=None)
    X_train_resid = X_train - Z_train @ beta
    X_test_resid = X_test - Z_test @ beta
    return X_train_resid, X_test_resid


def collect_oof_predictions(
    X: np.ndarray, y: np.ndarray,
    subject_ids: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    n_folds: int = 5,
    n_repeats: int = CV_N_REPEATS,
    cv_seed: int = CV_SEED,
    alpha_grid: List[float] = None,
    inner_cv_seed: int = CV_SEED,
) -> "pd.DataFrame":
    """Collect per-subject out-of-fold predictions across repeats
    (DUA-gated subject-level artifact; same seeds as repeated_cv_run
    so the r computed from these predictions matches the run's r).
    """
    import pandas as pd
    from sklearn.model_selection import KFold
    alpha_grid = alpha_grid if alpha_grid is not None else ALPHA_GRID
    rng = np.random.default_rng(cv_seed)
    frames = []
    for run_i in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True,
                   random_state=int(rng.integers(0, 2**31)))
        for fold_i, (tr, te) in enumerate(kf.split(X)):
            cov_tr = covariates[tr] if covariates is not None else None
            cov_te = covariates[te] if covariates is not None else None
            yp, best_alpha, _ba = fit_predict_outer_fold(
                X[tr], y[tr], X[te],
                covariate_train=cov_tr, covariate_test=cov_te,
                alpha_grid=alpha_grid, inner_cv_seed=inner_cv_seed,
                random_state=run_i,
            )
            frames.append(pd.DataFrame({
                "subject_id": subject_ids[te],
                "repeat": run_i,
                "fold": fold_i,
                "y_true": y[te],
                "y_pred": yp,
                "alpha": best_alpha,
            }))
    return pd.concat(frames, ignore_index=True)


def repeated_cv_run(
    X: np.ndarray, y: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    n_folds: int = 5,
    n_repeats: int = CV_N_REPEATS,
    cv_seed: int = CV_SEED,
    alpha_grid: List[float] = None,
    inner_cv_seed: int = CV_SEED,
) -> Tuple[List[float], List[float]]:
    """Repeated CV: R=10 runs, each with 5-fold split.

    Returns (all_r, all_boundary_hit_fractions).
    """
    from sklearn.model_selection import KFold
    alpha_grid = alpha_grid if alpha_grid is not None else ALPHA_GRID
    rng = np.random.default_rng(cv_seed)
    all_r, all_boundary = [], []
    for run_i in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True,
                   random_state=int(rng.integers(0, 2**31)))
        y_true, y_pred = [], []
        run_boundary = []
        for tr, te in kf.split(X):
            cov_tr = covariates[tr] if covariates is not None else None
            cov_te = covariates[te] if covariates is not None else None
            yp, _ba, ba = fit_predict_outer_fold(
                X[tr], y[tr], X[te],
                covariate_train=cov_tr, covariate_test=cov_te,
                alpha_grid=alpha_grid, inner_cv_seed=inner_cv_seed,
                random_state=run_i,
            )
            y_true.append(y[te])
            y_pred.append(yp)
            run_boundary.append(ba)
        all_r.append(_pooled_out_of_fold_r(y_true, y_pred))
        all_boundary.append(np.mean(run_boundary))
    return all_r, all_boundary


def adaptive_permutation(
    X: np.ndarray, y: np.ndarray,
    cfg: Any, observed_r: float,
    permutation_fn=None,
    covariates: Optional[np.ndarray] = None,
) -> Tuple[float, Tuple[float, float]]:
    """Adaptive permutation (S02): escalation_p_lo, escalation_p_hi.

    The escalation branch is REACHABLE. The test
    `test_escalation_branch_reachable` feeds p_hat = .02 and asserts
    the 10,000-perm path executes.
    """
    if permutation_fn is None:
        permutation_fn = _default_permutation_null
    # Stage 1: screening
    null_r = permutation_fn(X, y, cfg, B=cfg.n_permutations_screening,
                            covariates=covariates)
    B_used = cfg.n_permutations_screening
    p_hat = (1 + (null_r >= observed_r).sum()) / (1 + B_used)
    # Stage 2: escalate if p_hat in the decision margin
    escalated = (
        cfg.n_permutations_confirmatory > cfg.n_permutations_screening
        and cfg.escalation_p_lo < p_hat < cfg.escalation_p_hi
    )
    if escalated:
        null_r = permutation_fn(
            X, y, cfg, B=cfg.n_permutations_confirmatory,
            covariates=covariates)
        B_used = cfg.n_permutations_confirmatory
        p_hat = (1 + (null_r >= observed_r).sum()) / (1 + B_used)
    # MC CI on the permutation p (R-008: B tracked at decision time,
    # not re-evaluated after the final p_hat)
    p_lo = max(0.0, p_hat - 1.96 * np.sqrt(p_hat * (1 - p_hat) / B_used))
    p_hi = min(1.0, p_hat + 1.96 * np.sqrt(p_hat * (1 - p_hat) / B_used))
    return p_hat, (p_lo, p_hi)


def _default_permutation_null(
    X: np.ndarray, y: np.ndarray, cfg: Any, B: int,
    covariates: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Permutation null: shuffle y and rerun the COMPLETE nested
    pipeline (covariate residualization, inner-loop alpha tuning over
    the pinned grid, R repeats, seed lineage) per permutation — the
    same pipeline that produced the observed statistic (R-008)."""
    rng = np.random.default_rng(cfg.permutation_seed)
    n_repeats = getattr(cfg, "cv_n_repeats", CV_N_REPEATS)
    alpha_grid = getattr(cfg, "alpha_grid", ALPHA_GRID)
    inner_cv_seed = getattr(cfg, "inner_cv_seed", CV_SEED)
    null_r = np.zeros(B)
    for i in range(B):
        y_perm = rng.permutation(y)
        rep_r, _ = repeated_cv_run(
            X, y_perm,
            covariates=covariates,
            n_folds=cfg.cv_n_folds,
            n_repeats=n_repeats,
            cv_seed=cfg.cv_seed,
            alpha_grid=alpha_grid,
            inner_cv_seed=inner_cv_seed,
        )
        # Same statistic reduction as the observed run: mean over repeats
        null_r[i] = float(np.mean(rep_r))
    return null_r
