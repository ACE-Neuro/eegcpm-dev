"""
ISC regression: Freedman-Lane residual permutation (per spec §3.m + S13b).

Higher d -> LOWER ISC; lower-tail test. Freedman-Lane residual
permutation preserves the d-covariate association (the correct
null for a PARTIAL correlation).
"""

from typing import Any, List, Tuple

import numpy as np


def _partial_corr(x: np.ndarray, y: np.ndarray,
                  z: np.ndarray) -> float:
    """Compute partial correlation of x and y given z."""
    if z.size == 0:
        if x.std() == 0 or y.std() == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])
    # Regress x and y on z, correlate residuals
    Z = np.hstack([np.ones((len(z), 1)), z])
    beta_x, *_ = np.linalg.lstsq(Z, x, rcond=None)
    beta_y, *_ = np.linalg.lstsq(Z, y, rcond=None)
    e_x = x - Z @ beta_x
    e_y = y - Z @ beta_y
    if e_x.std() == 0 or e_y.std() == 0:
        return 0.0
    return float(np.corrcoef(e_x, e_y)[0, 1])


def isc_regression_freedman_lane(
    isc_df,
    d_df,
    cfg: Any,
) -> Tuple[float, float, List[float]]:
    """ISC~d test (S13: lower-tail, Freedman-Lane).

    Returns (r_obs, p_value, null_distribution).
    """
    df = isc_df.merge(d_df, on="subject_id")
    # Mandatory covariates (R-011): n_zeroed_channels AND
    # zeroed_topography_diversity MUST be present in the merged
    # DataFrame.
    mandatory = ("n_zeroed_channels", "zeroed_topography_diversity")
    for m in mandatory:
        if m not in df.columns:
            raise ValueError(
                f"isc_regression: mandatory covariate {m!r} missing "
                f"from the merged frame. The ISC arm schema MUST "
                f"include both n_zeroed_channels and "
                f"zeroed_topography_diversity (R-011)."
            )
    covariate_prefixes = ("age_spline_", "sex_M", "site_")
    covariates = df[[c for c in df.columns
                      if c.startswith(covariate_prefixes)
                      or c in mandatory]]
    cov = covariates.values
    y = df["isc"].values
    d = df["d"].values
    r_obs = _partial_corr(d, y, cov)
    # Freedman-Lane: regress d on cov (residuals e_d), regress ISC on
    # cov (residuals e_isc), permute e_d holding e_isc fixed, recombine
    Z = np.hstack([np.ones((len(cov), 1)), cov])
    beta_d, *_ = np.linalg.lstsq(Z, d, rcond=None)
    e_d = d - Z @ beta_d
    beta_y, *_ = np.linalg.lstsq(Z, y, rcond=None)
    e_isc = y - Z @ beta_y
    # Permutation null
    null = []
    for i in range(cfg.n_permutations_isc):
        rng = np.random.default_rng(cfg.permutation_seed + i)
        e_d_perm = rng.permutation(e_d)
        d_perm = Z @ beta_d + e_d_perm
        r_perm = _partial_corr(d_perm, e_isc, np.zeros_like(cov))
        null.append(r_perm)
    # Lower-tail p (S13: directional, higher d -> LOWER ISC)
    p = (1 + sum(r_obs >= r for r in null)) / (1 + len(null))
    return r_obs, p, null
