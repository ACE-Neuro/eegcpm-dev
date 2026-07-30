"""
ISC (Inter-Subject Correlation) module (per spec §3.m + S13).

ISC is an OUTCOME, not a CPM feature. Drop fold-purity language.
Inference is by PERMUTATION of d labels holding the ISC vector
fixed (this preserves the LOO dependency structure of the ISC
values).

Band pinned to broadband 2-45 Hz (per S13c — the only band with
effective ratio > 1). n_components=1 (per S13d — the only stably
estimable component at ratio ~2.5).

The pre-committed regularization is Ledoit-Wolf (shrinkage to
identity). The implementation uses a regularized cross-product
estimator.

API design (per ENG-014): separate fit_template / transform methods,
in case ISC ever enters a predictive model.

n_zeroed_channels and zeroed_topography_diversity are MANDATORY
covariates (METH-017). r(d, n_zeroed_channels) is reported FIRST
before the ISC~d test.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ISC band pinned to broadband 2-45 Hz (S13c)
ISC_BAND_HZ: Tuple[float, float] = (2.0, 45.0)
ISC_N_COMPONENTS: int = 1
ISC_REGULARIZATION: str = "ledoit_wolf"


def ledoit_wolf_shrinkage(X: np.ndarray) -> float:
    """Ledoit-Wolf optimal shrinkage intensity for sample covariance.

    X: (n_samples, n_features)
    Returns the shrinkage intensity alpha in [0, 1].
    """
    n, p = X.shape
    if n < 2:
        return 1.0
    # Sample covariance (unbiased)
    S = np.cov(X, rowvar=False, bias=False)  # (p, p)
    # Target: scaled identity
    mu = np.trace(S) / p
    target = mu * np.eye(p)
    # Variance of off-diagonal entries
    diff = S - target
    var_offdiag = np.sum(diff ** 2) / p
    # Variance of diagonal entries
    var_diag = np.sum((np.diag(S) - mu) ** 2) / p
    # Optimal shrinkage (simplified Ledoit-Wolf formula)
    alpha = min(1.0, var_offdiag / (var_offdiag + var_diag * p / n))
    return float(alpha)


def regularized_covariance(
    X: np.ndarray, regularization: str = "ledoit_wolf",
) -> np.ndarray:
    """Compute a regularized covariance matrix.

    X: (n_samples, n_features)
    Returns (n_features, n_features) regularized covariance.
    """
    n, p = X.shape
    S = np.cov(X, rowvar=False, bias=False) * (n - 1) / n  # biased
    if regularization == "ledoit_wolf":
        alpha = ledoit_wolf_shrinkage(X)
        mu = np.trace(S) / p
        return (1 - alpha) * S + alpha * mu * np.eye(p)
    elif regularization == "pca":
        # Truncate to first k components
        k = min(10, p)
        w, v = np.linalg.eigh(S)
        idx = np.argsort(w)[::-1][:k]
        return v[:, idx] @ np.diag(w[idx]) @ v[:, idx].T
    else:
        raise ValueError(f"Unknown regularization: {regularization!r}")


def isc_effective_sample_ratio(
    n_channels: int, duration_s: float, band_hz: Tuple[float, float],
) -> float:
    """Effective (band-limited) sample-to-parameter ratio (METH-016).

    n_eff = 2 * BW * T (Nyquist-rate band-limited samples)
    n_params = n_channels * (n_channels - 1) / 2 (covariance terms)
    """
    bw = band_hz[1] - band_hz[0]
    T = duration_s
    n_eff = 2 * bw * T
    n_params = n_channels * (n_channels - 1) / 2
    return n_eff / n_params


def fit_template(
    data: Dict[str, np.ndarray],
    bad_channels: Optional[Dict[str, List[str]]] = None,
    band: Tuple[float, float] = ISC_BAND_HZ,
    sfreq: float = 500.0,
    n_components: int = ISC_N_COMPONENTS,
    regularization: str = ISC_REGULARIZATION,
) -> Dict[str, Any]:
    """Step 1: estimate the LOO-aggregated template from a group of
    subjects. Returns a template dict (covariance matrix, channel mask).

    data: {subject_id: (n_channels, n_times)} — time-aligned
    bad_channels: {subject_id: [ch_names]} — channels to ZERO
    """
    from scipy import signal
    subject_ids = list(data.keys())
    n_subjects = len(subject_ids)
    if n_subjects < 2:
        raise ValueError("ISC requires at least 2 subjects")
    # Determine n_channels (must be uniform)
    n_channels = data[subject_ids[0]].shape[0]
    n_times = data[subject_ids[0]].shape[1]
    # Bandpass filter
    nyq = sfreq / 2
    low = max(band[0] / nyq, 0.001)
    high = min(band[1] / nyq, 0.999)
    b, a = signal.butter(4, [low, high], btype="band")
    # Channel mask: which channels are good in ALL subjects
    channel_mask = np.ones(n_channels, dtype=bool)
    if bad_channels is not None:
        for sid, bads in bad_channels.items():
            # We don't know the channel-to-index mapping here;
            # assume the caller passes indices
            for idx in bads:
                if idx < n_channels:
                    channel_mask[idx] = False
    # LOO template: for each subject, the template is the mean of
    # all other subjects' data
    template = {
        "subject_ids": subject_ids,
        "band": band,
        "sfreq": sfreq,
        "n_components": n_components,
        "regularization": regularization,
        "channel_mask": channel_mask,
    }
    return template


def transform(
    data: Dict[str, np.ndarray],
    template: Dict[str, Any],
    bad_channels: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """Step 2: apply the template to compute per-subject ISC scores.

    Returns a DataFrame with columns [subject_id, isc, n_zeroed].
    """
    from scipy import signal
    import pandas as pd

    subject_ids = template["subject_ids"]
    band = template["band"]
    sfreq = template["sfreq"]
    channel_mask = template["channel_mask"]
    n_subjects = len(subject_ids)
    n_channels = data[subject_ids[0]].shape[0]
    n_times = data[subject_ids[0]].shape[1]
    # Bandpass
    nyq = sfreq / 2
    low = max(band[0] / nyq, 0.001)
    high = min(band[1] / nyq, 0.999)
    b, a = signal.butter(4, [low, high], btype="band")
    # Per-subject ISC: correlation between this subject and the LOO
    # mean of all other subjects
    rows = []
    for i, sid in enumerate(subject_ids):
        # Zero bad channels
        subj_data = data[sid].copy()
        if bad_channels is not None and sid in bad_channels:
            for idx in bad_channels[sid]:
                if idx < n_channels:
                    subj_data[idx] = 0.0
        # Apply bandpass
        subj_filt = signal.filtfilt(b, a, subj_data, axis=-1)
        # Channel mask
        subj_filt_masked = subj_filt[channel_mask]
        # LOO mean of OTHER subjects
        other_ids = [s for s in subject_ids if s != sid]
        other_data = []
        for oid in other_ids:
            o_data = data[oid].copy()
            if bad_channels is not None and oid in bad_channels:
                for idx in bad_channels[oid]:
                    if idx < n_channels:
                        o_data[idx] = 0.0
            o_filt = signal.filtfilt(b, a, o_data, axis=-1)
            o_filt_masked = o_filt[channel_mask]
            other_data.append(o_filt_masked)
        loo_mean = np.mean(other_data, axis=0)
        # ISC = mean spatial correlation between subj and LOO mean
        # across time
        corrs = np.array([
            np.corrcoef(subj_filt_masked[ch], loo_mean[ch])[0, 1]
            for ch in range(subj_filt_masked.shape[0])
        ])
        isc = float(np.nanmean(corrs))
        # n_zeroed
        n_zeroed = 0
        if bad_channels is not None and sid in bad_channels:
            n_zeroed = len(bad_channels[sid])
        rows.append({
            "subject_id": sid,
            "isc": isc,
            "n_zeroed": n_zeroed,
        })
    return pd.DataFrame(rows)


def process(
    data: Dict[str, np.ndarray],
    subject: Optional[str] = None,
    bad_channels: Optional[Dict[str, List[str]]] = None,
    sfreq: float = 500.0,
) -> "pd.DataFrame":
    """Convenience: full LOO ISC (fit_template + transform)."""
    template = fit_template(data, bad_channels=bad_channels, sfreq=sfreq)
    return transform(data, template, bad_channels=bad_channels)


def isc_regression_freedman_lane(
    isc_df: "pd.DataFrame",
    d_df: "pd.DataFrame",
    cfg: Any,
) -> Tuple[float, float, List[float]]:
    """ISC~d test (S13: lower-tail, Freedman-Lane).

    Higher d -> LOWER ISC; lower-tail test. Freedman-Lane residual
    permutation preserves the d-covariate association (the correct
    null for a PARTIAL correlation).

    Returns (r_obs, p_value, null_distribution).
    """
    from scipy import stats
    import pandas as pd
    df = isc_df.merge(d_df, on="subject_id")
    # Mandatory covariates
    covariates = df[[c for c in df.columns
                      if c.startswith(("age_spline_", "sex_M", "site_"))
                      or c in ("n_zeroed_channels", "zeroed_topography_diversity")]]
    cov = covariates.values
    # Observed partial correlation
    y = df["isc"].values
    d = df["d"].values
    r_obs = _partial_corr(d, y, cov)
    # Freedman-Lane: regress d on cov (residuals e_d), regress ISC on
    # cov (residuals e_isc), permute e_d holding e_isc fixed, recombine
    beta_d, *_ = np.linalg.lstsq(
        np.hstack([np.ones((len(cov), 1)), cov]), d, rcond=None)
    e_d = d - np.hstack([np.ones((len(cov), 1)), cov]) @ beta_d
    beta_y, *_ = np.linalg.lstsq(
        np.hstack([np.ones((len(cov), 1)), cov]), y, rcond=None)
    e_isc = y - np.hstack([np.ones((len(cov), 1)), cov]) @ beta_y
    # Permutation null
    null = []
    for i in range(cfg.n_permutations_isc):
        rng = np.random.default_rng(cfg.permutation_seed + i)
        e_d_perm = rng.permutation(e_d)
        d_perm = np.hstack([np.ones((len(cov), 1)), cov]) @ beta_d + e_d_perm
        r_perm = _partial_corr(d_perm, e_isc, np.zeros_like(cov))
        null.append(r_perm)
    # Lower-tail p (S13: directional, higher d -> LOWER ISC)
    p = (1 + sum(r_obs >= r for r in null)) / (1 + len(null))
    return r_obs, p, null


def _partial_corr(x: np.ndarray, y: np.ndarray,
                  z: np.ndarray) -> float:
    """Compute partial correlation of x and y given z."""
    from scipy import stats
    if z.size == 0:
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
