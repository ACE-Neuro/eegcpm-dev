"""
ISC via one-component regularized Correlated Component Analysis
(per spec §3.m + R-011 + ENG-014).

ISC is an OUTCOME, not a CPM feature. One component only (the only
stably-estimable one at broadband effective ratio ~2.5; per S13d).

CorrCA (Dmochowski et al. 2012; Parra lab corrca):
    maximize  w' R_b w   subject to  w' R_w w = 1
where R_w = sum_i X_i X_i' (within-subject covariance sum) and
R_b = sum_{i != j} X_i X_j' (between-subject covariance sum), solved
as the generalized eigenproblem R_b w = lambda R_w w. R_b is computed
in O(N) via the identity  sum_{i!=j} X_i X_j' = S S' - R_w  with
S = sum_i X_i. R_w is Ledoit-Wolf shrunk toward scaled identity.

API (ENG-014): separate fit_template / transform, in case ISC ever
enters a predictive model. Bad channels are ZEROED (not interpolated)
per Langer et al. 2017, and the mandatory artifact covariates
n_zeroed_channels + zeroed_topography_diversity are emitted (METH-017).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


ISC_N_COMPONENTS: int = 1
ISC_BAND_HZ: Tuple[float, float] = (2.0, 45.0)
ISC_REGULARIZATION: str = "ledoit_wolf"


@dataclass
class ISCTemplate:
    """Result of fit_template: component weights + template time course."""
    weights: np.ndarray              # (n_channels, n_components)
    template_timecourse: np.ndarray  # (n_components, n_times)
    band: Tuple[float, float]
    sfreq: float
    n_components: int
    n_channels: int
    regularization: str
    shrinkage: float


def isc_effective_sample_ratio(
    n_channels: int, duration_s: float, band_hz: Tuple[float, float],
) -> float:
    """Effective (band-limited) sample-to-parameter ratio (METH-016)."""
    bw = band_hz[1] - band_hz[0]
    n_eff = 2 * bw * duration_s
    n_params = n_channels * (n_channels - 1) / 2
    return n_eff / n_params


def ledoit_wolf_shrinkage(X: np.ndarray) -> float:
    """Ledoit-Wolf-style shrinkage intensity in [0, 1].

    X: (n_samples, n_features). Scalar intensity toward scaled identity.
    """
    n, p = X.shape
    if n < 2:
        return 1.0
    S = np.cov(X, rowvar=False, bias=False)
    mu = np.trace(S) / p
    diff = S - mu * np.eye(p)
    var_offdiag = np.sum(diff ** 2) / p
    var_diag = np.sum((np.diag(S) - mu) ** 2) / p
    denom = var_offdiag + var_diag * p / n
    if denom <= 0:
        return 0.0
    return float(min(1.0, var_offdiag / denom))


def regularized_covariance(X: np.ndarray,
                            regularization: str = "ledoit_wolf"
                            ) -> np.ndarray:
    """LW-shrunk covariance of X (n_samples, n_features)."""
    n, p = X.shape
    S = np.cov(X, rowvar=False, bias=False) * (n - 1) / n
    if regularization == "ledoit_wolf":
        alpha = ledoit_wolf_shrinkage(X)
        mu = np.trace(S) / p
        return (1 - alpha) * S + alpha * mu * np.eye(p)
    raise ValueError(f"Unknown regularization: {regularization!r}")


def _bandpass(X: np.ndarray, band: Tuple[float, float], sfreq: float,
              b_a_cache: dict) -> np.ndarray:
    from scipy import signal
    key = (band, sfreq)
    if key not in b_a_cache:
        nyq = sfreq / 2
        low = max(band[0] / nyq, 0.001)
        high = min(band[1] / nyq, 0.999)
        b, a = signal.butter(4, [low, high], btype="band")
        b_a_cache[key] = (b, a)
    b, a = b_a_cache[key]
    return signal.filtfilt(b, a, X, axis=-1)


def _prepare(data, bad_channels, band, sfreq):
    """Zero bad channels, bandpass, center per channel. Returns
    (subject_ids, {sid: (n_channels, n_times)})."""
    subject_ids = list(data.keys())
    cache: dict = {}
    out = {}
    for sid in subject_ids:
        X = np.asarray(data[sid], dtype=np.float64).copy()
        if bad_channels is not None and sid in bad_channels:
            for idx in bad_channels[sid]:
                if idx < X.shape[0]:
                    X[idx] = 0.0
        X = _bandpass(X, band, sfreq, cache)
        X -= X.mean(axis=1, keepdims=True)
        out[sid] = X
    return subject_ids, out


def fit_template(
    data: Dict[str, np.ndarray],
    bad_channels: Optional[Dict[str, List[int]]] = None,
    band: Tuple[float, float] = ISC_BAND_HZ,
    sfreq: float = 500.0,
    n_components: int = ISC_N_COMPONENTS,
    regularization: str = "ledoit_wolf",
) -> ISCTemplate:
    """Fit the one-component regularized CorrCA template.

    data: {subject_id: (n_channels, n_times)}, time-aligned across
    subjects. bad_channels: {subject_id: [indices]} to zero.
    """
    from scipy.linalg import eigh as gen_eigh

    if len(data) < 2:
        raise ValueError("ISC requires at least 2 subjects")
    subject_ids, Xs = _prepare(data, bad_channels, band, sfreq)
    n_channels = Xs[subject_ids[0]].shape[0]

    # Within-subject covariance sum R_w and signal sum S (for R_b in O(N))
    R_w = np.zeros((n_channels, n_channels))
    S = np.zeros((n_channels, Xs[subject_ids[0]].shape[1]))
    for sid in subject_ids:
        X = Xs[sid]
        R_w += X @ X.T
        S += X
    # Between-subject covariance sum via the sum identity
    R_b = S @ S.T - R_w

    # LW shrinkage on R_w (pooled-sample intensity)
    pooled = np.hstack([Xs[sid] for sid in subject_ids])  # ch x total_times
    alpha = ledoit_wolf_shrinkage(pooled.T)
    p = n_channels
    mu = np.trace(R_w) / p
    R_w_reg = (1 - alpha) * R_w + alpha * mu * np.eye(p)

    # Generalized eigenproblem R_b w = lambda R_w_reg w
    eigvals, eigvecs = gen_eigh(R_b, R_w_reg)
    idx = np.argsort(eigvals)[::-1][:n_components]
    weights = eigvecs[:, idx]
    # Sign-align: largest-|weight| element positive
    for k in range(n_components):
        if weights[np.argmax(np.abs(weights[:, k])), k] < 0:
            weights[:, k] = -weights[:, k]

    # Template time course: mean projected component across subjects
    projected = np.stack([weights.T @ Xs[sid] for sid in subject_ids])
    template_tc = projected.mean(axis=0)  # (n_components, n_times)

    return ISCTemplate(
        weights=weights,
        template_timecourse=template_tc,
        band=band,
        sfreq=sfreq,
        n_components=n_components,
        n_channels=n_channels,
        regularization=regularization,
        shrinkage=float(alpha),
    )


def transform(
    data: Dict[str, np.ndarray],
    template: ISCTemplate,
    bad_channels: Optional[Dict[str, List[int]]] = None,
) -> "pd.DataFrame":
    """Per-subject ISC: correlation of the projected component time
    course with the template time course. Emits the mandatory artifact
    covariates n_zeroed_channels + zeroed_topography_diversity."""
    import pandas as pd

    subject_ids, Xs = _prepare(data, bad_channels, template.band,
                               template.sfreq)
    rows = []
    for sid in subject_ids:
        proj = template.weights.T @ Xs[sid]  # (n_components, n_times)
        iscs = []
        for k in range(template.n_components):
            r = np.corrcoef(proj[k], template.template_timecourse[k])[0, 1]
            iscs.append(r)
        bads = bad_channels.get(sid, []) if bad_channels else []
        rows.append({
            "subject_id": sid,
            "isc": float(np.mean(iscs)),
            "n_zeroed_channels": len(bads),
            "zeroed_topography_diversity": _topography_diversity(bads),
        })
    return pd.DataFrame(rows)


def _topography_diversity(bad_indices: List[int]) -> float:
    """0-1 diversity of bad-channel indices (scattered vs clustered)."""
    if len(bad_indices) < 2:
        return 0.0
    arr = np.asarray(bad_indices, dtype=float)
    return float(np.clip(arr.std() / 50.0, 0.0, 1.0))


def loo_transform(
    data: Dict[str, np.ndarray],
    bad_channels: Optional[Dict[str, List[int]]] = None,
    band: Tuple[float, float] = ISC_BAND_HZ,
    sfreq: float = 500.0,
) -> "pd.DataFrame":
    """Leave-one-out ISC: each subject's template is fitted on the
    OTHER subjects only, then that subject is projected. This is the
    primary ISC quantity (in-sample fitting inflates ISC, especially
    at small N)."""
    import pandas as pd

    subject_ids = list(data.keys())
    frames = []
    for sid in subject_ids:
        others = {s: data[s] for s in subject_ids if s != sid}
        tmpl = fit_template(others, bad_channels=bad_channels,
                            band=band, sfreq=sfreq)
        one = transform({sid: data[sid]}, tmpl, bad_channels=bad_channels)
        frames.append(one)
    return pd.concat(frames, ignore_index=True)


def process(
    data: Dict[str, np.ndarray],
    subject: Optional[str] = None,
    bad_channels: Optional[Dict[str, List[int]]] = None,
    sfreq: float = 500.0,
    loo: bool = True,
) -> "pd.DataFrame":
    """Convenience: full ISC. loo=True (default) uses leave-one-out
    templates (the primary quantity); loo=False fits one template on
    all subjects (in-sample — for diagnostics only)."""
    if loo:
        return loo_transform(data, bad_channels=bad_channels, sfreq=sfreq)
    template = fit_template(data, bad_channels=bad_channels, sfreq=sfreq)
    return transform(data, template, bad_channels=bad_channels)
