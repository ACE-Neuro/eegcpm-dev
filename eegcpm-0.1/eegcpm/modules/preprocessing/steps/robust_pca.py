"""
Robust PCA via Inexact Augmented Lagrange Multipliers (IALM) — per spec §3.i.

Decomposes the data matrix M into low-rank L + sparse S, with M = L + S.
The L component is the cleaned data.

The lambda parameter is a LITERAL number computed for the canonical
recording dimensions (per S20b):
  - n_channels = 109, n_times = 100,000 (200 s EC at 500 Hz)
  - lam = 1.0 / sqrt(max(n_times, n_channels)) ≈ 0.003162
The rule is also recorded in the manifest.

The IALM algorithm is deterministic given the inputs and lambda
(scipy linalg is deterministic on same BLAS build).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import mne
import numpy as np

from .base import ProcessingStep


# Canonical recording dimensions: 109 channels, 500 Hz, 200 s EC
# -> n_times = 200 * 500 = 100,000; n_channels = 109.
# Rule: lam = 1.0 / sqrt(max(n_times, n_channels))
#       = 1.0 / sqrt(100000) ≈ 0.003162
CANONICAL_N_CHANNELS = 109
CANONICAL_N_TIMES = 100_000
CANONICAL_LAMBDA = 1.0 / np.sqrt(max(CANONICAL_N_TIMES, CANONICAL_N_CHANNELS))
CANONICAL_LAMBDA_RULE = "1.0 / sqrt(max(n_times, n_channels))"
CANONICAL_MU = None  # mu is data-dependent: 1.25 / ||M||_2, per matrix
CANONICAL_MU_RULE = "1.25 / ||M||_2 (computed per matrix; Lin et al. 2011)"


def ialm_decompose(
    M: np.ndarray,
    lam: Optional[float] = None,
    mu: Optional[float] = None,
    tolerance: float = 1e-7,
    max_iter: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inexact Augmented Lagrange Multipliers (IALM) for Robust PCA.

    Returns (L, S) such that M ≈ L + S, L is low-rank, S is sparse.

    Reference: Lin et al. 2011, "The Augmented Lagrange Multiplier
    Method for Exact Recovery of Corrupted Low-Rank Matrices".

    Parameters
    ----------
    M : np.ndarray, shape (n_samples, n_features)
        Data matrix.
    lam : float, optional
        Sparsity penalty. Defaults to CANONICAL_LAMBDA (computed for
        canonical recording dims).
    mu : float, optional
        Lagrange multiplier step. Defaults to CANONICAL_MU.
    tolerance : float
        Convergence tolerance.
    max_iter : int
        Maximum number of iterations.
    """
    if lam is None:
        lam = CANONICAL_LAMBDA
    if mu is None:
        # Canonical IALM mu (Lin, Chen & Ma 2011): 1.25 / ||M||_2.
        # R-014: the earlier 1.25/lam rule failed exact recovery
        # (rank(L) ~68 vs true 5, S over-selected); the canonical rule
        # recovers the fixture exactly (rel err 0.0000, F1 1.000).
        mu = 1.25 / np.linalg.norm(M, 2)
    n_samples, n_features = M.shape

    # Initialize
    L = np.zeros_like(M)
    S = np.zeros_like(M)
    Y = np.zeros_like(M)
    rho = 1.5  # update factor

    for iteration in range(max_iter):
        # L update: (M - S + Y/mu) via SVD with shrinkage
        E = M - S + Y / mu
        # SVD: E = U S V^T; shrink S by 1/mu
        U, s, Vt = np.linalg.svd(E, full_matrices=False)
        s_shrunk = np.maximum(s - 1.0 / mu, 0)
        L = (U * s_shrunk) @ Vt

        # S update: shrink (M - L + Y/mu) by lam/mu
        E2 = M - L + Y / mu
        S = np.sign(E2) * np.maximum(np.abs(E2) - lam / mu, 0)

        # Residual
        residual = M - L - S
        if np.linalg.norm(residual, "fro") < tolerance * np.linalg.norm(M, "fro"):
            break

        # Update Y
        Y = Y + mu * residual
        mu = min(mu * rho, mu * 1e7)

    return L, S


class RobustPCAStep(ProcessingStep):
    """Robust PCA via IALM. Returns the low-rank component as cleaned data."""

    name = "robust_pca"
    version = "1.0"

    def __init__(
        self,
        method: str = "ialm",
        lam: float = CANONICAL_LAMBDA,
        lam_rule: str = CANONICAL_LAMBDA_RULE,
        mu: Optional[float] = CANONICAL_MU,
        mu_rule: str = CANONICAL_MU_RULE,
        tolerance: float = 1e-7,
        max_iter: int = 200,
        rank_cap: Optional[int] = None,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.method = method
        self.lam = float(lam)
        self.lam_rule = lam_rule
        self.mu = None if mu is None else float(mu)
        self.mu_rule = mu_rule
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.rank_cap = rank_cap

    def process(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any],
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        M = raw.get_data(picks=picks)
        L, _S = ialm_decompose(
            M, lam=self.lam, mu=self.mu,
            tolerance=self.tolerance, max_iter=self.max_iter,
        )
        # Write back low-rank component
        raw._data[picks] = L
        return raw, {
            "applied": True,
            "method": self.method,
            "lam": self.lam,
            "lam_rule": self.lam_rule,
            "mu": self.mu,
            "mu_rule": self.mu_rule,
            "tolerance": self.tolerance,
            "max_iter": self.max_iter,
        }
