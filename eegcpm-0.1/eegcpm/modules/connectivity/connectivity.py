"""
Connectivity module (per spec §3.k).

Implements the primary set {wPLI, dwPLI, AEC_orth} and the replication
set {coherence, PLV} for sensor-space connectivity. 109 scalp
channels, 5 canonical bands (delta 2-4, theta 4-8, alpha 8-13,
beta 13-30, gamma 30-45), 3 methods x 5 bands = 15 matrices
(5,886 edges per matrix, 88,290 total).

S13 (band adjustment): delta pinned to 2-4 Hz (NOT 1-4 Hz) for
consistency with the specparam fit floor.

The module exposes:
  - compute_wpli: weight phase lag index
  - compute_dwpli: debiased wPLI
  - compute_aec_orth: amplitude envelope correlation with symmetric
    orthogonalization
  - compute_coherence: magnitude-squared coherence
  - compute_plv: phase locking value
  - ConnectivityModule: orchestration
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal


# 5 canonical bands (per S13 / METH-028)
FREQUENCY_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (2.0, 4.0),    # 2-4 Hz (NOT 1-4; consistent with specparam floor)
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def bandpass(
    data: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass."""
    nyq = sfreq / 2
    low = band[0] / nyq
    high = band[1] / nyq
    if low <= 0:
        low = 0.001
    if high >= 1:
        high = 0.999
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, data, axis=-1)


def _analytic_signal(data: np.ndarray) -> np.ndarray:
    """Compute the analytic signal via Hilbert transform."""
    return signal.hilbert(data, axis=-1)


def compute_cross_spectrum(
    data: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
    nperseg: Optional[int] = None,
) -> np.ndarray:
    """Compute the complex cross-spectrum for all channel pairs.

    data: (n_channels, n_times)
    Returns: (n_channels, n_channels, n_freqs) complex cross-spectrum
    (band-limited).
    """
    if nperseg is None:
        nperseg = min(data.shape[-1], int(sfreq))
    # Bandpass first
    filtered = bandpass(data, sfreq, band)
    # Welch cross-spectrum
    f, Pxy = signal.csd(
        filtered[0], filtered[0], fs=sfreq, nperseg=nperseg,
        return_onesided=True,
    )
    n_channels = data.shape[0]
    n_freqs = len(f)
    S = np.zeros((n_channels, n_channels, n_freqs), dtype=complex)
    S[0, 0] = Pxy
    for i in range(1, n_channels):
        for j in range(i + 1, n_channels):
            f_ij, Pxy_ij = signal.csd(
                filtered[i], filtered[j], fs=sfreq, nperseg=nperseg,
                return_onesided=True,
            )
            S[i, j] = Pxy_ij
            S[j, i] = np.conj(Pxy_ij)
    # Band-pass the cross-spectrum (only keep freqs in the band)
    band_mask = (f >= band[0]) & (f <= band[1])
    return S[:, :, band_mask], f[band_mask]


def compute_wpli(
    data: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
) -> np.ndarray:
    """Weighted Phase Lag Index (Vinck et al. 2011).

    wPLI = |E[sign(Im(S_xy))]| / E[|Im(S_xy)|]
    where the expectation is over time/frequency bins.
    """
    S, f = compute_cross_spectrum(data, sfreq, band)
    n_channels = S.shape[0]
    M = np.zeros((n_channels, n_channels), dtype=float)
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            imag = np.imag(S[i, j])
            num = np.abs(np.mean(np.sign(imag)))
            denom = np.mean(np.abs(imag))
            if denom > 0:
                M[i, j] = num / denom
                M[j, i] = M[i, j]
    return M


def compute_dwpli(
    data: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
) -> np.ndarray:
    """Debiased wPLI (Vinck et al. 2011).

    dwPLI = (sum |Im(S)| * sign(Im(S))) / sum |Im(S)|
    """
    S, f = compute_cross_spectrum(data, sfreq, band)
    n_channels = S.shape[0]
    M = np.zeros((n_channels, n_channels), dtype=float)
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            imag = np.imag(S[i, j])
            denom = np.sum(np.abs(imag))
            if denom > 0:
                M[i, j] = np.sum(np.abs(imag) * np.sign(imag)) / denom
                M[j, i] = M[i, j]
    return M


def compute_aec_orth(
    data: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
) -> np.ndarray:
    """Amplitude Envelope Correlation with symmetric orthogonalization
    (Hipp et al. 2012)."""
    filtered = bandpass(data, sfreq, band)
    n_channels = filtered.shape[0]
    # Analytic signal
    analytic = _analytic_signal(filtered)
    # Amplitude envelopes
    envelopes = np.abs(analytic)
    # Symmetric orthogonalization: subtract the projection of each
    # envelope onto all others
    orthogonal = envelopes.copy()
    for i in range(n_channels):
        # Project onto all other envelopes
        others = np.delete(envelopes, i, axis=0)
        # Solve least-squares: others.T @ others @ beta = others.T @ env[i]
        others_flat = others.reshape(n_channels - 1, -1)
        env_i_flat = envelopes[i]
        beta, _, _, _ = np.linalg.lstsq(
            others_flat.T, env_i_flat, rcond=None)
        projected = (beta @ others_flat).reshape(envelopes.shape[1])
        orthogonal[i] = envelopes[i] - projected
    # Pearson correlation of orthogonalized envelopes
    M = np.corrcoef(orthogonal)
    # Diagonal = 1.0; off-diagonal = correlation
    return np.nan_to_num(M, nan=0.0)


def compute_coherence(
    data: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
) -> np.ndarray:
    """Magnitude-squared coherence."""
    S, f = compute_cross_spectrum(data, sfreq, band)
    n_channels = S.shape[0]
    # Coherence: |S_xy|^2 / (S_xx * S_yy)
    M = np.zeros((n_channels, n_channels), dtype=float)
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            num = np.mean(np.abs(S[i, j]) ** 2)
            denom = np.sqrt(
                np.mean(np.abs(S[i, i]) ** 2) * np.mean(np.abs(S[j, j]) ** 2))
            if denom > 0:
                M[i, j] = num / denom
                M[j, i] = M[i, j]
    return M


def compute_plv(
    data: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
) -> np.ndarray:
    """Phase Locking Value (Lachaux et al. 1999).

    PLV = |E[exp(i * phase_diff)]|
    """
    filtered = bandpass(data, sfreq, band)
    analytic = _analytic_signal(filtered)
    phases = np.angle(analytic)
    n_channels = phases.shape[0]
    M = np.zeros((n_channels, n_channels), dtype=float)
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase_diff = phases[i] - phases[j]
            M[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))
            M[j, i] = M[i, j]
    return M


# Primary metrics
PRIMARY_METHODS = ("wpli", "dwpli", "aec_orth")
REPLICATION_METHODS = ("coherence", "plv")
ALL_METHODS = PRIMARY_METHODS + REPLICATION_METHODS


def compute_connectivity(
    data: np.ndarray,
    sfreq: float,
    methods: Tuple[str, ...] = PRIMARY_METHODS,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute connectivity for all (method, band) combinations.

    Returns: {method: {band: (n_channels, n_channels) matrix}}
    """
    bands = bands if bands is not None else FREQUENCY_BANDS
    fn_map = {
        "wpli": compute_wpli,
        "dwpli": compute_dwpli,
        "aec_orth": compute_aec_orth,
        "coherence": compute_coherence,
        "plv": compute_plv,
    }
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for method in methods:
        if method not in fn_map:
            raise ValueError(
                f"Unknown method: {method!r}; must be one of {fn_map.keys()}")
        out[method] = {}
        for band_name, band_range in bands.items():
            out[method][band_name] = fn_map[method](data, sfreq, band_range)
    return out


def upper_triangle_indices(n_channels: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (i, j) indices for the upper triangle (i < j)."""
    i, j = np.triu_indices(n_channels, k=1)
    return i, j


def matrix_to_edges(M: np.ndarray) -> np.ndarray:
    """Convert a (n_channels, n_channels) matrix to an edge vector
    (upper triangle, n_channels*(n_channels-1)/2)."""
    i, j = upper_triangle_indices(M.shape[0])
    return M[i, j]


def edges_to_matrix(edges: np.ndarray, n_channels: int) -> np.ndarray:
    """Convert an edge vector back to a (n_channels, n_channels)
    symmetric matrix."""
    M = np.zeros((n_channels, n_channels), dtype=float)
    i, j = upper_triangle_indices(n_channels)
    M[i, j] = edges
    M[j, i] = edges
    return M


class ConnectivityModule:
    """Sensor-space connectivity orchestration (109 ch, 5 bands, 5 methods)."""

    def __init__(self, n_channels: int = 109, sfreq: float = 500.0,
                 methods: Tuple[str, ...] = PRIMARY_METHODS,
                 bands: Optional[Dict[str, Tuple[float, float]]] = None):
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.methods = methods
        self.bands = bands if bands is not None else FREQUENCY_BANDS

    def compute(self, data: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        return compute_connectivity(
            data, self.sfreq, methods=self.methods, bands=self.bands)

    def edges(self, data: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute and convert to edge vectors (long format)."""
        matrices = self.compute(data)
        out = {}
        for method, band_dict in matrices.items():
            out[method] = {}
            for band, M in band_dict.items():
                out[method][band] = matrix_to_edges(M)
        return out
