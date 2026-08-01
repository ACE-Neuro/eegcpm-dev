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
    n_channels = data.shape[0]
    # Welch cross-spectrum: freq grid from the first channel
    f, _ = signal.csd(
        filtered[0], filtered[0], fs=sfreq, nperseg=nperseg,
        return_onesided=True,
    )
    n_freqs = len(f)
    S = np.zeros((n_channels, n_channels, n_freqs), dtype=complex)
    # ALL pairs including diagonal (auto-spectra) and channel-0 pairs
    # (ENG-EEG3R2-005: the loop previously started at i=1, leaving
    # channel-0 pairs zero and misplacing the auto-spectrum).
    for i in range(n_channels):
        for j in range(i, n_channels):
            _, Pxy_ij = signal.csd(
                filtered[i], filtered[j], fs=sfreq, nperseg=nperseg,
                return_onesided=True,
            )
            S[i, j] = Pxy_ij
            S[j, i] = np.conj(Pxy_ij) if i != j else Pxy_ij
    # Band-pass the cross-spectrum (only keep freqs in the band)
    band_mask = (f >= band[0]) & (f <= band[1])
    return S[:, :, band_mask], f[band_mask]


def compute_cross_spectrum_segments(
    data: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
    nperseg: Optional[int] = None,
    overlap: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-segment complex cross-spectra for all channel pairs.

    The phase-lag estimators (wPLI/dwPLI) need the PER-OBSERVATION
    cross-spectra: Welch-averaging first and treating frequency bins as
    observations leaves too few observations for the debiasing to work
    (inflated values on independent channels — ENG-EEG3R-009).

    data: (n_channels, n_times)
    Returns: S of shape (n_segments, n_channels, n_channels, n_freqs)
             and the band-limited frequency vector.
    """
    if nperseg is None:
        nperseg = min(data.shape[-1], int(sfreq))
    filtered = bandpass(data, sfreq, band)
    n_ch, n_times = filtered.shape
    step = max(1, int(nperseg * (1 - overlap)))
    starts = list(range(0, n_times - nperseg + 1, step))
    if not starts:
        raise ValueError(
            f"recording too short ({n_times} samples) for nperseg={nperseg}"
        )
    win = signal.windows.hann(nperseg, sym=False)
    seg = np.stack([filtered[:, s:s + nperseg] * win for s in starts])
    seg = seg - seg.mean(axis=-1, keepdims=True)
    Xf = np.fft.rfft(seg, axis=-1)  # (n_seg, n_ch, n_freqs_rfft)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sfreq)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    Xf = Xf[:, :, band_mask]
    freqs = freqs[band_mask]
    # Cross-spectra per segment: S_ij = X_i * conj(X_j)
    S = np.einsum("scn,sdn->scdn", Xf, np.conj(Xf))  # (n_seg, ch, ch, F)
    return S, freqs


def compute_wpli(
    data: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
) -> np.ndarray:
    """Weighted Phase Lag Index (Vinck et al. 2011).

    wPLI = |E[Im(S_xy)]| / E[|Im(S_xy)|]
    (equivalently |E[|Im|*sign(Im)]| / E[|Im|]), expectation over
    independent observations (segments x frequency bins).
    Bounded in [0, 1].
    """
    S, f = compute_cross_spectrum_segments(data, sfreq, band)
    n_channels = S.shape[1]
    M = np.zeros((n_channels, n_channels), dtype=float)
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            imag = np.imag(S[:, i, j, :]).ravel()
            denom = np.mean(np.abs(imag))
            if denom > 0:
                M[i, j] = np.abs(np.mean(imag)) / denom
                M[j, i] = M[i, j]
    return M


def compute_dwpli(
    data: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
) -> np.ndarray:
    """Debiased squared wPLI (Vinck et al. 2011, eq. 8).

    dwPLI^2 = [ (sum Im_k)^2 - sum Im_k^2 ] / [ (sum |Im_k|)^2 - sum Im_k^2 ]
    over independent observations (segments x frequency bins).
    Bounded in [0, 1]; the debiasing removes the small-sample inflation.
    """
    S, f = compute_cross_spectrum_segments(data, sfreq, band)
    n_channels = S.shape[1]
    M = np.zeros((n_channels, n_channels), dtype=float)
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            imag = np.imag(S[:, i, j, :]).ravel()
            num = np.sum(imag) ** 2 - np.sum(imag ** 2)
            den = np.sum(np.abs(imag)) ** 2 - np.sum(imag ** 2)
            if den > 0:
                M[i, j] = max(num / den, 0.0)
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


# EGI HydroCel 129 montage: the 9 EOG channels (HydroCel numbers
# 8, 14, 17, 21, 25, 125, 126, 127, 128) are EXCLUDED from
# connectivity estimation (METH-EEGFULL-020): EOG-scalp edges carry
# the ocular signal EOG regression exists to remove, and blink/saccade
# rate covaries with age and inattention — a confound channel, not
# brain connectivity. EOG channels REMAIN in the recording (residual-
# ocular QC needs them); they are excluded at the feature step only.
# With 109 scalp channels: edges per matrix = 5,886; x 15 = 88,290.
EOG_HYDROCEL_NAMES = ("E8", "E14", "E17", "E21", "E25",
                      "E125", "E126", "E127", "E128")
NECK_FACE_HYDROCEL_NAMES = ("E38", "E43", "E44", "E48", "E49", "E56",
                            "E63", "E68", "E73", "E81", "E117")


def _egi_118_eog_positions() -> np.ndarray:
    """0-indexed EOG positions in the 118-channel array produced by the
    canonical chain (129 = E1..E128 + Cz, minus the 11 neck/face drops).
    Computed from the documented name mapping — not guessed."""
    full = [f"E{i}" for i in range(1, 129)] + ["Cz"]
    kept = [c for c in full if c not in NECK_FACE_HYDROCEL_NAMES]
    return np.array([kept.index(c) for c in EOG_HYDROCEL_NAMES])


def scalp_picks(n_channels: int,
                ch_names=None,
                ) -> np.ndarray:
    """Indices of scalp (non-EOG) channels. Name-based when ch_names is
    given; documented positional mapping for the canonical 118-channel
    EGI array; no-op when the array is already 109 scalp channels."""
    if ch_names is not None:
        return np.array([i for i, c in enumerate(ch_names)
                         if c not in EOG_HYDROCEL_NAMES])
    if n_channels == 118:
        eog_pos = set(_egi_118_eog_positions().tolist())
        picks = np.array([i for i in range(n_channels)
                          if i not in eog_pos])
        return picks
    if n_channels == 109:
        return np.arange(n_channels)
    import warnings
    warnings.warn(
        f"scalp_picks: n_channels={n_channels} is not the canonical EGI "
        f"118-array and no ch_names given; returning identity (no EOG "
        f"exclusion possible). Pass ch_names for EOG exclusion.",
        UserWarning)
    return np.arange(n_channels)


class ConnectivityModule:
    """Sensor-space connectivity orchestration (109 scalp ch, 5 bands,
    5 methods)."""

    def __init__(self, n_channels: int = 109, sfreq: float = 500.0,
                 methods: Tuple[str, ...] = PRIMARY_METHODS,
                 bands: Optional[Dict[str, Tuple[float, float]]] = None,
                 scalp_only: bool = True, ch_names=None):
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.methods = methods
        self.bands = bands if bands is not None else FREQUENCY_BANDS
        self.scalp_only = scalp_only
        self.ch_names = ch_names

    def _apply_picks(self, data: np.ndarray) -> np.ndarray:
        if self.scalp_only:
            picks = scalp_picks(data.shape[0], ch_names=self.ch_names)
            if len(picks) != self.n_channels:
                import warnings
                warnings.warn(
                    f"scalp picks give {len(picks)} channels, expected "
                    f"{self.n_channels}; using the picks anyway.",
                    UserWarning)
            return data[picks]
        return data

    def compute(self, data: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        return compute_connectivity(
            self._apply_picks(data), self.sfreq, methods=self.methods,
            bands=self.bands)

    def edges(self, data: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute and convert to edge vectors (long format)."""
        matrices = self.compute(data)
        out = {}
        for method, band_dict in matrices.items():
            out[method] = {}
            for band, M in band_dict.items():
                out[method][band] = matrix_to_edges(M)
        return out
