"""Tests for the connectivity module (per spec §3.k)."""

import numpy as np
import pytest

from eegcpm.modules.connectivity import (
    ALL_METHODS,
    FREQUENCY_BANDS,
    PRIMARY_METHODS,
    REPLICATION_METHODS,
    ConnectivityModule,
    bandpass,
    compute_aec_orth,
    compute_connectivity,
    compute_coherence,
    compute_cross_spectrum,
    compute_dwpli,
    compute_plv,
    compute_wpli,
    edges_to_matrix,
    matrix_to_edges,
    upper_triangle_indices,
)


# --------------------------------------------------------------- helpers

def _make_2ch_signal(sfreq: int = 500, duration_s: int = 10, seed: int = 0):
    """Two channels with a known phase coupling (channel 2 lags
    channel 1 by 5 samples = 10 ms)."""
    rng = np.random.RandomState(seed)
    n = int(duration_s * sfreq)
    t = np.arange(n) / sfreq
    x1 = rng.randn(n)
    x2 = np.roll(x1, 5) + 0.1 * rng.randn(n)
    return np.vstack([x1, x2]), sfreq


# --------------------------------------------------------------- band tests (S13)

def test_frequency_bands_delta_2_4_not_1_4():
    """S13: delta band is [2, 4] Hz (consistent with specparam 2 Hz
    floor), NOT [1, 4]."""
    delta = FREQUENCY_BANDS["delta"]
    assert delta == (2.0, 4.0), (
        f"delta band is {delta}; spec requires (2.0, 4.0) per S13."
    )


def test_frequency_bands_complete():
    """All 5 canonical bands present (delta, theta, alpha, beta, gamma)."""
    for band in ("delta", "theta", "alpha", "beta", "gamma"):
        assert band in FREQUENCY_BANDS


# --------------------------------------------------------------- method tests

def test_wpli_returns_symmetric_zero_diagonal():
    data, sfreq = _make_2ch_signal()
    M = compute_wpli(data, sfreq, band=(8, 13))
    n = M.shape[0]
    assert M.shape == (n, n)
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 0.0)


def test_dwpli_returns_symmetric_zero_diagonal():
    data, sfreq = _make_2ch_signal()
    M = compute_dwpli(data, sfreq, band=(8, 13))
    n = M.shape[0]
    assert M.shape == (n, n)
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 0.0)


def test_aec_orth_returns_symmetric_unit_diagonal():
    data, sfreq = _make_2ch_signal()
    M = compute_aec_orth(data, sfreq, band=(8, 13))
    n = M.shape[0]
    assert M.shape == (n, n)
    assert np.allclose(M, M.T)
    # Diagonal is 1.0 by construction
    assert np.allclose(np.diag(M), 1.0, atol=1e-6)


def test_coherence_returns_symmetric_zero_diagonal():
    data, sfreq = _make_2ch_signal()
    M = compute_coherence(data, sfreq, band=(8, 13))
    n = M.shape[0]
    assert M.shape == (n, n)
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 0.0)


def test_plv_returns_symmetric_zero_diagonal():
    data, sfreq = _make_2ch_signal()
    M = compute_plv(data, sfreq, band=(8, 13))
    n = M.shape[0]
    assert M.shape == (n, n)
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 0.0)


# --------------------------------------------------------------- edges tests

def test_upper_triangle_indices_count():
    """n_channels=109 -> 109*108/2 = 5886 upper-triangle edges."""
    i, j = upper_triangle_indices(109)
    assert len(i) == 109 * 108 // 2
    assert len(i) == 5886


def test_matrix_to_edges_roundtrip():
    n = 10
    rng = np.random.RandomState(0)
    M = rng.randn(n, n)
    M = (M + M.T) / 2
    edges = matrix_to_edges(M)
    M_back = edges_to_matrix(edges, n)
    # The diagonal is set to 0 in the back-conversion (off-diagonal
    # edges only); the off-diagonal must match.
    for i in range(n):
        for j in range(n):
            if i != j:
                assert abs(M[i, j] - M_back[i, j]) < 1e-12
    assert np.allclose(np.diag(M_back), 0.0)


def test_edges_equals_upper_triangle():
    n = 5
    M = np.arange(n * n).reshape(n, n).astype(float)
    M = M + M.T  # symmetric
    edges = matrix_to_edges(M)
    i, j = upper_triangle_indices(n)
    assert np.array_equal(edges, M[i, j])


# --------------------------------------------------------------- orchestration

def test_compute_connectivity_all_primary_methods():
    """PRIMARY methods = {wpli, dwpli, aec_orth}, 5 bands = 15 matrices."""
    data, sfreq = _make_2ch_signal(duration_s=5)
    out = compute_connectivity(data, sfreq, methods=PRIMARY_METHODS)
    assert set(out.keys()) == set(PRIMARY_METHODS)
    for method, band_dict in out.items():
        assert set(band_dict.keys()) == set(FREQUENCY_BANDS.keys())


def test_compute_connectivity_includes_replication():
    data, sfreq = _make_2ch_signal(duration_s=5)
    out = compute_connectivity(data, sfreq, methods=ALL_METHODS)
    assert set(out.keys()) == {"wpli", "dwpli", "aec_orth", "coherence", "plv"}


def test_connectivity_module_109ch_5bands():
    """109 ch -> 5,886 edges per matrix; 3 methods x 5 bands = 15 matrices
    = 88,290 edges. Use the new 2D-array interface."""
    from eegcpm.modules.connectivity.connectivity import (
        ConnectivityModule as _NewCM,
    )
    cm = _NewCM(n_channels=109, sfreq=500.0)
    rng = np.random.RandomState(0)
    data = rng.randn(109, 1000) * 1e-6
    matrices = cm.compute(data)
    assert set(matrices.keys()) == set(PRIMARY_METHODS)
    for method, band_dict in matrices.items():
        for band, M in band_dict.items():
            assert M.shape == (109, 109)


def test_connectivity_module_edges():
    """Use the new 2D-array interface for the edge-vector API."""
    from eegcpm.modules.connectivity.connectivity import (
        ConnectivityModule as _NewCM,
    )
    cm = _NewCM(n_channels=20, sfreq=500.0)
    rng = np.random.RandomState(0)
    data = rng.randn(20, 1000) * 1e-6
    edges = cm.edges(data)
    # 20 ch -> 20*19/2 = 190 edges per matrix
    for method, band_dict in edges.items():
        for band, e in band_dict.items():
            assert e.shape == (190,)


# --------------------------------------------------------------- mne-connectivity cross-check (METH-022)

def test_wpli_close_to_mne_connectivity_if_available():
    """METH-022: cross-check our wPLI against mne-connectivity on a
    synthetic known-coupling fixture. mne-connectivity is OPTIONAL
    (not installed in this venv); when absent, the test is skipped."""
    mne_conn = pytest.importorskip("mne_connectivity", reason="mne_connectivity not installed")
    data, sfreq = _make_2ch_signal(duration_s=5)
    # Our wPLI
    ours = compute_wpli(data, sfreq, band=(8, 13))
    # mne-connectivity
    wpli_mne, _freqs = mne_conn.spectral_connectivity(
        [data], sfreq=sfreq, method="wpli",
        fmin=[8], fmax=[13], faverage=True, verbose=False,
    )
    # wpli_mne shape: (n_pairs, n_connections)
    assert abs(ours[0, 1] - wpli_mne[0, 0]) < 0.2, (
        f"wPLI diverges from mne-connectivity: ours={ours[0, 1]}, "
        f"mne={wpli_mne[0, 0]}"
    )


def test_dwpli_close_to_mne_connectivity_if_available():
    mne_conn = pytest.importorskip("mne_connectivity", reason="mne_connectivity not installed")
    data, sfreq = _make_2ch_signal(duration_s=5)
    ours = compute_dwpli(data, sfreq, band=(8, 13))
    dwpli_mne, _freqs = mne_conn.spectral_connectivity(
        [data], sfreq=sfreq, method="dwpli",
        fmin=[8], fmax=[13], faverage=True, verbose=False,
    )
    assert abs(ours[0, 1] - dwpli_mne[0, 0]) < 0.2


# --------------------------------------------------------------- bandpass

def test_bandpass_filters_correctly():
    """Bandpass should attenuate frequencies outside the band."""
    sr = 500
    n = 5 * sr  # 5 seconds
    rng = np.random.RandomState(0)
    data = rng.randn(2, n)
    # Bandpass 8-13 Hz
    filtered = bandpass(data, sr, band=(8, 13))
    assert filtered.shape == data.shape
    # Power outside the band should be reduced
    from scipy import signal as sp_signal
    f, P = sp_signal.welch(filtered[0], fs=sr, nperseg=1024)
    band_mask = (f >= 8) & (f <= 13)
    power_in = np.sum(P[band_mask])
    power_total = np.sum(P)
    assert power_in / power_total > 0.5, (
        f"Bandpass did not concentrate power in [8, 13] Hz: "
        f"{power_in / power_total:.2%} in band"
    )
