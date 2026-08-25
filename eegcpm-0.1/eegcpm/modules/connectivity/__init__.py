"""Connectivity analysis module."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from eegcpm.pipeline.base import BaseModule, ModuleResult
from eegcpm.data.conn_rois import get_roi_names, get_network_indices, CONN_NETWORKS


class ConnectivityModule(BaseModule):
    """
    Compute functional connectivity matrices.

    Supports:
    - Correlation-based: Pearson, Spearman, partial correlation
    - Phase-based: PLV, PLI, wPLI, dwPLI (debiased)
    - Spectral: Coherence, imaginary coherence (icoh)
    - Amplitude: AEC (Amplitude Envelope Correlation), orthogonalized AEC
    - Directional: PDC (Partial Directed Coherence), DTF (Directed Transfer Function)
    - Information-theoretic: MI, TE (future)

    References:
    - PLV: Lachaux et al. (1999)
    - PLI: Stam et al. (2007)
    - wPLI: Vinck et al. (2011)
    - dwPLI: Vinck et al. (2011)
    - Imaginary Coherence: Nolte et al. (2004)
    - AEC: Brookes et al. (2011)
    - PDC: Baccala & Sameshima (2001)
    - DTF: Kaminski & Blinowska (1991)
    """

    name = "connectivity"
    version = "0.3.0"
    description = "Functional connectivity analysis"

    SUPPORTED_METHODS = [
        "correlation", "spearman", "partial_correlation",
        "plv", "pli", "wpli", "dwpli",
        "coherence", "icoh",
        "aec", "aec_orth",
        "pdc", "dtf",
    ]

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        super().__init__(config, output_dir)
        self.methods = config.get("methods", ["correlation", "plv"])
        self.frequency_bands = config.get("frequency_bands", {
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
        })
        self.time_windows = config.get("time_windows", [
            {"name": "prestim", "tmin": -0.5, "tmax": 0.0},
            {"name": "poststim", "tmin": 0.0, "tmax": 0.5},
        ])
        self.use_mne_connectivity = config.get("use_mne_connectivity", False)
        self._has_mne_connectivity = self._check_mne_connectivity()
        # MVAR model order for PDC/DTF
        self.mvar_order = config.get("mvar_order", 10)

    def _check_mne_connectivity(self) -> bool:
        """Check if mne-connectivity is available."""
        try:
            import mne_connectivity
            return True
        except ImportError:
            return False

    def validate_input(self, data: Any) -> bool:
        """Validate input is ROI data dict or epochs."""
        if isinstance(data, dict):
            return True
        try:
            import mne
            return isinstance(data, mne.Epochs)
        except ImportError:
            return False

    def process(
        self,
        data: Any,
        subject: Optional[Any] = None,
        sfreq: float = 500.0,
        **kwargs,
    ) -> ModuleResult:
        """
        Compute connectivity matrices.

        Args:
            data: ROI time courses dict or Epochs
            subject: Subject info
            sfreq: Sampling frequency

        Returns:
            ModuleResult with connectivity matrices
        """
        start_time = time.time()
        output_files = []
        warnings = []

        try:
            # Extract ROI data if needed
            if isinstance(data, dict):
                roi_data = data
            else:
                # Assume epochs - extract from channels
                roi_data = self._epochs_to_roi(data)
                sfreq = data.info["sfreq"]

            # Get conditions
            conditions = [k for k in roi_data.keys()
                         if not k.endswith("_times") and k not in ["roi_names", "sfreq"]]

            # Compute connectivity for each condition, method, band, window
            connectivity = {}

            for condition in conditions:
                tc = roi_data[condition]
                times = roi_data.get(f"{condition}_times", np.arange(tc.shape[-1]) / sfreq)

                # Check if data is trial-level (3D) or evoked (2D)
                is_trial_level = tc.ndim == 3

                if is_trial_level:
                    # Trial-level: (n_trials, n_rois, n_times)
                    n_trials = tc.shape[0]
                    warnings.append(f"Processing {n_trials} trials for condition '{condition}'")

                for window in self.time_windows:
                    # Get time window indices
                    t_mask = (times >= window["tmin"]) & (times < window["tmax"])

                    if is_trial_level:
                        # Extract window for all trials: (n_trials, n_rois, n_times_window)
                        tc_window = tc[:, :, t_mask]

                        for method in self.methods:
                            if method in ["plv", "pli", "wpli", "dwpli", "coherence", "icoh"]:
                                # Phase/spectral methods: estimate cross-spectra
                                # across ALL trials (not per-trial Hilbert on
                                # short windows, which is degenerate).
                                for band_name, (fmin, fmax) in self.frequency_bands.items():
                                    mean_matrix, std_matrix, var_matrix = \
                                        self._compute_spectral_trials(
                                            tc_window, method, sfreq, fmin, fmax
                                        )
                                    key_base = f"{condition}_{window['name']}_{method}_{band_name}"
                                    connectivity[f"{key_base}_mean"] = mean_matrix
                                    connectivity[f"{key_base}_std"] = std_matrix
                                    connectivity[f"{key_base}_variance"] = var_matrix
                            else:
                                # Amplitude/directional methods: per-trial then aggregate
                                trial_matrices = []

                                for trial_idx in range(n_trials):
                                    trial_tc = tc_window[trial_idx, :, :]  # (n_rois, n_times_window)

                                    if method in ["aec", "aec_orth", "pdc", "dtf"]:
                                        # Phase/spectral/directional - need frequency bands
                                        for band_name, (fmin, fmax) in self.frequency_bands.items():
                                            conn_matrix = self._compute_connectivity(
                                                trial_tc, method, sfreq, fmin, fmax
                                            )
                                            trial_matrices.append((f"{band_name}", conn_matrix))
                                    else:
                                        # Amplitude-based (correlation, etc.)
                                        conn_matrix = self._compute_connectivity(
                                            trial_tc, method, sfreq
                                        )
                                        trial_matrices.append((None, conn_matrix))

                                # Aggregate across trials
                                # Group by band (for frequency-based methods)
                                from collections import defaultdict
                                matrices_by_band = defaultdict(list)

                                for band_name, matrix in trial_matrices:
                                    matrices_by_band[band_name].append(matrix)

                                # Compute statistics for each band
                                for band_name, matrices in matrices_by_band.items():
                                    stacked = np.stack(matrices, axis=0)  # (n_trials, n_rois, n_rois)

                                    # Compute mean, std, variance
                                    mean_matrix = np.mean(stacked, axis=0)
                                    std_matrix = np.std(stacked, axis=0)
                                    var_matrix = np.var(stacked, axis=0)

                                    if band_name:
                                        # Frequency-based method
                                        key_base = f"{condition}_{window['name']}_{method}_{band_name}"
                                    else:
                                        # Amplitude-based method
                                        key_base = f"{condition}_{window['name']}_{method}"

                                    connectivity[f"{key_base}_mean"] = mean_matrix
                                    connectivity[f"{key_base}_std"] = std_matrix
                                    connectivity[f"{key_base}_variance"] = var_matrix

                    else:
                        # Evoked mode: (n_rois, n_times)
                        tc_window = tc[:, t_mask]

                        for method in self.methods:
                            if method in ["plv", "pli", "wpli", "dwpli", "coherence", "icoh", "aec", "aec_orth", "pdc", "dtf"]:
                                # Phase/spectral/directional - need frequency bands
                                for band_name, (fmin, fmax) in self.frequency_bands.items():
                                    key = f"{condition}_{window['name']}_{method}_{band_name}"
                                    conn_matrix = self._compute_connectivity(
                                        tc_window, method, sfreq, fmin, fmax
                                    )
                                    connectivity[key] = conn_matrix
                            else:
                                # Amplitude-based (correlation, etc.)
                                key = f"{condition}_{window['name']}_{method}"
                                conn_matrix = self._compute_connectivity(
                                    tc_window, method, sfreq
                                )
                                connectivity[key] = conn_matrix

            # Save connectivity matrices
            subject_id = subject.id if subject else "unknown"
            conn_path = self.output_dir / f"{subject_id}_connectivity.npz"
            np.savez(conn_path, **connectivity)
            output_files.append(conn_path)

            return ModuleResult(
                success=True,
                module_name=self.name,
                execution_time_seconds=time.time() - start_time,
                outputs={
                    "data": connectivity,
                    "connectivity": connectivity,
                },
                output_files=output_files,
                warnings=warnings,
                metadata={
                    "methods": self.methods,
                    "frequency_bands": list(self.frequency_bands.keys()),
                    "time_windows": [w["name"] for w in self.time_windows],
                    "n_matrices": len(connectivity),
                    "matrix_shape": list(connectivity.values())[0].shape if connectivity else None,
                    "supported_methods": self.SUPPORTED_METHODS,
                },
            )

        except Exception as e:
            return ModuleResult(
                success=False,
                module_name=self.name,
                execution_time_seconds=time.time() - start_time,
                errors=[str(e)],
            )

    def _compute_spectral_trials(
        self,
        X: np.ndarray,
        method: str,
        sfreq: float,
        fmin: float,
        fmax: float,
        n_bootstrap: int = 100,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Across-trials spectral connectivity (phase/spectral methods).

        Estimates per-trial cross-spectra (Hann window + rFFT) and computes
        metrics from trial-averaged spectra. This replaces per-trial
        Hilbert-based estimation, which is degenerate on short windows
        (~200-500 samples): narrowband filtfilt+hilbert on such windows
        yields near-saturated values even for independent noise.

        Metrics (Vinck et al. 2011; Nolte et al. 2004), with sample
        dimension = trials x band frequencies:
        - plv: |E[S/|S|]|
        - pli: |E[sign(Im S)]|
        - wpli: |E[Im S]| / E[|Im S|]
        - dwpli: debiased wPLI, ((sum Im)^2 - sum Im^2) / ((sum|Im|)^2 - sum Im^2)
        - coherence: |E[S]|^2 / (E[Pii] E[Pjj])
        - icoh: |Im E[S]| / sqrt(E[Pii] E[Pjj])

        Std is estimated by bootstrap over trials.

        Args:
            X: ROI time courses (n_trials, n_rois, n_times)
            method: Connectivity method
            sfreq: Sampling frequency
            fmin, fmax: Frequency band
            n_bootstrap: Bootstrap resamples for std estimation
            random_state: RNG seed for bootstrap

        Returns:
            (mean_matrix, std_matrix, var_matrix), each (n_rois, n_rois)
        """
        n_trials, n_rois, n_times = X.shape
        win = np.hanning(n_times)
        F = np.fft.rfft(X * win, axis=2)  # (trials, rois, freqs)
        freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
        bmask = (freqs >= fmin) & (freqs <= fmax)
        Fb = F[:, :, bmask]  # (trials, rois, band_freqs)

        # Per-trial cross-spectra: (trials, i, j, freqs)
        S = Fb[:, :, None, :] * np.conj(Fb[:, None, :, :])

        def _metric(Sx: np.ndarray) -> np.ndarray:
            # Sx: (n_samples, i, j, freqs)
            Im = Sx.imag
            if method == "plv":
                m = np.abs((Sx / np.maximum(np.abs(Sx), 1e-20)).mean(axis=(0, 3)))
            elif method == "pli":
                m = np.abs(np.sign(Im).mean(axis=(0, 3)))
            elif method == "wpli":
                m = (np.abs(Im.mean(axis=(0, 3)))
                     / np.maximum(np.abs(Im).mean(axis=(0, 3)), 1e-20))
            elif method == "dwpli":
                sum_im = Im.sum(axis=(0, 3))
                sum_im2 = (Im ** 2).sum(axis=(0, 3))
                sum_abs = np.abs(Im).sum(axis=(0, 3))
                num = sum_im ** 2 - sum_im2
                den = sum_abs ** 2 - sum_im2
                m = np.where(den > 0, num / np.maximum(den, 1e-20), 0.0)
                m = np.clip(m, 0.0, 1.0)
            elif method == "coherence":
                cs = Sx.mean(axis=(0, 3))
                p = Sx.real.diagonal(axis1=1, axis2=2).mean(axis=(0, 2))  # (i,)
                m = (np.abs(cs) ** 2
                     / np.maximum(np.outer(p, p), 1e-20))
            elif method == "icoh":
                cs = Sx.mean(axis=(0, 3))
                p = Sx.real.diagonal(axis1=1, axis2=2).mean(axis=(0, 2))
                m = (np.abs(cs.imag)
                     / np.maximum(np.sqrt(np.outer(p, p)), 1e-20))
            else:
                raise ValueError(f"Unsupported spectral method: {method}")
            return np.nan_to_num(m)

        mean_matrix = _metric(S)

        # Bootstrap over trials for variability
        rng = np.random.default_rng(random_state)
        boot = np.empty((n_bootstrap, n_rois, n_rois))
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_trials, n_trials)
            boot[b] = _metric(S[idx])
        std_matrix = boot.std(axis=0)
        var_matrix = boot.var(axis=0)

        # Diagonal carries no information; keep 0 for consistency with
        # previous outputs and feature extraction via upper triangle.
        for mtx in (mean_matrix, std_matrix, var_matrix):
            np.fill_diagonal(mtx, 0.0)

        return mean_matrix, std_matrix, var_matrix

    def _compute_connectivity(
        self,
        data: np.ndarray,
        method: str,
        sfreq: float,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute connectivity matrix.

        Args:
            data: ROI time courses (n_rois, n_times)
            method: Connectivity method
            sfreq: Sampling frequency
            fmin, fmax: Frequency band (for phase methods)

        Returns:
            Connectivity matrix (n_rois, n_rois)
        """
        n_rois = data.shape[0]

        # Use MNE-Connectivity if available and enabled
        if (self.use_mne_connectivity and self._has_mne_connectivity
                and method in ["plv", "pli", "wpli", "coh", "coherence"]):
            return self._compute_mne_connectivity(data, method, sfreq, fmin, fmax)

        if method == "correlation":
            return np.corrcoef(data)

        elif method == "spearman":
            from scipy.stats import spearmanr
            corr, _ = spearmanr(data.T)
            return corr

        elif method == "partial_correlation":
            return self._partial_correlation(data)

        elif method == "plv":
            return self._compute_plv(data, sfreq, fmin, fmax)

        elif method == "pli":
            return self._compute_pli(data, sfreq, fmin, fmax)

        elif method == "wpli":
            return self._compute_wpli(data, sfreq, fmin, fmax)

        elif method == "dwpli":
            return self._compute_dwpli(data, sfreq, fmin, fmax)

        elif method == "coherence":
            return self._compute_coherence(data, sfreq, fmin, fmax)

        elif method == "icoh":
            return self._compute_icoh(data, sfreq, fmin, fmax)

        elif method == "aec":
            return self._compute_aec(data, sfreq, fmin, fmax, orthogonalize=False)

        elif method == "aec_orth":
            return self._compute_aec(data, sfreq, fmin, fmax, orthogonalize=True)

        elif method == "pdc":
            return self._compute_pdc(data, sfreq, fmin, fmax)

        elif method == "dtf":
            return self._compute_dtf(data, sfreq, fmin, fmax)

        else:
            raise ValueError(f"Unknown method: {method}. Supported: {self.SUPPORTED_METHODS}")

    def _partial_correlation(self, data: np.ndarray) -> np.ndarray:
        """Compute partial correlation matrix."""
        from scipy import linalg

        corr = np.corrcoef(data)
        try:
            precision = linalg.inv(corr)
            d = np.sqrt(np.diag(precision))
            partial = -precision / np.outer(d, d)
            np.fill_diagonal(partial, 1.0)
            return partial
        except linalg.LinAlgError:
            return corr  # Fall back to correlation

    def _compute_plv(
        self,
        data: np.ndarray,
        sfreq: float,
        fmin: float,
        fmax: float,
    ) -> np.ndarray:
        """Compute Phase Locking Value."""
        from scipy.signal import hilbert, butter, filtfilt

        n_rois, n_times = data.shape

        # Bandpass filter
        nyq = sfreq / 2
        b, a = butter(4, [fmin / nyq, fmax / nyq], btype="band")
        filtered = filtfilt(b, a, data, axis=1)

        # Hilbert transform for phase
        analytic = hilbert(filtered, axis=1)
        phase = np.angle(analytic)

        # PLV
        plv = np.zeros((n_rois, n_rois))
        for i in range(n_rois):
            for j in range(i + 1, n_rois):
                phase_diff = phase[i] - phase[j]
                plv[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv[j, i] = plv[i, j]

        np.fill_diagonal(plv, 1.0)
        return plv

    def _compute_pli(
        self,
        data: np.ndarray,
        sfreq: float,
        fmin: float,
        fmax: float,
    ) -> np.ndarray:
        """Compute Phase Lag Index."""
        from scipy.signal import hilbert, butter, filtfilt

        n_rois, n_times = data.shape

        nyq = sfreq / 2
        b, a = butter(4, [fmin / nyq, fmax / nyq], btype="band")
        filtered = filtfilt(b, a, data, axis=1)

        analytic = hilbert(filtered, axis=1)
        phase = np.angle(analytic)

        pli = np.zeros((n_rois, n_rois))
        for i in range(n_rois):
            for j in range(i + 1, n_rois):
                phase_diff = phase[i] - phase[j]
                pli[i, j] = np.abs(np.mean(np.sign(np.sin(phase_diff))))
                pli[j, i] = pli[i, j]

        return pli

    def _compute_wpli(
        self,
        data: np.ndarray,
        sfreq: float,
        fmin: float,
        fmax: float,
    ) -> np.ndarray:
        """Compute Weighted Phase Lag Index."""
        from scipy.signal import hilbert, butter, filtfilt

        n_rois, n_times = data.shape

        nyq = sfreq / 2
        b, a = butter(4, [fmin / nyq, fmax / nyq], btype="band")
        filtered = filtfilt(b, a, data, axis=1)

        analytic = hilbert(filtered, axis=1)

        wpli = np.zeros((n_rois, n_rois))
        for i in range(n_rois):
            for j in range(i + 1, n_rois):
                cross = analytic[i] * np.conj(analytic[j])
                imag_cross = np.imag(cross)
                wpli[i, j] = np.abs(np.mean(imag_cross)) / np.mean(np.abs(imag_cross))
                wpli[j, i] = wpli[i, j]

        return np.nan_to_num(wpli)

    def _compute_coherence(
        self,
        data: np.ndarray,
        sfreq: float,
        fmin: float,
        fmax: float,
    ) -> np.ndarray:
        """Compute magnitude-squared coherence."""
        from scipy.signal import coherence as sig_coherence

        n_rois = data.shape[0]
        coh = np.zeros((n_rois, n_rois))

        for i in range(n_rois):
            for j in range(i + 1, n_rois):
                f, c = sig_coherence(data[i], data[j], fs=sfreq, nperseg=256)
                # Average coherence in frequency band
                mask = (f >= fmin) & (f <= fmax)
                coh[i, j] = np.mean(c[mask])
                coh[j, i] = coh[i, j]

        np.fill_diagonal(coh, 1.0)
        return coh

    def _compute_mne_connectivity(
        self,
        data: np.ndarray,
        method: str,
        sfreq: float,
        fmin: float,
        fmax: float,
    ) -> np.ndarray:
        """
        Compute connectivity using MNE-Connectivity.

        Args:
            data: ROI time courses (n_rois, n_times)
            method: Connectivity method (plv, pli, wpli, coh)
            sfreq: Sampling frequency
            fmin, fmax: Frequency band

        Returns:
            Connectivity matrix (n_rois, n_rois)
        """
        from mne_connectivity import spectral_connectivity_epochs

        n_rois = data.shape[0]

        # MNE-Connectivity expects (n_epochs, n_signals, n_times)
        # Wrap single trial data as single epoch
        data_3d = data[np.newaxis, :, :]

        # Map method names
        mne_method = method
        if method == "coherence":
            mne_method = "coh"

        # Compute connectivity
        conn = spectral_connectivity_epochs(
            data_3d,
            method=mne_method,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            mode="multitaper",
            mt_bandwidth=2.0,
            verbose=False,
        )

        # Get dense matrix
        conn_data = conn.get_data(output="dense")

        # Shape is (n_signals, n_signals, n_freqs) - squeeze freqs
        conn_matrix = np.squeeze(conn_data)

        # Ensure symmetric
        conn_matrix = (conn_matrix + conn_matrix.T) / 2
        np.fill_diagonal(conn_matrix, 1.0)

        return conn_matrix

    def _compute_dwpli(
        self,
        data: np.ndarray,
        sfreq: float,
        fmin: float,
        fmax: float,
    ) -> np.ndarray:
        """
        Compute Debiased Weighted Phase Lag Index.

        Removes sample size bias from wPLI by subtracting the expected value
        under the null hypothesis of no coupling.

        Reference: Vinck et al. (2011) NeuroImage
        """
        from scipy.signal import hilbert, butter, filtfilt

        n_rois, n_times = data.shape

        # Bandpass filter
        nyq = sfreq / 2
        b, a = butter(4, [fmin / nyq, fmax / nyq], btype="band")
        filtered = filtfilt(b, a, data, axis=1)

        # Hilbert transform
        analytic = hilbert(filtered, axis=1)

        dwpli = np.zeros((n_rois, n_rois))
        for i in range(n_rois):
            for j in range(i + 1, n_rois):
                cross = analytic[i] * np.conj(analytic[j])
                imag_cross = np.imag(cross)

                # dwPLI formula: (sum(imag)^2 - sum(imag^2)) / (sum(|imag|)^2 - sum(imag^2))
                numerator = np.sum(imag_cross)**2 - np.sum(imag_cross**2)
                denominator = np.sum(np.abs(imag_cross))**2 - np.sum(imag_cross**2)

                if denominator > 0:
                    dwpli[i, j] = numerator / denominator
                else:
                    dwpli[i, j] = 0.0

                dwpli[j, i] = dwpli[i, j]

        return np.nan_to_num(dwpli)

    def _compute_icoh(
        self,
        data: np.ndarray,
        sfreq: float,
        fmin: float,
        fmax: float,
    ) -> np.ndarray:
        """
        Compute Imaginary Coherence.

        Uses only the imaginary part of coherency, which is insensitive
        to volume conduction and zero-lag spurious interactions.

        Reference: Nolte et al. (2004) Clinical Neurophysiology
        """
        from scipy.signal import csd

        n_rois = data.shape[0]
        icoh = np.zeros((n_rois, n_rois))

        for i in range(n_rois):
            for j in range(i + 1, n_rois):
                # Cross-spectral density
                f, Pxy = csd(data[i], data[j], fs=sfreq, nperseg=256)

                # Auto-spectral densities
                _, Pxx = csd(data[i], data[i], fs=sfreq, nperseg=256)
                _, Pyy = csd(data[j], data[j], fs=sfreq, nperseg=256)

                # Coherency (complex)
                coherency = Pxy / np.sqrt(Pxx * Pyy)

                # Take imaginary part and average in band
                mask = (f >= fmin) & (f <= fmax)
                icoh[i, j] = np.mean(np.abs(np.imag(coherency[mask])))
                icoh[j, i] = icoh[i, j]

        return icoh

    def _compute_aec(
        self,
        data: np.ndarray,
        sfreq: float,
        fmin: float,
        fmax: float,
        orthogonalize: bool = False,
    ) -> np.ndarray:
        """
        Compute Amplitude Envelope Correlation.

        Measures correlation between amplitude envelopes of band-limited signals.
        Optionally applies symmetric orthogonalization to reduce signal leakage.

        Reference: Brookes et al. (2011) PNAS
        """
        from scipy.signal import hilbert, butter, filtfilt

        n_rois, n_times = data.shape

        # Bandpass filter
        nyq = sfreq / 2
        b, a = butter(4, [fmin / nyq, fmax / nyq], btype="band")
        filtered = filtfilt(b, a, data, axis=1)

        # Hilbert transform for envelope
        analytic = hilbert(filtered, axis=1)
        envelopes = np.abs(analytic)

        if orthogonalize:
            # Symmetric orthogonalization to reduce leakage
            aec = np.zeros((n_rois, n_rois))
            for i in range(n_rois):
                for j in range(i + 1, n_rois):
                    aec[i, j] = self._orthogonalized_aec_pair(
                        analytic[i], analytic[j]
                    )
                    aec[j, i] = aec[i, j]
            return aec
        else:
            # Standard AEC
            return np.corrcoef(envelopes)

    def _orthogonalized_aec_pair(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
    ) -> float:
        """
        Compute orthogonalized AEC between two signals.

        Symmetric orthogonalization: correlate envelope of signal1 with
        envelope of signal2 after removing the component of signal2 that
        is in-phase with signal1 (and vice versa).
        """
        # Orthogonalize signal2 w.r.t signal1
        beta = np.dot(signal2.real, signal1.real) / np.dot(signal1.real, signal1.real)
        signal2_orth = signal2 - beta * signal1

        # Orthogonalize signal1 w.r.t signal2
        beta = np.dot(signal1.real, signal2.real) / np.dot(signal2.real, signal2.real)
        signal1_orth = signal1 - beta * signal2

        # Correlation of envelopes
        env1 = np.abs(signal1_orth)
        env2 = np.abs(signal2_orth)

        return np.corrcoef(env1, env2)[0, 1]

    def _compute_pdc(
        self,
        data: np.ndarray,
        sfreq: float,
        fmin: float,
        fmax: float,
    ) -> np.ndarray:
        """
        Compute Partial Directed Coherence.

        Directional connectivity measure based on multivariate autoregressive
        (MVAR) modeling. PDC(i->j) quantifies the influence of source i on target j.

        Reference: Baccala & Sameshima (2001) Biological Cybernetics
        """
        n_rois = data.shape[0]

        # Fit MVAR model
        A, sigma = self._fit_mvar(data, order=self.mvar_order)

        # Compute PDC at frequencies
        freqs = np.linspace(0, sfreq / 2, 256)
        pdc = np.zeros((n_rois, n_rois))

        for freq_idx, freq in enumerate(freqs):
            if fmin <= freq <= fmax:
                # A(f) = I - sum_k A_k * exp(-2πifkΔt)
                omega = 2 * np.pi * freq / sfreq
                A_f = np.eye(n_rois, dtype=complex)
                for k in range(1, self.mvar_order + 1):
                    A_f -= A[:, :, k - 1] * np.exp(-1j * omega * k)

                # PDC(i->j, f) = |A_f(j,i)| / sqrt(sum_k |A_f(k,i)|^2)
                for i in range(n_rois):
                    for j in range(n_rois):
                        if i != j:
                            numerator = np.abs(A_f[j, i])
                            denominator = np.sqrt(np.sum(np.abs(A_f[:, i])**2))
                            pdc[i, j] += numerator / (denominator + 1e-10)

        # Average over frequency band
        n_freqs = np.sum((freqs >= fmin) & (freqs <= fmax))
        if n_freqs > 0:
            pdc /= n_freqs

        return pdc

    def _compute_dtf(
        self,
        data: np.ndarray,
        sfreq: float,
        fmin: float,
        fmax: float,
    ) -> np.ndarray:
        """
        Compute Directed Transfer Function.

        Directional connectivity based on the transfer function H(f) = A(f)^(-1).
        DTF(i->j) quantifies the influence of source i on target j in the frequency domain.

        Reference: Kaminski & Blinowska (1991) Biological Cybernetics
        """
        n_rois = data.shape[0]

        # Fit MVAR model
        A, sigma = self._fit_mvar(data, order=self.mvar_order)

        # Compute DTF at frequencies
        freqs = np.linspace(0, sfreq / 2, 256)
        dtf = np.zeros((n_rois, n_rois))

        for freq_idx, freq in enumerate(freqs):
            if fmin <= freq <= fmax:
                # A(f) = I - sum_k A_k * exp(-2πifkΔt)
                omega = 2 * np.pi * freq / sfreq
                A_f = np.eye(n_rois, dtype=complex)
                for k in range(1, self.mvar_order + 1):
                    A_f -= A[:, :, k - 1] * np.exp(-1j * omega * k)

                # H(f) = A(f)^(-1)
                try:
                    H_f = np.linalg.inv(A_f)

                    # DTF(i->j, f) = |H_f(j,i)|^2 / sum_k |H_f(j,k)|^2
                    for i in range(n_rois):
                        for j in range(n_rois):
                            if i != j:
                                numerator = np.abs(H_f[j, i])**2
                                denominator = np.sum(np.abs(H_f[j, :])**2)
                                dtf[i, j] += numerator / (denominator + 1e-10)
                except np.linalg.LinAlgError:
                    pass  # Singular matrix, skip this frequency

        # Average over frequency band
        n_freqs = np.sum((freqs >= fmin) & (freqs <= fmax))
        if n_freqs > 0:
            dtf /= n_freqs

        return dtf

    def _fit_mvar(
        self,
        data: np.ndarray,
        order: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit multivariate autoregressive (MVAR) model.

        Model: X(t) = sum_{k=1}^{order} A_k * X(t-k) + E(t)

        Args:
            data: ROI time courses (n_rois, n_times)
            order: MVAR model order

        Returns:
            A: Coefficient matrices (n_rois, n_rois, order)
            sigma: Noise covariance matrix (n_rois, n_rois)
        """
        n_rois, n_times = data.shape

        # Build design matrix
        X = []  # Predictors
        Y = []  # Targets

        for t in range(order, n_times):
            # Y(t) = [x1(t), x2(t), ..., xN(t)]
            Y.append(data[:, t])

            # X(t) = [x1(t-1), ..., xN(t-1), x1(t-2), ..., xN(t-2), ..., x1(t-p), ..., xN(t-p)]
            x_row = []
            for lag in range(1, order + 1):
                x_row.extend(data[:, t - lag])
            X.append(x_row)

        X = np.array(X)  # (n_samples, n_rois * order)
        Y = np.array(Y)  # (n_samples, n_rois)

        # Least squares: A = (X'X)^(-1) X'Y
        try:
            A_flat = np.linalg.lstsq(X, Y, rcond=None)[0]  # (n_rois * order, n_rois)
            A_flat = A_flat.T  # (n_rois, n_rois * order)

            # Reshape to (n_rois, n_rois, order)
            A = A_flat.reshape(n_rois, n_rois, order)

            # Residual covariance
            Y_pred = X @ A_flat.T
            residuals = Y - Y_pred
            sigma = np.cov(residuals.T)

            return A, sigma

        except np.linalg.LinAlgError:
            # Fallback: return identity matrices
            A = np.zeros((n_rois, n_rois, order))
            for k in range(order):
                A[:, :, k] = np.eye(n_rois) / order
            sigma = np.eye(n_rois)
            return A, sigma

    def _epochs_to_roi(self, epochs) -> Dict[str, np.ndarray]:
        """Convert epochs to ROI-like structure."""
        # Simple version: use channels as "ROIs"
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        avg_data = np.mean(data, axis=0)  # Average over epochs

        return {
            "sensor": avg_data,
            "sensor_times": epochs.times,
            "roi_names": epochs.ch_names,
        }

    def get_output_spec(self) -> Dict[str, str]:
        return {
            "connectivity": "Dict of connectivity matrices",
        }


# --- New (Stage 4) connectivity exports (per spec §3.k) ---
# These are the SENSOR-SPACE methods used by the EEG d-factor arm.
# They supplement the legacy SOURCE-SPACE ConnectivityModule above.

from .connectivity import (
    ALL_METHODS,
    FREQUENCY_BANDS,
    PRIMARY_METHODS,
    REPLICATION_METHODS,
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
