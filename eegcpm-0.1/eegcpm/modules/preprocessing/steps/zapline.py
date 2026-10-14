"""
Zapline-Plus line noise removal step.

Uses pyzaplineplus for adaptive removal of line noise and harmonics
via subspace decomposition. Automatically detects and removes 50/60 Hz
line noise without creating spectral notches.

Author: EEGCPM Development Team
Created: 2025-12
"""

from typing import Any, Dict, Optional, Tuple
import mne
import numpy as np

from .base import ProcessingStep


class ZaplineStep(ProcessingStep):
    """
    Zapline-Plus line noise removal step.

    Zapline-Plus uses PCA-based subspace decomposition to adaptively remove
    line noise (50/60 Hz) and harmonics without distorting brain signals.
    This is superior to notch filtering which creates spectral notches and
    can remove legitimate brain activity.

    Uses the pyzaplineplus library which automatically detects line frequency
    and adapts the cleaning process based on data characteristics.

    Based on: de Cheveigné (2020). ZapLine-plus: A Zapline extension for automatic
    and adaptive removal of frequency-specific noise artifacts in M/EEG.
    NeuroImage, 215, 116803.

    Parameters
    ----------
    fline : float or None
        Expected line noise frequency in Hz (50 for EU, 60 for US).
        If None, pyzaplineplus will automatically detect line frequency.
        Default: None (auto-detect)
    nremove : int or None
        Deprecated - pyzaplineplus automatically determines optimal number
    nfft : int
        FFT length for spectral analysis (default: 8192)
    nkeep : int or None
        Deprecated - pyzaplineplus handles automatically
    adaptive : bool
        Deprecated - pyzaplineplus is always adaptive (default: True)
    noisefreqs : list or None
        Specific frequencies to remove (e.g., [60, 120] for 60 Hz + harmonics).
        If empty/None, auto-detection is used. Default: None
    noiseCompDetectSigma : float
        Threshold in standard deviations for detecting noise components.
        Lower values = more aggressive cleaning. Range: 2.5-5.0
        Default: 3.0 (moderate). Try 2.5 for stronger cleaning.
    fixedNremove : int
        Fallback number of components to remove if adaptive detection fails.
        Default: 1. Increase to 2-3 for stronger cleaning.
    picks : str, list, or None
        Channels to process (default: 'eeg')
    verbose : bool
        Print diagnostic information (default: False)

    Examples
    --------
    Auto-detect line frequency (recommended):
    >>> step = ZaplineStep(fline=None)

    Specify expected frequency (60 Hz for US):
    >>> step = ZaplineStep(fline=60)

    EU 50 Hz line frequency:
    >>> step = ZaplineStep(fline=50)

    Verbose output:
    >>> step = ZaplineStep(verbose=True)

    Notes
    -----
    - Works best with > 30 seconds of data
    - Requires at least 2 channels
    - More robust than notch filtering to bad channels and artifacts
    - Preserves phase relationships and brain signals
    - Automatically handles harmonics
    - Automatically detects actual line frequency
    """

    name = "zapline"
    version = "1.0"

    def __init__(
        self,
        fline: Optional[float] = None,
        nremove: Optional[int] = None,
        nfft: int = 8192,
        nkeep: Optional[int] = None,
        adaptive: bool = True,
        noisefreqs: Optional[list] = None,
        noiseCompDetectSigma: float = 3.0,
        fixedNremove: int = 1,
        picks: str = 'eeg',
        verbose: bool = False,
        enabled: bool = True,
    ):
        """Initialize Zapline step."""
        super().__init__(enabled=enabled)

        self.fline = fline
        self.nremove = nremove
        self.nfft = nfft
        self.nkeep = nkeep
        self.adaptive = adaptive
        self.noisefreqs = noisefreqs or []
        self.noiseCompDetectSigma = noiseCompDetectSigma
        self.fixedNremove = fixedNremove
        self.picks = picks
        self.verbose = verbose

        # Validate parameters
        if fline is not None and fline not in [50, 60]:
            print(f"Warning: Unusual line frequency {fline} Hz. Standard values: 50 (EU), 60 (US), or None (auto-detect)")

    def process(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any]
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        """
        Apply Zapline-Plus line noise removal.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input raw data
        metadata : dict
            Metadata from previous steps

        Returns
        -------
        raw : mne.io.BaseRaw
            Zapline-processed raw data (in-place modification)
        step_metadata : dict
            Zapline metadata including components removed
        """
        # Import pyzaplineplus
        try:
            from pyzaplineplus import zapline_plus
        except ImportError:
            raise ImportError(
                "pyzaplineplus not installed. Install with: pip install pyzaplineplus"
            )

        # Get EEG channels
        if isinstance(self.picks, str):
            picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        else:
            picks = [raw.ch_names.index(ch) for ch in self.picks if ch in raw.ch_names]

        if len(picks) < 2:
            if self.verbose:
                print("Zapline: Skipping (< 2 EEG channels)")
            return raw, {
                'applied': False,
                'reason': 'insufficient_channels',
                'n_channels': len(picks)
            }

        # Get data
        data = raw.get_data(picks=picks)  # (n_channels, n_times)
        sfreq = raw.info['sfreq']

        # Check data duration
        duration = data.shape[1] / sfreq
        if duration < 10:
            if self.verbose:
                print(f"Zapline: Warning - short duration ({duration:.1f}s). Recommend > 30s for best results")

        # Apply Zapline (always print status to console)
        if self.fline is None:
            print(f"Zapline: Auto-detecting line noise frequency...")
        else:
            print(f"Zapline: Removing {self.fline} Hz line noise...")
        print(f"  Channels: {len(picks)}")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Mode: {'adaptive' if self.adaptive else 'fixed'}")
        print(f"  Detection threshold: {self.noiseCompDetectSigma}σ")
        print(f"  Fallback components: {self.fixedNremove}")

        # Transpose for pyzaplineplus: (n_times, n_channels)
        data_T = data.T

        # Apply Zapline-Plus
        # zapline_plus expects (n_times, n_channels)
        # Returns: (cleaned_data, zapline_config, analytics, plots)
        result = zapline_plus(
            data_T,
            sampling_rate=sfreq,
            noisefreqs=self.noisefreqs,
            noiseCompDetectSigma=self.noiseCompDetectSigma,
            fixedNremove=self.fixedNremove,
            plotResults=False,  # Disable plotting
        )
        cleaned_data = result[0]
        info_dict = result[1] if len(result) > 1 else {}

        # Transpose back: (n_channels, n_times)
        cleaned_data = cleaned_data.T

        # Update raw data in-place
        raw._data[picks, :] = cleaned_data

        # Extract info from pyzaplineplus output
        n_components_removed = info_dict.get('n_removed_components', 1) if info_dict else 1

        # Get detected frequencies
        detected_freqs = info_dict.get('noisefreqs', []) if info_dict else []
        detected_freq = detected_freqs[0] if len(detected_freqs) > 0 else self.fline

        # Check if auto-detection was used
        auto_detected = info_dict.get('automaticFreqDetection', False) if info_dict else False

        # Calculate noise reduction
        noise_power_before = np.var(data)
        noise_power_after = np.var(cleaned_data)
        noise_reduction_db = 10 * np.log10(noise_power_before / noise_power_after)

        # Calculate line noise power at detected frequency
        from scipy import signal
        freqs, psd_before = signal.welch(data, fs=sfreq, nperseg=min(self.nfft, data.shape[1]))
        _, psd_after = signal.welch(cleaned_data, fs=sfreq, nperseg=min(self.nfft, data.shape[1]))

        # Find power at line frequency (use detected freq, or default to 60 Hz if None)
        line_freq_for_calc = detected_freq if detected_freq is not None else 60.0
        line_idx = np.argmin(np.abs(freqs - line_freq_for_calc))
        line_power_before = np.mean(psd_before[:, line_idx])
        line_power_after = np.mean(psd_after[:, line_idx])
        line_reduction_db = 10 * np.log10(line_power_before / line_power_after)

        # Always print results to console
        if auto_detected:
            print(f"  ✓ Detected line frequency: {detected_freq} Hz")
        if len(detected_freqs) > 1:
            print(f"  ✓ Additional frequencies: {[f'{f:.1f}' for f in detected_freqs[1:]]}")
        print(f"  Components removed: {n_components_removed}")
        print(f"  Overall noise reduction: {noise_reduction_db:.1f} dB")
        print(f"  Line noise ({detected_freq} Hz) reduction: {line_reduction_db:.1f} dB")

        # Build metadata
        step_metadata = {
            'fline': self.fline,
            'detected_freq': float(detected_freq) if detected_freq is not None else None,
            'detected_freqs': [float(f) for f in detected_freqs],
            'auto_detected': auto_detected,
            'n_components_removed': n_components_removed,
            'n_channels_processed': len(picks),
            'duration_seconds': duration,
            'noise_reduction_db': float(noise_reduction_db),
            'line_reduction_db': float(line_reduction_db),
            'adaptive': self.adaptive,
            'applied': True,
        }

        return raw, step_metadata

    def check_dependencies(self) -> None:
        """Fail at construction when pyzaplineplus is missing.

        Never silently no-op the line-noise step: a missing backend
        must fail the run before any data is touched.
        """
        try:
            from pyzaplineplus import zapline_plus  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"ZaplineStep requires pyzaplineplus, which is not importable ({e}). "
                "Install with: pip install pyzaplineplus"
            ) from e

    def validate_inputs(self, raw: mne.io.BaseRaw) -> bool:
        """
        Validate that Zapline can be applied to this data.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data to validate

        Returns
        -------
        valid : bool
            True if validation passes

        Raises
        ------
        ValueError
            If validation fails
        """
        # Check sampling rate
        sfreq = raw.info['sfreq']
        nyquist = sfreq / 2

        if self.fline is not None and self.fline >= nyquist:
            raise ValueError(
                f"Line frequency {self.fline} Hz exceeds Nyquist frequency {nyquist} Hz. "
                f"Cannot remove line noise above Nyquist limit."
            )

        # Check sufficient channels
        picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        if len(picks) < 2:
            raise ValueError(
                f"Zapline requires at least 2 EEG channels, found {len(picks)}"
            )

        return True

    def get_config(self) -> Dict[str, Any]:
        """Get step configuration."""
        config = super().get_config()
        config.update({
            'fline': self.fline,
            'nremove': self.nremove,
            'nfft': self.nfft,
            'adaptive': self.adaptive,
            'noisefreqs': self.noisefreqs,
            'noiseCompDetectSigma': self.noiseCompDetectSigma,
            'fixedNremove': self.fixedNremove,
        })
        return config
