"""
Artifact Subspace Reconstruction (ASR) step.

Supports both eegprep (EEGLab port) and asrpy implementations.

Author: EEGCPM Development Team
Created: 2025-12
"""

from typing import Any, Dict, Tuple
import mne
import numpy as np

from .base import ProcessingStep


class ASRStep(ProcessingStep):
    """
    Artifact Subspace Reconstruction step.

    ASR identifies and removes high-amplitude artifacts by reconstructing
    corrupted signal subspaces. Two implementations are supported:
    - eegprep: Port of EEGLab's ASR (recommended, more robust)
    - asrpy: Standalone Python implementation

    The EEGLab recommended workflow uses two-stage ASR:
    1. Mild ASR (cutoff=40) before ICA - preserves artifacts for ICA to detect
    2. Aggressive ASR (cutoff=20) after ICA - cleans remaining artifacts

    Parameters
    ----------
    cutoff : float
        ASR cutoff parameter (standard deviations). Lower = more aggressive.
        Recommended: 40 for mild, 20 for aggressive
    mode : str
        Processing mode: 'mild', 'aggressive', or 'single'
        - 'mild': Preserves artifacts for later ICA detection (cutoff ~40)
        - 'aggressive': Removes more artifacts (cutoff ~20)
        - 'single': User-specified cutoff
    method : str
        Implementation: 'eegprep' (default, recommended) or 'asrpy'
    window_length : float
        Window length for ASR processing (seconds, default: 0.5)
    train_duration : float
        Duration of clean data for calibration (seconds, default: 60)
    max_bad_chans : float
        Maximum fraction of bad channels during calibration (default: 0.1)

    Examples
    --------
    Mild ASR before ICA (EEGLab workflow):
    >>> step = ASRStep(cutoff=40, mode='mild', method='eegprep')

    Aggressive ASR after ICA (EEGLab workflow):
    >>> step = ASRStep(cutoff=20, mode='aggressive', method='eegprep')

    Single-stage ASR:
    >>> step = ASRStep(cutoff=20, mode='single', method='eegprep')

    Using asrpy implementation:
    >>> step = ASRStep(cutoff=20, method='asrpy')
    """

    name = "asr"
    version = "1.0"

    def __init__(
        self,
        cutoff: float = 20.0,
        mode: str = 'single',
        method: str = 'eegprep',
        window_length: float = 0.5,
        train_duration: float = 60.0,
        max_bad_chans: float = 0.1,
        enabled: bool = True,
    ):
        """Initialize ASR step."""
        super().__init__(enabled=enabled)

        self.cutoff = cutoff
        self.mode = mode
        self.method = method
        self.window_length = window_length
        self.train_duration = train_duration
        self.max_bad_chans = max_bad_chans

        # Validate method
        if method not in ['eegprep', 'asrpy']:
            raise ValueError(f"Unknown ASR method: {method}. Use 'eegprep' or 'asrpy'")

    def process(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any]
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        """
        Apply ASR.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input raw data
        metadata : dict
            Metadata from previous steps

        Returns
        -------
        raw : mne.io.BaseRaw
            ASR-processed raw data
        step_metadata : dict
            ASR metadata
        """
        # Check for NaN/Inf before ASR
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        if len(eeg_picks) == 0:
            return raw, {'skipped': True, 'reason': 'no_eeg_channels'}

        data_sample = raw.get_data(picks=eeg_picks, start=0, stop=min(1000, raw.n_times))

        if not np.isfinite(data_sample).all():
            return raw, {
                'skipped': True,
                'reason': 'data_contains_nan_inf',
                'error': 'Data contains NaN/Inf values. Skipping ASR.'
            }

        # Apply ASR based on method
        if self.method == 'eegprep':
            raw_cleaned, step_metadata = self._apply_eegprep_asr(raw)
        else:  # asrpy
            raw_cleaned, step_metadata = self._apply_asrpy_asr(raw)

        # Add common metadata
        step_metadata.update({
            'cutoff': self.cutoff,
            'mode': self.mode,
            'method': self.method,
        })

        return raw_cleaned, step_metadata

    def _apply_eegprep_asr(self, raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, Dict]:
        """
        Apply ASR using eegprep (EEGLab port).

        This is the recommended method as it's a direct port of EEGLab's
        mature ASR implementation.
        """
        try:
            import eegprep
        except ImportError:
            raise ImportError(
                "eegprep not installed. Install with: pip install eegprep"
            )

        # Convert MNE Raw to EEGLab format
        eeg = eegprep.eeg_mne2eeg(raw)

        # Apply ASR
        # Note: eegprep.clean_artifacts() handles ASR via BurstCriterion parameter
        # We disable other preprocessing steps (flatline, highpass, channels, line noise)
        result = eegprep.clean_artifacts(
            eeg,
            BurstCriterion=self.cutoff,               # ASR cutoff
            WindowCriterion=self.window_length,       # Window length
            FlatlineCriterion='off',                  # Already handled
            Highpass='off',                           # Already filtered
            ChannelCriterion='off',                   # Already handled bad channels
            LineNoiseCriterion='off',                 # Optional, can enable if needed
        )

        eeg_cleaned = result[0]

        # Convert back to MNE Raw
        # IMPORTANT: eegprep stores data in µV, MNE expects Volts
        # eeg_mne2eeg() multiplied by 1e6 (V → µV), so we must divide by 1e6 here
        data = eeg_cleaned['data'] * 1e-6  # µV → V
        sfreq = eeg_cleaned['srate']
        ch_names = [ch['labels'] for ch in eeg_cleaned['chanlocs']]

        # Store original montage for restoration
        original_montage = raw.get_montage()
        montage_restored = False

        # Check for NaN/Inf values (indicates numerical problems in ASR)
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()

        nan_inf_metadata = {}
        annotations_added = []

        if has_nan or has_inf:
            # Find bad regions (any sample with NaN/Inf across any channel)
            nan_mask = np.isnan(data)
            inf_mask = np.isinf(data)
            bad_mask = nan_mask | inf_mask

            # Find bad time points (any channel has NaN/Inf)
            bad_timepoints = np.any(bad_mask, axis=0)

            n_bad_samples = np.sum(bad_timepoints)
            n_total_samples = len(bad_timepoints)
            pct_bad = 100 * n_bad_samples / n_total_samples

            # Per-channel statistics
            bad_per_channel = np.sum(bad_mask, axis=1)
            affected_channels = np.where(bad_per_channel > 0)[0]

            nan_inf_metadata = {
                'has_nan_inf': True,
                'n_nan_samples': int(np.sum(nan_mask)),
                'n_inf_samples': int(np.sum(inf_mask)),
                'n_bad_timepoints': int(n_bad_samples),
                'pct_bad_samples': float(pct_bad),
                'n_affected_channels': int(len(affected_channels)),
                'affected_channel_indices': affected_channels.tolist(),
            }

            # Replace NaN/Inf with zeros temporarily (for data integrity)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Create new Raw object
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw_cleaned = mne.io.RawArray(data, info, verbose=False)

        # Preserve original annotations/events - CRITICAL for epoching later
        if raw.annotations is not None and len(raw.annotations) > 0:
            raw_cleaned.set_annotations(raw.annotations)

        # Restore montage - CRITICAL for RANSAC, ICLabel, and spatial operations
        if original_montage is not None:
            try:
                raw_cleaned.set_montage(original_montage, on_missing='warn', verbose=False)
                montage_restored = True
            except Exception as e:
                # Log montage restoration failure - this is critical!
                import warnings
                warnings.warn(
                    f"Failed to restore montage after ASR: {e}. "
                    "This will cause RANSAC bad channel detection and ICLabel to fail.",
                    UserWarning
                )

        # Add annotations for bad segments (NaN/Inf regions)
        if has_nan or has_inf:
            # Find contiguous bad segments
            bad_timepoints = np.any(bad_mask, axis=0)
            times = np.arange(len(bad_timepoints)) / sfreq

            # Find transitions (start/stop of bad segments)
            transitions = np.diff(np.concatenate([[0], bad_timepoints.astype(int), [0]]))
            starts = np.where(transitions == 1)[0]
            stops = np.where(transitions == -1)[0]

            # Create annotations for each bad segment
            onsets = []
            durations = []
            descriptions = []

            for start, stop in zip(starts, stops):
                onset = times[start]
                duration = times[stop - 1] - times[start] if stop > 0 else 0
                onsets.append(onset)
                durations.append(duration)
                descriptions.append('BAD_asr_naninf')
                annotations_added.append({
                    'onset': float(onset),
                    'duration': float(duration),
                    'description': 'BAD_asr_naninf'
                })

            if onsets:
                # Add to raw_cleaned annotations
                new_annotations = mne.Annotations(
                    onset=onsets,
                    duration=durations,
                    description=descriptions,
                    orig_time=raw_cleaned.info['meas_date']
                )
                raw_cleaned.set_annotations(
                    raw_cleaned.annotations + new_annotations if raw_cleaned.annotations else new_annotations
                )

                nan_inf_metadata['n_annotations_added'] = len(onsets)
                nan_inf_metadata['total_bad_duration_s'] = float(np.sum(durations))

        metadata = {
            'applied': True,
            'implementation': 'eegprep',
            **nan_inf_metadata,  # Include NaN/Inf statistics if any
            'n_channels_before': len(raw.ch_names),
            'n_channels_after': len(raw_cleaned.ch_names),
            'montage_restored': montage_restored,
        }

        return raw_cleaned, metadata

    def _apply_asrpy_asr(self, raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, Dict]:
        """
        Apply ASR using asrpy (standalone Python implementation).

        Note: asrpy may have numerical stability issues with very poor quality data.
        eegprep is generally more robust.
        """
        try:
            from asrpy import ASR
        except ImportError:
            raise ImportError(
                "asrpy not installed. Install with: pip install asrpy"
            )

        # Initialize ASR
        asr = ASR(
            sfreq=raw.info['sfreq'],
            cutoff=self.cutoff,
            max_bad_chans=self.max_bad_chans
        )

        # Fit on calibration window
        train_samples = int(self.train_duration * raw.info['sfreq'])
        train_samples = min(train_samples, raw.n_times)

        asr.fit(raw, picks='eeg', start=0, stop=train_samples)

        # Transform entire dataset
        raw_cleaned = asr.transform(raw, picks='eeg')

        metadata = {
            'applied': True,
            'implementation': 'asrpy',
            'train_duration': self.train_duration,
            'train_samples': train_samples,
        }

        return raw_cleaned, metadata

    def validate_inputs(self, raw: mne.io.BaseRaw) -> bool:
        """Validate inputs."""
        # Check we have EEG channels
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        if len(eeg_picks) == 0:
            raise ValueError("No EEG channels available for ASR")

        # Check calibration data length
        required_samples = int(self.train_duration * raw.info['sfreq'])
        if raw.n_times < required_samples:
            raise ValueError(
                f"Data too short for ASR calibration. "
                f"Need {self.train_duration}s ({required_samples} samples), "
                f"have {raw.times[-1]:.1f}s ({raw.n_times} samples)"
            )

        return True

    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        config = super().get_config()
        config.update({
            'cutoff': self.cutoff,
            'mode': self.mode,
            'method': self.method,
            'train_duration': self.train_duration,
        })
        return config
