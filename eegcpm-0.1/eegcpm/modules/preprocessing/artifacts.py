"""
Automatic Artifact Annotation Module

Provides automatic detection and annotation of artifact segments in continuous
EEG data using amplitude, gradient, and flatline-based methods.

Artifact annotations improve ICA quality by excluding bad segments from fitting
and help with epoch rejection in event-related analyses.

Author: EEGCPM Development Team
Created: 2025-01
"""

from typing import List, Optional, Tuple
import mne
import numpy as np

from .steps.asr import append_bad_annotations


class ArtifactAnnotator:
    """
    Automatic artifact detection and annotation for continuous EEG.

    Methods:
    - Amplitude-based: Segments exceeding voltage thresholds
    - Gradient-based: Segments with rapid voltage changes
    - Flatline detection: Segments with near-zero variance
    - Muscle artifact: High-frequency power bursts (optional)

    Parameters
    ----------
    amplitude_threshold : float or dict, optional
        Peak-to-peak amplitude threshold in Volts.
        Can be float (e.g., 150e-6 for 150 µV) or dict per channel type.
        Example: {'eeg': 150e-6, 'eog': 250e-6}
    gradient_threshold : float, optional
        Maximum allowed gradient (voltage change per sample) in V/sample.
        Example: 75e-6 for 75 µV/sample
    flatline_duration : float, optional
        Minimum duration (seconds) of flatline to annotate.
        Example: 5.0 seconds
    muscle_threshold : float, optional
        Z-score threshold for muscle artifact detection (high-freq power).
        Set to None to disable. Example: 4.0
    freq_muscle : tuple, optional
        Frequency range for muscle artifact detection.
        Default: (110, 140) Hz
    min_duration : float, default=0.1
        Minimum duration (seconds) for annotations.
        Shorter artifacts are ignored.

    Examples
    --------
    Basic artifact detection:
    >>> annotator = ArtifactAnnotator(
    ...     amplitude_threshold=150e-6,
    ...     gradient_threshold=75e-6,
    ...     flatline_duration=5.0
    ... )
    >>> raw_annotated = annotator.annotate(raw)
    >>> print(f"Found {len(raw_annotated.annotations)} artifact segments")

    Conservative (fewer false positives):
    >>> annotator = ArtifactAnnotator(
    ...     amplitude_threshold=200e-6,
    ...     gradient_threshold=100e-6,
    ...     flatline_duration=10.0
    ... )

    Aggressive (catch more artifacts):
    >>> annotator = ArtifactAnnotator(
    ...     amplitude_threshold=100e-6,
    ...     gradient_threshold=50e-6,
    ...     flatline_duration=3.0,
    ...     muscle_threshold=3.0
    ... )
    """

    def __init__(
        self,
        amplitude_threshold: Optional[float] = None,
        gradient_threshold: Optional[float] = None,
        flatline_duration: Optional[float] = None,
        muscle_threshold: Optional[float] = None,
        freq_muscle: Tuple[float, float] = (110, 140),
        min_duration: float = 0.1,
    ):
        """Initialize artifact annotator."""
        self.amplitude_threshold = amplitude_threshold
        self.gradient_threshold = gradient_threshold
        self.flatline_duration = flatline_duration
        self.muscle_threshold = muscle_threshold
        self.freq_muscle = freq_muscle
        self.min_duration = min_duration

    def annotate(
        self,
        raw: mne.io.BaseRaw,
        copy: bool = True
    ) -> mne.io.BaseRaw:
        """
        Annotate artifacts in raw data.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw EEG data
        copy : bool, default=True
            Operate on copy (recommended)

        Returns
        -------
        mne.io.BaseRaw
            Raw data with artifact annotations added

        Notes
        -----
        Annotations are added with descriptions:
        - 'BAD_amplitude': Amplitude threshold exceeded
        - 'BAD_gradient': Gradient threshold exceeded
        - 'BAD_flatline': Flatline detected
        - 'BAD_muscle': Muscle artifact detected
        """
        if copy:
            raw = raw.copy()

        # Detect amplitude-based artifacts
        if self.amplitude_threshold is not None:
            raw = self._annotate_amplitude(raw)

        # Detect gradient-based artifacts
        if self.gradient_threshold is not None:
            raw = self._annotate_gradient(raw)

        # Detect flatlines
        if self.flatline_duration is not None:
            raw = self._annotate_flatline(raw)

        # Detect muscle artifacts
        if self.muscle_threshold is not None:
            raw = self._annotate_muscle(raw)

        return raw

    def _annotate_amplitude(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """
        Annotate segments exceeding peak-to-peak amplitude threshold.

        Uses sliding window peak-to-peak detection. MNE's annotate_amplitude
        actually uses sample-to-sample gradients, not true peak-to-peak amplitude.

        Windows where peak-to-peak exceeds threshold in ANY channel are marked.
        """
        # Convert threshold to float (use EEG threshold)
        if isinstance(self.amplitude_threshold, dict):
            threshold = self.amplitude_threshold.get('eeg', 150e-6)
        else:
            threshold = self.amplitude_threshold

        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')

        if len(eeg_picks) == 0:
            return raw

        data = raw.get_data(picks=eeg_picks)
        sfreq = raw.info['sfreq']

        # Use 100ms sliding windows for peak-to-peak calculation
        window_duration = 0.1  # 100ms windows
        window_samples = int(window_duration * sfreq)
        step_samples = window_samples // 2  # 50% overlap

        n_samples = data.shape[1]
        bad_mask = np.zeros(n_samples, dtype=bool)

        # Slide window across data
        for start in range(0, n_samples - window_samples + 1, step_samples):
            end = start + window_samples
            window_data = data[:, start:end]

            # Compute peak-to-peak per channel
            ptp = np.ptp(window_data, axis=1)

            # Mark as bad if ANY channel exceeds threshold
            if np.any(ptp > threshold):
                bad_mask[start:end] = True

        # Convert to annotations
        annotations = self._mask_to_annotations(
            bad_mask,
            sfreq,
            description='BAD_amplitude'
        )

        if len(annotations) > 0:
            # B3-K9: append-only. The four artifact sites previously used
            # the uncorrected `set_annotations(raw.annotations + annot)`
            # recipe, which shifts every existing annotation by
            # `first_time` whenever `meas_date is None` and
            # `first_samp != 0` (mne 1.13.2) — corrupting the timeline on
            # re-processed FIF derivatives. `append_bad_annotations`
            # applies the offset recipe with the frame correction, so
            # every pre-existing annotation keeps its sample.
            append_bad_annotations(
                raw,
                annotations[:, 0],
                annotations[:, 1],
                ['BAD_amplitude'] * len(annotations),
            )

        return raw

    def _annotate_gradient(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """
        Annotate segments with excessive gradients (rapid changes).

        Detects artifacts like muscle activity and electrode pops.
        """
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')

        if len(eeg_picks) == 0:
            return raw

        data = raw.get_data(picks=eeg_picks)

        # Compute gradient (first derivative)
        gradient = np.abs(np.diff(data, axis=1))

        # Find timepoints exceeding threshold
        bad_mask = np.any(gradient > self.gradient_threshold, axis=0)

        # Convert to annotations
        annotations = self._mask_to_annotations(
            bad_mask,
            raw.info['sfreq'],
            description='BAD_gradient'
        )

        if len(annotations) > 0:
            # B3-K9: append-only (see _annotate_amplitude).
            append_bad_annotations(
                raw,
                annotations[:, 0],
                annotations[:, 1],
                ['BAD_gradient'] * len(annotations),
            )

        return raw

    def _annotate_flatline(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """
        Annotate flatline segments (near-zero variance).

        Detects temporary amplifier disconnections or saturation where MULTIPLE
        channels go flatline simultaneously. Single-channel flatlines (dead electrodes)
        are handled separately by bad channel detection, not segment annotation.

        A segment is marked as BAD_flatline only if >10% of channels are flatline,
        indicating a recording problem rather than individual bad channels.
        """
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')

        if len(eeg_picks) == 0:
            return raw

        data = raw.get_data(picks=eeg_picks)
        sfreq = raw.info['sfreq']
        n_channels = len(eeg_picks)

        # Window size in samples
        window_samples = int(self.flatline_duration * sfreq)

        # Threshold: require >10% of channels to be flatline to mark as bad segment
        # This avoids marking entire recording just because of a few bad channels
        min_flatline_channels = max(1, int(0.1 * n_channels))

        # Sliding window variance
        n_windows = data.shape[1] - window_samples + 1
        bad_mask = np.zeros(data.shape[1], dtype=bool)

        # Use larger step size for efficiency (check every 1 second)
        step = int(sfreq)  # 1 second steps
        for i in range(0, n_windows, step):
            window = data[:, i:i+window_samples]
            # Count channels with near-zero variance
            variances = np.var(window, axis=1)
            n_flatline = np.sum(variances < 1e-20)

            # Mark as bad only if multiple channels are flatline
            if n_flatline >= min_flatline_channels:
                bad_mask[i:i+window_samples] = True

        # Convert to annotations
        annotations = self._mask_to_annotations(
            bad_mask,
            sfreq,
            description='BAD_flatline'
        )

        if len(annotations) > 0:
            # B3-K9: append-only (see _annotate_amplitude).
            append_bad_annotations(
                raw,
                annotations[:, 0],
                annotations[:, 1],
                ['BAD_flatline'] * len(annotations),
            )

        return raw

    def _annotate_muscle(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """
        Annotate muscle artifacts using high-frequency power.

        Muscle artifacts typically show increased power in 110-140 Hz range.
        """
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')

        if len(eeg_picks) == 0:
            return raw

        # Bandpass filter in muscle frequency range
        raw_muscle = raw.copy().filter(
            l_freq=self.freq_muscle[0],
            h_freq=self.freq_muscle[1],
            picks=eeg_picks,
            verbose=False
        )

        # Compute envelope (absolute value of analytic signal)
        data_muscle = np.abs(raw_muscle.get_data(picks=eeg_picks))

        # Average across channels
        muscle_power = np.mean(data_muscle, axis=0)

        # Z-score
        z_scores = (muscle_power - np.mean(muscle_power)) / np.std(muscle_power)

        # Threshold
        bad_mask = z_scores > self.muscle_threshold

        # Convert to annotations
        annotations = self._mask_to_annotations(
            bad_mask,
            raw.info['sfreq'],
            description='BAD_muscle'
        )

        if len(annotations) > 0:
            # B3-K9: append-only (see _annotate_amplitude).
            append_bad_annotations(
                raw,
                annotations[:, 0],
                annotations[:, 1],
                ['BAD_muscle'] * len(annotations),
            )

        return raw

    def _mask_to_annotations(
        self,
        mask: np.ndarray,
        sfreq: float,
        description: str
    ) -> np.ndarray:
        """
        Convert boolean mask to annotation onset/duration pairs.

        Parameters
        ----------
        mask : np.ndarray
            Boolean mask (True = artifact)
        sfreq : float
            Sampling frequency
        description : str
            Annotation description

        Returns
        -------
        np.ndarray
            Array of (onset, duration) pairs in seconds
        """
        # Find transitions
        diff = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
        starts = np.where(diff == 1)[0]
        stops = np.where(diff == -1)[0]

        # Convert to seconds
        annotations = []
        for start, stop in zip(starts, stops):
            onset = start / sfreq
            duration = (stop - start) / sfreq

            # Filter by minimum duration
            if duration >= self.min_duration:
                annotations.append([onset, duration])

        return np.array(annotations) if annotations else np.array([]).reshape(0, 2)

    def get_config(self) -> dict:
        """Get annotator configuration as dictionary."""
        return {
            'amplitude_threshold': self.amplitude_threshold,
            'gradient_threshold': self.gradient_threshold,
            'flatline_duration': self.flatline_duration,
            'muscle_threshold': self.muscle_threshold,
            'freq_muscle': self.freq_muscle,
            'min_duration': self.min_duration,
        }


def annotate_artifacts_from_config(
    raw: mne.io.BaseRaw,
    config: dict,
    copy: bool = True
) -> mne.io.BaseRaw:
    """
    Annotate artifacts from configuration dictionary.

    This is a convenience function for pipeline integration.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw EEG data
    config : dict
        Configuration dictionary with keys:
        - 'enabled': Enable artifact annotation (default: True)
        - 'amplitude_threshold': Amplitude threshold in V (optional)
        - 'gradient_threshold': Gradient threshold in V/sample (optional)
        - 'flatline_duration': Flatline duration in seconds (optional)
        - 'muscle_threshold': Muscle artifact z-score threshold (optional)
        - 'min_duration': Minimum annotation duration (default: 0.1)
    copy : bool
        Operate on copy

    Returns
    -------
    mne.io.BaseRaw
        Raw data with artifact annotations

    Examples
    --------
    >>> config = {
    ...     'enabled': True,
    ...     'amplitude_threshold': 150e-6,
    ...     'gradient_threshold': 75e-6,
    ...     'flatline_duration': 5.0
    ... }
    >>> raw_annotated = annotate_artifacts_from_config(raw, config)
    """
    if not config.get('enabled', True):
        return raw if not copy else raw.copy()

    annotator = ArtifactAnnotator(
        amplitude_threshold=config.get('amplitude_threshold'),
        gradient_threshold=config.get('gradient_threshold'),
        flatline_duration=config.get('flatline_duration'),
        muscle_threshold=config.get('muscle_threshold'),
        freq_muscle=config.get('freq_muscle', (110, 140)),
        min_duration=config.get('min_duration', 0.1),
    )

    return annotator.annotate(raw, copy=copy)
