"""
Bad Channel Detection and Interpolation Module

Provides automatic detection of bad EEG channels using multiple methods
and spherical spline interpolation to replace them.

Bad channels can corrupt ICA decomposition and connectivity analyses, so
proper detection and handling is critical for data quality.

Author: EEGCPM Development Team
Created: 2025-01
"""

from typing import List, Optional, Set, Tuple, Union
import mne
import numpy as np
from scipy import stats


class BadChannelDetector:
    """
    Automatic bad channel detection using multiple methods.

    Methods:
    - Variance-based: Channels with extreme variance (too high or too low)
    - Correlation-based: Channels poorly correlated with neighbors
    - RANSAC: RANdom SAmple Consensus for robust detection
    - Deviation: Channels deviating from robust average

    Parameters
    ----------
    methods : list of str
        Detection methods to use: ['variance', 'correlation', 'ransac', 'deviation']
    variance_threshold : float, default=5.0
        Standard deviations from median variance for variance method
    correlation_threshold : float, default=0.4
        Minimum correlation with neighbors for correlation method
    ransac_sample_prop : float, default=0.25
        Proportion of channels to sample in RANSAC
    ransac_corr_threshold : float, default=0.75
        Correlation threshold for RANSAC consensus
    deviation_threshold : float, default=5.0
        Standard deviations from robust mean for deviation method

    Examples
    --------
    Detect bad channels using multiple methods:
    >>> detector = BadChannelDetector(
    ...     methods=['variance', 'correlation'],
    ...     variance_threshold=5.0,
    ...     correlation_threshold=0.4
    ... )
    >>> bad_channels = detector.detect(raw)
    >>> print(f"Found {len(bad_channels)} bad channels: {bad_channels}")

    Conservative detection (fewer false positives):
    >>> detector = BadChannelDetector(
    ...     methods=['variance'],
    ...     variance_threshold=7.0
    ... )

    Aggressive detection (catch more bad channels):
    >>> detector = BadChannelDetector(
    ...     methods=['variance', 'correlation', 'deviation'],
    ...     variance_threshold=3.0
    ... )
    """

    VALID_METHODS = ['variance', 'correlation', 'ransac', 'deviation', 'prep']

    def __init__(
        self,
        methods: List[str] = ['variance', 'correlation'],
        variance_threshold: float = 5.0,
        correlation_threshold: float = 0.4,
        ransac_sample_prop: float = 0.25,
        ransac_corr_threshold: float = 0.75,
        deviation_threshold: float = 5.0,
    ):
        """Initialize bad channel detector."""
        # Validate methods
        invalid = [m for m in methods if m not in self.VALID_METHODS]
        if invalid:
            raise ValueError(
                f"Invalid methods: {invalid}. Valid: {self.VALID_METHODS}"
            )

        self.methods = methods
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.ransac_sample_prop = ransac_sample_prop
        self.ransac_corr_threshold = ransac_corr_threshold
        self.deviation_threshold = deviation_threshold

    def detect(
        self,
        raw: mne.io.BaseRaw,
        return_scores: bool = False
    ) -> Union[List[str], Tuple[List[str], dict]]:
        """
        Detect bad channels using configured methods.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw EEG data
        return_scores : bool, default=False
            If True, also return detection scores for each method

        Returns
        -------
        bad_channels : list of str
            Names of detected bad channels
        scores : dict, optional
            Detection scores per method (if return_scores=True)

        Notes
        -----
        A channel is marked as bad if flagged by ANY of the configured methods.
        For more conservative detection, use fewer methods or higher thresholds.
        """
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])

        if len(eeg_picks) == 0:
            return [] if not return_scores else ([], {})

        bad_channels_sets = []
        scores = {}

        # Run each detection method
        if 'variance' in self.methods:
            bad_var, score_var = self._detect_by_variance(raw, eeg_picks)
            bad_channels_sets.append(bad_var)
            scores['variance'] = score_var

        if 'correlation' in self.methods:
            bad_corr, score_corr = self._detect_by_correlation(raw, eeg_picks)
            bad_channels_sets.append(bad_corr)
            scores['correlation'] = score_corr

        if 'ransac' in self.methods:
            bad_ransac, score_ransac = self._detect_by_ransac(raw, eeg_picks)
            bad_channels_sets.append(bad_ransac)
            scores['ransac'] = score_ransac

        if 'deviation' in self.methods:
            bad_dev, score_dev = self._detect_by_deviation(raw, eeg_picks)
            bad_channels_sets.append(bad_dev)
            scores['deviation'] = score_dev

        if 'prep' in self.methods:
            bad_prep, score_prep = self._detect_by_prep(raw, eeg_picks)
            bad_channels_sets.append(bad_prep)
            scores['prep'] = score_prep

        # Combine: channel is bad if flagged by ANY method
        bad_channels = set()
        for bad_set in bad_channels_sets:
            bad_channels.update(bad_set)

        bad_channels = sorted(list(bad_channels))

        if return_scores:
            return bad_channels, scores
        return bad_channels

    def _detect_by_variance(
        self,
        raw: mne.io.BaseRaw,
        picks: np.ndarray
    ) -> Tuple[Set[str], dict]:
        """Detect channels with extreme variance."""
        data = raw.get_data(picks=picks)
        variances = np.var(data, axis=1)

        # Compute robust statistics
        median_var = np.median(variances)
        mad = np.median(np.abs(variances - median_var))
        # Convert MAD to std (assuming normal distribution)
        robust_std = 1.4826 * mad

        # Find outliers
        z_scores = np.abs((variances - median_var) / (robust_std + 1e-10))
        bad_indices = np.where(z_scores > self.variance_threshold)[0]

        bad_channels = {raw.ch_names[picks[i]] for i in bad_indices}
        scores = {raw.ch_names[picks[i]]: z_scores[i] for i in range(len(picks))}

        return bad_channels, scores

    def _detect_by_correlation(
        self,
        raw: mne.io.BaseRaw,
        picks: np.ndarray
    ) -> Tuple[Set[str], dict]:
        """Detect channels poorly correlated with neighbors."""
        data = raw.get_data(picks=picks)

        # Compute correlation matrix
        corr_matrix = np.corrcoef(data)

        # For each channel, compute median correlation with others
        median_corrs = np.median(corr_matrix, axis=1)

        # Find channels with low correlation
        bad_indices = np.where(median_corrs < self.correlation_threshold)[0]

        bad_channels = {raw.ch_names[picks[i]] for i in bad_indices}
        scores = {raw.ch_names[picks[i]]: median_corrs[i] for i in range(len(picks))}

        return bad_channels, scores

    def _detect_by_ransac(
        self,
        raw: mne.io.BaseRaw,
        picks: np.ndarray
    ) -> Tuple[Set[str], dict]:
        """
        Detect bad channels using RANSAC via Autoreject library.

        This is the method used in MNE-BIDS pipeline for robust bad channel detection.
        RANSAC works by iteratively selecting random subsets of channels, using them
        to predict other channels, and identifying channels that are consistently
        poorly predicted.

        Uses the Autoreject implementation which is optimized for EEG/MEG data.

        References
        ----------
        Jas, M., Engemann, D. A., Bekhti, Y., Raimondo, F., & Gramfort, A. (2017).
        Autoreject: Automated artifact rejection for MEG and EEG data.
        NeuroImage, 159, 417-429.

        Notes
        -----
        Requires montage to be set (for spatial interpolation).
        Returns empty set if montage is not available.
        """
        try:
            from autoreject import Ransac
        except ImportError as e:
            # Never degrade detection silently: a missing backend must
            # fail the run (F17). The step's check_dependencies()
            # preflight catches this at construction; this re-raise
            # covers direct detector use.
            raise ImportError(
                f"RANSAC bad-channel detection requires autoreject ({e}). "
                "Install with: pip install autoreject"
            ) from e

        # RANSAC requires montage for spatial interpolation
        if raw.get_montage() is None:
            print("Warning: RANSAC requires montage to be set. Skipping RANSAC detection.")
            return set(), {}

        # Get EEG picks
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(eeg_picks) < 4:
            # Need at least 4 channels for RANSAC
            return set(), {}

        try:
            # Create temporary epochs for RANSAC (Autoreject requires epochs, not raw)
            # Use short overlapping epochs to cover the full recording
            events = mne.make_fixed_length_events(raw, duration=1.0, overlap=0.5)
            epochs = mne.Epochs(
                raw,
                events,
                tmin=0,
                tmax=1.0,
                baseline=None,
                preload=True,
                verbose=False
            )

            # Create Ransac object
            # Parameters match MNE-BIDS pipeline defaults:
            # - n_jobs=1 for reproducibility
            # - min_channels=0.25 (use 25% of channels for prediction)
            # - min_corr=0.75 (correlation threshold for consensus)
            ransac = Ransac(
                n_resample=50,  # Number of RANSAC iterations
                min_channels=self.ransac_sample_prop,  # Proportion of channels to use
                min_corr=self.ransac_corr_threshold,   # Correlation threshold
                unbroken_time=True,  # Assume continuous time in epochs
                n_jobs=1,  # Single job for reproducibility
                verbose=False
            )

            # Fit RANSAC on epochs
            # This populates ransac.bad_chs_ with detected bad channels
            epochs_clean = ransac.fit_transform(epochs)

            # Get bad channels from RANSAC
            bad_channels = set(ransac.bad_chs_)

            # Create scores dictionary (correlation scores for each channel)
            # Higher score = better channel
            scores = {}
            if hasattr(ransac, 'bad_log_'):
                # bad_log_ contains boolean array (n_iterations x n_channels)
                # Count how often each channel was flagged as bad
                n_bad_iterations = ransac.bad_log_.sum(axis=0)
                n_iterations = ransac.bad_log_.shape[0]

                ch_names = [raw.ch_names[p] for p in eeg_picks]
                for i, ch_name in enumerate(ch_names):
                    # Convert to correlation-like score (1.0 = never bad, 0.0 = always bad)
                    scores[ch_name] = 1.0 - (n_bad_iterations[i] / n_iterations)
            else:
                # Fallback: just mark bad channels as 0, others as 1
                ch_names = [raw.ch_names[p] for p in eeg_picks]
                for ch_name in ch_names:
                    scores[ch_name] = 0.0 if ch_name in bad_channels else 1.0

            return bad_channels, scores

        except Exception as e:
            print(f"Warning: RANSAC detection failed: {e}")
            return set(), {}

    def _detect_by_deviation(
        self,
        raw: mne.io.BaseRaw,
        picks: np.ndarray
    ) -> Tuple[Set[str], dict]:
        """Detect channels deviating from robust average signal."""
        data = raw.get_data(picks=picks)

        # Compute robust average (median across channels at each timepoint)
        robust_avg = np.median(data, axis=0)

        # Compute deviation of each channel from robust average
        deviations = np.mean((data - robust_avg) ** 2, axis=1)

        # Find outliers
        median_dev = np.median(deviations)
        mad = np.median(np.abs(deviations - median_dev))
        robust_std = 1.4826 * mad

        z_scores = np.abs((deviations - median_dev) / (robust_std + 1e-10))
        bad_indices = np.where(z_scores > self.deviation_threshold)[0]

        bad_channels = {raw.ch_names[picks[i]] for i in bad_indices}
        scores = {raw.ch_names[picks[i]]: z_scores[i] for i in range(len(picks))}

        return bad_channels, scores

    def _detect_by_prep(
        self,
        raw: mne.io.BaseRaw,
        picks: np.ndarray
    ) -> Tuple[Set[str], dict]:
        """
        Detect bad channels using PREP pipeline (pyprep).

        PREP (Preprocessing Pipeline) is the gold standard for bad channel detection.
        It combines multiple metrics:
        - Deviation from robust average
        - Correlation with nearby channels
        - Noisiness (high-frequency noise)
        - RANSAC prediction error

        Uses iterative refinement for better accuracy.

        References
        ----------
        Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K. M., & Robbins, K. A. (2015).
        The PREP pipeline: standardized preprocessing for large-scale EEG analysis.
        Frontiers in neuroinformatics, 9, 16.

        Notes
        -----
        Requires montage to be set.
        Returns empty set if montage not available or pyprep not installed.
        """
        try:
            from pyprep.find_noisy_channels import NoisyChannels
        except ImportError as e:
            # Never degrade detection silently (F17): a missing
            # backend must fail the run.
            raise ImportError(
                f"PREP bad-channel detection requires pyprep ({e}). "
                "Install with: pip install pyprep"
            ) from e

        # PREP requires montage
        if raw.get_montage() is None:
            print("Warning: PREP requires montage to be set. Skipping PREP detection.")
            return set(), {}

        try:
            # Create NoisyChannels object
            # Note: NoisyChannels modifies raw in-place, so work on a copy
            raw_copy = raw.copy()

            nc = NoisyChannels(raw_copy, do_detrend=False)

            # Run all PREP bad channel detection methods
            # This finds:
            # - bad_by_deviation
            # - bad_by_correlation
            # - bad_by_hf_noise (high-frequency noise)
            # - bad_by_ransac
            nc.find_all_bads(ransac=True, channel_wise=True, max_chunk_size=None)

            # Get all bad channels detected
            bad_channels = set(nc.get_bads())

            # Create scores dictionary
            # Higher score = worse channel
            scores = {}
            all_channels = [raw.ch_names[p] for p in picks]

            for ch in all_channels:
                # Score based on which methods flagged it
                score = 0.0
                if hasattr(nc, 'bad_by_deviation') and ch in nc.bad_by_deviation:
                    score += 0.25
                if hasattr(nc, 'bad_by_correlation') and ch in nc.bad_by_correlation:
                    score += 0.25
                if hasattr(nc, 'bad_by_hf_noise') and ch in nc.bad_by_hf_noise:
                    score += 0.25
                if hasattr(nc, 'bad_by_ransac') and ch in nc.bad_by_ransac:
                    score += 0.25
                scores[ch] = score

            return bad_channels, scores

        except Exception as e:
            print(f"Warning: PREP detection failed: {e}")
            return set(), {}

    def get_config(self) -> dict:
        """Get detector configuration as dictionary."""
        return {
            'methods': self.methods,
            'variance_threshold': self.variance_threshold,
            'correlation_threshold': self.correlation_threshold,
            'ransac_sample_prop': self.ransac_sample_prop,
            'ransac_corr_threshold': self.ransac_corr_threshold,
            'deviation_threshold': self.deviation_threshold,
        }


def detect_and_interpolate_bad_channels(
    raw: mne.io.BaseRaw,
    config: dict,
    copy: bool = True
) -> Tuple[mne.io.BaseRaw, List[str]]:
    """
    Detect bad channels and interpolate them using spherical splines.

    This is a convenience function combining detection and interpolation.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw EEG data
    config : dict
        Configuration dictionary with keys:
        - 'auto_detect': Enable automatic detection (default: True)
        - 'methods': Detection methods (default: ['variance', 'correlation'])
        - 'variance_threshold': Threshold for variance method (default: 5.0)
        - 'correlation_threshold': Threshold for correlation (default: 0.4)
        - 'interpolate': Apply interpolation (default: True)
        - 'manual_bads': List of manually identified bad channels (optional)
    copy : bool
        Operate on copy

    Returns
    -------
    raw : mne.io.BaseRaw
        Processed raw data (with interpolated channels)
    bad_channels : list of str
        Names of detected/interpolated bad channels

    Examples
    --------
    >>> config = {
    ...     'auto_detect': True,
    ...     'methods': ['variance', 'correlation'],
    ...     'variance_threshold': 5.0,
    ...     'interpolate': True
    ... }
    >>> raw_clean, bad_chs = detect_and_interpolate_bad_channels(raw, config)
    >>> print(f"Interpolated {len(bad_chs)} channels: {bad_chs}")
    """
    if copy:
        raw = raw.copy()

    bad_channels = []

    # Add manually specified bad channels
    if 'manual_bads' in config and config['manual_bads']:
        manual_bads = config['manual_bads']
        if isinstance(manual_bads, str):
            manual_bads = [manual_bads]
        bad_channels.extend(manual_bads)

    # Automatic detection
    if config.get('auto_detect', True):
        detector = BadChannelDetector(
            methods=config.get('methods', ['variance', 'correlation']),
            variance_threshold=config.get('variance_threshold', 5.0),
            correlation_threshold=config.get('correlation_threshold', 0.4),
            ransac_sample_prop=config.get('ransac_sample_prop', 0.25),
            ransac_corr_threshold=config.get('ransac_corr_threshold', 0.75),
            deviation_threshold=config.get('deviation_threshold', 5.0),
        )

        auto_bad = detector.detect(raw)
        bad_channels.extend(auto_bad)

    # Remove duplicates
    bad_channels = sorted(list(set(bad_channels)))

    # Mark as bad
    raw.info['bads'] = bad_channels

    # Check for zero-variance channels (these cannot be interpolated)
    # These are likely reference/ground channels or completely disconnected
    zero_variance_channels = []
    if len(bad_channels) > 0:
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        data = raw.get_data(picks=eeg_picks)
        variances = np.var(data, axis=1)

        for ch_name in bad_channels:
            if ch_name in raw.ch_names:
                ch_idx = raw.ch_names.index(ch_name)
                if ch_idx in eeg_picks:
                    var_idx = np.where(eeg_picks == ch_idx)[0]
                    if len(var_idx) > 0 and variances[var_idx[0]] < 1e-20:
                        zero_variance_channels.append(ch_name)

    # Separate interpolatable and zero-variance channels
    interpolatable = [ch for ch in bad_channels if ch not in zero_variance_channels]

    # Interpolate if requested (only for non-zero variance channels)
    if config.get('interpolate', True) and len(interpolatable) > 0:
        # Need montage for interpolation
        if raw.get_montage() is None:
            try:
                montage = mne.channels.make_standard_montage('standard_1020')
                raw.set_montage(montage, on_missing='warn', verbose=False)
            except Exception as e:
                print(f"Warning: Could not set montage for interpolation: {e}")
                # Drop zero-variance channels instead
                if len(zero_variance_channels) > 0:
                    raw.drop_channels(zero_variance_channels)
                return raw, bad_channels

        # Set only interpolatable channels as bad
        raw.info['bads'] = interpolatable

        try:
            raw.interpolate_bads(reset_bads=True, verbose=False)
        except Exception as e:
            print(f"Warning: Interpolation failed: {e}")
            # If interpolation fails, just drop the bad channels
            if len(interpolatable) > 0:
                raw.drop_channels(interpolatable)

    # Drop zero-variance channels (cannot be interpolated)
    if len(zero_variance_channels) > 0:
        # Filter to only existing channels
        existing_zero_var = [ch for ch in zero_variance_channels if ch in raw.ch_names]
        if len(existing_zero_var) > 0:
            raw.drop_channels(existing_zero_var)

    return raw, bad_channels
