"""
Bad channel detection step.

Author: EEGCPM Development Team
Created: 2025-12
"""

from typing import Any, Dict, List, Tuple
import mne

from .base import ProcessingStep
from ..bad_channels import BadChannelDetector


class BadChannelDetectionStep(ProcessingStep):
    """
    Bad channel detection step.

    Detects bad channels using various methods (RANSAC, correlation, variance).
    Can either mark channels as bad or immediately interpolate them.

    Parameters
    ----------
    method : str or list of str
        Detection method(s): 'ransac', 'correlation', 'variance', 'deviation'
    mark_only : bool
        If True, only mark channels as bad (don't interpolate)
        If False, interpolate immediately (default: False)
    correlation_threshold : float
        Correlation threshold for detection (default: 0.4)
    variance_threshold : float
        Variance threshold in standard deviations (default: 5.0)
    ransac_sample_prop : float
        Proportion of samples for RANSAC (default: 0.25)
    ransac_corr_threshold : float
        RANSAC correlation threshold (default: 0.75)
    manual_bads : list of str
        Manually specified bad channels

    Examples
    --------
    RANSAC detection (mark only, don't interpolate):
    >>> step = BadChannelDetectionStep(
    ...     method='ransac',
    ...     mark_only=True
    ... )

    Multiple methods with immediate interpolation:
    >>> step = BadChannelDetectionStep(
    ...     method=['variance', 'correlation', 'ransac'],
    ...     mark_only=False
    ... )
    """

    name = "bad_channels"
    version = "1.0"

    def __init__(
        self,
        method: str = 'ransac',
        mark_only: bool = False,
        drop: bool = False,  # NEW: drop bad channels instead of interpolate
        correlation_threshold: float = 0.4,
        variance_threshold: float = 5.0,
        ransac_sample_prop: float = 0.25,
        ransac_corr_threshold: float = 0.75,
        deviation_threshold: float = 5.0,
        manual_bads: List[str] = None,
        enabled: bool = True,
    ):
        """Initialize bad channel detection step."""
        super().__init__(enabled=enabled)

        self.method = method if isinstance(method, list) else [method]
        self.mark_only = mark_only
        self.drop = drop
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.ransac_sample_prop = ransac_sample_prop
        self.ransac_corr_threshold = ransac_corr_threshold
        self.deviation_threshold = deviation_threshold
        self.manual_bads = manual_bads or []

    def process(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any]
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        """
        Detect bad channels.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input raw data
        metadata : dict
            Metadata from previous steps

        Returns
        -------
        raw : mne.io.BaseRaw
            Data with bad channels marked or interpolated
        step_metadata : dict
            Detection results
        """
        # Debug: check montage before detection
        has_montage = raw.get_montage() is not None
        n_channels = len(raw.ch_names)
        print(f"[BAD_CHANNELS DEBUG] N channels: {n_channels}, Has montage: {has_montage}")

        if has_montage:
            import numpy as np
            eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            bad_pos_count = 0
            for pick in eeg_picks[:5]:  # Check first 5
                ch_info = raw.info['chs'][pick]
                loc = ch_info['loc'][:3]
                if not np.isfinite(loc).all():
                    bad_pos_count += 1
            print(f"[BAD_CHANNELS DEBUG] First 5 channels have {bad_pos_count} with NaN/Inf positions")

        # Create detector
        detector = BadChannelDetector(
            methods=self.method,
            variance_threshold=self.variance_threshold,
            correlation_threshold=self.correlation_threshold,
            ransac_sample_prop=self.ransac_sample_prop,
            ransac_corr_threshold=self.ransac_corr_threshold,
            deviation_threshold=self.deviation_threshold,
        )

        # Detect bad channels
        print(f"[BAD_CHANNELS DEBUG] Running detector with methods: {self.method}")
        bad_channels = detector.detect(raw)
        print(f"[BAD_CHANNELS DEBUG] Detector returned: {len(bad_channels)} bad channels: {bad_channels}")

        # Add manual bads
        if self.manual_bads:
            bad_channels = list(set(bad_channels + self.manual_bads))

        # Get current EEG channel count
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        n_total = len(eeg_picks)

        # Mark, drop, or interpolate
        if self.drop:
            # Drop bad channels entirely
            if len(bad_channels) > 0:
                print(f"[BAD_CHANNELS] Dropping {len(bad_channels)} bad channels: {bad_channels}")
                raw.drop_channels(bad_channels)
                print(f"[BAD_CHANNELS] After drop: {len(raw.ch_names)} channels remain")
            n_interpolated = 0
            n_dropped = len(bad_channels)
        elif self.mark_only:
            # Just mark as bad
            raw.info['bads'] = bad_channels
            n_interpolated = 0
            n_dropped = 0
        else:
            # Interpolate immediately
            raw.info['bads'] = bad_channels
            if len(bad_channels) > 0:
                raw.interpolate_bads(reset_bads=True, verbose=False)
            n_interpolated = len(bad_channels)
            n_dropped = 0

        # Build metadata
        step_metadata = {
            'applied': True,
            'method': self.method,
            'n_bad_channels': len(bad_channels),
            'bad_channels': bad_channels,
            'n_interpolated': n_interpolated,
            'n_dropped': n_dropped,
            'mark_only': self.mark_only,
            'drop': self.drop,
            'pct_bad': 100 * len(bad_channels) / n_total if n_total > 0 else 0,
        }

        return raw, step_metadata

    def check_dependencies(self) -> None:
        """Fail at construction when a backend needed by the selected
        method(s) is missing.

        RANSAC detection silently returned zero bad channels when
        ``autoreject`` was absent — that is a silent quality
        degradation and is replaced by this hard preflight.
        """
        if 'ransac' in self.method:
            try:
                from autoreject import Ransac  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    f"BadChannelDetectionStep(method='ransac') requires autoreject ({e}). "
                    "Install with: pip install autoreject"
                ) from e

    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        config = super().get_config()
        config.update({
            'method': self.method,
            'mark_only': self.mark_only,
            'correlation_threshold': self.correlation_threshold,
            'ransac_corr_threshold': self.ransac_corr_threshold,
        })
        return config
