"""
Bad channel interpolation step.

Author: EEGCPM Development Team
Created: 2025-12
"""

from typing import Any, Dict, Tuple
import mne

from .base import ProcessingStep


class InterpolateBadChannelsStep(ProcessingStep):
    """
    Interpolate bad channels step.

    Interpolates channels marked as bad using spherical spline interpolation.

    Parameters
    ----------
    mode : str
        Interpolation mode: 'accurate' (default) or 'fast'
    max_bad_percent : float
        Maximum percentage of bad channels to interpolate (default: 50.0)
        If exceeded, channels are dropped instead
    reset_bads : bool
        Reset bad channel list after interpolation (default: True)

    Examples
    --------
    Standard interpolation:
    >>> step = InterpolateBadChannelsStep(mode='accurate')

    With safety check:
    >>> step = InterpolateBadChannelsStep(
    ...     mode='accurate',
    ...     max_bad_percent=50.0  # Drop if >50% bad
    ... )
    """

    name = "interpolate"
    version = "1.0"

    # Interpolating zero bad channels is a legitimate no-op (nothing to
    # do), not a failure. Any other not-applied result still fails a
    # required step.
    #
    # NOTE: the channel-drop branch below (when the bad-channel budget
    # `max_bad_percent` is exceeded) is a known P1-F5 defect — it drops
    # the bad channels and reports `applied: True` instead of failing
    # the recording. It is removed in B4 task 4 (P1-F5); do not rely on
    # it (the pinned detector is mark-only and never drops channels).
    ALLOWED_NOOP_REASONS = frozenset({'no_bad_channels'})

    def __init__(
        self,
        mode: str = 'accurate',
        max_bad_percent: float = 50.0,
        reset_bads: bool = True,
        enabled: bool = True,
    ):
        """Initialize interpolation step."""
        super().__init__(enabled=enabled)

        self.mode = mode
        self.max_bad_percent = max_bad_percent
        self.reset_bads = reset_bads

    def process(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any]
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        """
        Interpolate bad channels.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input raw data (with bad channels marked)
        metadata : dict
            Metadata from previous steps

        Returns
        -------
        raw : mne.io.BaseRaw
            Data with bad channels interpolated or dropped
        step_metadata : dict
            Interpolation results
        """
        bad_channels = raw.info['bads']

        if len(bad_channels) == 0:
            return raw, {'skipped': True, 'reason': 'no_bad_channels'}

        # Get channel counts
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        n_total = len(eeg_picks)
        n_bad = len(bad_channels)
        pct_bad = 100 * n_bad / n_total if n_total > 0 else 100

        # Safety check: too many bad channels?
        if pct_bad > self.max_bad_percent:
            # Drop instead of interpolate
            raw.drop_channels(bad_channels)
            return raw, {
                'applied': True,
                'action': 'dropped',
                'n_channels_dropped': n_bad,
                'pct_bad': pct_bad,
                'reason': f'Too many bad channels ({pct_bad:.1f}% > {self.max_bad_percent}%)',
            }

        # Interpolate
        raw.interpolate_bads(
            reset_bads=self.reset_bads,
            mode=self.mode,
            verbose=False
        )

        step_metadata = {
            'applied': True,
            'action': 'interpolated',
            'n_interpolated': n_bad,
            'pct_bad': pct_bad,
            'mode': self.mode,
        }

        return raw, step_metadata

    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        config = super().get_config()
        config.update({
            'mode': self.mode,
            'max_bad_percent': self.max_bad_percent,
        })
        return config
