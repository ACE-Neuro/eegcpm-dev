"""
Resampling step.

Author: EEGCPM Development Team
Created: 2025-12
"""

from typing import Any, Dict, Tuple
import mne

from .base import ProcessingStep


class ResampleStep(ProcessingStep):
    """
    Resampling step.

    Downsamples or upsamples EEG data to a new sampling frequency.

    Parameters
    ----------
    sfreq : float
        Target sampling frequency in Hz
    npad : str or int
        Padding type: 'auto' (default) or integer (samples)

    Examples
    --------
    Downsample to 250 Hz:
    >>> step = ResampleStep(sfreq=250)

    Downsample to 500 Hz:
    >>> step = ResampleStep(sfreq=500)
    """

    name = "resample"
    version = "1.0"

    # A recording already at the target sampling frequency is a
    # legitimate no-op (nothing to do), not a failure. Any other
    # not-applied result still fails a required step.
    ALLOWED_NOOP_REASONS = frozenset({'already_at_target_sfreq'})

    def __init__(
        self,
        sfreq: float,
        npad: str = 'auto',
        enabled: bool = True,
    ):
        """Initialize resample step."""
        super().__init__(enabled=enabled)

        self.sfreq = sfreq
        self.npad = npad

    def process(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any]
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        """
        Resample data.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input raw data
        metadata : dict
            Metadata from previous steps

        Returns
        -------
        raw : mne.io.BaseRaw
            Resampled data
        step_metadata : dict
            Resampling metadata
        """
        original_sfreq = raw.info['sfreq']

        if original_sfreq == self.sfreq:
            return raw, {'skipped': True, 'reason': 'already_at_target_sfreq'}

        raw.resample(self.sfreq, npad=self.npad, verbose=False)

        step_metadata = {
            'applied': True,
            'original_sfreq': original_sfreq,
            'new_sfreq': self.sfreq,
        }

        return raw, step_metadata

    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        config = super().get_config()
        config.update({'sfreq': self.sfreq})
        return config
