"""
Montage (channel location) step.

Author: EEGCPM Development Team
Created: 2025-12
"""

from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import mne

from .base import ProcessingStep


class MontageStep(ProcessingStep):
    """
    Set channel montage (electrode locations).

    Parameters
    ----------
    type : str
        Montage type: 'standard_1020', 'standard_1005', 'biosemi64', etc.
    file : str or Path, optional
        Custom montage file path
    on_missing : str
        How to handle missing channels: 'raise', 'warn', 'ignore'

    Examples
    --------
    Standard 10-20 montage:
    >>> step = MontageStep(type='standard_1020')

    Custom montage from file:
    >>> step = MontageStep(file='/path/to/montage.txt')

    Biosemi 64:
    >>> step = MontageStep(type='biosemi64')
    """

    name = "montage"
    version = "1.0"

    def __init__(
        self,
        type: str = 'standard_1020',
        file: Optional[str] = None,
        on_missing: str = 'warn',
        enabled: bool = True,
    ):
        """Initialize montage step."""
        super().__init__(enabled=enabled)

        self.type = type
        self.file = file
        self.on_missing = on_missing

    def process(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any]
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        """
        Set montage.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input raw data
        metadata : dict
            Metadata from previous steps

        Returns
        -------
        raw : mne.io.BaseRaw
            Data with montage set
        step_metadata : dict
            Montage metadata
        """
        # Check if montage already set
        if raw.get_montage() is not None:
            # Even if montage exists, drop channels with NaN/Inf positions
            import numpy as np

            eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            channels_without_positions = []

            for pick in eeg_picks:
                ch_info = raw.info['chs'][pick]
                loc = ch_info['loc'][:3]  # x, y, z positions
                if not np.isfinite(loc).all():
                    channels_without_positions.append(raw.ch_names[pick])

            # Drop channels without valid positions
            if channels_without_positions:
                print(f"[MONTAGE DEBUG] Dropping {len(channels_without_positions)} channels: {channels_without_positions}")
                raw.drop_channels(channels_without_positions)
                print(f"[MONTAGE DEBUG] After drop: {len(raw.ch_names)} channels remain")

            result_meta = {
                'skipped': True,
                'reason': 'montage_already_set',
                'channels_dropped': channels_without_positions,
                'n_dropped': len(channels_without_positions),
            }
            print(f"[MONTAGE DEBUG] Returning metadata: {result_meta}")
            return raw, result_meta

        # Load montage
        if self.file:
            montage = mne.channels.read_custom_montage(self.file)
            source = 'custom_file'
        else:
            montage = mne.channels.make_standard_montage(self.type)
            source = 'standard'

        # Set montage with match_alias=True so MNE's official channel aliases
        # (e.g., Cb1->POO7, Cb2->POO8, T3->T7, M1->TP9) get resolved automatically
        # See mne._fiff.constants.CHANNEL_LOC_ALIASES for the full alias table
        raw.set_montage(
            montage,
            match_alias=True,
            on_missing=self.on_missing,
            verbose=False,
        )

        # After setting montage, drop channels with NaN/Inf positions
        # This happens when the montage doesn't include all channels (e.g., E129 in GSN-HydroCel-129)
        import numpy as np

        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        channels_without_positions = []

        for pick in eeg_picks:
            ch_info = raw.info['chs'][pick]
            loc = ch_info['loc'][:3]  # x, y, z positions
            if not np.isfinite(loc).all():
                channels_without_positions.append(raw.ch_names[pick])

        # Drop channels without valid positions
        if channels_without_positions:
            raw.drop_channels(channels_without_positions)

        step_metadata = {
            'applied': True,
            'type': self.type,
            'source': source,
            'channels_dropped': channels_without_positions,
            'n_dropped': len(channels_without_positions),
        }

        return raw, step_metadata

    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        config = super().get_config()
        config.update({
            'type': self.type,
            'file': self.file,
        })
        return config
