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

    # A recording that already carries a montage is a legitimate no-op
    # (nothing to set), not a failure. Any other not-applied result
    # still fails a required step.
    ALLOWED_NOOP_REASONS = frozenset({'montage_already_set'})

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

    def _channels_without_valid_positions(self, raw: mne.io.BaseRaw) -> list:
        """EEG channels whose position is NaN/inf or all-zero (P1-F22)."""
        import numpy as np

        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        channels: list = []
        for pick in eeg_picks:
            loc = raw.info['chs'][pick]['loc'][:3]  # x, y, z positions
            if not np.isfinite(loc).all() or not np.any(loc):
                channels.append(raw.ch_names[pick])
        return channels

    def _supplement_from_locs_file(self, raw: mne.io.BaseRaw,
                                   channels: list) -> list:
        """Fill missing positions from the bundled Neuroscan .locs file.

        Returns the channels still without a position afterwards.
        """
        if not channels:
            return []
        locs_path = Path(__file__).parents[3] / 'data' / 'neuroscan_64ch.locs'
        if not locs_path.exists():
            return list(channels)
        extra_montage = mne.channels.read_custom_montage(str(locs_path))
        still_missing = []
        for ch_name in channels:
            if ch_name in extra_montage.ch_names:
                idx = raw.ch_names.index(ch_name)
                pos = extra_montage.get_positions()['ch_pos'][ch_name]
                raw.info['chs'][idx]['loc'][:3] = pos
            else:
                still_missing.append(ch_name)
        return still_missing

    def _assert_positions_valid(self, raw: mne.io.BaseRaw,
                                channels: list) -> None:
        """Raise for EEG channels without a usable position (P1-F22).

        Dropping them silently would change the analysis set behind the
        caller's back; an EGI recording whose montage did not match (NaN
        positions) must fail loudly instead.
        """
        if channels:
            raise ValueError(
                "MontageStep: EEG channel(s) without a valid position after "
                f"set_montage: {sorted(channels)}. Fix the montage/channel "
                "names (or remove the channels explicitly) — silently "
                "dropping them would change the analysis set."
            )

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

        Raises
        ------
        ValueError
            If any EEG channel still has a NaN/inf or all-zero position
            after the montage is applied (P1-F22).
        """
        # Check if montage already set
        if raw.get_montage() is not None:
            # Even if montage exists, repair channels with NaN/Inf positions
            # from the bundled Neuroscan .locs file; anything left without a
            # position fails loudly (never dropped silently, P1-F22).
            channels_without_positions = self._channels_without_valid_positions(raw)
            channels_without_positions = self._supplement_from_locs_file(
                raw, channels_without_positions)
            self._assert_positions_valid(raw, channels_without_positions)

            result_meta = {
                'skipped': True,
                'reason': 'montage_already_set',
                'channels_dropped': [],
                'n_dropped': 0,
            }
            return raw, result_meta

        # Load montage
        if self.file:
            montage = mne.channels.read_custom_montage(self.file)
            source = 'custom_file'
        else:
            montage = mne.channels.make_standard_montage(self.type)
            source = 'standard'

        # Rename channels to match standard montage naming conventions
        # Neuroscan uses uppercase (FP1, CZ) while MNE uses mixed case (Fp1, Cz)
        NEUROSCAN_RENAME = {
            'FP1': 'Fp1', 'FP2': 'Fp2', 'FPZ': 'Fpz',
            'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz', 'OZ': 'Oz',
            'FCZ': 'FCz', 'CPZ': 'CPz', 'POZ': 'POz',
            'AF3': 'AF3', 'AF4': 'AF4', 'AF7': 'AF7', 'AF8': 'AF8',
            'CB1': 'Cb1', 'CB2': 'Cb2',
            'HEOG': 'HEOG', 'VEOG': 'VEOG',
            'M1': 'M1', 'M2': 'M2',
        }
        rename_map = {}
        for ch in raw.ch_names:
            if ch in NEUROSCAN_RENAME:
                new_name = NEUROSCAN_RENAME[ch]
                if new_name != ch:
                    rename_map[ch] = new_name
            elif ch not in montage.ch_names and ch.upper() != ch:
                # Mixed-case channel not in montage — try lowercase first letter variant
                alt = ch[0].upper() + ch[1:].lower() if len(ch) > 1 else ch.lower()
                if alt in montage.ch_names:
                    rename_map[ch] = alt
        if rename_map:
            raw.rename_channels(rename_map)

        # Set montage with match_alias=True so MNE's official channel aliases
        # (e.g., Cb1->POO7, Cb2->POO8, T3->T7, M1->TP9) get resolved automatically
        raw.set_montage(
            montage,
            match_alias=True,
            on_missing=self.on_missing,
            verbose=False,
        )

        # Try to supplement missing positions from bundled Neuroscan .locs file
        channels_without_positions = self._channels_without_valid_positions(raw)
        channels_without_positions = self._supplement_from_locs_file(
            raw, channels_without_positions)

        # Any EEG channel still without a valid position fails the step
        # (P1-F22: never drop channels silently).
        self._assert_positions_valid(raw, channels_without_positions)

        step_metadata = {
            'applied': True,
            'type': self.type,
            'source': source,
            'channels_dropped': [],
            'n_dropped': 0,
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
