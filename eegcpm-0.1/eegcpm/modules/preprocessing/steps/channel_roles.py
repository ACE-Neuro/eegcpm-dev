"""
Channel roles step (per S19 in the spec).

Assigns channel roles for EGI HydroCel 129-channel net. The Cz channel
is kept `eeg`-typed with an `is_reference` flag consumed by the
bad-channel detector's exemption, so `picks: eeg` INCLUDES Cz in the
average-reference computation (this is required for reference recovery).

Cz is EXEMPT from:
  - flat-channel detection
  - bad-channel budget
  - the `<=11/109` interpolation cap (Cz is not interpolated)

Neck/face channels are dropped; 9 EOG channels are retained for EOG
regression; 109 scalp channels are the analysis set.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import mne

from .base import ProcessingStep


EGI_129_EOG_CHANNELS: List[str] = [
    "E8", "E14", "E17", "E21", "E25", "E125", "E126", "E127", "E128",
]

# Neck/face channels (EGI 128 layout, channels below the
# canthomeatal plane). Verified against the HydroCel documentation.
EGI_129_NECK_FACE_CHANNELS: List[str] = [
    "E38", "E43", "E44", "E48", "E49", "E56", "E63", "E68",
    "E73", "E81", "E117",
]

# Cz is the recording reference (identically flat pre-referencing).
EGI_129_REFERENCE_CHANNEL: str = "Cz"


class ChannelRolesStep(ProcessingStep):
    """Assign channel roles for EGI HydroCel 129-channel net.

    Cz is typed `eeg` with `is_reference: true`; the
    bad-channel detector's exemption consumes the flag, so Cz is
    exempt from flat-channel and bad-channel detection and from
    the bad-channel budget. Post-referencing, the post-ref Cz
    variance is asserted > 0; the post-ref Cz column is marked
    RECOVERED in the feature frame.
    """

    name = "channel_roles"
    version = "1.0"

    def __init__(
        self,
        net_type: str = "EGI_129",
        scalp_channels_count: int = 109,
        eog_channels: List[str] = None,
        neck_face_channels: List[str] = None,
        reference_channel: str = "Cz",
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.net_type = net_type
        self.scalp_channels_count = scalp_channels_count
        self.eog_channels = list(eog_channels) if eog_channels is not None else list(EGI_129_EOG_CHANNELS)
        self.neck_face_channels = list(neck_face_channels) if neck_face_channels is not None else list(EGI_129_NECK_FACE_CHANNELS)
        self.reference_channel = reference_channel

    def process(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any],
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        # Set EOG channel types (retained for EOG regression)
        eog_present = [ch for ch in self.eog_channels if ch in raw.ch_names]
        if eog_present:
            raw.set_channel_types({ch: "eog" for ch in eog_present})

        # Drop neck/face channels
        neck_face_present = [ch for ch in self.neck_face_channels if ch in raw.ch_names]
        if neck_face_present:
            raw.drop_channels(neck_face_present)

        # Set Cz as reference channel: keep eeg-typed with is_reference
        # flag. The flag is stored on the raw object as a custom
        # attribute (MNE info['chs'] keys are restricted). It is
        # consumed by the bad-channel detector's exemption and by
        # the reference step's Cz-include check.
        if self.reference_channel in raw.ch_names:
            ch_idx = raw.ch_names.index(self.reference_channel)
            # Reaffirm eeg typing
            raw.set_channel_types({self.reference_channel: "eeg"})
            # Store reference info on the raw object (not info dict,
            # which is restricted by MNE)
            if not hasattr(raw, "_eegcpm_reference_channels"):
                raw._eegcpm_reference_channels = {}
            raw._eegcpm_reference_channels[self.reference_channel] = True

        step_meta = {
            "eog_channels": eog_present,
            "neck_face_dropped": neck_face_present,
            "reference_channel": self.reference_channel,
            "scalp_channels_count": len(
                [ch for ch in raw.ch_names
                 if ch not in eog_present and ch != self.reference_channel]
            ),
        }
        return raw, step_meta
