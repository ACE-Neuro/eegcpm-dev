"""Set MNE channel types for named channels.

Some vendors and BIDS sidecars describe channels that the reader types
unhelpfully: Neuroscan BrainVision exports record a digital trigger line as an
EEG channel, and auxiliary electrodes (EOG/ECG/EMG) can be present but
physically unusable (flat or dominated by a DC offset) so that EOG-based ICA
detection and the ocular QC gate must not use them. This step applies an
explicit, config-declared mapping instead of a name heuristic.
"""

from typing import Any, Dict, Optional, Tuple

import mne

from .base import ProcessingStep


class SetChannelTypesStep(ProcessingStep):
    """Set channel types from a name -> MNE type mapping.

    Parameters
    ----------
    types : dict
        Channel name -> MNE channel type (e.g. ``{'VEOG': 'misc'}``). Names not
        present in the recording are reported (and are a hard error with
        ``on_missing='raise'``).
    on_missing : str
        ``'warn'`` (default) records missing names in the step metadata;
        ``'raise'`` fails the step when any requested channel is absent.
    enabled : bool

    Examples
    --------
    >>> step = SetChannelTypesStep(types={'VEOG': 'misc', 'HEOG': 'misc'})
    """

    name = "set_channel_types"
    version = "1.0"
    #: A mapping whose channels are all absent is a legitimate no-op (the
    #: caller may target an optional auxiliary montage).
    ALLOWED_NOOP_REASONS = frozenset({'no_matching_channels'})

    def __init__(
        self,
        types: Optional[Dict[str, str]] = None,
        on_missing: str = 'warn',
        enabled: bool = True,
    ):
        """Initialize the step."""
        super().__init__(enabled=enabled)
        self.types = dict(types or {})
        self.on_missing = on_missing

    def process(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any],
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        """Apply the channel-type mapping in place.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input data.
        metadata : dict
            Metadata from previous steps.

        Returns
        -------
        raw : mne.io.BaseRaw
            Data with the requested channel types.
        step_metadata : dict
            Requested/previous/new types, plus absent names.
        """
        present = {ch: t for ch, t in self.types.items() if ch in raw.ch_names}
        missing = sorted(set(self.types) - set(present))

        if self.on_missing == 'raise' and missing:
            raise ValueError(
                "SetChannelTypesStep: requested channel(s) not present in the "
                f"recording: {missing}"
            )

        if not present:
            return raw, {
                'skipped': True,
                'reason': 'no_matching_channels',
                'requested': dict(self.types),
                'missing': missing,
            }

        previous = {ch: raw.get_channel_types(picks=[ch])[0] for ch in present}
        raw.set_channel_types(present, verbose=False)

        return raw, {
            'applied': True,
            'types': present,
            'previous_types': previous,
            'n_channels': len(present),
            'missing': missing,
        }
