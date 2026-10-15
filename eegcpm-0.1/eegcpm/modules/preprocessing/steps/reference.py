"""
Re-referencing step.

Author: EEGCPM Development Team
Created: 2025-12
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import mne
import numpy as np

from .base import ProcessingStep


class ReferenceStep(ProcessingStep):
    """
    Re-referencing step.

    Changes the EEG reference to average reference or specific channels.
    REST referencing is **not implemented** (it needs a forward model) and
    raises instead of silently falling back to average (P1-F16).

    Parameters
    ----------
    type : str
        Reference type: 'average' or 'channels'. 'rest' raises
        ``NotImplementedError`` at construction (no silent average fallback);
        anything else raises ``ValueError``.
    channels : str or list of str
        Reference channel(s) for 'channels' type
    projection : bool
        Use reference projection (default: True)
    check_reference_channel_variance : str, optional
        Channel name whose post-reference variance is asserted finite and
        > 0 (P1-F7). The canonical chain passes ``'Cz'``: with Cz included
        in the average reference its column must be recovered (non-zero),
        so a zero/NaN column means the reference was not applied as
        intended. The measured variance is reported in the step metadata as
        ``reference_channel_post_ref_variance``. ``None`` disables the check.

    Examples
    --------
    Average reference:
    >>> step = ReferenceStep(type='average')

    Specific channel reference:
    >>> step = ReferenceStep(type='channels', channels=['Cz'])

    Linked mastoids:
    >>> step = ReferenceStep(type='channels', channels=['M1', 'M2'])
    """

    name = "reference"
    version = "1.1"  # 1.1: rest raises; post-ref variance check (P1-F7/F16)

    def __init__(
        self,
        type: str = 'average',
        channels: Union[str, List[str], None] = None,
        projection: bool = True,
        check_reference_channel_variance: Optional[str] = None,
        enabled: bool = True,
    ):
        """Initialize reference step."""
        super().__init__(enabled=enabled)

        if type == 'rest':
            # REST needs a forward model and is not implemented. The old
            # behaviour silently substituted the average reference while
            # reporting type 'rest_fallback_average' — a step that claims
            # one reference and applies another (P1-F16).
            raise NotImplementedError(
                "ReferenceStep(type='rest') is not implemented (REST needs a "
                "forward model); use type='average' or type='channels' "
                "explicitly instead of relying on a silent fallback"
            )
        if type not in ('average', 'channels'):
            raise ValueError(
                f"Unknown reference type: {type!r}. "
                "Use 'average' or 'channels'"
            )

        self.type = type
        self.channels = channels
        self.projection = projection
        self.check_reference_channel_variance = check_reference_channel_variance

    def process(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any]
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        """
        Apply re-referencing.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input raw data
        metadata : dict
            Metadata from previous steps

        Returns
        -------
        raw : mne.io.BaseRaw
            Re-referenced data
        step_metadata : dict
            Reference metadata

        Raises
        ------
        RuntimeError
            When ``check_reference_channel_variance`` is set and the named
            channel is missing, or its post-reference variance is not finite
            and > 0.
        """
        if self.type == 'average':
            raw.set_eeg_reference(
                ref_channels='average',
                projection=self.projection,
                verbose=False
            )
            ref_info = {'type': 'average'}

        elif self.type == 'channels':
            if not self.channels:
                raise ValueError("Must specify channels for 'channels' reference type")

            channels = self.channels if isinstance(self.channels, list) else [self.channels]
            raw.set_eeg_reference(
                ref_channels=channels,
                projection=self.projection,
                verbose=False
            )
            ref_info = {'type': 'channels', 'channels': channels}

        else:  # pragma: no cover - the constructor rejects everything else
            raise ValueError(f"Unknown reference type: {self.type}")

        post_ref_variance: Optional[float] = None
        if self.check_reference_channel_variance is not None:
            name = self.check_reference_channel_variance
            if name not in raw.ch_names:
                raise RuntimeError(
                    f"reference channel {name!r} is not present; cannot "
                    "verify the post-reference variance"
                )
            column = raw.get_data(picks=[name])[0]
            post_ref_variance = float(np.var(column))
            if not np.isfinite(post_ref_variance) or post_ref_variance <= 0.0:
                raise RuntimeError(
                    f"post-reference variance of {name!r} is "
                    f"{post_ref_variance!r}; expected finite and > 0 — the "
                    "reference was not applied as configured "
                    f"(reference type {self.type!r})"
                )

        step_metadata = {
            'applied': True,
            **ref_info,
            'projection': self.projection,
            'reference_channel_post_ref_variance': post_ref_variance,
        }

        return raw, step_metadata

    def get_config(self) -> Dict[str, Any]:
        """Get configuration (all constructor parameters)."""
        config = super().get_config()
        config.update({
            'type': self.type,
            'channels': self.channels,
            'projection': self.projection,
            'check_reference_channel_variance': self.check_reference_channel_variance,
        })
        return config
