"""
EOG regression step (per spec §3.i).

Linear regression of 9 EOG channels out of scalp EEG channels:
    X_scalp_clean = X_scalp - B @ X_eog
where B is fit on the clean (post-line-noise, post-filter) data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import mne
import numpy as np

from .base import ProcessingStep


class EOGRegressionStep(ProcessingStep):
    """Linear regression of EOG channels out of scalp EEG channels."""

    name = "eog_regression"
    version = "1.0"

    def __init__(
        self,
        eog_channels: List[str] = None,
        n_eog_channels: int = 9,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.eog_channels = list(eog_channels) if eog_channels is not None else [
            "E8", "E14", "E17", "E21", "E25", "E125", "E126", "E127", "E128",
        ]
        self.n_eog_channels = n_eog_channels

    def process(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any],
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        eog_present = [ch for ch in self.eog_channels if ch in raw.ch_names]
        if not eog_present:
            # No EOG channels: graceful skip
            return raw, {
                "applied": False,
                "reason": "no_eog_channels_present",
                "n_eog_channels": 0,
            }
        # Fit linear regression per scalp channel
        eog_picks = mne.pick_types(raw.info, eog=True, exclude="bads")
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        if len(eeg_picks) == 0 or len(eog_picks) == 0:
            return raw, {
                "applied": False,
                "reason": "no_picks",
                "n_eog_channels": len(eog_picks),
                "n_eeg_channels": len(eeg_picks),
            }
        data = raw.get_data(picks=np.concatenate([eeg_picks, eog_picks]))
        eeg_data = data[:len(eeg_picks)]
        eog_data = data[len(eeg_picks):]
        # Add intercept to EOG
        eog_with_int = np.vstack([eog_data, np.ones((1, eog_data.shape[1]))])
        # B = eeg @ eog_with_int.T @ inv(eog_with_int @ eog_with_int.T)
        Bt = eog_with_int @ eog_with_int.T
        # Regularize for numerical stability
        Bt += np.eye(Bt.shape[0]) * 1e-10
        B = eeg_data @ eog_with_int.T @ np.linalg.inv(Bt)
        # Reconstruct
        eeg_clean = eeg_data - B @ eog_with_int
        # Write back
        raw._data[eeg_picks] = eeg_clean
        return raw, {
            "applied": True,
            "n_eog_channels": len(eog_picks),
            "n_eeg_channels": len(eeg_picks),
            "n_eog_used": len(eog_present),
        }
