"""
EOG regression step (per spec §3.i + R-016).

Linear regression of 9 EOG channels out of scalp EEG channels:
    X_scalp_clean = X_scalp - B @ X_eog
where B is fit on the clean (post-line-noise, post-filter) data.

R-016: use lstsq (or SVD) for the solve, with condition number
diagnostics. Flag rank/conditioning violations; do NOT silently
fall back to a fixed 1e-10 perturbation that masks the true
rank of the EOG design matrix.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import mne
import numpy as np

from .base import ProcessingStep


# Condition number threshold above which the solve is flagged
# (per R-016: "flag rank/conditioning violations")
CONDITION_NUMBER_THRESHOLD: float = 1e6


class EOGRegressionStep(ProcessingStep):
    """Linear regression of EOG channels out of scalp EEG channels."""

    name = "eog_regression"
    version = "2.0"  # R-016: lstsq + condition diagnostics

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
            return raw, {
                "applied": False,
                "reason": "no_eog_channels_present",
                "n_eog_channels": 0,
            }
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

        # R-016: lstsq solve, NOT inverse-with-fixed-perturbation.
        # Per-channel fit: eeg_data = B @ eog_with_int + residual
        #   B = eeg_data @ eog_with_int.T @ pinv(eog_with_int @ eog_with_int.T)
        # We use np.linalg.lstsq on the SVD of (eog @ eog.T) for
        # the per-channel solve.
        n_eeg = eeg_data.shape[0]
        n_coef = eog_with_int.shape[0]
        B = np.zeros((n_eeg, n_coef))
        condition_numbers = np.zeros(n_eeg)
        rank_violations = []
        for i in range(n_eeg):
            # Solve eeg[i] = b @ eog_with_int
            # => b = eeg[i] @ eog_with_int.T @ inv(eog_with_int @ eog_with_int.T)
            # Use lstsq with the SVD of eog_with_int @ eog_with_int.T
            A = eog_with_int @ eog_with_int.T   # (n_coef, n_coef)
            b = eeg_data[i] @ eog_with_int.T    # (n_coef,)
            sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            B[i] = sol
            # Condition number
            if len(sv) > 0 and sv[0] > 0:
                cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
                condition_numbers[i] = cond
                if cond > CONDITION_NUMBER_THRESHOLD:
                    rank_violations.append(i)
                if rank < n_coef:
                    rank_violations.append(i)
        # Reconstruct
        eeg_clean = eeg_data - B @ eog_with_int
        raw._data[eeg_picks] = eeg_clean
        # Diagnostic metadata
        return raw, {
            "applied": True,
            "n_eog_channels": len(eog_picks),
            "n_eeg_channels": len(eeg_picks),
            "n_eog_used": len(eog_present),
            "condition_number_max": float(condition_numbers.max()),
            "condition_number_mean": float(condition_numbers.mean()),
            "n_rank_violations": len(rank_violations),
            "rank_violations": rank_violations,
        }
