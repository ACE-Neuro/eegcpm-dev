"""
EOG regression step (per spec §3.i + R-016, P1-F13/F16).

Linear regression of 9 EOG channels out of scalp EEG channels:
    X_scalp_clean = X_scalp - B @ X_eog
where B is fit on the clean (post-line-noise, post-filter) data.

P1-F13 (B6): the regression is a **single multi-target lstsq solve**
``B.T = lstsq(X.T, Y.T)`` (X = EOG rows + intercept row, Y = EEG rows)
instead of a per-channel normal-equations solve. The normal-equations
form squares the design condition number (and was computed once per
channel); the direct solve uses the SVD of the design matrix itself.

The reported conditioning is the condition number of the **z-scored**
EOG rows plus the intercept row, computed once: standardizing the rows
makes it invariant to the EOG channel units (a scaling of the EOG data
must not change the reported number). R-016 requires flagging
rank/conditioning violations; the step reports them and never falls back
to a fixed perturbation that would mask the true rank.

Spec deviation (recorded in the step metadata): EOG regression without
ICA can attenuate frontal ERPs; the sensitivity check runs at B8
(frontal ERP with vs without EOG regression, plan §B8 step 3).

B6 rev2 NEW-1 (regressor precondition): the regressors must be **band-passed**
by the preceding filter step (the canonical chain filters the EEG + EOG picks,
0.1-45 Hz). Regressing with unfiltered EOG channels re-injects their DC drift
and line noise into the cleaned EEG *after* zapline/low-pass (real-data probe:
> 50 Hz power x33, 59-61 Hz x~14,000, frontal std 51.6 uV vs 10.9 uV with
filtered EOG). This step does not filter internally — it consumes whatever the
chain's filter step produced; the version is bumped because the step's input
contract (band-passed regressors) is now documented and enforced by the chain
composition.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import mne
import numpy as np

from .base import ProcessingStep


# Condition number threshold above which the solve is flagged
# (per R-016: "flag rank/conditioning violations")
CONDITION_NUMBER_THRESHOLD: float = 1e6

#: Recorded in the step metadata (P1-F13): the regression removes ocular
#: variance without ICA, which can attenuate frontal ERPs. The empirical
#: sensitivity check is scheduled at B8 (real subject).
SPEC_DEVIATION = (
    "EOG regression without ICA; frontal ERP attenuation risk (checked in B8)"
)


class EOGRegressionStep(ProcessingStep):
    """Linear regression of EOG channels out of scalp EEG channels."""

    name = "eog_regression"
    version = "4.0"  # 4.0: band-passed-regressor contract documented (NEW-1);
    # 3.0: single multi-target lstsq + z-scored condition number

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
        # Design matrix X: EOG rows + intercept row.
        X = np.vstack([eog_data, np.ones((1, eog_data.shape[1]))])
        n_coef = X.shape[0]

        # Single multi-target least-squares solve (P1-F13):
        #   Y.T (n_times, n_eeg) = X.T (n_times, n_coef) @ B.T
        # -> B.T (n_coef, n_eeg); B (n_eeg, n_coef).
        B_T, residuals, rank, sv = np.linalg.lstsq(X.T, eeg_data.T, rcond=None)
        B = B_T.T

        # Condition number of the z-scored EOG rows + intercept, once.
        # The intercept row is already unit-scaled; z-scoring the EOG rows
        # removes their unit dependence (scaling EOG by any constant leaves
        # this number unchanged).
        eog_std = eog_data.std(axis=1, keepdims=True)
        eog_std = np.where(eog_std > 0, eog_std, 1.0)
        X_z = np.vstack([
            (eog_data - eog_data.mean(axis=1, keepdims=True)) / eog_std,
            np.ones((1, eog_data.shape[1])),
        ])
        condition_number = float(np.linalg.cond(X_z))
        rank_deficient = bool(rank < n_coef)
        condition_number_exceeded = bool(condition_number > CONDITION_NUMBER_THRESHOLD)

        # Reconstruct
        eeg_clean = eeg_data - B @ X
        raw._data[eeg_picks] = eeg_clean
        return raw, {
            "applied": True,
            "n_eog_channels": len(eog_picks),
            "n_eeg_channels": len(eeg_picks),
            "n_eog_used": len(eog_present),
            "solve": "lstsq_multi_target",
            "rank": int(rank),
            "condition_number": condition_number,
            "condition_number_threshold": CONDITION_NUMBER_THRESHOLD,
            "condition_number_exceeded": condition_number_exceeded,
            "rank_deficient": rank_deficient,
            "n_rank_violations": int(rank_deficient or condition_number_exceeded),
            "spec_deviation": SPEC_DEVIATION,
        }

    def get_config(self) -> Dict[str, Any]:
        """Get configuration (all constructor parameters, P1-F7)."""
        config = super().get_config()
        config.update({
            "eog_channels": list(self.eog_channels),
            "n_eog_channels": self.n_eog_channels,
        })
        return config
