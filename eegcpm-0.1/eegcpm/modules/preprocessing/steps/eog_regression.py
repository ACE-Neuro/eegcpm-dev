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
chain's filter step produced.

**B7 (carried B6 item): the precondition is now enforced, not just documented.**
``_check_band_limited`` runs two checks and raises ``ValueError`` when either
fails:

1. **declared band limit** — ``raw.info['lowpass']`` must be finite and
   ``<= HF_CUTOFF_HZ`` (50 Hz). The chain's filter step sets it to 45 Hz;
   an unfiltered ``RawArray`` carries the Nyquist frequency (125 Hz at 250 Hz
   sampling) and is refused outright, whatever its spectrum looks like.
2. **measured band limit** — the fraction of EOG power above ``HF_CUTOFF_HZ``
   (50 Hz = the filter edge plus its transition band; capped at 45 % of
   Nyquist) must be ``<= MAX_HF_POWER_FRACTION`` (0.05), measured with Welch's
   method. Broadband unfiltered regressors are caught here even when their
   metadata lies (measured on the fixture: 0.0075 filtered vs 0.59 unfiltered).

Both numbers (``eog_hf_power_fraction``, ``eog_lowpass_hz``) are recorded in
the metadata so a pass is inspectable. Set
``allow_unfiltered_regressors=True`` to bypass both checks explicitly (recorded
as ``eog_band_limited_check='bypassed'``) — a documented opt-out, not a silent
fallback.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import mne
import numpy as np

from .base import ProcessingStep


# Condition number threshold above which the solve is flagged
# (per R-016: "flag rank/conditioning violations")
CONDITION_NUMBER_THRESHOLD: float = 1e6

#: Upper edge of the band the regressors must be limited to: the chain's filter
#: edge (45 Hz) plus its transition band. Capped at 45 % of Nyquist for
#: low-rate recordings.
HF_CUTOFF_HZ: float = 50.0
#: Maximum fraction of EOG power above :data:`HF_CUTOFF_HZ` accepted before the
#: step refuses the regressors (B7, carried B6 item).
MAX_HF_POWER_FRACTION: float = 0.05

#: Recorded in the step metadata (P1-F13): the regression removes ocular
#: variance without ICA, which can attenuate frontal ERPs. The empirical
#: sensitivity check is scheduled at B8 (real subject).
SPEC_DEVIATION = (
    "EOG regression without ICA; frontal ERP attenuation risk (checked in B8)"
)


class EOGRegressionStep(ProcessingStep):
    """Linear regression of EOG channels out of scalp EEG channels."""

    name = "eog_regression"
    version = "5.0"  # 5.0: band-limited-regressor guard (B7, carried B6 item);
    # 4.0: band-passed-regressor contract documented (NEW-1);
    # 3.0: single multi-target lstsq + z-scored condition number

    def __init__(
        self,
        eog_channels: List[str] = None,
        n_eog_channels: int = 9,
        enabled: bool = True,
        allow_unfiltered_regressors: bool = False,
    ):
        super().__init__(enabled=enabled)
        self.eog_channels = list(eog_channels) if eog_channels is not None else [
            "E8", "E14", "E17", "E21", "E25", "E125", "E126", "E127", "E128",
        ]
        self.n_eog_channels = n_eog_channels
        self.allow_unfiltered_regressors = allow_unfiltered_regressors

    @staticmethod
    def _check_band_limited(
        raw: mne.io.BaseRaw, eog_picks: np.ndarray, *, allow_unfiltered: bool
    ) -> Tuple[float, float, bool]:
        """Refuse EOG regressors that are not band-limited (B7, carried B6 item).

        Returns ``(hf_fraction, cutoff_hz, bypassed)``. Raises ``ValueError``
        when more than :data:`MAX_HF_POWER_FRACTION` of the regressors' power
        sits above :data:`HF_CUTOFF_HZ` (capped at 45 % of Nyquist) and the
        caller has not opted out.

        Rationale: the canonical chain's filter step band-passes the EEG **and**
        EOG picks. An unfiltered EOG channel re-injects its 60 Hz line noise and
        DC drift into the cleaned EEG *after* zapline/low-pass (real-data probe:
        > 50 Hz power x33, 59-61 Hz x~14,000). The guard makes that contract
        explicit instead of relying on the chain composition.
        """
        sfreq = float(raw.info["sfreq"])
        nyquist = sfreq / 2.0
        cutoff = min(HF_CUTOFF_HZ, 0.45 * nyquist)
        lowpass = raw.info.get("lowpass")
        lowpass = float(lowpass) if lowpass is not None else float("nan")
        if (not np.isfinite(lowpass) or lowpass > HF_CUTOFF_HZ) and not allow_unfiltered:
            raise ValueError(
                f"EOG regression: the recording declares lowpass={lowpass} Hz, "
                f"above the {HF_CUTOFF_HZ:.0f} Hz limit; regressing with "
                f"unfiltered EOG channels re-injects their line noise and drift "
                f"into the cleaned EEG. Filter the EEG **and EOG** channels "
                f"first (the canonical chain does), or pass "
                f"allow_unfiltered_regressors=True to bypass explicitly."
            )
        from scipy.signal import welch

        data = raw.get_data(picks=eog_picks)
        nperseg = min(int(round(2 * sfreq)), data.shape[1])
        freqs, psd = welch(data, fs=sfreq, nperseg=max(nperseg, 8), axis=-1)
        if not np.isfinite(psd).any():
            # Cannot measure: do not invent a verdict, do not silently pass.
            raise ValueError(
                "EOG regression: could not estimate the regressors' spectrum; "
                "refusing to run without the band-limited check (pass "
                "allow_unfiltered_regressors=True to bypass explicitly)"
            )
        hf = freqs > cutoff
        total = float(np.trapezoid(psd, freqs, axis=-1).sum())
        if total <= 0:
            return 0.0, cutoff, bool(allow_unfiltered)
        hf_power = float(np.trapezoid(psd[:, hf], freqs[hf], axis=-1).sum())
        hf_fraction = hf_power / total
        if hf_fraction > MAX_HF_POWER_FRACTION and not allow_unfiltered:
            raise ValueError(
                f"EOG regression: the EOG regressors are not band-limited "
                f"({hf_fraction:.1%} of their power is above {cutoff:.0f} Hz, "
                f"limit {MAX_HF_POWER_FRACTION:.0%}); regressing with unfiltered "
                f"EOG channels re-injects their line noise and drift into the "
                f"cleaned EEG. Filter the EEG **and EOG** channels first (the "
                f"canonical chain does), or pass "
                f"allow_unfiltered_regressors=True to bypass explicitly."
            )
        return hf_fraction, cutoff, bool(allow_unfiltered)

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

        # B7 (carried B6 item): the regressors must be band-limited.
        hf_fraction, hf_cutoff, hf_bypassed = self._check_band_limited(
            raw, eog_picks, allow_unfiltered=self.allow_unfiltered_regressors
        )

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
            "eog_hf_power_fraction": hf_fraction,
            "eog_hf_cutoff_hz": hf_cutoff,
            "eog_lowpass_hz": float(raw.info.get("lowpass") or float("nan")),
            "eog_band_limited_check": (
                "bypassed" if hf_bypassed else "passed"
            ),
            "spec_deviation": SPEC_DEVIATION,
        }

    def get_config(self) -> Dict[str, Any]:
        """Get configuration (all constructor parameters, P1-F7)."""
        config = super().get_config()
        config.update({
            "eog_channels": list(self.eog_channels),
            "n_eog_channels": self.n_eog_channels,
            "allow_unfiltered_regressors": self.allow_unfiltered_regressors,
            # Algorithm constants (B6 NEW-4 principle: the resolved-config hash
            # must cover the numerics the step actually applies, not only the
            # constructor arguments).
            "algorithm": {
                "hf_cutoff_hz": HF_CUTOFF_HZ,
                "max_hf_power_fraction": MAX_HF_POWER_FRACTION,
                "condition_number_threshold": CONDITION_NUMBER_THRESHOLD,
            },
        })
        return config
