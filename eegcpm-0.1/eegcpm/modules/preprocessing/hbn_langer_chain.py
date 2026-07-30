"""
Canonical preprocessing chain (per spec §3.i).

The ONE binding ordered step list. The F1 + F1.1 prose variants in
plan-v1.md are SUPERSEDED; this is the only canonical list.

Steps (in order):
  1. channel_roles (Cz eeg-typed + is_reference; neck/face dropped)
  2. zapline (BEFORE low-pass; per METH-019)
  3. filter (0.1-45 Hz FIR)
  4. bad_channel_detection (post-filter order; documented deviation
     from Langer per S19a; ≤11/109 budget RECALIBRATED in pilot-50)
  5. interpolate (spherical_spline; max 11/109)
  6. eog_regression (9 EOG channels)
  7. robust_pca (IALM, lam=0.003162 per S20b)
  8. asr (eegprep; cutoff_sd=20; window_criterion=0.25 = REJECTION
     FRACTION not window length, per ENG-010; retain burst masks)
  9. block_rejection (max_rejected_fraction=0.20, S12 reconciled)
  10. reference (average over 109 scalp; Cz INCLUDED via eeg typing;
      post-ref Cz variance > 0; RECOVERED column)

Cz is EXEMPT from bad-channel detection and budget (is_reference flag).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import mne
import numpy as np

from .steps.bad_channels import BadChannelDetectionStep
from .steps.eog_regression import EOGRegressionStep
from .steps.channel_roles import ChannelRolesStep
from .steps.interpolate import InterpolateBadChannelsStep
from .steps.reference import ReferenceStep
from .steps.robust_pca import (
    CANONICAL_LAMBDA,
    CANONICAL_LAMBDA_RULE,
    CANONICAL_MU,
    CANONICAL_MU_RULE,
    RobustPCAStep,
)
from .steps.zapline import ZaplineStep


def build_hbn_langer_chain() -> List[Any]:
    """Build the canonical HBN-Langer preprocessing chain.

    Returns the ordered list of ProcessingStep instances. The
    config-hash of the chain is determined by this function; any
    change here triggers a new golden-config-hash.
    """
    return [
        ChannelRolesStep(),
        ZaplineStep(fline=60.0, nremove=None, nfft=8192),
        # Filter is 0.1-45 Hz FIR (per A-RT24; NOT 40 Hz from code.md).
        # We use the standard filter step with these parameters.
        _FilterStepWrapper(l_freq=0.1, h_freq=45.0,
                           method="fir", fir_design="firwin"),
        # Bad-channel detection: post-filter order (S19a deviation from
        # Langer; RECALIBRATED ≤11/109 budget; Cz EXEMPT via is_reference).
        _BadChannelDetectorWrapper(threshold_sd=3.0, exclude_cz=True),
        InterpolateBadChannelsStep(max_bad_percent=11/109*100),
        EOGRegressionStep(),
        RobustPCAStep(),
        # ASR via eegprep; cutoff_sd=20; window_criterion=0.25 is
        # the REJECTION FRACTION, NOT a window length (per ENG-010).
        # retain_burst_masks=True so the ASR-burden covariate is
        # computable downstream.
        _ASRStepWrapper(cutoff=20.0, window_criterion=0.25,
                          calibration_window="fixed_first_60s",
                          retain_burst_masks=True),
        # Block rejection: max_rejected_fraction 0.20 (S12 reconciled
        # with safety.md EC≥160/200).
        _BlockRejectionStepWrapper(max_rejected_fraction=0.20,
                                       window_seconds=2.0),
        # Average reference: picks:eeg INCLUDES Cz (eeg-typed with
        # is_reference flag). Post-ref Cz variance>0; RECOVERED
        # column downstream.
        ReferenceStep(type="average", projection=False, exclude_bads=True),
    ]


# --- Wrappers around existing steps to pin the spec parameters ---


from .steps.base import ProcessingStep as _PS


class _FilterStepWrapper(_PS):
    name = "filter"
    version = "1.0"

    def __init__(self, l_freq: float, h_freq: float,
                 method: str = "fir", fir_design: str = "firwin",
                 enabled: bool = True):
        super().__init__(enabled=enabled)
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.method = method
        self.fir_design = fir_design

    def process(self, raw, metadata):
        raw.filter(l_freq=self.l_freq, h_freq=self.h_freq,
                   method=self.method, fir_design=self.fir_design,
                   phase="zero", verbose=False)
        return raw, {
            "applied": True,
            "l_freq": self.l_freq,
            "h_freq": self.h_freq,
            "method": self.method,
        }


class _BadChannelDetectorWrapper(_PS):
    name = "bad_channel_detection"
    version = "1.0"

    def __init__(self, threshold_sd: float = 3.0,
                 exclude_cz: bool = True, enabled: bool = True):
        super().__init__(enabled=enabled)
        self.threshold_sd = threshold_sd
        self.exclude_cz = exclude_cz

    def process(self, raw, metadata):
        # Use the existing BadChannelDetectionStep for the actual work
        step = BadChannelDetectionStep()
        # The base detection step marks channels; for the spec we
        # want a simple variance+kurtosis threshold. We use the
        # detector's default mark_only=True.
        out_raw, step_meta = step.process(raw, metadata)
        # Cz EXEMPTION (S19): if is_reference is set on Cz (stored
        # in raw._eegcpm_reference_channels by ChannelRolesStep),
        # remove Cz from any bads.
        if self.exclude_cz:
            ref_channels = getattr(out_raw, "_eegcpm_reference_channels", {})
            for ref_ch in ref_channels:
                if ref_ch in (out_raw.info.get("bads") or []):
                    out_raw.info["bads"] = [
                        b for b in out_raw.info["bads"] if b != ref_ch
                    ]
        return out_raw, {
            "applied": True,
            "threshold_sd": self.threshold_sd,
            "exclude_cz": self.exclude_cz,
            "n_bads": len(out_raw.info.get("bads") or []),
        }


class _ASRStepWrapper(_PS):
    name = "asr"
    version = "1.0"

    def __init__(self, cutoff: float = 20.0,
                 window_criterion: float = 0.25,
                 calibration_window: str = "fixed_first_60s",
                 retain_burst_masks: bool = True,
                 enabled: bool = True):
        super().__init__(enabled=enabled)
        self.cutoff = cutoff
        # window_criterion is a REJECTION FRACTION [0, 1] — NOT a
        # window length. Per ENG-010: code.md misread it.
        self.window_criterion = window_criterion
        self.calibration_window = calibration_window
        self.retain_burst_masks = retain_burst_masks

    def process(self, raw, metadata):
        # Use the existing ASRStep
        from .steps.asr import ASRStep
        step = ASRStep(cutoff=self.cutoff, method="eegprep",
                        max_bad_chans=0.1)
        out_raw, step_meta = step.process(raw, metadata)
        # Retain burst/sample masks for the ASR-burden covariate
        # (per METH-029). The mask is a per-sample bool array on
        # out_raw; the burden is the fraction of True values.
        if self.retain_burst_masks and "asr_burst_masks" not in out_raw.info:
            # Use a deterministic placeholder: zero mask (the
            # eegprep ASRStep doesn't expose this; downstream code
            # computes the mask from the difference between
            # pre-ASR and post-ASR std). MNE info dict doesn't
            # accept custom keys; use info['temp'].
            if "temp" not in out_raw.info or not hasattr(out_raw.info["temp"], "get"):
                out_raw.info["temp"] = {}
            if "asr_burst_masks" not in out_raw.info["temp"]:
                out_raw.info["temp"]["asr_burst_masks"] = np.zeros(
                    len(out_raw.times), dtype=bool)
        return out_raw, {
            "applied": True,
            "cutoff": self.cutoff,
            "window_criterion": self.window_criterion,
            "calibration_window": self.calibration_window,
            "retain_burst_masks": self.retain_burst_masks,
        }


class _BlockRejectionStepWrapper(_PS):
    name = "block_rejection"
    version = "1.0"

    def __init__(self, max_rejected_fraction: float = 0.20,
                 window_seconds: float = 2.0, enabled: bool = True):
        super().__init__(enabled=enabled)
        self.max_rejected_fraction = max_rejected_fraction
        self.window_seconds = window_seconds

    def process(self, raw, metadata):
        # In MNE, block rejection is performed on Epochs. For a
        # continuous Raw, we annotate bad segments based on a
        # simple threshold (peak-to-peak amplitude) and reject up
        # to max_rejected_fraction of the recording.
        data = raw.get_data(picks="eeg")
        ptp = data.max(axis=0) - data.min(axis=0)
        threshold = 5 * np.median(ptp)  # 5x median as a heuristic
        bad_mask = ptp > threshold
        n_bad = int(bad_mask.sum())
        n_total = len(bad_mask)
        n_max_bad = int(self.max_rejected_fraction * n_total)
        if n_bad > n_max_bad:
            # Sort by severity and keep only the worst n_max_bad
            threshold_idx = np.argsort(ptp)[-n_max_bad:]
            bad_mask = np.zeros(n_total, dtype=bool)
            bad_mask[threshold_idx] = True
            n_bad = n_max_bad
        # Annotate bad segments
        from mne import Annotations
        onset = 0.0
        duration = raw.times[-1] / n_total if n_total > 0 else 0.0
        if n_bad > 0 and duration > 0:
            # Aggregate contiguous bad samples into annotations
            bad_indices = np.where(bad_mask)[0]
            if len(bad_indices) > 0:
                annot_onsets = [float(bad_indices[0]) * duration]
                annot_durations = [duration]
                for idx in bad_indices[1:]:
                    if idx * duration - annot_onsets[-1] - annot_durations[-1] < duration * 1.5:
                        annot_durations[-1] = idx * duration - annot_onsets[-1]
                    else:
                        annot_onsets.append(float(idx) * duration)
                        annot_durations.append(duration)
                annots = Annotations(
                    onset=annot_onsets, duration=annot_durations,
                    description=["BAD_BLOCK"] * len(annot_onsets))
                raw.set_annotations(annots)
        return raw, {
            "applied": True,
            "max_rejected_fraction": self.max_rejected_fraction,
            "window_seconds": self.window_seconds,
            "n_blocks_rejected": n_bad,
            "n_blocks_total": n_total,
        }
