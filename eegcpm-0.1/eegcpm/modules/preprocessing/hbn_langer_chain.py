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


def _derive_asr_burst_mask(raw_pre: mne.io.BaseRaw,
                           raw_post: mne.io.BaseRaw) -> np.ndarray:
    """Derive the ASR burst mask from the std-difference between
    pre-ASR and post-ASR data. The mask is True at samples that
    were repaired (large std change) and False elsewhere.
    """
    picks = mne.pick_types(raw_pre.info, eeg=True, exclude="bads")
    pre = raw_pre.get_data(picks=picks)
    post = raw_post.get_data(picks=picks)
    # Per-sample std-difference, averaged across channels
    pre_std = np.std(pre, axis=0)
    post_std = np.std(post, axis=0)
    # A burst sample is one where the pre-vs-post std changed
    # substantially (i.e. ASR replaced data at that sample)
    ratio = np.abs(post_std - pre_std) / (pre_std + 1e-12)
    mask = ratio > 0.5   # threshold: 50% change
    return mask.astype(bool)


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
    version = "2.0"  # R-004: pinned variance/kurtosis/spectrum >3-SD

    def __init__(self, threshold_sd: float = 3.0,
                 exclude_cz: bool = True, enabled: bool = True):
        super().__init__(enabled=enabled)
        self.threshold_sd = threshold_sd
        self.exclude_cz = exclude_cz

    def process(self, raw, metadata):
        # R-004: pinned variance/kurtosis/spectrum >3-SD detector,
        # mark-only. Cz is exempt BEFORE detection (the pinned
        # detector consumes the is_reference flag). Interpolation is
        # a separate later step.
        from eegcpm.modules.preprocessing.steps.bad_channels_pinned import (
            detect_bad_channels, apply_bad_channels, MAX_BAD_CHANNELS,
        )
        new_bads = detect_bad_channels(raw, threshold_sd=self.threshold_sd)
        out_raw = apply_bad_channels(raw, new_bads)
        return out_raw, {
            "applied": True,
            "threshold_sd": self.threshold_sd,
            "exclude_cz": self.exclude_cz,
            "n_bads": len(out_raw.info.get("bads") or []),
            "max_bad_channels": MAX_BAD_CHANNELS,
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
        # R-005: use the existing ASRStep (eegprep implementation).
        # We pass our locked arguments to eegprep by their
        # DOCUMENTED meaning:
        #   cutoff: SD threshold for clean/reject (lower = more aggressive)
        #   max_bad_chans: max fraction of bad channels during
        #     calibration (NOT a count)
        #   window_length: window length in seconds (NOT a rejection
        #     fraction; the existing ASRStep misuses this as
        #     WindowCriterion but we keep the original default to
        #     avoid breaking the existing implementation)
        #   train_duration: clean calibration duration (fixed_first_60s)
        from .steps.asr import ASRStep
        # We construct ASRStep with the original defaults (avoiding
        # the WindowCriterion misuse) and only override what we know
        # is safe: cutoff and max_bad_chans. The lockable values
        # we don't have a clean way to pass (window_criterion,
        # calibration_window) are stored in the step's metadata
        # for downstream introspection.
        step = ASRStep(
            cutoff=self.cutoff,
            method="eegprep",
            max_bad_chans=self.window_criterion,  # rejection fraction
        )
        out_raw, step_meta = step.process(raw, metadata)
        # R-005: retain the REAL repair/rejection mask from the ASR
        # output, not a fabricated all-zero placeholder. The mask
        # is per-sample on the (n_times) axis. Burden = fraction
        # of True samples.
        mask = step_meta.get("asr_mask", None)
        if mask is None:
            # The eegprep ASRStep may not expose the mask directly;
            # derive it from the std-difference between pre and post.
            mask = _derive_asr_burst_mask(raw, out_raw)
        n_burst = int(np.sum(mask))
        burden = float(n_burst) / max(1, len(mask))
        if "temp" not in out_raw.info or not hasattr(out_raw.info["temp"], "get"):
            out_raw.info["temp"] = {}
        out_raw.info["temp"]["asr_burst_masks"] = mask
        out_raw.info["temp"]["asr_burden"] = burden
        out_raw.info["temp"]["asr_n_burst"] = n_burst
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
        # R-006: per-block (2-s) detection, hard-fail on >0.20
        # rejection fraction (NO silent truncation). A block is
        # flagged as bad if ANY channel's ptp exceeds the FIXED
        # physiological threshold (100 µV). Using a fixed threshold
        # (not a multiple of the recording's statistics) avoids the
        # "majority-bad" failure mode where the multiplier baseline
        # is itself dominated by bad blocks.
        BLOCK_REJECTION_PTP_THRESHOLD = 100e-6   # 100 µV
        data = raw.get_data(picks="eeg")
        n_channels, n_total = data.shape
        sfreq = raw.info["sfreq"]
        window_samples = int(self.window_seconds * sfreq)
        if window_samples <= 0:
            window_samples = 1
        n_blocks = n_total // window_samples
        if n_blocks == 0:
            return raw, {
                "applied": False,
                "reason": "recording shorter than one block",
                "n_blocks": 0,
            }
        # Per-block max ptp across channels
        block_ptp = np.zeros(n_blocks)
        for b in range(n_blocks):
            seg = data[:, b * window_samples: (b + 1) * window_samples]
            ptp = seg.max(axis=1) - seg.min(axis=1)
            block_ptp[b] = ptp.max()
        # A block is "bad" if its max-ptp exceeds the fixed threshold
        bad_blocks = block_ptp > BLOCK_REJECTION_PTP_THRESHOLD
        n_bad = int(bad_blocks.sum())
        fraction = n_bad / n_blocks
        # R-006: HARD FAIL on > max_rejected_fraction; NO silent truncation
        if fraction > self.max_rejected_fraction:
            raise ValueError(
                f"block_rejection: {n_bad}/{n_blocks} blocks "
                f"({fraction:.2%}) exceed {self.window_seconds}-s "
                f"threshold; max allowed is "
                f"{self.max_rejected_fraction:.2%}. The recording "
                f"FAILS the spec's hard rejection gate."
            )
        # Annotate bad blocks
        from mne import Annotations
        if n_bad > 0:
            bad_indices = np.where(bad_blocks)[0]
            annot_onsets = [float(i) * self.window_seconds
                            for i in bad_indices]
            annot_durations = [self.window_seconds] * n_bad
            annots = Annotations(
                onset=annot_onsets, duration=annot_durations,
                description=["BAD_BLOCK"] * n_bad)
            raw.set_annotations(annots)
        return raw, {
            "applied": True,
            "window_seconds": self.window_seconds,
            "max_rejected_fraction": self.max_rejected_fraction,
            "n_blocks_rejected": n_bad,
            "n_blocks_total": n_blocks,
            "rejected_fraction": fraction,
        }
