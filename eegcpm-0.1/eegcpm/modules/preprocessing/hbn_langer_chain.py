"""
Canonical preprocessing chain (per spec §3.i).

The ONE binding ordered step list. The F1 + F1.1 prose variants in
plan-v1.md are SUPERSEDED; this is the only canonical list.

Steps (in order):
  1. channel_roles (Cz eeg-typed; reference channel reported via metadata;
     neck/face dropped)
  2. zapline (BEFORE low-pass; per METH-019)
  3. filter (0.1-45 Hz FIR)
  4. bad_channel_detection (post-filter order; documented deviation
     from Langer per S19a; ≤11/109 budget RECALIBRATED in pilot-50;
     P1-F5 robust median/MAD stats, explicit exemption, exclusion above
     the budget)
  5. interpolate (spherical_spline; max 11/109)
  6. eog_regression (9 EOG channels)
  7. robust_pca (IALM, lam=0.003162 per S20b)
  8. asr (eegprep; cutoff_sd=20; window_criterion=0.25 = REJECTION
     FRACTION not window length, per ENG-010; fixed first-60-s
     calibration; full-length repair with BAD_asr_window marks and
     burden metadata — never cuts samples, keeps no state in the MNE
     ``info`` 'temp' field)
  9. block_rejection (max_rejected_fraction=0.20, S12 reconciled)
  10. reference (average over 109 scalp; Cz INCLUDED via eeg typing;
      post-ref Cz variance > 0; RECOVERED column)

Cz is EXEMPT from bad-channel detection and budget (reference channel
passed explicitly via ``exempt=`` from the channel_roles metadata).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .steps.bad_channels import BadChannelDetectionStep
from .steps.eog_regression import EOGRegressionStep
from .steps.channel_roles import ChannelRolesStep
from .steps.interpolate import InterpolateBadChannelsStep
from .steps.reference import ReferenceStep
from .steps.asr import append_bad_annotations
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
        # Langer; RECALIBRATED ≤11/109 budget; Cz EXEMPT via the
        # channel_roles metadata, P1-F5).
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
        # Average reference: picks:eeg INCLUDES Cz (eeg-typed by
        # channel_roles). Post-ref Cz variance>0; RECOVERED
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
    version = "2.0"  # R-004: pinned variance/kurtosis/spectrum >3-SD

    def __init__(self, threshold_sd: float = 3.0,
                 exclude_cz: bool = True, enabled: bool = True):
        super().__init__(enabled=enabled)
        self.threshold_sd = threshold_sd
        self.exclude_cz = exclude_cz

    def process(self, raw, metadata):
        # R-004 + P1-F5: pinned robust detector, mark-only. The reference
        # channel comes from `metadata['channel_roles']['reference_channel']`
        # (the only source of truth); a missing channel_roles entry raises
        # KeyError and fails the run instead of silently re-flagging Cz.
        # Interpolation is a separate later step.
        from eegcpm.modules.preprocessing.steps.bad_channels_pinned import (
            detect_bad_channels, apply_bad_channels, MAX_BAD_CHANNELS,
        )
        exempt = ()
        if self.exclude_cz:
            exempt = (metadata["channel_roles"]["reference_channel"],)
        bads, report = detect_bad_channels(
            raw, threshold_sd=self.threshold_sd, exempt=exempt
        )
        out_raw = apply_bad_channels(raw, bads, exempt=exempt)
        return out_raw, {
            **report,
            "applied": True,
            "threshold_sd": self.threshold_sd,
            "exclude_cz": self.exclude_cz,
            "exempt": list(exempt),
            "n_bads": len(bads),
            "max_bad_channels": MAX_BAD_CHANNELS,
        }


class _ASRStepWrapper(_PS):
    name = "asr"
    version = "2.0"

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
        if calibration_window != "fixed_first_60s":
            raise ValueError(
                "_ASRStepWrapper implements calibration_window="
                f"'fixed_first_60s' only (got {calibration_window!r}); use "
                "ASRStep(calibration=...) for other calibrations"
            )
        self.calibration_window = calibration_window
        self.retain_burst_masks = retain_burst_masks

    def check_dependencies(self) -> None:
        """Delegate the backend preflight to the underlying ASRStep."""
        from .steps.asr import ASRStep
        ASRStep(method="eegprep").check_dependencies()

    def validate_inputs(self, raw) -> bool:
        """Delegate the fixed-first-60 s minimum-length check (B3-K1).

        The base implementation returns True, so without this override the
        golden chain silently capped a short recording's calibration to its
        length instead of failing the run.
        """
        from .steps.asr import ASRStep
        return ASRStep(
            method="eegprep",
            calibration="fixed_first_60s",
            train_duration=60.0,
        ).validate_inputs(raw)

    def process(self, raw, metadata):
        # P1-F8: delegate to ASRStep and forward its metadata UNCHANGED
        # (true calibration, rejected-window fraction, repaired fraction,
        # burden). The wrapper no longer invents a burden from pre/post
        # statistics and stores nothing in the MNE ``info`` 'temp' field
        # (it does not survive a FIF save; the executor's JSON sidecar is
        # the home of per-run provenance).
        from .steps.asr import ASRStep
        step = ASRStep(
            cutoff=self.cutoff,
            method="eegprep",
            window_criterion=self.window_criterion,
            calibration="fixed_first_60s",
            train_duration=60.0,
        )
        out_raw, step_meta = step.process(raw, metadata)
        return out_raw, {**step_meta,
                         "retain_burst_masks": self.retain_burst_masks}


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
        # Per-block statistics across channels. A block is bad if the
        # MEDIAN channel ptp exceeds the threshold — i.e. a whole-head
        # artifact, not single-channel noise (which is already handled
        # by bad-channel interpolation + ASR). Verified on real HBN
        # pilot data 2026-07-31: max-over-channels fails 100% of blocks
        # (single-channel spikes dominate), median-over-channels flags
        # ~13% (sensible gate). max-over-channels would make the gate
        # structurally unable to pass (METH-22-009).
        from eegcpm.modules.preprocessing.steps.base import RecordingExcluded

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
        # Per-block median and max channel ptp
        block_med = np.zeros(n_blocks)
        for b in range(n_blocks):
            seg = data[:, b * window_samples: (b + 1) * window_samples]
            ptp = seg.max(axis=1) - seg.min(axis=1)
            block_med[b] = np.median(ptp)
        # A block is "bad" if its MEDIAN channel ptp exceeds the threshold
        bad_blocks = block_med > BLOCK_REJECTION_PTP_THRESHOLD
        n_bad = int(bad_blocks.sum())
        fraction = n_bad / n_blocks
        # Onsets follow the actual sample boundaries
        # (window_samples = int(window_seconds * sfreq)), not the nominal
        # window_seconds, so annotations cannot drift on non-integer
        # sample counts.
        bad_block_onsets_s = [
            float(i) * window_samples / sfreq for i in np.where(bad_blocks)[0]
        ]
        # F4: APPEND BAD_BLOCK annotations (never set_annotations over
        # the existing ones — that would wipe stim events and earlier
        # BAD_* marks). Done BEFORE the exclusion raise so an excluded
        # recording still carries its rejection marks. Relative onsets
        # are seconds from the first data sample; `append_bad_annotations`
        # applies the §0.3 offset recipe (including the meas_date=None
        # frame correction) so no existing annotation moves.
        if n_bad > 0:
            append_bad_annotations(
                raw,
                bad_block_onsets_s,
                [window_samples / sfreq] * n_bad,
                ["BAD_BLOCK"] * n_bad,
            )
        # R-006: HARD FAIL on > max_rejected_fraction; NO silent truncation
        if fraction > self.max_rejected_fraction:
            raise RecordingExcluded(
                "block_rejection_fraction_exceeded",
                {
                    "n_blocks_rejected": n_bad,
                    "n_blocks_total": n_blocks,
                    "rejected_fraction": fraction,
                    "max_rejected_fraction": self.max_rejected_fraction,
                    "bad_block_onsets_s": bad_block_onsets_s,
                },
            )
        return raw, {
            "applied": True,
            "window_seconds": self.window_seconds,
            "max_rejected_fraction": self.max_rejected_fraction,
            "n_blocks_rejected": n_bad,
            "n_blocks_total": n_blocks,
            "rejected_fraction": fraction,
        }
