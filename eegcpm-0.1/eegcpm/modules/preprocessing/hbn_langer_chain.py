"""
Canonical preprocessing chain (per spec §3.i).

The ONE binding ordered step list. The F1 + F1.1 prose variants in
plan-v1.md are SUPERSEDED; this is the only canonical list.

Steps (in order):
  1. channel_roles (Cz eeg-typed; reference channel reported via metadata;
     neck/face dropped)
  2. zapline (BEFORE low-pass; per METH-019; EEG picks only — the filter
     below band-passes the EOG channels, which is what the EOG regressors
     need, B6 rev2 NEW-1)
  3. filter (0.1-45 Hz FIR over the EEG **and EOG** picks, not stim/misc —
     NEW-1: filtering only the EEG left DC drift and 60 Hz line noise in the
     EOG regressors, and ``eog_regression`` then re-injected both into the
     cleaned EEG after zapline/low-pass)
  4. bad_channel_detection (post-filter order; documented deviation
     from Langer per S19a; ≤11/109 budget RECALIBRATED in pilot-50;
     P1-F5 robust median/MAD stats, explicit exemption, exclusion above
     the budget; B4-D2 blink-aware log-kurtosis + relative near-flat floor)
  5. interpolate (spherical_spline; max 11/109)
  6. eog_regression (9 EOG channels; regressors band-passed by step 3)
  7. asr (eegprep; EEGLAB-standard input high-pass ``clean_drifts(0.25, 0.75)``
     before ASR, no add-back — the write-back is the high-passed repaired data
     and ``info['highpass']`` is updated to 0.75 Hz; the ASR pseudo-inverse
     runs under the MATLAB-equivalent tolerance (R1 — the actual fix for the
     real-data blow-ups);
     ``max_dims=0.66``; cutoff_sd=20 = the EEGLAB *conservative* setting
     (aggressive 3, default 5, GUI default 20; Chang et al. 2019 optimal 20-30),
     window_criterion=0.25 = REJECTION FRACTION not window length, per ENG-010;
     fixed first-60-s calibration; full-length repair with BAD_asr_window marks
     and burden metadata — never cuts samples, keeps no state in the MNE
     ``info`` 'temp' field). Plan §12.6 decision (2026-10-16; justification
     corrected by the ASR scoped audit R2): the drift high-pass exists for
     **EEGLAB-standard input + lower repair burden** (``repaired_fraction``
     0.762 → 0.481 on the e2e fixture), **not** as blow-up prevention — the
     blow-up fix is the pinv tolerance (R1). Per-dataset tuning of these ASR
     parameters goes through
     ``PreprocessingSpec.chain_params`` (``HbnLangerParams``).
  8. block_rejection (max_rejected_fraction=0.20, S12 reconciled)
  9. reference (average over 109 scalp; Cz INCLUDED via eeg typing;
     post-ref Cz variance > 0 enforced by the step; RECOVERED column)

**Spec deviation (P1-F2, Checkpoint A decision (a), 2026-10-15):**
``RobustPCAStep`` is NOT part of this chain. The pinned IALM solve
(lam=0.003162) absorbs most of the injected ERP into the sparse component
it discards (synthetic retention 2-28% in the audit probe; ~0.19-0.67 in
``tests/unit/test_robust_pca_erp.py``), so it must not run on ERP data.
The step class stays importable for standalone use (it was never registered).
The deviation is recorded in ``PreprocessingPipeline.get_config()
['spec_deviations']`` (see ``HBN_LANGER_SPEC_DEVIATIONS``) and in the
resolved-config hash.

Cz is EXEMPT from bad-channel detection and budget (reference channel
passed explicitly via ``exempt=`` from the channel_roles metadata).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .steps.bad_channels import BadChannelDetectionStep
from .steps.eog_regression import EOGRegressionStep
from .steps.channel_roles import ChannelRolesStep
from .steps.interpolate import InterpolateBadChannelsStep
from .steps.reference import ReferenceStep
from .steps.asr import append_bad_annotations
from .steps.zapline import ZaplineStep


#: Explicit, machine-readable spec deviations of the canonical chain. They
#: are part of ``PreprocessingPipeline.get_config()`` and therefore of the
#: golden resolved-config hash (plan §B6, Checkpoint A).
HBN_LANGER_SPEC_DEVIATIONS: List[str] = [
    "P1-F2: robust_pca removed (destroys ERP; synthetic retention 2-28%)",
]

#: R-006 block-rejection gate: a 2-s block is flagged ``BAD_BLOCK`` when its
#: MEDIAN channel peak-to-peak amplitude exceeds this fixed physiological
#: threshold (100 µV). Named at module level and read at call time, so
#: ``_BlockRejectionStepWrapper.get_config()`` — and therefore the golden
#: resolved-config hash — covers the number that actually decides the gate
#: (F2, final-verification-audit fix round). A fixed threshold (not a multiple
#: of the recording's own statistics) avoids the "majority-bad" failure mode
#: where the multiplier baseline is itself dominated by bad blocks.
BLOCK_REJECTION_PTP_THRESHOLD: float = 100e-6   # 100 µV


def build_hbn_langer_chain(
    asr_cutoff: float = 20.0,
    asr_drift_transition_hz: Optional[Tuple[float, float]] = (0.25, 0.75),
    asr_max_dims: float = 0.66,
    asr_calibration: str = "fixed_first_60s",
) -> List[Any]:
    """Build the canonical HBN-Langer preprocessing chain.

    Returns the ordered list of ProcessingStep instances. The
    config-hash of the chain is determined by this function; any
    change here triggers a new golden-config-hash.

    9 steps — ``RobustPCAStep`` is deliberately absent (P1-F2 spec
    deviation, see the module docstring and ``HBN_LANGER_SPEC_DEVIATIONS``).

    Parameters
    ----------
    asr_cutoff : float
        ASR cutoff (EEGLAB: aggressive 3, default 5, conservative 20).
        Default 20 = the conservative/GUI-default setting.
    asr_drift_transition_hz : (float, float) or None
        Transition band of the EEGLAB-standard ``clean_drifts`` high-pass
        applied to the ASR input (default ``(0.25, 0.75)`` — EEGLAB's own
        ``clean_artifacts`` band). ``None`` disables it (non-EEGLAB regime;
        the step warns).
    asr_max_dims : float
        Maximum fraction of dimensions ASR may reconstruct (default 0.66).
    asr_calibration : {'fixed_first_60s', 'data_driven'}
        ``'fixed_first_60s'`` (default): the canonical HBN-Langer
        calibration. ``'data_driven'``: EEGLAB's automatic clean-window
        calibration (eegprep selects the calibration data itself) — the
        built-in per-recording adaptivity, opt-in per dataset via
        ``PreprocessingSpec.chain_params``.

    The keyword names are exactly the ``HbnLangerParams`` fields, so
    ``build_hbn_langer_chain(**spec.chain_params.model_dump())`` is the
    per-dataset tuning path (plan §12.6).

    Returns a :class:`~eegcpm.modules.preprocessing.pipeline_builder
    .ResolvedChainSteps`: the step list carries ``HBN_LANGER_SPEC_DEVIATIONS``
    so every construction path — the chain name, the spec's
    ``pipeline_steps()`` and a direct call — resolves to the same config (and
    hash).
    """
    # Lazy import: ``pipeline_builder`` is the consumer and imports this module
    # lazily in turn, so a module-level import here would risk a cycle.
    from eegcpm.modules.preprocessing.pipeline_builder import ResolvedChainSteps

    return ResolvedChainSteps([
        ChannelRolesStep(),
        ZaplineStep(fline=60.0, nremove=None, nfft=8192),
        # Filter is 0.1-45 Hz FIR (per A-RT24; NOT 40 Hz from code.md) over
        # the EEG **and EOG** picks (B6 rev2 NEW-1). Filtering only the EEG
        # left the EOG regressors with DC drift and 60 Hz line noise, which
        # `eog_regression` then added back into the cleaned EEG (real-data
        # probe: >50 Hz power x33, 59-61 Hz x~14,000). Stim/misc channels
        # are not filtered (no downstream regressor reads them).
        _FilterStepWrapper(l_freq=0.1, h_freq=45.0,
                           method="fir", fir_design="firwin",
                           picks=("eeg", "eog")),
        # Bad-channel detection: post-filter order (S19a deviation from
        # Langer; RECALIBRATED ≤11/109 budget; Cz EXEMPT via the
        # channel_roles metadata, P1-F5).
        _BadChannelDetectorWrapper(threshold_sd=3.0, exclude_cz=True),
        InterpolateBadChannelsStep(max_bad_percent=11/109*100),
        EOGRegressionStep(),
        # ASR via eegprep; cutoff_sd=20 (EEGLAB *conservative*; aggressive 3,
        # default 5, GUI default 20); window_criterion=0.25 is the REJECTION
        # FRACTION, NOT a window length (per ENG-010). The ASR input is
        # high-passed with clean_drifts (EEGLAB-standard, plan §12.6) and the
        # write-back is the high-passed repaired data (no add-back).
        # retain_burst_masks=True so the ASR-burden covariate is
        # computable downstream.
        _ASRStepWrapper(cutoff=asr_cutoff, window_criterion=0.25,
                          calibration_window=asr_calibration,
                          drift_transition_hz=asr_drift_transition_hz,
                          max_dims=asr_max_dims,
                          retain_burst_masks=True),
        # Block rejection: max_rejected_fraction 0.20 (S12 reconciled
        # with safety.md EC≥160/200).
        _BlockRejectionStepWrapper(max_rejected_fraction=0.20,
                                       window_seconds=2.0),
        # Average reference: picks:eeg INCLUDES Cz (eeg-typed by
        # channel_roles). The step asserts post-ref Cz variance > 0
        # (P1-F7: the check is implemented, not a YAML no-op);
        # RECOVERED column downstream.
        ReferenceStep(type="average", projection=False,
                      check_reference_channel_variance="Cz"),
    ], HBN_LANGER_SPEC_DEVIATIONS)


# --- Wrappers around existing steps to pin the spec parameters ---


from .steps.base import ProcessingStep as _PS


class _FilterStepWrapper(_PS):
    name = "filter"
    version = "2.0"  # 2.0: EEG + EOG picks (B6 rev2 NEW-1)

    def __init__(self, l_freq: float, h_freq: float,
                 method: str = "fir", fir_design: str = "firwin",
                 picks: Sequence[str] = ("eeg", "eog"),
                 enabled: bool = True):
        super().__init__(enabled=enabled)
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.method = method
        self.fir_design = fir_design
        # NEW-1: the filter band-passes the EOG channels together with the
        # EEG, so the EOG regressors used by `eog_regression` (step 6) carry
        # no DC drift and no line noise. This is part of the resolved config
        # (and therefore of the golden hash).
        self.picks = list(picks)

    def process(self, raw, metadata):
        raw.filter(l_freq=self.l_freq, h_freq=self.h_freq,
                   method=self.method, fir_design=self.fir_design,
                   picks=self.picks, phase="zero", verbose=False)
        return raw, {
            "applied": True,
            "l_freq": self.l_freq,
            "h_freq": self.h_freq,
            "method": self.method,
            "picks": list(self.picks),
        }

    def get_config(self) -> Dict[str, Any]:
        """Get configuration (all constructor parameters, P1-F7)."""
        config = super().get_config()
        config.update({
            "l_freq": self.l_freq,
            "h_freq": self.h_freq,
            "method": self.method,
            "fir_design": self.fir_design,
            "picks": list(self.picks),
        })
        return config


class _BadChannelDetectorWrapper(_PS):
    name = "bad_channel_detection"
    version = "3.0"  # R-004 pinned detector; 3.0: B4-D2 blink-aware detector
    # + B6 rev2 (log-Pearson kurtosis, band-passed blink channel, algorithm
    # constants in get_config -> golden hash, NEW-3/NEW-4)

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

    def get_config(self) -> Dict[str, Any]:
        """Get configuration (all constructor parameters, P1-F7).

        NEW-4 (B6 rev2): the detector's module-level algorithm constants are
        returned under ``algorithm`` so the resolved-config hash (and the
        golden hash) covers them — the detector's numerics live in
        ``steps/bad_channels_pinned.py``, not in the constructor, so hashing
        the constructor parameters alone proved nothing about the algorithm.
        The values are read from the module at call time, so a change to any
        of them changes the hash (see the sensitivity test in
        ``test_config_hash.py``).
        """
        from eegcpm.modules.preprocessing.steps import bad_channels_pinned as bcp

        config = super().get_config()
        config.update({
            "threshold_sd": self.threshold_sd,
            "exclude_cz": self.exclude_cz,
            "algorithm": {
                "max_bad_channels": int(bcp.MAX_BAD_CHANNELS),
                "tails": dict(bcp.TAILS),
                "hf_band_hz": list(bcp.HF_BAND_HZ),
                "flat_variance_floor": bcp.FLAT_VARIANCE_FLOOR,
                "near_flat_variance_ratio": bcp.NEAR_FLAT_VARIANCE_RATIO,
                "degeneracy_rel_eps": bcp.DEGENERACY_REL_EPS,
                "blink_mask_half_width_s": bcp.BLINK_MASK_HALF_WIDTH_S,
                "blink_max_masked_fraction": bcp.BLINK_MAX_MASKED_FRACTION,
                "min_blink_free_fraction": bcp.MIN_BLINK_FREE_FRACTION,
                "blink_band_hz": list(bcp.BLINK_BANDPASS_HZ),
                "min_blink_rate_per_minute": bcp.MIN_BLINK_RATE_PER_MINUTE,
            },
        })
        return config


class _ASRStepWrapper(_PS):
    name = "asr"
    version = "2.5"  # 2.5: ASR scoped-audit R1 (MATLAB-equivalent pinv
    # tolerance, exposed in get_config -> golden hash); 2.4: plan §12.6
    # (2026-10-16) — EEGLAB-standard ASR: clean_drifts(drift_transition_hz) on
    # the ASR input, max_dims forwarded, data_driven calibration_window
    # accepted; 2.3: F3 (final-verification-audit fix round) — the effective
    # window_tolerances + ASR_OUTPUT_BLOWUP_FACTOR are read from ``steps.asr``
    # at call time and exposed in get_config (golden hash); 2.2: B7 rev2 fix 7
    # (re-interpolate the ASR-excluded interpolated channels) + B7-N2 output
    # blow-up guard; 2.1: B6 rev2 NEW-4 (method/train_duration/
    # ref_maxbadchannels exposed in get_config -> golden hash)

    def __init__(self, cutoff: float = 20.0,
                 window_criterion: float = 0.25,
                 calibration_window: str = "fixed_first_60s",
                 drift_transition_hz: Optional[Tuple[float, float]] = (0.25, 0.75),
                 max_dims: float = 0.66,
                 retain_burst_masks: bool = True,
                 enabled: bool = True):
        super().__init__(enabled=enabled)
        self.cutoff = cutoff
        # window_criterion is a REJECTION FRACTION [0, 1] — NOT a
        # window length. Per ENG-010: code.md misread it.
        self.window_criterion = window_criterion
        if calibration_window not in ("fixed_first_60s", "data_driven"):
            raise ValueError(
                "_ASRStepWrapper implements calibration_window="
                f"'fixed_first_60s' or 'data_driven' (got {calibration_window!r})"
            )
        self.calibration_window = calibration_window
        # Plan §12.6: the EEGLAB-standard input high-pass and the maximum
        # reconstructed dimensionality are part of the chain's resolved config.
        # Validation (None/'off'/pair, 0 < max_dims <= 1) is delegated to
        # ASRStep's constructor checks via a throwaway instance, so the two
        # layers cannot disagree.
        from .steps.asr import ASRStep
        probe = ASRStep(drift_transition_hz=drift_transition_hz,
                        max_dims=max_dims)
        self.drift_transition_hz = probe.drift_transition_hz
        self.max_dims = probe.max_dims
        self.retain_burst_masks = retain_burst_masks
        # NEW-4 (B6 rev2): the delegated ASRStep constants are pinned here so
        # `get_config()` (and the golden resolved hash) covers them.
        self.method = "eegprep"
        self.train_duration = 60.0

    def check_dependencies(self) -> None:
        """Delegate the backend preflight to the underlying ASRStep."""
        from .steps.asr import ASRStep
        ASRStep(method=self.method).check_dependencies()

    def validate_inputs(self, raw) -> bool:
        """Delegate the calibration minimum-length check (B3-K1).

        The base implementation returns True, so without this override the
        golden chain silently capped a short recording's calibration to its
        length instead of failing the run. The fixed-first-60 s calibration
        needs ``train_duration`` seconds; the data-driven calibration selects
        its own window and has no fixed minimum.
        """
        from .steps.asr import ASRStep
        return ASRStep(
            method=self.method,
            calibration=self.calibration_window,
            train_duration=self.train_duration,
        ).validate_inputs(raw)

    def process(self, raw, metadata):
        # P1-F8: delegate to ASRStep and forward its metadata UNCHANGED
        # (true calibration, rejected-window fraction, repaired fraction,
        # burden). The wrapper no longer invents a burden from pre/post
        # statistics and stores nothing in the MNE ``info`` 'temp' field
        # (it does not survive a FIF save; the executor's JSON sidecar is
        # the home of per-run provenance).
        # F3: window_tolerances is passed explicitly from the module constant
        # (read at call time), so the value reported by `get_config()` is the
        # value that decides the BAD_asr_window marks.
        from .steps import asr as asr_module
        step = asr_module.ASRStep(
            cutoff=self.cutoff,
            method=self.method,
            window_criterion=self.window_criterion,
            window_tolerances=asr_module.ASR_WINDOW_TOLERANCES,
            calibration=self.calibration_window,
            train_duration=self.train_duration,
            # Plan §12.6: EEGLAB-standard input high-pass + max_dims.
            drift_transition_hz=self.drift_transition_hz,
            max_dims=self.max_dims,
        )
        out_raw, step_meta = step.process(raw, metadata)
        return out_raw, {**step_meta,
                         "retain_burst_masks": self.retain_burst_masks}

    def get_config(self) -> Dict[str, Any]:
        """Get configuration (all constructor parameters, P1-F7).

        NEW-4 (B6 rev2, re-audit fix): the wrapper delegates to
        ``ASRStep(calibration='fixed_first_60s')``, so the resolved config must
        describe the **fixed** path truthfully: the calibration reference is the
        first ``train_duration`` seconds of the data (passed as an ndarray to
        ``clean_asr(ref_maxbadchannels=...)``) and ``ref_tolerances`` is not
        passed at all (eegprep default). Reporting the data-driven constants
        here claimed a calibration the canonical chain never runs. With
        ``calibration_window='data_driven'`` (plan §12.6 per-dataset option) the
        reported calibration fields switch to the data-driven constants the
        step actually passes.

        F3 (final-verification-audit fix round): the remaining numerics that
        live in ``steps/asr.py`` — ``ASR_WINDOW_TOLERANCES`` (the
        ``clean_windows`` z-thresholds that decide the ``BAD_asr_window``
        marks) and ``ASR_OUTPUT_BLOWUP_FACTOR`` (the B7-N2 guard) — are read
        from the module **at call time** and included, so a change to either
        changes the resolved-config hash (see the sensitivity test in
        ``test_config_hash.py``).

        R1 (ASR scoped audit, 2026-10-16): ``PINV_TOLERANCE_SPEC`` — the
        MATLAB-equivalent pseudo-inverse tolerance ``clean_asr`` runs under
        (it changes the repaired data) — is read from the module at call time
        and included for the same reason.

        Plan §12.6 (2026-10-16): ``drift_transition_hz`` (the EEGLAB-standard
        ``clean_drifts`` high-pass applied to the ASR input; ``None`` = the
        non-EEGLAB regime) and ``max_dims`` are part of the resolved config —
        changing either changes the golden hash, so a per-dataset
        ``chain_params`` override is visible in provenance.
        """
        from .steps import asr as asr_module

        config = super().get_config()
        config.update({
            "cutoff": self.cutoff,
            "window_criterion": self.window_criterion,
            "window_tolerances": [
                float(t) for t in asr_module.ASR_WINDOW_TOLERANCES
            ],
            "output_blowup_factor": float(asr_module.ASR_OUTPUT_BLOWUP_FACTOR),
            "calibration_window": self.calibration_window,
            "retain_burst_masks": self.retain_burst_masks,
            "method": self.method,
            "train_duration": self.train_duration,
            # Plan §12.6: EEGLAB-standard ASR input high-pass + dimensionality.
            "drift_transition_hz": (
                None if self.drift_transition_hz is None
                else [float(t) for t in self.drift_transition_hz]
            ),
            "max_dims": float(self.max_dims),
            # R1 (ASR scoped audit): the eegprep ASR pseudo-inverse runs under
            # the MATLAB-equivalent tolerance (it changes the repaired data).
            "pinv_tolerance": asr_module.PINV_TOLERANCE_SPEC,
        })
        if self.calibration_window == "fixed_first_60s":
            config.update({
                "calibration_reference": "first_train_duration_s_of_data",
                "ref_maxbadchannels": "calibration_array",
                "ref_tolerances": "eegprep_default",
            })
        else:
            config.update({
                "calibration_reference": "eegprep_clean_windows_selection",
                "ref_maxbadchannels": float(
                    asr_module.DATA_DRIVEN_REF_MAXBADCHANNELS),
                "ref_tolerances": [
                    float(t) for t in asr_module.DATA_DRIVEN_REF_TOLERANCES
                ],
            })
        return config


class _BlockRejectionStepWrapper(_PS):
    name = "block_rejection"
    version = "1.1"  # 1.1: F2 (final-verification-audit fix round) — the
    # 100 µV ptp threshold is a module constant (read at call time) exposed in
    # get_config -> golden hash; 1.0: R-006 median-ptp gate

    def __init__(self, max_rejected_fraction: float = 0.20,
                 window_seconds: float = 2.0, enabled: bool = True):
        super().__init__(enabled=enabled)
        self.max_rejected_fraction = max_rejected_fraction
        self.window_seconds = window_seconds

    def process(self, raw, metadata):
        # R-006: per-block (2-s) detection, hard-fail on >0.20
        # rejection fraction (NO silent truncation). A block is
        # flagged as bad if its MEDIAN channel ptp exceeds the FIXED
        # physiological threshold ``BLOCK_REJECTION_PTP_THRESHOLD``
        # (100 µV, module constant, read at call time). Using a fixed
        # threshold (not a multiple of the recording's statistics)
        # avoids the "majority-bad" failure mode where the multiplier
        # baseline is itself dominated by bad blocks.
        # Per-block statistics across channels. A block is bad if the
        # MEDIAN channel ptp exceeds the threshold — i.e. a whole-head
        # artifact, not single-channel noise (which is already handled
        # by bad-channel interpolation + ASR). Verified on real HBN
        # pilot data 2026-07-31: max-over-channels fails 100% of blocks
        # (single-channel spikes dominate), median-over-channels flags
        # ~13% (sensible gate). max-over-channels would make the gate
        # structurally unable to pass (METH-22-009).
        from eegcpm.modules.preprocessing.steps.base import RecordingExcluded

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
                "ptp_threshold_v": float(BLOCK_REJECTION_PTP_THRESHOLD),
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
            "ptp_threshold_v": float(BLOCK_REJECTION_PTP_THRESHOLD),
            "n_blocks_rejected": n_bad,
            "n_blocks_total": n_blocks,
            "rejected_fraction": fraction,
        }

    def get_config(self) -> Dict[str, Any]:
        """Get configuration (all constructor parameters, P1-F7).

        F2 (final-verification-audit fix round): the 100 µV ptp threshold that
        decides the gate lives in the module constant
        ``BLOCK_REJECTION_PTP_THRESHOLD`` (read at call time by ``process`` and
        here), so it is part of the resolved config — and the golden hash.
        """
        config = super().get_config()
        config.update({
            "max_rejected_fraction": self.max_rejected_fraction,
            "window_seconds": self.window_seconds,
            "ptp_threshold_v": float(BLOCK_REJECTION_PTP_THRESHOLD),
        })
        return config
