"""
Artifact Subspace Reconstruction (ASR) step.

ASR **repairs** artifacts at full length. eegprep's ``clean_asr`` reconstructs
the corrupted signal subspace without cutting samples, so after this step the
recording keeps its sample count, its event samples, its ``info``
(``meas_date`` / ``highpass`` / ``lowpass`` / ``bads`` / montage) and its
existing annotations. Windows flagged by ``clean_windows`` are **annotated**
(``BAD_asr_window``) instead of being removed, and repaired (burst-corrected)
intervals are recorded in the step metadata only (``repaired_intervals_s`` /
``repaired_fraction``) — repaired data are correct data, not bad data.

**EEGLAB-standard input high-pass (2026-10-16 design decision, plan §12.6;
justification corrected 2026-10-16 — ASR scoped audit R2).**
``clean_asr`` requires zero-mean (high-passed) input: its docstring states the
data are *"assumed to be zero mean, e.g., appropriately high-passed (e.g.
>0.5Hz or with a 0.5Hz - 1.0Hz transition band)"*, and EEGLAB's
``clean_artifacts`` applies ``clean_drifts(EEG, (0.25, 0.75))`` internally
before ASR (verified in the installed eegprep source,
``plugins/clean_rawdata/clean_artifacts.py`` line ~203). The filter's
justification is **EEGLAB-standard input + lower repair burden**: on the chain's
0.1-Hz data ASR rewrote most of a *clean* recording (``repaired_fraction``
0.762 on the e2e fixture; 0.481 with the filter), i.e. most of the old "repair"
was low-frequency content EEGLAB removes before ASR. It is **not** the fix for
the real-data blow-ups: the auditor reproduced ``asr_output_magnitude_blowup``
on 4 of 6 pilot recordings that reach ASR *with this filter on* — that failure
mode is eegprep's default pseudo-inverse tolerance (R1 below). This step
therefore applies ``eegprep.clean_drifts(eeg, drift_transition_hz)`` to the ASR
**input** (default ``(0.25, 0.75)``, EEGLAB's own transition band) and passes
``max_dims`` (default 0.66, eegprep/EEGLAB default) to ``clean_asr``.
``drift_transition_hz = None``/``'off'`` skips the filter (a warning is emitted
and recorded).

**No add-back + truthful ``info['highpass']``:** the write-back is the
high-passed repaired data — exactly EEGLAB's behaviour (``clean_artifacts``
returns the high-passed output and does not add the pre-high-pass signal back).
Because the write-back really is high-passed, the step updates
``info['highpass']`` to the filter's passband edge (``drift_transition_hz[1]``,
default 0.75 Hz) — the convention MNE's own ``filter`` uses — so the FIF no
longer claims the chain's 0.1 Hz passband for data that no longer has it (ASR
scoped audit R4). The metadata keeps the full transition band
(``drift_transition_hz``, ``drift_filter_applied``, ``info_highpass_hz``).
``repaired_fraction`` is measured against the **ASR input** (post-drift) so it
keeps meaning "samples ASR rewrote", and ``drift_filter_modified_fraction``
records the separate high-pass effect.

**R1 (ASR scoped audit, 2026-10-16) — MATLAB-equivalent pinv tolerance.**
eegprep 0.4.0's ASR reconstruction calls ``np.linalg.pinv`` with numpy's
default ``rcond=1e-15`` (``plugins/clean_rawdata/asr_process.py`` line ~214,
via the module-global ``finite_pinv(Q)``), while MATLAB's ``pinv`` uses the
**absolute** tolerance ``max(size(A)) * eps(norm(A))`` (≈2.2e-14 for ~100
channels; ``eps`` = ``np.spacing``, ``norm`` = largest singular value). The
near-singular directions numpy inverts make ``clean_asr`` return ~1e15 µV
instead of repaired data on real recordings. This step runs ``clean_asr``
under a temporary patch of exactly that call site (replacing the
``finite_pinv`` module global for the duration of the call, restored in
``finally``) and records ``pinv_tolerance: 'matlab_max_size_eps(norm)'``. The
patch is guarded: if the installed eegprep does not expose the expected call
site, the step refuses to run (``eegprep_pinv_assumption_broken``) instead of
silently falling back to the default tolerance.

Supports both eegprep (EEGLab port) and asrpy implementations. The asrpy
branch is kept functional but is not part of any golden chain: it does not
annotate rejected windows, does not report the eegprep burden fields, and does
not apply the drift high-pass (recorded honestly in its metadata).

Probe findings (2026-10-14, eegprep 0.4.0 / mne 1.13.2, recorded per the
skill's "verify, never trust memory" rule; probe under ``/tmp/opencode/``):

- ``eegprep.eeg_mne2eeg(raw)`` returns an EEGLAB dict whose ``data`` is
  ``(n_channels, n_times)`` in **µV** (MNE Volts × 1e6), dtype float32.
- ``eegprep.clean_asr(EEG, ...)`` returns a **new** EEG dict (the input is
  deep-copied internally) with ``data`` at the same ``(n_channels, n_times)``
  shape (never cut), dtype float64, still in µV.
- ``clean_asr(..., ref_maxbadchannels=<ndarray>)`` uses the array *itself* as
  the calibration reference data and requires shape ``(n_channels,
  n_samples)`` — a transposed array raises ``ValueError``. That is how the
  ``fixed_first_60s`` calibration is implemented (the array is the first
  ``train_duration`` seconds of the µV data).
- ``clean_asr(..., ref_maxbadchannels=0.075, ref_tolerances=(-inf, 5.5))``
  selects the calibration data internally via ``clean_windows`` (data-driven).
- ``eegprep.clean_windows(EEG, max_bad_channels=<fraction>, zthresholds=...)``
  returns ``(EEG, sample_mask)`` with ``sample_mask.shape == (n_times,)``,
  bool, ``True`` = retained. A value in (0, 1) is interpreted as a *fraction
  of bad channels per window*; other values as an absolute channel count.
  The mask is used for annotations only; the data are never cut here.
- ``eegprep.clean_drifts(EEG, transition=(0.25, 0.75))`` (probed 2026-10-16)
  applies a forward-backward Kaiser-windowed FIR high-pass (``transition`` =
  lower/upper edge of the transition band in Hz), mutates and returns the
  *same* EEG dict, and works directly on the ``eeg_mne2eeg`` output (which
  carries an empty ``etc`` dict; ``clean_drifts`` writes its kernel into it).
- ``eegprep.clean_asr(..., max_dims=0.66)`` — the fraction of dimensions ASR
  may reconstruct; 0.66 is the eegprep/EEGLAB default and is now configurable
  through the step and the chain.

**B7-N1 (probed 2026-10-16) — interpolated channels must not enter ASR.**
Spherical-spline interpolation (``InterpolateBadChannelsStep``, chain step 5)
reconstructs each bad channel as an *exact* linear combination of the
remaining channels, so a single interpolated channel makes the ASR calibration
covariance exactly singular (measured condition number 1.2e15 on the EGI-129
fixture with one interpolated channel; 0 interpolated → 35). ``eegprep``'s
``clean_asr`` takes ``sqrtm`` of that matrix (``LinAlgWarning: Matrix is
singular``) and returns ~1e15 µV instead of repaired data; the block-rejection
gate then sees every 2-s block above its 100 µV threshold and the recording is
excluded. **Re-probed 2026-10-16 after the §12.6 drift high-pass and again
after the R1 pinv fix:** the exactly-singular EGI-129 case is still
rank-deficient (it is caught by the ``asr_calibration_covariance_degenerate``
check below, with or without the drift filter), and the *near*-singular
32-channel control (condition number ~2e4) that used to blow up is handled by
the R1 MATLAB-equivalent pinv tolerance (the drift high-pass alone did **not**
fix the real-data blow-ups — see the R1 note above) — so the exclusion is
still required, and the failure mode stays loud either way. This step therefore
excludes the channels a previous ``interpolate``
step reconstructed (``metadata['interpolate']['interpolated_channels']``) from
the ASR **input** — they are synthetic, carry no independent information, and
stay in the recording (the average reference re-mixes them). Excluding them
restores a full-rank covariance (probed: 0 singular values, sane output).
The alternative fix (moving interpolation after ASR) changes the chain
composition and the golden resolved-config hash; it is recorded as a
follow-up, not applied here. **rev2 (B7 rev2 fix 7):** after the repair, the
excluded interpolated channels are **re-interpolated from the ASR-cleaned
data** (same interpolation mode) so the recording stays internally consistent
— their pre-ASR values are built from the pre-repair neighbours and would
otherwise contradict the repaired data (measured 1.02 / 1.09 relative-norm
divergence from a fresh re-interpolation on the EGI-129 chain fixture).

**B7-N2 (probed 2026-10-16) — ASR output magnitude blow-up guard.**
``eegprep.clean_asr`` at ``cutoff=20`` can return garbage at a huge scale
instead of repaired data: the audit measured 2.6e16–5.7e16 µV with moderate
spatial mixing (calibration covariance condition number 9e4–1.4e6, i.e. *not*
exactly singular) — the root cause is the default pseudo-inverse tolerance,
fixed by R1 above. Probed here on a 32-channel fixture with three
near-duplicate channel pairs (condition number 2.2e8): ``clean_asr`` returned
**4.6e15 µV** from a 13 µV input before R1. The guard stays as
defense-in-depth: it covers the ``drift_transition_hz=None`` opt-out, other
degenerate inputs, and any future backend regression, so a blow-up can never
flow into the block-rejection gate, which would see every block above its
100 µV threshold and report the recording as *excluded for block
contamination* — a catastrophic misattribution of a backend failure to the
data. This step therefore fails loudly
(``RuntimeError``, reason ``asr_output_magnitude_blowup``) whenever the
repaired output's maximum absolute amplitude exceeds
:data:`ASR_OUTPUT_BLOWUP_FACTOR` × the input maximum, and records the
calibration-covariance condition number (fixed calibration) in the metadata.
The guard sits **inside** the ASR step, so a blow-up can never be reported as
a block-rejection exclusion.

Author: EEGCPM Development Team
Created: 2025-12
"""

import warnings
from typing import Any, Dict, Optional, Sequence, Tuple

import mne
import numpy as np

from .base import ProcessingStep


# Data-driven calibration constants (``ASRStep(calibration='data_driven')``),
# passed to ``eegprep.clean_asr`` (probed 2026-10-14, eegprep 0.4.0). Named so
# the hbn_langer ASR wrapper can expose them in ``get_config()`` (B6 rev2
# NEW-4) — the resolved-config hash must cover the ASR constants, not only the
# wrapper's constructor arguments.
DATA_DRIVEN_REF_MAXBADCHANNELS: float = 0.075
DATA_DRIVEN_REF_TOLERANCES: Tuple[float, float] = (-np.inf, 5.5)

#: B7-N2: an ASR output whose maximum |amplitude| exceeds this factor times the
#: input maximum is a backend blow-up (``sqrtm`` of a degenerate calibration
#: covariance), not a repair — fail loudly instead of letting the block gate
#: misattribute it to the data. Probed: 4.6e15 µV output from a 13 µV input on
#: a near-duplicate-channel fixture (ratio 3.6e14), while legitimate ASR
#: repairs never *increase* the maximum amplitude by even 2×.
ASR_OUTPUT_BLOWUP_FACTOR: float = 1e3

#: Canonical ``(lower, upper)`` RMS z-thresholds passed to
#: ``clean_windows(zthresholds=...)``; they decide which windows are marked
#: ``BAD_asr_window`` (``(-inf, 7.0)`` flags only over-power windows). Named at
#: module level and read at call time so ``_ASRStepWrapper.get_config()`` — and
#: therefore the golden resolved-config hash — covers the value the canonical
#: chain actually uses (F3, final-verification-audit fix round). This is the
#: default of :class:`ASRStep` and what the hbn_langer wrapper passes.
ASR_WINDOW_TOLERANCES: Tuple[float, float] = (-np.inf, 7.0)

#: R1 (ASR scoped audit, 2026-10-16): metadata/get_config value recording that
#: the ASR pseudo-inverse ran with MATLAB's default ``pinv`` tolerance
#: (``max(size(A)) * eps(norm(A))``) instead of numpy's ``rcond=1e-15``. Named
#: at module level so the hbn_langer wrapper's ``get_config()`` — and therefore
#: the golden resolved-config hash — covers it.
PINV_TOLERANCE_SPEC: str = "matlab_max_size_eps(norm)"

#: eegprep release the R1 call-site assumption was verified against. The
#: runtime guard checks the installed **source** (not only the version, which a
#: patch release can keep while changing the call); a mismatch between the
#: installed version and this value is recorded in the metadata, and a changed
#: call site refuses the run loudly.
EEGPREP_PINV_VERIFIED_VERSION: str = "0.4.0"


def _matlab_pinv_tolerance(matrix: np.ndarray) -> float:
    """MATLAB's default ``pinv`` tolerance: ``max(size(A)) * eps(norm(A))``.

    MATLAB's ``eps(x)`` is the distance from ``x`` to the next larger float
    (``np.spacing``) and ``norm(A)`` is the matrix 2-norm (largest singular
    value), so this is an **absolute** cutoff compared against the singular
    values of ``A`` — numpy's ``rcond`` is relative (``rcond * s_max``), which
    is why the patched pinv converts this value.
    """
    matrix = np.asarray(matrix)
    if matrix.size == 0:
        return 0.0
    spectral_norm = float(np.linalg.norm(matrix, 2))
    return max(matrix.shape) * float(np.spacing(spectral_norm))


def _pinv_with_matlab_tolerance(matrix: np.ndarray) -> np.ndarray:
    """``np.linalg.pinv`` with MATLAB's default tolerance (R1).

    Mirrors numpy's own pinv algorithm (SVD, then ``s > cutoff`` masking) but
    uses MATLAB's absolute cutoff :func:`_matlab_pinv_tolerance` instead of
    ``rcond * s_max`` with numpy's default ``rcond=1e-15``. The difference is
    the whole point: near-singular directions in eegprep's ASR reconstruction
    matrix are inverted under the numpy default and produce ~1e15 µV instead of
    repaired data (measured on 4 of 6 pilot recordings that reach ASR), while
    the MATLAB cutoff truncates them.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.size == 0:
        return np.linalg.pinv(matrix)
    if np.iscomplexobj(matrix):
        # N4 (scoped re-audit): numpy's pinv conjugates; this implementation
        # does not. eegprep's ASR path forces a real matrix, so this cannot be
        # reached today — fail loudly if that ever changes.
        raise TypeError(
            "matlab-tolerance pinv is implemented for real matrices only; "
            "eegprep's ASR reconstruction matrix changed to complex input"
        )
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    if s.size == 0:
        return np.zeros((matrix.shape[1], matrix.shape[0]), dtype=float)
    tol = max(matrix.shape) * float(np.spacing(float(s[0])))
    large = s > tol
    s = np.divide(1, s, where=large, out=s)
    s[~large] = 0
    # numpy's own pinv reconstruction: ``vt.T @ (s[:, None] * u.T)`` (scales
    # the rows of ``u.T``, i.e. ``vt.T @ diag(s) @ u.T``).
    return vt.T @ (s[:, np.newaxis] * u.T)


def _eegprep_asr_process_module():
    """Import and return ``eegprep.plugins.clean_rawdata.asr_process``.

    ``from eegprep.plugins.clean_rawdata import asr_process`` yields the
    *function* (the package re-exports it), so the module is fetched through
    ``importlib`` — the R1 patch needs the module object whose globals
    ``asr_process`` resolves ``finite_pinv`` from.
    """
    import importlib

    return importlib.import_module(
        "eegprep.plugins.clean_rawdata.asr_process")


def _verify_eegprep_pinv_assumption(asr_module) -> None:
    """Fail loudly when eegprep's ASR no longer calls ``finite_pinv(Q)`` (R1).

    The R1 patch replaces the ``finite_pinv`` module global for the duration of
    ``clean_asr``. If a future eegprep renames the call, moves it to another
    module, or otherwise changes the pseudo-inverse path, the patch would
    silently do nothing and the blow-up regime would return unnoticed — so the
    assumption is verified against the installed source before every patched
    call and the run refuses to continue on a mismatch.
    """
    import inspect

    import eegprep

    problems = []
    if not hasattr(asr_module, "finite_pinv"):
        problems.append(
            f"{asr_module.__name__} has no module-level finite_pinv")
    else:
        try:
            asr_source = inspect.getsource(asr_module.asr_process)
        except (OSError, TypeError):  # pragma: no cover - source unavailable
            problems.append(
                "asr_process source is unavailable for verification")
        else:
            if "finite_pinv(Q)" not in asr_source:
                problems.append(
                    "asr_process no longer calls finite_pinv(Q)")
    if problems:
        raise RuntimeError(
            "eegprep_pinv_assumption_broken: the R1 MATLAB-pinv patch "
            "(ASR scoped audit 2026-10-16) cannot be applied to the installed "
            f"eegprep {getattr(eegprep, '__version__', 'unknown')}: "
            + "; ".join(problems)
            + ". Re-verify the eegprep ASR pseudo-inverse call site and update "
            "steps/asr.py before running ASR (the default numpy tolerance "
            "returns ~1e15 µV instead of repaired data)."
        )


class _MatlabPinvTolerancePatch:
    """Context manager temporarily replacing eegprep's ASR ``finite_pinv`` (R1).

    The replacement is the module global in
    ``eegprep.plugins.clean_rawdata.asr_process`` — the minimal scoped target:
    ``clean_asr`` calls ``asr_process`` from that module and ``asr_process``
    resolves ``finite_pinv`` from its module globals at call time. Restored in
    ``finally``, so an exception inside ``clean_asr`` never leaves the process
    with a patched pinv. Nested/repeated use is safe (the saved original is the
    current value, which is the patch itself only if a caller nests it).

    N5 (scoped re-audit): the swap is a module-global mutation and is therefore
    **not thread-safe** — ASR must run one recording at a time per process
    (the pipeline's single-writer assumption). The assumption guard is a source
    substring check on ``finite_pinv(Q)``; if eegprep changes the call site the
    patch raises ``eegprep_pinv_assumption_broken`` rather than silently
    reverting to the buggy tolerance.
    """

    def __init__(self) -> None:
        self._asr_module = None
        self._original = None

    def __enter__(self):
        asr_module = _eegprep_asr_process_module()
        _verify_eegprep_pinv_assumption(asr_module)
        self._asr_module = asr_module
        self._original = asr_module.finite_pinv
        asr_module.finite_pinv = _pinv_with_matlab_tolerance
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._asr_module is not None and self._original is not None:
            self._asr_module.finite_pinv = self._original
        return False



def _true_runs(mask: np.ndarray) -> list:
    """Sample runs of a boolean mask as ``[(start_sample, n_samples), ...]``."""
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.size == 0 or not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(start), int(stop - start))
            for start, stop in zip(edges[::2], edges[1::2])]


def append_bad_annotations(
    raw: mne.io.BaseRaw,
    rel_onsets: Sequence[float],
    durations: Sequence[float],
    descriptions: Sequence[str],
) -> None:
    """Append ``BAD_*`` annotations to ``raw`` (plan §0.3 offset recipe).

    ``rel_onsets`` are seconds from the **first data sample** of ``raw`` (the
    data frame), so the marks land on their samples whatever ``first_samp``
    and ``meas_date`` are. Existing annotations are never moved or replaced
    (append-only invariant).

    The recipe is::

        off = raw.first_time if raw.annotations.orig_time is not None else 0.0
        new = mne.Annotations(rel_onsets + off, durations, descriptions,
                              orig_time=raw.annotations.orig_time)
        raw.set_annotations(raw.annotations + new)

    with one correction required by mne 1.13.2: when ``orig_time`` is None,
    ``set_annotations`` adds ``raw.first_time`` to *every* onset it is handed
    (probed: an identity ``set_annotations(raw.annotations)`` moves a stored
    onset from 5.0 s to 9.0 s at 250 Hz with ``first_samp=1000``). Calling
    the recipe as written would therefore shift the already-present
    annotations by ``first_time`` whenever ``first_samp != 0`` — the exact
    timeline corruption this step exists to prevent. In that branch the
    existing onsets are pre-shifted back by ``first_time``, so the stored
    result keeps every annotation on its original sample and the new onsets
    land ``rel_onsets`` seconds after the first data sample.
    """
    rel_onsets = np.atleast_1d(np.asarray(rel_onsets, dtype=float))
    durations = np.atleast_1d(np.asarray(durations, dtype=float))
    descriptions = list(descriptions)
    if len(rel_onsets) == 0:
        return
    if not (len(rel_onsets) == len(durations) == len(descriptions)):
        raise ValueError(
            "rel_onsets, durations and descriptions must have the same length "
            f"(got {len(rel_onsets)}, {len(durations)}, {len(descriptions)})"
        )

    # Self-check (B3-K6): capture the pre-existing annotations' data-sample
    # positions and verify none moved or disappeared after the rewrite. The
    # correction below depends on MNE's onset convention; if that convention
    # ever differs (e.g. a desynced orig_time set through private state), the
    # rewrite could silently drop the whole timeline — refuse loudly instead.
    sfreq = float(raw.info["sfreq"])
    first_time = float(raw.first_time)

    def _sample_keys(annotations):
        return {
            (round((onset - first_time) * sfreq), round(duration * sfreq), str(descr))
            for onset, duration, descr in zip(
                annotations.onset,
                annotations.duration,
                annotations.description,
            )
        }

    prev_keys = _sample_keys(raw.annotations)

    current = raw.annotations
    if current.orig_time is None:
        # `set_annotations` will add `raw.first_time` to everything; hand it
        # the existing onsets in the pre-shift frame so they stay put.
        base = current.copy()
        base.onset = base.onset - raw.first_time
        new = mne.Annotations(rel_onsets, durations, descriptions,
                              orig_time=None)
    else:
        base = current
        new = mne.Annotations(rel_onsets + raw.first_time, durations,
                              descriptions, orig_time=current.orig_time)
    raw.set_annotations(base + new)

    missing = prev_keys - _sample_keys(raw.annotations)
    if missing:
        raise RuntimeError(
            "append_bad_annotations would drop or move pre-existing "
            f"annotations ({len(missing)} missing/moved, e.g. {sorted(missing)[:3]}); "
            "refusing to rewrite the timeline"
        )


class ASRStep(ProcessingStep):
    """
    Artifact Subspace Reconstruction step (full-length repair).

    ASR identifies and reconstructs high-amplitude artifact subspaces. The
    EEGLab recommended workflow uses two-stage ASR (mild before ICA,
    conservative after ICA); the canonical HBN-Langer chain uses a single
    **conservative** pass (``cutoff=20``) after ICA-free artifact regression.
    EEGLAB's cutoff semantics: **aggressive 3, default 5, conservative 20**
    (``clean_asr`` docstring; the EEGLAB GUI's default is 20 and Chang et al.
    2019 report the optimal range as 20-30), so 20 is the *conservative*
    setting — not "aggressive" as an earlier version of this docstring said.

    The ASR **input** is high-passed with ``eegprep.clean_drifts`` first
    (EEGLAB's ``clean_artifacts`` does the same with ``(0.25, 0.75)``; the
    ``clean_asr`` docstring requires zero-mean/high-passed input) and the
    write-back is the high-passed repaired data — **no add-back**, exactly
    EEGLAB's behaviour. ``info['highpass']`` is updated to the filter's
    passband edge so the FIF describes the data it holds (R4). ``clean_asr``
    runs under the MATLAB-equivalent pinv tolerance (R1). See the module
    docstring.

    Parameters
    ----------
    cutoff : float
        ASR cutoff parameter (standard deviations). Lower = more aggressive.
        EEGLAB: aggressive 3, default 5, conservative 20 (GUI default 20;
        Chang et al. 2019 optimal 20-30).
    mode : str
        Processing label only ('mild', 'conservative' or 'single'); it does not
        change the numerics — ``cutoff`` does. Reported in the metadata.
    method : str
        Implementation: 'eegprep' (default, recommended) or 'asrpy'.
    window_criterion : float
        Bad-channel **fraction** per window in (0, 1) passed to
        ``clean_windows(max_bad_channels=...)``; windows exceeding it are
        annotated ``BAD_asr_window``. This is a rejection fraction, NOT a
        window length (ENG-010) and it never falls back to ``window_length``.
    window_tolerances : tuple of float
        ``(lower, upper)`` RMS z-thresholds for window flagging, passed as
        ``clean_windows(zthresholds=...)``. Default
        :data:`ASR_WINDOW_TOLERANCES` = ``(-inf, 7.0)``: only over-power
        windows are flagged.
    calibration : {'data_driven', 'fixed_first_60s'}
        'data_driven' (default): eegprep picks clean calibration data itself
        (``ref_maxbadchannels=0.075``, ``ref_tolerances=(-inf, 5.5)``).
        'fixed_first_60s': the first ``train_duration`` seconds of the data
        are the calibration reference (``ref_maxbadchannels=<ndarray>``),
        the canonical HBN-Langer calibration.
        Ignored by ``method='asrpy'`` (asrpy always fits on the first
        ``train_duration`` seconds); that backend reports
        ``calibration='asrpy_fit_first_train_duration'`` so the metadata
        never claims a calibration that was not used.
    train_duration : float
        Calibration duration in seconds for ``calibration='fixed_first_60s'``.
    window_length : float, optional
        ASR statistics window length in seconds; passed to
        ``clean_asr(window_len=...)`` only when set (``None`` = eegprep's own
        default ``max(0.5, 1.5 * n_channels / sfreq)``).
    drift_transition_hz : (float, float) or None or 'off'
        Transition band of the ``eegprep.clean_drifts`` high-pass applied to
        the ASR **input** before calibration/repair (default ``(0.25, 0.75)``
        — EEGLAB's own ``clean_artifacts`` band; ``clean_asr`` requires
        zero-mean/high-passed input). The write-back is the high-passed
        repaired data (no add-back) and ``info['highpass']`` is set to the
        band's passband edge (R4). The upper edge must stay below the Nyquist
        frequency (checked at runtime). ``None``/``'off'`` skips the filter;
        that is the non-EEGLAB configuration the blow-up guard exists for, so
        it emits a warning and records ``drift_filter_applied: False``.
        eegprep backend only (asrpy is not part of any chain and does not
        apply it; its metadata says so).
    max_dims : float
        Maximum fraction of dimensions ``clean_asr`` may reconstruct; passed
        as ``max_dims`` (eegprep/EEGLAB default 0.66).
    exclude_channels : sequence of str
        Channels that must not enter ASR (in addition to bad channels). The
        recording's reference channel from
        ``metadata['channel_roles']['reference_channel']`` (flat Cz before
        re-referencing) is always excluded.
    max_bad_chans : float
        asrpy-only: max fraction of bad channels during calibration.

    Examples
    --------
    Conservative ASR on EEGLAB-high-passed input (chain default):

    >>> step = ASRStep(cutoff=20, method='eegprep')

    Fixed first-60-s calibration (HBN-Langer):

    >>> step = ASRStep(cutoff=20, calibration='fixed_first_60s',
    ...                train_duration=60.0)

    The pre-2026-10-16 non-EEGLAB regime (no input high-pass; the blow-up
    guard exists for it):

    >>> step = ASRStep(cutoff=20, drift_transition_hz=None)
    """

    name = "asr"
    version = "2.3"  # 2.3: ASR scoped-audit R1 (MATLAB-equivalent pinv
    # tolerance on clean_asr, guarded + recorded) and R4 (info['highpass']
    # updated to the drift filter's passband edge); 2.2: EEGLAB-standard input
    # high-pass (clean_drifts 0.25-0.75 before ASR, no add-back) + configurable
    # max_dims (plan §12.6); 2.1: B7 rev2 fix 7 (re-interpolate the
    # ASR-excluded interpolated channels from the repaired data) + B7-N2
    # output blow-up guard; 2.0: full-length repair + BAD_asr_window + sidecar
    # metadata

    def __init__(
        self,
        cutoff: float = 20.0,
        mode: str = "single",
        method: str = "eegprep",
        window_criterion: float = 0.25,
        window_tolerances: Tuple[float, float] = ASR_WINDOW_TOLERANCES,
        calibration: str = "data_driven",
        train_duration: float = 60.0,
        window_length: Optional[float] = None,
        drift_transition_hz: Optional[Tuple[float, float]] = (0.25, 0.75),
        max_dims: float = 0.66,
        exclude_channels: Sequence[str] = (),
        max_bad_chans: float = 0.1,
        enabled: bool = True,
    ):
        """Initialize ASR step."""
        super().__init__(enabled=enabled)

        if method not in ("eegprep", "asrpy"):
            raise ValueError(
                f"Unknown ASR method: {method}. Use 'eegprep' or 'asrpy'")
        if calibration not in ("data_driven", "fixed_first_60s"):
            raise ValueError(
                f"Unknown ASR calibration: {calibration!r}. "
                "Use 'data_driven' or 'fixed_first_60s'")
        if window_criterion is None or not 0.0 < float(window_criterion) < 1.0:
            raise ValueError(
                "window_criterion must be a bad-channel FRACTION in (0, 1); "
                f"got {window_criterion!r}. clean_windows interprets values "
                "outside (0, 1) as an absolute channel count (ENG-010: this "
                "is a rejection fraction, never a window length).")
        if not float(train_duration) > 0:
            raise ValueError(
                f"train_duration must be > 0 s; got {train_duration!r}")
        if drift_transition_hz is None or (
            isinstance(drift_transition_hz, str) and drift_transition_hz == "off"
        ):
            # The non-EEGLAB regime (ASR on non-high-passed data). Kept
            # expressible for probing/back-compat; the run warns and the
            # metadata records that the filter was skipped.
            self.drift_transition_hz: Optional[Tuple[float, float]] = None
        else:
            try:
                values = tuple(float(v) for v in drift_transition_hz)
            except (TypeError, ValueError):
                raise ValueError(
                    "drift_transition_hz must be None, 'off', or a "
                    f"(low, high) transition band in Hz; got "
                    f"{drift_transition_hz!r}")
            if len(values) != 2 or not 0.0 < values[0] < values[1]:
                raise ValueError(
                    "drift_transition_hz must be a (low, high) pair with "
                    f"0 < low < high (Hz); got {drift_transition_hz!r}")
            self.drift_transition_hz = values
        if not 0.0 < float(max_dims) <= 1.0:
            raise ValueError(
                "max_dims must be a fraction in (0, 1] (eegprep's maximum "
                f"reconstructed dimensionality); got {max_dims!r}")

        self.cutoff = float(cutoff)
        self.mode = mode
        self.method = method
        self.window_criterion = float(window_criterion)
        self.window_tolerances = tuple(float(t) for t in window_tolerances)
        self.calibration = calibration
        self.train_duration = float(train_duration)
        self.window_length = None if window_length is None else float(window_length)
        self.max_dims = float(max_dims)
        self.exclude_channels = tuple(exclude_channels)
        self.max_bad_chans = max_bad_chans

    # ------------------------------------------------------------------
    # picks / validation
    # ------------------------------------------------------------------

    def _channel_exclusions(
        self, raw: mne.io.BaseRaw, metadata: Optional[Dict[str, Any]]
    ) -> list:
        """Names of EEG channels that must not enter ASR.

        Explicit ``exclude_channels`` plus:

        - the recording's reference channel
          (``metadata['channel_roles']['reference_channel']``): a flat
          pre-referencing Cz must not drive the ASR covariance estimate;
        - the channels a previous ``interpolate`` step reconstructed
          (``metadata['interpolate']['interpolated_channels']``). **B7-N1
          (probed 2026-10-16):** spherical-spline interpolation makes each
          interpolated channel an exact linear combination of the remaining
          ones, so the ASR calibration covariance becomes exactly singular
          (measured condition number 1.2e15 with a *single* interpolated
          channel on the EGI-129 fixture). eegprep's ``clean_asr`` then takes
          ``sqrtm`` of that singular matrix (``LinAlgWarning``) and returns
          ~1e15 µV instead of repaired data; every 2-s block then exceeds the
          100 µV block-rejection threshold and the recording is excluded. The
          interpolated channels are synthetic anyway, so leaving them out of
          the ASR filter loses nothing — they must simply not be allowed to
          make the filter degenerate. They stay in the recording (and are
          re-mixed by the average reference), only the ASR *input* excludes
          them.
        """
        roles = (metadata or {}).get("channel_roles") or {}
        reference = roles.get("reference_channel")
        names = list(self.exclude_channels)
        if reference and reference in raw.ch_names and reference not in names:
            names.append(reference)
        interpolate_meta = (metadata or {}).get("interpolate") or {}
        for name in interpolate_meta.get("interpolated_channels") or []:
            if name in raw.ch_names and name not in names:
                names.append(str(name))
        return names

    def _asr_picks(
        self, raw: mne.io.BaseRaw, metadata: Optional[Dict[str, Any]]
    ) -> Tuple[np.ndarray, list]:
        """EEG picks for ASR, minus excluded channels.

        Returns ``(picks, excluded_from_asr)`` where ``excluded_from_asr``
        lists only channels that would otherwise have been picked (honest
        reporting: a name that is not an EEG pick was not excluded by us).
        """
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        candidates = self._channel_exclusions(raw, metadata)
        picked_names = {raw.ch_names[i] for i in eeg_picks}
        excluded = [name for name in candidates if name in picked_names]
        picks = np.asarray(
            [i for i in eeg_picks if raw.ch_names[i] not in excluded],
            dtype=int,
        )
        if len(picks) < 2:
            raise ValueError(
                "ASR needs at least 2 EEG channels after exclusions "
                f"(got {len(picks)}; excluded={excluded})")
        return picks, excluded

    def validate_inputs(self, raw: mne.io.BaseRaw) -> bool:
        """Validate inputs.

        Only the ``fixed_first_60s`` calibration has a length requirement:
        it needs ``train_duration`` seconds of data to build the reference.
        The data-driven calibration selects its own window and works on
        arbitrarily short recordings (eegprep raises if there is not enough
        data for even one window).

        R6 (ASR scoped audit): the ``drift_transition_hz`` upper edge must
        stay below the Nyquist frequency — a transition band reaching Nyquist
        is not a meaningful high-pass and ``clean_drifts`` would build a
        degenerate kernel.
        """
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        if len(eeg_picks) == 0:
            raise ValueError("No EEG channels available for ASR")

        self._check_drift_nyquist(float(raw.info["sfreq"]))

        if self.calibration == "fixed_first_60s":
            required_samples = int(round(self.train_duration * raw.info["sfreq"]))
            if raw.n_times < required_samples:
                raise ValueError(
                    f"Data too short for ASR calibration "
                    f"(calibration='fixed_first_60s'): need {self.train_duration}s "
                    f"({required_samples} samples), have {raw.n_times} samples")
        return True

    def _check_drift_nyquist(self, sfreq: float) -> None:
        """Refuse a drift transition band whose upper edge reaches Nyquist (R6)."""
        if self.drift_transition_hz is None:
            return
        upper = float(self.drift_transition_hz[1])
        nyquist = sfreq / 2.0
        if upper >= nyquist:
            raise ValueError(
                "drift_transition_hz upper edge must be below the Nyquist "
                f"frequency: got {upper:g} Hz with sfreq {sfreq:g} Hz "
                f"(Nyquist {nyquist:g} Hz). clean_drifts cannot apply a "
                "high-pass that reaches Nyquist."
            )

    # ------------------------------------------------------------------
    # processing
    # ------------------------------------------------------------------

    def process(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any]
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        """
        Apply ASR in place (full length) and annotate rejected windows.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input raw data. The executor passes a copy; the repaired data are
            written back with ``raw._data[picks] = ...`` so ``info``,
            ``meas_date`` and annotations survive.
        metadata : dict
            Metadata from previous steps (``channel_roles`` supplies the
            reference channel to exclude).

        Returns
        -------
        raw : mne.io.BaseRaw
            The same object with repaired EEG data and ``BAD_asr_window``
            annotations appended.
        step_metadata : dict
            ASR metadata (see ``_apply_eegprep_asr``).
        """
        if self.method == "asrpy":
            return self._apply_asrpy_asr(raw)
        return self._apply_eegprep_asr(raw, metadata)

    def _reinterpolate_after_asr(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any],
        excluded_from_asr: Sequence[str],
    ) -> Tuple[list, Dict[str, float]]:
        """Re-interpolate the ASR-excluded interpolated channels (B7 rev2 fix 7).

        The channels a previous ``interpolate`` step reconstructed were excluded
        from the ASR *filter* (B7-N1), so their data still reflect the pre-ASR
        interpolation of the pre-repair neighbours — while every neighbour has
        since been rewritten by ASR. That contradiction is invisible in a FIF
        (the channel is a plausible-looking waveform) but breaks the invariant
        that an interpolated channel is the spline reconstruction of the data
        it sits in (measured 1.02/1.09 relative-norm divergence on the EGI-129
        chain fixture). Re-interpolating them from the ASR-cleaned data with the
        same mode as the original interpolation restores the invariant.

        Returns ``(reinterpolated_names, relative_norms)`` where the relative
        norms measure ``||old - new|| / ||new||`` per channel (recorded, so a
        no-op regression is visible in the sidecar).
        """
        interpolate_meta = (metadata or {}).get("interpolate") or {}
        candidates = [
            str(name) for name in (interpolate_meta.get("interpolated_channels") or [])
        ]
        excluded = set(excluded_from_asr)
        names = [
            name for name in candidates
            if name in excluded and name in raw.ch_names
        ]
        if not names:
            return [], {}
        mode = interpolate_meta.get("mode") or "accurate"
        old = {name: raw.get_data(picks=[name])[0].copy() for name in names}
        bads_before = list(raw.info["bads"])
        try:
            # `interpolate_bads` interpolates `info['bads']` and uses only the
            # non-bad channels as sources (probed, mne 1.13.2), so the synthetic
            # channels are not allowed to reconstruct each other.
            raw.info["bads"] = names
            raw.interpolate_bads(reset_bads=False, mode=mode, verbose=False)
        finally:
            # Never leave the recording with the synthetic channels marked bad.
            raw.info["bads"] = bads_before
        relative_norms = {}
        for name in names:
            new = raw.get_data(picks=[name])[0]
            denom = float(np.linalg.norm(new))
            relative_norms[name] = (
                float(np.linalg.norm(old[name] - new) / denom)
                if denom > 0 else 0.0
            )
        return names, relative_norms

    def _apply_eegprep_asr(
        self, raw: mne.io.BaseRaw, metadata: Dict[str, Any]
    ) -> Tuple[mne.io.BaseRaw, Dict]:
        """Repair artifacts with eegprep's ASR at full length.

        The recording is never rebuilt: only the picked rows of the original
        ``raw._data`` are overwritten, so ``meas_date``, ``highpass``,
        ``lowpass``, ``bads``, the montage and every annotation survive.

        The ASR **input** is high-passed with ``eegprep.clean_drifts`` first
        (EEGLAB-standard; ``clean_asr`` requires zero-mean input — see the
        module docstring) and the write-back is the high-passed repaired data
        (no add-back, exactly EEGLAB). ``pre_max_uv`` and the
        ``repaired_fraction`` mask are measured against that post-drift ASR
        input; the separate high-pass effect is recorded as
        ``drift_filter_modified_fraction``.
        """
        import eegprep  # imported lazily; check_dependencies() preflights it

        picks, excluded_from_asr = self._asr_picks(raw, metadata)

        pre = raw.get_data(picks=picks)          # V, (n_picks, n_times)
        n_samples_before = int(pre.shape[1])     # captured before the write-back (B3-K10)
        if not np.isfinite(pre).all():
            raise RuntimeError(
                "ASR input contains non-finite samples; refusing to run "
                "(eegprep's calibration zeroes non-finite samples silently)")

        # eegprep works on EEGLAB dicts in µV: (n_channels, n_samples).
        eeg = eegprep.eeg_mne2eeg(raw.copy().pick(picks))
        sfreq = float(eeg["srate"])
        n_samples = int(np.asarray(eeg["data"]).shape[1])

        # R6: a drift transition band reaching Nyquist is refused at runtime
        # too (validate_inputs is the executor's pre-flight; a direct
        # process() call must not bypass the check).
        self._check_drift_nyquist(sfreq)

        # --- EEGLAB-standard input high-pass (plan §12.6) ---
        # clean_asr requires high-passed (zero-mean) input; EEGLAB's
        # clean_artifacts applies clean_drifts((0.25, 0.75)) internally before
        # ASR and keeps the high-passed output. Applied to the ASR input only
        # (after the picks/exclusions, before the calibration reference is
        # extracted). `clean_drifts` mutates and returns the same EEG dict —
        # this is a fresh dict from `eeg_mne2eeg`, so the caller's Raw is
        # untouched.
        drift_filter_applied = self.drift_transition_hz is not None
        if drift_filter_applied:
            eeg = eegprep.clean_drifts(eeg, tuple(self.drift_transition_hz))
        else:
            warnings.warn(
                "ASRStep: drift_transition_hz is off — running clean_asr on "
                "non-high-passed data. EEGLAB requires zero-mean/high-passed "
                "input (clean_asr docstring); this is the non-EEGLAB "
                "configuration, numerically less conditioned than the "
                "EEGLAB-standard path, and the B7-N2 magnitude guard stays in "
                "place for it. Set drift_transition_hz=(0.25, 0.75) for the "
                "EEGLAB-standard path.",
                RuntimeWarning,
                stacklevel=2,
            )

        pre_uv = np.asarray(eeg["data"], dtype=float)   # ASR input, post-drift
        pre_max_uv = float(np.max(np.abs(pre_uv))) if pre_uv.size else 0.0

        # --- calibration ---
        asr_kwargs: Dict[str, Any] = {}
        n_calibration_samples: Optional[int] = None
        calibration_cov_cond: Optional[float] = None
        calibration_cov_min_eig: Optional[float] = None
        if self.calibration == "fixed_first_60s":
            # The golden-chain wrapper rejects recordings shorter than
            # train_duration via validate_inputs (B3-K1); a direct process()
            # call on a short fixture stays usable by capping the reference,
            # and the real count is reported in `n_calibration_samples`.
            n_cal = min(int(round(self.train_duration * sfreq)), n_samples)
            calibration = np.asarray(pre_uv)[:, :n_cal]
            asr_kwargs["ref_maxbadchannels"] = calibration
            n_calibration_samples = n_cal
            # B7-N2 (optional check): record the calibration covariance
            # condition number. A rank-deficient calibration (interpolated
            # channel left in the filter) makes `clean_asr` take `sqrtm` of a
            # singular matrix and return garbage at ~1e15 µV — the magnitude
            # guard below catches the garbage, this records the precursor.
            cov = np.cov(calibration)
            eigvals = np.linalg.eigvalsh(cov)
            calibration_cov_min_eig = float(eigvals[0])
            calibration_cov_cond = (
                float(eigvals[-1] / eigvals[0]) if eigvals[0] > 0 else np.inf
            )
            if not np.isfinite(calibration_cov_cond):
                raise RuntimeError(
                    "asr_calibration_covariance_degenerate: the fixed ASR "
                    "calibration covariance is exactly singular (condition "
                    f"number {calibration_cov_cond}, min eigenvalue "
                    f"{calibration_cov_min_eig:.3g}); eegprep's clean_asr would "
                    "take sqrtm of it and return garbage instead of repaired "
                    "data (B7-N1/B7-N2). Exclude the rank-deficient channels "
                    "(e.g. a previous interpolate step's channels) from the "
                    "ASR input."
                )
        else:
            # Data-driven calibration: eegprep selects clean data internally.
            # The number of calibration samples it ends up with is not
            # returned, so it is reported as None rather than invented.
            asr_kwargs["ref_maxbadchannels"] = DATA_DRIVEN_REF_MAXBADCHANNELS
            asr_kwargs["ref_tolerances"] = DATA_DRIVEN_REF_TOLERANCES
        if self.window_length is not None:
            asr_kwargs["window_len"] = self.window_length

        # R1 (ASR scoped audit, 2026-10-16): eegprep 0.4.0's ASR reconstruction
        # calls ``np.linalg.pinv`` with numpy's default ``rcond=1e-15``, while
        # MATLAB's ``pinv`` uses the absolute tolerance
        # ``max(size(A)) * eps(norm(A))``. The default inverts near-singular
        # directions and returns ~1e15 µV instead of repaired data on real
        # recordings (4 of 6 pilot recordings that reach ASR). ``clean_asr``
        # runs under the MATLAB-equivalent tolerance; the patch replaces the
        # module global in ``asr_process`` and is restored in ``finally``, and
        # the call site is source-verified first (a changed eegprep raises
        # ``eegprep_pinv_assumption_broken`` instead of silently reverting).
        with _MatlabPinvTolerancePatch():
            eeg_repaired = eegprep.clean_asr(
                eeg, cutoff=self.cutoff, max_dims=self.max_dims, **asr_kwargs)
        eegprep_version = str(getattr(eegprep, "__version__", "unknown"))
        post_uv = np.asarray(eeg_repaired["data"], dtype=float)

        # --- output checks (fail loudly; never zero-fill, never cut) ---
        if post_uv.shape != pre.shape:
            raise RuntimeError(
                f"ASR changed the data shape ({post_uv.shape} != {pre.shape}); "
                "ASR must repair at full length and never cut samples")
        if not np.isfinite(post_uv).all():
            n_bad = int((~np.isfinite(post_uv)).sum())
            raise RuntimeError(
                f"ASR output contains {n_bad} non-finite values; refusing to "
                "zero-fill them (they would silently corrupt the recording)")

        # B7-N2 magnitude guard: a backend blow-up must fail HERE (reason
        # `asr_output_magnitude_blowup`) and can never be misattributed to the
        # block-rejection gate downstream (which would see every 2-s block above
        # its 100 µV threshold and exclude the recording as contaminated).
        post_max_uv = float(np.max(np.abs(post_uv))) if post_uv.size else 0.0
        if pre_max_uv > 0:
            magnitude_ratio = post_max_uv / pre_max_uv
        else:
            magnitude_ratio = np.inf if post_max_uv > 0 else 1.0
        if post_max_uv > ASR_OUTPUT_BLOWUP_FACTOR * pre_max_uv:
            raise RuntimeError(
                "asr_output_magnitude_blowup: ASR output max |amplitude| "
                f"{post_max_uv:.4g} µV exceeds {ASR_OUTPUT_BLOWUP_FACTOR:.0f} x "
                f"the input max {pre_max_uv:.4g} µV (ratio {magnitude_ratio:.3g}). "
                "This is eegprep clean_asr taking sqrtm of a degenerate "
                "calibration covariance (B7-N1/B7-N2), not a repair; failing "
                "here instead of letting the block-rejection gate report a "
                "false 'block contamination' exclusion."
            )

        post = post_uv * 1e-6                     # µV → V
        raw._data[picks] = post

        # R4 (ASR scoped audit, 2026-10-16): the write-back is the
        # clean_drifts-high-passed data (no add-back), so `info['highpass']`
        # must describe the data the recording now holds. MNE's own
        # `filter(l_freq, h_freq)` records `info['highpass'] = l_freq` (the
        # passband edge, verified on mne 1.13.2); the equivalent edge here is
        # the transition band's upper edge (0.75 Hz for the default
        # (0.25, 0.75)). Before this, the FIF kept the chain's stale 0.1 Hz
        # value and claimed a passband the data no longer has. The full
        # transition band stays in the metadata (`drift_transition_hz`).
        info_highpass_hz: Optional[float] = None
        if drift_filter_applied:
            # N1 (scoped re-audit): never *lower* an existing stronger
            # high-pass (explicit configs use 0.5–1 Hz); report the effective
            # edge, i.e. the max of the prior value and this band's edge.
            prior_highpass = float(raw.info["highpass"])
            info_highpass_hz = max(prior_highpass, float(self.drift_transition_hz[1]))
            with raw.info._unlock():
                raw.info["highpass"] = info_highpass_hz

        # B7 rev2 fix 7: the excluded interpolated channels still hold their
        # pre-ASR spline values; re-interpolate them from the repaired data.
        reinterpolated, reinterpolation_norms = self._reinterpolate_after_asr(
            raw, metadata, excluded_from_asr
        )

        # --- rejected-window mask (annotation only, no cutting) ---
        _, sample_mask = eegprep.clean_windows(
            eeg_repaired,
            max_bad_channels=self.window_criterion,
            zthresholds=self.window_tolerances,
        )
        sample_mask = np.asarray(sample_mask, dtype=bool).ravel()
        if sample_mask.shape != (raw.n_times,):
            raise RuntimeError(
                f"clean_windows returned a sample mask of shape "
                f"{sample_mask.shape}; expected ({raw.n_times},)")

        bad_runs = _true_runs(~sample_mask)
        # `n_bad_windows` counts the contiguous rejected intervals (= the
        # annotations): `clean_windows` scores overlapping 1-s windows, so
        # raw window counts would not match `bad_window_intervals_s`.
        bad_window_intervals_s = [
            [start / sfreq, n_run / sfreq] for start, n_run in bad_runs
        ]
        if bad_runs:
            append_bad_annotations(
                raw,
                [start / sfreq for start, _ in bad_runs],
                [n_run / sfreq for _, n_run in bad_runs],
                ["BAD_asr_window"] * len(bad_runs),
            )

        # --- repaired-sample mask (metadata only, not annotated) ---
        # A sample counts as repaired when the cross-channel RMS of the
        # post−pre difference exceeds half the cross-channel RMS of the
        # pre-ASR data: a conservative detector of samples ASR rewrote.
        # `pre` here is the ASR INPUT (post-drift, in V): the drift high-pass
        # is a deliberate EEGLAB-standard transform, not an ASR repair, so it
        # must not inflate `repaired_fraction` (its own effect is recorded
        # separately below as `drift_filter_modified_fraction`).
        asr_input = pre_uv * 1e-6                 # µV → V (post-drift)
        pre_rms = np.sqrt(np.mean(asr_input ** 2, axis=0))
        diff_rms = np.sqrt(np.mean((post - asr_input) ** 2, axis=0))
        repaired_mask = diff_rms > 0.5 * pre_rms
        repaired_runs = _true_runs(repaired_mask)
        repaired_fraction = float(repaired_mask.mean())
        # Separate record of the drift high-pass's own effect on the write-back
        # (post-drift ASR input vs the original pre-ASR data): without it the
        # metadata would not show that the output is high-passed relative to
        # the chain input. None when the filter was skipped.
        if drift_filter_applied:
            raw_rms = np.sqrt(np.mean(pre ** 2, axis=0))
            drift_diff_rms = np.sqrt(np.mean((asr_input - pre) ** 2, axis=0))
            drift_filter_modified_fraction: Optional[float] = float(
                (drift_diff_rms > 0.5 * raw_rms).mean())
        else:
            drift_filter_modified_fraction = None
        # The mask is computed on the REPAIRED data (B3-K7): these numbers
        # measure RESIDUAL post-repair badness. Windows ASR fully repaired are
        # deliberately NOT marked — repaired data are correct data (user
        # decision: mark-only, repaired intervals in metadata).
        window_rejected_fraction = float((~sample_mask).mean())
        # Burden: fraction of samples either in a rejected window or
        # repaired (union — a sample is counted once).
        asr_burden = float((repaired_mask | ~sample_mask).mean())

        step_metadata = {
            "applied": True,
            "implementation": "eegprep",
            "mode": self.mode,
            "calibration": self.calibration,
            "n_calibration_samples": n_calibration_samples,
            "cutoff": self.cutoff,
            "max_dims": self.max_dims,
            # Plan §12.6 (EEGLAB-standard): the ASR input was high-passed with
            # clean_drifts(drift_transition_hz) and the write-back is the
            # high-passed repaired data (no add-back — EEGLAB keeps the HP
            # output). `drift_filter_applied=False` is the non-EEGLAB regime
            # (warned about above).
            "drift_transition_hz": (
                None if self.drift_transition_hz is None
                else list(self.drift_transition_hz)
            ),
            "drift_filter_applied": drift_filter_applied,
            "drift_filter_modified_fraction": drift_filter_modified_fraction,
            # R4: the passband edge written into info['highpass'] (None when
            # the filter was skipped — the recording keeps the chain's value).
            "info_highpass_hz": info_highpass_hz,
            # R1: the pseudo-inverse tolerance clean_asr ran under (MATLAB's
            # default) and the eegprep release it was verified against. The
            # call site is source-verified before the patch; a version that
            # differs from the verified one while still exposing the expected
            # call site is recorded here so provenance shows the drift.
            "pinv_tolerance": PINV_TOLERANCE_SPEC,
            "eegprep_version": eegprep_version,
            "eegprep_pinv_verified_version": EEGPREP_PINV_VERIFIED_VERSION,
            "window_criterion": self.window_criterion,
            "window_tolerances": list(self.window_tolerances),
            "window_length": self.window_length,
            "n_channels_asr": int(len(picks)),
            "excluded_from_asr": excluded_from_asr,
            # B7-N1: which of the exclusions came from a previous interpolate
            # step (their rank-deficient columns must not enter the covariance).
            "interpolated_excluded_from_asr": [
                name for name in excluded_from_asr
                if name in (
                    ((metadata or {}).get("interpolate") or {}).get(
                        "interpolated_channels") or []
                )
            ],
            # B7 rev2 fix 7: the excluded interpolated channels were
            # re-interpolated from the ASR-cleaned data (same mode) so they
            # stay consistent with their repaired neighbours; the relative
            # norms measure how far the pre-ASR spline values were from the
            # repaired reconstruction (0.0 would mean the re-interpolation was
            # a no-op).
            "reinterpolated_after_asr": reinterpolated,
            "reinterpolation_mode": (
                ((metadata or {}).get("interpolate") or {}).get("mode")
                if reinterpolated else None
            ),
            "reinterpolation_relative_norm": reinterpolation_norms,
            # B7-N2: input/output amplitude maxima (µV) and the calibration
            # covariance condition number (fixed calibration only; the
            # data-driven branch's internal calibration is not exposed by
            # eegprep, so it is reported as None rather than invented).
            "pre_max_uv": pre_max_uv,
            "post_max_uv": post_max_uv,
            "output_magnitude_ratio": (
                float(magnitude_ratio) if np.isfinite(magnitude_ratio) else None
            ),
            "calibration_cov_cond": calibration_cov_cond,
            "calibration_cov_min_eig": calibration_cov_min_eig,
            # Full-length guarantee: ASR must never change the timeline.
            "n_samples_before": n_samples_before,
            "n_samples_after": int(raw.n_times),
            "window_rejected_fraction": window_rejected_fraction,
            "n_bad_windows": len(bad_runs),
            "bad_window_intervals_s": bad_window_intervals_s,
            "repaired_fraction": repaired_fraction,
            "repaired_intervals_s": [
                [start / sfreq, n_run / sfreq] for start, n_run in repaired_runs
            ],
            "asr_burden": asr_burden,
            # eegprep's eeg_mne2eeg round-trips through a float32 µV EEGLAB
            # .set file, so every ASR sample is re-quantized (max relative L2
            # error ~4e-8; far below ADC precision, but ASR is not bit-neutral).
            "data_roundtrip": "eeglab_set_float32_uV",
            # Kept for run_processor's log line. ASR no longer rebuilds the
            # Raw, so this reports that the recording still carries its
            # montage (a fact, not a phantom "restoration" claim — B3-K8).
            "montage_present": raw.get_montage() is not None,
        }
        return raw, step_metadata

    def _apply_asrpy_asr(self, raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, Dict]:
        """
        Apply ASR using asrpy (standalone Python implementation).

        Kept functional but not part of any golden chain: it returns a new
        Raw (asrpy's own copy) and does not annotate rejected windows or
        report the eegprep burden fields. asrpy always fits on the first
        ``train_duration`` seconds, so its ``calibration`` metadata value is
        ``'asrpy_fit_first_train_duration'``, never the eegprep value. The
        EEGLAB-standard drift high-pass is **not** applied here (``max_dims``
        is an eegprep-only parameter too); the metadata records both facts
        instead of silently implying the EEGLAB path.

        **Recorded limitation (B7-N1):** this branch receives no ``metadata``,
        so it does **not** exclude interpolated channels from the fit. asrpy
        computes its own covariance with a different (regularized) estimator and
        is not part of any golden chain, so the rank-deficiency failure mode of
        the eegprep path was not observed here; if asrpy is ever promoted into a
        chain, route ``metadata`` in and apply the same exclusion.
        """
        try:
            from asrpy import ASR
        except ImportError:
            raise ImportError(
                "asrpy not installed. Install with: pip install asrpy"
            )

        asr = ASR(
            sfreq=raw.info['sfreq'],
            cutoff=self.cutoff,
            max_bad_chans=self.max_bad_chans
        )

        train_samples = int(self.train_duration * raw.info['sfreq'])
        train_samples = min(train_samples, raw.n_times)

        asr.fit(raw, picks='eeg', start=0, stop=train_samples)
        raw_cleaned = asr.transform(raw, picks='eeg')

        step_metadata = {
            'applied': True,
            'implementation': 'asrpy',
            'calibration': 'asrpy_fit_first_train_duration',
            'train_duration': self.train_duration,
            'n_calibration_samples': train_samples,
            'cutoff': self.cutoff,
            'mode': self.mode,
            # Truthfulness: the EEGLAB drift high-pass / max_dims are not part
            # of this backend's path.
            'drift_transition_hz': None,
            'drift_filter_applied': False,
            'drift_filter_note': 'asrpy backend: clean_drifts not applied',
            'max_dims': None,
            # R1: the MATLAB-pinv patch is an eegprep-path fix; asrpy does not
            # use numpy's pinv for its reconstruction.
            'pinv_tolerance': None,
            'pinv_tolerance_note': 'asrpy backend: numpy pinv tolerance not applicable',
            'n_samples_before': int(raw.n_times),
            'n_samples_after': int(raw_cleaned.n_times),
        }

        return raw_cleaned, step_metadata

    def check_dependencies(self) -> None:
        """Fail at construction when the ASR backend is missing.

        A missing backend must never leave the step as a silent no-op;
        the run has to fail before any data is touched.
        """
        if self.method == 'eegprep':
            try:
                import eegprep  # noqa: F401
                import eeglabio  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    f"ASRStep(method='eegprep') requires eegprep and eeglabio ({e}). "
                    "Install with: pip install eegprep eeglabio"
                ) from e
        elif self.method == 'asrpy':
            try:
                import asrpy  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    f"ASRStep(method='asrpy') requires asrpy ({e}). "
                    "Install with: pip install asrpy"
                ) from e

    def get_config(self) -> Dict[str, Any]:
        """Get configuration (all behaviour-affecting parameters)."""
        config = super().get_config()
        config.update({
            'cutoff': self.cutoff,
            'mode': self.mode,
            'method': self.method,
            'window_criterion': self.window_criterion,
            'window_tolerances': list(self.window_tolerances),
            'calibration': self.calibration,
            'train_duration': self.train_duration,
            'window_length': self.window_length,
            # Plan §12.6: the ASR input high-pass (clean_drifts) and the
            # maximum reconstructed dimensionality are behaviour-affecting.
            'drift_transition_hz': (
                None if self.drift_transition_hz is None
                else list(self.drift_transition_hz)
            ),
            'max_dims': self.max_dims,
            'exclude_channels': list(self.exclude_channels),
            'max_bad_chans': self.max_bad_chans,
            # R1 (ASR scoped audit): the eegprep ASR pseudo-inverse runs under
            # the MATLAB-equivalent tolerance; the spec string is a
            # behaviour-affecting constant of the eegprep path (it changes the
            # repaired data), so it is part of the resolved config.
            'pinv_tolerance': PINV_TOLERANCE_SPEC,
        })
        return config
