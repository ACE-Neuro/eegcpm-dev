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

Supports both eegprep (EEGLab port) and asrpy implementations. The asrpy
branch is kept functional but is not part of any golden chain: it does not
annotate rejected windows and does not report the eegprep burden fields.

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

Author: EEGCPM Development Team
Created: 2025-12
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import mne
import numpy as np

from .base import ProcessingStep


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


class ASRStep(ProcessingStep):
    """
    Artifact Subspace Reconstruction step (full-length repair).

    ASR identifies and reconstructs high-amplitude artifact subspaces. The
    EEGLab recommended workflow uses two-stage ASR (mild before ICA,
    aggressive after ICA); the canonical HBN-Langer chain uses a single
    aggressive pass (``cutoff=20``) after ICA-free artifact regression.

    Parameters
    ----------
    cutoff : float
        ASR cutoff parameter (standard deviations). Lower = more aggressive.
        Recommended: 40 for mild, 20 for aggressive.
    mode : str
        Processing label only ('mild', 'aggressive' or 'single'); it does not
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
        ``clean_windows(zthresholds=...)``. Default ``(-inf, 7.0)``: only
        over-power windows are flagged.
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
    exclude_channels : sequence of str
        Channels that must not enter ASR (in addition to bad channels). The
        recording's reference channel from
        ``metadata['channel_roles']['reference_channel']`` (flat Cz before
        re-referencing) is always excluded.
    max_bad_chans : float
        asrpy-only: max fraction of bad channels during calibration.

    Examples
    --------
    Aggressive ASR (chain default):

    >>> step = ASRStep(cutoff=20, method='eegprep')

    Fixed first-60-s calibration (HBN-Langer):

    >>> step = ASRStep(cutoff=20, calibration='fixed_first_60s',
    ...                train_duration=60.0)
    """

    name = "asr"
    version = "2.0"  # 2.0: full-length repair + BAD_asr_window + sidecar metadata

    def __init__(
        self,
        cutoff: float = 20.0,
        mode: str = "single",
        method: str = "eegprep",
        window_criterion: float = 0.25,
        window_tolerances: Tuple[float, float] = (-np.inf, 7.0),
        calibration: str = "data_driven",
        train_duration: float = 60.0,
        window_length: Optional[float] = None,
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

        self.cutoff = float(cutoff)
        self.mode = mode
        self.method = method
        self.window_criterion = float(window_criterion)
        self.window_tolerances = tuple(float(t) for t in window_tolerances)
        self.calibration = calibration
        self.train_duration = float(train_duration)
        self.window_length = None if window_length is None else float(window_length)
        self.exclude_channels = tuple(exclude_channels)
        self.max_bad_chans = max_bad_chans

    # ------------------------------------------------------------------
    # picks / validation
    # ------------------------------------------------------------------

    def _channel_exclusions(
        self, raw: mne.io.BaseRaw, metadata: Optional[Dict[str, Any]]
    ) -> list:
        """Names of EEG channels that must not enter ASR.

        Explicit ``exclude_channels`` plus the recording's reference channel
        (``metadata['channel_roles']['reference_channel']``): a flat
        pre-referencing Cz must not drive the ASR covariance estimate.
        """
        roles = (metadata or {}).get("channel_roles") or {}
        reference = roles.get("reference_channel")
        names = list(self.exclude_channels)
        if reference and reference in raw.ch_names and reference not in names:
            names.append(reference)
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
        """
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        if len(eeg_picks) == 0:
            raise ValueError("No EEG channels available for ASR")

        if self.calibration == "fixed_first_60s":
            required_samples = int(round(self.train_duration * raw.info["sfreq"]))
            if raw.n_times < required_samples:
                raise ValueError(
                    f"Data too short for ASR calibration "
                    f"(calibration='fixed_first_60s'): need {self.train_duration}s "
                    f"({required_samples} samples), have {raw.n_times} samples")
        return True

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

    def _apply_eegprep_asr(
        self, raw: mne.io.BaseRaw, metadata: Dict[str, Any]
    ) -> Tuple[mne.io.BaseRaw, Dict]:
        """Repair artifacts with eegprep's ASR at full length.

        The recording is never rebuilt: only the picked rows of the original
        ``raw._data`` are overwritten, so ``meas_date``, ``highpass``,
        ``lowpass``, ``bads``, the montage and every annotation survive.
        """
        import eegprep  # imported lazily; check_dependencies() preflights it

        picks, excluded_from_asr = self._asr_picks(raw, metadata)

        pre = raw.get_data(picks=picks)          # V, (n_picks, n_times)
        if not np.isfinite(pre).all():
            raise RuntimeError(
                "ASR input contains non-finite samples; refusing to run "
                "(eegprep's calibration zeroes non-finite samples silently)")

        # eegprep works on EEGLAB dicts in µV: (n_channels, n_samples).
        eeg = eegprep.eeg_mne2eeg(raw.copy().pick(picks))
        sfreq = float(eeg["srate"])
        n_samples = int(np.asarray(eeg["data"]).shape[1])

        # --- calibration ---
        asr_kwargs: Dict[str, Any] = {}
        n_calibration_samples: Optional[int] = None
        if self.calibration == "fixed_first_60s":
            # The golden-chain wrapper rejects recordings shorter than
            # train_duration via validate_inputs (B3-K1); a direct process()
            # call on a short fixture stays usable by capping the reference,
            # and the real count is reported in `n_calibration_samples`.
            n_cal = min(int(round(self.train_duration * sfreq)), n_samples)
            asr_kwargs["ref_maxbadchannels"] = np.asarray(eeg["data"])[:, :n_cal]
            n_calibration_samples = n_cal
        else:
            # Data-driven calibration: eegprep selects clean data internally.
            # The number of calibration samples it ends up with is not
            # returned, so it is reported as None rather than invented.
            asr_kwargs["ref_maxbadchannels"] = 0.075
            asr_kwargs["ref_tolerances"] = (-np.inf, 5.5)
        if self.window_length is not None:
            asr_kwargs["window_len"] = self.window_length

        eeg_repaired = eegprep.clean_asr(eeg, cutoff=self.cutoff, **asr_kwargs)
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

        post = post_uv * 1e-6                     # µV → V
        raw._data[picks] = post

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
        pre_rms = np.sqrt(np.mean(pre ** 2, axis=0))
        diff_rms = np.sqrt(np.mean((post - pre) ** 2, axis=0))
        repaired_mask = diff_rms > 0.5 * pre_rms
        repaired_runs = _true_runs(repaired_mask)
        repaired_fraction = float(repaired_mask.mean())
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
            "window_criterion": self.window_criterion,
            "window_tolerances": list(self.window_tolerances),
            "window_length": self.window_length,
            "n_channels_asr": int(len(picks)),
            "excluded_from_asr": excluded_from_asr,
            # Full-length guarantee: ASR must never change the timeline.
            "n_samples_before": int(raw.n_times),
            "n_samples_after": int(raw.n_times),
            "window_rejected_fraction": window_rejected_fraction,
            "n_bad_windows": len(bad_runs),
            "bad_window_intervals_s": bad_window_intervals_s,
            "repaired_fraction": repaired_fraction,
            "repaired_intervals_s": [
                [start / sfreq, n_run / sfreq] for start, n_run in repaired_runs
            ],
            "asr_burden": asr_burden,
            # Kept for run_processor's log line. ASR no longer rebuilds the
            # Raw, so this reports that the recording still carries its
            # montage (computed, not a phantom "restoration" claim).
            "montage_restored": raw.get_montage() is not None,
        }
        return raw, step_metadata

    def _apply_asrpy_asr(self, raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, Dict]:
        """
        Apply ASR using asrpy (standalone Python implementation).

        Kept functional but not part of any golden chain: it returns a new
        Raw (asrpy's own copy) and does not annotate rejected windows or
        report the eegprep burden fields. asrpy always fits on the first
        ``train_duration`` seconds, so its ``calibration`` metadata value is
        ``'asrpy_fit_first_train_duration'``, never the eegprep value.
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
            'exclude_channels': list(self.exclude_channels),
            'max_bad_chans': self.max_bad_chans,
        })
        return config
