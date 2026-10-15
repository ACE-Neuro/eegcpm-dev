"""
Pinned bad-channel detector (per spec §3.i + R-004 + P1-F5).

Implements the variance/kurtosis/spectrum >3-SD detector directly (NOT RANSAC,
NOT immediate interpolation; the existing BadChannelDetectionStep uses RANSAC +
immediate interpolation which silently bypasses the spec's >3-SD rule).

P1-F5 corrections:

- Exemption is explicit: ``exempt=(...)`` (the reference channel comes from
  ``metadata['channel_roles']['reference_channel']`` in the chain wrapper). The
  old ``info['chs'][i]['is_reference']`` flag hack is gone — it could not be set
  (MNE restricts ``info['chs']`` keys) and was silently never true.
- Robust NaN-safe statistics (median/MAD, falling back to std, then to a
  degenerate-metric marker) replace mean/std, so a single flat channel (NaN
  kurtosis) no longer disables the criterion for every channel.
- Zero-variance/flat non-exempt channels are bad outright and are excluded from
  the statistics.
- More than ``max_bad`` detections raise :class:`RecordingExcluded` with the full
  detected list; there is no silent truncation.
- High-frequency power is measured in **25-40 Hz** (``hf_band_hz`` in the
  report). This is a documented deviation from Langer's higher band: the
  detector runs *after* the chain's 0.1-45 Hz low-pass, so 40-100 Hz would be
  empty (P1-F5: "compute before, or document the band").

B4-D1 recalibration (Opus 5.5 final audit, user decision: log-scale robust z):

- **Variance and 25-40 Hz power are z-scored on the ``log`` scale with the
  upper tail only** (``z > threshold_sd`` flags; the reported z is signed).
  Healthy recordings have a wide, skewed low-amplitude spread (Cz-referenced
  HBN recordings in particular); the previous raw-scale two-sided rule flagged
  11-97 channels on real HBN pilot recordings (8/10 would have been excluded).
  High variance / high HF power remain the badness markers; low values are not
  flagged by *this* rule (they are caught by the near-flat floor below).
- **Degeneracy is a relative-epsilon test** (B4-K3): a scale is degenerate when
  ``scale <= 1e-12 * max(abs(median), 1.0)`` — a float-exact ``mad == 0`` check
  is rounding-dependent (``np.nanstd`` of identical ~1e-12 values can be
  ulp-nonzero). The fallback order stays MAD → std → degenerate (``z = 0`` for
  every channel, metric recorded in ``degenerate_metrics``); the test applies on
  the log scale for variance/HF power.
- Non-finite z on an eligible channel is a detection (reason
  ``non_finite_<criterion>``) as before.
- The report gains ``tails`` (per-criterion tail semantics); all previous keys
  are kept.

B4-D2 recalibration (Opus B4 re-audit, user decision 2026-10-15: blink-aware
design; plan §12.5):

- **Kurtosis is computed on blink-free samples.** The detector runs *before*
  EOG regression, so frontal blink clusters dominated the kurtosis criterion on
  real HBN pilot recordings (up to 23 channels flagged). Blinks are located
  with ``mne.preprocessing.find_eog_events`` on the **1-10 Hz band-passed**
  EOG-typed channel with the highest band-passed variance (B6 rev2, NEW-2:
  ranking raw variance picks a drift-dominated infraorbital channel and
  under-detects; the band-pass is the conventional blink band) when an
  EOG-typed channel exists, otherwise on a synthetic frontal proxy (the most
  anterior non-flat EEG channel in the head frame); ``±200 ms`` around each
  event is masked out. If the mask would leave too little data (``> 50%``
  masked, or fewer than ``MIN_BLINK_FREE_FRACTION`` of the samples usable) the
  full signal is used instead and the reason is recorded in the report. A
  detected blink rate below ``MIN_BLINK_RATE_PER_MINUTE`` is implausible for a
  human recording and is recorded as ``blink_mask['plausibility_reason']``
  (the mask is still applied; this is a provenance note, not a fallback). The
  ICLabel cross-check of the user decision is **not** implemented: the
  detector is step 4 of the chain, before any ICA exists, so there are no
  ICLabel components to cross-check (documented, not silently dropped).
- **Kurtosis uses the upper tail on the log scale** (plan §12.5 item 2
  "log-scale / upper-tail kurtosis", B6 rev2 NEW-3): the metric is
  ``log(Pearson kurtosis) = log(excess + 3)``, which is always defined because
  sample Pearson kurtosis ``m4 / m2**2 >= 1`` (Cauchy-Schwarz; the earlier
  "log undefined" premise came from applying the log to *signed excess*
  kurtosis). This matches the log-scale rule used for variance and HF power —
  only *heavy*-tailed channels are flagged, which is the artifact signature.
  (This supersedes the B4-D1 "kurtosis stays two-sided" note; the two-sided
  rule also flagged sub-Gaussian channels, e.g. large sinusoids.)
- **Relative near-flat floor** (plan §12.5 item 3): a channel is near-flat when
  its variance is ``< 1e-3 x median`` of the statistics-eligible channels (the
  ratio disjunct only). This closes the blind spot above the absolute
  ``1e-20 V**2`` floor (e.g. a 1 nV dead electrode, ``1e-18 V**2``). The
  decision's lower-tail z alternative (``log-variance z < -threshold_sd``) is
  deliberately **not** used: the log-variance spread of a healthy recording
  makes z a non-flatness measure (measured: the lowest channel of the healthy
  regression fixture, at 29 % of the median variance, has z = -54, i.e. the z
  rule flags healthy channels). Near-flat channels are bad with reason ``flat``
  and are reported under ``near_flat`` (``flat`` stays the absolute-floor list).
- The report gains ``blink_mask`` (source channel, n_blinks, masked fraction,
  fallback reason) and ``near_flat``; the budget is unchanged.

The detector is MARK-ONLY; the pipeline applies interpolation as a separate later
step (per S12 reconciliation).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import mne
import numpy as np
from scipy import signal, stats

from .base import RecordingExcluded


# Pinned thresholds (per spec §3.i)
THRESHOLD_SD: float = 3.0
MAX_BAD_CHANNELS: int = 11   # <=10% of 109

# Documented deviation (P1-F5): the detector runs after the 0.1-45 Hz low-pass,
# so the high-frequency band is 25-40 Hz instead of Langer's 40-100 Hz.
HF_BAND_HZ: Tuple[float, float] = (25.0, 40.0)

# A channel is flat when its peak-to-peak range is exactly zero or its variance
# is at/below this floor (V**2). Only non-exempt channels are checked.
FLAT_VARIANCE_FLOOR: float = 1e-20

# Relative near-flat floor (B4-D2, plan §12.5 item 3): a channel whose variance
# is below this fraction of the median variance of the statistics-eligible
# channels is near-flat. Closes the blind spot above FLAT_VARIANCE_FLOOR (e.g.
# a 1 nV electrode: 1e-18 V**2 is far above the absolute floor but ~1e-8 of a
# typical 10 uV**2 EEG variance).
NEAR_FLAT_VARIANCE_RATIO: float = 1e-3

# Relative-epsilon degeneracy floor (B4-K3): a robust scale is degenerate when
# it is at/below ``DEGENERACY_REL_EPS * max(abs(median), 1.0)``. A float-exact
# ``mad == 0`` test is rounding-dependent: ``np.nanstd`` of identical ~1e-12
# values can be ulp-nonzero.
DEGENERACY_REL_EPS: float = 1e-12

# Blink masking (B4-D2): +/-200 ms around each detected blink event.
BLINK_MASK_HALF_WIDTH_S: float = 0.2
# If blinks cover more than this fraction of the recording, the blink-free
# samples are too few to trust: fall back to the full signal (reason recorded).
BLINK_MAX_MASKED_FRACTION: float = 0.5
# Minimum usable fraction of blink-free samples for the kurtosis criterion.
MIN_BLINK_FREE_FRACTION: float = 0.5
# Blink channel selection band (B6 rev2, NEW-2): the EOG-typed channel used for
# `find_eog_events` is chosen by variance on the 1-10 Hz band-passed signal,
# the conventional blink band. Raw-variance ranking is dominated by electrode
# DC drift (an infraorbital EOG channel can have a huge drift variance but few
# detectable blinks) and under-detected blinks on real pilot data.
BLINK_BANDPASS_HZ: Tuple[float, float] = (1.0, 10.0)
# Plausibility floor for the detected blink rate (blinks per minute): humans
# blink roughly 10-20 times/min, so a rate below this floor means the detection
# aid found implausibly few events (e.g. a drift-dominated channel was picked
# before the band-pass ranking). Recorded as a reason in the report; the mask
# is still applied.
MIN_BLINK_RATE_PER_MINUTE: float = 1.0

# Detection tail per criterion (B4-D1/D2). Variance and HF power are one-sided
# (upper tail: only abnormally *high* values are bad); kurtosis is upper-tail
# too (B4-D2/B6-NEW-3: heavy tails are the artifact signature; the metric is
# log-Pearson, which is always defined). The lower variance tail is handled by
# the near-flat floor.
TAILS: Dict[str, str] = {
    "variance": "upper",
    "hf_power": "upper",
    "kurtosis": "upper",
}


def _frontal_proxy_channel(
    raw: mne.io.BaseRaw,
    eligible_picks: Sequence[int],
) -> Optional[str]:
    """Most anterior non-flat eligible EEG channel (synthetic blink proxy).

    ``None`` when there are no eligible channels or none has a usable
    head-frame position. B4-D2: used when no EOG-typed channel exists, and
    retried when the EOG channel's ``find_eog_events`` fails (B6-K5).
    """
    if not len(eligible_picks):
        return None
    positions = np.array(
        [raw.info["chs"][p]["loc"][:3] for p in eligible_picks], dtype=float
    )
    if not (np.isfinite(positions).all() and np.any(positions)):
        return None
    ptp = np.ptp(raw.get_data(picks=list(eligible_picks)), axis=1)
    # most anterior (head-frame +y) non-flat EEG channel
    order = np.argsort(-positions[:, 1])
    for idx in order:
        if ptp[idx] > 0.0:
            return raw.info["ch_names"][eligible_picks[int(idx)]]
    return None


def _find_blink_mask(
    raw: mne.io.BaseRaw,
    eligible_picks: Sequence[int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Blink mask (True = inside a +/-200 ms window around a blink).

    B4-D2, refined in B6 rev2 (NEW-2). The blink channel is the EOG-typed
    channel with the highest variance on the **1-10 Hz band-passed** signal
    (the conventional blink band; raw-variance ranking is dominated by
    electrode DC drift and under-detects blinks), otherwise a synthetic frontal
    proxy: the most anterior non-flat EEG channel in the head frame. B6-K5: a
    dead/flat EOG electrode makes ``find_eog_events`` raise; the frontal proxy
    is then retried before giving up (every attempt is recorded under
    ``attempts``). A blink rate below ``MIN_BLINK_RATE_PER_MINUTE`` is recorded
    under ``plausibility_reason`` (never silent, never a fallback). Every
    fallback is recorded in the returned report; the mask is all-False when no
    mask could be built, so the caller falls back to the full signal.
    """
    n_times = int(raw.n_times)
    sfreq = float(raw.info["sfreq"])
    report: Dict[str, Any] = {
        "applied": False,
        "source": None,
        "channel": None,
        "n_blinks": 0,
        "blinks_per_minute": None,
        "masked_fraction": 0.0,
        "half_width_s": BLINK_MASK_HALF_WIDTH_S,
        "plausibility_reason": None,
        "reason": None,
    }
    empty_mask = np.zeros(n_times, dtype=bool)

    # --- candidate blink channels (B6-K5) ---
    # Highest-variance EOG-typed channel when one exists; the synthetic
    # frontal proxy is always built when available so that a dead EOG
    # electrode (find_eog_events raises) can retry the proxy instead of
    # silently falling back to the full signal.
    candidates: List[Tuple[str, str]] = []
    eog_picks = mne.pick_types(raw.info, eog=True, exclude="bads")
    if len(eog_picks):
        eog_data = raw.get_data(picks=eog_picks)
        # NEW-2: rank on the 1-10 Hz band-passed signal (see module docstring).
        selection_band: Optional[List[float]] = list(BLINK_BANDPASS_HZ)
        try:
            ranked = mne.filter.filter_data(
                eog_data, sfreq, BLINK_BANDPASS_HZ[0], BLINK_BANDPASS_HZ[1],
                verbose=False,
            )
        except Exception as e:
            # Never silent: record why the raw signal was used for ranking.
            report["selection_error"] = f"{type(e).__name__}: {e}"
            ranked = eog_data
            selection_band = None
        variances = np.var(ranked, axis=1)
        order = np.argsort(-variances)
        report["selection"] = {
            "metric": "variance",
            "band_hz": selection_band,
            "ranking": [
                {
                    "channel": raw.info["ch_names"][eog_picks[int(i)]],
                    "variance": float(variances[int(i)]),
                }
                for i in order
            ],
        }
        candidates.append(("eog", raw.info["ch_names"][eog_picks[int(order[0])]]))
    proxy = _frontal_proxy_channel(raw, eligible_picks)
    if proxy is not None:
        candidates.append(("frontal_proxy", proxy))

    if not candidates:
        report["reason"] = (
            "no_eog_channel_and_no_usable_anterior_proxy"
            if len(eligible_picks)
            else "no_eog_channel_and_no_eligible_eeg_channels"
        )
        return empty_mask, report

    # --- blink events ---
    # Try each candidate in order. A detection aid must not fail the whole
    # run: the fallback is explicit in the report (never silent), and the
    # kurtosis criterion then uses the full signal. Probed 2026-10-16 on
    # mne 1.13.2: a flat/dead channel makes find_eog_events raise TypeError.
    attempts: List[Dict[str, Any]] = []
    events: Optional[np.ndarray] = None
    for cand_source, cand_channel in candidates:
        try:
            events = np.asarray(
                mne.preprocessing.find_eog_events(
                    raw, ch_name=cand_channel, verbose=False
                )
            )
        except Exception as e:
            attempts.append({
                "source": cand_source,
                "channel": cand_channel,
                "error": f"{type(e).__name__}: {e}",
            })
            continue
        attempts.append({
            "source": cand_source, "channel": cand_channel, "error": None,
        })
        report["source"] = cand_source
        report["channel"] = cand_channel
        break
    report["attempts"] = attempts

    if events is None:
        report["reason"] = "find_eog_events_failed: " + "; ".join(
            f"{a['source']}({a['channel']}): {a['error']}" for a in attempts
        )
        return empty_mask, report

    n_blinks = int(events.shape[0])
    report["n_blinks"] = n_blinks
    duration_min = (n_times / sfreq) / 60.0
    blinks_per_minute = (
        n_blinks / duration_min if duration_min > 0 else float("inf")
    )
    report["blinks_per_minute"] = blinks_per_minute
    if n_blinks > 0 and blinks_per_minute < MIN_BLINK_RATE_PER_MINUTE:
        # NEW-2: plausibility note (never silent). The mask is still applied;
        # a rate this low usually means the wrong channel was ranked first.
        report["plausibility_reason"] = (
            f"low_blink_rate: {blinks_per_minute:.2f}/min < "
            f"{MIN_BLINK_RATE_PER_MINUTE}/min"
        )
    if n_blinks == 0:
        report["reason"] = "no_blinks_detected"
        return empty_mask, report

    # --- +/-200 ms mask ---
    # `find_eog_events` returns samples in the **raw** frame (probed 2026-10-16:
    # the returned samples include `raw.first_samp`, mne maps them back through
    # `times * sfreq + first_samp`), so they are converted to the data frame
    # (0..n_times-1) before indexing the mask.
    first_samp = int(raw.first_samp)
    half_width = int(round(BLINK_MASK_HALF_WIDTH_S * sfreq))
    mask = np.zeros(n_times, dtype=bool)
    for sample in np.asarray(events)[:, 0]:
        sample = int(sample) - first_samp
        start = max(0, sample - half_width)
        stop = min(n_times, sample + half_width)
        mask[start:stop] = True
    masked_fraction = float(mask.mean())
    report["masked_fraction"] = masked_fraction

    if masked_fraction > BLINK_MAX_MASKED_FRACTION:
        report["reason"] = (
            f"blinks_too_frequent: masked_fraction={masked_fraction:.3f} > "
            f"{BLINK_MAX_MASKED_FRACTION}"
        )
        return empty_mask, report

    report["applied"] = True
    return mask, report


def _robust_z(
    metric: np.ndarray, eligible: np.ndarray, tail: str = "two-sided"
) -> Tuple[np.ndarray, bool]:
    """NaN-safe robust z-scores (median/MAD) over the eligible channels.

    ``z = (metric - median) / scale`` with ``scale = 1.4826 * MAD``; when the MAD
    is degenerate the standard deviation is used; when that is degenerate too the
    metric is identical (up to float noise) across eligible channels and is
    reported as degenerate (``z = 0``, no detections from this criterion). A
    scale is degenerate when ``scale <= 1e-12 * max(abs(median), 1.0)`` — a
    relative-epsilon test (B4-K3), not ``mad == 0``. ``tail`` documents the
    detection tail (``'upper'`` or ``'two-sided'``); the returned z is always
    **signed**. Non-eligible channels get ``z = 0``.

    Callers pass the metric on the scale they want z-scored (variance/HF power
    are passed as ``log`` values, B4-D1; kurtosis as ``log`` Pearson, B6-NEW-3).

    Returns ``(z, degenerate)``.
    """
    z = np.zeros(metric.shape, dtype=float)
    values = metric[eligible]
    if values.size == 0:
        return z, True

    median = float(np.nanmedian(values))
    mad = float(1.4826 * np.nanmedian(np.abs(values - median)))
    eps = DEGENERACY_REL_EPS * max(abs(median), 1.0)
    if not np.isfinite(mad) or mad <= eps:
        scale = float(np.nanstd(values))
        if not np.isfinite(scale) or scale <= eps:
            return z, True
    else:
        scale = mad

    z[eligible] = (metric[eligible] - median) / scale
    return z, False


def _z_value(z: np.ndarray, idx: int, computed: bool) -> Any:
    """JSON-friendly z entry (``None`` when not computed / non-finite)."""
    if not computed:
        return None
    value = float(z[idx])
    return value if np.isfinite(value) else None


def detect_bad_channels(
    raw: mne.io.BaseRaw,
    threshold_sd: float = THRESHOLD_SD,
    exempt: Sequence[str] = (),
    max_bad: int = MAX_BAD_CHANNELS,
) -> Tuple[List[str], Dict[str, Any]]:
    """Mark-only bad-channel detector per spec §3.i (P1-F5, B4-D1).

    Algorithm: robust (median/MAD) z-scores on ``log`` variance (upper tail),
    ``log`` 25-40 Hz power (upper tail) and ``log`` Pearson kurtosis (upper
    tail, B4-D2; log scale per B6 rev2 NEW-3 — ``log(excess + 3)``), each
    compared against ``threshold_sd``. ``exempt`` channels are excluded before
    detection; flat non-exempt channels are bad outright and are removed from
    the statistics; a degenerate scale (relative-epsilon test) yields ``z = 0``
    and is recorded in ``degenerate_metrics``; a non-finite z on an eligible
    channel is itself a detection.

    Returns ``(bad_channel_names, report)``. More than ``max_bad`` detections
    raise :class:`RecordingExcluded` (``too_many_bad_channels``) — the list is
    never silently truncated.
    """
    info = raw.info
    exempt_set = set(str(ch) for ch in exempt)

    eeg_picks = mne.pick_types(info, eeg=True, exclude="bads")
    eligible_picks = [
        p for p in eeg_picks if info["ch_names"][p] not in exempt_set
    ]
    names = [info["ch_names"][p] for p in eligible_picks]
    n_channels = len(names)

    if n_channels == 0:
        report = {
            "detected": [],
            "reasons": {},
            "z_scores": {},
            "n_detected_precap": 0,
            "flat": [],
            "near_flat": [],
            "exempt": sorted(exempt_set),
            "hf_band_hz": list(HF_BAND_HZ),
            "tails": dict(TAILS),
            "degenerate_metrics": [],
            "blink_mask": {
                "applied": False,
                "source": None,
                "channel": None,
                "n_blinks": 0,
                "blinks_per_minute": None,
                "masked_fraction": 0.0,
                "half_width_s": BLINK_MASK_HALF_WIDTH_S,
                "plausibility_reason": None,
                "reason": "no_eligible_channels",
            },
        }
        return [], report

    data = raw.get_data(picks=eligible_picks)  # (n_channels, n_times)

    # --- flat channels (bad outright, removed from the statistics) ---
    variance = np.var(data, axis=1)
    ptp = np.ptp(data, axis=1)
    flat_mask = (ptp == 0.0) | (variance <= FLAT_VARIANCE_FLOOR)
    stats_mask = ~flat_mask

    bad_mask = flat_mask.copy()
    reasons: Dict[str, List[str]] = {}
    for idx in np.where(flat_mask)[0]:
        reasons.setdefault(names[idx], []).append("flat")

    # --- B4-D2: blink mask; kurtosis is computed on blink-free samples ---
    blink_mask, blink_report = _find_blink_mask(raw, eligible_picks)
    n_blink_free = int((~blink_mask).sum())
    min_blink_free = int(MIN_BLINK_FREE_FRACTION * data.shape[1])
    if blink_report["applied"] and n_blink_free < min_blink_free:
        # The mask leaves too little data (find_eog_events found many events):
        # use the full signal and record why (never silently).
        blink_report = {
            **blink_report,
            "applied": False,
            "reason": (
                f"too_few_blink_free_samples: {n_blink_free} < "
                f"{min_blink_free} ({MIN_BLINK_FREE_FRACTION:.0%} of samples)"
            ),
        }
    kurtosis_data = data[:, ~blink_mask] if blink_report["applied"] else data

    # --- per-channel metrics on the remaining eligible channels ---
    # NEW-3 (B6 rev2): Pearson kurtosis ``m4 / m2**2 >= 1`` by Cauchy-Schwarz,
    # so the log scale is always defined (the earlier "log undefined" premise
    # came from signed *excess* kurtosis). ``fisher=False`` returns
    # ``excess + 3``; z-scoring on the log matches the variance/HF rule.
    pearson_kurtosis = stats.kurtosis(
        kurtosis_data, axis=1, fisher=False, nan_policy="omit"
    )

    n_times = data.shape[1]
    nperseg = min(int(2 * float(info["sfreq"])), n_times)
    freqs, psd = signal.welch(
        data, fs=float(info["sfreq"]), nperseg=nperseg
    )
    hf_mask = (freqs >= HF_BAND_HZ[0]) & (freqs <= HF_BAND_HZ[1])
    if hf_mask.any():
        hf_power = np.mean(psd[:, hf_mask], axis=1)
    else:
        hf_power = np.full(n_channels, np.nan)

    # --- robust z-scores (B4-D1: log scale for variance/HF, upper tail only) ---
    # The log is taken only on the statistics-eligible channels: flat channels
    # (var <= FLAT_VARIANCE_FLOOR) would make it log(0) = -inf and are excluded
    # from the statistics anyway; NaN marks "not computed" for them.
    log_variance = np.full(variance.shape, np.nan, dtype=float)
    log_variance[stats_mask] = np.log(variance[stats_mask])
    log_hf_power = np.full(hf_power.shape, np.nan, dtype=float)
    log_hf_power[stats_mask] = np.log(hf_power[stats_mask])
    log_kurtosis = np.full(pearson_kurtosis.shape, np.nan, dtype=float)
    log_kurtosis[stats_mask] = np.log(pearson_kurtosis[stats_mask])

    z_var, deg_var = _robust_z(
        log_variance, stats_mask, tail=TAILS["variance"]
    )
    z_kurt, deg_kurt = _robust_z(
        log_kurtosis, stats_mask, tail=TAILS["kurtosis"]
    )
    z_hf, deg_hf = _robust_z(
        log_hf_power, stats_mask, tail=TAILS["hf_power"]
    )

    # --- relative near-flat floor (B4-D2, plan §12.5 item 3) ---
    # A channel is near-flat when its variance is far below the median of the
    # statistics-eligible channels: a 1 nV dead electrode is ~1e-8 of a
    # typical 10 uV channel and sits far above the absolute 1e-20 V**2 floor,
    # which is the blind spot this closes. Near-flat channels are bad with
    # reason 'flat'; they stay in the statistics (the median/MAD are robust
    # and the upper tails do not flag low values).
    #
    # The lower-tail alternative of the decision (`log-variance z < -K`) is
    # deliberately NOT used: the log-variance spread of a healthy recording is
    # tiny, so the z is not a flatness measure. Measured on the healthy
    # regression fixture (test_bad_channel_detector), the *lowest* channel
    # (29% of the median variance — perfectly healthy) has z = -54, i.e. the
    # z rule flags healthy channels exactly like the lower-tail rule the
    # B4-D1 decision rejected on real pilot data. The relative ratio is the
    # physically meaningful criterion and is unit/scale invariant.
    near_flat_mask = np.zeros_like(flat_mask)
    if stats_mask.any():
        median_var = float(np.nanmedian(variance[stats_mask]))
        if np.isfinite(median_var) and median_var > 0:
            near_flat_mask |= stats_mask & (
                variance < NEAR_FLAT_VARIANCE_RATIO * median_var
            )
    for idx in np.where(near_flat_mask)[0]:
        bad_mask[idx] = True
        reasons.setdefault(names[idx], []).append("flat")

    # Tail semantics: 'upper' flags z > threshold; 'two-sided' flags |z| >
    # threshold. The reported z stays signed in both cases.
    for criterion, z in (
        ("variance", z_var),
        ("kurtosis", z_kurt),
        ("hf_power", z_hf),
    ):
        if TAILS[criterion] == "upper":
            flagged = stats_mask & (z > threshold_sd)
        else:
            flagged = stats_mask & (np.abs(z) > threshold_sd)
        for idx in np.where(flagged)[0]:
            bad_mask[idx] = True
            reasons.setdefault(names[idx], []).append(criterion)

    # Non-finite z on an eligible channel is itself a detection (the metric
    # could not be evaluated reliably). A non-finite z that already satisfied
    # its tail comparison is only recorded once (the criterion above).
    for criterion, z in (
        ("variance", z_var),
        ("kurtosis", z_kurt),
        ("hf_power", z_hf),
    ):
        if TAILS[criterion] == "upper":
            flagged = stats_mask & (z > threshold_sd)
        else:
            flagged = stats_mask & (np.abs(z) > threshold_sd)
        for idx in np.where(stats_mask & ~np.isfinite(z) & ~flagged)[0]:
            bad_mask[idx] = True
            reasons.setdefault(names[idx], []).append(f"non_finite_{criterion}")

    bads = [names[idx] for idx in np.where(bad_mask)[0]]
    n_detected_precap = len(bads)

    report = {
        "detected": bads,
        "reasons": reasons,
        "z_scores": {
            names[idx]: {
                "var": _z_value(z_var, idx, bool(stats_mask[idx])),
                "kurtosis": _z_value(z_kurt, idx, bool(stats_mask[idx])),
                "hf": _z_value(z_hf, idx, bool(stats_mask[idx])),
            }
            for idx in range(n_channels)
        },
        "n_detected_precap": int(n_detected_precap),
        "flat": [names[idx] for idx in np.where(flat_mask)[0]],
        "near_flat": [names[idx] for idx in np.where(near_flat_mask)[0]],
        "exempt": sorted(exempt_set),
        "hf_band_hz": list(HF_BAND_HZ),
        "tails": dict(TAILS),
        "blink_mask": blink_report,
        "degenerate_metrics": [
            metric
            for metric, degenerate in (
                ("variance", deg_var),
                ("kurtosis", deg_kurt),
                ("hf_power", deg_hf),
            )
            if degenerate
        ],
    }

    if n_detected_precap > max_bad:
        raise RecordingExcluded(
            "too_many_bad_channels",
            {
                "n_detected": int(n_detected_precap),
                "max_bad_channels": int(max_bad),
                "detected": bads,
            },
        )

    return bads, report


def apply_bad_channels(
    raw: mne.io.BaseRaw,
    new_bads: Sequence[str],
    exempt: Sequence[str] = (),
) -> mne.io.BaseRaw:
    """Mark the given channels as bad on the raw object (mark-only).

    Appends to ``info['bads']`` and never adds an exempt channel (the reference
    channel is exempt by construction, P1-F5). Existing bad channels are kept.
    """
    exempt_set = set(str(ch) for ch in exempt)
    final_bads = list(raw.info.get("bads", []))
    for ch_name in new_bads:
        if ch_name in exempt_set:
            continue
        if ch_name not in final_bads:
            final_bads.append(ch_name)
    raw.info["bads"] = final_bads
    return raw
