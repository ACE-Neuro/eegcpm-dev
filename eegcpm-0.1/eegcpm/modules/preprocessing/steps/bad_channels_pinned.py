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
  flagged (flat channels are detected explicitly, low HF power is not an
  artifact).
- **Kurtosis stays two-sided on the raw scale**
  (``|kurt - median(kurt)| / scale > threshold_sd``): both heavy tails and
  extreme dips are suspicious.
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

The detector is MARK-ONLY; the pipeline applies interpolation as a separate later
step (per S12 reconciliation).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

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

# Relative-epsilon degeneracy floor (B4-K3): a robust scale is degenerate when
# it is at/below ``DEGENERACY_REL_EPS * max(abs(median), 1.0)``. A float-exact
# ``mad == 0`` test is rounding-dependent: ``np.nanstd`` of identical ~1e-12
# values can be ulp-nonzero.
DEGENERACY_REL_EPS: float = 1e-12

# Detection tail per criterion (B4-D1). Variance and HF power are one-sided
# (upper tail: only abnormally *high* values are bad); kurtosis is two-sided.
TAILS: Dict[str, str] = {
    "variance": "upper",
    "hf_power": "upper",
    "kurtosis": "two-sided",
}


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
    are passed as ``log`` values, B4-D1).

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
    ``log`` 25-40 Hz power (upper tail) and raw kurtosis (two-sided), each
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
            "exempt": sorted(exempt_set),
            "hf_band_hz": list(HF_BAND_HZ),
            "tails": dict(TAILS),
            "degenerate_metrics": [],
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

    # --- per-channel metrics on the remaining eligible channels ---
    kurtosis = stats.kurtosis(data, axis=1, fisher=True, nan_policy="omit")

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

    z_var, deg_var = _robust_z(
        log_variance, stats_mask, tail=TAILS["variance"]
    )
    z_kurt, deg_kurt = _robust_z(
        kurtosis, stats_mask, tail=TAILS["kurtosis"]
    )
    z_hf, deg_hf = _robust_z(
        log_hf_power, stats_mask, tail=TAILS["hf_power"]
    )

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
        "exempt": sorted(exempt_set),
        "hf_band_hz": list(HF_BAND_HZ),
        "tails": dict(TAILS),
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
