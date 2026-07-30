"""
Pinned bad-channel detector (per spec §3.i + R-004).

Implements the variance/kurtosis/spectrum >3-SD detector
directly (NOT RANSAC, NOT immediate interpolation; the existing
BadChannelDetectionStep uses RANSAC + immediate interpolation
which silently bypasses the spec's >3-SD rule).

Cz is exempt BEFORE detection (not after): the
`is_reference` flag is consumed by the detector's exemption so
that Cz (a zero-variance channel by construction) is NEVER flagged
and NEVER consumes the <=11/109 budget.

The detector is MARK-ONLY; the pipeline applies interpolation as
a separate later step (per S12 reconciliation).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import mne
import numpy as np


# Pinned thresholds (per spec §3.i)
THRESHOLD_SD: float = 3.0
MAX_BAD_CHANNELS: int = 11   # <=10% of 109


def _is_cz(info: mne.Info, idx: int) -> bool:
    """Check if channel idx is the Cz reference (is_reference flag)."""
    return bool(info["chs"][idx].get("is_reference", False))


def detect_bad_channels(
    raw: mne.io.BaseRaw,
    threshold_sd: float = THRESHOLD_SD,
) -> List[str]:
    """Mark-only bad-channel detector per spec §3.i.

    Algorithm: variance/kurtosis/spectrum >3-SD. Cz is exempt
    BEFORE detection. Returns a list of bad-channel NAMES.

    The existing BadChannelDetectionStep in the codebase uses
    RANSAC + immediate interpolation which does not match the
    spec; this function is the spec-compliant replacement.
    """
    info = raw.info
    picks = mne.pick_types(info, eeg=True, exclude="bads")
    if len(picks) == 0:
        return list(info.get("bads", []))

    data = raw.get_data(picks=picks)  # (n_channels, n_times)
    n_channels, n_times = data.shape

    # Build Cz-exempt mask
    is_cz = np.array([_is_cz(info, picks[i]) for i in range(n_channels)])
    eligible = ~is_cz

    # Per-channel statistics (only over eligible channels)
    bad_mask = np.zeros(n_channels, dtype=bool)
    if eligible.sum() == 0:
        return [info["ch_names"][picks[i]] for i in range(n_channels)
                if bad_mask[i]]

    # Variance score
    var = np.var(data, axis=1)
    var_eligible = var[eligible]
    var_mu = var_eligible.mean()
    var_sd = var_eligible.std() if var_eligible.std() > 0 else 1e-12
    var_z = np.abs(var - var_mu) / var_sd
    bad_mask |= (var_z > threshold_sd) & eligible

    # Kurtosis score
    try:
        from scipy import stats
        kurt = stats.kurtosis(data, axis=1, fisher=True, nan_policy="omit")
        kurt_eligible = kurt[eligible]
        kurt_mu = kurt_eligible.mean()
        kurt_sd = kurt_eligible.std() if kurt_eligible.std() > 0 else 1e-12
        kurt_z = np.abs(kurt - kurt_mu) / kurt_sd
        bad_mask |= (kurt_z > threshold_sd) & eligible
    except ImportError:
        pass

    # Spectrum score: high-frequency (40-100 Hz) power
    try:
        from scipy import signal
        # Use a quick PSD in the high-frequency range
        nperseg = min(int(raw.info["sfreq"] * 2), n_times)
        f, Pxx = signal.welch(data, fs=raw.info["sfreq"], nperseg=nperseg)
        hf_mask = (f >= 40) & (f <= min(100, raw.info["sfreq"] / 2 - 1))
        if hf_mask.any() and hf_mask.sum() > 1:
            hf_power = np.mean(Pxx[:, hf_mask], axis=1)
            hf_eligible = hf_power[eligible]
            hf_mu = hf_eligible.mean()
            hf_sd = hf_eligible.std() if hf_eligible.std() > 0 else 1e-12
            hf_z = np.abs(hf_power - hf_mu) / hf_sd
            bad_mask |= (hf_z > threshold_sd) & eligible
    except (ImportError, ValueError):
        pass

    # Cap at MAX_BAD_CHANNELS (the spec's <=10% budget; RECALIBRATED
    # in pilot-50; per S19). Truncate to the WORST z-scores.
    if bad_mask.sum() > MAX_BAD_CHANNELS:
        # Compute a combined z-score for ranking
        combined_z = np.zeros(n_channels)
        combined_z[eligible] = var_z[eligible]
        eligible_idx = np.where(eligible)[0]
        if "kurt_z" in dir():
            try:
                combined_z[eligible] = np.maximum(
                    combined_z[eligible], kurt_z[eligible])
            except (NameError, IndexError):
                pass
        # Sort by combined_z descending, keep the worst MAX_BAD_CHANNELS
        sorted_idx = eligible_idx[np.argsort(combined_z[eligible])[::-1]]
        new_bad = np.zeros(n_channels, dtype=bool)
        for j in sorted_idx[:MAX_BAD_CHANNELS]:
            new_bad[j] = True
        bad_mask = new_bad

    return [info["ch_names"][picks[i]] for i in range(n_channels)
            if bad_mask[i]]


def apply_bad_channels(raw: mne.io.BaseRaw,
                       new_bads: List[str]) -> mne.io.BaseRaw:
    """Mark the given channels as bad on the raw object (mark-only).

    Cz is re-verified: even if `new_bads` includes Cz, we remove it
    (Cz is exempt per spec §3.i).
    """
    info = raw.info
    final_bads = list(info.get("bads", []))
    for ch_name in new_bads:
        if ch_name not in final_bads:
            final_bads.append(ch_name)
    # Cz-exemption: never mark Cz as bad
    final_bads = [b for b in final_bads if not _is_cz_by_name(info, b)]
    raw.info["bads"] = final_bads
    return raw


def _is_cz_by_name(info: mne.Info, ch_name: str) -> bool:
    if ch_name not in info["ch_names"]:
        return False
    idx = info["ch_names"].index(ch_name)
    return bool(info["chs"][idx].get("is_reference", False))
