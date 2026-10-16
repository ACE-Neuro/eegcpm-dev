"""Per-recording QC metrics for the HBN canonical chain.

Promoted from the d-factor project driver (24-pilot5-local.py) into the
toolbox so any project gets the same QC surface (spec §3.2 + pilot-50
design). One row per recording (subject x task).

Metrics:
- duration_post_s / n_times_post: usable duration after the chain
- line_noise_residual_ratio: 60 Hz / 20-40 Hz spectral ratio (post)
- blink_residual_ratio: **the ocular gate** (P1-F10). Blinks are detected on
  the *pre-chain* recording (``raw_pre``) through the first available EOG
  channel (EGI HydroCel names first, else ``find_eog_events``' synthetic
  frontal proxy), and the ratio of blink-locked frontal peak-to-peak amplitude
  post/pre is reported. ``blink_residual_pass`` = ``ratio < OCULAR_GATE``.
  ``None``/``NaN`` when ``raw_pre`` is not supplied (no gate, no false pass).
  Two guards from the B7 rev2 audit: (1) the pre/post timelines must match
  (``sfreq``, ``n_times``, ``first_samp``) or the ratio is ``None`` with
  ``reason='timeline_mismatch'`` — blink sample indices from ``raw_pre`` are
  meaningless on a resampled/trimmed ``raw_post``; (2) the peak-to-peak is
  measured on **identically band-passed copies** (1–15 Hz) of pre and post.
  Measuring the pre amplitude on unfiltered data (line noise, drift) while the
  post is filtered biases the ratio toward pass (probed: 0.78/0.55/0.20 for
  0/20/100 µV 60 Hz line noise in the pre recording vs the 0.30 gate — the
  100 µV case is a false pass with the blinks fully retained).
- residual_ocular_maxcorr_insample: informational only. Max |corr| of a bipolar
  EOG proxy against scalp channels *in the post-chain data*. After the OLS EOG
  regression this is near zero **by construction** — the EOG regressors are
  subtracted from the EEG, so an in-sample correlation cannot detect residual
  ocular activity. It is kept for continuity and must not be used as a gate.
- emg_proxy_30_45_ratio: 30-45 Hz / 2-45 Hz spectral ratio (partial
  EMG proxy; the 45 Hz ceiling caps the classic 70-110 Hz EMG band)
- asr_burden / asr_rejected_fraction / asr_repaired_fraction: from the
  ASR step metadata (`chain_metadata['asr']`); `None` when the chain
  metadata is not supplied. The ASR step stores no state in the MNE
  ``info`` 'temp' field (it does not survive a FIF save) — the
  executor's JSON sidecar and this metadata are the only provenance.
- bad_channels_n: number of channels the chain's bad-channel detector flagged
  (`chain_metadata['bad_channel_detection']['detected']`, P1-F5); falls back to
  ``len(raw.info['bads'])`` when the chain metadata is not supplied (the
  interpolate step resets ``info['bads']``, so the metadata is authoritative).
- block_rejected_fraction / block_n_rejected: from
  ``chain_metadata['block_rejection']`` (`rejected_fraction`,
  `n_blocks_rejected` — the names the block-rejection step emits).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# EGI HydroCel EOG channel names (9)
EOG_HYDROCEL_NAMES: List[str] = [
    "E8", "E14", "E17", "E21", "E25", "E125", "E126", "E127", "E128",
]

#: Gate on ``blink_residual_ratio`` (blink-locked frontal ptp post/pre).
OCULAR_GATE = 0.30
#: Band-pass applied to **identical copies** of pre and post before the
#: blink-locked peak-to-peak is measured (B7 rev2 fix 4). Without it the pre
#: amplitude carries unfiltered line noise / drift while the post amplitude is
#: filtered, biasing the ratio toward pass (probed: a false pass at 100 µV
#: 60 Hz in the pre recording with the blinks fully retained).
BLINK_BANDPASS_HZ: Tuple[float, float] = (1.0, 15.0)
#: Frontal *scalp* channels used for the blink-locked amplitude ratio when no
#: EGI EOG channel is present (matches the detector's synthetic frontal proxy
#: idea). EGI EOG channel names are deliberately absent: the ratio measures the
#: residual ocular activity in the **scalp EEG**, and an EOG channel keeps its
#: blink by definition (the EOG regression only rewrites EEG channels), which
#: would pin the ratio near 1.
FRONTAL_PROXY_NAMES: Tuple[str, ...] = (
    "Fp1", "Fp2", "Fpz", "AF3", "AF4", "F7", "F8",
    "E1", "E22", "E26", "E33", "E34", "E45", "E46", "E122", "E123",
)


def _first_eog_channel(raw, eog_names: Sequence[str] = EOG_HYDROCEL_NAMES) -> Optional[str]:
    """First EOG channel present in ``raw`` (EGI names, then EOG-typed picks)."""
    import mne

    for name in eog_names:
        if name in raw.ch_names:
            return name
    eog_picks = mne.pick_types(raw.info, eog=True, exclude=[])
    if len(eog_picks):
        return raw.ch_names[eog_picks[0]]
    return None


def _blink_events(raw, proxy_name: str) -> np.ndarray:
    """Sample indices of blink peaks detected on ``raw`` (``find_eog_events``).

    ``find_eog_events`` returns **absolute** sample numbers (they include
    ``raw.first_samp``); the returned array is converted to data-relative
    indices so it can index ``raw.get_data(start=...)`` directly.
    """
    import mne

    events = mne.preprocessing.find_eog_events(
        raw, ch_name=proxy_name, verbose=False
    )
    return events[:, 0] - int(raw.first_samp)


def _frontal_channels(raw) -> List[str]:
    """Names of frontal *scalp* channels used for the blink-locked amplitude ratio.

    EOG-typed channels are excluded: the gate measures the residual ocular
    activity in the scalp EEG, and an EOG channel keeps its blink by definition
    (the EOG regression only rewrites EEG channels) — including one would pin
    the ratio near 1 for every recording.
    """
    import mne

    eog_names = {
        raw.ch_names[i] for i in mne.pick_types(raw.info, eog=True, exclude=[])
    }
    names = [
        ch for ch in raw.ch_names
        if ch in FRONTAL_PROXY_NAMES
        and ch not in eog_names
        and ch not in (raw.info.get("bads") or [])
    ]
    if not names:
        # Fall back to the first third of the EEG channels; never empty.
        eeg_names = [
            raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])
        ]
        names = eeg_names[: max(1, len(eeg_names) // 3)]
    return names


def _blink_locked_ptp(raw, events: np.ndarray, names: Sequence[str],
                      window_s: float = 0.25) -> Optional[float]:
    """Mean peak-to-peak amplitude in a +/- ``window_s`` window around blinks.

    Channels are resolved by name in ``raw`` (the chain drops/renames channels,
    so pre- and post-chain indices are not interchangeable).
    """
    picks = [raw.ch_names.index(name) for name in names if name in raw.ch_names]
    if not len(events) or not picks:
        return None
    sfreq = raw.info["sfreq"]
    half = int(round(window_s * sfreq))
    n_times = raw.n_times
    ptps = []
    for onset in events:
        start = int(onset) - half
        stop = int(onset) + half
        if start < 0 or stop > n_times or stop <= start:
            continue
        segment = raw.get_data(picks=picks, start=start, stop=stop)
        ptps.append(float(np.mean(np.ptp(segment, axis=1))))
    if not ptps:
        return None
    return float(np.mean(ptps))


def _bandpassed_copy(raw, names: Sequence[str]):
    """Copy of ``raw`` with the frontal channels band-passed for the ptp (B7 rev2 fix 4).

    Only the channels the ratio actually reads are filtered (a copy is made so
    the caller's recording is never modified). Pre and post go through the
    *same* filter, so line noise / drift cannot inflate the pre amplitude and
    bias the ratio toward pass.
    """
    picks = [name for name in names if name in raw.ch_names]
    filtered = raw.copy()
    if picks:
        filtered.filter(
            l_freq=BLINK_BANDPASS_HZ[0],
            h_freq=BLINK_BANDPASS_HZ[1],
            picks=picks,
            verbose=False,
        )
    return filtered


def _timeline(raw) -> Tuple[float, int, int]:
    """``(sfreq, n_times, first_samp)`` — the sample frame of a recording."""
    return (float(raw.info["sfreq"]), int(raw.n_times), int(raw.first_samp))


def blink_residual_ratio(
    raw_pre,
    raw_post,
    *,
    eog_names: Sequence[str] = EOG_HYDROCEL_NAMES,
) -> Dict[str, Any]:
    """Blink-locked frontal amplitude ratio post/pre (P1-F10 ocular gate).

    Returns a dict with ``ratio``, ``pass`` (``ratio < OCULAR_GATE``),
    ``n_blinks``, ``blink_channel`` (the channel the blinks were detected on)
    and ``frontal_channels``. When blinks or frontal channels cannot be
    resolved the ratio/pass are ``None`` (a gate that cannot run must not
    report a pass).

    B7 rev2 guards:

    - ``reason='timeline_mismatch'`` (ratio/pass ``None``) when the pre and
      post recordings do not share ``sfreq``, ``n_times`` and ``first_samp``:
      the blink sample indices come from ``raw_pre`` and would silently index
      the wrong samples of a resampled/trimmed ``raw_post`` (probed: a
      resampled post yields a meaningless ratio while still reporting a
      pass/fail verdict);
    - the pre and post peak-to-peak amplitudes are measured on identically
      band-passed copies (:data:`BLINK_BANDPASS_HZ`).
    """
    result: Dict[str, Any] = {
        "ratio": None,
        "pass": None,
        "n_blinks": 0,
        "blink_channel": None,
        "frontal_channels": [],
        "reason": None,
    }
    if raw_pre is None or raw_post is None:
        result["reason"] = "raw_pre_missing"
        return result

    pre_timeline = _timeline(raw_pre)
    post_timeline = _timeline(raw_post)
    if pre_timeline != post_timeline:
        # Blink sample indices from `raw_pre` are only meaningful on a post
        # recording with the identical sample frame (B7 rev2 fix 3).
        result["reason"] = "timeline_mismatch"
        result["timeline"] = {
            "pre": {
                "sfreq": pre_timeline[0],
                "n_times": pre_timeline[1],
                "first_samp": pre_timeline[2],
            },
            "post": {
                "sfreq": post_timeline[0],
                "n_times": post_timeline[1],
                "first_samp": post_timeline[2],
            },
        }
        return result

    blink_channel = _first_eog_channel(raw_pre, eog_names)
    if blink_channel is None:
        # Synthetic frontal proxy: the most anterior available channel.
        candidates = [ch for ch in raw_pre.ch_names if ch in FRONTAL_PROXY_NAMES]
        if not candidates:
            result["reason"] = "no_eog_or_frontal_channel"
            return result
        blink_channel = candidates[0]
    result["blink_channel"] = blink_channel

    try:
        events = _blink_events(raw_pre, blink_channel)
    except Exception as exc:  # mne raises for degenerate/flat reference channels
        result["reason"] = f"find_eog_events_failed: {exc}"
        return result

    frontal_names = _frontal_channels(raw_pre)
    # Identical band-pass on both sides before the ptp (fix 4): the pre
    # amplitude must not carry line noise / drift the post no longer has.
    ptp_pre = _blink_locked_ptp(
        _bandpassed_copy(raw_pre, frontal_names), events, frontal_names)
    ptp_post = _blink_locked_ptp(
        _bandpassed_copy(raw_post, frontal_names), events, frontal_names)
    result["n_blinks"] = int(len(events))
    result["frontal_channels"] = list(frontal_names)
    if not events.size or ptp_pre is None or ptp_post is None or ptp_pre <= 0:
        result["reason"] = "no_usable_blink_window"
        return result

    ratio = float(ptp_post / ptp_pre)
    result["ratio"] = ratio
    result["pass"] = bool(ratio < OCULAR_GATE)
    return result


def recording_qc_metrics(
    raw_post,
    task: str,
    subject_id: Optional[str] = None,
    chain_metadata: Optional[Dict[str, Any]] = None,
    raw_pre=None,
) -> Dict[str, Any]:
    """Compute the per-recording QC row from a post-chain mne Raw.

    Parameters
    ----------
    raw_post : mne.io.BaseRaw
        The preprocessed recording (chain output, EOG channels retained).
    task : str
        Task label.
    subject_id : str, optional
    chain_metadata : dict, optional
        Per-step metadata from the preprocessing chain. The ASR burden
        fields are read from ``chain_metadata['asr']`` (the ASR step
        keeps no state in the MNE ``info`` 'temp' field), the detected bad
        channels from ``chain_metadata['bad_channel_detection']`` and the
        block-rejection counters from ``chain_metadata['block_rejection']``.
    raw_pre : mne.io.BaseRaw, optional
        The pre-chain recording. Required for the ocular gate
        (``blink_residual_ratio``); without it the gate fields are ``None``
        (never a false pass).
    """
    data = raw_post.get_data()
    sfreq = raw_post.info["sfreq"]
    row: Dict[str, Any] = {
        "task": task,
        "subject_id": subject_id,
        "duration_post_s": float(data.shape[1] / sfreq),
        "n_times_post": int(data.shape[1]),
    }

    # Spectral QC on the post-chain signal
    win = np.hanning(data.shape[1])
    F = np.abs(np.fft.rfft(data * win, axis=-1))
    freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / sfreq)

    m60 = (freqs > 58) & (freqs < 62)
    mbb = (freqs > 20) & (freqs < 40)
    row["line_noise_residual_ratio"] = float(
        (F[:, m60].mean() + 1e-30) / (F[:, mbb].mean() + 1e-30))

    # Residual ocular — informational only (vacuous as a gate after the OLS
    # EOG regression, P1-F10): bipolar EOG proxy vs scalp max |corr|.
    ch_names = raw_post.ch_names
    eog_idx = [ch_names.index(c) for c in EOG_HYDROCEL_NAMES
               if c in ch_names]
    scalp_idx = [i for i in range(len(ch_names))
                 if i not in eog_idx and ch_names[i] != "Cz"]
    if len(eog_idx) >= 2 and scalp_idx:
        bipolar = data[eog_idx[0]] - data[eog_idx[1]]
        corr_mat = np.corrcoef(np.vstack([bipolar, data[scalp_idx]]))
        row["residual_ocular_maxcorr_insample"] = float(
            np.nanmax(np.abs(corr_mat[0, 1:])))
    else:
        row["residual_ocular_maxcorr_insample"] = np.nan

    # Ocular gate: blink-locked frontal ptp post/pre on the pre-chain recording.
    blink = blink_residual_ratio(raw_pre, raw_post)
    row["blink_residual_ratio"] = blink["ratio"]
    row["blink_residual_pass"] = blink["pass"]
    row["blink_n"] = blink["n_blinks"]
    row["blink_channel"] = blink["blink_channel"]
    if blink["reason"]:
        row["blink_residual_reason"] = blink["reason"]

    # EMG proxy: 30-45 Hz band power ratio (partial; 45 Hz ceiling)
    memg = (freqs >= 30) & (freqs <= 45)
    mall = (freqs >= 2) & (freqs <= 45)
    row["emg_proxy_30_45_ratio"] = float(
        (F[:, memg].mean() + 1e-30) / (F[:, mall].mean() + 1e-30))

    # ASR burden fields: read from the ASR step metadata
    # (chain_metadata['asr'], produced by ASRStep / _ASRStepWrapper).
    # The step deliberately keeps no state in the MNE ``info`` 'temp' field.
    asr_meta = (chain_metadata or {}).get("asr") or {}
    row["asr_burden"] = asr_meta.get("asr_burden")
    row["asr_rejected_fraction"] = asr_meta.get("window_rejected_fraction")
    row["asr_repaired_fraction"] = asr_meta.get("repaired_fraction")

    # Bad channels: the detector's report is authoritative (the interpolate
    # step resets ``info['bads']``); without chain metadata fall back to info.
    detector_meta = (chain_metadata or {}).get("bad_channel_detection") or {}
    if detector_meta:
        detected = detector_meta.get("detected") or []
        row["bad_channels_n"] = int(len(detected))
        row["bad_channels_n_precap"] = detector_meta.get("n_detected_precap")
        row["bad_channels_detected"] = [str(ch) for ch in detected]
    else:
        row["bad_channels_n"] = len(raw_post.info.get("bads") or [])
        row["bad_channels_n_precap"] = None
        row["bad_channels_detected"] = list(raw_post.info.get("bads") or [])

    # Chain metadata fallbacks
    if chain_metadata:
        br = chain_metadata.get("block_rejection")
        if isinstance(br, dict):
            row["block_rejected_fraction"] = br.get("rejected_fraction")
            row["block_n_rejected"] = br.get("n_blocks_rejected")

    return row
