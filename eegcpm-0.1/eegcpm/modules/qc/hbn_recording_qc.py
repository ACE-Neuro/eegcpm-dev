"""Per-recording QC metrics for the HBN canonical chain.

Promoted from the d-factor project driver (24-pilot5-local.py) into the
toolbox so any project gets the same QC surface (spec §3.2 + pilot-50
design). One row per recording (subject x task).

Metrics:
- duration_post_s / n_times_post: usable duration after the chain
- line_noise_residual_ratio: 60 Hz / 20-40 Hz spectral ratio (post)
- residual_ocular_maxcorr: max |corr| of a bipolar EOG proxy against
  scalp channels (gate: < 0.30; requires EOG channels retained)
- emg_proxy_30_45_ratio: 30-45 Hz / 2-45 Hz spectral ratio (partial
  EMG proxy; the 45 Hz ceiling caps the classic 70-110 Hz EMG band)
- asr_burden / asr_rejected_fraction / asr_repaired_fraction: from the
  persisted ASR metadata (clean_sample_mask lineage)
- bad_channels_n: len(raw.info['bads']) after the chain
"""

from typing import Any, Dict, List, Optional

import numpy as np

# EGI HydroCel EOG channel names (9)
EOG_HYDROCEL_NAMES: List[str] = [
    "E8", "E14", "E17", "E21", "E25", "E125", "E126", "E127", "E128",
]

OCULAR_GATE = 0.30  # max |corr| of bipolar EOG proxy vs scalp


def recording_qc_metrics(
    raw_post,
    task: str,
    subject_id: Optional[str] = None,
    chain_metadata: Optional[Dict[str, Any]] = None,
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
        Per-step metadata from the preprocessing chain (for ASR burden
        fields when info['temp'] is absent).
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

    # Residual ocular: bipolar EOG proxy vs scalp max |corr|
    ch_names = raw_post.ch_names
    eog_idx = [ch_names.index(c) for c in EOG_HYDROCEL_NAMES
               if c in ch_names]
    scalp_idx = [i for i in range(len(ch_names))
                 if i not in eog_idx and ch_names[i] != "Cz"]
    if len(eog_idx) >= 2 and scalp_idx:
        bipolar = data[eog_idx[0]] - data[eog_idx[1]]
        corr_mat = np.corrcoef(np.vstack([bipolar, data[scalp_idx]]))
        row["residual_ocular_maxcorr"] = float(
            np.nanmax(np.abs(corr_mat[0, 1:])))
        row["residual_ocular_pass"] = bool(
            row["residual_ocular_maxcorr"] < OCULAR_GATE)
    else:
        row["residual_ocular_maxcorr"] = np.nan
        row["residual_ocular_pass"] = None

    # EMG proxy: 30-45 Hz band power ratio (partial; 45 Hz ceiling)
    memg = (freqs >= 30) & (freqs <= 45)
    mall = (freqs >= 2) & (freqs <= 45)
    row["emg_proxy_30_45_ratio"] = float(
        (F[:, memg].mean() + 1e-30) / (F[:, mall].mean() + 1e-30))

    # ASR burden fields: from info['temp'] (persisted by the ASR step)
    temp = raw_post.info.get("temp")
    if isinstance(temp, dict):
        row["asr_burden"] = temp.get("asr_burden")
        row["asr_rejected_fraction"] = temp.get("asr_rejected_fraction")
        row["asr_repaired_fraction"] = temp.get("asr_repaired_fraction")
    row["bad_channels_n"] = len(raw_post.info.get("bads") or [])

    # Chain metadata fallbacks
    if chain_metadata:
        br = chain_metadata.get("block_rejection")
        if isinstance(br, dict):
            row["block_rejected_fraction"] = br.get("fraction")
            row["block_n_bad"] = br.get("n_bad")

    return row
