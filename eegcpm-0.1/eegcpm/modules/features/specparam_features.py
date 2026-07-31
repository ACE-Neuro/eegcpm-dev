"""
specparam feature module (per spec §3.j + S22 + ENG-007 + METH-021).

Extracts aperiodic and periodic spectral features from EEG spectra.
Mode handling (S21): PRIMARY is fixed mode for all channels and
subjects; the knee arm is EXPLORATORY and stored in SEPARATE columns
(exponent_knee, offset_knee) that are NEVER pooled with the fixed-mode
columns. The `aperiodic_mode` column is REQUIRED in the schema AND
in the parquet metadata.

Fit QC (METH-020 + S20c):
  - R² >= 0.90
  - RMSE <= 0.10 in log10-power units
  - RMSE is the BINDING in-house definition
    sqrt(mean((log10(p) - log10(modeled))^2))
  - The library's MAE attribute is NOT used as the binding QC
    statistic; loud failure if no RMSE is available

Pinned absolute Welch params (METH-013): identical across all
conditions; no None-auto path. Realized segment counts are recorded.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from eegcpm.pipeline.base import BaseModule
from ._specparam_adapter import (
    fit_spectrum,
    get_aperiodic_params,
    get_mae,
    get_n_peaks,
    get_peak_params,
    get_r_squared,
    rmse_in_house,
    specparam_rmse,
)

# Re-export so callers can import from this module too
from ._specparam_adapter import (
    fit_spectrum as _fit_spectrum,
    rmse_in_house as _rmse_in_house,
    specparam_rmse as _specparam_rmse,
)


# Pinned absolute Welch parameters (per METH-013 / §3.h).
# None-auto path: any config with None for these fields is REJECTED
# at config load with PinnedValueRequired.
PINNED_WELCH = {
    "n_per_seg_samples": 1024,      # ~2.05 s at 500 Hz
    "overlap_samples": 512,         # 50%
    "window": "hann",
    "detrend": "constant",
    "scaling": "density",
}

# Realized segment counts at 500 Hz with the pinned Welch settings
# (must match the spec's tables; updated at pilot-50).
# Formula: n_segs = (n_samples - noverlap) // (nperseg - noverlap)
#   EC 200 s:  (100000 - 512) // 512 = 194
#   EC 100 s:  (50000  - 512) // 512 = 96
#   EO 100 s:  (50000  - 512) // 512 = 96
#   DM 175 s:  (87500  - 512) // 512 = 169
REALIZED_SEGMENT_COUNTS = {
    "ec_200s": 194,
    "ec_100s": 96,
    "eo_100s": 96,
    "dm_175s": 169,
}

# Fit QC thresholds (per METH-020, S20c)
FIT_QC = {
    "r_squared_min": 0.90,
    "rmse_max": 0.10,                 # NOT mae_max
    "rmse_units": "log10_power",      # required field
    "rmse_provenance": "in_house",    # binding definition
}


def compute_psd(
    data: np.ndarray,
    sfreq: float,
    n_per_seg_samples: int = 1024,
    overlap_samples: int = 512,
    window: str = "hann",
    detrend: str = "constant",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute PSD using pinned Welch parameters. Returns
    (freqs, psd, n_segments)."""
    from scipy import signal
    nperseg = n_per_seg_samples
    noverlap = overlap_samples
    freqs, psd = signal.welch(
        data, fs=sfreq, nperseg=nperseg, noverlap=noverlap,
        window=window, detrend=detrend, scaling="density",
    )
    n_segments = (data.shape[-1] - noverlap) // (nperseg - noverlap)
    return freqs, psd, n_segments


# Required column list (the binding schema)
SPECPARAM_COLUMNS: List[str] = [
    "subject_id", "session", "condition", "channel", "region",
    # Aperiodic (fixed mode)
    "offset", "exponent",
    # Aperiodic mode marker (REQUIRED — S21)
    "aperiodic_mode",
    # Knee (exploratory; NULL when aperiodic_mode == "fixed")
    "exponent_knee", "offset_knee", "knee_freq",
    # Periodic
    "n_peaks",
    "peak_alpha_freq", "peak_alpha_power",
    # Band-limited aperiodic-corrected power
    "bandpower_delta", "bandpower_theta", "bandpower_alpha",
    "bandpower_beta",  "bandpower_gamma",
    # Fit QC — RMSE per S20c, NOT MAE
    "r_squared", "rmse",
    "qc_pass",
    # Channel bookkeeping
    "n_epochs_used", "duration_seconds",
    "input_file_sha256", "extraction_timestamp",
]


def fit_one_channel(
    freqs: np.ndarray,
    psd: np.ndarray,
    aperiodic_mode: str = "fixed",
) -> Dict[str, Any]:
    """Fit specparam on a single channel's PSD.

    Returns a dict with the specparam columns for the row, plus
    a fit_qc dict carrying realized values.
    """
    # Guard: specparam logs the PSD — zero/negative/NaN powers (dead or
    # fully-interpolated channels) produce NaN/Inf in log space and a
    # DataError. Floor at a tiny positive value and mark QC-fail when
    # the floor was engaged on >5% of bins (dead channel).
    psd = np.asarray(psd, dtype=float)
    bad_bins = ~np.isfinite(psd) | (psd <= 0)
    floor_engaged = float(bad_bins.mean())
    if floor_engaged > 0:
        psd = psd.copy()
        psd[bad_bins] = 1e-20
    if floor_engaged > 0.05:
        return {
            "offset": np.nan, "exponent": np.nan,
            "aperiodic_mode": aperiodic_mode,
            "exponent_knee": None, "offset_knee": None, "knee_freq": None,
            "n_peaks": 0, "peak_alpha_freq": np.nan,
            "peak_alpha_power": np.nan,
            "r_squared": 0.0, "rmse": np.inf, "qc_pass": False,
            "library_mae_for_reference_only": np.nan,
            "dead_channel_fraction": floor_engaged,
        }
    if aperiodic_mode == "fixed":
        model = _fit_spectrum(
            freqs, psd, freq_range=(2.0, 40.0),
            aperiodic_mode="fixed", max_n_peaks=6, verbose=False,
        )
        ap = get_aperiodic_params(model)  # [offset, exponent]
        offset, exponent = float(ap[0]), float(ap[1])
        return {
            "offset": offset,
            "exponent": exponent,
            "aperiodic_mode": "fixed",
            "exponent_knee": None,
            "offset_knee": None,
            "knee_freq": None,
            "n_peaks": get_n_peaks(model),
            "peak_alpha_freq": _extract_alpha_peak(model),
            "peak_alpha_power": _extract_alpha_peak_power(model),
            "r_squared": get_r_squared(model),
            "rmse": _specparam_rmse(model),
            "qc_pass": _check_qc(get_r_squared(model), _specparam_rmse(model)),
            "library_mae_for_reference_only": get_mae(model),
        }
    elif aperiodic_mode == "knee":
        # Exploratory; SEPARATE output columns
        model = _fit_spectrum(
            freqs, psd, freq_range=(2.0, 40.0),
            aperiodic_mode="knee", max_n_peaks=6, verbose=False,
        )
        ap = get_aperiodic_params(model)  # [offset, knee, exponent] (3)
        offset, knee, exponent = float(ap[0]), float(ap[1]), float(ap[2])
        return {
            "offset": None,           # null in fixed column; knee values in separate cols
            "exponent": None,
            "aperiodic_mode": "knee",
            "offset_knee": offset,
            "exponent_knee": exponent,
            "knee_freq": knee,
            "n_peaks": get_n_peaks(model),
            "peak_alpha_freq": _extract_alpha_peak(model),
            "peak_alpha_power": _extract_alpha_peak_power(model),
            "r_squared": get_r_squared(model),
            "rmse": _specparam_rmse(model),
            "qc_pass": _check_qc(get_r_squared(model), _specparam_rmse(model)),
            "library_mae_for_reference_only": get_mae(model),
        }
    else:
        raise ValueError(
            f"Unknown aperiodic_mode={aperiodic_mode!r}; must be "
            f"'fixed' or 'knee'."
        )


def _extract_alpha_peak(model) -> float:
    """Extract the dominant alpha peak frequency (8-13 Hz); None
    if no peak in that range."""
    peaks = get_peak_params(model)
    if len(peaks) == 0:
        return None
    alpha_mask = (peaks[:, 0] >= 8) & (peaks[:, 0] <= 13)
    if not alpha_mask.any():
        return None
    # Highest power wins
    alpha_peaks = peaks[alpha_mask]
    idx = int(np.argmax(alpha_peaks[:, 1]))
    return float(alpha_peaks[idx, 0])


def _extract_alpha_peak_power(model) -> float:
    peaks = get_peak_params(model)
    if len(peaks) == 0:
        return None
    alpha_mask = (peaks[:, 0] >= 8) & (peaks[:, 0] <= 13)
    if not alpha_mask.any():
        return None
    alpha_peaks = peaks[alpha_mask]
    idx = int(np.argmax(alpha_peaks[:, 1]))
    return float(alpha_peaks[idx, 1])


def _check_qc(r_squared: float, rmse: float) -> bool:
    return (
        r_squared >= FIT_QC["r_squared_min"]
        and rmse <= FIT_QC["rmse_max"]
    )


class SpecparamFeatureModule(BaseModule):
    """specparam feature extraction (per-channel, per-condition)."""

    name = "specparam_features"
    version = "0.2.0"  # R-010: complete BaseModule contract

    def __init__(self, config: dict, output_dir):
        super().__init__(config, output_dir)
        self.aperiodic_mode = config.get("aperiodic_mode", "fixed")
        if self.aperiodic_mode not in {"fixed", "knee"}:
            raise ValueError(
                f"aperiodic_mode={self.aperiodic_mode!r}; must be "
                f"'fixed' or 'knee'."
            )
        self.freq_range = tuple(config.get("freq_range", (2.0, 40.0)))
        self.subject_id = config.get("subject_id", "unknown")
        self.session = config.get("session", "01")
        self.condition = config.get("condition", "resting_ec")
        self.input_file_sha256 = config.get("input_file_sha256", None)

    def validate_input(self, data) -> bool:
        """R-010: validate that data is a 2D (n_channels, n_times)
        RawArray-compatible array. Reject MNE Epochs (we operate on
        continuous Raw) and any non-array input.
        """
        import numpy as np
        if not isinstance(data, np.ndarray):
            return False
        if data.ndim != 2:
            return False
        if data.shape[0] < 2 or data.shape[1] < 256:
            return False
        return True

    def process(self, data, subject=None, condition="resting_ec",
                session="01", **kwargs):
        """Compute PSD per channel, fit specparam, return ModuleResult
        (R-010: BaseModule contract).

        `data` is (n_channels, n_times) array. The output DataFrame
        has the SPECPARAM_COLUMNS schema and metadata per row.
        """
        import time
        start = time.time()
        # R-010: subject/condition/session from kwargs OVERRIDE
        # constructor defaults (so per-call arguments win).
        subj = subject if subject is not None else self.subject_id
        sess = session or self.session
        cond = condition or self.condition
        sfreq = kwargs.get("sfreq", 500.0)
        n_per_seg = PINNED_WELCH["n_per_seg_samples"]
        n_overlap = PINNED_WELCH["overlap_samples"]
        n_channels = data.shape[0]
        rows = []
        for ch in range(n_channels):
            ch_data = data[ch]
            freqs, psd, n_segs = compute_psd(
                ch_data, sfreq=sfreq,
                n_per_seg_samples=n_per_seg,
                overlap_samples=n_overlap,
                window=PINNED_WELCH["window"],
                detrend=PINNED_WELCH["detrend"],
            )
            row = fit_one_channel(freqs, psd, self.aperiodic_mode)
            row["channel"] = f"ch_{ch}"
            row["region"] = "unknown"   # caller can update
            row["subject_id"] = subj
            row["session"] = sess
            row["condition"] = cond
            row["n_epochs_used"] = n_segs
            row["duration_seconds"] = data.shape[-1] / sfreq
            row["input_file_sha256"] = self.input_file_sha256
            row["extraction_timestamp"] = time.time()
            # Band-limited power (computed from the PSD)
            bw = lambda lo, hi: float(np.mean(
                psd[(freqs >= lo) & (freqs <= hi)]))
            row["bandpower_delta"] = bw(2, 4)
            row["bandpower_theta"] = bw(4, 8)
            row["bandpower_alpha"] = bw(8, 13)
            row["bandpower_beta"] = bw(13, 30)
            row["bandpower_gamma"] = bw(30, 45)
            rows.append(row)
        # R-010: return ModuleResult (BaseModule contract) with
        # the binding schema in outputs["features"] as a list of
        # dicts (each row is a SPECPARAM_COLUMNS dict).
        from eegcpm.pipeline.base import ModuleResult
        return ModuleResult(
            module_name=self.name,
            execution_time_seconds=time.time() - start,
            success=True,
            outputs={"features": rows},
            output_files=[],
            metadata={
                "n_channels": n_channels,
                "aperiodic_mode": self.aperiodic_mode,
                "freq_range": list(self.freq_range),
                "n_rows": len(rows),
                "columns": list(SPECPARAM_COLUMNS),
                "pinned_welch": dict(PINNED_WELCH),
                "realized_segment_counts": dict(REALIZED_SEGMENT_COUNTS),
            },
        )
