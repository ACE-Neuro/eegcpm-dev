"""Quality Control module for subject-level artifact assessment."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np

from eegcpm.pipeline.base import BaseModule, ModuleResult, RawDataModule
from .base import BaseQC, QCMetric, QCResult
from .html_report import HTMLReportBuilder, QCIndexBuilder

logger = logging.getLogger(__name__)


@dataclass
class QCMetrics:
    """Quality control metrics for a single recording."""

    subject_id: str
    run_id: Optional[str] = None

    # Data quality metrics
    recording_duration_s: float = 0.0
    n_channels: int = 0
    n_bad_channels: int = 0
    bad_channel_names: List[str] = field(default_factory=list)
    percent_bad_channels: float = 0.0

    # Artifact metrics
    percent_bad_segments: float = 0.0
    total_bad_duration_s: float = 0.0
    n_bad_annotations: int = 0

    # Signal quality
    mean_amplitude_uv: float = 0.0
    max_amplitude_uv: float = 0.0
    std_amplitude_uv: float = 0.0
    line_noise_power: float = 0.0  # 50/60 Hz

    # Flatline detection
    n_flatline_channels: int = 0
    flatline_channel_names: List[str] = field(default_factory=list)

    # Overall assessment
    usable: bool = True
    exclusion_reasons: List[str] = field(default_factory=list)
    qc_score: float = 1.0  # 0-1 quality score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subject_id": self.subject_id,
            "run_id": self.run_id,
            "recording_duration_s": self.recording_duration_s,
            "n_channels": self.n_channels,
            "n_bad_channels": self.n_bad_channels,
            "bad_channel_names": self.bad_channel_names,
            "percent_bad_channels": self.percent_bad_channels,
            "percent_bad_segments": self.percent_bad_segments,
            "total_bad_duration_s": self.total_bad_duration_s,
            "n_bad_annotations": self.n_bad_annotations,
            "mean_amplitude_uv": self.mean_amplitude_uv,
            "max_amplitude_uv": self.max_amplitude_uv,
            "std_amplitude_uv": self.std_amplitude_uv,
            "line_noise_power": self.line_noise_power,
            "n_flatline_channels": self.n_flatline_channels,
            "flatline_channel_names": self.flatline_channel_names,
            "usable": self.usable,
            "exclusion_reasons": self.exclusion_reasons,
            "qc_score": self.qc_score,
        }


@dataclass
class QCThresholds:
    """Thresholds for quality control decisions."""

    # Channel rejection
    max_bad_channel_percent: float = 20.0  # Exclude if >20% channels bad

    # Segment rejection
    max_bad_segment_percent: float = 50.0  # Exclude if >50% data is bad

    # Amplitude thresholds (in microvolts)
    max_amplitude_uv: float = 500.0  # Exclude if max > 500 µV
    min_amplitude_uv: float = 0.1  # Flatline if std < 0.1 µV

    # Minimum recording duration
    min_duration_s: float = 60.0  # At least 1 minute

    # Line noise threshold (relative power)
    max_line_noise_ratio: float = 5.0

    @classmethod
    def lenient(cls) -> "QCThresholds":
        """Lenient thresholds - keep more data."""
        return cls(
            max_bad_channel_percent=30.0,
            max_bad_segment_percent=60.0,
            max_amplitude_uv=1000.0,
        )

    @classmethod
    def strict(cls) -> "QCThresholds":
        """Strict thresholds - cleaner data."""
        return cls(
            max_bad_channel_percent=10.0,
            max_bad_segment_percent=30.0,
            max_amplitude_uv=300.0,
        )


class QualityControlModule(RawDataModule):
    """
    Quality control assessment for raw EEG data.

    Identifies subjects/runs with excessive artifacts that should be excluded
    from further analysis BEFORE preprocessing.

    Metrics computed:
    - Bad channel detection (flatlines, excessive noise)
    - Bad segment identification
    - Amplitude statistics
    - Line noise assessment
    - Overall usability score
    """

    name = "quality_control"
    version = "0.1.0"
    description = "Subject-level quality control assessment"

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        super().__init__(config, output_dir)

        # Parse thresholds
        thresh_config = config.get("thresholds", {})
        if isinstance(thresh_config, str):
            if thresh_config == "lenient":
                self.thresholds = QCThresholds.lenient()
            elif thresh_config == "strict":
                self.thresholds = QCThresholds.strict()
            else:
                self.thresholds = QCThresholds()
        else:
            self.thresholds = QCThresholds(**thresh_config)

        self.line_freq = config.get("line_freq", 50.0)  # 50 Hz (EU) or 60 Hz (US)
        self.check_flatlines = config.get("check_flatlines", True)
        self.flatline_duration_s = config.get("flatline_duration_s", 5.0)

    def process(
        self,
        data: mne.io.BaseRaw,
        subject: Optional[Any] = None,
        run_id: Optional[str] = None,
        **kwargs,
    ) -> ModuleResult:
        """
        Run quality control assessment.

        Args:
            data: Raw EEG data
            subject: Subject info
            run_id: Run identifier

        Returns:
            ModuleResult with QC metrics
        """
        start_time = time.time()
        raw = data
        output_files = []
        warnings = []

        try:
            subject_id = subject.id if subject else "unknown"

            # Initialize metrics
            metrics = QCMetrics(
                subject_id=subject_id,
                run_id=run_id,
            )

            # Basic info
            metrics.recording_duration_s = raw.times[-1]
            metrics.n_channels = len(raw.ch_names)

            # Check bad channels (already marked)
            if raw.info["bads"]:
                metrics.bad_channel_names = list(raw.info["bads"])
                metrics.n_bad_channels = len(metrics.bad_channel_names)
            metrics.percent_bad_channels = (
                100 * metrics.n_bad_channels / metrics.n_channels
            )

            # Compute amplitude statistics
            eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
            if len(eeg_picks) > 0:
                data_subset = raw.get_data(picks=eeg_picks)
                metrics.mean_amplitude_uv = float(np.mean(np.abs(data_subset)) * 1e6)
                metrics.max_amplitude_uv = float(np.max(np.abs(data_subset)) * 1e6)
                metrics.std_amplitude_uv = float(np.std(data_subset) * 1e6)

            # Check for flatlines
            if self.check_flatlines:
                flatlines = self._detect_flatlines(raw, eeg_picks)
                metrics.n_flatline_channels = len(flatlines)
                metrics.flatline_channel_names = flatlines

            # Check bad annotations/segments
            if raw.annotations:
                bad_annots = [a for a in raw.annotations if a["description"].startswith("BAD")]
                metrics.n_bad_annotations = len(bad_annots)
                metrics.total_bad_duration_s = sum(a["duration"] for a in bad_annots)
                metrics.percent_bad_segments = (
                    100 * metrics.total_bad_duration_s / metrics.recording_duration_s
                )

            # Compute line noise
            metrics.line_noise_power = self._compute_line_noise(raw, eeg_picks)
            if not np.isfinite(metrics.line_noise_power):
                # NaN, never a false 0.0; the caller records why (B7 rev2 fix 9).
                warnings.append(
                    "Line noise power could not be computed (see log); "
                    "reported as NaN instead of a false 0.0"
                )

            # Assess usability
            metrics = self._assess_usability(metrics)

            # Save QC report
            qc_path = self.output_dir / f"{subject_id}_qc.json"
            import json
            with open(qc_path, "w") as f:
                json.dump(metrics.to_dict(), f, indent=2)
            output_files.append(qc_path)

            return ModuleResult(
                success=True,
                module_name=self.name,
                execution_time_seconds=time.time() - start_time,
                outputs={
                    "data": raw,  # Pass through unchanged
                    "qc_metrics": metrics,
                    "usable": metrics.usable,
                },
                output_files=output_files,
                warnings=warnings,
                metadata=metrics.to_dict(),
            )

        except Exception as e:
            return ModuleResult(
                success=False,
                module_name=self.name,
                execution_time_seconds=time.time() - start_time,
                errors=[str(e)],
            )

    def _detect_flatlines(
        self,
        raw: mne.io.BaseRaw,
        picks: np.ndarray,
    ) -> List[str]:
        """Detect channels with flatline segments."""
        flatline_channels = []
        n_samples_flat = int(self.flatline_duration_s * raw.info["sfreq"])

        for pick in picks:
            ch_data = raw.get_data(picks=[pick])[0]

            # Check for near-zero variance windows
            for start in range(0, len(ch_data) - n_samples_flat, n_samples_flat):
                window = ch_data[start : start + n_samples_flat]
                if np.std(window) < self.thresholds.min_amplitude_uv * 1e-6:
                    flatline_channels.append(raw.ch_names[pick])
                    break

        return flatline_channels

    def _compute_line_noise(
        self,
        raw: mne.io.BaseRaw,
        picks: np.ndarray,
    ) -> float:
        """Compute relative line noise power.

        Returns NaN (never a false 0.0) when the measurement cannot be taken
        (no picks, PSD failure); the caller records a warning (B7 rev2 fix 9).
        """
        if len(picks) == 0:
            logger.warning("line-noise power: no channels to compute a PSD from")
            return float("nan")

        try:
            # Compute PSD
            psd = raw.compute_psd(
                picks=picks,
                fmin=1,
                fmax=100,
                verbose=False,
            )
            freqs = psd.freqs
            powers = psd.get_data().mean(axis=0)  # Average across channels

            # Find line noise frequency band
            line_mask = (freqs >= self.line_freq - 2) & (freqs <= self.line_freq + 2)
            neighbor_mask = (
                ((freqs >= self.line_freq - 10) & (freqs < self.line_freq - 2)) |
                ((freqs > self.line_freq + 2) & (freqs <= self.line_freq + 10))
            )

            line_power = np.mean(powers[line_mask]) if np.any(line_mask) else 0
            neighbor_power = np.mean(powers[neighbor_mask]) if np.any(neighbor_mask) else 1

            return float(line_power / neighbor_power) if neighbor_power > 0 else 0

        except Exception as exc:
            # A failed measurement is not a clean recording (B7 rev2 fix 9).
            logger.warning("line-noise power could not be computed: %s", exc)
            return float("nan")

    def _assess_usability(self, metrics: QCMetrics) -> QCMetrics:
        """Assess overall usability based on thresholds."""
        exclusion_reasons = []
        qc_score = 1.0

        # Check duration
        if metrics.recording_duration_s < self.thresholds.min_duration_s:
            exclusion_reasons.append(
                f"Recording too short: {metrics.recording_duration_s:.1f}s "
                f"< {self.thresholds.min_duration_s}s"
            )
            qc_score -= 0.3

        # Check bad channels
        if metrics.percent_bad_channels > self.thresholds.max_bad_channel_percent:
            exclusion_reasons.append(
                f"Too many bad channels: {metrics.percent_bad_channels:.1f}% "
                f"> {self.thresholds.max_bad_channel_percent}%"
            )
            qc_score -= 0.25

        # Check bad segments
        if metrics.percent_bad_segments > self.thresholds.max_bad_segment_percent:
            exclusion_reasons.append(
                f"Too many bad segments: {metrics.percent_bad_segments:.1f}% "
                f"> {self.thresholds.max_bad_segment_percent}%"
            )
            qc_score -= 0.25

        # Check amplitude
        if metrics.max_amplitude_uv > self.thresholds.max_amplitude_uv:
            exclusion_reasons.append(
                f"Excessive amplitude: {metrics.max_amplitude_uv:.1f}µV "
                f"> {self.thresholds.max_amplitude_uv}µV"
            )
            qc_score -= 0.2

        # Check flatlines
        if metrics.n_flatline_channels > 0:
            pct_flatline = 100 * metrics.n_flatline_channels / metrics.n_channels
            if pct_flatline > 10:
                exclusion_reasons.append(
                    f"Flatline channels: {metrics.n_flatline_channels} ({pct_flatline:.1f}%)"
                )
                qc_score -= 0.2

        # Check line noise
        if metrics.line_noise_power > self.thresholds.max_line_noise_ratio:
            exclusion_reasons.append(
                f"High line noise: {metrics.line_noise_power:.1f}x "
                f"> {self.thresholds.max_line_noise_ratio}x"
            )
            qc_score -= 0.1

        metrics.qc_score = max(0.0, qc_score)
        metrics.exclusion_reasons = exclusion_reasons
        metrics.usable = len(exclusion_reasons) == 0

        return metrics

    def get_output_spec(self) -> Dict[str, str]:
        return {
            "qc_metrics": "QCMetrics dataclass with quality measures",
            "usable": "Boolean indicating if subject data is usable",
        }


def aggregate_qc_report(
    qc_metrics_list: List[QCMetrics],
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Aggregate QC metrics across subjects into a summary report.

    Args:
        qc_metrics_list: List of QCMetrics from multiple subjects
        output_path: Optional path to save CSV report

    Returns:
        Summary statistics dict
    """
    import pandas as pd

    # Convert to dataframe
    records = [m.to_dict() for m in qc_metrics_list]
    df = pd.DataFrame(records)

    # Summary
    summary = {
        "n_subjects": len(df),
        "n_usable": int(df["usable"].sum()),
        "n_excluded": int((~df["usable"]).sum()),
        "percent_usable": 100 * df["usable"].mean(),
        "mean_qc_score": df["qc_score"].mean(),
        "mean_bad_channel_percent": df["percent_bad_channels"].mean(),
        "mean_bad_segment_percent": df["percent_bad_segments"].mean(),
        "excluded_subjects": df[~df["usable"]]["subject_id"].tolist(),
    }

    if output_path:
        df.to_csv(output_path, index=False)

    return summary


from .raw_qc import RawQC, run_raw_qc_batch
from .preprocessed_qc import PreprocessedQC
from .erp_qc import ERPQC
from .preprocessing_comparison_qc import PreprocessingComparisonQC

__all__ = [
    "QCMetric",
    "QCResult",
    "BaseQC",
    "HTMLReportBuilder",
    "QCIndexBuilder",
    "QCMetrics",
    "QCThresholds",
    "QualityControlModule",
    "aggregate_qc_report",
    "RawQC",
    "run_raw_qc_batch",
    "PreprocessedQC",
    "ERPQC",
    "PreprocessingComparisonQC",
]
