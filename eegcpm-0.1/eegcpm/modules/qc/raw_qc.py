"""
Raw EEG data quality control with diagnostic plots and HTML reports.

Provides RawQC class for computing quality metrics on raw EEG data and
generating diagnostic plots (PSD, channel variance, time series, correlations)
with HTML report output.
"""

import logging

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np

from .base import BaseQC, QCMetric, QCResult


logger = logging.getLogger(__name__)

class RawQC(BaseQC):
    """
    Quality control for raw EEG data with diagnostic plots.

    Generates:
    1. PSD plot (all channels overlaid)
    2. Channel variance bar plot (with bad channel highlighting)
    3. Time series plot (first 10 seconds, sample channels)
    4. Channel correlation matrix
    5. Topographic map of mean amplitude (if montage available)

    Attributes:
        line_freq: Line frequency in Hz (50 for EU, 60 for US)
        dpi: DPI for PNG rendering of figures
        thresh_bad_chan_pct: Threshold for bad channel percentage
        thresh_max_amplitude_uv: Threshold for maximum amplitude in µV
        thresh_flatline_std_uv: Threshold for flatline detection in µV
    """

    def __init__(
        self,
        output_dir: Path,
        config: Optional[Dict] = None,
        line_freq: float = 50.0,
        dpi: int = 100,
    ):
        """
        Initialize RawQC module.

        Args:
            output_dir: Directory for output files
            config: Configuration dictionary with optional thresholds
            line_freq: Line frequency (50Hz EU, 60Hz US)
            dpi: DPI for figure rendering
        """
        super().__init__(output_dir, config)
        self.line_freq = line_freq
        self.dpi = dpi

        # Default thresholds - can be overridden in config
        self.thresh_bad_chan_pct = (
            config.get("thresh_bad_chan_pct", 20.0) if config else 20.0
        )
        self.thresh_max_amplitude_uv = (
            config.get("thresh_max_amplitude_uv", 500.0) if config else 500.0
        )
        self.thresh_flatline_std_uv = (
            config.get("thresh_flatline_std_uv", 0.1) if config else 0.1
        )

    def compute(self, data: mne.io.BaseRaw, subject_id: str, **kwargs) -> QCResult:
        """
        Compute QC metrics and generate diagnostic plots.

        Args:
            data: MNE Raw object
            subject_id: Subject identifier
            **kwargs: Additional keyword arguments (unused)

        Returns:
            QCResult with metrics and embedded figures
        """
        raw = data
        result = QCResult(subject_id=subject_id)

        # 1. Basic metrics
        result.add_metric(QCMetric("Duration", raw.times[-1], "s"))
        result.add_metric(QCMetric("Sampling Rate", raw.info["sfreq"], "Hz"))
        result.add_metric(QCMetric("N Channels", len(raw.ch_names), ""))

        # Count EEG channels
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        n_eeg = len(eeg_picks)
        result.add_metric(QCMetric("N EEG Channels", n_eeg, ""))

        # Bad channels
        n_bad = len(raw.info["bads"])
        bad_pct = 100 * n_bad / n_eeg if n_eeg > 0 else 0
        status = "ok" if bad_pct < 10 else ("warning" if bad_pct < 20 else "bad")
        result.add_metric(QCMetric("Bad Channels", n_bad, "", status))
        result.add_metric(QCMetric("Bad Channel %", bad_pct, "%", status))

        # 2. Amplitude statistics
        eeg_picks_good = mne.pick_types(raw.info, eeg=True, exclude="bads")
        if len(eeg_picks_good) > 0:
            data_eeg = raw.get_data(picks=eeg_picks_good)
            mean_amp = np.mean(np.abs(data_eeg)) * 1e6
            max_amp = np.max(np.abs(data_eeg)) * 1e6
            std_amp = np.std(data_eeg) * 1e6

            status = "ok" if max_amp < 300 else ("warning" if max_amp < 500 else "bad")
            result.add_metric(QCMetric("Mean Amplitude", mean_amp, "µV"))
            result.add_metric(QCMetric("Max Amplitude", max_amp, "µV", status))
            result.add_metric(QCMetric("Std Amplitude", std_amp, "µV"))
        else:
            result.add_note("No good EEG channels available for amplitude analysis")

        # 3. Detect flatlines
        flatlines = self._detect_flatlines(raw, eeg_picks_good)
        status = "ok" if len(flatlines) == 0 else "warning"
        result.add_metric(QCMetric("Flatline Channels", len(flatlines), "", status))
        if flatlines:
            result.add_note(f"Flatline channels: {', '.join(flatlines)}")

        # 4. Line noise
        line_noise = self._compute_line_noise_ratio(raw, eeg_picks_good)
        if np.isfinite(line_noise):
            status = "ok" if line_noise < 3 else ("warning" if line_noise < 5 else "bad")
        else:
            # A failed measurement must not look like a clean recording (B7 rev2).
            status = "warning"
            result.add_note(
                "Line noise ratio could not be computed (see log); reported "
                "as NaN, not a false 0.0"
            )
        result.add_metric(QCMetric("Line Noise Ratio", line_noise, "x", status))

        # 5. Generate plots
        try:
            # PSD plot
            fig_psd = self._plot_psd(raw, eeg_picks_good)
            result.add_figure("psd", self.fig_to_base64(fig_psd, self.dpi))
            plt.close(fig_psd)
        except Exception as e:
            result.add_note(f"PSD plot generation failed: {str(e)}")

        try:
            # Channel variance plot
            fig_var = self._plot_channel_variance(raw, eeg_picks)
            result.add_figure("channel_variance", self.fig_to_base64(fig_var, self.dpi))
            plt.close(fig_var)
        except Exception as e:
            result.add_note(f"Channel variance plot generation failed: {str(e)}")

        try:
            # Time series plot
            fig_ts = self._plot_time_series(raw, eeg_picks_good)
            result.add_figure("time_series", self.fig_to_base64(fig_ts, self.dpi))
            plt.close(fig_ts)
        except Exception as e:
            result.add_note(f"Time series plot generation failed: {str(e)}")

        try:
            # Channel correlation matrix
            fig_corr = self._plot_correlation_matrix(raw, eeg_picks_good)
            result.add_figure("correlation", self.fig_to_base64(fig_corr, self.dpi))
            plt.close(fig_corr)
        except Exception as e:
            result.add_note(f"Correlation plot generation failed: {str(e)}")

        try:
            # Temporal artifact distribution
            fig_temporal = self._plot_temporal_artifact_distribution(raw, eeg_picks_good)
            result.add_figure("temporal_artifacts", self.fig_to_base64(fig_temporal, self.dpi))
            plt.close(fig_temporal)
        except Exception as e:
            result.add_note(f"Temporal artifact plot generation failed: {str(e)}")

        # Overall status
        bad_metrics = [m for m in result.metrics if m.status == "bad"]
        warn_metrics = [m for m in result.metrics if m.status == "warning"]
        if bad_metrics:
            result.status = "bad"
        elif warn_metrics:
            result.status = "warning"
        else:
            result.status = "ok"

        return result

    def _detect_flatlines(
        self, raw: mne.io.BaseRaw, picks: np.ndarray, window_s: float = 5.0
    ) -> List[str]:
        """
        Detect channels with near-zero variance segments.

        Args:
            raw: Raw EEG data
            picks: Channel picks to check
            window_s: Window duration in seconds for flatline detection

        Returns:
            List of channel names with flatlines
        """
        flatlines = []
        n_samples = int(window_s * raw.info["sfreq"])

        if n_samples == 0 or len(picks) == 0:
            return flatlines

        for pick in picks:
            ch_data = raw.get_data(picks=[pick])[0]
            # Check windows
            for start in range(0, len(ch_data) - n_samples, n_samples):
                window = ch_data[start : start + n_samples]
                if np.std(window) * 1e6 < self.thresh_flatline_std_uv:
                    flatlines.append(raw.ch_names[pick])
                    break

        return flatlines

    def _compute_line_noise_ratio(
        self, raw: mne.io.BaseRaw, picks: np.ndarray
    ) -> float:
        """
        Compute relative power at line frequency vs neighbors.

        Args:
            raw: Raw EEG data
            picks: Channel picks to analyze

        Returns:
            Ratio of line frequency power to neighboring frequency power
        """
        if len(picks) == 0:
            # No usable channels: NaN, not a false 0.0 "clean" reading (B7 rev2).
            logger.warning("line-noise ratio: no channels to compute a PSD from")
            return float("nan")

        try:
            psd = raw.compute_psd(picks=picks, fmin=1, fmax=100, verbose=False)
            freqs = psd.freqs
            powers = psd.get_data().mean(axis=0)

            line_mask = (freqs >= self.line_freq - 2) & (freqs <= self.line_freq + 2)
            neighbor_mask = (
                ((freqs >= self.line_freq - 10) & (freqs < self.line_freq - 2))
                | ((freqs > self.line_freq + 2) & (freqs <= self.line_freq + 10))
            )

            line_power = np.mean(powers[line_mask]) if np.any(line_mask) else 0
            neighbor_power = (
                np.mean(powers[neighbor_mask]) if np.any(neighbor_mask) else 1
            )
            return float(line_power / neighbor_power) if neighbor_power > 0 else 0
        except Exception as exc:
            # A failed measurement is NOT a clean recording: report NaN and
            # record why, so the caller can flag it (B7 rev2 fix 9).
            logger.warning("line-noise ratio could not be computed: %s", exc)
            return float("nan")

    def _plot_psd(
        self, raw: mne.io.BaseRaw, picks: np.ndarray
    ) -> plt.Figure:
        """
        Plot power spectral density.

        Args:
            raw: Raw EEG data
            picks: Channel picks to plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 4))

        if len(picks) > 0:
            psd = raw.compute_psd(picks=picks, fmin=0.5, fmax=80, verbose=False)
            psd.plot(axes=ax, show=False)
            ax.axvline(
                self.line_freq,
                color="r",
                linestyle="--",
                alpha=0.5,
                label=f"{self.line_freq}Hz",
            )
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "No EEG channels",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_title("Power Spectral Density")
        fig.tight_layout()
        return fig

    def _plot_channel_variance(
        self, raw: mne.io.BaseRaw, picks: np.ndarray
    ) -> plt.Figure:
        """
        Plot channel variance as bar chart with bad channel highlighting.

        Args:
            raw: Raw EEG data
            picks: Channel picks to plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 4))

        if len(picks) > 0:
            data = raw.get_data(picks=picks)
            variances = np.var(data, axis=1) * 1e12  # Convert to µV²
            ch_names = [raw.ch_names[p] for p in picks]

            colors = [
                "red" if ch in raw.info["bads"] else "steelblue" for ch in ch_names
            ]
            ax.bar(range(len(variances)), variances, color=colors)

            # Only show subset of labels if too many
            if len(ch_names) > 30:
                step = len(ch_names) // 20
                ax.set_xticks(range(0, len(ch_names), step))
                ax.set_xticklabels(
                    [ch_names[i] for i in range(0, len(ch_names), step)],
                    rotation=45,
                    ha="right",
                )
            else:
                ax.set_xticks(range(len(ch_names)))
                ax.set_xticklabels(ch_names, rotation=45, ha="right")

            ax.set_ylabel("Variance (µV²)")
            ax.set_xlabel("Channel")
        else:
            ax.text(
                0.5,
                0.5,
                "No EEG channels",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_title("Channel Variance (red = bad)")
        fig.tight_layout()
        return fig

    def _plot_time_series(
        self,
        raw: mne.io.BaseRaw,
        picks: np.ndarray,
        duration: float = 10.0,
        n_channels: int = 8,
    ) -> plt.Figure:
        """
        Plot sample time series.

        Args:
            raw: Raw EEG data
            picks: Channel picks to plot
            duration: Duration in seconds to display
            n_channels: Maximum number of channels to show

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        if len(picks) > 0:
            # Select subset of channels
            if len(picks) > n_channels:
                step = len(picks) // n_channels
                selected = picks[::step][:n_channels]
            else:
                selected = picks

            n_samples = int(duration * raw.info["sfreq"])
            n_samples = min(n_samples, raw.n_times)
            data = raw.get_data(picks=selected)[:, :n_samples]
            times = raw.times[:n_samples]

            # Stack with offset
            offset = np.max(np.abs(data)) * 2.5
            for i, pick in enumerate(selected):
                ax.plot(times, data[i] * 1e6 + i * offset, "k", linewidth=0.5)

            ax.set_yticks([i * offset for i in range(len(selected))])
            ax.set_yticklabels([raw.ch_names[p] for p in selected])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Channel")
        else:
            ax.text(
                0.5,
                0.5,
                "No EEG channels",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_title(f"Time Series (first {duration}s)")
        fig.tight_layout()
        return fig

    def _plot_correlation_matrix(
        self, raw: mne.io.BaseRaw, picks: np.ndarray, sample_duration: float = 30.0
    ) -> plt.Figure:
        """
        Plot channel correlation matrix.

        Args:
            raw: Raw EEG data
            picks: Channel picks to correlate
            sample_duration: Duration in seconds to use for correlation

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 7))

        if len(picks) > 2:
            n_samples = int(sample_duration * raw.info["sfreq"])
            n_samples = min(n_samples, raw.n_times)
            data = raw.get_data(picks=picks)[:, :n_samples]

            # Filter out channels with zero variance (dead channels)
            variances = np.var(data, axis=1)
            valid_mask = variances > 1e-20
            if np.sum(valid_mask) > 2:
                data_valid = data[valid_mask]
                valid_picks = picks[valid_mask]
                corr = np.corrcoef(data_valid)
                # Handle any remaining NaN/Inf values
                corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
            else:
                corr = np.corrcoef(data)
                corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
                valid_picks = picks

            im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
            fig.colorbar(im, ax=ax, label="Correlation")

            # Labels (show subset if too many)
            ch_names = [raw.ch_names[p] for p in picks]
            if len(ch_names) > 20:
                step = len(ch_names) // 10
                ax.set_xticks(range(0, len(ch_names), step))
                ax.set_yticks(range(0, len(ch_names), step))
                ax.set_xticklabels(
                    [ch_names[i] for i in range(0, len(ch_names), step)],
                    rotation=45,
                    ha="right",
                )
                ax.set_yticklabels([ch_names[i] for i in range(0, len(ch_names), step)])
            else:
                ax.set_xticks(range(len(ch_names)))
                ax.set_yticks(range(len(ch_names)))
                ax.set_xticklabels(ch_names, rotation=45, ha="right")
                ax.set_yticklabels(ch_names)
        else:
            ax.text(
                0.5,
                0.5,
                "Not enough channels",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_title("Channel Correlation Matrix")
        fig.tight_layout()
        return fig

    def _plot_temporal_artifact_distribution(
        self, raw: mne.io.BaseRaw, picks: np.ndarray, window_sec: float = 5.0
    ) -> plt.Figure:
        """
        Plot temporal distribution of artifacts over recording.

        Shows when artifacts occur during recording using sliding window
        peak-to-peak amplitude. Helps identify:
        - Subject movement periods
        - Equipment failures
        - Systematic temporal patterns

        Args:
            raw: Raw EEG data
            picks: Channel picks to analyze
            window_sec: Window size in seconds for artifact detection

        Returns:
            Matplotlib figure with two subplots:
            1. Peak-to-peak amplitude over time
            2. Histogram of amplitude distribution
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

        if len(picks) == 0:
            ax1.text(0.5, 0.5, "No valid channels", ha="center", va="center",
                    transform=ax1.transAxes)
            ax2.text(0.5, 0.5, "No valid channels", ha="center", va="center",
                    transform=ax2.transAxes)
            fig.tight_layout()
            return fig

        sfreq = raw.info["sfreq"]
        window_samples = int(window_sec * sfreq)
        step_samples = window_samples // 2  # 50% overlap

        # Get data
        data = raw.get_data(picks=picks)  # (n_channels, n_times)
        n_times = data.shape[1]

        # Calculate sliding window peak-to-peak amplitude
        window_starts = range(0, n_times - window_samples, step_samples)
        ptp_values = []
        time_centers = []

        for start in window_starts:
            end = start + window_samples
            window_data = data[:, start:end]

            # Peak-to-peak amplitude (max - min) across channels
            ptp = np.max(window_data) - np.min(window_data)
            ptp_values.append(ptp * 1e6)  # Convert to µV

            # Time at center of window
            time_centers.append((start + window_samples / 2) / sfreq)

        ptp_values = np.array(ptp_values)
        time_centers = np.array(time_centers)

        # Subplot 1: Temporal evolution
        ax1.plot(time_centers, ptp_values, linewidth=1, alpha=0.7)
        ax1.fill_between(time_centers, ptp_values, alpha=0.3)

        # Add threshold lines
        median_ptp = np.median(ptp_values)
        threshold_high = np.percentile(ptp_values, 90)
        threshold_very_high = np.percentile(ptp_values, 95)

        ax1.axhline(median_ptp, color='green', linestyle='--', linewidth=1,
                   label=f'Median: {median_ptp:.1f} µV', alpha=0.7)
        ax1.axhline(threshold_high, color='orange', linestyle='--', linewidth=1,
                   label=f'90th %ile: {threshold_high:.1f} µV', alpha=0.7)
        ax1.axhline(threshold_very_high, color='red', linestyle='--', linewidth=1,
                   label=f'95th %ile: {threshold_very_high:.1f} µV', alpha=0.7)

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Peak-to-Peak Amplitude (µV)')
        ax1.set_title(f'Temporal Artifact Distribution ({window_sec}s windows, 50% overlap)')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Histogram of amplitudes
        ax2.hist(ptp_values, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        ax2.axvline(median_ptp, color='green', linestyle='--', linewidth=2,
                   label=f'Median: {median_ptp:.1f} µV')
        ax2.axvline(threshold_high, color='orange', linestyle='--', linewidth=2,
                   label=f'90th %ile: {threshold_high:.1f} µV')
        ax2.axvline(threshold_very_high, color='red', linestyle='--', linewidth=2,
                   label=f'95th %ile: {threshold_very_high:.1f} µV')

        ax2.set_xlabel('Peak-to-Peak Amplitude (µV)')
        ax2.set_ylabel('Count')
        ax2.set_title('Amplitude Distribution Across Windows')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add statistics text
        stats_text = (
            f'Mean: {np.mean(ptp_values):.1f} µV\n'
            f'Std: {np.std(ptp_values):.1f} µV\n'
            f'High artifact windows (>90th %ile): {np.sum(ptp_values > threshold_high)}/{len(ptp_values)}'
        )
        ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.tight_layout()
        return fig

    def generate_html_report(
        self, result: QCResult, save_path: Optional[Path] = None
    ) -> str:
        """
        Generate HTML report from QCResult.

        Args:
            result: QCResult with metrics and figures
            save_path: Optional path to save HTML

        Returns:
            HTML string
        """
        from .html_report import HTMLReportBuilder

        builder = HTMLReportBuilder(title=f"QC Report: {result.subject_id}")

        # Status badge
        builder.add_header(f"Overall Status: {result.status.upper()}", level=2)

        # Metrics table
        builder.add_header("Metrics", level=2)
        builder.add_metrics_table(result.metrics)

        # Notes
        if result.notes:
            builder.add_notes(result.notes)

        # Figures
        builder.add_header("Diagnostic Plots", level=2)

        figure_captions = {
            "psd": "Power Spectral Density across all EEG channels",
            "channel_variance": "Variance per channel (red = marked as bad)",
            "time_series": "Sample time series showing data quality",
            "correlation": "Channel correlation matrix",
            "temporal_artifacts": "Temporal distribution of artifacts - shows when artifacts occur during recording",
        }

        for name, fig_bytes in result.figures.items():
            caption = figure_captions.get(name, name)
            builder.add_figure(name, fig_bytes, caption)

        html = builder.build()

        if save_path:
            Path(save_path).write_text(html)

        return html


def run_raw_qc_batch(
    raw_files: List[Path],
    output_dir: Path,
    config: Optional[Dict] = None,
    line_freq: float = 50.0,
) -> Tuple[List[QCResult], Path]:
    """
    Run raw QC on multiple files and generate index.html.

    Args:
        raw_files: List of paths to raw files
        output_dir: Output directory for reports
        config: Optional configuration dict
        line_freq: Line frequency (50 or 60 Hz)

    Returns:
        Tuple of (list of QCResults, path to index.html)
    """
    from eegcpm.data.loaders import load_raw

    from .html_report import QCIndexBuilder

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qc = RawQC(output_dir, config=config, line_freq=line_freq)
    index_builder = QCIndexBuilder(title="Raw QC Index")
    results = []

    for raw_path in raw_files:
        # Extract subject ID from filename
        subject_id = raw_path.stem.split("_")[0]

        try:
            raw = load_raw(raw_path, preload=True, verbose=False)
            result = qc.compute(raw, subject_id)

            # Save individual report
            html_filename = f"{subject_id}_qc.html"
            qc.generate_html_report(result, output_dir / html_filename)

            # Save JSON
            result.save_json(output_dir / f"{subject_id}_qc.json")

            # Add to index
            index_builder.add_subject(subject_id, result.status, html_filename)
            results.append(result)

        except Exception as exc:
            # One broken recording must not stop the batch, but it must not be
            # silently absent from the results either (P1-F11).
            logger.warning("error processing %s: %s", raw_path, exc)
            continue

    # Save index
    index_path = output_dir / "index.html"
    index_builder.save(index_path)

    return results, index_path
