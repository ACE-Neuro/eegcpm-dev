"""
Source reconstruction Quality Control module.

Generates QC reports for source reconstruction results including:
- ROI coverage (which ROIs have valid data)
- ROI time courses for key ROIs
- SNR estimates per ROI
- Source power distribution
- Network activation summary
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import mne

from .base import BaseQC, QCMetric, QCResult
from .html_report import HTMLReportBuilder


class SourceQC(BaseQC):
    """Quality control for source reconstruction results."""

    @staticmethod
    def _format_signal_value(value: float) -> str:
        """Format source values without rounding tiny eLORETA signals to zero."""
        if value != 0 and abs(value) < 0.01:
            return f"{value:.2e}"
        return f"{value:.4f}"

    def _parse_roi_data(self, roi_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse ROI data into a consistent format.

        The source module stores data as:
            Evoked mode:
                roi_data['condition'] = (n_rois, n_times)
            Trial-level mode:
                roi_data['condition'] = (n_trials, n_rois, n_times)

            roi_data['condition_times'] = (n_times,)
            roi_data['roi_names'] = ['ROI1', 'ROI2', ...]

        This method extracts condition names, times, and the data matrix.

        Returns
        -------
        dict with keys:
            'roi_names': list of ROI names
            'conditions': list of condition names
            'times': dict of condition -> time array
            'data': dict of condition -> (n_rois, n_times) or (n_trials, n_rois, n_times) array
        """
        roi_names = roi_data.get('roi_names', [])
        if isinstance(roi_names, np.ndarray):
            roi_names = list(roi_names)

        # Find condition keys (exclude metadata keys)
        metadata_keys = {'roi_names', 'roi_coords_mni', 'roi_radius_mm', 'roi_vertex_counts', 'sfreq'}
        condition_keys = []
        for key in roi_data.keys():
            if key not in metadata_keys and not key.endswith('_times'):
                condition_keys.append(key)

        candidate_conditions = sorted(condition_keys)
        times = {}
        data = {}

        for cond in candidate_conditions:
            if cond in roi_data:
                arr = roi_data[cond]
                # Accept both 2D (evoked) and 3D (trial-level) data
                if isinstance(arr, np.ndarray) and arr.ndim in [2, 3]:
                    if arr.ndim == 3:
                        arr = arr.mean(axis=0)  # Average trials -> (n_rois, n_times)
                    data[cond] = arr
                    # Get times if available
                    times_key = f"{cond}_times"
                    if times_key in roi_data:
                        times[cond] = roi_data[times_key]

        # Only report conditions that produced valid ROI arrays.
        conditions = sorted(data.keys())

        return {
            'roi_names': roi_names,
            'conditions': conditions,
            'times': times,
            'data': data,
        }

    def compute(
        self,
        data: Dict[str, Any],
        subject_id: str,
        **kwargs
    ) -> QCResult:
        """
        Compute QC metrics for source reconstruction.

        Parameters
        ----------
        data : dict
            Dictionary with keys:
            - 'stcs': Dict[str, mne.SourceEstimate] - Source estimates per condition
            - 'roi_data': Dict[str, np.ndarray] - ROI time courses
            - 'method': str - Reconstruction method (dSPM, sLORETA, etc.)
            - 'parcellation': str - Parcellation name
        subject_id : str
            Subject identifier
        **kwargs : dict
            Additional arguments (session_id, sfreq, etc.)

        Returns
        -------
        QCResult
            QC result with metrics and figures
        """
        result = QCResult(subject_id=subject_id)

        stcs = data.get('stcs', {})
        roi_data = data.get('roi_data', {})
        method = data.get('method', 'unknown')
        parcellation = data.get('parcellation', 'unknown')

        # Parse ROI data into consistent format
        parsed = self._parse_roi_data(roi_data) if roi_data else None

        # Add metadata. In trial-level source reconstruction, STCs are intentionally
        # omitted to avoid huge files, so condition metadata must come from ROI data.
        roi_conditions = parsed.get('conditions', []) if parsed else []
        stc_conditions = list(stcs.keys()) if stcs else []
        conditions = roi_conditions or stc_conditions
        result.metadata['method'] = method
        result.metadata['parcellation'] = parcellation
        result.metadata['n_conditions'] = len(conditions)
        result.metadata['conditions'] = conditions

        # Compute metrics
        if parsed and parsed['roi_names']:
            n_rois = len(parsed['roi_names'])
            n_rois_with_data = self._count_rois_with_data(parsed)
            result.add_metric(QCMetric(
                name="ROI Time Series Present",
                value=n_rois_with_data,
                unit=f"of {n_rois} ROIs",
                status="ok" if n_rois_with_data == n_rois and n_rois > 0 else "warning"
            ))

            # Average signal strength across all ROIs and conditions
            if parsed['data']:
                all_powers = []
                for cond, arr in parsed['data'].items():
                    # arr is (n_rois, n_times)
                    all_powers.append(np.mean(np.abs(arr)))
                if all_powers:
                    avg_power = np.mean(all_powers)
                    result.add_metric(QCMetric(
                        name="Average ROI Signal",
                        value=float(avg_power),
                        unit="a.u.",
                        status="ok"
                    ))

        # Generate figures
        if stcs is not None and len(stcs) > 0:
            fig_power = self._plot_source_power_distribution(stcs)
            if fig_power:
                result.add_figure("source_power", self.fig_to_base64(fig_power))
                plt.close(fig_power)

        if parsed and parsed['roi_names']:
            fig_coverage = self._plot_roi_coverage(parsed)
            if fig_coverage:
                result.add_figure("roi_coverage", self.fig_to_base64(fig_coverage))
                plt.close(fig_coverage)

            fig_tc = self._plot_roi_timecourses(parsed)
            if fig_tc:
                result.add_figure("roi_timecourses", self.fig_to_base64(fig_tc))
                plt.close(fig_tc)

            if parcellation == 'conn_networks':
                fig_network = self._plot_network_summary(parsed)
                if fig_network:
                    result.add_figure("network_summary", self.fig_to_base64(fig_network))
                    plt.close(fig_network)

            fig_erp = self._plot_source_erp_waveforms(parsed, **kwargs)
            if fig_erp:
                result.add_figure("source_erp", self.fig_to_base64(fig_erp))
                plt.close(fig_erp)

            # Figure 6: ROI correlation matrix
            fig_corr = self._plot_roi_correlation_matrix(parsed)
            if fig_corr:
                result.add_figure("roi_correlation", self.fig_to_base64(fig_corr))
                plt.close(fig_corr)

        # Figure 7: Source crosstalk matrix (if forward/inverse operators available)
        fwd = data.get('forward')
        inv_op = data.get('inverse_operator')
        labels = data.get('labels')

        if fwd is not None and inv_op is not None and labels is not None:
            try:
                # Compute or load crosstalk matrix
                crosstalk = data.get('crosstalk_matrix')
                if crosstalk is None:
                    # Compute crosstalk (expensive, should ideally be pre-computed)
                    result.add_note("Computing crosstalk matrix (this may take several minutes)...")
                    crosstalk = self._compute_crosstalk_matrix(fwd, inv_op, labels, method)

                # Plot crosstalk matrix
                roi_names = parsed['roi_names'] if parsed else [label.name for label in labels]
                fig_crosstalk = self._plot_crosstalk_matrix(crosstalk, roi_names, threshold=0.3)
                if fig_crosstalk:
                    result.add_figure("crosstalk_matrix", self.fig_to_base64(fig_crosstalk))
                    plt.close(fig_crosstalk)

                    # Add crosstalk statistics to metadata
                    mask = ~np.eye(crosstalk.shape[0], dtype=bool)
                    off_diag = crosstalk[mask]
                    n_high = np.sum(np.triu(crosstalk, k=1) > 0.3)
                    result.metadata['crosstalk_mean'] = float(np.mean(off_diag))
                    result.metadata['crosstalk_high_pairs'] = int(n_high)

            except Exception as e:
                result.add_note(f"Crosstalk matrix computation failed: {str(e)}")

        # Overall status
        has_roi = parsed is not None and len(parsed.get('roi_names', [])) > 0
        has_roi_data = parsed is not None and bool(parsed.get('data'))
        has_stc = stcs is not None and len(stcs) > 0
        # Trial-level source mode is valid with ROI data and no saved STCs.
        result.status = "ok" if has_roi and (has_roi_data or has_stc) else "bad"

        return result

    def _count_rois_with_data(self, parsed: Dict[str, Any]) -> int:
        """Count ROIs with finite, non-zero samples in at least one condition."""
        roi_names = parsed.get('roi_names', [])
        data = parsed.get('data', {})
        if not roi_names or not data:
            return 0

        has_data = np.zeros(len(roi_names), dtype=bool)
        for arr in data.values():
            if not isinstance(arr, np.ndarray):
                continue
            if arr.ndim == 3:
                roi_has_data = np.any(np.isfinite(arr) & (arr != 0), axis=(0, 2))
            elif arr.ndim == 2:
                roi_has_data = np.any(np.isfinite(arr) & (arr != 0), axis=1)
            else:
                continue
            has_data[: len(roi_has_data)] |= roi_has_data[: len(has_data)]

        return int(has_data.sum())

    def _plot_source_power_distribution(
        self,
        stcs: Dict[str, mne.SourceEstimate]
    ) -> Optional[plt.Figure]:
        """Plot distribution of source amplitudes."""
        if not stcs:
            return None

        fig, axes = plt.subplots(1, len(stcs), figsize=(5 * len(stcs), 4))
        if len(stcs) == 1:
            axes = [axes]

        for ax, (condition, stc) in zip(axes, stcs.items()):
            # Get absolute source amplitudes
            amplitudes = np.abs(stc.data).flatten()

            # Plot histogram
            ax.hist(amplitudes, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Source Amplitude (a.u.)')
            ax.set_ylabel('Count')
            ax.set_title(f'{condition}')
            ax.grid(True, alpha=0.3)

            # Add mean/median lines
            mean_amp = np.mean(amplitudes)
            median_amp = np.median(amplitudes)
            ax.axvline(mean_amp, color='red', linestyle='--', label=f'Mean: {mean_amp:.2f}')
            ax.axvline(median_amp, color='blue', linestyle='--', label=f'Median: {median_amp:.2f}')
            ax.legend()

        fig.suptitle('Source Power Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return fig

    def _plot_roi_coverage(self, parsed: Dict[str, Any]) -> Optional[plt.Figure]:
        """Plot which ROIs have valid data."""
        roi_names = parsed.get('roi_names', [])
        if not roi_names:
            return None

        n_rois = len(roi_names)
        has_data = np.zeros(n_rois, dtype=int)
        data = parsed.get('data', {})
        for arr in data.values():
            if not isinstance(arr, np.ndarray):
                continue
            if arr.ndim == 3:
                roi_has_data = np.any(np.isfinite(arr) & (arr != 0), axis=(0, 2))
            elif arr.ndim == 2:
                roi_has_data = np.any(np.isfinite(arr) & (arr != 0), axis=1)
            else:
                continue
            has_data[: len(roi_has_data)] |= roi_has_data[: len(has_data)].astype(int)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Bar plot showing all ROIs have data
        colors = ['green' if val else 'red' for val in has_data]
        ax.bar(range(n_rois), has_data, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(n_rois))
        ax.set_xticklabels(roi_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Has Data')
        ax.set_ylim([0, 1.2])
        ax.set_title(
            f'ROI Time Series Present ({int(has_data.sum())}/{n_rois} ROIs)',
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def _plot_roi_timecourses(self, parsed: Dict[str, Any]) -> Optional[plt.Figure]:
        """Plot time courses for top 6 ROIs by power."""
        roi_names = parsed.get('roi_names', [])
        conditions = parsed.get('conditions', [])
        data = parsed.get('data', {})
        times_dict = parsed.get('times', {})

        if not roi_names or not conditions or not data:
            return None

        n_rois = len(roi_names)

        # Calculate mean power for each ROI across all conditions
        roi_powers = []
        for roi_idx, roi_name in enumerate(roi_names):
            powers = []
            for cond in conditions:
                if cond in data:
                    arr = data[cond]
                    # Handle both 2D (n_rois, n_times) and 3D (n_trials, n_rois, n_times)
                    if arr.ndim == 3:
                        # Average across trials then compute power
                        roi_tc = arr[:, roi_idx, :].mean(axis=0)
                        powers.append(np.mean(np.abs(roi_tc)))
                    else:
                        powers.append(np.mean(np.abs(arr[roi_idx, :])))
            if powers:
                mean_power = np.mean(powers)
                roi_powers.append((roi_idx, roi_name, mean_power))

        if not roi_powers:
            return None

        # Sort by power and take top 6
        roi_powers.sort(key=lambda x: x[2], reverse=True)
        top_rois = roi_powers[:6]

        # Get time axis from first condition (or use samples)
        first_cond = conditions[0]
        arr = data[first_cond]

        # Handle both 2D (n_rois, n_times) and 3D (n_trials, n_rois, n_times)
        if arr.ndim == 3:
            n_times = arr.shape[2]
        else:
            n_times = arr.shape[1]

        if first_cond in times_dict and times_dict[first_cond] is not None:
            times = times_dict[first_cond]
            # Ensure times is 1D
            if isinstance(times, np.ndarray) and times.ndim > 1:
                times = times.flatten()
            time_label = 'Time (s)'
        else:
            times = np.arange(n_times)
            time_label = 'Time (samples)'

        # Plot
        n_plots = len(top_rois)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2 * n_plots))
        if n_plots == 1:
            axes = [axes]

        colors = plt.cm.tab10.colors

        for ax, (roi_idx, roi_name, power) in zip(axes, top_rois):
            # Plot all conditions for this ROI
            for cond_idx, cond in enumerate(conditions):
                if cond in data:
                    arr = data[cond]
                    # Handle both 2D (n_rois, n_times) and 3D (n_trials, n_rois, n_times)
                    if arr.ndim == 3:
                        # Average across trials
                        tc = arr[:, roi_idx, :].mean(axis=0)
                    else:
                        tc = arr[roi_idx, :]  # Time course for this ROI
                    ax.plot(times, tc, label=cond, alpha=0.8, color=colors[cond_idx % 10])

            ax.set_ylabel('Amplitude (a.u.)')
            ax.set_title(f'{roi_name} (mean |signal|: {self._format_signal_value(power)})')
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.axvline(0, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
            if len(conditions) > 1:
                ax.legend(loc='upper right', fontsize=8)

        axes[-1].set_xlabel(time_label)
        fig.suptitle('ROI Time Courses (Top 6 by Mean Absolute Signal)', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return fig

    def _plot_network_summary(self, parsed: Dict[str, Any]) -> Optional[plt.Figure]:
        """Plot network activation summary for CONN parcellation."""
        # CONN network definitions (short names from parcellation)
        # ROI names in data are like "DefaultMode.MPFC", "SensoriMotor.Lateral", etc.
        networks = {
            'DefaultMode': ['MPFC', 'LP', 'PCC'],
            'SensoriMotor': ['Lateral', 'Superior'],
            'Visual': ['Medial', 'Occipital', 'Lateral'],
            'Salience': ['ACC', 'AInsula', 'RPFC', 'SMG'],
            'DorsalAttention': ['FEF', 'IPS'],
            'FrontoParietal': ['LPFC', 'PPC'],
            'Language': ['IFG', 'pSTG'],
            'Cerebellar': ['Anterior', 'Posterior']
        }

        roi_names = parsed.get('roi_names', [])
        conditions = parsed.get('conditions', [])
        data = parsed.get('data', {})

        if not roi_names or not conditions or not data:
            return None

        # Calculate average signal per network across conditions. Use the network
        # prefix in names like "DefaultMode.LP" so duplicated L/R ROI names are
        # both included instead of overwritten in a short-name lookup.
        network_powers = {}
        for network_name in networks:
            powers = []
            roi_indices = [
                idx for idx, full_name in enumerate(roi_names)
                if str(full_name).startswith(f"{network_name}.")
            ]
            for roi_idx in roi_indices:
                for cond in conditions:
                    if cond in data:
                        arr = data[cond]
                        mean_power = np.mean(np.abs(arr[roi_idx, :]))
                        powers.append(mean_power)

            # Backward-compatible fallback for any ROI names without network prefix.
            if not powers:
                for short_name in networks[network_name]:
                    for idx, full_name in enumerate(roi_names):
                        roi_short = str(full_name).split('.', 1)[-1]
                        if roi_short == short_name:
                            for cond in conditions:
                                if cond in data:
                                    arr = data[cond]
                                    mean_power = np.mean(np.abs(arr[idx, :]))
                                    powers.append(mean_power)

            if powers:
                network_powers[network_name] = np.mean(powers)
            else:
                network_powers[network_name] = 0

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        network_names = list(network_powers.keys())
        powers = list(network_powers.values())

        ax.bar(range(len(network_names)), powers, alpha=0.7, edgecolor='black', color='steelblue')
        ax.set_xticks(range(len(network_names)))
        ax.set_xticklabels(network_names, rotation=45, ha='right')
        ax.set_ylabel('Mean |Signal| (a.u.)')
        ax.set_title('Network Mean Signal Summary (CONN 32 ROIs)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def _plot_source_erp_waveforms(
        self,
        parsed: Dict[str, Any],
        **kwargs
    ) -> Optional[plt.Figure]:
        """
        Plot source-level ERP waveforms for ALL ROIs with 95% confidence intervals.

        Shows event-related potentials at the ROI level across all conditions.
        When trial-level data (3D) is available, plots mean ± 95% CI.
        When evoked data (2D) is available, plots single waveform.

        Includes left/right hemisphere labels for paired ROIs.
        """
        roi_names = parsed.get('roi_names', [])
        conditions = parsed.get('conditions', [])
        data = parsed.get('data', {})
        times_dict = parsed.get('times', {})

        if not roi_names or not conditions or not data:
            return None

        # Get sampling rate from kwargs
        sfreq = kwargs.get('sfreq', 500.0)

        # Add laterality labels (L/R) to ROI names
        # ROIs are stored with duplicates representing left/right hemispheres
        display_names = []
        roi_counts = {}
        for name in roi_names:
            if name in roi_counts:
                # Second occurrence - add R suffix
                display_names.append(f"{name}.R")
                roi_counts[name] += 1
            else:
                # First occurrence - check if there will be a duplicate
                if list(roi_names).count(name) > 1:
                    display_names.append(f"{name}.L")
                else:
                    # Midline ROI - no laterality
                    display_names.append(name)
                roi_counts[name] = 1

        n_rois = len(roi_names)

        # Get time axis from first condition
        first_cond = conditions[0]
        if first_cond in times_dict:
            times = times_dict[first_cond]
        else:
            # Fallback: construct time axis
            # Determine n_times from data (handle both 2D and 3D)
            first_data = data[first_cond]
            n_times = first_data.shape[-1]  # Last dimension is always time
            times = np.arange(n_times) / sfreq
            # Center at 0 (assume epoch from -tmin to tmax)
            times = times - times[len(times) // 2]

        # Detect if data is trial-level (3D) or evoked (2D)
        first_data = data[first_cond]
        is_trial_level = first_data.ndim == 3

        # Create subplots: 4 columns for better layout with 32 ROIs
        n_cols = 4
        n_rows = (n_rois + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
        axes = axes.flatten() if n_rois > 1 else [axes]

        colors = plt.cm.tab10.colors

        for roi_idx in range(n_rois):
            ax = axes[roi_idx]
            roi_display_name = display_names[roi_idx]

            # Plot all conditions for this ROI
            for cond_idx, cond in enumerate(conditions):
                if cond not in data:
                    continue

                color = colors[cond_idx % 10]
                cond_data = data[cond]

                if is_trial_level:
                    # 3D data: (n_trials, n_rois, n_times)
                    # Extract trials for this ROI
                    roi_trials = cond_data[:, roi_idx, :]  # (n_trials, n_times)
                    n_trials = roi_trials.shape[0]

                    # Compute mean and 95% CI
                    mean_tc = np.mean(roi_trials, axis=0)  # (n_times,)
                    std_tc = np.std(roi_trials, axis=0)    # (n_times,)
                    sem = std_tc / np.sqrt(n_trials)       # Standard error of mean
                    ci_95 = 1.96 * sem                     # 95% confidence interval

                    # Plot mean with CI band
                    ax.plot(times, mean_tc, label=cond, alpha=0.9, linewidth=1.5, color=color)
                    ax.fill_between(
                        times,
                        mean_tc - ci_95,
                        mean_tc + ci_95,
                        alpha=0.2,
                        color=color,
                        label=f'{cond} 95% CI'
                    )
                else:
                    # 2D data: (n_rois, n_times) - evoked response
                    tc = cond_data[roi_idx, :]  # (n_times,)
                    ax.plot(times, tc, label=cond, alpha=0.9, linewidth=1.5, color=color)

            # Reference lines
            ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.axvline(0, color='r', linestyle='--', linewidth=0.5, alpha=0.5)

            # Shading for common ERP component windows (if in time range)
            if times[0] <= 0.1 <= times[-1]:
                ax.axvspan(0.08, 0.12, alpha=0.05, color='blue', label='P1')
            if times[0] <= 0.16 <= times[-1]:
                ax.axvspan(0.14, 0.18, alpha=0.05, color='green', label='N1')
            if times[0] <= 0.4 <= times[-1]:
                ax.axvspan(0.3, 0.5, alpha=0.05, color='orange', label='P3')

            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Amplitude (a.u.)', fontsize=8)
            ax.set_title(f'{roi_display_name}', fontweight='bold', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

            # Legend only for first subplot (avoid clutter)
            if roi_idx == 0:
                ax.legend(fontsize=6, loc='upper right', framealpha=0.8)

        # Hide unused subplots
        for idx in range(n_rois, len(axes)):
            axes[idx].axis('off')

        # Title indicates whether CI is shown
        if is_trial_level:
            title = f'Source-Level Event-Related Potentials (All {n_rois} ROIs with 95% CI)'
        else:
            title = f'Source-Level Event-Related Potentials (All {n_rois} ROIs)'

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        return fig

    def _plot_roi_correlation_matrix(self, parsed: Dict[str, Any]) -> Optional[plt.Figure]:
        """
        Plot ROI-to-ROI correlation matrix using sliding time windows.

        Computes temporal correlation between all pairs of ROIs across
        multiple time windows to show how connectivity evolves over time.
        """
        roi_names = parsed.get('roi_names', [])
        conditions = parsed.get('conditions', [])
        data = parsed.get('data', {})
        times_dict = parsed.get('times', {})

        if not roi_names or not data:
            return None

        n_rois = len(roi_names)

        # Get time axis
        first_cond = conditions[0]
        if first_cond not in times_dict:
            return None

        times = times_dict[first_cond]
        sfreq = 1.0 / np.median(np.diff(times)) if len(times) > 1 else 500.0

        # Define sliding windows (in seconds)
        window_size = 0.1  # 100 ms
        window_step = 0.05  # 50 ms step
        tmin, tmax = times[0], times[-1]

        # Generate window starts
        window_starts = np.arange(tmin, tmax - window_size + window_step, window_step)
        n_windows = len(window_starts)

        if n_windows < 3:
            # Fall back to single full-window correlation
            return self._plot_single_correlation_matrix(parsed)

        # Collect ROI timeseries for each condition
        roi_timeseries = []
        for roi_idx in range(n_rois):
            all_times = []
            for cond in conditions:
                if cond in data:
                    arr = data[cond]
                    # Handle both 2D (n_rois, n_times) and 3D (n_trials, n_rois, n_times)
                    if arr.ndim == 3:
                        # Average across trials
                        roi_data = arr[:, roi_idx, :].mean(axis=0)
                    else:
                        roi_data = arr[roi_idx, :]
                    all_times.append(roi_data)
            if all_times:
                roi_ts = np.concatenate(all_times)
                roi_timeseries.append(roi_ts)

        if not roi_timeseries:
            return None

        roi_timeseries = np.array(roi_timeseries)  # (n_rois, n_times_total)

        # Add laterality labels (L/R) to ROI names
        # ROIs are stored with duplicates representing left/right hemispheres
        display_names = []
        roi_counts = {}
        for name in roi_names:
            if name in roi_counts:
                # Second occurrence - add R suffix
                display_names.append(f"{name}.R")
                roi_counts[name] += 1
            else:
                # First occurrence - check if there will be a duplicate
                if list(roi_names).count(name) > 1:
                    display_names.append(f"{name}.L")
                else:
                    # Midline ROI - no laterality
                    display_names.append(name)
                roi_counts[name] = 1

        # Show ALL consecutive overlapping windows
        # Use 3 columns for larger matrices with readable labels
        n_cols = 3
        n_rows = (n_windows + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
        axes = axes.flatten() if n_windows > 1 else [axes]

        im = None  # Will hold last imshow for colorbar

        for win_idx in range(n_windows):
            win_start = window_starts[win_idx]
            win_end = win_start + window_size

            # Convert to sample indices
            start_sample = int((win_start - tmin) * sfreq)
            end_sample = int((win_end - tmin) * sfreq)
            start_sample = max(0, start_sample)
            end_sample = min(roi_timeseries.shape[1], end_sample)

            ax = axes[win_idx]

            if end_sample <= start_sample:
                ax.axis('off')
                continue

            # Extract window data
            window_data = roi_timeseries[:, start_sample:end_sample]

            # Compute correlation
            corr_matrix = np.corrcoef(window_data)

            # Plot
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

            # Show ALL tick labels on leftmost column and bottom row
            if win_idx % n_cols == 0:  # Leftmost column
                ax.set_yticks(range(n_rois))
                ax.set_yticklabels(display_names, fontsize=6)
            else:
                ax.set_yticks([])

            if win_idx >= n_windows - n_cols:  # Bottom row
                ax.set_xticks(range(n_rois))
                ax.set_xticklabels(display_names, rotation=90, fontsize=6, ha='center')
            else:
                ax.set_xticks([])

            ax.set_title(f'{round(win_start*1000)} to {round(win_end*1000)} ms', fontsize=10, fontweight='bold')

            # Grid
            ax.set_xticks(np.arange(n_rois) - 0.5, minor=True)
            ax.set_yticks(np.arange(n_rois) - 0.5, minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.2, alpha=0.2)

        # Hide unused subplots
        for idx in range(n_windows, len(axes)):
            axes[idx].axis('off')

        # Add single colorbar for all subplots
        if im is not None:
            fig.subplots_adjust(right=0.94)
            cbar_ax = fig.add_axes([0.95, 0.15, 0.015, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Correlation', rotation=270, labelpad=15, fontsize=10)

        fig.suptitle(
            f'ROI-to-ROI Correlation Matrix (Sliding Windows: {int(window_size*1000)}ms window, {int(window_step*1000)}ms step)',
            fontsize=14,
            fontweight='bold'
        )
        # Avoid tight_layout with a manually added colorbar axis.
        fig.subplots_adjust(left=0.08, right=0.94, top=0.95, bottom=0.05, hspace=0.35, wspace=0.15)

        return fig

    def _plot_single_correlation_matrix(self, parsed: Dict[str, Any]) -> Optional[plt.Figure]:
        """Plot single full-window correlation matrix (fallback)."""
        roi_names = parsed.get('roi_names', [])
        conditions = parsed.get('conditions', [])
        data = parsed.get('data', {})

        if not roi_names or not data:
            return None

        n_rois = len(roi_names)

        # Collect ROI timeseries
        roi_timeseries = []
        for roi_idx in range(n_rois):
            all_times = []
            for cond in conditions:
                if cond in data:
                    arr = data[cond]
                    # Handle both 2D (n_rois, n_times) and 3D (n_trials, n_rois, n_times)
                    if arr.ndim == 3:
                        # Average across trials
                        roi_data = arr[:, roi_idx, :].mean(axis=0)
                    else:
                        roi_data = arr[roi_idx, :]
                    all_times.append(roi_data)
            if all_times:
                roi_ts = np.concatenate(all_times)
                roi_timeseries.append(roi_ts)

        if not roi_timeseries:
            return None

        roi_timeseries = np.array(roi_timeseries)
        corr_matrix = np.corrcoef(roi_timeseries)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Correlation', rotation=270, labelpad=20)

        # Shorten ROI names
        short_names = []
        for name in roi_names:
            if '.' in name:
                network, roi = name.split('.', 1)
                network_abbr = ''.join([c for c in network if c.isupper()])
                short_names.append(f"{network_abbr}.{roi}")
            else:
                short_names.append(name)

        ax.set_xticks(range(n_rois))
        ax.set_yticks(range(n_rois))
        ax.set_xticklabels(short_names, rotation=90, fontsize=8, ha='center')
        ax.set_yticklabels(short_names, fontsize=8)

        # Grid
        ax.set_xticks(np.arange(n_rois) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_rois) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

        ax.set_title('ROI-to-ROI Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('ROI', fontsize=12)
        ax.set_ylabel('ROI', fontsize=12)

        plt.tight_layout()
        return fig

    def _compute_crosstalk_matrix(
        self,
        fwd: mne.Forward,
        inv_op: mne.minimum_norm.InverseOperator,
        labels: List[mne.Label],
        method: str = 'dSPM'
    ) -> np.ndarray:
        """
        Compute source crosstalk (point spread function) matrix.

        Simulates a point source at each ROI, applies forward-inverse round trip,
        and measures how much signal leaks into other ROIs.

        This quantifies the spatial resolution limitations of the inverse solution
        and identifies ROI pairs that will show spurious correlation due to
        volume conduction/spatial leakage.

        Parameters
        ----------
        fwd : mne.Forward
            Forward solution (head model + sensor positions)
        inv_op : mne.minimum_norm.InverseOperator
            Inverse operator (dSPM, sLORETA, etc.)
        labels : list of mne.Label
            ROI parcellation (one Label per ROI)
        method : str
            Inverse method ('dSPM', 'sLORETA', 'eLORETA')

        Returns
        -------
        crosstalk : ndarray (n_rois, n_rois)
            Crosstalk matrix where C[i,j] = correlation between:
            - Simulated source at ROI i
            - Reconstructed signal at ROI j
            Diagonal = 1.0 (perfect reconstruction of own ROI)
            Off-diagonal = spatial leakage (should be low, e.g., < 0.3)

        Notes
        -----
        High crosstalk (e.g., > 0.5) indicates ROI pairs whose connectivity
        estimates are unreliable due to spatial leakage.

        This computation is expensive (O(n_rois)) but only needs to be done
        once per source reconstruction configuration. The result should be
        saved and reused for all downstream connectivity/feature analyses.

        References
        ----------
        Hauk, O., & Stenroos, M. (2014). A framework for the design of flexible
        cross-talk functions for spatial filtering of EEG/MEG data.
        Human brain mapping, 35(4), 1792-1802.
        """
        n_rois = len(labels)
        n_times = 100  # Short simulation (crosstalk is time-independent)
        sfreq = inv_op['info']['sfreq']

        # Initialize crosstalk matrix
        crosstalk = np.zeros((n_rois, n_rois))

        # Extract label centers (vertices with peak in each ROI)
        label_centers = []
        for label in labels:
            # Use the vertex closest to the label's center of mass
            if len(label.vertices) > 0:
                center_vertex = label.vertices[len(label.vertices) // 2]
                label_centers.append((label.hemi, center_vertex))
            else:
                label_centers.append(None)

        # For each ROI, simulate point source and measure leakage
        for i, label_i in enumerate(labels):
            if label_centers[i] is None:
                continue

            hemi, vertex = label_centers[i]

            # Create source estimate with delta function at ROI i
            # (1 µA·m at t=0, zero elsewhere)
            stc_simulated = mne.SourceEstimate(
                data=np.zeros((len(fwd['src'][0]['vertno']) + len(fwd['src'][1]['vertno']), n_times)),
                vertices=[fwd['src'][0]['vertno'], fwd['src'][1]['vertno']],
                tmin=0,
                tstep=1.0 / sfreq
            )

            # Set delta at vertex
            if hemi == 'lh':
                vertex_idx = np.where(fwd['src'][0]['vertno'] == vertex)[0]
                if len(vertex_idx) > 0:
                    stc_simulated.data[vertex_idx[0], 0] = 1e-9  # 1 nA·m
            else:
                vertex_idx = np.where(fwd['src'][1]['vertno'] == vertex)[0]
                if len(vertex_idx) > 0:
                    lh_nverts = len(fwd['src'][0]['vertno'])
                    stc_simulated.data[lh_nverts + vertex_idx[0], 0] = 1e-9

            # Forward model: source -> sensors
            try:
                evoked_simulated = mne.apply_forward(fwd, stc_simulated, inv_op['info'])

                # Inverse solution: sensors -> reconstructed sources
                stc_reconstructed = mne.minimum_norm.apply_inverse(
                    evoked_simulated,
                    inv_op,
                    lambda2=1.0 / 9.0,  # Standard SNR=3
                    method=method,
                    verbose=False
                )

                # Extract ROI time courses from reconstructed source
                roi_timeseries_reconstructed = mne.extract_label_time_course(
                    stc_reconstructed,
                    labels,
                    fwd['src'],
                    mode='mean',
                    verbose=False
                )  # (n_rois, n_times)

                # Measure correlation between simulated (delta at i) and reconstructed (all ROIs)
                # Simulated signal at ROI i
                simulated_roi_i = np.zeros(n_times)
                simulated_roi_i[0] = 1.0  # Delta function

                # Correlation with each reconstructed ROI
                for j in range(n_rois):
                    reconstructed_roi_j = roi_timeseries_reconstructed[j, :]

                    # Correlation (handles zero-variance case)
                    if np.std(reconstructed_roi_j) > 1e-10:
                        corr = np.corrcoef(simulated_roi_i, reconstructed_roi_j)[0, 1]
                        crosstalk[i, j] = np.abs(corr)  # Use absolute value
                    else:
                        crosstalk[i, j] = 0.0

            except Exception as e:
                print(f"Warning: Crosstalk computation failed for ROI {i} ({labels[i].name}): {e}")
                # Leave as zeros for this ROI
                continue

        # Ensure diagonal is 1.0 (ROI reconstructs itself perfectly)
        np.fill_diagonal(crosstalk, 1.0)

        return crosstalk

    def _plot_crosstalk_matrix(
        self,
        crosstalk: np.ndarray,
        roi_names: List[str],
        threshold: float = 0.3
    ) -> plt.Figure:
        """
        Plot source crosstalk (point spread function) matrix.

        Parameters
        ----------
        crosstalk : ndarray (n_rois, n_rois)
            Crosstalk matrix from _compute_crosstalk_matrix()
        roi_names : list of str
            ROI names for axis labels
        threshold : float
            Threshold for highlighting problematic ROI pairs (default: 0.3)
            Pairs with crosstalk > threshold shown in red

        Returns
        -------
        fig : matplotlib.Figure
            Figure with crosstalk heatmap and statistics

        Notes
        -----
        Visualization includes:
        - Full crosstalk matrix (heatmap)
        - Red boxes around high-crosstalk ROI pairs (> threshold)
        - Statistics: % of ROI pairs with high crosstalk
        - Interpretation guide
        """
        n_rois = len(roi_names)

        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1],
                              hspace=0.4, wspace=0.3)

        # Main crosstalk matrix (large subplot)
        ax_main = fig.add_subplot(gs[0, :])

        # Plot heatmap
        im = ax_main.imshow(crosstalk, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.02)
        cbar.set_label('Crosstalk (Absolute Correlation)', rotation=270, labelpad=25, fontsize=12)

        # Highlight high-crosstalk pairs with red boxes
        high_crosstalk = np.where(crosstalk > threshold)
        for i, j in zip(high_crosstalk[0], high_crosstalk[1]):
            if i != j:  # Skip diagonal
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor='red', linewidth=2)
                ax_main.add_patch(rect)

        # Shorten ROI names for display
        short_names = []
        for name in roi_names:
            if '.' in name:
                network, roi = name.split('.', 1)
                network_abbr = ''.join([c for c in network if c.isupper()])
                short_names.append(f"{network_abbr}.{roi}")
            else:
                short_names.append(name[:15])  # Truncate long names

        ax_main.set_xticks(range(n_rois))
        ax_main.set_yticks(range(n_rois))
        ax_main.set_xticklabels(short_names, rotation=90, fontsize=8, ha='center')
        ax_main.set_yticklabels(short_names, fontsize=8)

        ax_main.set_title('Source Crosstalk Matrix (Point Spread Function)',
                         fontsize=14, fontweight='bold', pad=15)
        ax_main.set_xlabel('Target ROI', fontsize=12)
        ax_main.set_ylabel('Source ROI', fontsize=12)

        # Add grid for readability
        ax_main.set_xticks(np.arange(n_rois) - 0.5, minor=True)
        ax_main.set_yticks(np.arange(n_rois) - 0.5, minor=True)
        ax_main.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.2)

        # Bottom left: Histogram of off-diagonal crosstalk values
        ax_hist = fig.add_subplot(gs[1, 0])

        # Get off-diagonal values
        mask = ~np.eye(n_rois, dtype=bool)
        off_diag_values = crosstalk[mask]

        ax_hist.hist(off_diag_values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax_hist.axvline(threshold, color='red', linestyle='--', linewidth=2,
                       label=f'Threshold: {threshold}')
        ax_hist.axvline(np.median(off_diag_values), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(off_diag_values):.3f}')

        ax_hist.set_xlabel('Crosstalk Value', fontsize=11)
        ax_hist.set_ylabel('Count (ROI Pairs)', fontsize=11)
        ax_hist.set_title('Distribution of Off-Diagonal Crosstalk', fontsize=12, fontweight='bold')
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, alpha=0.3, axis='y')

        # Bottom right: Statistics and interpretation guide
        ax_text = fig.add_subplot(gs[1, 1])
        ax_text.axis('off')

        # Compute statistics
        n_pairs = n_rois * (n_rois - 1) // 2  # Unique pairs (upper triangle)
        n_high_crosstalk = np.sum(np.triu(crosstalk, k=1) > threshold)
        pct_high_crosstalk = 100 * n_high_crosstalk / n_pairs if n_pairs > 0 else 0

        mean_crosstalk = np.mean(off_diag_values)
        median_crosstalk = np.median(off_diag_values)
        max_crosstalk = np.max(off_diag_values)

        stats_text = f"""
CROSSTALK STATISTICS

Total ROI pairs: {n_pairs}
High crosstalk (>{threshold}): {n_high_crosstalk} ({pct_high_crosstalk:.1f}%)

Mean crosstalk: {mean_crosstalk:.3f}
Median crosstalk: {median_crosstalk:.3f}
Max crosstalk: {max_crosstalk:.3f}

INTERPRETATION GUIDE

Crosstalk < 0.3: Low leakage
  → Connectivity trustworthy

Crosstalk 0.3-0.5: Moderate leakage
  → Use with caution

Crosstalk > 0.5: High leakage
  → Connectivity unreliable
  → Consider leakage correction

RED BOXES indicate problematic
ROI pairs requiring correction.
        """

        ax_text.text(0.05, 0.95, stats_text, transform=ax_text.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        return fig

    def generate_html_report(
        self,
        result: QCResult,
        output_path: Path,
        **kwargs
    ) -> Path:
        """
        Generate HTML QC report for source reconstruction.

        Parameters
        ----------
        result : QCResult
            QC result with metrics and figures
        output_path : Path
            Path to save HTML report
        **kwargs : dict
            Additional arguments (session_id, variant, etc.)

        Returns
        -------
        Path
            Path to saved HTML file
        """
        builder = HTMLReportBuilder(title=f"Source QC: {result.subject_id}")

        # Header section
        method = result.metadata.get('method', 'unknown')
        parcellation = result.metadata.get('parcellation', 'unknown')
        n_conditions = result.metadata.get('n_conditions', 0)

        builder.add_header("Overview", level=2)
        builder.add_text(f"Subject: {result.subject_id}")
        builder.add_text(f"Method: {method}")
        builder.add_text(f"Parcellation: {parcellation}")
        builder.add_text(f"Conditions: {n_conditions}")
        conditions = result.metadata.get('conditions', [])
        if conditions:
            builder.add_text(f"Condition names: {', '.join(map(str, conditions))}")

        # Metrics section
        if result.metrics:
            builder.add_header("Quality Metrics", level=2)
            builder.add_metrics_table(result.metrics)

        # Figures (base64 encoded strings)
        if 'source_power' in result.figures:
            builder.add_header("Source Power Distribution", level=2)
            builder.add_raw_html(f'<div class="figure-container"><img src="data:image/png;base64,{result.figures["source_power"]}" alt="Source Power"><div class="figure-caption">Source amplitudes across all source points</div></div>')

        if 'roi_coverage' in result.figures:
            builder.add_header("ROI Time Series Present", level=2)
            builder.add_raw_html(f'<div class="figure-container"><img src="data:image/png;base64,{result.figures["roi_coverage"]}" alt="ROI Time Series Present"><div class="figure-caption">ROIs with finite, non-zero time-series samples in at least one condition. This confirms ROI time-course presence, not anatomical coverage quality.</div></div>')

        if 'roi_timecourses' in result.figures:
            builder.add_header("ROI Time Courses", level=2)
            builder.add_raw_html(f'<div class="figure-container"><img src="data:image/png;base64,{result.figures["roi_timecourses"]}" alt="ROI Time Courses"><div class="figure-caption">Time courses for top ROIs by mean absolute signal.</div></div>')

        if 'network_summary' in result.figures:
            builder.add_header("Network Summary", level=2)
            builder.add_raw_html(f'<div class="figure-container"><img src="data:image/png;base64,{result.figures["network_summary"]}" alt="Network Summary"><div class="figure-caption">Average absolute ROI signal per CONN network.</div></div>')

        if 'source_erp' in result.figures:
            builder.add_header("Source-Level Event-Related Potentials", level=2)
            builder.add_raw_html(f'<div class="figure-container"><img src="data:image/png;base64,{result.figures["source_erp"]}" alt="Source ERP"><div class="figure-caption">Event-locked waveforms for all ROIs with left/right hemisphere labels. Shaded bands show 95% confidence intervals (when trial-level data available). P1, N1, P3 component windows highlighted.</div></div>')

        if 'roi_correlation' in result.figures:
            builder.add_header("ROI Correlation Matrix (Sliding Windows)", level=2)
            builder.add_raw_html(f'<div class="figure-container"><img src="data:image/png;base64,{result.figures["roi_correlation"]}" alt="ROI Correlation"><div class="figure-caption">ROI-to-ROI correlation in 100ms sliding windows (50ms steps) showing temporal evolution of functional connectivity</div></div>')

        if 'crosstalk_matrix' in result.figures:
            builder.add_header("Source Crosstalk Matrix (Point Spread Function)", level=2)

            # Add explanation
            builder.add_raw_html('''
                <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                    <p style="margin: 5px 0;"><strong>What is Source Crosstalk?</strong></p>
                    <p style="margin: 5px 0; font-size: 0.9em;">
                        Source crosstalk measures the spatial resolution limitations of the inverse solution.
                        It quantifies how much signal from one ROI "leaks" into other ROIs due to volume conduction
                        and the ill-posed nature of the inverse problem.
                    </p>
                    <p style="margin: 5px 0; font-size: 0.9em;">
                        <strong>Why it matters:</strong> High crosstalk between two ROIs means their connectivity
                        estimates may be inflated by methodological artifact rather than true neural communication.
                        This is computed once per inverse method + parcellation configuration and reused for all
                        downstream connectivity analyses.
                    </p>
                </div>
            ''')

            builder.add_raw_html(f'<div class="figure-container"><img src="data:image/png;base64,{result.figures["crosstalk_matrix"]}" alt="Crosstalk Matrix"><div class="figure-caption">Point spread function showing spatial leakage between ROI pairs. Red boxes indicate high-crosstalk pairs (> 0.3) requiring leakage correction for reliable connectivity estimation.</div></div>')

            # Add crosstalk statistics if available
            if 'crosstalk_mean' in result.metadata:
                mean_ct = result.metadata['crosstalk_mean']
                n_high = result.metadata.get('crosstalk_high_pairs', 0)
                builder.add_raw_html(f'''
                    <div style="background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin-top: 10px;">
                        <p style="margin: 5px 0; font-size: 0.9em;">
                            <strong>Crosstalk Statistics:</strong><br>
                            Mean off-diagonal crosstalk: {mean_ct:.3f}<br>
                            High-crosstalk pairs (&gt; 0.3): {n_high}
                        </p>
                    </div>
                ''')

        # Save
        html = builder.build()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html)

        return output_path
