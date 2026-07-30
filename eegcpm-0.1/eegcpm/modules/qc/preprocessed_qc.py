"""
Preprocessed EEG data quality control with diagnostic plots.

Provides PreprocessedQC class for evaluating quality of preprocessed data,
focusing on ICA effectiveness, filtering results, and cleaned signal quality.

This module also supports Event-Related Potential (ERP) quality control by
generating condition-based ERP waveforms and topographic maps using MNE's
built-in plotting functions. ERP QC is critical for validating that
preprocessing parameters are appropriate before proceeding to connectivity
analysis.

Author: EEGCPM Development Team
Created: 2025-01
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml

import matplotlib.pyplot as plt
import mne
import numpy as np

from .base import BaseQC, QCMetric, QCResult
from .erp_qc import ERPQC
from ..preprocessing.channel_clustering import (
    compute_bad_channel_clustering,
    visualize_channel_clustering,
    get_clustering_recommendation
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_NEUROSCAN_CH_TO_MNE = {
    "FP1": "Fp1", "FP2": "Fp2", "FPZ": "Fpz",
    "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "OZ": "Oz",
    "FCZ": "FCz", "CPZ": "CPz", "POZ": "POz",
    "CB1": "Cb1", "CB2": "Cb2",
}
NEUROSCAN_LOCS_CACHE: Optional[Dict[str, np.ndarray]] = None


def _load_neuroscan_locs() -> Dict[str, np.ndarray]:
    global NEUROSCAN_LOCS_CACHE
    if NEUROSCAN_LOCS_CACHE is not None:
        return NEUROSCAN_LOCS_CACHE
    locs_file = DATA_DIR / "neuroscan_64ch.locs"
    result = {}
    if locs_file.exists():
        with open(locs_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                name = parts[3]
                theta_deg = float(parts[1])
                radius = float(parts[2])
                theta_rad = np.radians(theta_deg)
                x = radius * np.sin(theta_rad)
                y = radius * np.cos(theta_rad)
                mapped = _NEUROSCAN_CH_TO_MNE.get(name, name)
                result[mapped] = np.array([x, y])
    NEUROSCAN_LOCS_CACHE = result
    return result


def _draw_head_outline(ax, *, head_radius=0.5, linewidth=1.5):
    """Draw EEGLAB-style head outline (circle + nose + ears) centered at origin.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    head_radius : float
        Radius of the head circle (0.5 for Neuroscan .locs convention).
    linewidth : float
        Line width for all outline elements.
    """
    from matplotlib.patches import Circle
    r = head_radius
    ax.add_patch(Circle((0, 0), r, fill=False, linewidth=linewidth, color='k'))
    nh = r * 0.18
    nw = r * 0.08
    ax.plot([-nw, nw, 0, -nw], [r, r, r + nh, r], 'k-', lw=linewidth)
    ear_ext = r * 0.07
    ear_y = r * 0.15
    ear_t = np.linspace(0, np.pi, 20)
    ax.plot(-r - ear_ext * np.sin(ear_t), -ear_y * np.cos(ear_t), 'k-', lw=linewidth)
    ax.plot(r + ear_ext * np.sin(ear_t), -ear_y * np.cos(ear_t), 'k-', lw=linewidth)


def _get_ica_reject_reason(idx: int, excluded_set: set, eog_set: set, ecg_set: set,
                           auto_detected: dict, manual_set: Optional[set] = None) -> str:
    """Determine ICA rejection reason from step metadata."""
    if idx not in excluded_set:
        return '-'
    if idx in eog_set:
        return 'ICA auto-detected EOG'
    if idx in ecg_set:
        return 'ICA auto-detected ECG'
    if manual_set and idx in manual_set:
        return 'Manually excluded'
    return 'ICA excluded'


class PreprocessedQC(BaseQC):
    """
    Quality control for preprocessed EEG data.

    Evaluates preprocessing effectiveness including:
    1. PSD before/after comparison (if raw available)
    2. ICA component overview
    3. Cleaned signal amplitude distribution
    4. Channel correlation after preprocessing
    5. Bad segment summary

    Attributes:
        dpi: DPI for PNG rendering of figures
    """

    def __init__(
        self,
        output_dir: Path,
        config: Optional[Dict] = None,
        dpi: int = 100,
    ):
        """
        Initialize PreprocessedQC module.

        Args:
            output_dir: Directory for output files
            config: Configuration dictionary
            dpi: DPI for figure rendering
        """
        super().__init__(output_dir, config)
        self.dpi = dpi

    def compute(
        self,
        data: mne.io.BaseRaw,
        subject_id: str,
        ica: Optional[mne.preprocessing.ICA] = None,
        raw_before: Optional[mne.io.BaseRaw] = None,
        events: Optional[np.ndarray] = None,
        event_id: Optional[Dict[str, int]] = None,
        epochs_params: Optional[Dict[str, Any]] = None,
        erp_channels: Optional[List[str]] = None,
        removed_channels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> QCResult:
        """
        Compute QC metrics and generate diagnostic plots for preprocessed data.

        Generates comprehensive QC including signal quality metrics and optional
        event-related potential (ERP) analysis for task-based paradigms.

        Args:
            data: Preprocessed MNE Raw object
            subject_id: Subject identifier
            ica: Optional ICA object used in preprocessing
            raw_before: Optional raw data before preprocessing (for comparison)
            events: Optional event array (N_events x 3) for ERP generation
            event_id: Optional dictionary mapping condition names to event codes
                Example: {'left': 8, 'right': 9} for left/right stimulus
            epochs_params: Optional parameters for epoching (tmin, tmax, baseline, etc.)
                Default: {'tmin': -0.2, 'tmax': 0.8, 'baseline': (-0.2, 0)}
            erp_channels: Optional list of channels for ERP analysis (e.g., visual ROI)
                If None, uses all EEG channels
            removed_channels: Dict mapping channel names to removal reasons
                Example: {'E1': 'flatline', 'E2': 'bridged', 'E3': 'high_variance'}
            metadata: Optional dict with ICLabel and preprocessing information
                Expected structure: {'iclabel': {'components': [{'index': int, 'label': str,
                'probability': float, 'rejected': bool}, ...]}, ...}
            **kwargs: Additional keyword arguments (session_id, run_id, etc.)

        Returns:
            QCResult with metrics, embedded figures, optional ICA components table, and ERP report path

        Notes:
            - ERP analysis is only performed if both events and event_id are provided
            - ERP reports are saved as separate HTML files in the output directory
            - ERP QC helps validate preprocessing quality before connectivity analysis
            - ICLabel information is extracted from metadata if available
        """
        raw = data
        result = QCResult(subject_id=subject_id)

        # Use provided removed_channels or empty dict
        channel_reasons = removed_channels if removed_channels else {}

        # Safety net: catch channels dropped/marked by any step not in removed_channels
        safety_net_channels = {}
        if raw_before is not None:
            before_eeg = set(mne.pick_types(raw_before.info, eeg=True, exclude=[]))
            before_eeg = {raw_before.ch_names[i] for i in before_eeg}
            after_chs = set(raw.ch_names)
            dropped = before_eeg - after_chs
            for ch in dropped:
                if ch not in channel_reasons:
                    channel_reasons[ch] = 'dropped'
                    safety_net_channels[ch] = 'dropped'
            for ch in raw.info["bads"]:
                if ch not in channel_reasons:
                    channel_reasons[ch] = 'marked_bad'
                    safety_net_channels[ch] = 'marked_bad'
        if safety_net_channels:
            ch_list = list(safety_net_channels.keys())
            ch_str = ', '.join(ch_list[:10])
            if len(ch_list) > 10:
                ch_str += f'... ({len(ch_list)} total)'
            result.add_note(f"Bad/removed channel(s): {ch_str}")

        # Extract preprocessing metadata if provided
        preprocessing_metadata = kwargs.get('preprocessing_metadata', {})

        # Extract and store ICLabel information from metadata
        ica_components_info = None
        if metadata and 'iclabel' in metadata:
            iclabel_data = metadata.get('iclabel', {})
            ica_components_info = iclabel_data.get('components', None)
            if ica_components_info:
                n_classified = len(ica_components_info)
                n_rejected = sum(1 for c in ica_components_info if c.get('rejected', False))
                result.add_note(f"ICA classification: ICLabel ({n_classified} components, {n_rejected} rejected)")

        # Fallback: build component info from ICA step when ICLabel unavailable
        ica_step_meta = metadata.get('ica', {}) if metadata else {}
        if ica_components_info is None and ica is not None and ica_step_meta:
            auto_detected = ica_step_meta.get('auto_detected', {})
            excluded = ica_step_meta.get('excluded_indices', [])
            excluded_set = set(excluded) if excluded else set()
            eog_indices = set(auto_detected.get('eog', []) or [])
            ecg_indices = set(auto_detected.get('ecg', []) or [])

            # Compute variance explained from ICA object
            variance_ratios = None
            if hasattr(ica, 'pca_explained_variance_') and ica.pca_explained_variance_ is not None:
                total_variance = ica.pca_explained_variance_.sum()
                if total_variance > 0:
                    variance_ratios = (ica.pca_explained_variance_ / total_variance * 100)

            fallback_components = []
            for i in range(ica.n_components_):
                label = 'other'
                if i in eog_indices:
                    label = 'eye'
                elif i in ecg_indices:
                    label = 'heart'
                variance_explained = None
                if variance_ratios is not None and i < len(variance_ratios):
                    variance_explained = float(variance_ratios[i])
                fallback_components.append({
                    'index': i,
                    'label': label,
                    'probability': 1.0 if label != 'other' else 0.0,
                    'variance_explained': variance_explained,
                    'rejected': i in excluded_set,
                    'reject_reason': _get_ica_reject_reason(i, excluded_set, eog_indices, ecg_indices, auto_detected),
                })
            ica_components_info = fallback_components
            if fallback_components:
                n_labeled = sum(1 for c in fallback_components if c['label'] != 'other')
                result.add_note(f"ICA classification: Using EOG/ECG auto-detection ({n_labeled} labeled, ICLabel not available)")

        # Merge ICA exclusions into component info (even when ICLabel provided data)
        n_ica_merged = 0
        if ica_components_info is not None and ica_step_meta:
            excluded = ica_step_meta.get('excluded_indices', [])
            auto_detected = ica_step_meta.get('auto_detected', {})
            excluded_set = set(excluded) if excluded else set()
            eog_set = set(auto_detected.get('eog', []) or [])
            ecg_set = set(auto_detected.get('ecg', []) or [])
            manual_excluded = ica_step_meta.get('manual_excluded', [])
            manual_set = set(manual_excluded) if manual_excluded else set()
            for comp in ica_components_info:
                idx = comp.get('index')
                if idx in excluded_set and not comp.get('rejected', False):
                    comp['rejected'] = True
                    comp['reject_reason'] = _get_ica_reject_reason(
                        idx, excluded_set, eog_set, ecg_set, auto_detected, manual_set
                    )
                    n_ica_merged += 1
            if n_ica_merged > 0:
                result.add_note(f"ICA: {n_ica_merged} additional component(s) excluded by ICA but not flagged by ICLabel — marked as rejected in table")

        # Add preprocessing summary notes
        if preprocessing_metadata:
            bad_ch_meta = preprocessing_metadata.get('bad_channels', {})
            if bad_ch_meta:
                n_bad_detected = bad_ch_meta.get('n_bad', 0)
                if n_bad_detected > 0:
                    bad_list = bad_ch_meta.get('detected', [])
                    bad_str = ', '.join(bad_list[:10])  # Show first 10
                    if len(bad_list) > 10:
                        bad_str += f'... ({len(bad_list)} total)'
                    result.add_note(f"Preprocessing: {n_bad_detected} bad channel(s) interpolated: {bad_str}")

            artifact_meta = preprocessing_metadata.get('artifacts', {})
            if artifact_meta:
                n_artifacts = artifact_meta.get('n_annotations_added', 0)
                if n_artifacts > 0:
                    result.add_note(f"Preprocessing: {n_artifacts} artifact segment(s) annotated")

            filter_meta = preprocessing_metadata.get('filter', {})
            if filter_meta:
                l_freq = filter_meta.get('l_freq')
                h_freq = filter_meta.get('h_freq')
                result.add_note(f"Preprocessing: Filtered {l_freq}-{h_freq} Hz")

            zapline_meta = preprocessing_metadata.get('zapline', {})
            if zapline_meta and zapline_meta.get('applied', False):
                detected_freq = zapline_meta.get('detected_freq')
                n_components = zapline_meta.get('n_components_removed', 0)
                line_reduction_db = zapline_meta.get('line_reduction_db', 0)

                if detected_freq is not None:
                    result.add_note(
                        f"Zapline: Detected {detected_freq:.1f} Hz, removed {n_components} component(s), "
                        f"{line_reduction_db:.1f} dB reduction"
                    )

        # 1. Basic metrics
        result.add_metric(QCMetric("Duration", raw.times[-1], "s"))
        result.add_metric(QCMetric("Sampling Rate", raw.info["sfreq"], "Hz"))
        result.add_metric(QCMetric("N Channels", len(raw.ch_names), ""))

        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        n_eeg = len(eeg_picks)
        result.add_metric(QCMetric("N EEG Channels", n_eeg, ""))

        # Bad channels - count dropped AND marked bad
        n_bad_marked = len(raw.info["bads"])

        # If raw_before is available, calculate n_dropped
        n_dropped = 0
        if raw_before is not None:
            n_channels_before = len(mne.pick_types(raw_before.info, eeg=True, exclude=[]))
            n_channels_after = len(mne.pick_types(raw.info, eeg=True, exclude=[]))
            n_dropped = n_channels_before - n_channels_after

        n_bad_total = n_bad_marked + n_dropped
        bad_pct = 100 * n_bad_total / n_eeg if n_eeg > 0 else 0
        status = "ok" if bad_pct < 10 else ("warning" if bad_pct < 20 else "bad")

        # Unit field should only contain the description, not the value
        bad_unit = f"({n_dropped} dropped)" if n_dropped > 0 else ""
        result.add_metric(QCMetric("Bad Channels", n_bad_total, bad_unit, status))

        # 2. Amplitude statistics (post-preprocessing should be cleaner)
        # Filter out dead channels for amplitude statistics
        eeg_picks_good = mne.pick_types(raw.info, eeg=True, exclude="bads")
        valid_picks = self._filter_dead_channels(raw, eeg_picks_good)

        n_dead = len(eeg_picks_good) - len(valid_picks)
        if n_dead > 0:
            result.add_note(f"{n_dead} dead channel(s) excluded from statistics")

        if len(valid_picks) > 0:
            # Get data EXCLUDING bad segments (reject_by_annotation equivalent)
            # Build mask of good samples
            good_mask = self._get_good_samples_mask(raw)
            data_eeg = raw.get_data(picks=valid_picks)

            # Apply mask - only use good (non-bad) samples
            if good_mask.sum() > 0:
                data_good = data_eeg[:, good_mask]
                # Also filter out any NaN/Inf values
                data_good = data_good[np.isfinite(data_good)]
                if len(data_good) > 0:
                    mean_amp = np.mean(np.abs(data_good)) * 1e6
                    max_amp = np.max(np.abs(data_good)) * 1e6
                    std_amp = np.std(data_good) * 1e6

                    # Post-preprocessing thresholds are tighter
                    status = "ok" if max_amp < 150 else ("warning" if max_amp < 300 else "bad")
                    result.add_metric(QCMetric("Mean Amplitude", mean_amp, "µV"))
                    result.add_metric(QCMetric("Max Amplitude", max_amp, "µV", status))
                    result.add_metric(QCMetric("Std Amplitude", std_amp, "µV"))
                else:
                    result.add_note("No valid amplitude data after filtering NaN/Inf values")
            else:
                result.add_note("All samples marked as bad - no valid data for amplitude statistics")
        else:
            result.add_note("No valid EEG channels for amplitude statistics")

        # 2b. Before/After amplitude metrics comparison
        before_after_metrics = {}
        if raw_before is not None:
            try:
                # Get valid picks for raw_before
                eeg_picks_before = mne.pick_types(raw_before.info, eeg=True, exclude="bads")
                valid_picks_before = self._filter_dead_channels(raw_before, eeg_picks_before)

                if len(valid_picks_before) > 0:
                    # Get data EXCLUDING bad segments
                    good_mask_before = self._get_good_samples_mask(raw_before)
                    data_eeg_before = raw_before.get_data(picks=valid_picks_before)

                    if good_mask_before.sum() > 0:
                        data_good_before = data_eeg_before[:, good_mask_before]
                        data_good_before = data_good_before[np.isfinite(data_good_before)]
                        if len(data_good_before) > 0:
                            mean_amp_before = np.mean(np.abs(data_good_before)) * 1e6
                            std_amp_before = np.std(data_good_before) * 1e6
                            min_amp_before = np.min(np.abs(data_good_before)) * 1e6
                            max_amp_before = np.max(np.abs(data_good_before)) * 1e6
                            before_after_metrics['before'] = {
                                'mean': mean_amp_before,
                                'std': std_amp_before,
                                'min': min_amp_before,
                                'max': max_amp_before,
                            }
            except Exception as e:
                result.add_note(f"Before/after comparison failed: {str(e)}")

            # Get metrics for after preprocessing (already computed above as valid_picks data)
            if len(valid_picks) > 0:
                good_mask_after = self._get_good_samples_mask(raw)
                data_eeg_after = raw.get_data(picks=valid_picks)

                if good_mask_after.sum() > 0:
                    data_good_after = data_eeg_after[:, good_mask_after]
                    data_good_after = data_good_after[np.isfinite(data_good_after)]
                    if len(data_good_after) > 0:
                        mean_amp_after = np.mean(np.abs(data_good_after)) * 1e6
                        std_amp_after = np.std(data_good_after) * 1e6
                        min_amp_after = np.min(np.abs(data_good_after)) * 1e6
                        max_amp_after = np.max(np.abs(data_good_after)) * 1e6
                        before_after_metrics['after'] = {
                            'mean': mean_amp_after,
                            'std': std_amp_after,
                            'min': min_amp_after,
                            'max': max_amp_after,
                        }

            # Store metrics in result metadata for HTML report
            if before_after_metrics:
                result.metadata['before_after_metrics'] = before_after_metrics

        # 3. Zapline metrics (if applied)
        if preprocessing_metadata and 'zapline' in preprocessing_metadata:
            zapline_meta = preprocessing_metadata['zapline']
            if zapline_meta.get('applied', False):
                detected_freq = zapline_meta.get('detected_freq')
                line_reduction_db = zapline_meta.get('line_reduction_db', 0)
                n_components = zapline_meta.get('n_components_removed', 0)

                # Status: good if >10 dB, warning if 5-10 dB, bad if <5 dB
                if line_reduction_db >= 10:
                    status = "ok"
                elif line_reduction_db >= 5:
                    status = "warning"
                else:
                    status = "bad"

                result.add_metric(QCMetric(
                    "Zapline Line Reduction",
                    line_reduction_db,
                    "dB",
                    status
                ))

                if detected_freq is not None:
                    result.add_metric(QCMetric("Zapline Detected Freq", detected_freq, "Hz"))

                result.add_metric(QCMetric("Zapline Components", n_components, ""))

                if line_reduction_db < 5:
                    result.add_note(f"⚠️ Zapline: Low line noise reduction ({line_reduction_db:.1f} dB). Consider increasing aggressiveness.")

        # 4. ICA metrics
        if ica is not None:
            result.add_metric(QCMetric("ICA Components", ica.n_components_, ""))
            result.add_metric(QCMetric("ICA Excluded", len(ica.exclude), ""))

            if len(ica.exclude) > ica.n_components_ * 0.5:
                result.add_note(f"Warning: {len(ica.exclude)}/{ica.n_components_} ICA components excluded")
                status = "warning"
            else:
                status = "ok"
            result.add_metric(QCMetric("ICA Excluded %", 100 * len(ica.exclude) / ica.n_components_, "%", status))

            # Store ICLabel information if available
            if ica_components_info:
                result.metadata['ica_components'] = ica_components_info

        # 4. Bad annotations/segments
        # Use sample mask to correctly handle overlapping annotations
        good_mask = self._get_good_samples_mask(raw)
        n_total_samples = len(good_mask)
        n_good_samples = good_mask.sum()
        n_bad_samples = n_total_samples - n_good_samples
        bad_pct = 100 * n_bad_samples / n_total_samples if n_total_samples > 0 else 0

        # Count number of bad annotation segments
        n_bad_annots = 0
        if raw.annotations:
            n_bad_annots = len([a for a in raw.annotations if a["description"].startswith("BAD")])

        status = "ok" if bad_pct < 20 else ("warning" if bad_pct < 40 else "bad")
        result.add_metric(QCMetric("Bad Segments", n_bad_annots, ""))
        result.add_metric(QCMetric("Bad Segment %", bad_pct, "%", status))

        # Add note about good data duration
        good_duration_s = n_good_samples / raw.info['sfreq']
        result.add_note(f"Good data: {good_duration_s:.1f}s ({100-bad_pct:.1f}% of recording)")

        # Bad channel topography map
        try:
            raw_for_topo = raw_before if raw_before is not None and raw_before.get_montage() else raw
            if channel_reasons and raw_for_topo.get_montage():
                fig_topo_bad = self._plot_bad_channels_topomap(raw_for_topo, channel_reasons)
                result.add_figure("bad_channels_topo", self.fig_to_base64(fig_topo_bad, self.dpi))
                plt.close(fig_topo_bad)
        except Exception as e:
            result.add_note(f"Bad channel topography failed: {str(e)}")

        # Bad channel clustering analysis
        try:
            if channel_reasons and raw_for_topo.get_montage():
                bad_channels = list(channel_reasons.keys())
                clustering_result = compute_bad_channel_clustering(raw_for_topo, bad_channels)

                if 'error' not in clustering_result:
                    # Add clustering metrics
                    n_clustered = clustering_result['n_clustered_channels']
                    pct_clustered = clustering_result['pct_clustered']
                    severity = clustering_result['severity']

                    result.add_metric(QCMetric(
                        "Clustered Bad Channels",
                        n_clustered,
                        f"({pct_clustered:.1f}%)",
                        "bad" if severity == 'severe' else "warning" if severity in ['moderate', 'mild'] else "ok"
                    ))

                    # Add severity warning if problematic
                    if clustering_result.get('warning'):
                        result.add_note(f"⚠️ {clustering_result['warning']}")

                    # Add recommendation
                    recommendation = get_clustering_recommendation(clustering_result)
                    result.add_note(f"Clustering Recommendation: {recommendation}")

                    # Generate visualization
                    fig_clustering = visualize_channel_clustering(raw_for_topo, bad_channels, clustering_result)
                    result.add_figure("bad_channel_clustering", self.fig_to_base64(fig_clustering, self.dpi))
                    plt.close(fig_clustering)
        except Exception as e:
            result.add_note(f"Bad channel clustering analysis failed: {str(e)}")

        # Raw vs preprocessed overlay (middle 20s)
        if raw_before:
            try:
                fig_overlay = self._plot_raw_vs_preprocessed_overlay(raw_before, raw)
                result.add_figure("raw_vs_preprocessed_overlay", self.fig_to_base64(fig_overlay, self.dpi))
                plt.close(fig_overlay)
            except Exception as e:
                result.add_note(f"Raw vs preprocessed overlay failed: {str(e)}")

        # 5. Generate plots
        # Inter-channel correlation matrix
        try:
            fig_corr = self._plot_correlation_matrix(raw, eeg_picks_good)
            result.add_figure("correlation_matrix", self.fig_to_base64(fig_corr, self.dpi))
            plt.close(fig_corr)
        except Exception as e:
            result.add_note(f"Correlation matrix plot failed: {str(e)}")


        try:
            # Power spectral density
            fig_psd = self._plot_psd(raw, eeg_picks_good)
            result.add_figure("psd", self.fig_to_base64(fig_psd, self.dpi))
            plt.close(fig_psd)
        except Exception as e:
            result.add_note(f"PSD plot failed: {str(e)}")

        try:
            # Amplitude distribution
            fig_dist = self._plot_amplitude_distribution(raw, eeg_picks_good)
            result.add_figure("amplitude_dist", self.fig_to_base64(fig_dist, self.dpi))
            plt.close(fig_dist)
        except Exception as e:
            result.add_note(f"Amplitude distribution plot failed: {str(e)}")

        try:
            # Channel variance
            fig_var = self._plot_channel_variance(raw, eeg_picks)
            result.add_figure("channel_variance", self.fig_to_base64(fig_var, self.dpi))
            plt.close(fig_var)
        except Exception as e:
            result.add_note(f"Variance plot failed: {str(e)}")

        # ICA components plot (time series)
        if ica is not None:
            try:
                fig_ica = self._plot_ica_components(ica, raw, ica_components_info)
                result.add_figure("ica_components", self.fig_to_base64(fig_ica, self.dpi))
                plt.close(fig_ica)
            except Exception as e:
                result.add_note(f"ICA time series plot failed: {str(e)}")

            # ICA topography plot (spatial maps)
            try:
                fig_topo = self._plot_ica_topographies(ica, raw, ica_components_info)
                result.add_figure("ica_topographies", self.fig_to_base64(fig_topo, self.dpi))
                plt.close(fig_topo)
            except Exception as e:
                result.add_note(f"ICA topography plot failed: {str(e)}")

            # ICA component power spectra
            try:
                fig_spectra = self._plot_ica_component_spectra(ica, raw, ica_components_info)
                result.add_figure("ica_component_spectra", self.fig_to_base64(fig_spectra, self.dpi))
                plt.close(fig_spectra)
            except Exception as e:
                result.add_note(f"ICA component spectra plot failed: {str(e)}")

        # Before/after comparison if raw_before provided
        if raw_before is not None:
            try:
                fig_compare = self._plot_before_after(raw_before, raw, eeg_picks_good)
                result.add_figure("before_after", self.fig_to_base64(fig_compare, self.dpi))
                plt.close(fig_compare)
            except Exception as e:
                result.add_note(f"Before/after plot failed: {str(e)}")

            # Residual artifact quantification
            try:
                fig_residual = self._plot_residual_artifacts(raw_before, raw, eeg_picks_good)
                result.add_figure("residual_artifacts", self.fig_to_base64(fig_residual, self.dpi))
                plt.close(fig_residual)
            except Exception as e:
                result.add_note(f"Residual artifact plot failed: {str(e)}")

        # 6. Event-Related Potential (ERP) Visualization
        # Plot ERP waveforms if events/annotations are available (even without event_id parameter)
        try:
            fig_erp = self._plot_erp_waveforms(raw)
            if fig_erp is not None:
                result.add_figure("erp_waveforms", self.fig_to_base64(fig_erp, self.dpi))
                plt.close(fig_erp)
                result.add_note("ERP waveforms detected and plotted from annotations")
        except Exception as e:
            # Silently skip if ERP plotting fails (expected for resting-state data)
            pass

        # 7. Full Event-Related Potential (ERP) QC Report
        # Generate ERP reports if events and event_id are provided (task-based data)
        if events is not None and event_id is not None:
            try:
                erp_report_path = self._generate_erp_qc(
                    raw=raw,
                    subject_id=subject_id,
                    events=events,
                    event_id=event_id,
                    epochs_params=epochs_params,
                    erp_channels=erp_channels,
                    session_id=kwargs.get("session_id"),
                    task_name=kwargs.get("task_name"),
                )
                if erp_report_path:
                    result.add_note(f"ERP QC report generated: {erp_report_path.name}")
                    # Store path in result for reference
                    result.metadata["erp_qc_path"] = str(erp_report_path)
            except Exception as e:
                result.add_note(f"ERP QC generation failed: {str(e)}")

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

    def _create_ica_components_table(self, ica_components_info: List[Dict[str, Any]]) -> str:
        """
        Create HTML table for ICA components with ICLabel information.

        Displays ALL components with their classifications, variance explained, and rejection status.

        Args:
            ica_components_info: List of dicts with keys: index, label, probability, rejected, variance_explained

        Returns:
            HTML string with formatted table
        """
        if not ica_components_info:
            return "<p>No ICA component information available.</p>"

        rows = []
        total_variance = 0.0

        # First pass: calculate total variance for percentage calculation
        for comp in ica_components_info:
            var_explained = comp.get('variance_explained', 0.0)
            if isinstance(var_explained, (int, float)):
                total_variance += var_explained

        # Second pass: generate table rows
        for comp in ica_components_info:
            comp_idx = comp.get('index', '?')
            label = comp.get('label', 'Unknown')
            prob = comp.get('probability', 0.0)
            rejected = comp.get('rejected', False)
            reject_reason = comp.get('reject_reason', '-')
            var_explained = comp.get('variance_explained', 0.0)

            # Format probability as percentage
            prob_str = f"{prob*100:.1f}%" if isinstance(prob, (int, float)) else str(prob)

            # Format variance explained as percentage (already in percentage from ICLabel)
            if isinstance(var_explained, (int, float)):
                var_str = f"{var_explained:.2f}%"
            else:
                var_str = "N/A"

            # Status color: rejected=red, brain=green, other=yellow
            if rejected:
                status_color = "#ffcccc"  # Light red
            elif label.lower() == 'brain':
                status_color = "#ccffcc"  # Light green
            else:
                status_color = "#ffffcc"  # Light yellow (artifact/other)

            rejected_str = "Yes" if rejected else "No"

            rows.append(f"""
            <tr style="background-color: {status_color};">
                <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">{comp_idx}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{label}</td>
                <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">{prob_str}</td>
                <td style="padding: 8px; text-align: right; border: 1px solid #ddd;">{var_str}</td>
                <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">{rejected_str}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{reject_reason}</td>
            </tr>
            """)

        # Add total variance row (already in percentage from ICLabel)
        total_var_str = f"{total_variance:.2f}%" if isinstance(total_variance, (int, float)) else "N/A"
        rows.append(f"""
        <tr style="background-color: #e8e8e8; font-weight: bold;">
            <td colspan="3" style="padding: 8px; border: 1px solid #ddd; text-align: right;">Total Variance Explained:</td>
            <td style="padding: 8px; text-align: right; border: 1px solid #ddd;">{total_var_str}</td>
            <td colspan="2" style="padding: 8px; border: 1px solid #ddd;"></td>
        </tr>
        """)

        table_html = f"""
        <table style="width: 100%; border-collapse: collapse; border: 1px solid #ddd; margin-bottom: 20px;">
            <thead style="background-color: #f0f0f0; font-weight: bold;">
                <tr>
                    <th style="padding: 8px; text-align: center; border: 1px solid #ddd;">Comp.</th>
                    <th style="padding: 8px; border: 1px solid #ddd;">ICLabel Classification</th>
                    <th style="padding: 8px; text-align: center; border: 1px solid #ddd;">Probability</th>
                    <th style="padding: 8px; text-align: center; border: 1px solid #ddd;">Variance Explained</th>
                    <th style="padding: 8px; text-align: center; border: 1px solid #ddd;">Rejected</th>
                    <th style="padding: 8px; border: 1px solid #ddd;">Rejection Reason</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>

        <p style="font-size: 0.85em; color: #666; margin-top: 10px;">
            <strong>Color coding:</strong>
            <span style="background-color: #ccffcc; padding: 2px 5px; border-radius: 2px;">Brain</span>
            <span style="background-color: #ffffcc; padding: 2px 5px; border-radius: 2px; margin-left: 5px;">Artifact/Other</span>
            <span style="background-color: #ffcccc; padding: 2px 5px; border-radius: 2px; margin-left: 5px;">Rejected</span>
        </p>
        """
        return table_html

    def _create_before_after_metrics_table(self, before_after_metrics: Dict[str, Any]) -> str:
        """
        Create HTML table comparing before/after preprocessing metrics.

        Args:
            before_after_metrics: Dict with 'before' and 'after' keys, each containing metrics

        Returns:
            HTML string with formatted table
        """
        if not before_after_metrics:
            return "<p>No before/after comparison data available.</p>"

        before = before_after_metrics.get('before', {})
        after = before_after_metrics.get('after', {})

        # Extract metrics
        mean_before = before.get('mean', None)
        mean_after = after.get('mean', None)
        std_before = before.get('std', None)
        std_after = after.get('std', None)
        min_before = before.get('min', None)
        min_after = after.get('min', None)
        max_before = before.get('max', None)
        max_after = after.get('max', None)

        # Calculate changes
        mean_change = None
        mean_pct_change = None
        if mean_before is not None and mean_after is not None:
            mean_change = mean_after - mean_before
            mean_pct_change = 100 * (mean_after - mean_before) / mean_before if mean_before != 0 else 0

        min_change = None
        min_pct_change = None
        if min_before is not None and min_after is not None:
            min_change = min_after - min_before
            min_pct_change = 100 * (min_after - min_before) / min_before if min_before != 0 else 0

        max_change = None
        max_pct_change = None
        if max_before is not None and max_after is not None:
            max_change = max_after - max_before
            max_pct_change = 100 * (max_after - max_before) / max_before if max_before != 0 else 0

        std_change = None
        std_pct_change = None
        if std_before is not None and std_after is not None:
            std_change = std_after - std_before
            std_pct_change = 100 * (std_after - std_before) / std_before if std_before != 0 else 0

        rows = []

        # Mean amplitude row
        if mean_before is not None and mean_after is not None:
            rows.append(f"""
            <tr>
                <td style="padding: 8px; font-weight: bold;">Mean Amplitude</td>
                <td style="padding: 8px; text-align: right;">{mean_before:.2f} µV</td>
                <td style="padding: 8px; text-align: right;">{mean_after:.2f} µV</td>
                <td style="padding: 8px; text-align: right;">{mean_change:+.2f} µV</td>
                <td style="padding: 8px; text-align: right;">{mean_pct_change:+.1f}%</td>
            </tr>
            """)

        # Min amplitude row
        if min_before is not None and min_after is not None:
            rows.append(f"""
            <tr>
                <td style="padding: 8px; font-weight: bold;">Min Amplitude</td>
                <td style="padding: 8px; text-align: right;">{min_before:.2f} µV</td>
                <td style="padding: 8px; text-align: right;">{min_after:.2f} µV</td>
                <td style="padding: 8px; text-align: right;">{min_change:+.2f} µV</td>
                <td style="padding: 8px; text-align: right;">{min_pct_change:+.1f}%</td>
            </tr>
            """)

        # Max amplitude row
        if max_before is not None and max_after is not None:
            rows.append(f"""
            <tr>
                <td style="padding: 8px; font-weight: bold;">Max Amplitude</td>
                <td style="padding: 8px; text-align: right;">{max_before:.2f} µV</td>
                <td style="padding: 8px; text-align: right;">{max_after:.2f} µV</td>
                <td style="padding: 8px; text-align: right;">{max_change:+.2f} µV</td>
                <td style="padding: 8px; text-align: right;">{max_pct_change:+.1f}%</td>
            </tr>
            """)

        # Std deviation row
        if std_before is not None and std_after is not None:
            rows.append(f"""
            <tr>
                <td style="padding: 8px; font-weight: bold;">Std Deviation</td>
                <td style="padding: 8px; text-align: right;">{std_before:.2f} µV</td>
                <td style="padding: 8px; text-align: right;">{std_after:.2f} µV</td>
                <td style="padding: 8px; text-align: right;">{std_change:+.2f} µV</td>
                <td style="padding: 8px; text-align: right;">{std_pct_change:+.1f}%</td>
            </tr>
            """)

        if not rows:
            return "<p>No valid comparison data.</p>"

        table_html = f"""
        <table style="width: 100%; border-collapse: collapse; border: 1px solid #ddd;">
            <thead style="background-color: #f0f0f0; font-weight: bold;">
                <tr>
                    <th style="padding: 8px; border: 1px solid #ddd;">Metric</th>
                    <th style="padding: 8px; text-align: right; border: 1px solid #ddd;">Before</th>
                    <th style="padding: 8px; text-align: right; border: 1px solid #ddd;">After</th>
                    <th style="padding: 8px; text-align: right; border: 1px solid #ddd;">Absolute Change</th>
                    <th style="padding: 8px; text-align: right; border: 1px solid #ddd;">% Change</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>

        <p style="font-size: 0.9em; color: #666; margin-top: 10px;">
            <strong>Note:</strong> Mean amplitude for preprocessed data should be close to 0 µV due to average reference.
            Negative % change indicates reduction in signal amplitude/variability.
        </p>
        """
        return table_html

    def _get_good_samples_mask(self, raw: mne.io.BaseRaw) -> np.ndarray:
        """Get boolean mask of good (non-bad) samples.

        Handles overlapping annotations correctly by marking each sample
        as bad if it falls within ANY bad annotation.

        Returns:
            Boolean array of shape (n_times,) where True = good sample
        """
        n_samples = raw.n_times
        sfreq = raw.info['sfreq']
        good_mask = np.ones(n_samples, dtype=bool)

        if raw.annotations:
            for annot in raw.annotations:
                if annot['description'].startswith('BAD'):
                    # Convert onset/duration to sample indices
                    start_sample = int(annot['onset'] * sfreq)
                    end_sample = int((annot['onset'] + annot['duration']) * sfreq)
                    # Clamp to valid range
                    start_sample = max(0, start_sample)
                    end_sample = min(n_samples, end_sample)
                    # Mark as bad
                    good_mask[start_sample:end_sample] = False

        return good_mask

    def _filter_dead_channels(self, raw: mne.io.BaseRaw, picks: np.ndarray) -> np.ndarray:
        """Filter out dead channels (zero/near-zero variance) from picks.

        Dead channels cause infinite/NaN PSD values which appear as negative
        power in dB scale, corrupting the plots.
        """
        if len(picks) == 0:
            return picks

        data = raw.get_data(picks=picks)
        variances = np.var(data, axis=1)
        # Threshold: variance < 1e-20 V² (essentially zero)
        valid_mask = variances > 1e-20
        return picks[valid_mask]

    def _plot_psd(self, raw: mne.io.BaseRaw, picks: np.ndarray) -> plt.Figure:
        """Plot power spectral density."""
        fig, ax = plt.subplots(figsize=(10, 4))

        # Filter out dead channels to avoid infinite PSD values
        valid_picks = self._filter_dead_channels(raw, picks)

        if len(valid_picks) > 0:
            # Note: reject_by_annotation=False to show actual PSD including bad segments for QC
            psd = raw.compute_psd(picks=valid_picks, fmin=0.5, fmax=80,
                                 reject_by_annotation=False, verbose=False)
            psd.plot(axes=ax, show=False)

            n_excluded = len(picks) - len(valid_picks)
            if n_excluded > 0:
                ax.text(0.02, 0.98, f"({n_excluded} dead channel(s) excluded)",
                        transform=ax.transAxes, ha="left", va="top", fontsize=8,
                        color="red", style="italic")
        else:
            ax.text(0.5, 0.5, "No valid EEG channels", ha="center", va="center", transform=ax.transAxes)

        ax.set_title("Power Spectral Density (Preprocessed)")
        fig.tight_layout()
        return fig

    def _plot_amplitude_distribution(self, raw: mne.io.BaseRaw, picks: np.ndarray) -> plt.Figure:
        """Plot amplitude distribution histogram."""
        fig, ax = plt.subplots(figsize=(10, 4))

        if len(picks) > 0:
            data = raw.get_data(picks=picks).flatten() * 1e6  # to µV

            # Remove extreme outliers for plotting
            p1, p99 = np.percentile(data, [1, 99])
            data_clipped = data[(data >= p1) & (data <= p99)]

            ax.hist(data_clipped, bins=100, color="steelblue", alpha=0.7, edgecolor="none")
            ax.axvline(0, color="red", linestyle="--", alpha=0.5)

            # Add stats
            ax.text(0.95, 0.95, f"Mean: {np.mean(data):.2f} µV\nStd: {np.std(data):.2f} µV",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            ax.set_xlabel("Amplitude (µV)")
            ax.set_ylabel("Count")
        else:
            ax.text(0.5, 0.5, "No EEG channels", ha="center", va="center", transform=ax.transAxes)

        ax.set_title("Amplitude Distribution")
        fig.tight_layout()
        return fig

    def _plot_channel_variance(self, raw: mne.io.BaseRaw, picks: np.ndarray) -> plt.Figure:
        """Plot channel variance as bar chart."""
        fig, ax = plt.subplots(figsize=(12, 4))

        if len(picks) > 0:
            data = raw.get_data(picks=picks)
            variances = np.var(data, axis=1) * 1e12  # to µV²
            ch_names = [raw.ch_names[p] for p in picks]

            colors = ["red" if ch in raw.info["bads"] else "steelblue" for ch in ch_names]
            ax.bar(range(len(variances)), variances, color=colors)

            # Median line
            median_var = np.median(variances)
            ax.axhline(median_var, color="orange", linestyle="--", label=f"Median: {median_var:.1f}")

            if len(ch_names) > 30:
                step = len(ch_names) // 20
                ax.set_xticks(range(0, len(ch_names), step))
                ax.set_xticklabels([ch_names[i] for i in range(0, len(ch_names), step)], rotation=45, ha="right")
            else:
                ax.set_xticks(range(len(ch_names)))
                ax.set_xticklabels(ch_names, rotation=45, ha="right")

            ax.set_ylabel("Variance (µV²)")
            ax.set_xlabel("Channel")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No EEG channels", ha="center", va="center", transform=ax.transAxes)

        ax.set_title("Channel Variance (Preprocessed)")
        fig.tight_layout()
        return fig

    def _plot_correlation_matrix(self, raw: mne.io.BaseRaw, picks: np.ndarray) -> plt.Figure:
        """Plot inter-channel correlation matrix.

        Helps identify bridged channels (very high correlation) and
        disconnected channels (very low correlation with all others).
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Filter out dead channels
        valid_picks = self._filter_dead_channels(raw, picks)

        if len(valid_picks) < 2:
            ax.text(0.5, 0.5, "Insufficient valid channels for correlation",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Inter-Channel Correlation Matrix")
            return fig

        # Get data and compute correlation
        data = raw.get_data(picks=valid_picks)
        ch_names = [raw.ch_names[p] for p in valid_picks]

        # Compute correlation matrix
        corr_matrix = np.corrcoef(data)

        # Handle NaN values (from constant channels)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Plot heatmap
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation', fontsize=10)

        # Axis labels
        n_ch = len(ch_names)
        if n_ch <= 30:
            ax.set_xticks(range(n_ch))
            ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
            ax.set_yticks(range(n_ch))
            ax.set_yticklabels(ch_names, fontsize=6)
        else:
            # Show every Nth label for large matrices
            step = max(1, n_ch // 20)
            ax.set_xticks(range(0, n_ch, step))
            ax.set_xticklabels([ch_names[i] for i in range(0, n_ch, step)], rotation=90, fontsize=6)
            ax.set_yticks(range(0, n_ch, step))
            ax.set_yticklabels([ch_names[i] for i in range(0, n_ch, step)], fontsize=6)

        # Statistics
        # Get off-diagonal elements
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        off_diag = corr_matrix[mask]
        mean_corr = np.mean(off_diag)
        high_corr_count = np.sum(np.abs(off_diag) > 0.95)
        low_corr_count = np.sum(np.abs(off_diag) < 0.1)

        stats_text = (f"Mean corr: {mean_corr:.3f}\n"
                      f"Very high (>0.95): {high_corr_count//2} pairs\n"
                      f"Very low (<0.1): {low_corr_count//2} pairs")
        ax.text(1.02, 0.02, stats_text, transform=ax.transAxes, fontsize=8,
                va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_title("Inter-Channel Correlation Matrix (Preprocessed)")
        fig.tight_layout()
        return fig

    def _plot_ica_components(self, ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw,
                            ica_components_info: Optional[List[Dict]] = None) -> plt.Figure:
        """Plot ICA component time series summary with ICLabel classifications."""
        n_components = ica.n_components_  # Show ALL components

        # Calculate grid size to fit all components
        n_cols = 5
        n_rows = int(np.ceil(n_components / n_cols))

        # Use 2.5 inches per row to accommodate titles and axis decorations
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows))
        axes = axes.flatten() if n_components > 1 else [axes]

        # Build lookup for component info
        comp_info_map = {}
        if ica_components_info:
            for comp in ica_components_info:
                comp_info_map[comp['index']] = comp

        try:
            # Get component sources
            sources = ica.get_sources(raw).get_data()

            for i in range(n_components):
                ax = axes[i]
                # Plot variance over time (simplified)
                segment = sources[i, :int(10 * raw.info["sfreq"])]  # First 10s

                # Determine color based on rejection status
                line_color = "k"
                is_rejected = i in ica.exclude

                # Check ICLabel info if available
                if i in comp_info_map:
                    comp_info = comp_info_map[i]
                    is_rejected = comp_info.get('rejected', False)

                # Use red for rejected components
                if is_rejected:
                    line_color = "#E41A1C"  # Red
                    ax.set_facecolor("#ffeeee")  # Light red background

                ax.plot(segment, line_color, linewidth=0.5)

                # Build title with classification
                title = f"IC{i}"
                if i in comp_info_map:
                    comp_info = comp_info_map[i]
                    label = comp_info.get('label', '?')
                    prob = comp_info.get('probability', 0)
                    # Abbreviate labels for space
                    label_abbrev = {
                        'brain': 'Br', 'muscle': 'Mu', 'eye': 'Ey',
                        'heart': 'Hr', 'line_noise': 'LN',
                        'channel_noise': 'CN', 'other': 'Ot'
                    }
                    label_short = label_abbrev.get(label, label[:2])
                    title += f" {label_short}"
                    if is_rejected:
                        title += " ✗"
                elif is_rejected:
                    title += " ✗"

                ax.set_title(title, fontsize=8, color='red' if is_rejected else 'black')
                ax.set_xticks([])
                ax.set_yticks([])

            # Hide unused axes
            for i in range(n_components, len(axes)):
                axes[i].set_visible(False)

        except Exception:
            axes[0].text(0.5, 0.5, "Could not plot ICA components", ha="center", va="center")

        fig.suptitle(f"ICA Components ({ica.n_components_} total, {len(ica.exclude)} excluded)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle (3% at top)
        return fig

    def _plot_ica_topographies(self, ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw,
                              ica_components_info: Optional[List[Dict]] = None) -> plt.Figure:
        """Plot ICA component topographies (scalp maps) with ICLabel classifications.

        Shows spatial patterns of ICA components, useful for identifying
        artifact components (e.g., eye blinks have frontal pattern).
        """
        n_components = ica.n_components_  # Show ALL components

        # Build lookup for component info
        comp_info_map = {}
        if ica_components_info:
            for comp in ica_components_info:
                comp_info_map[comp['index']] = comp

        # Use MNE's built-in ICA component plotting
        try:
            # ica.plot_components returns a list of figures (one per 20 components)
            figs = ica.plot_components(
                picks=range(n_components),
                show=False,
                title=None,
            )

            # Replace MNE's head outline + sensor spots with EEGLAB-style versions
            if isinstance(figs, list):
                for source_fig in figs:
                    self._fix_ica_head_outlines(source_fig, ica)
            else:
                self._fix_ica_head_outlines(figs, ica)

            # Annotate with ICLabel classifications before combining
            if isinstance(figs, list):
                for fig_idx, source_fig in enumerate(figs):
                    comp_start = fig_idx * 20
                    self._annotate_ica_topographies(source_fig, comp_start, comp_info_map)

                n_figs = len(figs)
                if n_figs == 1:
                    fig = figs[0]
                    fig.suptitle(f"ICA Component Topographies ({n_components} components)")
                else:
                    combined_fig, combined_axes = plt.subplots(
                        n_figs, 1,
                        figsize=(12, 8 * n_figs),
                        constrained_layout=True
                    )
                    if n_figs == 1:
                        combined_axes = [combined_axes]

                    for idx, (source_fig, ax) in enumerate(zip(figs, combined_axes)):
                        source_fig.canvas.draw()
                        img = np.frombuffer(source_fig.canvas.tostring_rgb(), dtype=np.uint8)
                        img = img.reshape(source_fig.canvas.get_width_height()[::-1] + (3,))
                        ax.imshow(img)
                        ax.axis('off')
                        plt.close(source_fig)

                    combined_fig.suptitle(f"ICA Component Topographies ({n_components} components total)",
                                         fontsize=14)
                    fig = combined_fig
            else:
                fig = figs
                self._annotate_ica_topographies(fig, 0, comp_info_map)
                fig.suptitle(f"ICA Component Topographies ({n_components} components)")
                fig.subplots_adjust(top=0.92)

            return fig

        except Exception as e:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5,
                    f"ICA topographies unavailable\n(requires montage with channel locations)\n\nError: {str(e)[:100]}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, wrap=True)
            ax.set_title("ICA Component Topographies")
            ax.axis("off")
            return fig

    def _annotate_ica_topographies(self, fig: plt.Figure, comp_start: int,
                                   comp_info_map: Dict[int, Dict]) -> None:
        """Annotate ICA topography figure with ICLabel classifications."""
        for ax in fig.axes:
            title = ax.get_title()
            if title and title.startswith('ICA'):
                try:
                    comp_idx = int(title[3:])

                    if comp_idx in comp_info_map:
                        comp_info = comp_info_map[comp_idx]
                        label = comp_info.get('label', '?')
                        prob = comp_info.get('probability', 0)
                        is_rejected = comp_info.get('rejected', False)

                        label_abbrev = {
                            'brain': 'Brain', 'muscle': 'Muscle', 'eye': 'Eye',
                            'heart': 'Heart', 'line_noise': 'LineNoise',
                            'channel_noise': 'ChanNoise', 'other': 'Other'
                        }
                        label_text = label_abbrev.get(label, label)

                        new_title = f"{title}\n{label_text} ({prob:.2f})"
                        if is_rejected:
                            new_title += " ✗"
                            ax.set_title(new_title, fontsize=9, color='red', fontweight='bold')
                            for spine in ax.spines.values():
                                spine.set_edgecolor('red')
                                spine.set_linewidth(2)
                        else:
                            ax.set_title(new_title, fontsize=9)

                except (ValueError, IndexError):
                    pass

    def _fix_ica_head_outlines(self, fig: plt.Figure, ica: mne.preprocessing.ICA) -> None:
        """Replace MNE's head outline + sensor spots with our EEGLAB-style versions."""
        locs_pos = _load_neuroscan_locs()
        for ax in fig.axes:
            lines = list(ax.lines)
            if not lines:
                continue
            head_radius = None
            for line in lines:
                x, y = line.get_data()
                if len(x) > 50:
                    head_radius = np.mean(np.sqrt(x**2 + y**2))
                    break
            if head_radius is None:
                continue
            locs_scale = head_radius / 0.5
            n_ica_ch = len(ica.ch_names)
            for coll in ax.collections:
                if not hasattr(coll, 'get_offsets'):
                    continue
                off = coll.get_offsets()
                if len(off) != n_ica_ch:
                    continue
                new_off = np.zeros_like(off)
                for i, name in enumerate(ica.ch_names):
                    if name in locs_pos:
                        new_off[i] = locs_pos[name] * locs_scale
                    else:
                        new_off[i] = off[i] if i < len(off) else [0, 0]
                coll.set_offsets(new_off)
            for line in lines:
                line.remove()
            _draw_head_outline(ax, head_radius=head_radius, linewidth=1.5)
            ax.set_aspect('equal')

    def _plot_ica_component_spectra(self, ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw,
                                    ica_components_info: Optional[List[Dict]] = None) -> plt.Figure:
        """
        Plot power spectral density for each ICA component.

        Shows PSD for each ICA component, color-coded by ICLabel classification.
        Helps validate component classifications:
        - Brain components: should show alpha peaks (~10 Hz)
        - Eye components: high low-frequency power
        - Muscle components: high frequency power (>20 Hz)
        - Line noise: sharp peak at 50/60 Hz

        Args:
            ica: ICA object with fitted components
            raw: Raw data (used to get sources)
            ica_components_info: Optional ICLabel classification info

        Returns:
            Matplotlib figure with PSD subplots for each component
        """
        n_components = ica.n_components_

        # Calculate grid size
        n_cols = 4
        n_rows = int(np.ceil(n_components / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 2.5 * n_rows))
        axes = axes.flatten() if n_components > 1 else [axes]

        # Build lookup for component info
        comp_info_map = {}
        if ica_components_info:
            for comp in ica_components_info:
                comp_info_map[comp['index']] = comp

        # Define color scheme for ICLabel classes
        label_colors = {
            'brain': '#4DAF4A',           # Green
            'muscle': '#E41A1C',          # Red
            'eye': '#377EB8',             # Blue
            'heart': '#FF7F00',           # Orange
            'line_noise': '#984EA3',      # Purple
            'channel_noise': '#A65628',   # Brown
            'other': '#999999',           # Gray
        }

        try:
            # Get component sources
            sources = ica.get_sources(raw)

            for i in range(n_components):
                ax = axes[i]

                # Compute PSD for this component
                # Use first 60 seconds or all data if shorter
                sfreq = raw.info['sfreq']
                max_samples = min(int(60 * sfreq), sources.get_data().shape[1])
                component_data = sources.get_data()[i, :max_samples]

                # Compute PSD using Welch's method
                from scipy.signal import welch
                freqs, psd = welch(component_data, fs=sfreq, nperseg=int(2*sfreq),
                                  noverlap=int(sfreq), nfft=int(4*sfreq))

                # Convert to dB
                psd_db = 10 * np.log10(psd)

                # Get component classification
                label = 'other'
                prob = 0.0
                is_rejected = i in ica.exclude

                if i in comp_info_map:
                    comp_info = comp_info_map[i]
                    label = comp_info.get('label', 'other')
                    prob = comp_info.get('probability', 0.0)
                    is_rejected = comp_info.get('rejected', False)

                # Choose color based on classification
                color = label_colors.get(label, '#999999')

                # Plot PSD (0-80 Hz)
                freq_mask = freqs <= 80
                ax.plot(freqs[freq_mask], psd_db[freq_mask], color=color, linewidth=1.5, alpha=0.8)

                # Highlight if rejected
                if is_rejected:
                    ax.set_facecolor('#fff0f0')  # Light red background
                    ax.spines['bottom'].set_color('red')
                    ax.spines['top'].set_color('red')
                    ax.spines['left'].set_color('red')
                    ax.spines['right'].set_color('red')
                    ax.spines['bottom'].set_linewidth(2)
                    ax.spines['top'].set_linewidth(2)
                    ax.spines['left'].set_linewidth(2)
                    ax.spines['right'].set_linewidth(2)

                # Title with classification
                title = f"IC{i}: {label.replace('_', ' ').title()}"
                if prob > 0:
                    title += f" ({prob:.2f})"
                if is_rejected:
                    title += " ✗"

                ax.set_title(title, fontsize=8, color='red' if is_rejected else 'black',
                           fontweight='bold' if is_rejected else 'normal')

                # Add subtle alpha band marker for brain components
                if label == 'brain':
                    ax.axvspan(8, 12, alpha=0.2, color='green', label='Alpha')

                # Add high-frequency marker for muscle
                if label == 'muscle':
                    ax.axvspan(30, 80, alpha=0.2, color='red', label='Muscle')

                # Add line noise markers (60 Hz only, within 0-80 Hz range)
                if label == 'line_noise':
                    ax.axvline(60, color='orange', linestyle='--', alpha=0.5, linewidth=1)

                ax.set_xlabel('Frequency (Hz)', fontsize=7)
                ax.set_ylabel('PSD (dB)', fontsize=7)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.3)

                # Set consistent axis limits for all components
                ax.set_xlim([0, 80])  # Consistent x-axis range
                ax.set_ylim([psd_db[freq_mask].min() - 5, psd_db[freq_mask].max() + 5])

            # Hide unused axes
            for i in range(n_components, len(axes)):
                axes[i].set_visible(False)

        except Exception as e:
            axes[0].text(0.5, 0.5, f"Could not compute ICA component PSDs\n{str(e)[:100]}",
                        ha="center", va="center", fontsize=8)

        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=label_colors['brain'], label='Brain'),
            Patch(facecolor=label_colors['eye'], label='Eye'),
            Patch(facecolor=label_colors['muscle'], label='Muscle'),
            Patch(facecolor=label_colors['heart'], label='Heart'),
            Patch(facecolor=label_colors['line_noise'], label='Line Noise'),
            Patch(facecolor=label_colors['channel_noise'], label='Channel Noise'),
            Patch(facecolor=label_colors['other'], label='Other'),
        ]

        fig.suptitle(f"ICA Component Power Spectra ({n_components} components, {len(ica.exclude)} rejected)",
                    fontsize=12, fontweight='bold', y=1.0)
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98),
                  fontsize=9, ncol=7, framealpha=0.9)
        fig.subplots_adjust(top=0.95, hspace=0.4, wspace=0.3)
        return fig

    def _plot_bad_channels_topomap(self, raw: mne.io.BaseRaw, channel_reasons: Dict[str, str]) -> plt.Figure:
        """Plot topography showing bad/removed channels with color-coded markers by reason.

        Uses MNE's layout system for channel positions with a custom head outline.
        Bad channels are highlighted with larger colored markers.
        """
        from matplotlib.lines import Line2D

        # Check montage
        if raw.get_montage() is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'Montage required for topography\n(set montage before preprocessing)',
                    ha='center', va='center', fontsize=12)
            ax.set_title('Bad Channel Topography')
            ax.axis('off')
            return fig

        # Color scheme for reasons (colorblind-friendly)
        # Map both old and new reason formats
        reason_colors = {
            'good': '#888888',                    # Gray for good channels
            'dropped_ransac': '#E41A1C',          # Red - RANSAC dropped
            'dropped_variance': '#E41A1C',        # Red - variance dropped
            'dropped_correlation': '#E41A1C',     # Red - correlation dropped
            'dropped_deviation': '#E41A1C',       # Red - deviation dropped
            'interpolated_ransac': '#377EB8',     # Blue - RANSAC interpolated
            'interpolated_variance': '#377EB8',   # Blue - variance interpolated
            'marked_bad_ransac': '#FF7F00',       # Orange - marked bad
            'no_position': '#984EA3',             # Purple - no montage position
            'dropped': '#E41A1C',                 # Red - dropped (generic)
            'marked_bad': '#FF7F00',              # Orange - marked bad (generic)
            'dropped_flat': '#E41A1C',            # Red - dropped flat channel
            # Bare method-name reasons (from legacy/converted configs)
            'variance': '#E41A1C',
            # Legacy reasons (for backwards compatibility)
            'flatline': '#E41A1C',
            'high_variance': '#FF7F00',
            'bridged': '#984EA3',
            'interpolated': '#377EB8',
        }

        reason_labels = {
            'good': 'Good',
            'dropped_ransac': 'Dropped (RANSAC)',
            'dropped_variance': 'Dropped (Variance)',
            'dropped_correlation': 'Dropped (Correlation)',
            'dropped_deviation': 'Dropped (Deviation)',
            'interpolated_ransac': 'Interpolated (RANSAC)',
            'interpolated_variance': 'Interpolated (Variance)',
            'marked_bad_ransac': 'Marked Bad (RANSAC)',
            'no_position': 'Dropped (No Position)',
            'dropped': 'Dropped',
            'marked_bad': 'Marked Bad',
            'dropped_flat': 'Dropped (Flat)',
            # Bare method-name reasons
            'variance': 'Dropped (Variance)',
            # Legacy labels
            'flatline': 'Flatline (dropped)',
            'high_variance': 'High Variance (dropped)',
            'bridged': 'Bridged (dropped)',
            'interpolated': 'Interpolated (RANSAC)',
        }

        # Get EEG channel indices and names
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        eeg_ch_names = [raw.ch_names[i] for i in eeg_picks]

        # Get channel positions from Neuroscan .locs (EEGLAB convention)
        locs_pos = _load_neuroscan_locs()
        if locs_pos:
            pos_dict = {name: locs_pos[name] for name in eeg_ch_names if name in locs_pos}
        else:
            layout = mne.channels.find_layout(raw.info, ch_type='eeg')
            pos_dict = {name: pos for name, pos in zip(layout.names, layout.pos[:, :2])}

        # Build position array for EEG channels
        layout_pos = []
        valid_ch_indices = []
        for i, ch_name in enumerate(eeg_ch_names):
            if ch_name in pos_dict:
                layout_pos.append(pos_dict[ch_name])
                valid_ch_indices.append(i)

        if not layout_pos:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'Could not get channel positions from layout',
                    ha='center', va='center')
            ax.set_title('Bad Channel Topography')
            return fig

        pos = np.array(layout_pos)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw head outline using shared EEGLAB-style function
        _draw_head_outline(ax, head_radius=0.5, linewidth=2)

        # Track which reasons are present for legend
        reasons_present = set()

        # Categorize channels
        good_indices = []
        bad_indices = {reason: [] for reason in reason_colors if reason != 'good'}

        for i, ch_idx in enumerate(valid_ch_indices):
            ch_name = eeg_ch_names[ch_idx]
            reason = channel_reasons.get(ch_name, 'good')
            reasons_present.add(reason)
            if reason == 'good':
                good_indices.append(i)
            else:
                if reason not in bad_indices:
                    bad_indices[reason] = []
                bad_indices[reason].append(i)

        # Plot good channels (smaller, in background)
        if good_indices:
            good_pos = pos[good_indices]
            ax.scatter(good_pos[:, 0], good_pos[:, 1],
                      c=reason_colors['good'], s=50, marker='o',
                      alpha=0.6, zorder=1, linewidths=0)

        # Plot bad channels by reason (larger markers, on top)
        for reason, indices in bad_indices.items():
            if indices:
                bad_pos = pos[indices]
                # Use color for this reason, or default to red if unknown
                color = reason_colors.get(reason, '#E41A1C')
                ax.scatter(bad_pos[:, 0], bad_pos[:, 1],
                          c=color, s=200, marker='o',
                          edgecolors='black', linewidths=2, zorder=2)

        # Add channel names for bad channels
        for i, ch_idx in enumerate(valid_ch_indices):
            ch_name = eeg_ch_names[ch_idx]
            if ch_name in channel_reasons:
                ax.annotate(ch_name, (pos[i, 0], pos[i, 1]),
                           fontsize=8, ha='center', va='bottom',
                           xytext=(0, 8), textcoords='offset points',
                           fontweight='bold')

        # Set asymmetric axis limits (more space at top for nose, legend at bottom)
        ax.set_xlim(-0.65, 0.65)
        ax.set_ylim(-0.65, 0.75)
        ax.set_aspect('equal')
        ax.axis('off')

        # Legend (only show reasons that are present)
        legend_elements = []
        for reason in reasons_present:
            # Get label and color, with fallbacks for unknown reasons
            label = reason_labels.get(reason, reason.replace('_', ' ').title())
            color = reason_colors.get(reason, '#E41A1C')
            size = 8 if reason == 'good' else 12

            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', label=label,
                       markerfacecolor=color, markersize=size,
                       markeredgecolor='black' if reason != 'good' else 'none',
                       markeredgewidth=1.5 if reason != 'good' else 0)
            )

        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02),
                 fontsize=10, frameon=True, ncol=3)

        # Count bad channels
        n_bad = len(channel_reasons)
        n_total = len(eeg_ch_names)
        ax.set_title(f'Bad/Removed Channels ({n_bad}/{n_total} channels affected)', fontsize=14, pad=10)

        fig.tight_layout()
        return fig

    def _plot_raw_vs_preprocessed_overlay(self, raw_before: mne.io.BaseRaw, raw_after: mne.io.BaseRaw) -> plt.Figure:
        """Plot raw vs preprocessed signal comparison in stacked subplots.

        Shows a 10-second segment from the middle of the recording for 6 channels,
        with raw (demeaned) in light color and preprocessed (demeaned) overlaid.
        """
        sfreq = raw_before.info['sfreq']
        duration = 10.0  # 10 seconds for clarity
        n_samples = int(duration * sfreq)

        # Middle segment
        mid_start = max(0, (raw_before.n_times - n_samples) // 2)
        t_slice = slice(mid_start, mid_start + n_samples)
        times = raw_before.times[t_slice] - raw_before.times[mid_start]  # Start from 0

        # Good EEG channels (common to both)
        picks_before = mne.pick_types(raw_before.info, eeg=True, exclude=[])
        picks_after = mne.pick_types(raw_after.info, eeg=True, exclude=[])
        common_ch_names = sorted(
            set([raw_before.ch_names[i] for i in picks_before]) &
            set([raw_after.ch_names[i] for i in picks_after])
        )

        # Select 6 channels: try to get spatial spread (frontal, central, parietal, occipital)
        priority_prefixes = ['Fz', 'Cz', 'Pz', 'Oz', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
        selected_channels = []
        for prefix in priority_prefixes:
            for ch in common_ch_names:
                if ch.startswith(prefix) and ch not in selected_channels:
                    selected_channels.append(ch)
                    break
            if len(selected_channels) >= 6:
                break

        # Fill remaining slots if needed
        for ch in common_ch_names:
            if ch not in selected_channels:
                selected_channels.append(ch)
            if len(selected_channels) >= 6:
                break

        n_channels = len(selected_channels)
        if n_channels == 0:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.text(0.5, 0.5, 'No common channels between raw and preprocessed data',
                    ha='center', va='center')
            return fig

        fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2 * n_channels), sharex=True)
        if n_channels == 1:
            axes = [axes]

        # Get data
        picks_b = [raw_before.ch_names.index(ch) for ch in selected_channels]
        picks_a = [raw_after.ch_names.index(ch) for ch in selected_channels]

        data_before = raw_before.get_data(picks=picks_b)[:, t_slice]
        data_after = raw_after.get_data(picks=picks_a)[:, t_slice]

        # Demean each channel
        data_before = data_before - np.mean(data_before, axis=-1, keepdims=True)
        data_after = data_after - np.mean(data_after, axis=-1, keepdims=True)

        # Convert to µV
        data_before *= 1e6
        data_after *= 1e6

        for i, (ax, ch_name) in enumerate(zip(axes, selected_channels)):
            # Raw signal (light, in background)
            ax.plot(times, data_before[i], color='#CCCCCC', linewidth=1.0,
                    label='Raw' if i == 0 else '', alpha=0.8)
            # Preprocessed signal (dark, in foreground)
            ax.plot(times, data_after[i], color='#2166AC', linewidth=0.8,
                    label='Preprocessed' if i == 0 else '')

            ax.set_ylabel(f'{ch_name}\n(µV)', fontsize=9)
            ax.yaxis.set_label_coords(-0.08, 0.5)

            # Set reasonable y-limits based on preprocessed data
            y_max = np.percentile(np.abs(data_after[i]), 99) * 1.5
            y_max = max(y_max, 20)  # Minimum ±20 µV
            ax.set_ylim(-y_max, y_max)

            # Light grid
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)

            # Remove top/right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        axes[-1].set_xlabel('Time (s)', fontsize=10)

        # Add legend at top
        axes[0].legend(loc='upper right', fontsize=9, framealpha=0.9)

        # Calculate time offset for title
        mid_time = raw_before.times[mid_start]
        fig.suptitle(f'Raw vs Preprocessed Signal Comparison\n(10s segment starting at {mid_time:.1f}s, demeaned)',
                     fontsize=11, y=0.98)

        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        return fig

    def _plot_before_after(
        self,
        raw_before: mne.io.BaseRaw,
        raw_after: mne.io.BaseRaw,
        picks: np.ndarray
    ) -> plt.Figure:
        """Plot before/after PSD comparison using MNE's native plotting."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # For BEFORE: use ALL EEG channels
        picks_before_all = mne.pick_types(raw_before.info, eeg=True, exclude=[])
        valid_picks_before = self._filter_dead_channels(raw_before, picks_before_all)

        # For AFTER: also use ALL EEG channels present in raw_after
        # (some may have been dropped, but interpolated channels should be included)
        picks_after_all = mne.pick_types(raw_after.info, eeg=True, exclude=[])
        valid_picks_after = self._filter_dead_channels(raw_after, picks_after_all)

        # Plot before
        if len(valid_picks_before) > 0:
            psd_before = raw_before.compute_psd(
                picks=valid_picks_before, fmin=0.5, fmax=80,
                reject_by_annotation=False, verbose=False
            )
            psd_before.plot(axes=axes[0], show=False, spatial_colors=True)
            axes[0].set_title(f"Before Preprocessing ({len(valid_picks_before)} channels)")
        else:
            axes[0].text(0.5, 0.5, "No valid channels", ha="center", va="center",
                        transform=axes[0].transAxes)
            axes[0].set_title("Before Preprocessing")

        # Plot after
        n_axes_before = len(fig.axes)
        if len(valid_picks_after) > 0:
            psd_after = raw_after.compute_psd(
                picks=valid_picks_after, fmin=0.5, fmax=80,
                reject_by_annotation=False, verbose=False
            )
            psd_after.plot(axes=axes[1], show=False, spatial_colors=True)

            # Fix after panel spatial color legend: replace MNE 3D→2D with .locs positions
            if len(fig.axes) > n_axes_before:
                host_ax = fig.axes[-1]
                orig_coll = next((c for c in host_ax.collections
                                  if hasattr(c, 'get_offsets') and hasattr(c, 'get_array')), None)
                if orig_coll is not None and len(orig_coll.get_offsets()) > 0:
                    orig_offsets = orig_coll.get_offsets().copy()

                    head_r_axes = 0.095
                    for line in host_ax.lines:
                        r = float(np.sqrt(line.get_xdata()**2 + line.get_ydata()**2).max())
                        if 0.08 < r < 0.15:
                            head_r_axes = r
                            break

                    locs_scale = head_r_axes / 0.5
                    locs_pos = _load_neuroscan_locs()
                    ch_names_after = [raw_after.ch_names[i] for i in valid_picks_after]

                    new_offsets = np.zeros_like(orig_offsets)
                    for i, ch_name in enumerate(ch_names_after):
                        if i < len(orig_offsets) and ch_name in locs_pos:
                            new_offsets[i] = locs_pos[ch_name] * locs_scale
                        elif i < len(orig_offsets):
                            new_offsets[i] = orig_offsets[i]

                    orig_coll.set_offsets(new_offsets)

                    for line in host_ax.lines:
                        line.remove()

                    nh = head_r_axes * 0.18
                    _draw_head_outline(host_ax, head_radius=head_r_axes)

                    max_ext = max(float(np.abs(new_offsets).max()), head_r_axes + nh) * 1.15
                    host_ax.set_xlim(-max_ext, max_ext)
                    host_ax.set_ylim(-max_ext, max_ext)
                    host_ax.set_aspect('equal')
                    host_ax.axis('off')

            # Show channel count and indicate if channels were dropped
            n_dropped = len(valid_picks_before) - len(valid_picks_after)
            if n_dropped > 0:
                # Get dropped channel names for informative title
                ch_names_before = [raw_before.ch_names[i] for i in valid_picks_before]
                ch_names_after = [raw_after.ch_names[i] for i in valid_picks_after]
                dropped_names = set(ch_names_before) - set(ch_names_after)

                title_text = f"After Preprocessing ({len(valid_picks_after)} channels, {n_dropped} dropped)"
                if n_dropped <= 5:  # Show names if not too many
                    title_text += f"\nDropped: {', '.join(sorted(dropped_names))}"
                axes[1].set_title(title_text, fontsize=10)
            else:
                axes[1].set_title(f"After Preprocessing ({len(valid_picks_after)} channels)")
        else:
            axes[1].text(0.5, 0.5, "No valid channels", ha="center", va="center",
                        transform=axes[1].transAxes)
            axes[1].set_title("After Preprocessing")

        # Synchronize y-axis limits AFTER both plots are created
        if len(valid_picks_before) > 0 and len(valid_picks_after) > 0:
            # Get current y-limits from both plots
            ylim_before = axes[0].get_ylim()
            ylim_after = axes[1].get_ylim()

            # Use the wider range
            y_min = min(ylim_before[0], ylim_after[0])
            y_max = max(ylim_before[1], ylim_after[1])

            # Apply to both
            axes[0].set_ylim(y_min, y_max)
            axes[1].set_ylim(y_min, y_max)

        fig.suptitle("Power Spectral Density Comparison", fontsize=12, y=1.02)
        fig.subplots_adjust(top=0.88, wspace=0.3)
        return fig

    def _plot_residual_artifacts(
        self,
        raw_before: mne.io.BaseRaw,
        raw_after: mne.io.BaseRaw,
        picks: np.ndarray,
        window_sec: float = 2.0
    ) -> plt.Figure:
        """
        Quantify remaining artifacts after preprocessing.

        Shows temporal density of artifacts before/after preprocessing to assess
        whether preprocessing was sufficient or if residual artifacts remain.

        Args:
            raw_before: Raw data before preprocessing
            raw_after: Raw data after preprocessing
            picks: Channel picks to analyze
            window_sec: Window size in seconds for artifact detection

        Returns:
            Matplotlib figure with three subplots:
            1. Peak-to-peak amplitude over time (before vs after)
            2. Amplitude distribution histograms
            3. Artifact density (artifacts per second)
        """
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])  # Temporal comparison
        ax2 = fig.add_subplot(gs[1, 0])  # Before histogram
        ax3 = fig.add_subplot(gs[1, 1])  # After histogram
        ax4 = fig.add_subplot(gs[2, :])  # Artifact density

        if len(picks) == 0:
            ax1.text(0.5, 0.5, "No valid channels", ha="center", va="center",
                    transform=ax1.transAxes)
            return fig

        sfreq = raw_after.info["sfreq"]
        window_samples = int(window_sec * sfreq)
        step_samples = window_samples // 2  # 50% overlap

        # Get data (use same time range for both)
        n_times = min(raw_before.n_times, raw_after.n_times)
        data_before = raw_before.get_data(picks=picks)[:, :n_times]
        data_after = raw_after.get_data(picks=picks)[:, :n_times]

        # Calculate sliding window peak-to-peak amplitude
        window_starts = range(0, n_times - window_samples, step_samples)
        ptp_before = []
        ptp_after = []
        time_centers = []

        for start in window_starts:
            end = start + window_samples

            # Peak-to-peak (max - min) across all channels in window
            ptp_b = (np.max(data_before[:, start:end]) - np.min(data_before[:, start:end])) * 1e6
            ptp_a = (np.max(data_after[:, start:end]) - np.min(data_after[:, start:end])) * 1e6

            ptp_before.append(ptp_b)
            ptp_after.append(ptp_a)
            time_centers.append((start + window_samples / 2) / sfreq)

        ptp_before = np.array(ptp_before)
        ptp_after = np.array(ptp_after)
        time_centers = np.array(time_centers)

        # Subplot 1: Temporal comparison
        ax1.plot(time_centers, ptp_before, label='Before Preprocessing',
                alpha=0.7, linewidth=1, color='red')
        ax1.plot(time_centers, ptp_after, label='After Preprocessing',
                alpha=0.7, linewidth=1, color='green')
        ax1.fill_between(time_centers, ptp_before, alpha=0.2, color='red')
        ax1.fill_between(time_centers, ptp_after, alpha=0.2, color='green')

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Peak-to-Peak Amplitude (µV)')
        ax1.set_title(f'Residual Artifact Assessment ({window_sec}s windows, 50% overlap)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Before histogram
        ax2.hist(ptp_before, bins=50, alpha=0.7, edgecolor='black', color='red')
        ax2.axvline(np.median(ptp_before), color='darkred', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(ptp_before):.1f} µV')
        ax2.axvline(np.percentile(ptp_before, 95), color='orange', linestyle='--', linewidth=1,
                   label=f'95th %ile: {np.percentile(ptp_before, 95):.1f} µV')
        ax2.set_xlabel('Peak-to-Peak Amplitude (µV)')
        ax2.set_ylabel('Count')
        ax2.set_title('Before Preprocessing')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')

        # Subplot 3: After histogram
        ax3.hist(ptp_after, bins=50, alpha=0.7, edgecolor='black', color='green')
        ax3.axvline(np.median(ptp_after), color='darkgreen', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(ptp_after):.1f} µV')
        ax3.axvline(np.percentile(ptp_after, 95), color='orange', linestyle='--', linewidth=1,
                   label=f'95th %ile: {np.percentile(ptp_after, 95):.1f} µV')
        ax3.set_xlabel('Peak-to-Peak Amplitude (µV)')
        ax3.set_ylabel('Count')
        ax3.set_title('After Preprocessing')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')

        # Subplot 4: Artifact density (artifacts per second)
        # Define artifact threshold as 95th percentile of preprocessed data
        threshold = np.percentile(ptp_after, 95)

        # Count artifacts in 10-second bins
        bin_size_sec = 10.0
        bin_size_samples = int(bin_size_sec * sfreq)
        n_bins = int(np.ceil(n_times / bin_size_samples))

        artifact_density_before = []
        artifact_density_after = []
        bin_centers = []

        for bin_idx in range(n_bins):
            bin_start = bin_idx * bin_size_samples
            bin_end = min((bin_idx + 1) * bin_size_samples, n_times)

            # Find which windows fall in this bin
            mask = (time_centers * sfreq >= bin_start) & (time_centers * sfreq < bin_end)

            if np.sum(mask) > 0:
                n_artifacts_before = np.sum(ptp_before[mask] > threshold)
                n_artifacts_after = np.sum(ptp_after[mask] > threshold)

                # Normalize by actual duration
                actual_duration = (bin_end - bin_start) / sfreq
                artifact_density_before.append(n_artifacts_before / actual_duration)
                artifact_density_after.append(n_artifacts_after / actual_duration)
                bin_centers.append((bin_start + (bin_end - bin_start) / 2) / sfreq)

        ax4.bar(bin_centers, artifact_density_before, width=bin_size_sec * 0.4,
               alpha=0.7, label='Before', color='red', align='edge')
        ax4.bar([x + bin_size_sec * 0.4 for x in bin_centers], artifact_density_after,
               width=bin_size_sec * 0.4, alpha=0.7, label='After', color='green', align='edge')

        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel(f'Artifacts per second (>{threshold:.0f} µV)')
        ax4.set_title(f'Artifact Density Over Time ({bin_size_sec}s bins)')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # Add summary statistics
        reduction_pct = 100 * (1 - np.median(ptp_after) / np.median(ptp_before))
        stats_text = (
            f'Median amplitude reduction: {reduction_pct:.1f}%\n'
            f'Before: {np.median(ptp_before):.1f} µV → After: {np.median(ptp_after):.1f} µV\n'
            f'Artifact windows reduced: {np.sum(ptp_before > threshold)} → {np.sum(ptp_after > threshold)}'
        )
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle('Residual Artifact Quantification', fontsize=14, fontweight='bold')
        return fig

    def _plot_erp_waveforms(self, raw: mne.io.BaseRaw) -> Optional[plt.Figure]:
        """
        Plot Event-Related Potential (ERP) waveforms if events/annotations are available.

        Creates epochs from the first event type with at least 10 trials and generates
        an ERP plot showing the grand average waveform across channels.

        Args:
            raw: Preprocessed MNE Raw object with annotations/events

        Returns:
            Matplotlib Figure with ERP waveform plot, or None if insufficient events
        """
        try:
            # Check if data has events/annotations
            events, event_id = mne.events_from_annotations(raw, verbose=False)

            if len(events) < 10:
                return None

            # Find first event type with >= 10 trials
            selected_event_name = None
            selected_event_code = None

            for event_name, event_code in event_id.items():
                n_trials = np.sum(events[:, 2] == event_code)
                if n_trials >= 10:
                    selected_event_name = event_name
                    selected_event_code = event_code
                    break

            if selected_event_name is None:
                return None

            # Create epochs
            # Baseline: -200 to 0 ms
            # Window: -200 to 800 ms
            epochs = mne.Epochs(
                raw,
                events,
                event_id={selected_event_name: selected_event_code},
                tmin=-0.2,
                tmax=0.8,
                baseline=(-0.2, 0),
                reject=None,  # No rejection for QC visualization
                preload=True,
                verbose=False,
            )

            if len(epochs) < 10:
                return None

            # Compute grand average ERP
            evoked = epochs.average()

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            # Get EEG data
            times = evoked.times
            data = evoked.get_data() * 1e6  # Convert to microvolts

            # Plot individual channel waveforms (light gray)
            for ch_idx in range(len(data)):
                ax.plot(times, data[ch_idx], color='lightgray', alpha=0.3, linewidth=0.5)

            # Plot grand average (mean across channels) in bold
            grand_avg = np.mean(data, axis=0)
            ax.plot(times, grand_avg, color='#2166AC', linewidth=2.5, label='Grand Average')

            # Add zero lines
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.axvline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='Stimulus Onset')

            # Labels and title
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Amplitude (µV)', fontsize=11)
            ax.set_title(f'Event-Related Potential (ERP)\nCondition: {selected_event_name} ({len(epochs)} trials)', fontsize=12)

            # Add legend
            ax.legend(loc='upper right', fontsize=9)

            # Grid
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)

            # Info text
            info_text = f"N channels: {len(data)}\nBaseline: -200 to 0 ms"
            ax.text(0.02, 0.98, info_text,
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            fig.tight_layout()
            return fig

        except Exception as e:
            # If anything fails, return None (no ERP plot)
            return None

    def _load_task_config(self, task_name: str) -> Optional[Dict[str, Any]]:
        """
        Load task-specific epoch configuration from YAML file.

        Args:
            task_name: Task name (e.g., 'contrastdetection', 'nback')

        Returns:
            Dictionary with epoch parameters (tmin, tmax, baseline, conditions) or None if not found
        """
        # Standard locations to search for task configs
        search_paths = [
            # 1. Relative to current working directory (most common for project configs)
            Path(f"configs/tasks/{task_name}.yaml"),
            Path(f"eegcpm/configs/tasks/{task_name}.yaml"),
            # 2. In the eegcpm package (for default/example configs)
            Path(__file__).parent.parent.parent / "configs" / "tasks" / f"{task_name}.yaml",
        ]

        for config_path in search_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        task_config = yaml.safe_load(f)

                    # Extract epoch parameters
                    epochs_params = {
                        'tmin': task_config.get('tmin', -0.2),
                        'tmax': task_config.get('tmax', 0.8),
                        'baseline': tuple(task_config.get('baseline', [-0.2, 0.0])),
                        'reject': task_config.get('reject'),
                        'decim': 1,
                        'conditions': task_config.get('conditions', []),  # Pass through conditions
                    }

                    print(f"Loaded task config from: {config_path}")
                    print(f"  tmin={epochs_params['tmin']}, tmax={epochs_params['tmax']}, baseline={epochs_params['baseline']}")
                    if epochs_params['conditions']:
                        print(f"  conditions: {[c['name'] for c in epochs_params['conditions']]}")

                    return epochs_params

                except Exception as e:
                    print(f"Warning: Failed to load task config from {config_path}: {e}")
                    continue

        print(f"Warning: No task config found for '{task_name}', using defaults")
        return None

    def _generate_erp_qc(
        self,
        raw: mne.io.BaseRaw,
        subject_id: str,
        events: np.ndarray,
        event_id: Dict[str, int],
        epochs_params: Optional[Dict[str, Any]] = None,
        erp_channels: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        task_name: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Generate Event-Related Potential quality control report.

        Creates condition-based ERP waveforms and topographic maps to validate
        preprocessing quality for task-based paradigms. This is essential for
        ensuring preprocessing parameters are appropriate before proceeding to
        connectivity analysis.

        Args:
            raw: Preprocessed MNE Raw object
            subject_id: Subject identifier
            events: Event array (N_events x 3) with [sample, 0, event_id]
            event_id: Dictionary mapping condition names to event codes
                Example: {'left_stimulus': 8, 'right_stimulus': 9}
            epochs_params: Optional epoching parameters
                Default: {'tmin': -0.2, 'tmax': 0.8, 'baseline': (-0.2, 0), 'reject': None}
            erp_channels: Optional list of channels to focus on (e.g., visual ROI)
            session_id: Optional session identifier
            task_name: Optional task name to load task-specific config

        Returns:
            Path to generated ERP QC HTML report, or None if generation failed

        Notes:
            - Automatically creates epochs for each condition in event_id
            - Generates evoked responses (averaged ERPs) by condition
            - Produces HTML report with waveforms, topomaps, and peak metrics
            - Uses MNE's built-in plotting for publication-quality figures
        """
        # Try to load task-specific config if task_name provided
        if task_name and epochs_params is None:
            epochs_params = self._load_task_config(task_name)

        # Default epoching parameters if no task config found
        if epochs_params is None:
            epochs_params = {
                'tmin': -0.2,
                'tmax': 0.8,
                'baseline': (-0.2, 0),
                'reject': None,  # No rejection for preprocessed data
                'decim': 1,      # No decimation to avoid aliasing
            }

        # Build condition -> event_codes mapping from task config
        # This allows merging multiple event codes into single conditions
        condition_to_codes = {}
        if 'conditions' in epochs_params and epochs_params['conditions']:
            for cond in epochs_params['conditions']:
                cond_name = cond['name']
                event_codes_list = cond.get('event_codes', [])

                # Resolve annotation names to numeric event codes
                numeric_codes = []
                for code in event_codes_list:
                    # code might be a string (annotation name) or int (event code)
                    if str(code) in event_id:
                        numeric_codes.append(event_id[str(code)])
                    elif isinstance(code, int) and code in event_id.values():
                        numeric_codes.append(code)

                if numeric_codes:
                    condition_to_codes[cond_name] = numeric_codes
        else:
            # No task config - map each event_id directly to a condition
            for annotation_name, event_code in event_id.items():
                condition_to_codes[annotation_name] = [event_code]

        # Extract epochs for each condition (merging event codes as needed)
        evokeds = {}

        for condition, event_codes_for_condition in condition_to_codes.items():
            # Collect all events for this condition (may span multiple event codes)
            condition_events = events[np.isin(events[:, 2], event_codes_for_condition)]

            if len(condition_events) == 0:
                print(f"Warning: No events found for condition '{condition}'")
                continue

            try:
                # Create epochs (merges all event codes for this condition)
                epochs = mne.Epochs(
                    raw,
                    condition_events,
                    tmin=epochs_params.get('tmin', -0.2),
                    tmax=epochs_params.get('tmax', 0.8),
                    baseline=epochs_params.get('baseline', (-0.2, 0)),
                    reject=epochs_params.get('reject'),
                    decim=epochs_params.get('decim', 1),
                    preload=True,
                    verbose=False,
                )

                # Check minimum epochs
                if len(epochs) < 5:
                    print(f"Warning: Only {len(epochs)} epochs for condition '{condition}' (minimum 5 recommended)")
                    continue

                # Compute evoked response
                evoked = epochs.average()
                evokeds[condition] = evoked

                print(f"  Created ERP for condition '{condition}': {len(epochs)} epochs from {len(event_codes_for_condition)} event code(s)")

            except Exception as e:
                print(f"Warning: Could not create epochs for condition '{condition}': {e}")
                continue

        # Generate ERP QC report if we have at least one condition
        if not evokeds:
            print("Warning: No valid evoked responses to generate ERP QC")
            return None

        try:
            # Initialize ERP QC generator
            erp_qc = ERPQC(self.output_dir)

            # Generate report
            report_path = erp_qc.generate_report(
                evokeds=evokeds,
                subject_id=subject_id,
                session_id=session_id,
                roi_channels=erp_channels,
                peak_windows=None,  # Use defaults (P1, N1, P3)
            )

            return report_path

        except Exception as e:
            print(f"Error generating ERP QC report: {e}")
            return None

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

        builder = HTMLReportBuilder(title=f"Preprocessed QC: {result.subject_id}")

        builder.add_header(f"Overall Status: {result.status.upper()}", level=2)

        # ========== Section 1: Preprocessing Metrics ==========
        builder.add_header("1. Preprocessing Metrics", level=2)
        builder.add_metrics_table(result.metrics)

        if result.notes:
            builder.add_notes(result.notes)

        # ========== Section 2: Before/After Preprocessing Comparison ==========
        if "before_after_metrics" in result.metadata:
            builder.add_header("2. Before/After Preprocessing Comparison", level=2)
            before_after_html = self._create_before_after_metrics_table(
                result.metadata["before_after_metrics"]
            )
            builder.add_raw_html(before_after_html)

        # ========== Diagnostic Plots ==========
        builder.add_header("Diagnostic Plots", level=2)

        # Define figure order following user's requested sequence
        figure_order = [
            # 3. Channel Layout
            ("bad_channels_topo", "3a. Channel Layout - Bad/removed channels topography (color-coded by reason)"),
            ("bad_channel_clustering", "3c. Channel Layout - Bad channel clustering analysis (spatial proximity)"),

            # 4. Power Spectral Density Comparison
            ("before_after", "4. Power Spectral Density - Before vs after preprocessing comparison"),

            # 4b. Residual Artifact Quantification
            ("residual_artifacts", "4b. Residual Artifact Quantification - Temporal artifact density before/after preprocessing"),

            # 5. Amplitude Distribution (after preprocessing)
            ("amplitude_dist", "5. Amplitude Distribution - Signal amplitude after preprocessing (should be roughly Gaussian)"),

            # 6. Channel Variance (preprocessed)
            ("channel_variance", "6. Channel Variance - Variance per channel after preprocessing"),

            # 7. Inter-Channel Correlation Matrix (preprocessed)
            ("correlation_matrix", "7. Inter-Channel Correlation Matrix - Check for bridged/disconnected channels (preprocessed data)"),

            # 8. Raw vs Preprocessed
            ("raw_vs_preprocessed_overlay", "8. Raw vs Preprocessed - Signal comparison (demeaned overlay)"),

            # 9. ICA Component Topography
            ("ica_topographies", "9. ICA Component Topography - Scalp maps showing spatial patterns"),

            # 10. ICA Component Timeseries
            ("ica_components", "10. ICA Component Timeseries - Component activations over time (excluded marked in red)"),

            # 10b. ICA Component Power Spectra
            ("ica_component_spectra", "10b. ICA Component Power Spectra - Frequency content by component (color-coded by ICLabel classification)"),

            # ERPs (if available)
            ("erp_waveforms", "Event-Related Potential (ERP) waveforms - Grand average across channels with baseline correction"),
        ]

        # Add figures in defined order
        for name, caption in figure_order:
            if name in result.figures:
                builder.add_figure(name, result.figures[name], caption)

        # ========== Section 11: ICA Components Table (with ICLabel Classification) ==========
        # Add ICA components table AFTER the ICA plots (sections 9 & 10)
        if "ica_components" in result.metadata:
            builder.add_header("11. ICA Components (with ICLabel Classification)", level=2)
            ica_table_html = self._create_ica_components_table(
                result.metadata["ica_components"]
            )
            builder.add_raw_html(ica_table_html)

        # Add link to ERP QC report if it was generated
        if "erp_qc_path" in result.metadata:
            erp_path = Path(result.metadata["erp_qc_path"])
            # Use file:// protocol for absolute path to work when opening HTML locally
            erp_file_url = erp_path.as_uri()
            builder.add_header("Event-Related Potential (ERP) Analysis", level=2)
            builder.add_raw_html(f"""
                <div style="background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <p style="margin: 5px 0;">
                        <strong>ERP QC Report Available:</strong>
                        <a href="{erp_file_url}" target="_blank"
                           style="color: #2980b9; text-decoration: none; font-weight: bold;">
                            View Condition-based ERP Analysis →
                        </a>
                    </p>
                    <p style="margin: 5px 0; font-size: 0.9em; color: #555;">
                        The ERP report shows averaged waveforms by experimental condition,
                        topographic maps at key latencies, and detected peak components (P1, N1, P3).
                        Review to validate preprocessing quality before proceeding to connectivity analysis.
                    </p>
                </div>
            """)

        # Add any remaining figures not in the predefined order
        for name, fig_bytes in result.figures.items():
            if name not in [fig_name for fig_name, _ in figure_order]:
                builder.add_figure(name, fig_bytes, name)

        html = builder.build()

        if save_path:
            Path(save_path).write_text(html)

        return html
