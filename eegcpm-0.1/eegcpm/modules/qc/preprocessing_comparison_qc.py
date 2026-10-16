"""
Preprocessing Pipeline Comparison QC Module

Generates comprehensive diagnostic plots comparing different preprocessing methods:
- Original (unprocessed) data
- EEGPrep 3-step pipeline (optional)
- EEGCPM Full pipeline

Produces side-by-side visualizations:
1. Channel locations (excluded channels marked)
2. Time series segments (first/middle/last third)
3. Statistical comparison (violin plots + statistics table)
4. Per-channel variance (log scale)
5. Artifactual segments comparison

Author: EEGCPM Development Team
Date: 2025-11-29
"""

import logging

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import mne

from eegcpm.modules.preprocessing.steps.asr import append_bad_annotations

from .base import BaseQC, QCResult

# Check for EEGPrep availability
try:
    import eegprep
    EEGPREP_AVAILABLE = True
except ImportError:
    EEGPREP_AVAILABLE = False


logger = logging.getLogger(__name__)


def append_eegprep_removed_mark(
    raw: mne.io.BaseRaw,
    onset_s: float,
    duration_s: float,
    description: str,
) -> None:
    """Append a ``BAD_eegprep_removed_*`` mark to a display copy (F6).

    ``raw_annotated`` is the original-duration *display* copy used by the
    comparison figure: it must keep every annotation the recording already
    carries (stim events, earlier ``BAD_*`` marks) and only **add** the
    removed-segment mark. The pre-F6 code called ``set_annotations(annot)``,
    which replaced the whole timeline and silently dropped those annotations
    from the figure. This helper appends via
    :func:`~eegcpm.modules.preprocessing.steps.asr.append_bad_annotations`,
    which applies the §0.3 offset recipe, so the mark lands ``onset_s``
    seconds after the first data sample whatever ``first_samp``/``orig_time``
    are. Display-only: the copy is never processed or saved.
    """
    append_bad_annotations(raw, [onset_s], [duration_s], [description])


class PreprocessingComparisonQC(BaseQC):
    """
    QC module for comparing preprocessing methods.

    Generates comprehensive diagnostic plots comparing:
    - Original unprocessed data
    - EEGPrep 3-step pipeline (if available)
    - EEGCPM full preprocessing pipeline

    Configuration options:
    - enable_eegprep: bool - Enable EEGPrep comparison (default: True if available)
    - eegprep_config_mild: dict - EEGPrep Step 1 config (mild ASR)
    - eegprep_config_aggressive: dict - EEGPrep Step 3 config (aggressive ASR)
    - figure_dpi: int - DPI for figure generation (default: 150)
    """

    # Default EEGPrep configurations
    DEFAULT_EEGPREP_MILD = {
        'FlatlineCriterion': 5,
        'Highpass': (0.25, 0.75),
        'BurstCriterion': 40,  # MILD
        'ChannelCriterion': 0.8,
        'WindowCriterion': 0.25,
        'LineNoiseCriterion': 4,
    }

    DEFAULT_EEGPREP_AGGRESSIVE = {
        'FlatlineCriterion': 'off',
        'Highpass': 'off',
        'BurstCriterion': 20,  # AGGRESSIVE
        'ChannelCriterion': 'off',
        'WindowCriterion': 0.25,
        'LineNoiseCriterion': 'off',
    }

    def __init__(self, output_dir: Path, config: Optional[Dict] = None):
        """Initialize preprocessing comparison QC module."""
        super().__init__(output_dir, config)

        self.enable_eegprep = self.config.get('enable_eegprep', EEGPREP_AVAILABLE)
        if self.enable_eegprep and not EEGPREP_AVAILABLE:
            print("Warning: EEGPrep comparison enabled but eegprep not installed")
            self.enable_eegprep = False

        self.eegprep_config_mild = self.config.get('eegprep_config_mild', self.DEFAULT_EEGPREP_MILD)
        self.eegprep_config_aggressive = self.config.get('eegprep_config_aggressive', self.DEFAULT_EEGPREP_AGGRESSIVE)
        self.figure_dpi = self.config.get('figure_dpi', 150)

    def compute(
        self,
        data: Dict[str, Any],
        subject_id: str,
        **kwargs
    ) -> QCResult:
        """
        Compute preprocessing comparison QC metrics and generate diagnostic plots.

        Args:
            data: Dictionary with keys:
                - 'raw_original': Original unprocessed MNE Raw object
                - 'raw_processed': Processed MNE Raw object (full pipeline)
            subject_id: Subject identifier
            **kwargs: Additional arguments (task, session, etc.)

        Returns:
            QCResult with metrics and embedded figures
        """
        raw_original = data['raw_original']
        raw_full = data['raw_processed']
        task = kwargs.get('task', 'unknown')

        result = QCResult(subject_id=subject_id)

        # Process with EEGPrep if enabled
        raw_eegprep_processed = None
        raw_eegprep_annotated = None
        stats_eegprep = {}

        if self.enable_eegprep:
            try:
                raw_eegprep_processed, raw_eegprep_annotated, stats_eegprep = \
                    self._preprocess_with_eegprep(raw_original.copy())
                result.add_note("EEGPrep comparison completed successfully")
            except Exception as e:
                result.add_note(f"EEGPrep comparison failed: {e}")
                print(f"Warning: EEGPrep processing failed for {subject_id}: {e}")

        # Generate diagnostic plot
        fig = self._generate_diagnostic_plot(
            raw_original=raw_original,
            raw_eegprep_processed=raw_eegprep_processed,
            raw_eegprep_annotated=raw_eegprep_annotated,
            raw_full=raw_full,
            subject_id=subject_id,
            task=task
        )

        # Convert figure to bytes
        fig_bytes = self.fig_to_base64(fig, dpi=self.figure_dpi)
        result.add_figure('preprocessing_comparison', fig_bytes)
        plt.close(fig)

        # Add comparison metrics
        if stats_eegprep:
            result.metadata['eegprep_stats'] = stats_eegprep

        return result

    def _preprocess_with_eegprep(
        self,
        raw_mne: mne.io.Raw
    ) -> Tuple[mne.io.Raw, mne.io.Raw, Dict]:
        """
        Apply EEGPrep 3-step preprocessing.

        Returns:
            - raw_processed: Processed data (shorter duration, filtered, ICA'd)
            - raw_annotated: Original-duration data with bad channels/segments marked
            - stats: Processing statistics
        """
        raw_original = raw_mne.copy()

        stats = {
            'n_channels_before': len(raw_mne.ch_names),
            'duration_s': raw_mne.times[-1],
            'sfreq': raw_mne.info['sfreq'],
        }

        # Convert MNE -> EEGPrep
        eeg = eegprep.eeg_mne2eeg(raw_mne)
        original_channels = [ch['labels'] for ch in eeg['chanlocs']]

        # STEP 1: Mild ASR
        result = eegprep.clean_artifacts(eeg, **self.eegprep_config_mild)
        eeg_mild = result[0]

        channels_after_step1 = [ch['labels'] for ch in eeg_mild['chanlocs']]
        removed_channels_step1 = list(set(original_channels) - set(channels_after_step1))

        # STEP 1.5: Estimate rank for ICA
        data = eeg_mild['data']
        cov_matrix = np.cov(data)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        threshold = 1e-6 * eigenvalues[0]
        effective_rank_initial = np.sum(eigenvalues > threshold)
        effective_rank = max(int(effective_rank_initial * 0.8), 20)

        stats['effective_rank_initial'] = effective_rank_initial
        stats['effective_rank_reduced'] = effective_rank

        # STEP 2: ICA
        try:
            eeg_ica = eegprep.eeg_picard(eeg_mild, n_components=effective_rank)
            stats['ica_algorithm'] = 'picard'
        except Exception:
            try:
                eeg_ica = eegprep.pop_runica(eeg_mild, icatype='runica', extended=1, pca=effective_rank)
                stats['ica_algorithm'] = 'runica'
            except Exception:
                stats['ica_algorithm'] = 'failed'
                eeg_ica = eeg_mild

        # STEP 2b: ICLabel
        if 'icaweights' in eeg_ica and eeg_ica['icaweights'] is not None:
            stats['ica_components'] = eeg_ica['icaweights'].shape[0]
            eeg_labeled = eegprep.iclabel(eeg_ica)

            if 'etc' in eeg_labeled and 'ic_classification' in eeg_labeled['etc']:
                ic_class = eeg_labeled['etc']['ic_classification']
                labels = ic_class['ICLabel']['classifications']

                # Identify artifacts (Eye/Muscle >80%)
                threshold = 0.8
                artifact_comps = []
                for idx in range(labels.shape[0]):
                    if labels[idx, 2] > threshold or labels[idx, 1] > threshold:
                        artifact_comps.append(idx)

                stats['ica_components_removed'] = len(artifact_comps)

                try:
                    eeg_after_ica = eegprep.pop_subcomp(eeg_labeled, artifact_comps)
                except Exception:
                    eeg_after_ica = eeg_labeled
            else:
                stats['ica_components_removed'] = 0
                eeg_after_ica = eeg_labeled
        else:
            stats['ica_components'] = 0
            stats['ica_components_removed'] = 0
            eeg_after_ica = eeg_mild

        # STEP 3: Aggressive ASR
        result = eegprep.clean_artifacts(eeg_after_ica, **self.eegprep_config_aggressive)
        eeg_final = result[0]

        # Convert back to MNE
        data = eeg_final['data'] * 1e-6  # µV → V
        sfreq = eeg_final['srate']
        ch_names = [ch['labels'] for ch in eeg_final['chanlocs']]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw_processed = mne.io.RawArray(data, info, verbose=False)

        # Set montage. A montage failure changes every downstream topography,
        # so it is recorded (and surfaces as a missing-position montage) rather
        # than silently skipped (P1-F11).
        try:
            montage = mne.channels.make_dig_montage(
                ch_pos={ch['labels']: [ch['X'], ch['Y'], ch['Z']]
                        for ch in eeg_final['chanlocs'] if 'X' in ch},
                coord_frame='head'
            )
            raw_processed.set_montage(montage, on_missing='warn')
        except (ValueError, KeyError, RuntimeError) as exc:
            logger.warning(
                "preprocessing comparison: could not set montage from the "
                "EEGLAB chanlocs (%s); topographies will be unavailable", exc)

        # Create annotated version
        final_channels = [ch['labels'] for ch in eeg_final['chanlocs']]
        all_removed_channels = list(set(original_channels) - set(final_channels))

        raw_annotated = raw_original.copy()
        if all_removed_channels:
            raw_annotated.info['bads'] = all_removed_channels

        # Annotate removed data segments
        duration_before = stats['duration_s']
        duration_after = raw_processed.times[-1]
        data_removed_pct = ((duration_before - duration_after) / duration_before) * 100

        if duration_after < duration_before:
            onset = duration_after
            duration = duration_before - duration_after
            description = f"BAD_eegprep_removed_{data_removed_pct:.1f}pct"
            # F6 (final-verification-audit fix round): APPEND to the display
            # copy's annotations (append-only invariant). `set_annotations`
            # replaced the copy's whole timeline — any stim/event annotations
            # were dropped from the comparison figure.
            append_eegprep_removed_mark(
                raw_annotated, onset, duration, description
            )

        stats['removed_channels'] = all_removed_channels
        stats['data_removed_seconds'] = duration_before - duration_after
        stats['data_removed_pct'] = data_removed_pct

        return raw_processed, raw_annotated, stats

    def _generate_diagnostic_plot(
        self,
        raw_original: mne.io.Raw,
        raw_eegprep_processed: Optional[mne.io.Raw],
        raw_eegprep_annotated: Optional[mne.io.Raw],
        raw_full: mne.io.Raw,
        subject_id: str,
        task: str
    ) -> plt.Figure:
        """Generate comprehensive diagnostic comparison plot."""

        fig = plt.figure(figsize=(20, 20))
        gs = GridSpec(5, 3, figure=fig, hspace=0.35, wspace=0.3)

        fig.suptitle(f'{subject_id} - {task}\nPreprocessing Comparison',
                     fontsize=16, fontweight='bold')

        # Row 1: Channel Locations
        ax = fig.add_subplot(gs[0, 0])
        self._plot_channel_locations(raw_original, ax, "Original (128 channels)")

        if raw_eegprep_processed is not None:
            ax = fig.add_subplot(gs[0, 1])
            excluded_eegprep = list(set(raw_original.ch_names) - set(raw_eegprep_processed.ch_names))
            self._plot_channel_locations(
                raw_original, ax,
                f"EEGPrep ({len(raw_eegprep_processed.ch_names)} kept, {len(excluded_eegprep)} excluded)",
                excluded_channels=excluded_eegprep
            )

        ax = fig.add_subplot(gs[0, 2])
        excluded_full = list(set(raw_original.ch_names) - set(raw_full.ch_names))
        self._plot_channel_locations(
            raw_original, ax,
            f"Full Pipeline ({len(raw_full.ch_names)} kept, {len(excluded_full)} excluded)",
            excluded_channels=excluded_full
        )

        # Row 2: Time Series Segments
        sfreq = raw_original.info['sfreq']
        duration = raw_original.times[-1]

        t_start_1 = duration / 6
        self._plot_time_series_segment(fig, gs[1, 0], raw_original, raw_eegprep_processed,
                                       raw_full, t_start_1, 10, sfreq, "First Third")

        t_start_2 = duration / 2 - 5
        self._plot_time_series_segment(fig, gs[1, 1], raw_original, raw_eegprep_processed,
                                       raw_full, t_start_2, 10, sfreq, "Middle Third")

        t_start_3 = 5 * duration / 6
        self._plot_time_series_segment(fig, gs[1, 2], raw_original, raw_eegprep_processed,
                                       raw_full, t_start_3, 10, sfreq, "Last Third")

        # Row 3: Statistical Comparison
        ax = fig.add_subplot(gs[2, :2])
        self._plot_statistical_comparison(raw_original, raw_eegprep_processed, raw_full, ax)

        ax = fig.add_subplot(gs[2, 2])
        self._plot_variance_comparison(raw_original, raw_eegprep_processed, raw_full, ax)

        # Row 4: Artifactual Segments - Full Pipeline
        ax = fig.add_subplot(gs[3, :])
        self._plot_artifact_segments(ax, raw_original, raw_full, "Full Pipeline")

        # Row 5: Artifactual Segments - EEGPrep
        if raw_eegprep_annotated is not None:
            ax = fig.add_subplot(gs[4, :])
            self._plot_artifact_segments(ax, raw_original, raw_eegprep_annotated, "EEGPrep")

        return fig

    # All plotting helper methods from the script
    # (Copied from preprocess_comparison_with_diagnostics.py)

    def _plot_channel_locations(self, raw, ax, title, excluded_channels=None):
        """Plot channel locations with excluded channels marked."""
        montage = raw.get_montage()
        if montage is None:
            ax.text(0.5, 0.5, 'No montage available', ha='center', va='center')
            ax.set_title(title)
            return

        pos = montage.get_positions()
        ch_pos = pos['ch_pos']

        for ch_name in raw.ch_names:
            if ch_name not in ch_pos:
                continue

            xyz = ch_pos[ch_name]
            x, y = xyz[0], xyz[1]

            if excluded_channels and ch_name in excluded_channels:
                ax.plot(x, y, 'rx', markersize=8, markeredgewidth=2)
            else:
                ax.plot(x, y, 'bo', markersize=5)

        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

        if excluded_channels:
            kept_patch = mpatches.Patch(color='blue', label='Kept')
            excl_patch = mpatches.Patch(color='red', label='Excluded')
            ax.legend(handles=[kept_patch, excl_patch], loc='upper right', fontsize=8)

    def _plot_time_series_segment(self, fig, gs_pos, raw_orig, raw_eegprep, raw_full,
                                   t_start, duration, sfreq, segment_name):
        """Plot 10s time series segment for comparison (demeaned)."""
        ax = fig.add_subplot(gs_pos)

        i_start = int(t_start * sfreq)
        i_end = int((t_start + duration) * sfreq)

        common_channels = list(set(raw_orig.ch_names) & set(raw_full.ch_names))
        if raw_eegprep is not None:
            common_channels = list(set(common_channels) & set(raw_eegprep.ch_names))

        plot_channels = common_channels[:10]

        orig_data = raw_orig.copy().pick_channels(plot_channels).get_data()[:, i_start:i_end] * 1e6
        orig_data = orig_data - orig_data.mean(axis=1, keepdims=True)

        full_data = raw_full.copy().pick_channels(plot_channels).get_data()[:, i_start:i_end] * 1e6
        full_data = full_data - full_data.mean(axis=1, keepdims=True)

        eegprep_data = None
        eegprep_times = None
        if raw_eegprep is not None:
            eegprep_sfreq = raw_eegprep.info['sfreq']
            eegprep_duration = raw_eegprep.times[-1]

            if t_start < eegprep_duration:
                eegprep_i_start = int(min(t_start, eegprep_duration - duration) * eegprep_sfreq)
                eegprep_i_end = int(min(eegprep_i_start + duration * eegprep_sfreq, raw_eegprep.n_times))

                if eegprep_i_end > eegprep_i_start:
                    eegprep_data = raw_eegprep.copy().pick_channels(plot_channels).get_data()[:, eegprep_i_start:eegprep_i_end] * 1e6
                    if eegprep_data.shape[1] > 0:
                        eegprep_data = eegprep_data - eegprep_data.mean(axis=1, keepdims=True)
                        eegprep_times = np.arange(eegprep_data.shape[1]) / eegprep_sfreq
                    else:
                        eegprep_data = None

        times = np.arange(orig_data.shape[1]) / sfreq

        offset = 100  # µV
        for i in range(len(plot_channels)):
            y_offset = i * offset
            ax.plot(times, orig_data[i, :] + y_offset, 'gray', alpha=0.5, lw=0.5, label='Original' if i == 0 else '')
            ax.plot(times, full_data[i, :] + y_offset, 'green', lw=0.7, label='Full' if i == 0 else '')

            if eegprep_data is not None and eegprep_times is not None:
                ax.plot(eegprep_times, eegprep_data[i, :] + y_offset, 'blue', lw=0.7, label='EEGPrep' if i == 0 else '')

        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Channel', fontsize=9)
        ax.set_title(f'{segment_name} ({t_start:.0f}-{t_start+duration:.0f}s, demeaned)', fontsize=10, fontweight='bold')
        ax.set_yticks([])
        if plt.gca() == ax:
            ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)

    def _get_good_data(self, raw, channels):
        """Get data excluding bad annotations and bad channels."""
        picks = [ch for ch in channels if ch not in raw.info['bads']]
        if not picks:
            return np.array([])

        raw_copy = raw.copy().pick_channels(picks)

        if raw_copy.annotations is None or len(raw_copy.annotations) == 0:
            return raw_copy.get_data().flatten() * 1e6

        sfreq = raw_copy.info['sfreq']
        n_samples = raw_copy.n_times
        good_mask = np.ones(n_samples, dtype=bool)

        for annot in raw_copy.annotations:
            if 'bad' in annot['description'].lower():
                onset_sample = int(annot['onset'] * sfreq)
                duration_samples = int(annot['duration'] * sfreq)
                end_sample = min(onset_sample + duration_samples, n_samples)
                good_mask[onset_sample:end_sample] = False

        data = raw_copy.get_data() * 1e6
        good_data = data[:, good_mask]

        return good_data.flatten()

    def _plot_statistical_comparison(self, raw_orig, raw_eegprep, raw_full, ax):
        """Violin plot statistical comparison with separate scales and statistics table."""
        common_channels = list(set(raw_orig.ch_names) & set(raw_full.ch_names))
        if raw_eegprep is not None:
            common_channels = list(set(common_channels) & set(raw_eegprep.ch_names))

        orig_data = self._get_good_data(raw_orig, common_channels)
        full_data = self._get_good_data(raw_full, common_channels)

        data_to_plot = [orig_data, full_data]
        labels = ['Original', 'Full']
        colors = ['gray', 'green']

        if raw_eegprep is not None:
            eegprep_data = self._get_good_data(raw_eegprep, common_channels)
            data_to_plot.insert(1, eegprep_data)
            labels.insert(1, 'EEGPrep')
            colors.insert(1, 'blue')

        inner_gs = GridSpecFromSubplotSpec(1, 4, subplot_spec=ax.get_subplotspec(),
                                          width_ratios=[1, 1, 1, 1.2], wspace=0.3)
        ax.remove()
        axs = [plt.gcf().add_subplot(inner_gs[i]) for i in range(4)]

        for i, (data, label, color) in enumerate(zip(data_to_plot, labels, colors)):
            parts = axs[i].violinplot([data], positions=[0], showmeans=True,
                                      showmedians=True, widths=0.7)

            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.7)

            y_min = np.percentile(data, 0.5)
            y_max = np.percentile(data, 99.5)
            axs[i].set_ylim([y_min * 1.1, y_max * 1.1])

            axs[i].set_xticks([0])
            axs[i].set_xticklabels([label], fontsize=9)
            axs[i].set_ylabel('Amplitude (µV)', fontsize=9)
            axs[i].grid(True, alpha=0.3, axis='y')
            axs[i].tick_params(axis='y', labelsize=8)

        # Statistics table
        stats_ax = axs[3]
        stats_ax.axis('off')

        stats_table = []
        if raw_eegprep:
            stats_table.append(['Metric', 'Original', 'EEGPrep', 'Full'])
        else:
            stats_table.append(['Metric', 'Original', 'Full'])

        for stat_name, stat_func in [('Mean', np.mean), ('Median', np.median),
                                       ('Std', np.std), ('Min', np.min), ('Max', np.max),
                                       ('P95', lambda x: np.percentile(x, 95)),
                                       ('P99', lambda x: np.percentile(x, 99))]:
            row = [stat_name]
            for data in data_to_plot:
                val = stat_func(data)
                row.append(f'{val:.2f}')
            stats_table.append(row)

        # N samples row
        n_samples_row = ['N samples', f'{len(orig_data):,}']
        if raw_eegprep:
            n_samples_row.append(f'{len(data_to_plot[1]):,}')
        n_samples_row.append(f'{len(full_data):,}')
        stats_table.append(n_samples_row)

        # Adjust column widths based on number of columns
        if raw_eegprep:
            col_widths = [0.3, 0.23, 0.23, 0.23]
        else:
            col_widths = [0.4, 0.3, 0.3]

        table = stats_ax.table(cellText=stats_table, loc='center', cellLoc='left',
                              colWidths=col_widths)
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.5)

        for i in range(len(stats_table[0])):
            table[(0, i)].set_facecolor('#E0E0E0')
            table[(0, i)].set_text_props(weight='bold')

        for i in range(1, len(stats_table)):
            table[(i, 1)].set_facecolor('#F5F5F5')
            if raw_eegprep:
                table[(i, 2)].set_facecolor('#E3F2FD')
                table[(i, 3)].set_facecolor('#E8F5E9')
            else:
                table[(i, 2)].set_facecolor('#E8F5E9')

        stats_ax.set_title('Statistics (µV)', fontsize=9, fontweight='bold', pad=10)

    def _plot_variance_comparison(self, raw_orig, raw_eegprep, raw_full, ax):
        """Plot variance comparison across channels (grouped bar plot with log scale)."""
        all_orig_channels = sorted(raw_orig.ch_names)[:20]

        orig_var = raw_orig.copy().pick_channels(all_orig_channels).get_data().var(axis=1) * 1e12

        full_var = np.zeros(len(all_orig_channels))
        for i, ch in enumerate(all_orig_channels):
            if ch in raw_full.ch_names:
                full_var[i] = raw_full.copy().pick_channels([ch]).get_data().var() * 1e12
            else:
                full_var[i] = np.nan

        eegprep_var = np.zeros(len(all_orig_channels))
        if raw_eegprep is not None:
            for i, ch in enumerate(all_orig_channels):
                if ch in raw_eegprep.ch_names:
                    eegprep_var[i] = raw_eegprep.copy().pick_channels([ch]).get_data().var() * 1e12
                else:
                    eegprep_var[i] = np.nan

        x = np.arange(len(all_orig_channels))
        width = 0.28

        # Replace zeros with small value to avoid log(0) issues
        orig_var_plot = np.where(orig_var > 0, orig_var, 0.01)
        full_var_plot = np.where(full_var > 0, full_var, 0.01)
        eegprep_var_plot = np.where(eegprep_var > 0, eegprep_var, 0.01) if raw_eegprep is not None else None

        ax.bar(x - width, orig_var_plot, width, label='Original', color='gray', alpha=0.7)
        ax.bar(x, full_var_plot, width, label='Full', color='green', alpha=0.7)
        if raw_eegprep is not None:
            ax.bar(x + width, eegprep_var_plot, width, label='EEGPrep', color='blue', alpha=0.7)

        ax.set_xlabel('Channel', fontsize=9)
        ax.set_ylabel('Variance (µV²) [log scale]', fontsize=9)
        ax.set_title('Per-Channel Variance', fontsize=10, fontweight='bold')
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(all_orig_channels, rotation=45, ha='right', fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y', which='both')

    def _plot_artifact_segments(self, ax, raw_orig, raw_processed, method_name):
        """Plot artifact segments for a single method."""
        if raw_processed.annotations is not None and len(raw_processed.annotations) > 0:
            annots = raw_processed.annotations
            bad_segments = [(annot['onset'], annot['onset'] + annot['duration'])
                           for annot in annots if 'bad' in annot['description'].lower()]
        else:
            bad_segments = []

        sfreq_orig = raw_orig.info['sfreq']
        sfreq_proc = raw_processed.info['sfreq']
        window = int(1 * sfreq_orig)

        orig_rms = self._compute_rms_trace(raw_orig, window)

        window_proc = int(1 * sfreq_proc)
        processed_rms = self._compute_rms_trace(raw_processed, window_proc)

        times_orig = np.arange(len(orig_rms)) * window / sfreq_orig
        times_proc = np.arange(len(processed_rms)) * window_proc / sfreq_proc

        ax.plot(times_orig, orig_rms, 'gray', alpha=0.6, label='Original', lw=1)

        color = 'green' if 'Full' in method_name else 'blue'
        ax.plot(times_proc, processed_rms, color=color, label=method_name, lw=1)

        max_time_proc = times_proc[-1] if len(times_proc) > 0 else 0
        for onset, offset in bad_segments:
            if onset < max_time_proc:
                ax.axvspan(onset, min(offset, max_time_proc), color='red', alpha=0.2)

        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('RMS Amplitude (µV)', fontsize=10)
        ax.set_title(f'{method_name}: Artifact Segments (red = marked bad)', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    def _compute_rms_trace(self, raw, window):
        """Compute RMS trace over time using sliding window, excluding bad channels."""
        picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude='bads')
        data = raw.get_data(picks=picks) * 1e6
        n_samples = data.shape[1]
        n_windows = (n_samples - window) // window + 1

        rms = np.zeros(n_windows)
        for i in range(n_windows):
            start = i * window
            end = start + window
            if end > n_samples:
                end = n_samples
            rms[i] = np.sqrt(np.mean(data[:, start:end] ** 2))

        return rms
