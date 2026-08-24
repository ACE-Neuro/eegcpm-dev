"""Epochs Quality Control Report Generation.

Unified QC report generation for epochs, used by both interactive and batch modes.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import base64
import io
import numpy as np
import mne
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_epochs_qc_report(
    epochs: mne.Epochs,
    task_config: Dict[str, Any],
    output_path: Path,
    runs_included: Optional[List[str]] = None,
    runs_excluded: Optional[List[str]] = None,
    subject_id: str = "unknown",
    session: str = "01",
    task: str = "unknown",
) -> Path:
    """
    Generate standardized HTML QC report for epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data
    task_config : dict
        Task configuration used for epoching
    output_path : Path
        Output HTML file path
    runs_included : list, optional
        List of run IDs that were included
    runs_excluded : list, optional
        List of run IDs that were excluded
    subject_id : str
        Subject identifier
    session : str
        Session identifier
    task : str
        Task name

    Returns
    -------
    Path
        Path to generated HTML report
    """

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect QC metrics
    n_epochs = len(epochs)
    n_conditions = len(epochs.event_id)
    drop_log_stats = epochs.drop_log_stats()

    # Per-condition trial counts
    condition_counts = {}
    for condition in epochs.event_id:
        if condition in epochs.event_id:
            condition_counts[condition] = len(epochs[condition])

    # Generate HTML report
    html = _generate_html_template(
        subject_id=subject_id,
        session=session,
        task=task,
        task_config=task_config,
        n_epochs=n_epochs,
        n_conditions=n_conditions,
        condition_counts=condition_counts,
        drop_log_stats=drop_log_stats,
        runs_included=runs_included,
        runs_excluded=runs_excluded,
        epochs=epochs,
    )

    # Write to file
    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def _generate_html_template(
    subject_id: str,
    session: str,
    task: str,
    task_config: Dict[str, Any],
    n_epochs: int,
    n_conditions: int,
    condition_counts: Dict[str, int],
    drop_log_stats: float,
    runs_included: Optional[List[str]],
    runs_excluded: Optional[List[str]],
    epochs: mne.Epochs,
) -> str:
    """Generate HTML template for epochs QC report."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build run info section
    run_info_html = ""
    if runs_included:
        run_info_html += f"<p><strong>Runs Included:</strong> {', '.join(runs_included)}</p>\n"
    if runs_excluded:
        run_info_html += f"<p><strong>Runs Excluded:</strong> {', '.join(runs_excluded)}</p>\n"

    # Build condition table
    condition_rows = ""
    for condition, count in sorted(condition_counts.items()):
        condition_rows += f"""
        <tr>
            <td>{condition}</td>
            <td>{count}</td>
            <td>{count / n_epochs * 100:.1f}%</td>
        </tr>
        """

    # Task config details
    config_details = f"""
    <p><strong>Time Window:</strong> {task_config.get('tmin', 'N/A')} to {task_config.get('tmax', 'N/A')} s</p>
    <p><strong>Baseline:</strong> {task_config.get('baseline', 'N/A')}</p>
    """

    if 'conditions' in task_config:
        config_details += "<p><strong>Conditions Defined:</strong></p><ul>"
        for cond in task_config['conditions']:
            config_details += f"<li>{cond.get('name', 'Unknown')}: {cond.get('event_codes', [])}</li>"
        config_details += "</ul>"

    # Generate ERP plots as base64 images
    erp_plots_html = _generate_erp_plots_html(epochs, condition_counts.keys())

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Epochs QC Report - {subject_id}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px;
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 5px;
            min-width: 150px;
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .info-box {{
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
        }}
        .success {{
            background-color: #d4edda;
            border-left: 4px solid #28a745;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 12px;
            text-align: right;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Epochs Quality Control Report</h1>

        <div class="timestamp">Generated: {timestamp}</div>

        <h2>Subject Information</h2>
        <p><strong>Subject:</strong> {subject_id}</p>
        <p><strong>Session:</strong> {session}</p>
        <p><strong>Task:</strong> {task}</p>
        {run_info_html}

        <h2>Epoch Summary</h2>
        <div class="metric">
            <div class="metric-label">Total Epochs</div>
            <div class="metric-value">{n_epochs}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Conditions</div>
            <div class="metric-value">{n_conditions}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Drop Rate</div>
            <div class="metric-value">{drop_log_stats:.1f}%</div>
        </div>

        <h2>Task Configuration</h2>
        <div class="info-box">
            {config_details}
        </div>

        <h2>Per-Condition Trial Counts</h2>
        <table>
            <thead>
                <tr>
                    <th>Condition</th>
                    <th>Trial Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
                {condition_rows}
            </tbody>
        </table>

        <h2>Event-Related Potentials</h2>
        {erp_plots_html}

        <h2>Rejection Statistics</h2>
        <div class="info-box">
            <p><strong>Drop Rate:</strong> {drop_log_stats:.1f}% of epochs rejected</p>
            <p><strong>Criteria:</strong> {epochs.reject if hasattr(epochs, 'reject') else 'N/A'}</p>
        </div>

        <div class="info-box success">
            <p><strong>✓ Quality Check Complete</strong></p>
            <p>Epochs extracted successfully. Review ERP plots and trial counts above to verify data quality.</p>
        </div>
    </div>
</body>
</html>
    """

    return html


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64-encoded PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


def _spatial_colors(locs: np.ndarray) -> np.ndarray:
    """Map 3D channel positions to RGB colors (same scheme as MNE spatial_colors)."""
    rgb = np.array(locs, dtype=float)
    rgb -= np.nanmin(rgb, axis=0)
    rgb /= np.maximum(np.nanmax(rgb, axis=0), 1e-16)  # avoid div by zero
    # Reduce RGB intensity for overly light colors
    mask = rgb.sum(axis=1) > 2.5
    rgb[mask] = rgb[mask] - 0.3
    return rgb


def _plot_condition_butterfly(evoked: mne.Evoked, condition: str, n_trials: int) -> plt.Figure:
    """Plot ERP butterfly (all channels) with a sensor-location head inset.

    Traces and sensor spots are colored by channel position (MNE
    spatial_colors scheme), so each trace matches its head spot.
    """
    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 4], wspace=0.15)

    # Per-channel colors from 3D positions
    eeg_picks = mne.pick_types(evoked.info, eeg=True)
    eeg_names = [evoked.ch_names[i] for i in eeg_picks]
    locs = np.array([evoked.info['chs'][i]['loc'][:3] for i in eeg_picks])
    colors = _spatial_colors(locs)
    color_by_name = dict(zip(eeg_names, colors))

    # Sensor-location head with the project EEGLAB-style outline
    from eegcpm.modules.qc.preprocessed_qc import _draw_head_outline, _load_neuroscan_locs

    locs_pos = _load_neuroscan_locs()
    ax_head = fig.add_subplot(gs[0])
    _draw_head_outline(ax_head, head_radius=0.5, linewidth=1.5)
    for name in eeg_names:
        if name in locs_pos:
            x, y = locs_pos[name]
            ax_head.plot(x, y, 'o', markersize=3.5, color=color_by_name[name])
    ax_head.set_xlim(-0.65, 0.65)
    ax_head.set_ylim(-0.65, 0.72)
    ax_head.set_aspect('equal')
    ax_head.axis('off')

    # Butterfly traces (EEG channels only, in µV), colored by position
    ax = fig.add_subplot(gs[1])
    data_uv = evoked.data[eeg_picks] * 1e6
    for i, name in enumerate(eeg_names):
        ax.plot(evoked.times, data_uv[i], color=color_by_name[name], linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_xlim(evoked.times[0], evoked.times[-1])
    ax.set_title(f'ERP: {condition} ({n_trials} trials, butterfly, {len(eeg_picks)} channels)',
                 fontsize=11)
    return fig


def _generate_erp_plots_html(epochs: mne.Epochs, conditions) -> str:
    """Generate HTML with ERP plots for each condition.

    Creates a butterfly plot per condition plus an overlaid condition
    comparison plot, embedded as base64 PNG images.
    """
    valid_conditions = [c for c in conditions if c in epochs.event_id and len(epochs[c]) > 0]

    # Trial-count summary (kept as caption above the plots)
    plots_html = '<div class="info-box">'
    plots_html += '<p><strong>Trials per condition:</strong></p>'
    plots_html += '<ul>'
    for condition in valid_conditions:
        evoked = epochs[condition].average()
        times = evoked.times
        plots_html += (
            f'<li><strong>{condition}</strong>: {len(epochs[condition])} trials, '
            f'{len(evoked.ch_names)} channels, {times[0]:.2f} to {times[-1]:.2f} s</li>'
        )
    plots_html += '</ul>'
    plots_html += '</div>'

    if not valid_conditions:
        return plots_html

    # Per-condition butterfly plots (custom drawing so the sensor head
    # matches the project EEGLAB-style outline used in preprocessed QC)
    for condition in valid_conditions:
        evoked = epochs[condition].average()
        try:
            fig = _plot_condition_butterfly(evoked, condition, len(epochs[condition]))
            plots_html += (
                f'<h3>{condition}</h3>'
                f'<img src="{_fig_to_base64(fig)}" style="max-width:100%;">'
            )
        except Exception as e:
            plots_html += f'<p><em>Could not plot {condition}: {e}</em></p>'

    # Condition comparison plot (overlaid)
    if len(valid_conditions) > 1:
        try:
            evokeds = {c: epochs[c].average() for c in valid_conditions}
            fig = mne.viz.plot_compare_evokeds(evokeds, show=False)
            figs = fig if isinstance(fig, list) else [fig]
            # Move legends outside the axes so they don't overlap the traces
            for f in figs:
                for ax in f.axes:
                    leg = ax.get_legend()
                    if leg is not None:
                        leg.set_loc('upper left')
                        leg.set_bbox_to_anchor((1.02, 1.0))
                        leg.set_frame_on(True)
            for i, f in enumerate(figs):
                title = 'Condition comparison' if len(figs) == 1 else \
                    f'Condition comparison (channel group {i + 1})'
                plots_html += (
                    f'<h3>{title}</h3>'
                    f'<img src="{_fig_to_base64(f)}" style="max-width:100%;">'
                )
        except Exception as e:
            plots_html += f'<p><em>Could not generate comparison plot: {e}</em></p>'

    return plots_html
