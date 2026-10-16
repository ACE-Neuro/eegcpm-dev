"""Quality assessment logic for QC metrics.

This module provides functions to assess quality status and extract metrics
from QC results for JSON export.

JSON Schema
-----------
{
  "subject_id": "NDARAA306NT2",
  "session": "01",
  "task": "contdet",
  "run": "1",
  "pipeline": "standard",
  "timestamp": "2025-12-09T15:42:00",
  "quality_metrics": {
    "quality_status": "good",  // excellent/good/acceptable/poor
    "pct_bad_channels": 8.5,
    "n_bad_channels": 3,
    "clustering_severity": "mild",  // none/mild/moderate/severe
    "ica_success": true,
    "n_ica_components": 25,
    "n_components_rejected": 5,
    "recommended_action": "accept"  // accept/review/reject
  },
  "qc_report_path": "relative/path/to/qc.html"
}
"""

from typing import Dict, Any, Optional
from datetime import datetime

from .base import QCResult


def assess_quality_status(bad_pct: float, clustering: str) -> str:
    """
    Assess overall quality status based on bad channel percentage and clustering.

    Parameters
    ----------
    bad_pct : float
        Percentage of bad channels (0-100)
    clustering : str
        Clustering severity ('none', 'mild', 'moderate', 'severe')

    Returns
    -------
    str
        Quality status: 'excellent', 'good', 'acceptable', or 'poor'
    """
    # Base quality on bad channel percentage
    if bad_pct < 10:
        quality = 'excellent'
    elif bad_pct < 20:
        quality = 'good'
    elif bad_pct < 30:
        quality = 'acceptable'
    else:
        quality = 'poor'

    # Downgrade for severe clustering
    if clustering == 'severe' and quality in ['excellent', 'good']:
        quality = 'acceptable'

    return quality


def get_recommended_action(quality_status: str, ica_success: bool,
                           gates_ok: Optional[bool] = None) -> str:
    """
    Get recommended action based on quality status and ICA success.

    Parameters
    ----------
    quality_status : str
        Quality status ('excellent', 'good', 'acceptable', 'poor')
    ica_success : bool
        Whether ICA completed successfully
    gates_ok : bool, optional
        Verdict of the per-recording QC gate(s) (e.g. the ocular
        ``blink_residual_pass``). ``False`` vetoes an ``accept`` — the action
        must not contradict the gate's evidence (2026-10-17: 25SusAttn
        sub-007 stroopvis failed the ocular gate and still reported
        ``accept``). ``None`` (no gate ran / not supplied) leaves the previous
        behaviour unchanged.

    Returns
    -------
    str
        Recommended action: 'accept', 'review', or 'reject'
    """
    if quality_status == 'poor':
        return 'reject'
    elif quality_status == 'acceptable':
        return 'review'
    elif quality_status == 'good' and not ica_success:
        return 'review'
    elif gates_ok is False:
        return 'review'
    else:
        return 'accept'


def extract_metrics_from_qc_result(
    qc_result: QCResult,
    subject_id: str,
    session: str = "01",
    task: str = "unknown",
    run: str = "1",
    pipeline: str = "standard",
    qc_report_path: Optional[str] = None,
    gates_ok: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Extract standardized metrics from QCResult for JSON export.

    Parameters
    ----------
    qc_result : QCResult
        QC result from PreprocessedQC.compute()
    subject_id : str
        Subject identifier
    session : str
        Session identifier
    task : str
        Task name
    run : str
        Run number
    pipeline : str
        Pipeline name
    qc_report_path : str, optional
        Relative path to QC HTML report
    gates_ok : bool, optional
        Verdict of the per-recording QC gate(s) (e.g. the ocular
        ``blink_residual_pass``). ``False`` vetoes an ``accept`` recommendation
        (the action must not contradict the gate's evidence); ``None`` leaves
        the previous behaviour unchanged.

    Returns
    -------
    dict
        Standardized QC metrics dictionary (see JSON schema above)
    """
    # Extract metrics from QCResult
    metrics_dict = {m.name: m.value for m in qc_result.metrics}

    # Extract key metrics with defaults
    n_bad_channels = int(metrics_dict.get('Bad Channels', 0))
    n_eeg_channels = int(metrics_dict.get('N EEG Channels', 128))
    pct_bad_channels = float(metrics_dict.get('Bad Channels', 0)) / n_eeg_channels * 100 if n_eeg_channels > 0 else 0.0

    # ICA metrics
    ica_success = 'ICA Components' in metrics_dict
    n_ica_components = int(metrics_dict.get('ICA Components', 0))
    n_components_rejected = int(metrics_dict.get('ICA Excluded', 0))

    # Clustering metrics
    n_clustered_bad = int(metrics_dict.get('Clustered Bad Channels', 0))

    # Determine clustering severity from metrics first (more reliable than parsing notes)
    clustering_severity = 'none'

    # Try to find "Clustered Bad Channels" metric status
    for metric in qc_result.metrics:
        if metric.name == "Clustered Bad Channels":
            # Metric value indicates clustering present
            if metric.value > 0:
                # Check status: 'bad' = severe, 'warning' = moderate/mild, 'ok' = none
                if metric.status == 'bad':
                    clustering_severity = 'severe'
                elif metric.status == 'warning':
                    # Distinguish mild vs moderate from notes
                    clustering_severity = 'moderate'  # Default to moderate for warnings
                    for note in qc_result.notes:
                        if 'mild clustering' in note.lower():
                            clustering_severity = 'mild'
                            break
                        elif 'moderate clustering' in note.lower():
                            clustering_severity = 'moderate'
                            break
            break

    # Fallback: parse from notes if metric not found
    if clustering_severity == 'none':
        for note in qc_result.notes:
            if 'Clustering' in note or 'clustering' in note:
                if 'severe' in note.lower():
                    clustering_severity = 'severe'
                elif 'moderate' in note.lower():
                    clustering_severity = 'moderate'
                elif 'mild' in note.lower():
                    clustering_severity = 'mild'
                break

    # Assess quality
    quality_status = assess_quality_status(pct_bad_channels, clustering_severity)
    recommended_action = get_recommended_action(quality_status, ica_success, gates_ok)

    # Build output dictionary
    data = {
        'subject_id': subject_id,
        'session': session,
        'task': task,
        'run': run,
        'pipeline': pipeline,
        'timestamp': datetime.now().isoformat(),
        'quality_metrics': {
            'quality_status': quality_status,
            'pct_bad_channels': round(pct_bad_channels, 2),
            'n_bad_channels': n_bad_channels,
            'n_eeg_channels': n_eeg_channels,
            'clustering_severity': clustering_severity,
            'n_clustered_bad': n_clustered_bad,
            'ica_success': ica_success,
            'n_ica_components': n_ica_components,
            'n_components_rejected': n_components_rejected,
            'gates_ok': gates_ok,
            'recommended_action': recommended_action,
        },
        'qc_report_path': qc_report_path if qc_report_path else f"{subject_id}_ses-{session}_task-{task}_run-{run}_preprocessed_qc.html"
    }

    return data
