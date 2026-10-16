"""Aggregate reporting across all subjects and runs.

This module generates index pages and summary reports for browsing
QC results across an entire dataset.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json
from datetime import datetime
from ..core.paths import find_preprocessed_raw
from ..workflow.state import WorkflowStateManager

logger = logging.getLogger(__name__)


@dataclass
class SubjectSummary:
    """Summary of processing status for one subject."""
    subject_id: str
    session: str
    task: str
    n_runs_total: int = 0
    n_runs_processed: int = 0
    n_runs_accepted: int = 0
    n_runs_rejected: int = 0
    combined_epochs: Optional[int] = None
    combined_qc_path: Optional[Path] = None
    run_qc_paths: List[Path] = field(default_factory=list)
    quality_summary: Dict[str, int] = field(default_factory=dict)  # {excellent: 2, good: 1, ...}


class AggregateReportGenerator:
    """Generate aggregate reports across all subjects."""

    def __init__(self, derivatives_root: Path, pipeline_name: str):
        """
        Initialize aggregate report generator.

        Parameters
        ----------
        derivatives_root : Path
            Root derivatives directory
        pipeline_name : str
            Pipeline name (e.g., 'pipeline-asr-test')
        """
        self.derivatives_root = Path(derivatives_root)
        self.pipeline_name = pipeline_name
        self.pipeline_dir = self.derivatives_root / pipeline_name
        self.qc_output = self.pipeline_dir / "qc"

        # Initialize state manager if database exists
        state_db = self.derivatives_root / ".eegcpm" / "state.db"
        self.state_manager = WorkflowStateManager(state_db) if state_db.exists() else None

    def scan_subjects(self) -> List[SubjectSummary]:
        """
        Scan pipeline directory for all processed subjects.

        Returns
        -------
        List[SubjectSummary]
            Summary for each subject
        """
        summaries = []

        # Scan for subject directories
        if not self.pipeline_dir.exists():
            return summaries

        for subject_dir in sorted(self.pipeline_dir.iterdir()):
            if not subject_dir.is_dir() or subject_dir.name.startswith('.'):
                continue

            subject_id = subject_dir.name

            # Scan sessions
            for session_dir in sorted(subject_dir.iterdir()):
                if not session_dir.is_dir() or not session_dir.name.startswith('ses-'):
                    continue

                session = session_dir.name.replace('ses-', '')

                # Scan tasks
                for task_dir in sorted(session_dir.iterdir()):
                    if not task_dir.is_dir() or not task_dir.name.startswith('task-'):
                        continue

                    task = task_dir.name.replace('task-', '')

                    # Get subject summary
                    summary = self._scan_subject_task(
                        subject_dir=task_dir,
                        subject_id=subject_id,
                        session=session,
                        task=task
                    )

                    if summary:
                        summaries.append(summary)

        return summaries

    def _scan_subject_task(
        self,
        subject_dir: Path,
        subject_id: str,
        session: str,
        task: str
    ) -> Optional[SubjectSummary]:
        """Scan a subject's task directory for processing results."""

        summary = SubjectSummary(
            subject_id=subject_id,
            session=session,
            task=task
        )

        # Count run directories
        run_dirs = [d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith('run-')]
        summary.n_runs_total = len(run_dirs)

        # Check each run for preprocessing output
        quality_counts = {'excellent': 0, 'good': 0, 'acceptable': 0, 'poor': 0}

        for run_dir in run_dirs:
            # Check if the preprocessed file exists (stem carries
            # session/run, so glob for it rather than hard-coding the
            # subject-only name).
            preproc_file = find_preprocessed_raw(run_dir)
            if preproc_file is not None:
                summary.n_runs_processed += 1

                # Extract run number from directory name
                run_id = run_dir.name.replace('run-', '')

                # Try to read quality metrics from workflow state database
                if self.state_manager:
                    try:
                        # Extract pipeline name from self.pipeline_name (remove 'pipeline-' prefix)
                        pipeline = self.pipeline_name.replace('pipeline-', '')

                        # Load state for this run
                        state = self.state_manager.load_state(
                            subject_id=subject_id,
                            task=task,
                            pipeline=pipeline,
                            session=session,
                            run=run_id
                        )

                        if state:
                            # Get preprocessing step metadata
                            for step in state.steps:
                                if step.step_name == "preprocessing" and step.metadata:
                                    metadata = step.metadata

                                    # Extract quality info
                                    quality = self._assess_quality(metadata)
                                    quality_counts[quality] += 1

                                    if quality in ['excellent', 'good']:
                                        summary.n_runs_accepted += 1
                                    elif quality == 'poor':
                                        summary.n_runs_rejected += 1
                                    break

                    except Exception as exc:
                        # A broken state row or unexpected metadata must not
                        # silently drop the run from the quality summary —
                        # record it (P1-F11).
                        logger.warning(
                            "quality summary: state/quality lookup failed for "
                            "sub-%s ses-%s task-%s run-%s (%s)",
                            subject_id, session, task, run_id, exc)

        summary.quality_summary = quality_counts

        # Check for combined epochs
        combined_dir = subject_dir / "combined"
        if combined_dir.exists():
            combined_epo = combined_dir / f"{subject_id}_ses-{session}_task-{task}_combined_epo.fif"
            if combined_epo.exists():
                # Count epochs
                try:
                    import mne
                    epochs = mne.read_epochs(combined_epo, preload=False, verbose=False)
                    summary.combined_epochs = len(epochs)
                except (OSError, ValueError, RuntimeError) as exc:
                    # An unreadable combined epoch file is reported, not
                    # silently rendered as "0 epochs" (P1-F11).
                    logger.warning(
                        "combined epochs unreadable for sub-%s ses-%s task-%s: %s",
                        subject_id, session, task, exc)

            # Check for combined QC
            combined_qc = combined_dir / f"{subject_id}_ses-{session}_task-{task}_combined_qc.html"
            if combined_qc.exists():
                summary.combined_qc_path = combined_qc

        return summary if summary.n_runs_processed > 0 else None

    def _assess_quality(self, metadata: Dict[str, Any]) -> str:
        """Assess run quality from metadata."""
        bad_channels_info = metadata.get('bad_channels', {})
        n_bad = len(bad_channels_info.get('detected', []))
        n_total = metadata.get('n_original_channels', 128)

        if n_total == 0:
            return 'unknown'

        pct_bad = (n_bad / n_total) * 100

        if pct_bad > 30:
            return 'poor'
        elif pct_bad > 20:
            return 'acceptable'
        elif pct_bad > 10:
            return 'good'
        else:
            return 'excellent'

    def generate_index(self, summaries: List[SubjectSummary]) -> Path:
        """
        Generate main index HTML page.

        Parameters
        ----------
        summaries : List[SubjectSummary]
            Summaries for all subjects

        Returns
        -------
        Path
            Path to generated index.html
        """
        # Ensure QC output directory exists
        self.qc_output.mkdir(parents=True, exist_ok=True)

        # Organize by task
        by_task = {}
        for summary in summaries:
            if summary.task not in by_task:
                by_task[summary.task] = []
            by_task[summary.task].append(summary)

        # Generate HTML
        index_path = self.qc_output / "index.html"

        html_content = self._generate_index_html(by_task, summaries)
        index_path.write_text(html_content)

        return index_path

    def _generate_index_html(
        self,
        by_task: Dict[str, List[SubjectSummary]],
        all_summaries: List[SubjectSummary]
    ) -> str:
        """Generate HTML content for index page."""

        # Calculate overall statistics
        total_subjects = len(all_summaries)
        total_runs_processed = sum(s.n_runs_processed for s in all_summaries)
        total_runs_accepted = sum(s.n_runs_accepted for s in all_summaries)
        total_combined = sum(1 for s in all_summaries if s.combined_epochs is not None)

        # Generate task sections
        task_sections = []
        for task, summaries in sorted(by_task.items()):
            task_sections.append(self._generate_task_section(task, summaries))

        task_html = "\n".join(task_sections)

        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>EEGCPM Aggregate QC Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .container {{
            max-width: 1400px;
            margin: 20px auto;
            padding: 0 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #3498db;
            margin: 10px 0;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            margin-top: 0;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .badge-excellent {{
            background-color: #27ae60;
            color: white;
        }}
        .badge-good {{
            background-color: #3498db;
            color: white;
        }}
        .badge-acceptable {{
            background-color: #f39c12;
            color: white;
        }}
        .badge-poor {{
            background-color: #e74c3c;
            color: white;
        }}
        .timestamp {{
            text-align: center;
            color: #999;
            margin-top: 30px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>EEGCPM Pipeline QC Report</h1>
        <p>Pipeline: {self.pipeline_name}</p>
    </div>

    <div class="container">
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Subjects</div>
                <div class="stat-value">{total_subjects}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Runs Processed</div>
                <div class="stat-value">{total_runs_processed}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Runs Accepted</div>
                <div class="stat-value">{total_runs_accepted}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Combined Datasets</div>
                <div class="stat-value">{total_combined}</div>
            </div>
        </div>

        {task_html}

        <div class="timestamp">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
        """

    def _generate_task_section(self, task: str, summaries: List[SubjectSummary]) -> str:
        """Generate HTML section for one task."""

        # Build table rows
        rows = []
        for summary in sorted(summaries, key=lambda s: s.subject_id):
            # Quality badges
            quality_badges = []
            for quality, count in summary.quality_summary.items():
                if count > 0:
                    quality_badges.append(
                        f'<span class="badge badge-{quality}">{count} {quality}</span>'
                    )
            quality_html = " ".join(quality_badges) if quality_badges else "—"

            # Combined QC link
            if summary.combined_qc_path:
                # Make path relative to the qc folder (where index.html lives)
                # Combined QC is at: pipeline-dir/subject/ses-XX/task-XX/combined/qc.html
                # Index is at: pipeline-dir/qc/index.html
                # So we need: ../subject/ses-XX/task-XX/combined/qc.html
                rel_path = summary.combined_qc_path.relative_to(self.pipeline_dir)
                combined_link = f'<a href="../{rel_path}">View QC</a>'
            else:
                combined_link = "—"

            # Epochs count
            epochs_str = str(summary.combined_epochs) if summary.combined_epochs else "—"

            rows.append(f"""
            <tr>
                <td>{summary.subject_id}</td>
                <td>{summary.session}</td>
                <td>{summary.n_runs_processed} / {summary.n_runs_total}</td>
                <td>{quality_html}</td>
                <td>{summary.n_runs_accepted}</td>
                <td>{epochs_str}</td>
                <td>{combined_link}</td>
            </tr>
            """)

        rows_html = "".join(rows)

        return f"""
        <div class="section">
            <h2>Task: {task}</h2>
            <p>{len(summaries)} subjects processed</p>
            <table>
                <thead>
                    <tr>
                        <th>Subject</th>
                        <th>Session</th>
                        <th>Runs Processed</th>
                        <th>Run Quality</th>
                        <th>Accepted</th>
                        <th>Combined Epochs</th>
                        <th>QC Report</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """

    def generate_summary_json(self, summaries: List[SubjectSummary]) -> Path:
        """
        Generate machine-readable summary JSON.

        Parameters
        ----------
        summaries : List[SubjectSummary]
            Summaries for all subjects

        Returns
        -------
        Path
            Path to generated summary.json
        """
        # Ensure QC output directory exists
        self.qc_output.mkdir(parents=True, exist_ok=True)

        summary_data = {
            'pipeline': self.pipeline_name,
            'generated': datetime.now().isoformat(),
            'total_subjects': len(summaries),
            'subjects': []
        }

        for summary in summaries:
            summary_data['subjects'].append({
                'subject_id': summary.subject_id,
                'session': summary.session,
                'task': summary.task,
                'n_runs_total': summary.n_runs_total,
                'n_runs_processed': summary.n_runs_processed,
                'n_runs_accepted': summary.n_runs_accepted,
                'n_runs_rejected': summary.n_runs_rejected,
                'combined_epochs': summary.combined_epochs,
                'quality_summary': summary.quality_summary,
                'combined_qc_path': str(summary.combined_qc_path) if summary.combined_qc_path else None
            })

        summary_path = self.qc_output / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

        return summary_path
