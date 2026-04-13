"""Run-level preprocessing orchestrator.

This module handles processing multiple runs of the same task for a subject,
including per-run preprocessing, QC, and preparation for epoch combination.
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import mne
from ..data.bids_utils import BIDSFile, find_subject_runs, read_raw_eeg
from ..modules.preprocessing import PreprocessingPipeline
from ..modules.qc.preprocessed_qc import PreprocessedQC
from ..modules.qc.multirun_qc_report import MultiRunQCReport
from ..modules.qc.quality_assessment import extract_metrics_from_qc_result
from ..modules.qc.metrics_io import save_qc_metrics_json
from ..workflow.state import (
    WorkflowState,
    WorkflowStateManager,
    ProcessingStatus,
    StepRecord,
)
from .processing_state import ProcessingState, create_processing_state
from datetime import datetime


@dataclass
class RunQualityMetrics:
    """Quality metrics for a single run."""
    run: str
    n_original_channels: int
    n_bad_channels: int
    pct_bad_channels: float
    n_clustered_bad: int
    pct_clustered: float
    clustering_severity: str  # 'none', 'mild', 'moderate', 'severe'
    ica_success: bool
    n_ica_components_rejected: int = 0
    duration_seconds: float = 0.0
    quality_status: str = "unknown"  # 'excellent', 'good', 'acceptable', 'poor'
    recommended_action: str = "accept"  # 'accept', 'review', 'reject'
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'run': self.run,
            'n_original_channels': self.n_original_channels,
            'n_bad_channels': self.n_bad_channels,
            'pct_bad_channels': round(self.pct_bad_channels, 2),
            'n_clustered_bad': self.n_clustered_bad,
            'pct_clustered': round(self.pct_clustered, 2),
            'clustering_severity': self.clustering_severity,
            'ica_success': self.ica_success,
            'n_ica_components_rejected': self.n_ica_components_rejected,
            'duration_seconds': round(self.duration_seconds, 2),
            'quality_status': self.quality_status,
            'recommended_action': self.recommended_action,
            'warnings': self.warnings,
        }


@dataclass
class RunProcessingResult:
    """Result of processing a single run."""
    run: str
    success: bool
    raw_preprocessed: Optional[mne.io.Raw] = None
    output_path: Optional[Path] = None
    qc_path: Optional[Path] = None
    quality_metrics: Optional[RunQualityMetrics] = None
    error: Optional[str] = None
    logs: Optional[List[str]] = None  # Processing logs for UI display


class RunProcessor:
    """Process multiple runs of a task for a single subject."""

    def __init__(
        self,
        bids_root: Path,
        output_root: Path,
        config: Dict[str, Any],
        state_manager: Optional[WorkflowStateManager] = None,
        verbose: bool = True
    ):
        """
        Initialize run processor.

        Parameters
        ----------
        bids_root : Path
            BIDS dataset root
        output_root : Path
            Output directory for derivatives
        config : dict
            Preprocessing configuration
        state_manager : WorkflowStateManager, optional
            State manager for workflow tracking
        verbose : bool
            Whether to print progress messages
        """
        self.bids_root = Path(bids_root)
        self.output_root = Path(output_root)
        self.config = config
        self.state_manager = state_manager
        self.verbose = verbose

    def _print(self, *args, **kwargs):
        """Print only if verbose mode is enabled."""
        if self.verbose:
            try:
                print(*args, **kwargs)
            except BrokenPipeError:
                pass  # Ignore broken pipe errors in non-verbose contexts

    def process_subject_task(
        self,
        subject_id: str,
        task: str,
        session: str = "01",
        pipeline: str = "preprocessing"
    ) -> List[RunProcessingResult]:
        """
        Process all runs for a subject/task.

        Parameters
        ----------
        subject_id : str
            Subject identifier
        task : str
            Task name (BIDS-compliant, e.g., 'saiit')
        session : str
            Session identifier
        pipeline : str
            Pipeline name

        Returns
        -------
        List[RunProcessingResult]
            Results for each run
        """
        # Find all runs for this subject/task
        runs = find_subject_runs(self.bids_root, subject_id, task, session=session)
        if not runs:
            self._print(f"No runs found for {subject_id} task {task}")
            return []

        self._print(f"\nProcessing {subject_id} task {task}: {len(runs)} runs")

        results = []
        for bids_file in runs:
            result = self._process_single_run(
                bids_file,
                subject_id,
                session,
                pipeline
            )
            results.append(result)

        # Generate unified multi-run QC report after processing all runs
        if results:
            try:
                multirun_qc_path = self.generate_multirun_qc_report(
                    subject_id=subject_id,
                    task=task,
                    session=session,
                    results=results
                )
                if multirun_qc_path:
                    self._print(f"✓ Multi-run QC report: {multirun_qc_path.name}")
            except Exception as e:
                self._print(f"Warning: Failed to generate multi-run QC report: {e}")

        return results

    def _process_single_run(
        self,
        bids_file: BIDSFile,
        subject_id: str,
        session: str,
        pipeline: str
    ) -> RunProcessingResult:
        """Process a single run."""
        run = bids_file.run or "01"
        task = bids_file.task

        # Collect logs for UI display
        logs = []
        logs.append(f"Processing run-{run} for subject {subject_id}")

        self._print(f"  Processing run-{run}...")

        # Create output directory (add sub- prefix for BIDS compliance)
        subject_prefix = subject_id if subject_id.startswith('sub-') else f"sub-{subject_id}"
        output_dir = self.output_root / subject_prefix / f"ses-{session}" / f"task-{task}" / f"run-{run}"
        output_dir.mkdir(parents=True, exist_ok=True)
        logs.append(f"Output directory: {output_dir}")

        # Create processing state (filesystem-based, for CLI/UI sync)
        processing_state = create_processing_state(
            subject_id=subject_id,
            session=session,
            task=task,
            run=run,
            source="CLI" if not self.state_manager else "UI",
            config=self.config
        )

        # Initialize workflow state (database-based, UI only)
        if self.state_manager:
            workflow = WorkflowState(
                subject_id=subject_id,
                session=session,
                task=task,
                run=run,
                pipeline=pipeline,
                status=ProcessingStatus.IN_PROGRESS
            )
            self.state_manager.save_state(workflow)

        try:
            # Load data
            load_step = StepRecord(
                step_name="load_data",
                status=ProcessingStatus.IN_PROGRESS,
                started_at=datetime.now()
            )

            # Read FIF file (Legacy)
            # raw = mne.io.read_raw_fif(bids_file.path, preload=True, verbose=False)

            # Automatically read different eeg standard file with mne
            raw = read_raw_eeg(bids_file.path)

            n_original_channels = len(raw.ch_names)
            logs.append(f"Loaded raw data: {n_original_channels} channels, {raw.n_times} samples ({raw.times[-1]:.1f}s)")

            load_step.status = ProcessingStatus.COMPLETED
            load_step.completed_at = datetime.now()
            load_step.output_path = str(bids_file.path)

            if self.state_manager:
                workflow.add_step(load_step)
                self.state_manager.save_state(workflow)

            # Preprocess
            logs.append("Starting preprocessing...")
            preproc_step = StepRecord(
                step_name="preprocessing",
                status=ProcessingStatus.IN_PROGRESS,
                started_at=datetime.now()
            )

            # Get steps from config
            preproc_config = self.config.get('preprocessing', self.config)
            steps = preproc_config.get('steps', [])

            # Create modular pipeline
            pipeline = PreprocessingPipeline(steps=steps, output_dir=output_dir)
            preproc_result = pipeline.process(raw, subject_id=subject_id)

            # Add detailed preprocessing logs from modular pipeline
            if preproc_result.metadata:
                meta = preproc_result.metadata
                print(f"[RUN_PROCESSOR DEBUG] Metadata keys: {list(meta.keys())}")

                # Iterate through step results
                for step_name, step_meta in meta.items():
                    print(f"[RUN_PROCESSOR DEBUG] Processing step: {step_name}, meta: {step_meta}")

                    # Montage (process this BEFORE generic skip handling)
                    if step_name == 'montage':
                        n_ch_before = step_meta.get('n_channels_before', '?')
                        n_ch_after = step_meta.get('n_channels_after', '?')

                        if step_meta.get('applied'):
                            logs.append(f"✓ Montage: {step_meta.get('type', 'unknown')} ({n_ch_before}→{n_ch_after} ch)")
                        elif step_meta.get('skipped'):
                            reason = step_meta.get('reason', 'unknown')
                            logs.append(f"⊘ montage: {reason} ({n_ch_before}→{n_ch_after} ch)")

                        # Show dropped channels (for both applied and skipped)
                        n_dropped = step_meta.get('n_dropped', 0)
                        dropped = step_meta.get('channels_dropped', [])
                        print(f"[RUN_PROCESSOR DEBUG] Montage step - n_dropped={n_dropped}, dropped={dropped}")

                        if n_dropped > 0:
                            logs.append(f"  ⚠️ Dropped {len(dropped)} channel(s) without positions: {', '.join(dropped)}")

                    # Filter
                    elif step_name == 'filter' and step_meta.get('applied'):
                        l_freq = step_meta.get('l_freq', 'None')
                        h_freq = step_meta.get('h_freq', 'None')
                        logs.append(f"✓ Filter: {l_freq}-{h_freq} Hz")

                    # Bad channels
                    elif step_name == 'bad_channels' and step_meta.get('applied'):
                        n_bad = step_meta.get('n_bad_channels', 0)
                        logs.append(f"✓ Bad channels: {n_bad} detected")

                    # Zapline
                    elif step_name == 'zapline' and step_meta.get('applied'):
                        detected_freq = step_meta.get('detected_freq')
                        n_components = step_meta.get('n_components_removed', 0)
                        line_reduction = step_meta.get('line_reduction_db', 0)
                        if detected_freq is not None:
                            logs.append(f"✓ Zapline: {detected_freq:.1f} Hz, {n_components} comp, {line_reduction:.1f} dB")

                    # Artifacts
                    elif step_name == 'artifacts' and step_meta.get('applied'):
                        n_annot = step_meta.get('n_annotations_added', 0)
                        logs.append(f"✓ Artifacts: {n_annot} segments annotated")

                    # ASR (can have multiple - asr_4, asr_7, etc.)
                    elif step_name.startswith('asr') and step_meta.get('applied'):
                        cutoff = step_meta.get('cutoff', 'unknown')
                        mode = step_meta.get('mode', '')
                        montage_restored = step_meta.get('montage_restored', False)

                        # Show montage restoration status
                        montage_str = "✓" if montage_restored else "✗"
                        logs.append(f"✓ ASR ({mode}): cutoff={cutoff}, montage {montage_str}")

                        # Warn about NaN/Inf if present
                        if step_meta.get('has_nan_inf'):
                            pct = step_meta.get('pct_bad_samples', 0)
                            n_ch = step_meta.get('n_affected_channels', 0)
                            n_annot = step_meta.get('n_annotations_added', 0)
                            bad_dur = step_meta.get('total_bad_duration_s', 0)
                            logs.append(f"  ⚠️ ASR NaN/Inf: {pct:.2f}% samples, {n_ch} channels → {n_annot} segments marked BAD ({bad_dur:.1f}s)")

                        # Warn about montage loss (critical issue)
                        if not montage_restored:
                            logs.append(f"  ⚠️ ASR lost montage - RANSAC and ICLabel will fail")

                    # ICA
                    elif step_name == 'ica' and step_meta.get('applied'):
                        n_comp = step_meta.get('n_components', 'unknown')
                        logs.append(f"✓ ICA: {n_comp} components fitted")

                    # ICLabel
                    elif step_name == 'iclabel' and step_meta.get('applied'):
                        n_removed = step_meta.get('n_components_removed', 0)
                        removed_labels = step_meta.get('removed_labels', [])
                        logs.append(f"✓ ICLabel: {n_removed} components removed ({', '.join(removed_labels)})")

                    # Interpolate
                    elif step_name == 'interpolate' and step_meta.get('applied'):
                        action = step_meta.get('action', 'interpolated')
                        n_channels = step_meta.get('n_interpolated', step_meta.get('n_dropped', 0))
                        logs.append(f"✓ Interpolate: {n_channels} channels {action}")

                    # Reference
                    elif step_name == 'reference' and step_meta.get('applied'):
                        ref_type = step_meta.get('type', 'unknown')
                        logs.append(f"✓ Reference: {ref_type}")

                    # Generic handler for skipped steps (except montage which is handled above)
                    elif step_meta.get('skipped') and step_name != 'montage':
                        reason = step_meta.get('reason', 'unknown')
                        logs.append(f"⊘ {step_name}: {reason}")

                    # Resample
                    elif step_name == 'resample' and step_meta.get('applied'):
                        orig_sf = step_meta.get('original_sfreq', 0)
                        new_sf = step_meta.get('new_sfreq', 0)
                        logs.append(f"✓ Resample: {orig_sf:.1f} Hz → {new_sf:.1f} Hz")

                # Warnings
                if preproc_result.warnings:
                    for warning in preproc_result.warnings:
                        logs.append(f"⚠️  {warning}")

            preproc_step.status = ProcessingStatus.COMPLETED if preproc_result.success else ProcessingStatus.FAILED
            preproc_step.completed_at = datetime.now()
            preproc_step.output_path = str(output_dir)
            preproc_step.metadata = preproc_result.metadata

            if not preproc_result.success:
                preproc_step.error_message = str(preproc_result.errors)

            if self.state_manager:
                workflow.add_step(preproc_step)
                workflow.status = ProcessingStatus.COMPLETED if preproc_result.success else ProcessingStatus.FAILED
                self.state_manager.save_state(workflow)

            # Get processed raw from outputs (key is 'data', not 'raw')
            raw_processed = preproc_result.outputs.get('data') if preproc_result.success else None

            # Generate QC report first, then extract quality metrics from it
            qc_path = None
            qc_metrics_json_path = None
            if preproc_result.success and raw_processed is not None:
                try:
                    logs.append("Generating QC report...")

                    # Initialize QC generator
                    qc_generator = PreprocessedQC(
                        output_dir=output_dir,
                        config=self.config
                    )

                    # Get ICA object from outputs if available
                    ica = preproc_result.outputs.get('ica', None)

                    # Extract removed channels from preprocessing metadata
                    removed_channels = {}
                    if preproc_result.metadata:
                        # Check for bad_channels step
                        bad_ch_step = preproc_result.metadata.get('bad_channels', {})
                        if bad_ch_step.get('bad_channels'):
                            bad_list = bad_ch_step['bad_channels']
                            method_raw = bad_ch_step.get('method', 'unknown')

                            # Handle method as list or string
                            if isinstance(method_raw, list):
                                method = method_raw[0] if method_raw else 'unknown'
                            else:
                                method = method_raw

                            # Determine action taken
                            if bad_ch_step.get('drop'):
                                action = 'dropped'
                            elif bad_ch_step.get('mark_only'):
                                action = 'marked_bad'
                            else:
                                action = 'interpolated'

                            # Add to removed_channels dict
                            for ch_name in bad_list:
                                removed_channels[ch_name] = f'{action}_{method}'

                        # Check for montage step (channels dropped due to missing positions)
                        montage_step = preproc_result.metadata.get('montage', {})
                        if montage_step.get('channels_dropped'):
                            for ch_name in montage_step['channels_dropped']:
                                removed_channels[ch_name] = 'no_position'

                    # Compute QC
                    qc_result = qc_generator.compute(
                        data=raw_processed,  # Preprocessed data
                        subject_id=subject_id,
                        ica=ica,
                        raw_before=raw,  # Original raw data for comparison
                        removed_channels=removed_channels,  # Pass removed channels info
                        session_id=session,
                        run_id=run,
                        metadata=preproc_result.metadata  # Pass preprocessing metadata
                    )

                    # Generate HTML report
                    qc_filename = f"{subject_id}_ses-{session}_task-{task}_run-{run}_preprocessed_qc.html"
                    qc_path = output_dir / qc_filename

                    qc_generator.generate_html_report(qc_result, qc_path)
                    logs.append(f"QC report saved: {qc_filename}")

                    # Save QC result as JSON (legacy format for compatibility)
                    import json
                    qc_json_path = output_dir / qc_filename.replace('.html', '.json')
                    with open(qc_json_path, 'w') as f:
                        json.dump(qc_result.to_dict(), f, indent=2)
                    logs.append(f"QC metrics saved: {qc_json_path.name}")

                    # Save standardized QC metrics JSON for CLI/UI sync
                    metrics_data = extract_metrics_from_qc_result(
                        qc_result,
                        subject_id=subject_id,
                        session=session,
                        task=task,
                        run=run,
                        pipeline="preprocessing",
                        qc_report_path=qc_filename
                    )
                    qc_metrics_json_path = output_dir / f"{subject_id}_ses-{session}_task-{task}_run-{run}_qc_metrics.json"
                    save_qc_metrics_json(metrics_data, qc_metrics_json_path)
                    logs.append(f"Standardized QC metrics saved: {qc_metrics_json_path.name}")
                except Exception as qc_error:
                    import traceback
                    logs.append(f"Warning: QC generation failed: {qc_error}")
                    logs.append(f"Traceback: {traceback.format_exc()}")
                    qc_path = None

            # Extract quality metrics AFTER QC report generation so we can use the QC JSON
            quality_metrics = self._extract_quality_metrics(
                run,
                n_original_channels,
                preproc_result.metadata,
                qc_metrics_json=qc_metrics_json_path  # Now available from QC generation
            )

            logs.append(f"Preprocessing completed: {quality_metrics.quality_status} ({quality_metrics.recommended_action})")
            self._print(f"    ✓ run-{run}: {quality_metrics.quality_status} ({quality_metrics.recommended_action})")

            # Save processing state to filesystem
            processing_state.update_status('completed')
            processing_state.logs = logs
            processing_state.set_metadata(preproc_result.metadata)
            processing_state.set_quality(quality_metrics.quality_status)
            processing_state.output_files = {
                'preprocessed': str(output_dir / f"{subject_id}_preprocessed_raw.fif"),
                'qc_report': str(qc_path) if qc_path else None,
            }
            processing_state.save(output_dir)

            return RunProcessingResult(
                run=run,
                success=preproc_result.success,
                raw_preprocessed=raw_processed,
                output_path=output_dir,
                qc_path=qc_path if qc_path and qc_path.exists() else None,
                quality_metrics=quality_metrics,
                logs=logs
            )

        except Exception as e:
            logs.append(f"ERROR: {str(e)}")
            self._print(f"    ✗ run-{run}: {e}")

            # Save failed processing state to filesystem
            processing_state.update_status('failed', error=str(e))
            processing_state.logs = logs
            processing_state.save(output_dir)

            if self.state_manager:
                workflow.status = ProcessingStatus.FAILED
                workflow.add_step(StepRecord(
                    step_name="error",
                    status=ProcessingStatus.FAILED,
                    error_message=str(e)
                ))
                self.state_manager.save_state(workflow)

            return RunProcessingResult(
                run=run,
                success=False,
                error=str(e),
                logs=logs
            )

    def _extract_quality_metrics(
        self,
        run: str,
        n_original: int,
        metadata: Dict[str, Any],
        qc_metrics_json: Optional[Path] = None
    ) -> RunQualityMetrics:
        """Extract quality metrics from preprocessing metadata and QC JSON."""

        # Handle None or empty metadata
        if not metadata:
            return RunQualityMetrics(
                run=run,
                n_original_channels=n_original,
                n_bad_channels=0,
                pct_bad_channels=0.0,
                n_clustered_bad=0,
                pct_clustered=0.0,
                clustering_severity='none',
                ica_success=False,
                quality_status='unknown',
                recommended_action='reject',
                warnings=['Preprocessing failed - no metadata available']
            )

        # Try to load clustering info from QC metrics JSON file (most reliable)
        n_clustered = 0
        pct_clustered = 0.0
        severity = 'none'

        if qc_metrics_json and qc_metrics_json.exists():
            try:
                import json
                with open(qc_metrics_json, 'r') as f:
                    qc_data = json.load(f)

                qm = qc_data.get('quality_metrics', {})
                n_clustered = qm.get('n_clustered_bad', 0)
                severity = qm.get('clustering_severity', 'none')

                # Calculate pct_clustered from n_clustered and n_bad
                n_bad = qm.get('n_bad_channels', 0)
                pct_clustered = (n_clustered / n_bad * 100) if n_bad > 0 else 0.0
            except Exception as e:
                # QC JSON loading failed - continue with defaults
                pass

        # Extract bad channels info from bad_channels step
        bad_channels_step = metadata.get('bad_channels', {})
        n_bad = bad_channels_step.get('n_bad_channels', 0)
        pct_bad = (n_bad / n_original * 100) if n_original > 0 else 0

        # Extract ICA info from ica step
        ica_step = metadata.get('ica', {})
        ica_success = ica_step.get('applied', False)

        # Get number of components excluded from iclabel step
        iclabel_step = metadata.get('iclabel', {})
        n_rejected = iclabel_step.get('n_components_removed', 0)

        # Determine quality status based on bad channel percentage
        if pct_bad > 30:
            quality_status = "poor"
            recommended = "reject"
        elif pct_bad > 20:
            quality_status = "acceptable"
            recommended = "review"
        elif pct_bad > 10:
            quality_status = "good"
            recommended = "accept"
        else:
            quality_status = "excellent"
            recommended = "accept"

        # Adjust for clustering - only reject/review for severe clustering with high bad channels
        if severity == 'severe':
            if pct_bad > 15:
                # Severe clustering + moderate bad channels → reject
                recommended = "reject"
            elif recommended == "accept":
                # Severe clustering + low bad channels → review (not auto-reject)
                recommended = "review"
        elif severity == 'moderate' and pct_bad > 10:
            # Moderate clustering + moderate bad channels → review
            if recommended == "accept":
                recommended = "review"

        # Adjust for ICA failure - only if moderate/high bad channels
        if not ica_success and pct_bad > 10:
            if recommended == "accept":
                recommended = "review"

        # Warnings
        warnings = []
        if pct_bad > 20:
            warnings.append(f"High bad channel percentage: {pct_bad:.1f}%")
        if severity in ['moderate', 'severe']:
            warnings.append(f"Bad channel clustering: {severity}")
        if not ica_success:
            warnings.append("ICA artifact removal failed")

        return RunQualityMetrics(
            run=run,
            n_original_channels=n_original,
            n_bad_channels=n_bad,
            pct_bad_channels=pct_bad,
            n_clustered_bad=n_clustered,
            pct_clustered=pct_clustered,
            clustering_severity=severity,
            ica_success=ica_success,
            n_ica_components_rejected=n_rejected,
            quality_status=quality_status,
            recommended_action=recommended,
            warnings=warnings
        )

    def load_processed_runs(
        self,
        subject_id: str,
        task: str,
        session: str = "01",
        pipeline: str = "preprocessing"
    ) -> List[RunProcessingResult]:
        """
        Load already-processed run results from workflow state (no reprocessing).

        Parameters
        ----------
        subject_id : str
            Subject identifier
        task : str
            Task name
        session : str
            Session identifier
        pipeline : str
            Pipeline name

        Returns
        -------
        List[RunProcessingResult]
            Results loaded from existing workflow states
        """
        results = []

        # Find run directories
        subject_task_dir = self.output_root / subject_id / f"ses-{session}" / f"task-{task}"

        if not subject_task_dir.exists():
            return results

        run_dirs = sorted([d for d in subject_task_dir.iterdir()
                          if d.is_dir() and d.name.startswith('run-')])

        for run_dir in run_dirs:
            run_id = run_dir.name.replace('run-', '')

            # Check if preprocessed file exists
            preproc_file = run_dir / f"{subject_id}_preprocessed_raw.fif"

            if not preproc_file.exists():
                continue

            # Try to load workflow state
            quality_metrics = None
            success = True  # Assume success if preprocessed file exists

            if self.state_manager:
                state = self.state_manager.load_state(
                    subject_id=subject_id,
                    task=task,
                    pipeline=pipeline,
                    session=session,
                    run=run_id
                )

                if state and state.steps:
                    # Extract quality metrics from preprocessing step
                    for step in state.steps:
                        if step.step_name == "preprocessing" and step.metadata:
                            # Get original channel count (need to infer from current + dropped)
                            metadata = step.metadata
                            if 'bad_channels' in metadata:
                                bc_info = metadata['bad_channels']
                                n_detected = len(bc_info.get('detected', []))

                                # Load raw to get current channel count
                                try:
                                    import mne
                                    raw = mne.io.read_raw_fif(preproc_file, preload=False, verbose=False)
                                    n_current = len(raw.ch_names)
                                    n_original = n_current + len(bc_info.get('dropped', []))
                                except:
                                    n_original = 128  # Default assumption
                                    n_current = len(bc_info.get('detected', [])) if bc_info.get('detected') else 0

                                # Try to find QC metrics JSON file
                                qc_json_path = run_dir / f"{subject_id}_ses-{session}_task-{task}_run-{run_id}_qc_metrics.json"

                                quality_metrics = self._extract_quality_metrics(
                                    run_id,
                                    n_original,
                                    metadata,
                                    qc_metrics_json=qc_json_path if qc_json_path.exists() else None
                                )
                                break

                    # Get success status from workflow state
                    success = state.status.value == 'completed'

            # If no quality metrics from workflow state, create a placeholder
            # This happens when workflow state DB is missing or incomplete
            if quality_metrics is None:
                self._print(f"    Warning: No quality metrics in workflow state for run-{run_id}")
                quality_metrics = RunQualityMetrics(
                    run=run_id,
                    n_original_channels=0,
                    n_bad_channels=0,
                    pct_bad_channels=0.0,
                    n_clustered_bad=0,
                    pct_clustered=0.0,
                    clustering_severity='unknown',
                    ica_success=False,
                    quality_status='unknown',
                    recommended_action='review',
                    warnings=['Workflow state missing - reprocess to get quality metrics']
                )

            results.append(RunProcessingResult(
                run=run_id,
                success=success,
                raw_preprocessed=None,  # Don't load into memory
                output_path=run_dir,
                quality_metrics=quality_metrics
            ))

        return results

    def get_run_selection_recommendations(
        self,
        results: List[RunProcessingResult]
    ) -> Dict[str, bool]:
        """
        Get recommendations for which runs to include in combination.

        Parameters
        ----------
        results : List[RunProcessingResult]
            Processing results for all runs

        Returns
        -------
        dict
            {run: should_include} mapping
        """
        recommendations = {}

        for result in results:
            if not result.success or not result.quality_metrics:
                recommendations[result.run] = False
            else:
                # Auto-accept if recommended action is 'accept'
                # Auto-reject if recommended action is 'reject'
                # Flag for review if recommended action is 'review'
                recommendations[result.run] = result.quality_metrics.recommended_action == "accept"

        return recommendations

    def generate_multirun_qc_report(
        self,
        subject_id: str,
        task: str,
        session: str = "01",
        results: Optional[List[RunProcessingResult]] = None
    ) -> Optional[Path]:
        """
        Generate unified multi-run QC report for all runs of a task.

        Creates a single HTML containing:
        - Summary statistics across all runs
        - Overview table with key metrics
        - Collapsible detailed reports for each run

        Parameters
        ----------
        subject_id : str
            Subject identifier
        task : str
            Task name
        session : str
            Session identifier
        results : List[RunProcessingResult], optional
            Processing results. If None, will load from existing QC reports.

        Returns
        -------
        Path or None
            Path to generated multi-run QC report, or None if generation failed
        """
        # If no results provided, scan for existing QC reports
        if results is None:
            subject_task_dir = self.output_root / subject_id / f"ses-{session}" / f"task-{task}"
            if not subject_task_dir.exists():
                self._print(f"No output directory found for {subject_id} task {task}")
                return None

            # Find all run directories
            run_dirs = sorted([d for d in subject_task_dir.iterdir()
                              if d.is_dir() and d.name.startswith('run-')])

            if not run_dirs:
                self._print(f"No runs found for {subject_id} task {task}")
                return None

            results = []
            for run_dir in run_dirs:
                run_id = run_dir.name.replace('run-', '')
                qc_file = run_dir / f"{subject_id}_ses-{session}_task-{task}_run-{run_id}_preprocessed_qc.html"
                qc_json_file = run_dir / f"{subject_id}_ses-{session}_task-{task}_run-{run_id}_preprocessed_qc.json"

                if qc_file.exists():
                    # Try to load quality metrics from JSON if available
                    quality_metrics = None
                    if qc_json_file.exists():
                        try:
                            import json
                            with open(qc_json_file) as f:
                                qc_data = json.load(f)
                            # Store QC data in result for later use
                            results.append(RunProcessingResult(
                                run=run_id,
                                success=True,
                                output_path=run_dir,
                                qc_path=qc_file,
                                quality_metrics=None,  # We'll reconstruct from QC data
                                error=None,
                                logs=None
                            ))
                            # Store qc_data as attribute for later access
                            results[-1].__dict__['qc_data'] = qc_data
                        except Exception as e:
                            self._print(f"Warning: Could not load QC JSON for run-{run_id}: {e}")
                            results.append(RunProcessingResult(
                                run=run_id,
                                success=True,
                                output_path=run_dir,
                                qc_path=qc_file
                            ))
                    else:
                        results.append(RunProcessingResult(
                            run=run_id,
                            success=True,
                            output_path=run_dir,
                            qc_path=qc_file
                        ))

        if not results:
            self._print(f"No QC reports found for {subject_id} task {task}")
            return None

        # Build multi-run report
        report_title = f"Multi-Run QC: {subject_id} | Task: {task} | Session: {session}"
        builder = MultiRunQCReport(title=report_title)

        for result in results:
            if not result.success or not result.qc_path:
                continue

            # Read the individual QC report HTML
            try:
                qc_html = result.qc_path.read_text()
            except Exception as e:
                self._print(f"Warning: Could not read QC report for run-{result.run}: {e}")
                continue

            # Load QC result from JSON if available
            from ..modules.qc.base import QCResult, QCMetric

            # Check if we have QC data from JSON
            qc_data = result.__dict__.get('qc_data')

            if qc_data:
                # Reconstruct QCResult from JSON data
                qc_result = QCResult(subject_id=qc_data.get('subject_id', subject_id))
                qc_result.status = qc_data.get('status', 'ok')
                qc_result.notes = qc_data.get('notes', [])

                # Reconstruct metrics
                for metric_data in qc_data.get('metrics', []):
                    qc_result.add_metric(QCMetric(
                        name=metric_data['name'],
                        value=metric_data['value'],
                        unit=metric_data.get('unit', ''),
                        status=metric_data.get('status', 'ok'),
                        threshold_warning=metric_data.get('threshold_warning'),
                        threshold_bad=metric_data.get('threshold_bad')
                    ))
            elif result.quality_metrics:
                # Fallback: create QC result from quality metrics
                qc_result = QCResult(subject_id=subject_id)
                qm = result.quality_metrics
                qc_result.status = qm.quality_status if qm.quality_status in ['ok', 'warning', 'bad'] else 'ok'

                qc_result.add_metric(QCMetric("N EEG Channels", qm.n_original_channels, ""))
                qc_result.add_metric(QCMetric("Bad Channels", qm.n_bad_channels, ""))
                qc_result.add_metric(QCMetric("Duration", qm.duration_seconds, "s"))
                qc_result.add_metric(QCMetric("ICA Excluded", qm.n_ica_components_rejected, ""))
                qc_result.add_metric(QCMetric("Bad Segment %", 0.0, "%", "ok"))
            else:
                # No data available - create placeholder
                qc_result = QCResult(subject_id=subject_id)
                qc_result.status = "unknown"

            # Add run to multi-run report
            builder.add_run(
                run_id=f"run-{result.run}",
                qc_result=qc_result,
                detailed_html=qc_html
            )

        # Save multi-run report
        output_dir = self.output_root / subject_id / f"ses-{session}" / f"task-{task}"
        output_dir.mkdir(parents=True, exist_ok=True)

        multirun_qc_path = output_dir / f"{subject_id}_ses-{session}_task-{task}_multirun_qc.html"
        builder.save(multirun_qc_path)

        self._print(f"Multi-run QC report saved: {multirun_qc_path.name}")
        return multirun_qc_path
