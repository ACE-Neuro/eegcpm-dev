"""Epoch combination module for merging multiple runs.

This module combines epochs from multiple accepted runs of the same task,
enabling better SNR and more trials for subsequent analysis.
"""

import logging

from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import mne
import numpy as np

from ..workflow.state import (
    WorkflowState,
    WorkflowStateManager,
    ProcessingStatus,
    StepRecord,
)
from .combined_qc import CombinedQC
from ..core.paths import find_preprocessed_raw
from ..data.event_mapping import translate_event_codes, get_event_mapping_for_run
from datetime import datetime


@dataclass
class EpochCombinationResult:
    """Result of combining epochs from multiple runs."""
    success: bool
    combined_epochs: Optional[mne.Epochs] = None
    n_runs_combined: int = 0
    runs_included: List[str] = None
    n_total_epochs: int = 0
    n_epochs_per_run: Dict[str, int] = None
    output_path: Optional[Path] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.runs_included is None:
            self.runs_included = []
        if self.n_epochs_per_run is None:
            self.n_epochs_per_run = {}


logger = logging.getLogger(__name__)


class EpochCombiner:
    """Combine epochs from multiple runs of the same task."""

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Path,
        state_manager: Optional[WorkflowStateManager] = None,
        verbose: bool = True,
        bids_root: Optional[Path] = None
    ):
        """
        Initialize epoch combiner.

        Parameters
        ----------
        config : dict
            Epoching configuration
        output_dir : Path
            Output directory for combined epochs
        state_manager : WorkflowStateManager, optional
            State manager for workflow tracking
        verbose : bool
            Whether to print progress messages
        bids_root : Path, optional
            BIDS dataset root for event mapping translation
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_manager = state_manager
        self.verbose = verbose
        self.bids_root = Path(bids_root) if bids_root else None

    def _print(self, *args, **kwargs):
        """Print only if verbose mode is enabled."""
        if self.verbose:
            try:
                print(*args, **kwargs)
            except BrokenPipeError as exc:
                # stdout was closed (piped into `head` etc.). Record it and
                # stop printing instead of silently swallowing the condition
                # (P1-F11).
                logger.debug("stdout closed, dropping verbose output: %s", exc)
                self.verbose = False

    def combine_runs(
        self,
        raw_files: List[Path],
        run_ids: List[str],
        subject_id: str,
        session: str,
        task: str,
        pipeline: str = "preprocessing"
    ) -> EpochCombinationResult:
        """
        Combine epochs from multiple preprocessed run files.

        Parameters
        ----------
        raw_files : List[Path]
            Paths to preprocessed raw FIF files
        run_ids : List[str]
            Run identifiers (e.g., ['01', '02'])
        subject_id : str
            Subject identifier
        session : str
            Session identifier
        task : str
            Task name
        pipeline : str
            Pipeline name

        Returns
        -------
        EpochCombinationResult
            Result of combination
        """
        self._print(f"\nCombining {len(raw_files)} runs for {subject_id} {task}")

        # Initialize workflow state for combined
        if self.state_manager:
            workflow = WorkflowState(
                subject_id=subject_id,
                session=session,
                task=task,
                run=None,  # Combined workflow
                pipeline=pipeline,
                status=ProcessingStatus.IN_PROGRESS
            )
            self.state_manager.save_state(workflow)

        try:
            # Load all runs
            load_step = StepRecord(
                step_name="load_runs",
                status=ProcessingStatus.IN_PROGRESS,
                started_at=datetime.now()
            )

            raw_list = []
            for raw_file, run_id in zip(raw_files, run_ids):
                self._print(f"  Loading run-{run_id}: {raw_file.name}")
                raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)
                raw_list.append(raw)

            load_step.status = ProcessingStatus.COMPLETED
            load_step.completed_at = datetime.now()

            if self.state_manager:
                workflow.add_step(load_step)
                self.state_manager.save_state(workflow)

            # Harmonize channels across runs
            harmonize_step = StepRecord(
                step_name="harmonize_channels",
                status=ProcessingStatus.IN_PROGRESS,
                started_at=datetime.now()
            )

            self._print(f"  Harmonizing channels across {len(raw_list)} runs...")
            raw_list = self._harmonize_channels(raw_list, run_ids)

            harmonize_step.status = ProcessingStatus.COMPLETED
            harmonize_step.completed_at = datetime.now()

            if self.state_manager:
                workflow.add_step(harmonize_step)
                self.state_manager.save_state(workflow)

            # Concatenate raw files
            concat_step = StepRecord(
                step_name="concatenate",
                status=ProcessingStatus.IN_PROGRESS,
                started_at=datetime.now()
            )

            self._print(f"  Concatenating {len(raw_list)} runs...")
            # Concatenate with annotations
            raw_combined = mne.concatenate_raws(raw_list, preload=True, verbose=False)

            # Re-apply average reference projector (was deleted during harmonization)
            # This is required for source reconstruction
            raw_combined.set_eeg_reference(projection=True, verbose=False)

            self._print(f"  Combined data: {raw_combined.times[-1]:.1f}s, {len(raw_combined.ch_names)} channels, {len(raw_combined.annotations)} annotations")

            concat_step.status = ProcessingStatus.COMPLETED
            concat_step.completed_at = datetime.now()

            if self.state_manager:
                workflow.add_step(concat_step)
                self.state_manager.save_state(workflow)

            # Create epochs
            epoch_step = StepRecord(
                step_name="create_epochs",
                status=ProcessingStatus.IN_PROGRESS,
                started_at=datetime.now()
            )

            self._print(f"  Creating epochs...")
            epochs = self._create_epochs(raw_combined)

            epoch_step.status = ProcessingStatus.COMPLETED
            epoch_step.completed_at = datetime.now()
            epoch_step.metadata = {
                'n_epochs': len(epochs),
                'event_id': epochs.event_id if epochs else {},
                'tmin': self.config.get('epochs', {}).get('tmin', -0.2),
                'tmax': self.config.get('epochs', {}).get('tmax', 0.5),
            }

            if self.state_manager:
                workflow.add_step(epoch_step)
                workflow.status = ProcessingStatus.COMPLETED
                self.state_manager.save_state(workflow)

            # Save combined epochs
            output_file = self.output_dir / f"{subject_id}_ses-{session}_task-{task}_combined_epo.fif"
            self._print(f"  Saving combined epochs: {output_file.name}")
            epochs.save(output_file, overwrite=True, verbose=False)

            # Calculate per-run epoch counts (approximate based on duration)
            total_duration = raw_combined.times[-1]
            n_epochs_per_run = {}
            cumulative_duration = 0

            for raw, run_id in zip(raw_list, run_ids):
                run_duration = raw.times[-1]
                run_pct = run_duration / total_duration
                n_epochs_per_run[run_id] = int(len(epochs) * run_pct)

            self._print(f"  ✓ Combined {len(epochs)} epochs from {len(run_ids)} runs")

            return EpochCombinationResult(
                success=True,
                combined_epochs=epochs,
                n_runs_combined=len(run_ids),
                runs_included=run_ids,
                n_total_epochs=len(epochs),
                n_epochs_per_run=n_epochs_per_run,
                output_path=output_file
            )

        except Exception as e:
            self._print(f"  ✗ Error combining runs: {e}")

            if self.state_manager:
                workflow.status = ProcessingStatus.FAILED
                workflow.add_step(StepRecord(
                    step_name="error",
                    status=ProcessingStatus.FAILED,
                    error_message=str(e)
                ))
                self.state_manager.save_state(workflow)

            return EpochCombinationResult(
                success=False,
                error=str(e)
            )

    def _harmonize_channels(
        self,
        raw_list: List[mne.io.Raw],
        run_ids: List[str]
    ) -> List[mne.io.Raw]:
        """
        Harmonize channels across runs by keeping only common channels.

        Parameters
        ----------
        raw_list : List[mne.io.Raw]
            List of raw objects from different runs
        run_ids : List[str]
            Run identifiers

        Returns
        -------
        List[mne.io.Raw]
            Raw objects with harmonized channels
        """
        # Find common channels across all runs
        all_channel_sets = [set(raw.ch_names) for raw in raw_list]
        common_channels = set.intersection(*all_channel_sets)

        # Report channel differences
        for i, (raw, run_id) in enumerate(zip(raw_list, run_ids)):
            n_channels = len(raw.ch_names)
            n_common = len(common_channels)
            n_dropped = n_channels - n_common

            if n_dropped > 0:
                self._print(f"    run-{run_id}: {n_channels} channels → {n_common} common ({n_dropped} unique dropped)")

        # Keep only common channels in all runs
        harmonized_list = []
        common_channels_sorted = sorted(common_channels)

        for raw in raw_list:
            raw_copy = raw.copy()

            # Pick common channels
            if set(raw.ch_names) != set(common_channels_sorted):
                raw_copy.pick_channels(common_channels_sorted, ordered=True)

            # Remove SSP projectors (they may differ across runs)
            # We'll apply average reference instead
            raw_copy.del_proj()

            harmonized_list.append(raw_copy)

        return harmonized_list

    def _create_epochs(self, raw: mne.io.Raw) -> mne.Epochs:
        """
        Create epochs from continuous data.

        Parameters
        ----------
        raw : mne.io.Raw
            Continuous EEG data

        Returns
        -------
        mne.Epochs
            Epoched data
        """
        # Get epoching parameters from config
        epochs_config = self.config.get('epochs', {})

        # Find events from annotations
        try:
            events, event_dict = mne.events_from_annotations(raw, verbose=False)
        except ValueError as e:
            # No annotations found, try stim channel
            stim_channels = mne.pick_types(raw.info, stim=True)
            if len(stim_channels) > 0:
                events = mne.find_events(raw, shortest_event=1, verbose=False)
                event_dict = None
            else:
                raise ValueError(f"No events found in data: {e}")

        if events is None or len(events) == 0:
            raise ValueError("No events found in data")

        # Filter events based on task config (if provided)
        event_codes_filter = epochs_config.get('event_codes_filter', None)

        if event_codes_filter and event_dict:
            # Try to filter using semantic names first
            filtered_event_id = {
                name: code for name, code in event_dict.items()
                if name in event_codes_filter
            }

            # If no matches and we have BIDS root, try translating semantic names to numeric codes
            if not filtered_event_id and self.bids_root:
                # Get event mapping from first file (assumes all runs have same mapping)
                # This is passed via config metadata
                event_mapping = epochs_config.get('event_mapping', {})

                if event_mapping:
                    # Translate semantic event names to numeric codes
                    translated_codes = translate_event_codes(event_codes_filter, event_mapping)
                    self._print(f"  🔄 Translated event codes: {dict(zip(event_codes_filter, translated_codes))}")

                    # Filter using translated codes
                    filtered_event_id = {
                        name: code for name, code in event_dict.items()
                        if name in translated_codes
                    }

            if filtered_event_id:
                self._print(f"  🎯 Filtering to {len(filtered_event_id)} event types: {list(filtered_event_id.keys())}")
                event_id = filtered_event_id
            else:
                # No matching events found, use all events
                self._print(f"  ⚠️ No matching events found for filter: {event_codes_filter}")
                event_id = event_dict
        elif 'event_id' in epochs_config and epochs_config['event_id'] is not None:
            # Use explicitly provided event_id
            event_id = epochs_config['event_id']
        elif event_dict:
            # Use all events from annotations
            event_id = event_dict
        else:
            # Auto-generate event_id from unique event codes
            unique_events = np.unique(events[:, 2])
            event_id = {f'event_{code}': code for code in unique_events}

        # Create epochs
        tmin = epochs_config.get('tmin', -0.2)
        tmax = epochs_config.get('tmax', 0.5)
        baseline = epochs_config.get('baseline', (None, 0))
        reject = epochs_config.get('reject', None)

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            proj=False,  # Keep projectors but don't apply yet (for source reconstruction)
            reject=reject,
            reject_by_annotation=False,  # Don't reject based on BAD annotations
            preload=True,
            verbose=False
        )

        return epochs

    def combine_with_selection(
        self,
        run_results: List[Any],  # List[RunProcessingResult]
        selection: Dict[str, bool],
        subject_id: str,
        session: str,
        task: str,
        pipeline: str = "preprocessing",
        generate_qc: bool = True
    ) -> EpochCombinationResult:
        """
        Combine epochs from selected runs based on quality recommendations.

        Parameters
        ----------
        run_results : List[RunProcessingResult]
            Processing results for all runs
        selection : dict
            {run_id: include} mapping
        subject_id : str
            Subject identifier
        session : str
            Session identifier
        task : str
            Task name
        pipeline : str
            Pipeline name
        generate_qc : bool
            Whether to generate QC report

        Returns
        -------
        EpochCombinationResult
            Result of combination
        """
        # Filter runs based on selection
        selected_runs = []
        selected_ids = []

        for result in run_results:
            if result.success and selection.get(result.run, False):
                # Find the preprocessed raw file (stem is
                # `{subject}{_ses-X}{_run-Y}`, not the bare subject id).
                raw_file = (find_preprocessed_raw(result.output_path)
                            if result.output_path else None)
                if raw_file is not None:
                    selected_runs.append(raw_file)
                    selected_ids.append(result.run)

        if len(selected_runs) == 0:
            return EpochCombinationResult(
                success=False,
                error="No runs selected for combination"
            )

        if len(selected_runs) == 1:
            self._print(f"  Warning: Only 1 run selected, but combining anyway")

        # Combine runs
        result = self.combine_runs(
            raw_files=selected_runs,
            run_ids=selected_ids,
            subject_id=subject_id,
            session=session,
            task=task,
            pipeline=pipeline
        )

        # Generate QC report if requested and combination succeeded
        if generate_qc and result.success:
            qc = CombinedQC(self.output_dir)
            qc_result = qc.generate_report(
                combination_result=result,
                run_results=run_results,
                subject_id=subject_id,
                session=session,
                task=task
            )

            if qc_result.success:
                self._print(f"  ✓ QC report: {qc_result.html_path}")
            else:
                self._print(f"  ✗ QC report failed: {qc_result.error}")

        return result
