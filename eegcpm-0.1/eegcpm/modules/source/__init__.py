"""Source reconstruction module."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mne
import numpy as np

from eegcpm.pipeline.base import BaseModule, ModuleResult, EpochsModule
from eegcpm.data.conn_rois import get_conn_rois, get_mni_coordinates, get_roi_names
from eegcpm.modules.qc.source_qc import SourceQC


class SourceReconstructionModule(EpochsModule):
    """
    Source reconstruction from sensor-level EEG.

    Supports:
    - dSPM, sLORETA, eLORETA, MNE
    - Template or individual head models
    - ROI-based parcellation (CONN 32 networks default)
    """

    name = "source_reconstruction"
    version = "0.1.0"
    description = "Source localization and ROI extraction"

    def __init__(self, config: Dict[str, Any], output_dir: Path, task_config: Optional[Dict[str, Any]] = None):
        super().__init__(config, output_dir)
        self.method = config.get("method", "sLORETA")
        self.parcellation = config.get("parcellation", "conn_networks")
        self.snr = config.get("snr", 3.0)
        self.loose = config.get("loose", 0.2)
        self.depth = config.get("depth", 0.8)
        self.spacing = config.get("spacing", "oct6")
        self.task_config = task_config  # Optional: for grouping epochs by condition
        self.trial_level = config.get("trial_level", True)  # Default to trial-level for connectivity

    def process(
        self,
        data: mne.Epochs,
        subject: Optional[Any] = None,
        forward: Optional[mne.Forward] = None,
        **kwargs,
    ) -> ModuleResult:
        """
        Perform source reconstruction.

        Args:
            data: Epochs data
            subject: Subject info
            forward: Optional pre-computed forward model

        Returns:
            ModuleResult with source estimates
        """
        start_time = time.time()
        epochs = data.copy()  # Copy to avoid modifying original
        output_files = []
        warnings = []

        try:
            # Ensure average reference projector is set (required for source modeling)
            if not any('EEG' in proj['desc'] and 'average' in proj['desc'].lower()
                      for proj in epochs.info['projs']):
                epochs.set_eeg_reference(projection=True, verbose=False)
                warnings.append("Applied average EEG reference projector (required for source modeling)")

            # Get or compute forward model
            if forward is None:
                forward, warnings_fwd = self._get_template_forward(epochs.info)
                warnings.extend(warnings_fwd)

            # Compute noise covariance
            noise_cov = mne.compute_covariance(
                epochs,
                tmin=None,
                tmax=0,  # Baseline period
                method="empirical",
                verbose=False,
            )

            # Make inverse operator
            inverse_operator = mne.minimum_norm.make_inverse_operator(
                epochs.info,
                forward,
                noise_cov,
                loose=self.loose,
                depth=self.depth,
                verbose=False,
            )

            # Compute source estimates
            lambda2 = 1.0 / self.snr ** 2

            if self.trial_level:
                # TRIAL-BY-TRIAL MODE: Process each epoch individually
                roi_data = self._process_trial_level(epochs, inverse_operator, forward, lambda2)
                stcs = {}  # Don't save individual STCs in trial mode (too large)
            else:
                # EVOKED MODE: Average epochs first, then compute source estimates
                stcs = {}

                # If task config provided, group epochs by conditions
                if self.task_config and 'conditions' in self.task_config:
                    # Group epochs by task config conditions
                    for condition_spec in self.task_config['conditions']:
                        condition_name = condition_spec['name']
                        event_codes = condition_spec.get('event_codes', [])

                        # Find which epochs belong to this condition
                        matching_events = [str(ec) for ec in event_codes if str(ec) in epochs.event_id]

                        if matching_events:
                            # Combine all matching events for this condition
                            condition_epochs = epochs[matching_events]
                            evoked = condition_epochs.average()
                            stc = mne.minimum_norm.apply_inverse(
                                evoked,
                                inverse_operator,
                                lambda2=lambda2,
                                method=self.method,
                                verbose=False,
                            )
                            stcs[condition_name] = stc
                else:
                    # No task config - process each event separately
                    for condition in epochs.event_id:
                        evoked = epochs[condition].average()
                        stc = mne.minimum_norm.apply_inverse(
                            evoked,
                            inverse_operator,
                            lambda2=lambda2,
                            method=self.method,
                            verbose=False,
                        )
                        stcs[condition] = stc

                # Extract ROI time courses from evoked responses
                roi_data = {}
                if self.parcellation == "conn_networks":
                    roi_data = self._extract_conn_rois(stcs, forward)

            # Save results
            subject_id = subject.id if subject else "unknown"

            # Save source estimates (only in evoked mode)
            for condition, stc in stcs.items():
                stc_path = self.output_dir / f"{subject_id}_{condition}_stc"
                stc.save(stc_path, overwrite=True, verbose=False)
                output_files.append(Path(f"{stc_path}-lh.stc"))

            # Save ROI data
            if roi_data:
                roi_path = self.output_dir / f"{subject_id}_roi_tc.npz"

                if self.trial_level:
                    # Save with compression for trial-level data (can be large)
                    np.savez_compressed(roi_path, **roi_data)
                else:
                    # Regular save for evoked data
                    np.savez(roi_path, **roi_data)

                output_files.append(roi_path)

            # Generate QC report
            qc_metrics = []
            try:
                qc = SourceQC(self.output_dir)

                # Get session info
                session_id = subject.session if hasattr(subject, 'session') else "01"

                # Prepare data for QC
                qc_data = {
                    'stcs': stcs,
                    'roi_data': roi_data,
                    'method': self.method,
                    'parcellation': self.parcellation,
                }

                # Compute QC
                qc_result = qc.compute(
                    data=qc_data,
                    subject_id=subject_id,
                    session_id=session_id,
                    sfreq=epochs.info['sfreq']
                )

                # Generate HTML report
                html_filename = f"{subject_id}_ses-{session_id}_source_qc.html"
                qc.generate_html_report(qc_result, output_path=self.output_dir / html_filename)
                output_files.append(self.output_dir / html_filename)

                # Extract metrics
                if qc_result.metrics:
                    qc_metrics = qc_result.metrics

                warnings.append(f"Generated QC report: {html_filename}")

            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                warnings.append(f"QC generation failed: {str(e)}")
                warnings.append(f"Error details: {error_details[:500]}")

            return ModuleResult(
                success=True,
                module_name=self.name,
                execution_time_seconds=time.time() - start_time,
                outputs={
                    "data": roi_data if roi_data else stcs,
                    "stcs": stcs,
                    "roi_data": roi_data,
                    "inverse_operator": inverse_operator,
                },
                output_files=output_files,
                warnings=warnings,
                metadata={
                    "method": self.method,
                    "parcellation": self.parcellation,
                    "trial_level": self.trial_level,
                    "n_sources": stcs[list(stcs.keys())[0]].data.shape[0] if stcs else 0,
                    "n_rois": len([k for k in roi_data.keys() if not k.endswith('_times') and k not in ['roi_names', 'sfreq']]) if roi_data else 0,
                    "n_trials": roi_data[list(roi_data.keys())[1]].shape[0] if self.trial_level and roi_data and len(roi_data) > 1 else 0,
                    "conditions": list(stcs.keys()) if stcs else [k for k in roi_data.keys() if not k.endswith('_times') and k not in ['roi_names', 'sfreq']],
                    "qc_metrics": {m.name: m.value for m in qc_metrics} if qc_metrics else {},
                },
            )

        except Exception as e:
            return ModuleResult(
                success=False,
                module_name=self.name,
                execution_time_seconds=time.time() - start_time,
                errors=[str(e)],
            )

    def _get_template_forward(
        self,
        info: mne.Info,
    ) -> tuple:
        """
        Get template forward model (fsaverage).

        Returns:
            (forward, warnings)
        """
        warnings = []

        try:
            # Use fsaverage template
            fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
            subjects_dir = Path(fs_dir).parent

            # Setup source space
            src = mne.setup_source_space(
                "fsaverage",
                spacing=self.spacing,
                subjects_dir=subjects_dir,
                add_dist=False,
                verbose=False,
            )

            # BEM model
            bem_path = Path(fs_dir) / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
            if not bem_path.exists():
                # Use simpler model
                bem_path = Path(fs_dir) / "bem" / "fsaverage-ico-5-src.fif"
                bem = mne.make_bem_model(
                    "fsaverage",
                    subjects_dir=subjects_dir,
                    conductivity=(0.3,),  # Single shell
                    verbose=False,
                )
                bem_sol = mne.make_bem_solution(bem, verbose=False)
            else:
                bem_sol = mne.read_bem_solution(bem_path, verbose=False)

            # Compute forward
            forward = mne.make_forward_solution(
                info,
                trans="fsaverage",
                src=src,
                bem=bem_sol,
                eeg=True,
                mindist=5.0,
                verbose=False,
            )

            return forward, warnings

        except Exception as e:
            warnings.append(f"Template forward model failed: {e}")
            raise

    def _get_roi_vertex_mapping(
        self,
        forward: mne.Forward,
        roi_radius: float = 10.0,
    ) -> List[np.ndarray]:
        """
        Build a mapping from each CONN ROI to source-space vertex indices.

        For each ROI, finds all source-space vertices within ``roi_radius``
        (mm, in MNI space) of the ROI's MNI centre on the appropriate
        hemisphere.  Falls back to the single nearest vertex when no vertices
        fall within the radius (this happens, e.g., for the cerebellar ROIs
        which are not on the cortical surface).

        Returns
        -------
        list of np.ndarray
            ``roi_vertex_indices[i]`` is a 1-D int array of row indices into
            ``stc.data`` for ROI *i*.  Row indices already account for the
            left-then-right ordering of MNE surface source estimates.
        """
        rois = get_conn_rois()
        mni_coords = get_mni_coordinates()

        # Source-space vertex positions in head coordinates
        src = forward["src"]
        mri_head_t = forward["mri_head_t"]

        # fsaverage subjects_dir (where template forward was built)
        subjects_dir = Path(mne.datasets.fetch_fsaverage(verbose=False)).parent

        lh_vertno = src[0]["vertno"]
        rh_vertno = src[1]["vertno"]
        lh_pos_head = src[0]["rr"][lh_vertno]
        rh_pos_head = src[1]["rr"][rh_vertno]

        # Convert head -> MNI
        lh_mni = mne.head_to_mni(
            lh_pos_head, "fsaverage", mri_head_t, subjects_dir
        )
        rh_mni = mne.head_to_mni(
            rh_pos_head, "fsaverage", mri_head_t, subjects_dir
        )

        n_lh = len(lh_vertno)  # offset for RH indices in stc.data
        roi_vertex_indices: List[np.ndarray] = []

        for roi, mni in zip(rois, mni_coords):
            mni_arr = np.asarray(mni, dtype=float)
            dl = np.sqrt(((lh_mni - mni_arr) ** 2).sum(axis=1))
            dr = np.sqrt(((rh_mni - mni_arr) ** 2).sum(axis=1))

            if roi.hemisphere == "L":
                idx = np.where(dl < roi_radius)[0]
            elif roi.hemisphere == "R":
                idx = np.where(dr < roi_radius)[0]
            else:
                # Midline: pool both hemispheres
                idx = np.concatenate(
                    [np.where(dl < roi_radius)[0],
                     n_lh + np.where(dr < roi_radius)[0]]
                )

            if len(idx) == 0:
                # Fallback: use the single nearest vertex
                if roi.hemisphere == "L":
                    idx = np.array([int(np.argmin(dl))])
                elif roi.hemisphere == "R":
                    idx = np.array([n_lh + int(np.argmin(dr))])
                else:
                    if dl.min() <= dr.min():
                        idx = np.array([int(np.argmin(dl))])
                    else:
                        idx = np.array([n_lh + int(np.argmin(dr))])

            roi_vertex_indices.append(idx)
        return roi_vertex_indices

    def _extract_roi_tc_from_stc(
        self,
        stc: mne.SourceEstimate,
        roi_vertex_indices: List[np.ndarray],
    ) -> np.ndarray:
        """
        Average ``stc.data`` across vertices for each ROI.

        Parameters
        ----------
        stc : mne.SourceEstimate
        roi_vertex_indices : list of np.ndarray
            Row indices into ``stc.data`` for each ROI.

        Returns
        -------
        np.ndarray, shape (n_rois, n_times)
        """
        n_rois = len(roi_vertex_indices)
        roi_tc = np.zeros((n_rois, stc.data.shape[1]), dtype=stc.data.dtype)
        for i, idx in enumerate(roi_vertex_indices):
            roi_tc[i, :] = stc.data[idx, :].mean(axis=0)
        return roi_tc

    def _extract_conn_rois(
        self,
        stcs: Dict[str, mne.SourceEstimate],
        forward: mne.Forward,
    ) -> Dict[str, np.ndarray]:
        """
        Extract time courses for the CONN 32 ROIs by averaging source
        estimates across vertices within ``roi_radius`` of each ROI centre.

        Args:
            stcs: Source estimates per condition
            forward: Forward model

        Returns:
            Dict with ROI time courses per condition
        """
        roi_names = get_roi_names()
        roi_radius = self.config.get("roi_radius", 10.0)
        roi_vertex_indices = self._get_roi_vertex_mapping(forward, roi_radius)

        roi_data = {"roi_names": roi_names}

        for condition, stc in stcs.items():
            roi_tc = self._extract_roi_tc_from_stc(stc, roi_vertex_indices)
            roi_data[condition] = roi_tc
            roi_data[f"{condition}_times"] = stc.times

        # Add sampling frequency (required for connectivity analysis)
        if stcs:
            first_stc = list(stcs.values())[0]
            if len(first_stc.times) > 1:
                sfreq = 1.0 / (first_stc.times[1] - first_stc.times[0])
                roi_data['sfreq'] = sfreq

        return roi_data

    def _process_trial_level(
        self,
        epochs: mne.Epochs,
        inverse_operator: mne.minimum_norm.InverseOperator,
        forward: mne.Forward,
        lambda2: float,
    ) -> Dict[str, np.ndarray]:
        """
        Process epochs trial-by-trial for connectivity analysis.

        Args:
            epochs: MNE Epochs object
            inverse_operator: Inverse operator
            forward: Forward model
            lambda2: Regularization parameter

        Returns:
            Dictionary with trial-level ROI time courses
        """
        roi_names = get_roi_names()
        n_rois = len(roi_names)

        # Build ROI -> vertex mapping once
        roi_radius = self.config.get("roi_radius", 10.0)
        roi_vertex_indices = self._get_roi_vertex_mapping(forward, roi_radius)

        # Group epochs by condition if task config provided
        if self.task_config and 'conditions' in self.task_config:
            condition_groups = {}
            for condition_spec in self.task_config['conditions']:
                condition_name = condition_spec['name']
                event_codes = condition_spec.get('event_codes', [])
                matching_events = [str(ec) for ec in event_codes if str(ec) in epochs.event_id]

                if matching_events:
                    condition_groups[condition_name] = epochs[matching_events]
        else:
            # No task config - use event types as conditions
            condition_groups = {condition: epochs[condition] for condition in epochs.event_id}

        # Process each condition
        roi_data = {'roi_names': roi_names}

        for condition_name, condition_epochs in condition_groups.items():
            n_trials = len(condition_epochs)
            n_times = len(condition_epochs.times)

            # Initialize array for all trials: (n_trials, n_rois, n_times)
            trial_roi_data = np.zeros((n_trials, n_rois, n_times))

            print(f"   Processing {n_trials} trials for condition '{condition_name}'...")

            # Process each trial
            for trial_idx in range(n_trials):
                # Get single trial as evoked (MNE requires Evoked for apply_inverse)
                single_trial = condition_epochs[trial_idx].average()

                # Compute source estimate for this trial
                stc = mne.minimum_norm.apply_inverse(
                    single_trial,
                    inverse_operator,
                    lambda2=lambda2,
                    method=self.method,
                    verbose=False,
                )

                # Extract ROI time courses for this trial
                trial_roi_data[trial_idx] = self._extract_roi_tc_from_stc(
                    stc, roi_vertex_indices
                )

            # Store trials for this condition
            roi_data[condition_name] = trial_roi_data  # (n_trials, n_rois, n_times)
            roi_data[f"{condition_name}_times"] = condition_epochs.times

        # Add sampling frequency
        if len(epochs.times) > 1:
            sfreq = epochs.info['sfreq']
            roi_data['sfreq'] = sfreq

        return roi_data

    def get_output_spec(self) -> Dict[str, str]:
        return {
            "stcs": "Dict of condition -> SourceEstimate",
            "roi_data": "Dict with ROI time courses",
            "inverse_operator": "MNE InverseOperator",
        }
