"""Source reconstruction module."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np

# In mne 1.13 the inverse operators live under mne.minimum_norm; they are imported as
# module-level names so tests can monkeypatch ``eegcpm.modules.source.apply_inverse_epochs``.
from mne.minimum_norm import (
    apply_inverse,
    apply_inverse_epochs,
    make_inverse_operator,
)

from eegcpm.pipeline.base import ModuleResult, EpochsModule
from eegcpm.modules.source.labels import build_conn_labels
from eegcpm.modules.qc.source_qc import SourceQC


class SourceReconstructionModule(EpochsModule):
    """
    Source reconstruction from sensor-level EEG.

    Supports:
    - dSPM, sLORETA, eLORETA, MNE
    - Template or individual head models
    - ROI-based parcellation (CONN 32 networks default)

    ROI time courses are anatomically mapped (CONN MNI centres -> source-space
    vertices; P2-F1) with unique names (P2-F2) and extracted from a
    ``pick_ori='normal'`` source estimate with ``mode='mean_flip'`` so the output is
    **signed** (P2-F4) and can represent N1/P2 polarity.
    """

    name = "source_reconstruction"
    version = "0.2.0"
    description = "Source localization and ROI extraction"

    def __init__(self, config: Dict[str, Any], output_dir: Path, task_config: Optional[Dict[str, Any]] = None):
        super().__init__(config, output_dir)
        self.method = config.get("method", "sLORETA")
        self.parcellation = config.get("parcellation", "conn_networks")
        self.snr = config.get("snr", 3.0)
        self.loose = config.get("loose", 0.2)
        self.depth = config.get("depth", 0.8)
        self.spacing = config.get("spacing", "oct6")
        self.roi_radius = config.get("roi_radius", 10.0)
        self.roi_radius_midline = config.get("roi_radius_midline", 15.0)
        self.roi_radius_overrides = config.get("roi_radius_overrides") or {}
        self.roi_min_vertices = config.get("roi_min_vertices", 3)
        # F2: store the config value as-is (may be None) and resolve lazily in the
        # ``subjects_dir`` property. ``fetch_fsaverage`` downloads ~500 MB when
        # fsaverage is not cached, so ``__init__`` must not call it (~25 unmarked
        # tests construct the module).
        subjects_dir = config.get("subjects_dir")
        self._subjects_dir: Optional[Path] = (
            Path(subjects_dir) if subjects_dir is not None else None
        )
        self.task_config = task_config  # Optional: for grouping epochs by condition
        # ERP polarity/amplitude features are defined on evoked (2-D) ROI time courses;
        # per-trial arrays are opt-in (plan §0.5: default mode='evoked').
        self.trial_level = config.get("trial_level", False)
        # Labels are built on the source space actually used by the inverse operator and
        # cached on ``self`` keyed by that source space's subject + vertno arrays (N1/CF-1).
        self._labels: Optional[List[Any]] = None
        self._label_info: Optional[Dict[str, Any]] = None
        self._labels_key: Optional[Tuple[Optional[str], Tuple[Tuple[int, ...], ...]]] = None

    @property
    def subjects_dir(self) -> Path:
        """Subjects dir containing fsaverage (F2).

        Returns the configured value when set; otherwise resolves
        ``mne.datasets.fetch_fsaverage(verbose=False)`` on **first use** and caches
        the parent directory. Construction never touches the network; both
        ``_get_conn_labels`` and ``_get_template_forward`` read this property, so
        labels and the template forward always come from one fsaverage copy.
        """
        if self._subjects_dir is None:
            self._subjects_dir = Path(
                mne.datasets.fetch_fsaverage(verbose=False)
            ).parent
        return self._subjects_dir

    def validate_input(self, data: Any) -> bool:
        """Return True only for epochs with usable EEG montage positions.

        Deviation from the :class:`~eegcpm.pipeline.base.BaseModule` contract: this
        override returns ``False`` instead of raising for invalid input (other modules
        raise). This is safe only because :meth:`process` calls this method first and
        fails loudly with ``success=False``; ``pipeline/executor.py`` ignores the
        return value of ``validate_input`` (it only relies on exceptions).
        """
        if not isinstance(data, mne.BaseEpochs):
            return False
        eeg_picks = mne.pick_types(data.info, eeg=True)
        if len(eeg_picks) == 0:
            return False
        positions = np.array(
            [data.info["chs"][p]["loc"][:3] for p in eeg_picks], dtype=float
        )
        return bool(
            np.all(np.isfinite(positions))
            and np.all(np.any(positions != 0.0, axis=1))
        )

    def process(
        self,
        data: mne.BaseEpochs,
        subject: Optional[Any] = None,
        forward: Optional[mne.Forward] = None,
        **kwargs,
    ) -> ModuleResult:
        """
        Perform source reconstruction.

        Args:
            data: Epochs data (baseline-corrected; the baseline window is used as the
                noise covariance window)
            subject: Subject info
            forward: Optional pre-computed forward model

        Returns:
            ModuleResult with source estimates
        """
        start_time = time.time()

        if not self.validate_input(data):
            return ModuleResult(
                success=False,
                module_name=self.name,
                execution_time_seconds=time.time() - start_time,
                errors=["Missing montage: EEG channel positions are absent or zero"],
            )

        # EOG/ECG channels must not enter the covariance or the inverse operator.
        # The pick runs inside the ``try`` below so that an all-bad-channels case
        # (``pick`` raises ValueError) returns ``success=False`` instead of
        # propagating out of ``process`` (B2 rev1 robustness fix).
        output_files = []
        warnings = []

        try:
            epochs = data.copy().pick("eeg", exclude="bads")

            # Ensure average reference projector is set (required for source modeling)
            if not any('EEG' in proj['desc'] and 'average' in proj['desc'].lower()
                      for proj in epochs.info['projs']):
                epochs.set_eeg_reference(projection=True, verbose=False)
                warnings.append("Applied average EEG reference projector (required for source modeling)")

            # Get or compute forward model
            if forward is None:
                forward, warnings_fwd = self._get_template_forward(epochs.info)
                warnings.extend(warnings_fwd)

            # Compute noise covariance (baseline-corrected epochs, P2-F11 estimator).
            # Runs after the forward so a forward failure remains errors[0].
            noise_cov, cov_warnings = self._compute_noise_covariance(epochs)
            warnings.extend(cov_warnings)

            # Make inverse operator
            inverse_operator = make_inverse_operator(
                epochs.info,
                forward,
                noise_cov,
                loose=self.loose,
                depth=self.depth,
                verbose=False,
            )

            # Anatomical CONN labels on the source space the inverse operator uses.
            labels, label_info = self._get_conn_labels(inverse_operator["src"])

            lambda2 = 1.0 / self.snr ** 2

            if self.trial_level:
                roi_data, conditions = self._process_trial_level(
                    epochs, inverse_operator, lambda2, labels, label_info
                )
                stcs: Dict[str, mne.SourceEstimate] = {}
                n_trials = int(roi_data[conditions[0]].shape[0]) if conditions else 0
            else:
                stcs, roi_data, conditions = self._process_evoked(
                    epochs, inverse_operator, lambda2, labels, label_info
                )
                n_trials = 0

            n_sources = int(sum(len(s["vertno"]) for s in inverse_operator["src"]))

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
                    "n_sources": n_sources,
                    "n_rois": len(label_info["roi_names"]),
                    "n_trials": n_trials,
                    "conditions": conditions,
                    "roi_names": list(label_info["roi_names"]),
                    "roi_vertex_counts": label_info["roi_vertex_counts"].tolist(),
                    "excluded_rois": list(label_info["excluded_rois"]),
                    "sfreq": float(epochs.info["sfreq"]),
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

    def _compute_noise_covariance(
        self, epochs: mne.BaseEpochs
    ) -> Tuple[mne.Covariance, List[str]]:
        """Estimate the noise covariance from the baseline (P2-F11).

        Uses the approved estimator (``shrunk``, ``rank=None``,
        ``keep_sample_mean=True``, ``tmin=None``, ``tmax=0.0``) with loud guards.
        Baseline-corrected epochs are correct for epoch covariance (audit H4 was
        refuted); the guards only verify that a baseline actually exists.
        """
        warnings: List[str] = []
        if epochs.baseline is None:
            raise ValueError(
                "epochs must be baseline-corrected (e.g. baseline=(None, 0)) so a "
                "noise covariance window exists"
            )
        if epochs.tmin >= 0:
            raise ValueError(
                f"no pre-stimulus interval: epochs.tmin={epochs.tmin} >= 0; a "
                "negative tmin is required to estimate the noise covariance"
            )
        data = epochs.get_data()
        if not np.isfinite(data).all():
            raise ValueError(
                "non-finite values in epochs data; cannot estimate the noise covariance"
            )

        n_prestim = int(round(-epochs.tmin * epochs.info["sfreq"]))
        n_eeg = len(mne.pick_types(epochs.info, eeg=True, exclude="bads"))
        if len(epochs) * n_prestim < 10 * n_eeg:
            warnings.append(
                f"Few baseline samples for the noise covariance: "
                f"{len(epochs)} epochs x {n_prestim} samples < 10 x {n_eeg} channels"
            )

        noise_cov = mne.compute_covariance(
            epochs,
            tmin=None,
            tmax=0.0,
            method="shrunk",
            rank=None,
            keep_sample_mean=True,
            verbose=False,
        )
        return noise_cov, warnings

    @staticmethod
    def _source_space_key(
        src: mne.SourceSpaces,
    ) -> Optional[Tuple[Optional[str], Tuple[Tuple[int, ...], ...]]]:
        """Cache key for a source space: subject name + vertno arrays (N1/CF-1).

        The subject is part of the key because ``build_conn_labels`` rejects
        non-fsaverage source spaces (N2): if a cached lookup skipped that check, a
        scaled template with the same vertno but a different ``subject_his_id``
        would silently reuse fsaverage labels.

        Returns ``None`` for a malformed source space so ``build_conn_labels``
        (which validates and raises) is always called instead of the cache serving
        stale labels.
        """
        if len(src) != 2:
            return None
        return (
            getattr(src, "_subject", None),
            tuple(
                tuple(np.asarray(src[h]["vertno"], dtype=int).tolist()) for h in (0, 1)
            ),
        )

    def _get_conn_labels(self, src: mne.SourceSpaces) -> Tuple[List[Any], Dict[str, Any]]:
        """Build (and cache) the CONN labels for this module instance (P2-F1).

        The cache is keyed on the source space subject + vertno arrays (N1/CF-1):
        labels built for one source space must never be reused for another, or their
        vertex numbers silently refer to the wrong surface. Different key →
        **rebuild** (not raise): reusing one module instance across subjects with
        individual head models is legitimate. The N2 subject guard in
        ``build_conn_labels`` still rejects individual-subject source spaces mapped
        onto the fsaverage surface (including on cache hits, via the subject key).
        """
        key = self._source_space_key(src)
        if (
            self._labels is None
            or self._label_info is None
            or key is None
            or self._labels_key != key
        ):
            self._labels, self._label_info = build_conn_labels(
                src,
                self.subjects_dir,
                radius_mm=self.roi_radius,
                midline_radius_mm=self.roi_radius_midline,
                radius_overrides=self.roi_radius_overrides,
                min_vertices=self.roi_min_vertices,
            )
            self._labels_key = key
        return self._labels, self._label_info

    @staticmethod
    def _roi_metadata(label_info: Dict[str, Any]) -> Dict[str, Any]:
        """ROI metadata block stored in every npz (P2-F1/F3)."""
        return {
            "roi_names": np.asarray(label_info["roi_names"], dtype=str),
            "roi_coords_mni": np.asarray(label_info["roi_coords_mni"], dtype=float),
            "roi_radius_mm": np.asarray(label_info["roi_radius_mm"], dtype=float),
            "roi_vertex_counts": np.asarray(label_info["roi_vertex_counts"], dtype=int),
            "roi_hemi": np.asarray(label_info["roi_hemi"], dtype=str),
            "excluded_rois": np.asarray(label_info["excluded_rois"], dtype=str),
        }

    def _condition_groups(self, epochs: mne.BaseEpochs) -> Dict[str, mne.BaseEpochs]:
        """Group epochs by condition (task-config names, else ``event_id`` keys)."""
        if self.task_config and "conditions" in self.task_config:
            groups: Dict[str, mne.BaseEpochs] = {}
            for condition_spec in self.task_config["conditions"]:
                condition_name = condition_spec["name"]
                event_codes = condition_spec.get("event_codes", [])
                matching_events = [
                    str(ec) for ec in event_codes if str(ec) in epochs.event_id
                ]
                if matching_events:
                    groups[condition_name] = epochs[matching_events]
            return groups
        return {condition: epochs[condition] for condition in epochs.event_id}

    def _process_evoked(
        self,
        epochs: mne.BaseEpochs,
        inverse_operator: Any,
        lambda2: float,
        labels: List[Any],
        label_info: Dict[str, Any],
    ) -> Tuple[Dict[str, mne.SourceEstimate], Dict[str, Any], List[str]]:
        """Average per condition and extract signed ROI time courses (P2-F4)."""
        stcs: Dict[str, mne.SourceEstimate] = {}
        roi_data = self._roi_metadata(label_info)
        conditions: List[str] = []

        for condition, condition_epochs in self._condition_groups(epochs).items():
            evoked = condition_epochs.average()
            stc = apply_inverse(
                evoked,
                inverse_operator,
                lambda2=lambda2,
                method=self.method,
                pick_ori="normal",
                verbose=False,
            )
            stcs[condition] = stc
            roi_data[condition] = mne.extract_label_time_course(
                stc,
                labels,
                inverse_operator["src"],
                mode="mean_flip",
                allow_empty=False,
                verbose=False,
            )
            roi_data[f"{condition}_times"] = stc.times
            conditions.append(condition)

        roi_data["sfreq"] = float(epochs.info["sfreq"])
        return stcs, roi_data, conditions

    def _process_trial_level(
        self,
        epochs: mne.BaseEpochs,
        inverse_operator: Any,
        lambda2: float,
        labels: List[Any],
        label_info: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Per-trial inversion via ``apply_inverse_epochs`` (P2-F12 scaffolding).

        Trial lambda2 (snr_trial) and method_params are wired in B5; the exact same
        ``pick_ori='normal'`` / ``mean_flip`` extraction as the evoked path is used so
        single-trial output is signed as well.
        """
        roi_data = self._roi_metadata(label_info)
        conditions: List[str] = []

        for condition_name, condition_epochs in self._condition_groups(epochs).items():
            stcs = apply_inverse_epochs(
                condition_epochs,
                inverse_operator,
                lambda2,
                method=self.method,
                pick_ori="normal",
                return_generator=True,
                verbose=False,
            )
            tcs = mne.extract_label_time_course(
                stcs,
                labels,
                inverse_operator["src"],
                mode="mean_flip",
                allow_empty=False,
                return_generator=False,
                verbose=False,
            )
            # (n_trials, n_rois, n_times)
            roi_data[condition_name] = np.stack(tcs)
            roi_data[f"{condition_name}_times"] = condition_epochs.times
            conditions.append(condition_name)

        roi_data["sfreq"] = float(epochs.info["sfreq"])
        return roi_data, conditions

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
            # fsaverage template from the same subjects dir the labels use (F2).
            subjects_dir = self.subjects_dir
            fs_dir = subjects_dir / "fsaverage"

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

    def get_output_spec(self) -> Dict[str, str]:
        return {
            "stcs": "Dict of condition -> SourceEstimate",
            "roi_data": "Dict with ROI time courses",
            "inverse_operator": "MNE InverseOperator",
        }
