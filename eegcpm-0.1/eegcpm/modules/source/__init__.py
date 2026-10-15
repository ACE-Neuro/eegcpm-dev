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

from eegcpm.core.config import SourceConfig
from eegcpm.pipeline.base import ModuleResult, EpochsModule
from eegcpm.modules.source.labels import build_conn_labels
from eegcpm.modules.qc.source_qc import SourceQC


# Output npz keys a condition name must never collide with (B4-K2/N2): the label
# metadata block (plus any ``roi_*`` key) and the per-condition companion keys
# ``{cond}_times`` / ``{cond}_nave``.
RESERVED_CONDITION_NAMES = frozenset(
    {"roi_names", "sfreq", "excluded_rois", "excluded_reasons"}
)


class SourceReconstructionModule(EpochsModule):
    """
    Source reconstruction from sensor-level EEG.

    Supports:
    - dSPM, sLORETA, eLORETA, MNE
    - Template or individual head models
    - ROI-based parcellation (CONN 32 networks default)
    - Evoked ERP extraction (default), single trials, or both

    ROI time courses are anatomically mapped (CONN MNI centres -> source-space
    vertices; P2-F1) with unique names (P2-F2) and extracted from a
    ``pick_ori='normal'`` source estimate with ``mode='mean_flip'`` so the output is
    **signed** (P2-F4) and can represent N1/P2 polarity.

    Configuration (P2-F5) is validated through :class:`~eegcpm.core.config.SourceConfig`
    (``extra='forbid'``): nested ``forward``/``inverse`` sections are consumed and the
    legacy flat dialect raises instead of silently no-op'ing.
    """

    name = "source_reconstruction"
    version = "0.2.0"
    description = "Source localization and ROI extraction"

    def __init__(self, config: Dict[str, Any], output_dir: Path, task_config: Optional[Dict[str, Any]] = None):
        super().__init__(config, output_dir)
        # P2-F5: one validated schema. Flat keys (``snr``, ``trial_level``,
        # ``spacing``, ...) raise ValidationError instead of being ignored.
        self.cfg = SourceConfig.model_validate(config)
        # Test-facing attributes (kept stable across the B4 migration).
        self.method = self.cfg.method
        self.parcellation = self.cfg.parcellation
        self.snr = self.cfg.inverse.snr
        self.snr_trial = self.cfg.inverse.snr_trial
        self.loose = self.cfg.inverse.loose
        self.depth = self.cfg.inverse.depth
        self.spacing = self.cfg.forward.spacing
        self.roi_radius = self.cfg.roi_radius
        self.roi_radius_midline = self.cfg.roi_radius_midline
        self.roi_radius_overrides = dict(self.cfg.roi_radius_overrides)
        self.roi_min_vertices = self.cfg.roi_min_vertices
        self.mode = self.cfg.mode
        self.min_trials = self.cfg.min_trials
        self.equalize_trials = self.cfg.equalize_trials
        self.random_seed = self.cfg.random_seed
        # F2: store the config value as-is (may be None) and resolve lazily in the
        # ``subjects_dir`` property. ``fetch_fsaverage`` downloads ~500 MB when
        # fsaverage is not cached, so ``__init__`` must not call it (~25 unmarked
        # tests construct the module).
        subjects_dir = self.cfg.subjects_dir
        self._subjects_dir: Optional[Path] = (
            Path(subjects_dir) if subjects_dir is not None else None
        )
        self.task_config = task_config  # Optional: for grouping epochs by condition
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

            # Conditions: requested/missing/skipped/equalized (P2-F7/F8/F9). This runs
            # after the inverse operator so the covariance guards and the forward
            # error keep their precedence (test_source.py asserts the forward error
            # is errors[0]; the covariance guard tests assert their own messages).
            selected, selection = self._select_condition_epochs(epochs)
            warnings.extend(selection["warnings"])

            if not selected:
                # Loud failure: never write an empty/zero-trial output (P2-F8/F9).
                return ModuleResult(
                    success=False,
                    module_name=self.name,
                    execution_time_seconds=time.time() - start_time,
                    errors=[
                        "no conditions produced; requested="
                        f"{selection['conditions_requested']}, missing="
                        f"{selection['conditions_missing']}, skipped="
                        f"{selection['conditions_skipped']}"
                    ],
                    warnings=warnings,
                    metadata=self._selection_metadata(selection),
                )

            # Anatomical CONN labels on the source space the inverse operator uses.
            labels, label_info = self._get_conn_labels(inverse_operator["src"])

            lambda2 = 1.0 / self.snr ** 2
            # NOTE (B5): the single-trial path still uses the evoked lambda2; the
            # snr_trial -> lambda2_trial wiring lands in B5 (task 2) with method_params.
            stcs: Dict[str, mne.SourceEstimate] = {}
            evoked_data: Optional[Dict[str, Any]] = None
            trial_data: Optional[Dict[str, Any]] = None

            if self.mode in ("evoked", "both"):
                stcs, evoked_data = self._process_evoked(
                    selected, inverse_operator, lambda2, labels, label_info, selection["nave"]
                )
            if self.mode in ("trials", "both"):
                trial_data = self._process_trial_level(
                    selected, inverse_operator, lambda2, labels, label_info, selection["nave"]
                )

            n_sources = int(sum(len(s["vertno"]) for s in inverse_operator["src"]))
            n_trials = 0 if self.mode == "evoked" else int(sum(selection["nave"].values()))
            conditions = list(selection["nave"].keys())

            # Save results
            subject_id = subject.id if subject else "unknown"

            # Save source estimates (evoked mode only; honour output.save_stc).
            if self.cfg.output.save_stc:
                for condition, stc in stcs.items():
                    stc_path = self.output_dir / f"{subject_id}_{condition}_stc"
                    stc.save(stc_path, overwrite=True, verbose=False)
                    output_files.append(Path(f"{stc_path}-lh.stc"))

            # Save ROI data (evoked -> 2-D; trials -> 3-D; both -> two files).
            if self.cfg.output.save_roi_tc:
                if evoked_data is not None:
                    roi_path = self.output_dir / f"{subject_id}_roi_tc.npz"
                    np.savez(roi_path, **evoked_data)
                    output_files.append(roi_path)
                if trial_data is not None:
                    roi_path = self.output_dir / f"{subject_id}_roi_tc_trials.npz"
                    np.savez_compressed(roi_path, **trial_data)
                    output_files.append(roi_path)

            # Generate QC report
            qc_metrics = []
            qc_roi_data = evoked_data if evoked_data is not None else trial_data
            if self.cfg.output.generate_qc and qc_roi_data:
                try:
                    qc = SourceQC(self.output_dir)

                    # Get session info
                    session_id = subject.session if hasattr(subject, 'session') else "01"

                    # Prepare data for QC
                    qc_data = {
                        'stcs': stcs,
                        'roi_data': qc_roi_data,
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

            data_for_next = evoked_data if evoked_data is not None else trial_data
            metadata = {
                "method": self.method,
                "parcellation": self.parcellation,
                "mode": self.mode,
                "n_sources": n_sources,
                "n_rois": len(label_info["roi_names"]),
                "n_trials": n_trials,
                "conditions": conditions,
                "roi_names": list(label_info["roi_names"]),
                "roi_vertex_counts": label_info["roi_vertex_counts"].tolist(),
                "excluded_rois": list(label_info["excluded_rois"]),
                "sfreq": float(epochs.info["sfreq"]),
                "qc_metrics": {m.name: m.value for m in qc_metrics} if qc_metrics else {},
            }
            metadata.update(self._selection_metadata(selection))

            return ModuleResult(
                success=True,
                module_name=self.name,
                execution_time_seconds=time.time() - start_time,
                outputs={
                    "data": data_for_next if data_for_next else stcs,
                    "stcs": stcs,
                    "roi_data": data_for_next,
                    "inverse_operator": inverse_operator,
                },
                output_files=output_files,
                warnings=warnings,
                metadata=metadata,
            )

        except Exception as e:
            return ModuleResult(
                success=False,
                module_name=self.name,
                execution_time_seconds=time.time() - start_time,
                errors=[str(e)],
            )

    # ------------------------------------------------------------------
    # Condition selection (P2-F7/F8/F9)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_condition_names(names: List[str]) -> None:
        """Raise ``ValueError`` for condition names that collide with output keys.

        Loud guard (B4-K2/N2):

        - **duplicates**: the task-config loop would silently keep only the last
          spec (probe: two ``{"name": "dup"}`` -> requested ``['dup']``, first
          spec lost);
        - **reserved names**: ``roi_names``/``sfreq``/``excluded_rois``/
          ``excluded_reasons`` (and any ``roi_*`` key) are label-metadata keys
          written into the same npz;
        - **``*_times`` / ``*_nave``**: per-condition companion keys written by
          ``_process_evoked``/``_process_trial_level``.
        """
        seen: Dict[str, int] = {}
        for name in names:
            seen[name] = seen.get(name, 0) + 1
        duplicates = sorted(name for name, count in seen.items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate condition name(s): {duplicates}")

        for name in names:
            if name in RESERVED_CONDITION_NAMES or name.startswith("roi_"):
                raise ValueError(
                    f"condition name {name!r} is reserved: it collides with an "
                    f"output metadata key; rename the condition"
                )
            if name.endswith("_times") or name.endswith("_nave"):
                raise ValueError(
                    f"condition name {name!r} must not end in '_times'/'_nave': "
                    f"it collides with the per-condition companion keys"
                )

    def _select_condition_epochs(
        self, epochs: mne.BaseEpochs
    ) -> Tuple[Dict[str, mne.BaseEpochs], Dict[str, Any]]:
        """Select the epochs per condition, loudly.

        - ``requested`` = task-config condition names (else every ``event_id`` key).
        - Duplicate/reserved condition names -> ``ValueError`` (B4-K2/N2).
        - No matching events (or zero trials) -> warning + ``conditions_missing``.
        - ``n < min_trials`` -> warning + ``conditions_skipped`` (``below_min_trials``).
        - ``equalize_trials=N`` -> seeded random subset (``random_seed``), iterated in
          sorted condition order so the draw does not depend on task-config order;
          conditions with ``n < N`` are skipped (``below_equalize_trials``).

        Never returns an empty condition selection, so ``.average()`` is never called
        on zero trials and zero-trial arrays are never written.
        """
        task_conditions: Dict[str, List[str]] = {}
        if self.task_config and "conditions" in self.task_config:
            specs = self.task_config["conditions"]
            self._validate_condition_names([str(spec["name"]) for spec in specs])
            for spec in specs:
                name = str(spec["name"])
                task_conditions[name] = [str(c) for c in spec.get("event_codes", [])]
            requested = list(task_conditions)
        else:
            requested = list(epochs.event_id)
            self._validate_condition_names(requested)
            task_conditions = {name: [name] for name in requested}

        missing: List[str] = []
        skipped: Dict[str, Dict[str, Any]] = {}
        selected: Dict[str, mne.BaseEpochs] = {}
        warnings: List[str] = []

        for name in requested:
            matching = [code for code in task_conditions[name] if code in epochs.event_id]
            n_trials = len(epochs[matching]) if matching else 0
            if n_trials == 0:
                warnings.append(f"condition '{name}': no matching events")
                missing.append(name)
                continue
            if n_trials < self.min_trials:
                warnings.append(
                    f"condition '{name}': {n_trials} trials < min_trials={self.min_trials}"
                )
                skipped[name] = {"n_trials": int(n_trials), "reason": "below_min_trials"}
                continue
            selected[name] = epochs[matching]

        if self.equalize_trials is not None:
            n_target = int(self.equalize_trials)
            rng = np.random.default_rng(self.random_seed)
            for name in sorted(selected):
                condition_epochs = selected[name]
                n_trials = len(condition_epochs)
                if n_trials < n_target:
                    warnings.append(
                        f"condition '{name}': {n_trials} trials < equalize_trials={n_target}"
                    )
                    skipped[name] = {
                        "n_trials": int(n_trials),
                        "reason": "below_equalize_trials",
                    }
                    continue
                idx = np.sort(rng.choice(n_trials, n_target, replace=False))
                selected[name] = condition_epochs[idx]
            for name in skipped:
                selected.pop(name, None)

        nave = {name: int(len(condition_epochs)) for name, condition_epochs in selected.items()}
        return selected, {
            "conditions_requested": requested,
            "conditions_missing": missing,
            "conditions_skipped": skipped,
            "nave": nave,
            "warnings": warnings,
        }

    def _selection_metadata(self, selection: Dict[str, Any]) -> Dict[str, Any]:
        """Bookkeeping metadata for condition selection (P2-F7/F8/F9)."""
        return {
            "conditions_requested": list(selection["conditions_requested"]),
            "conditions_produced": list(selection["nave"].keys()),
            "conditions_missing": list(selection["conditions_missing"]),
            "conditions_skipped": dict(selection["conditions_skipped"]),
            "nave": dict(selection["nave"]),
            "nave_confound_warning": bool(
                self.method in {"dSPM", "sLORETA", "eLORETA"}
                and self.equalize_trials is None
            ),
        }

    # ------------------------------------------------------------------
    # Covariance
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Extraction paths
    # ------------------------------------------------------------------

    def _process_evoked(
        self,
        selected: Dict[str, mne.BaseEpochs],
        inverse_operator: Any,
        lambda2: float,
        labels: List[Any],
        label_info: Dict[str, Any],
        nave: Dict[str, int],
    ) -> Tuple[Dict[str, mne.SourceEstimate], Dict[str, Any]]:
        """Average per condition and extract signed ROI time courses (P2-F4).

        Returns ``(stcs, roi_data)`` with ``roi_data[cond]`` shaped
        ``(n_rois, n_times)`` plus ``{cond}_times`` / ``{cond}_nave`` / ``sfreq``
        and the label metadata.
        """
        stcs: Dict[str, mne.SourceEstimate] = {}
        roi_data = self._roi_metadata(label_info)

        for condition, condition_epochs in selected.items():
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
            roi_data[f"{condition}_nave"] = int(nave[condition])

        roi_data["sfreq"] = float(condition_epochs.info["sfreq"])
        return stcs, roi_data

    def _process_trial_level(
        self,
        selected: Dict[str, mne.BaseEpochs],
        inverse_operator: Any,
        lambda2: float,
        labels: List[Any],
        label_info: Dict[str, Any],
        nave: Dict[str, int],
    ) -> Dict[str, Any]:
        """Per-trial inversion via ``apply_inverse_epochs`` (P2-F12 scaffolding).

        The trial lambda2 (``snr_trial``) and ``method_params`` are wired in B5; the
        exact same ``pick_ori='normal'`` / ``mean_flip`` extraction as the evoked
        path is used so single-trial output is signed as well. Each condition array
        is ``(n_trials, n_rois, n_times)`` with at least ``min_trials`` trials.
        """
        roi_data = self._roi_metadata(label_info)

        for condition_name, condition_epochs in selected.items():
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
            roi_data[f"{condition_name}_nave"] = int(nave[condition_name])

        roi_data["sfreq"] = float(condition_epochs.info["sfreq"])
        return roi_data

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
