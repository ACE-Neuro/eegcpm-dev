"""Source reconstruction module.

ROI time courses are extracted with ``mode='mean_flip'`` (P2-F4). Two
documented caveats (user decision 2026-10-15, plan §12 item 4 — keep
``mean_flip``, do not switch to ``pca_flip``):

1. **Attenuation under uniform patches.** In a uniform-patch simulation
   (same dSPM / loose-0.2 pipeline, 40 epochs) the injected ROI was dominant
   in **11/30** ROIs under ``mean_flip`` vs **22/30** under ``pca_flip``: the
   signed mean of the surface normals within a label cancels part of the
   source amplitude. Mean absolute source energy still ranks the true ROIs
   first — the loss is cancellation during extraction, not the inverse — but
   single-ROI ERP *amplitudes* are attenuated and must not be compared
   against sensor-level amplitudes.
2. **Midline polarity convention.** Midline ROIs (``BiHemiLabel``, |x| ≤ 5 mm)
   have near-tied flip axes (|mean(dot)| 0.008–0.21; e.g. ``MPFC`` lh −0.70 /
   rh +0.71), so the *sign* of a midline time course follows the
   ``mean_flip`` polarity convention rather than anatomy, and one ROI
   (``Language.IFG_R``) comes out polarity-inverted. Cross-ROI sign
   comparisons on midline labels are therefore not meaningful without an
   explicit polarity convention; lateral ROIs are stable.

The extraction itself is anatomically mapped (CONN MNI centres -> source-space
vertices), uniquely named, and **signed** (``pick_ori='normal'``) so N1/P2
polarity survives. See the class docstring for the hardening/provenance
behaviour.
"""

import glob
import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import scipy
import yaml
from mne.io.constants import FIFF

# In mne 1.13 the inverse operators live under mne.minimum_norm; they are imported as
# module-level names so tests can monkeypatch ``eegcpm.modules.source.apply_inverse_epochs``.
from mne.minimum_norm import (
    apply_inverse,
    apply_inverse_epochs,
    make_inverse_operator,
    write_inverse_operator,
)
from scipy.spatial import cKDTree

from eegcpm.core.config import SourceConfig
from eegcpm.core.provenance import (
    atomic_save,
    atomic_write_json,
    atomic_write_npz,
    atomic_write_text,
    package_version,
    sha256_file,
    sha256_json,
    stale_output_patterns,
    to_jsonable,
)
from eegcpm.pipeline.base import ModuleResult, EpochsModule
from eegcpm.modules.source.labels import build_conn_labels
from eegcpm.modules.qc.source_qc import (
    SourceQC,
    compute_crosstalk_matrix,
    compute_crosstalk_matrix_numpy,
)


# Output npz keys a condition name must never collide with (B4-K2/N2): the label
# metadata block (plus any ``roi_*`` key) and the per-condition companion keys
# ``{cond}_times`` / ``{cond}_nave``.
RESERVED_CONDITION_NAMES = frozenset(
    {"roi_names", "sfreq", "excluded_rois", "excluded_reasons"}
)

#: The only accepted BEM for the fsaverage template forward (P2-F13 / plan §0.5).
BEM_FILENAME = "fsaverage-5120-5120-5120-bem-sol.fif"
#: Head->MRI transform of the fsaverage template, relative to ``<subject>/bem``.
TRANS_FILENAME = "fsaverage-trans.fif"
#: Scalp surface used by the montage pre-flight (relative to ``<subject>/bem``).
HEAD_SURFACE_FILENAME = "fsaverage-head.fif"
#: Covariance estimator (P2-F11, approved in plan §0.5).
COV_METHOD = "shrunk"
#: Montage pre-flight thresholds in mm (P2-F13): warn above the median, fail above
#: the maximum. Probe 2026-10-13: GSN-HydroCel-129 on fsaverage gives 7.2/18.7 mm.
MONTAGE_MEDIAN_DISTANCE_MM = 10.0
MONTAGE_MAX_DISTANCE_MM = 25.0


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

    Hardening + provenance (P2-F10/F12/F13, B5):

    - the noise covariance is logged with ``mne.compute_rank`` (data + covariance
      rank) and every run writes ``{subject}_source_config.yaml``,
      ``{subject}-cov.fif``, ``{subject}-inv.fif`` and ``{subject}_provenance.json``
      next to the data artifacts;
    - single trials are inverted with ``lambda2 = 1/snr_trial**2`` via
      ``apply_inverse_epochs`` (never per-trial ``apply_inverse``), and eLORETA with
      a loose orientation constraint gets ``method_params={'force_equal': True}``;
    - the template forward uses the 3-layer fsaverage BEM only (missing → loud
      ``FileNotFoundError``, no single-shell fallback) and is cached per
      montage/spacing/BEM/mne version;
    - the montage is pre-flighted (head-frame positions, cardinal fiducials, and
      the distance of the transformed electrodes to the fsaverage scalp) before the
      forward is computed;
    - artifact writes are atomic (temp + rename) and a re-run removes its own stale
      outputs first, so a failed run never leaves partial or mixed artifacts.
    """

    name = "source_reconstruction"
    version = "0.2.0"
    description = "Source localization and ROI extraction"

    #: Class-level ``(src, forward)`` cache (P2-F13) keyed by
    #: ``sha256(ch_names + rounded loc[:3]) | spacing | BEM | mne version | subjects_dir
    #: | BEM/trans mtime_ns:size``.
    #: Building the fsaverage forward costs ~10 s; a batch of subjects on the same
    #: montage reuses one forward. ``subjects_dir`` is part of the key (beyond the
    #: plan's list) so two different fsaverage copies can never share a forward, and
    #: the BEM/trans mtime+size (B5-A4) invalidates the entry when the template
    #: files are replaced in place. A benign consequence of the class-level cache is
    #: a possible double build under GIL-free concurrency (two threads miss the same
    #: key and both build) — wasteful, never incorrect.
    _FORWARD_CACHE: Dict[str, Tuple[mne.SourceSpaces, mne.Forward]] = {}

    #: Class-level ROI x ROI crosstalk cache (P2-F14/B7 task 8). Keyed by the
    #: forward cache key **plus** the noise-covariance digest, method and lambda2:
    #: the resolution matrix is built from the inverse operator, which is
    #: whitened by the noise covariance, so two subjects with the same montage
    #: must not share a matrix (the plan's "per montage hash" would silently
    #: reuse subject 1's covariance). The cached value is the small
    #: ``(n_rois, n_rois)`` matrices — the primary flip-aligned ``crosstalk``
    #: and the library ``crosstalk_unflipped`` (B7 rev2 fix 6) — not the
    #: ~0.5 GB resolution matrix.
    _CROSSTALK_CACHE: Dict[str, Dict[str, np.ndarray]] = {}

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
        epochs_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> ModuleResult:
        """
        Perform source reconstruction.

        Args:
            data: Epochs data (baseline-corrected; the baseline window is used as the
                noise covariance window)
            subject: Subject info
            forward: Optional pre-computed forward model
            epochs_path: Optional path of the epoch file the data came from; recorded
                in the provenance with its streamed sha256 (P2-F10)

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

            # Get or compute forward model. The montage pre-flight (P2-F13) runs
            # inside the template-forward path, immediately before the forward is
            # computed: head-frame positions, cardinal fiducials, and the
            # fsaverage scalp-distance check. A caller-supplied forward may be
            # built on an individual head model, where the fsaverage trans is
            # meaningless, so the alignment check is recorded as not run.
            montage_check: Dict[str, Any]
            if forward is None:
                forward_source = "template"
                forward, warnings_fwd, montage_check = self._get_template_forward(
                    epochs.info
                )
                warnings.extend(warnings_fwd)
                warnings.extend(montage_check["warnings"])
            else:
                forward_source = "caller"
                montage_check = {
                    "median_mm": None,
                    "max_mm": None,
                    "n_electrodes": None,
                    "status": "not_run: forward supplied by the caller",
                    "warnings": [],
                }

            # Compute noise covariance (baseline-corrected epochs, P2-F11 estimator).
            # Runs after the forward so a forward failure remains errors[0].
            noise_cov, cov_warnings = self._compute_noise_covariance(epochs)
            warnings.extend(cov_warnings)

            # Rank logging (P2-F11, B5 task 1): surfaced in the provenance JSON.
            data_rank = mne.compute_rank(epochs, rank=None, verbose=False)
            cov_rank = mne.compute_rank(noise_cov, info=epochs.info, verbose=False)

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

            # Inverse parameters (P2-F12, B5 task 2): evoked lambda2 from ``snr``,
            # single-trial lambda2 from ``snr_trial`` (a missing kwarg would silently
            # fall back to the evoked lambda2), eLORETA loose orientation needs
            # ``force_equal`` (mne 1.13.2 default is False).
            method_params = self._resolve_method_params(
                self.method, self.loose, self.cfg.inverse.method_params
            )
            lambda2_evoked = 1.0 / self.snr ** 2
            lambda2_trial = 1.0 / self.snr_trial ** 2
            stcs: Dict[str, mne.SourceEstimate] = {}
            evoked_data: Optional[Dict[str, Any]] = None
            trial_data: Optional[Dict[str, Any]] = None

            if self.mode in ("evoked", "both"):
                stcs, evoked_data = self._process_evoked(
                    selected, inverse_operator, lambda2_evoked, labels, label_info,
                    selection["nave"], method_params,
                )
            if self.mode in ("trials", "both"):
                trial_data = self._process_trial_level(
                    selected, inverse_operator, lambda2_trial, labels, label_info,
                    selection["nave"], method_params,
                )

            n_sources = int(sum(len(s["vertno"]) for s in inverse_operator["src"]))
            n_trials = 0 if self.mode == "evoked" else int(sum(selection["nave"].values()))
            conditions = list(selection["nave"].keys())

            # Save results
            subject_id = subject.id if subject else "unknown"

            # Crosstalk matrices (P2-F14/B7 task 8, B7 rev2 fix 6): a QC-only
            # fixed-orientation forward/inverse pair gives the ROI x ROI
            # relative leakage that the connectivity/ERP interpretation needs.
            # Two matrices are produced per (forward, noise covariance, method,
            # lambda2) and cached:
            #   * `crosstalk` — the PRIMARY flip-aligned estimator
            #     (`compute_crosstalk_matrix_numpy(align_signs=True)`), matching
            #     the `mean_flip` ROI extraction;
            #   * `crosstalk_unflipped` — the mne library `get_cross_talk(mode=
            #     'mean')` matrix, kept for continuity/cross-check.
            # Written as `{subject}_crosstalk.npz` next to the other artifacts.
            crosstalk_matrices: Optional[Dict[str, np.ndarray]] = None
            crosstalk_reason: Optional[str] = None
            if self.cfg.output.compute_crosstalk:
                try:
                    crosstalk_matrices = self._get_crosstalk_matrices(
                        forward=forward,
                        info=epochs.info,
                        noise_cov=noise_cov,
                        labels=labels,
                        lambda2=lambda2_evoked,
                    )
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                    # QC-only artifact: recorded, never allowed to invalidate an
                    # otherwise successful reconstruction (the matrix is
                    # diagnostic; the ROI time courses are the product). The
                    # broad-but-explicit tuple covers the known failure modes:
                    # `ValueError` for methods without a resolution matrix
                    # (eLORETA), numeric failures, and mne's runtime errors. A
                    # `MemoryError` is deliberately NOT caught — it must
                    # propagate.
                    crosstalk_reason = f"{type(exc).__name__}: {exc}"
                    warnings.append(f"Crosstalk matrix not computed: {exc}")
            else:
                crosstalk_reason = (
                    "compute_crosstalk disabled by config "
                    "(output.compute_crosstalk=false)"
                )
            crosstalk_matrix = (
                crosstalk_matrices["crosstalk"] if crosstalk_matrices else None
            )
            crosstalk_unflipped = (
                crosstalk_matrices["crosstalk_unflipped"] if crosstalk_matrices else None
            )

            # Artifact writes (B5 task 8): atomic (temp + rename) and stale-safe
            # (a previous run's outputs for this subject are removed first), so a
            # failed write cannot leave partial artifacts behind success=False.
            # Provenance/cov/inv are written whenever the run produces data
            # artifacts; with ``save_stc``/``save_roi_tc`` both off nothing at all
            # is written (B4 contract).
            write_artifacts = bool(self.cfg.output.save_stc or self.cfg.output.save_roi_tc)
            if write_artifacts:
                provenance = self._build_provenance(
                    subject_id=subject_id,
                    epochs=epochs,
                    inverse_operator=inverse_operator,
                    labels=labels,
                    label_info=label_info,
                    selection=selection,
                    lambda2_evoked=lambda2_evoked,
                    lambda2_trial=lambda2_trial,
                    method_params=method_params,
                    data_rank=data_rank,
                    cov_rank=cov_rank,
                    montage_check=montage_check,
                    epochs_path=epochs_path,
                    forward_source=forward_source,
                    crosstalk_matrix=crosstalk_matrix,
                    crosstalk_unflipped=crosstalk_unflipped,
                    crosstalk_reason=crosstalk_reason,
                )
                output_files.extend(
                    self._write_source_artifacts(
                        subject_id=subject_id,
                        stcs=stcs,
                        evoked_data=evoked_data,
                        trial_data=trial_data,
                        noise_cov=noise_cov,
                        inverse_operator=inverse_operator,
                        provenance=provenance,
                        crosstalk_matrix=crosstalk_matrix,
                        crosstalk_unflipped=crosstalk_unflipped,
                    )
                )

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
                        'labels': labels,
                        'roi_min_vertices': self.roi_min_vertices,
                    }
                    if crosstalk_matrix is not None:
                        qc_data['crosstalk_matrix'] = crosstalk_matrix
                        # State which matrix the QC statistics are computed on
                        # (B7 rev2 fix 6).
                        qc_data['crosstalk_matrix_kind'] = 'crosstalk_flip_aligned'

                    # Compute QC
                    qc_result = qc.compute(
                        data=qc_data,
                        subject_id=subject_id,
                        session_id=session_id,
                        sfreq=epochs.info['sfreq'],
                        roi_min_vertices=self.roi_min_vertices,
                    )

                    # Generate HTML report (atomic: temp + rename, B5 task 8)
                    html_filename = f"{subject_id}_ses-{session_id}_source_qc.html"
                    html_path = self.output_dir / html_filename
                    atomic_save(
                        html_path,
                        lambda tmp: qc.generate_html_report(qc_result, output_path=tmp),
                    )
                    output_files.append(html_path)

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
                "data_rank": dict(data_rank),
                "cov_rank": dict(cov_rank),
                "montage_check": montage_check,
                "config_hash": self.config_hash(),
                "run_identity_hash": self.run_identity_hash(selection),
                "resolved_event_codes": to_jsonable(
                    selection["resolved_event_codes"]
                ),
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
          ``_process_evoked``/``_process_trial_level``;
        - **path separators** (``/``, ``\\``): condition names become file names
          (``{subject}_{condition}_stc``), so a ``/`` used to fail the whole run
          with a ``FileNotFoundError`` deep inside ``stc.save`` (B5-N4).
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
            if "/" in name or "\\" in name:
                raise ValueError(
                    f"condition name {name!r} contains a path separator: it would "
                    f"be used as part of an output file name "
                    f"({{subject}}_{{condition}}_stc) and escape the output "
                    f"directory; rename the condition"
                )

    def _select_condition_epochs(
        self, epochs: mne.BaseEpochs
    ) -> Tuple[Dict[str, mne.BaseEpochs], Dict[str, Any]]:
        """Select the epochs per condition, loudly.

        - ``requested`` = task-config condition names (else every ``event_id`` key).
        - Duplicate/reserved condition names -> ``ValueError`` (B4-K2/N2).
        - Matching is by **event-code value** (:meth:`_match_event_codes`, B5 task 7):
          ``np.isin(epochs.events[:, 2], values)`` so a task-config code ``a`` cannot
          drag in an epoch tagged ``a/b`` via MNE's hierarchical-tag lookup, and the
          numeric-code fallback of ``EventSorter.sort_by_condition`` resolves codes
          that are numbers (``7`` or ``"7"``) rather than ``event_id`` names.
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
                # Raw code types are kept (names, ints or numeric strings) so
                # ``_match_event_codes`` can resolve each dialect (B5 task 7).
                task_conditions[name] = list(spec.get("event_codes", []))
            requested = list(task_conditions)
        else:
            requested = list(epochs.event_id)
            self._validate_condition_names(requested)
            task_conditions = {name: [name] for name in requested}

        missing: List[str] = []
        skipped: Dict[str, Dict[str, Any]] = {}
        selected: Dict[str, mne.BaseEpochs] = {}
        warnings: List[str] = []
        # Resolved condition -> event-code mapping (B5-N3): the task-config
        # codes as given *and* the epoch event values they matched. Recorded
        # in the provenance and hashed into the run identity, so a run is
        # attributable to the exact condition mapping that produced it.
        resolved_event_codes: Dict[str, Dict[str, Any]] = {}

        for name in requested:
            event_values = self._match_event_codes(epochs, task_conditions[name])
            resolved_event_codes[name] = {
                "codes": [str(code) for code in task_conditions[name]],
                "matched_event_values": [int(value) for value in event_values],
            }
            mask = (
                np.isin(epochs.events[:, 2], event_values)
                if event_values
                else np.zeros(len(epochs), dtype=bool)
            )
            n_trials = int(np.count_nonzero(mask))
            if n_trials == 0:
                warnings.append(
                    f"condition '{name}': no matching events (codes="
                    f"{task_conditions[name]})"
                )
                missing.append(name)
                continue
            if n_trials < self.min_trials:
                warnings.append(
                    f"condition '{name}': {n_trials} trials < min_trials={self.min_trials}"
                )
                skipped[name] = {"n_trials": int(n_trials), "reason": "below_min_trials"}
                continue
            selected[name] = epochs[mask]

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
            "resolved_event_codes": resolved_event_codes,
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
            "resolved_event_codes": dict(selection["resolved_event_codes"]),
            "nave_confound_warning": bool(
                self.method in {"dSPM", "sLORETA", "eLORETA"}
                and self.equalize_trials is None
            ),
        }

    @staticmethod
    def _match_event_codes(epochs: mne.BaseEpochs, codes: Any) -> List[int]:
        """Resolve task-config event codes to the epoch event **values** (B5 task 7).

        Accepted code dialects:

        - an ``epochs.event_id`` name (exact key) -> that key's value;
        - a numeric value, either an ``int`` or a numeric string (``7`` / ``"7"``),
          matched against every ``event_id`` value (the numeric-code fallback of
          ``EventSorter.sort_by_condition``, ``core/task_config.py``);
        - anything else is ignored (it simply does not match).

        The caller then selects with ``np.isin(epochs.events[:, 2], values)`` instead
        of ``epochs[[names]]``: MNE's hierarchical event tags would make a code ``a``
        also select epochs tagged ``a/b`` (probed 2026-10-15 on 1.13.2).
        """
        values: set = set()
        for code in codes:
            if isinstance(code, str) and code in epochs.event_id:
                values.add(int(epochs.event_id[code]))
                continue
            numeric: Optional[int] = None
            if isinstance(code, bool):
                numeric = None
            elif isinstance(code, (int, np.integer)):
                numeric = int(code)
            elif isinstance(code, str):
                try:
                    numeric = int(code.strip())
                except ValueError:
                    numeric = None
            if numeric is not None:
                values.update(
                    int(value)
                    for value in epochs.event_id.values()
                    if int(value) == numeric
                )
        return sorted(values)

    # ------------------------------------------------------------------
    # Inverse parameters / config identity (P2-F12, B5 tasks 2 and 5)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_method_params(
        method: str, loose: float, user: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extra ``apply_inverse``/``apply_inverse_epochs`` parameters (P2-F12).

        eLORETA with a loose orientation constraint (``0 < loose < 1``) needs
        ``force_equal=True``: mne 1.13.2 defaults it to ``False``, which produces a
        free-orientation solution that is not the smooth loose-orientation
        transition (verified in ``mne/minimum_norm/_eloreta.py``). Explicit user
        parameters win over the derived default.
        """
        if method == "eLORETA" and 0.0 < float(loose) < 1.0:
            return {"force_equal": True, **(user or {})}
        return user

    def config_hash(self) -> str:
        """sha256 of the resolved source config (P2-F10, plan §B5 task 5).

        Hashes ``json.dumps(cfg.model_dump(mode='json'), sort_keys=True)`` — the
        resolved parameters, not the raw YAML bytes — so the same configuration
        always yields the same hash and any parameter change is visible.
        """
        return sha256_json(self.cfg.model_dump(mode="json"))

    def run_identity_hash(self, selection: Dict[str, Any]) -> str:
        """sha256 of the run identity: config hash + resolved condition mapping.

        B5-N3: ``config_hash`` alone does not identify a run — two runs with the
        same ``SourceConfig`` can select different conditions (a task config
        whose codes were translated differently, or epochs with different
        ``event_id`` values). The identity combines the resolved
        ``{condition: codes, matched_event_values}`` mapping with the config
        hash, so the features consumer can tell whether two source archives
        were produced by the same *run identity*, not merely the same config.
        """
        payload = {
            "config_hash": self.config_hash(),
            "resolved_event_codes": to_jsonable(
                selection["resolved_event_codes"]
            ),
        }
        return sha256_json(payload)

    # ------------------------------------------------------------------
    # Montage pre-flight (P2-F13, B5 task 4)
    # ------------------------------------------------------------------

    @staticmethod
    def _trans_path(subjects_dir: Union[str, Path]) -> Path:
        """fsaverage head->MRI transform the template forward must use."""
        return Path(subjects_dir) / "fsaverage" / "bem" / TRANS_FILENAME

    @staticmethod
    def _load_bem(subjects_dir: Union[str, Path]) -> mne.bem.ConductorModel:
        """Load the fsaverage 3-layer BEM solution (P2-F13).

        The 3-layer BEM is the only accepted model: a single-shell/1-layer BEM
        raises for EEG, so a missing file fails loudly instead of silently falling
        back to an unusable head model (plan §0.5).
        """
        bem_path = Path(subjects_dir) / "fsaverage" / "bem" / BEM_FILENAME
        if not bem_path.is_file():
            raise FileNotFoundError(
                f"3-layer BEM solution not found: {bem_path}; the fsaverage template "
                "forward requires the 3-layer BEM (no single-shell fallback)"
            )
        return mne.read_bem_solution(bem_path, verbose=False)

    @staticmethod
    def _preflight_montage(
        info: mne.Info, subjects_dir: Union[str, Path]
    ) -> Dict[str, Any]:
        """Validate the montage before the template forward (P2-F13, B5 task 4).

        Three checks, all loud:

        (a) every EEG channel has finite, non-zero ``loc[:3]`` (message contains
            "montage");
        (b) ``info['dig']`` carries the LPA/Nasion/RPA cardinal points and every
            digitised point is in the head frame (message contains "fiducials").
            Note (B5-A3): mne's ``set_montage`` fabricates these cardinal points,
            so (b) only catches dig lists without fiducials (or stripped
            fiducials); the scalp-distance check (c) is the accepted real guard;
        (c) the EEG positions, transformed with
            ``<subjects_dir>/fsaverage/bem/fsaverage-trans.fif`` (head->MRI), are
            within reach of the ``fsaverage-head.fif`` scalp vertices
            (``scipy.spatial.cKDTree``): a median above
            :data:`MONTAGE_MEDIAN_DISTANCE_MM` warns, a maximum above
            :data:`MONTAGE_MAX_DISTANCE_MM` raises.

        Returns ``{'median_mm', 'max_mm', 'n_electrodes', 'status', 'warnings'}``
        for the provenance JSON (task 5).
        """
        eeg_picks = mne.pick_types(info, eeg=True)
        if len(eeg_picks) == 0:
            raise ValueError("Missing montage: no EEG channels to localise")
        positions = np.array(
            [info["chs"][pick]["loc"][:3] for pick in eeg_picks], dtype=float
        )
        if not np.all(np.isfinite(positions)) or not np.all(
            np.any(positions != 0.0, axis=1)
        ):
            raise ValueError(
                "Missing montage: EEG channel positions are absent or zero"
            )

        dig = list(info.get("dig") or [])
        cardinal_idents = {
            int(point["ident"])
            for point in dig
            if int(point["kind"]) == int(FIFF.FIFFV_POINT_CARDINAL)
        }
        fiducials = {
            "LPA": int(FIFF.FIFFV_POINT_LPA),
            "Nasion": int(FIFF.FIFFV_POINT_NASION),
            "RPA": int(FIFF.FIFFV_POINT_RPA),
        }
        missing_fiducials = sorted(
            name for name, ident in fiducials.items() if ident not in cardinal_idents
        )
        if missing_fiducials:
            raise ValueError(
                "Missing fiducials: info['dig'] has no cardinal "
                + ", ".join(missing_fiducials)
                + " point(s); a digitised montage with fiducials is required for the "
                "fsaverage template forward"
            )
        bad_frames = sorted(
            {int(point["coord_frame"]) for point in dig}
            - {int(FIFF.FIFFV_COORD_HEAD)}
        )
        if bad_frames:
            raise ValueError(
                "Montage digitisation is not in the head frame: coordinate frame(s) "
                f"{bad_frames} found (expected FIFFV_COORD_HEAD="
                f"{int(FIFF.FIFFV_COORD_HEAD)}); re-apply the montage so the dig "
                "points are in head coordinates"
            )

        trans_path = SourceReconstructionModule._trans_path(subjects_dir)
        if not trans_path.is_file():
            raise FileNotFoundError(
                f"fsaverage head->MRI transform not found: {trans_path}; required by "
                "the montage pre-flight and the template forward"
            )
        head_path = (
            Path(subjects_dir) / "fsaverage" / "bem" / HEAD_SURFACE_FILENAME
        )
        if not head_path.is_file():
            raise FileNotFoundError(
                f"fsaverage scalp surface not found: {head_path}; required by the "
                "montage pre-flight"
            )

        trans = mne.read_trans(trans_path, verbose=False)
        surfaces = mne.read_bem_surfaces(head_path, verbose=False)
        scalp_rr = np.concatenate(
            [np.asarray(surface["rr"], dtype=float) for surface in surfaces]
        )
        tree = cKDTree(scalp_rr)
        distances_mm = np.asarray(
            tree.query(mne.transforms.apply_trans(trans, positions))[0], dtype=float
        ) * 1000.0
        median_mm = float(np.median(distances_mm))
        max_mm = float(distances_mm.max())

        if max_mm > MONTAGE_MAX_DISTANCE_MM:
            raise ValueError(
                f"Montage alignment failed: the worst EEG electrode is {max_mm:.1f} mm "
                f"from the fsaverage scalp (limit {MONTAGE_MAX_DISTANCE_MM:.0f} mm; "
                f"median {median_mm:.1f} mm); the electrode positions do not match the "
                "fsaverage head surface"
            )
        warnings: List[str] = []
        if median_mm > MONTAGE_MEDIAN_DISTANCE_MM:
            warnings.append(
                f"Montage alignment warning: median EEG-to-scalp distance "
                f"{median_mm:.1f} mm > {MONTAGE_MEDIAN_DISTANCE_MM:.0f} mm "
                f"(max {max_mm:.1f} mm)"
            )
        return {
            "median_mm": median_mm,
            "max_mm": max_mm,
            "n_electrodes": int(len(positions)),
            "status": "ok",
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Provenance + atomic artifact writes (P2-F10, B5 tasks 5 and 8)
    # ------------------------------------------------------------------

    @staticmethod
    def _roi_flip_summary(
        labels: List[Any], src: mne.SourceSpaces, roi_names: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Per-ROI polarity summary of the ``mean_flip`` extraction (Opus B2 cond. 5).

        ``extract_label_time_course(mode='mean_flip')`` multiplies each vertex by
        ``mne.label_sign_flip``. Recording the mean sign (``mean_dot``; plus the
        per-hemisphere means for midline ``BiHemiLabel`` ROIs, whose flip axes can
        cancel) makes a silent polarity flip across mne versions/configurations
        detectable from the provenance alone.
        """
        summary: Dict[str, Dict[str, Any]] = {}
        for name, label in zip(roi_names, labels):
            flips = np.asarray(mne.label_sign_flip(label, src), dtype=float)
            entry: Dict[str, Any] = {
                "n_vertices": int(flips.size),
                "mean_dot": float(np.mean(flips)) if flips.size else None,
            }
            if isinstance(label, mne.BiHemiLabel):
                n_lh = int(len(label.lh.vertices))
                entry["lh_mean_dot"] = (
                    float(np.mean(flips[:n_lh])) if n_lh else None
                )
                entry["rh_mean_dot"] = (
                    float(np.mean(flips[n_lh:])) if flips.size > n_lh else None
                )
            summary[name] = entry
        return summary

    def _build_provenance(
        self,
        *,
        subject_id: str,
        epochs: mne.BaseEpochs,
        inverse_operator: Any,
        labels: List[Any],
        label_info: Dict[str, Any],
        selection: Dict[str, Any],
        lambda2_evoked: float,
        lambda2_trial: float,
        method_params: Optional[Dict[str, Any]],
        data_rank: Dict[str, int],
        cov_rank: Dict[str, int],
        montage_check: Dict[str, Any],
        epochs_path: Optional[Union[str, Path]],
        forward_source: str = "template",
        crosstalk_matrix: Optional[np.ndarray] = None,
        crosstalk_unflipped: Optional[np.ndarray] = None,
        crosstalk_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assemble the ``{subject}_provenance.json`` payload (P2-F10, B5 task 5).

        Covers library versions, the fsaverage assets actually used (subjects dir,
        BEM, trans), source-space size, covariance estimator + data/covariance rank,
        channel/epoch counts, both lambda2 values and the resolved
        ``method_params``, per-condition ``nave``, the full label block including
        ``excluded_reasons``, the per-ROI flip summary, montage distances, the
        condition bookkeeping (B4), the resolved condition -> event-code
        mapping + ``run_identity_hash`` (B5-N3), the epoch file path + streamed
        sha256 and the resolved ``config_hash``.
        """
        epochs_path = Path(epochs_path) if epochs_path is not None else None
        selection_meta = self._selection_metadata(selection)
        return {
            "stage": "source",
            "module": self.name,
            "module_version": self.version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "subject_id": subject_id,
            "eegcpm": package_version("eegcpm"),
            "mne": mne.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "method": self.method,
            "parcellation": self.parcellation,
            "mode": self.mode,
            "forward_source": forward_source,
            "spacing": self.spacing if forward_source == "template" else None,
            "subjects_dir": str(self.subjects_dir),
            "bem": BEM_FILENAME if forward_source == "template" else None,
            "trans": (
                str(self._trans_path(self.subjects_dir))
                if forward_source == "template"
                else None
            ),
            "n_sources": int(sum(len(s["vertno"]) for s in inverse_operator["src"])),
            "n_channels": int(len(epochs.ch_names)),
            "n_epochs": int(len(epochs)),
            "sfreq": float(epochs.info["sfreq"]),
            "cov_method": COV_METHOD,
            "cov_rank": dict(cov_rank),
            "data_rank": dict(data_rank),
            "loose": float(self.loose),
            "depth": float(self.depth),
            "pick_ori": "normal",
            "label_extraction_mode": "mean_flip",
            "lambda2_evoked": float(lambda2_evoked),
            "lambda2_trial": float(lambda2_trial),
            "method_params": method_params,
            "nave": dict(selection_meta["nave"]),
            "nave_confound_warning": selection_meta["nave_confound_warning"],
            "conditions_requested": selection_meta["conditions_requested"],
            "conditions_produced": selection_meta["conditions_produced"],
            "conditions_missing": selection_meta["conditions_missing"],
            "conditions_skipped": selection_meta["conditions_skipped"],
            # B5-N3: the resolved condition -> event-code mapping and the
            # identity hash combining it with the config hash.
            "resolved_event_codes": to_jsonable(
                selection_meta["resolved_event_codes"]
            ),
            "run_identity_hash": self.run_identity_hash(selection),
            "labels": {
                "roi_names": list(label_info["roi_names"]),
                "roi_coords_mni": np.asarray(
                    label_info["roi_coords_mni"], dtype=float
                ).tolist(),
                "roi_radius_mm": np.asarray(
                    label_info["roi_radius_mm"], dtype=float
                ).tolist(),
                "roi_vertex_counts": np.asarray(
                    label_info["roi_vertex_counts"], dtype=int
                ).tolist(),
                "roi_hemi": list(label_info["roi_hemi"]),
                "excluded_rois": list(label_info["excluded_rois"]),
                "excluded_reasons": dict(label_info["excluded_reasons"]),
            },
            "roi_flip_summary": self._roi_flip_summary(
                labels, inverse_operator["src"], list(label_info["roi_names"])
            ),
            # P2-F14: the crosstalk matrices themselves are written to
            # `{subject}_crosstalk.npz`; the provenance records how they were
            # built and the leakage summary of the PRIMARY matrix (the matrices
            # are not duplicated here). B7 rev2 fix 6: the primary matrix is the
            # flip-aligned estimator (`mean_flip`-compatible); the library
            # `get_cross_talk(mode='mean')` matrix is kept as
            # `crosstalk_unflipped` and summarised alongside.
            "crosstalk": (
                {
                    "computed": True,
                    "primary_matrix": "crosstalk_flip_aligned",
                    "definition": (
                        "C[i,j] = mean(|CTF_i in ROI j|) / mean(|CTF_i in ROI i|) "
                        "on a QC-only fixed-orientation operator pair "
                        f"(fixed=True, depth={self.depth}, lambda2=lambda2_evoked); "
                        "documented approximation of the production loose=0.2 "
                        "operator. The resolution matrix's rows are cross-talk "
                        "functions (mne 1.13.2: 'the columns ... are the PSFs and "
                        "the rows are the CTFs'), so C[i, j] is the leakage FROM "
                        "ROI j (source, column) INTO ROI i (target, row); "
                        "C[i, i] == 1. The PRIMARY `crosstalk` matrix is the "
                        "flip-aligned estimator "
                        "(compute_crosstalk_matrix_numpy(align_signs=True), the "
                        "sign alignment matching the pipeline's mean_flip ROI "
                        "extraction); `crosstalk_unflipped` is the mne "
                        "get_cross_talk(mode='mean') library matrix, whose signed "
                        "mean cancels across sulcal walls. C is directional "
                        "(C[i, j] != C[j, i] in general) and "
                        "`n_directed_above_0.3` counts directed entries > 0.3 "
                        "(i != j) of the PRIMARY flip-aligned matrix — a "
                        "reciprocal pair above the threshold counts twice "
                        "(B7-A3/B7 rev2)."
                    ),
                    "n_rois": int(crosstalk_matrix.shape[0]),
                    "off_diagonal_max": float(
                        np.max(
                            crosstalk_matrix[
                                ~np.eye(crosstalk_matrix.shape[0], dtype=bool)
                            ]
                        )
                    ),
                    "off_diagonal_mean": float(
                        np.mean(
                            crosstalk_matrix[
                                ~np.eye(crosstalk_matrix.shape[0], dtype=bool)
                            ]
                        )
                    ),
                    "n_directed_above_0.3": int(
                        np.sum(
                            (crosstalk_matrix > 0.3)
                            & ~np.eye(crosstalk_matrix.shape[0], dtype=bool)
                        )
                    ),
                    "unflipped": (
                        {
                            "off_diagonal_max": float(
                                np.max(
                                    crosstalk_unflipped[
                                        ~np.eye(crosstalk_unflipped.shape[0], dtype=bool)
                                    ]
                                )
                            ),
                            "off_diagonal_mean": float(
                                np.mean(
                                    crosstalk_unflipped[
                                        ~np.eye(crosstalk_unflipped.shape[0], dtype=bool)
                                    ]
                                )
                            ),
                            "n_directed_above_0.3": int(
                                np.sum(
                                    (crosstalk_unflipped > 0.3)
                                    & ~np.eye(crosstalk_unflipped.shape[0], dtype=bool)
                                )
                            ),
                        }
                        if crosstalk_unflipped is not None
                        else None
                    ),
                    "max_abs_delta_primary_vs_unflipped": (
                        float(np.max(np.abs(crosstalk_matrix - crosstalk_unflipped)))
                        if crosstalk_unflipped is not None
                        else None
                    ),
                }
                if crosstalk_matrix is not None
                else {"computed": False, "reason": crosstalk_reason}
            ),
            "montage_check": montage_check,
            "epochs_path": str(epochs_path) if epochs_path is not None else None,
            "epochs_sha256": (
                sha256_file(epochs_path) if epochs_path is not None else None
            ),
            "config_hash": self.config_hash(),
            "config": self.cfg.model_dump(mode="json"),
        }

    def _get_crosstalk_matrices(
        self,
        *,
        forward: mne.Forward,
        info: mne.Info,
        noise_cov: mne.Covariance,
        labels: List[Any],
        lambda2: float,
    ) -> Dict[str, np.ndarray]:
        """ROI x ROI crosstalk matrices, cached per forward+covariance+method+lambda2.

        Returns ``{"crosstalk": <flip-aligned primary>,
        "crosstalk_unflipped": <mne get_cross_talk(mode='mean') library>}``
        (B7 rev2 fix 6). The resolution matrix (``n_dipoles x n_dipoles`` at
        oct6 ≈ 0.5 GB) is never cached — only the resulting ``(n_rois, n_rois)``
        matrices are. The cache key contains a digest of the noise covariance
        because the inverse operator (and therefore the whitened resolution
        matrix) depends on it: caching "per montage hash" alone would hand
        subject 2 a matrix computed from subject 1's covariance (documented
        deviation from the plan wording; the forward cache key already covers
        montage/spacing/BEM/trans/mne version/subjects_dir).
        """
        fwd_key = self._forward_cache_key(info, self.spacing, self.subjects_dir)
        cov_digest = hashlib.sha256(
            np.ascontiguousarray(noise_cov["data"]).tobytes()
        ).hexdigest()
        key = sha256_json({
            "forward": fwd_key,
            "noise_cov": cov_digest,
            "method": self.method,
            "lambda2": float(lambda2),
            "labels": [str(label.name) for label in labels],
        })
        cached = self._CROSSTALK_CACHE.get(key)
        if cached is not None:
            return cached
        # The primary matrix is the flip-aligned numpy estimator: it matches the
        # `mean_flip` sign alignment the pipeline uses to extract ROI time
        # courses (rank agreement 0.76 vs 0.61 for the library's signed mean).
        # The library matrix stays available as `crosstalk_unflipped` and is the
        # exact cross-check target of the numpy mirror (test_source.py).
        matrix = compute_crosstalk_matrix_numpy(
            forward,
            info,
            noise_cov,
            labels,
            method=self.method,
            lambda2=lambda2,
            depth=self.depth,
            align_signs=True,
        )
        matrix_unflipped = compute_crosstalk_matrix(
            forward,
            info,
            noise_cov,
            labels,
            method=self.method,
            lambda2=lambda2,
            depth=self.depth,
        )
        matrices = {
            "crosstalk": matrix,
            "crosstalk_unflipped": matrix_unflipped,
        }
        self._CROSSTALK_CACHE[key] = matrices
        return matrices

    def _remove_stale_outputs(self, subject_id: str) -> List[Path]:
        """Delete this subject's artifacts from a previous run before writing (B5 task 8).

        A re-run that produces fewer conditions must not leave the previous run's
        stc/npz files behind (old and new files must never mix). Only this module's
        own naming patterns for this subject are touched.
        """
        removed: List[Path] = []
        for pattern in stale_output_patterns(subject_id):
            for path in sorted(self.output_dir.glob(pattern)):
                if path.is_file():
                    path.unlink()
                    removed.append(path)
        return removed

    def _save_stc_atomic(
        self, subject_id: str, condition: str, stc: mne.SourceEstimate
    ) -> List[Path]:
        """Save one condition's source estimate atomically (temp stem + rename).

        ``stc.save`` writes one file per hemisphere derived from the stem; every
        file it wrote is renamed onto the final stem, and any leftover temp file is
        removed if the save fails.
        """
        final_stem = self.output_dir / f"{subject_id}_{condition}_stc"
        tmp_stem = self.output_dir / f".{subject_id}_{condition}_stc.tmp"
        pattern = f"{glob.escape(tmp_stem.name)}*"
        try:
            stc.save(tmp_stem, overwrite=True, verbose=False)
            written = sorted(self.output_dir.glob(pattern))
            if not written:
                raise RuntimeError(
                    f"SourceEstimate.save wrote no file for {tmp_stem}"
                )
            finals: List[Path] = []
            for tmp_file in written:
                final = self.output_dir / (
                    final_stem.name + tmp_file.name[len(tmp_stem.name):]
                )
                os.replace(tmp_file, final)
                finals.append(final)
            return finals
        except BaseException:
            for leftover in sorted(self.output_dir.glob(pattern)):
                if leftover.is_file():
                    leftover.unlink()
            raise

    def _write_source_artifacts(
        self,
        *,
        subject_id: str,
        stcs: Dict[str, mne.SourceEstimate],
        evoked_data: Optional[Dict[str, Any]],
        trial_data: Optional[Dict[str, Any]],
        noise_cov: mne.Covariance,
        inverse_operator: Any,
        provenance: Dict[str, Any],
        crosstalk_matrix: Optional[np.ndarray] = None,
        crosstalk_unflipped: Optional[np.ndarray] = None,
    ) -> List[Path]:
        """Write every artifact atomically, stale-safe, provenance last (B5 task 8).

        Order: stale outputs of a previous run are removed first, then the data
        artifacts (stc / ROI npz), then the config/cov/inv files, then the
        provenance JSON that records the sha256 of everything written before it. If
        any write fails, the files this run already wrote are removed and the
        exception propagates, so a failed run never leaves a partial or mixed set of
        artifacts behind a ``success=False`` result.
        """
        written: List[Path] = []
        try:
            self._remove_stale_outputs(subject_id)

            if self.cfg.output.save_stc:
                for condition, stc in stcs.items():
                    written.extend(self._save_stc_atomic(subject_id, condition, stc))

            artifacts: Dict[str, str] = {}
            if self.cfg.output.save_roi_tc:
                for name, data, compressed in (
                    (f"{subject_id}_roi_tc.npz", evoked_data, False),
                    (f"{subject_id}_roi_tc_trials.npz", trial_data, True),
                ):
                    if data is None:
                        continue
                    path = self.output_dir / name
                    atomic_write_npz(path, data, compressed=compressed)
                    written.append(path)
                    artifacts[name] = sha256_file(path)

            if crosstalk_matrix is not None:
                # P2-F14: the ROI x ROI crosstalk matrices travel with the run so
                # downstream connectivity/ERP interpretation can flag leakage
                # pairs without recomputing the (expensive) resolution matrix.
                # B7 rev2 fix 6: `crosstalk` is the primary flip-aligned matrix
                # (matches the mean_flip ROI extraction); `crosstalk_unflipped`
                # is the mne get_cross_talk(mode='mean') library matrix.
                crosstalk_path = self.output_dir / f"{subject_id}_crosstalk.npz"
                payload = {
                    "crosstalk": np.asarray(crosstalk_matrix, dtype=float),
                    "roi_names": np.array(
                        [str(name) for name in provenance["labels"]["roi_names"]],
                        dtype=str,
                    ),
                    "method": np.array([str(self.method)]),
                    "primary": np.array(["crosstalk_flip_aligned"]),
                    "definition": np.array([
                        "crosstalk: primary flip-aligned estimator "
                        "(compute_crosstalk_matrix_numpy(align_signs=True), "
                        "label_sign_flip-aligned CTF rows; matches mean_flip). "
                        "crosstalk_unflipped: mne get_cross_talk(mode='mean') "
                        "library matrix. Both: C[i,j] = mean(|CTF_i in ROI j|) / "
                        "mean(|CTF_i in ROI i|) on a fixed-orientation QC "
                        "operator pair (depth=%.2f, lambda2=lambda2_evoked); "
                        "rows are cross-talk functions, so C[i, j] is the "
                        "leakage FROM ROI j (source, column) INTO ROI i "
                        "(target, row); C[i, i] == 1 (B7 rev2 fixes 5/6)."
                        % self.depth
                    ]),
                }
                if crosstalk_unflipped is not None:
                    payload["crosstalk_unflipped"] = np.asarray(
                        crosstalk_unflipped, dtype=float
                    )
                atomic_write_npz(crosstalk_path, payload)
                written.append(crosstalk_path)
                artifacts[crosstalk_path.name] = sha256_file(crosstalk_path)

            config_path = self.output_dir / f"{subject_id}_source_config.yaml"
            atomic_write_text(
                config_path,
                yaml.safe_dump(self.cfg.model_dump(mode="json"), sort_keys=False),
            )
            written.append(config_path)
            artifacts[config_path.name] = sha256_file(config_path)

            cov_path = self.output_dir / f"{subject_id}-cov.fif"
            atomic_save(
                cov_path,
                lambda tmp: noise_cov.save(tmp, overwrite=True, verbose=False),
                tmp_name=f".{cov_path.stem}.tmp-cov.fif",
            )
            written.append(cov_path)
            artifacts[cov_path.name] = sha256_file(cov_path)

            inv_path = self.output_dir / f"{subject_id}-inv.fif"
            atomic_save(
                inv_path,
                lambda tmp: write_inverse_operator(
                    tmp, inverse_operator, overwrite=True, verbose=False
                ),
                tmp_name=f".{inv_path.stem}.tmp-inv.fif",
            )
            written.append(inv_path)
            artifacts[inv_path.name] = sha256_file(inv_path)

            # Provenance last: it records the hashes of every artifact above, so a
            # consumer that finds it can verify the archive it belongs to.
            payload = dict(provenance)
            payload["artifacts"] = artifacts
            provenance_path = self.output_dir / f"{subject_id}_provenance.json"
            atomic_write_json(provenance_path, payload)
            written.append(provenance_path)
        except BaseException:
            for path in written:
                Path(path).unlink(missing_ok=True)
            raise
        return written

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
            method=COV_METHOD,
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
        method_params: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, mne.SourceEstimate], Dict[str, Any]]:
        """Average per condition and extract signed ROI time courses (P2-F4).

        Returns ``(stcs, roi_data)`` with ``roi_data[cond]`` shaped
        ``(n_rois, n_times)`` plus ``{cond}_times`` / ``{cond}_nave`` / ``sfreq``
        and the label metadata. ``lambda2`` is the evoked lambda2 (``1/snr**2``)
        and ``method_params`` carries the eLORETA ``force_equal`` flag (P2-F12).
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
                method_params=method_params,
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
        method_params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Per-trial inversion via ``apply_inverse_epochs`` (P2-F12).

        Single trials use their own lambda2 (``1/snr_trial**2``, passed
        positionally) — a missing kwarg would silently fall back to the evoked
        lambda2 (B4-N5) — and ``apply_inverse_epochs`` re-prepares the operator once
        instead of per trial (~10x faster). The same ``pick_ori='normal'`` /
        ``mean_flip`` extraction as the evoked path is used so single-trial output is
        signed as well. Each condition array is ``(n_trials, n_rois, n_times)`` with
        at least ``min_trials`` trials.
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
                method_params=method_params,
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

    @staticmethod
    def _template_asset_key(subjects_dir: Union[str, Path]) -> str:
        """``mtime_ns:size`` of the template BEM/trans files (B5-A4).

        The cache key used to cover the BEM/trans *filenames* only: replacing
        the file contents (a different fsaverage BEM built in place, a
        re-downloaded template) kept handing out the stale forward. Including
        ``st_mtime_ns`` and ``st_size`` makes the key content-sensitive
        without hashing multi-hundred-MB files.
        """
        subjects_dir = Path(subjects_dir)
        parts: List[str] = []
        for path in (
            subjects_dir / "fsaverage" / "bem" / BEM_FILENAME,
            SourceReconstructionModule._trans_path(subjects_dir),
        ):
            try:
                stat = path.stat()
                parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
            except OSError:
                # Missing assets are reported by the explicit loaders below
                # (FileNotFoundError with a clear message); the key just
                # records that they were absent.
                parts.append(f"{path.name}:missing")
        return "|".join(parts)

    @staticmethod
    def _forward_cache_key(
        info: mne.Info, spacing: str, subjects_dir: Union[str, Path]
    ) -> str:
        """Cache key for ``(src, forward)`` (P2-F13).

        ``sha256(ch_names + rounded loc[:3] bytes) + spacing + BEM filename + mne
        version`` (plan §B5 task 3) plus the resolved ``subjects_dir`` (two
        different fsaverage copies with the same BEM filename must never share a
        forward) and the BEM/trans ``mtime_ns:size`` (B5-A4: replacing the file
        contents in place must invalidate the cache). Rounding the positions to
        1e-6 m keeps the key stable against float noise in the montage
        round-trip.
        """
        digest = hashlib.sha256()
        digest.update("\x00".join(info["ch_names"]).encode("utf-8"))
        positions = np.array(
            [channel["loc"][:3] for channel in info["chs"]], dtype=float
        )
        digest.update(np.round(positions, 6).tobytes())
        return "|".join(
            (
                digest.hexdigest(),
                spacing,
                BEM_FILENAME,
                mne.__version__,
                str(Path(subjects_dir)),
                SourceReconstructionModule._template_asset_key(subjects_dir),
            )
        )

    def _get_template_forward(
        self,
        info: mne.Info,
    ) -> Tuple[mne.Forward, List[str], Dict[str, Any]]:
        """
        Get (and cache) the fsaverage template forward model (P2-F13).

        Runs the montage pre-flight (:meth:`_preflight_montage`) first, so an
        invalid montage or a misaligned electrode set fails before the ~10 s forward
        build and before a cached forward is handed out. Only the 3-layer BEM
        solution is accepted (:meth:`_load_bem`), the head->MRI transform is read
        from the same ``subjects_dir`` the pre-flight validated (``trans='fsaverage'``
        resolves to mne's *bundled* template transform in mne 1.13.2, which need not
        be the validated file), and the result is cached class-wide per
        montage/spacing/BEM/mne version.

        Returns:
            (forward, warnings, montage_check)
        """
        warnings: List[str] = []
        subjects_dir = self.subjects_dir
        montage_check = self._preflight_montage(info, subjects_dir)

        cache_key = self._forward_cache_key(info, self.spacing, subjects_dir)
        cached = self._FORWARD_CACHE.get(cache_key)
        if cached is not None:
            return cached[1], warnings, montage_check

        src = mne.setup_source_space(
            "fsaverage",
            spacing=self.spacing,
            subjects_dir=subjects_dir,
            add_dist=False,
            verbose=False,
        )
        bem_sol = self._load_bem(subjects_dir)
        trans_path = self._trans_path(subjects_dir)
        if not trans_path.is_file():
            raise FileNotFoundError(
                f"fsaverage head->MRI transform not found: {trans_path}; the template "
                "forward requires the transform validated by the montage pre-flight"
            )
        forward = mne.make_forward_solution(
            info,
            trans=str(trans_path),
            src=src,
            bem=bem_sol,
            eeg=True,
            meg=False,
            mindist=5.0,
            verbose=False,
        )
        self._FORWARD_CACHE[cache_key] = (src, forward)
        return forward, warnings, montage_check

    def get_output_spec(self) -> Dict[str, str]:
        return {
            "stcs": "Dict of condition -> SourceEstimate",
            "roi_data": "Dict with ROI time courses",
            "inverse_operator": "MNE InverseOperator",
        }
