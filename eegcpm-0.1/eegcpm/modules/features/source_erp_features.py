"""
Source-Level ERP Feature Extraction Module

Extracts Event-Related Potential features from source-reconstructed ROI time courses.
This allows observing ERP patterns at the source level.

Features include:
- Peak amplitudes and latencies for defined components at each ROI
- Mean amplitudes in time windows per ROI
- Per-condition columns (never averaged across conditions)
- Source-level ERP QC reports

Input contract (P2-F6 / B4): the evoked ROI archive written by the source module
(``{subject}_roi_tc.npz``):

- ``{cond}: (n_rois, n_times)`` arrays, one per condition
- ``{cond}_times``: the condition's time axis (negative tmin preserved)
- ``{cond}_nave``: trials averaged into ``{cond}`` (optional but written by B4)
- ``roi_names``: ROI names in row order

Accepted inputs: a ``Mapping``, an ``np.lib.npyio.NpzFile`` or a path to such an
``.npz``. Conditions are the keys that have a ``{key}_times`` partner — there is no
metadata deny-list, so new condition names are picked up automatically. Single-trial
(3-D) arrays raise: use the evoked ``*_roi_tc.npz`` (the trial archive is
``*_roi_tc_trials.npz``).

Provenance requirement (B5 task 8): an archive read **from disk** must have the
provenance sidecar written by the source stage next to it
(``{subject}_provenance.json``, ``eegcpm.core.provenance``); a missing/invalid
provenance — or a recorded sha256 that does not match the file — is rejected
loudly instead of silently consuming a stale or partially written archive. The
run's ``config_hash`` and ``epochs_sha256`` are reported in the result metadata.
In-memory ``Mapping`` input has no sidecar and is accepted as-is (the caller owns
the freshness of data it holds directly).
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eegcpm.core.provenance import verify_archive_provenance
from eegcpm.pipeline.base import BaseModule, ModuleResult


class SourceERPFeatureModule(BaseModule):
    """Extract ERP features from source-reconstructed ROI time courses."""

    name = "source_erp_features"
    version = "0.2.0"
    description = "ERP feature extraction (source-level)"

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        super().__init__(config, output_dir)

        # Feature extraction parameters
        self.components = config.get("components", {})
        self.mean_amplitude_windows = config.get("mean_amplitude_windows", {})
        self.metrics = config.get("metrics", ["peak_amplitude", "peak_latency"])
        self.rois = config.get("rois", "all")  # Which ROIs to extract

        # Dependencies
        depends_on = config.get("depends_on", {}) or {}
        self.preprocessing = depends_on.get("preprocessing", "unknown")
        self.task = depends_on.get("task", "unknown")
        self.source_variant = depends_on.get("source", "unknown")

        # Variant name
        self.variant = config.get("variant", "default")

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    @staticmethod
    def _as_mapping(data: Any) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Coerce a Mapping / ``NpzFile`` / path into a plain dict (P2-F6).

        Returns ``(mapping, provenance)``. Loading eagerly means callers may close
        the file and every downstream branch sees the same plain-dict contract.

        Archives read from disk must carry the source stage's provenance sidecar
        (B5 task 8): :func:`verify_archive_provenance` raises ``ValueError`` when it
        is missing, unusable, or does not match the archive on disk. In-memory
        mappings cannot have a sidecar and are accepted as-is.
        """
        if isinstance(data, np.lib.npyio.NpzFile):
            mapping = {key: data[key] for key in data.files}
            # numpy >= 2.5 no longer exposes ``NpzFile.filename``; the path lives on
            # the underlying zipfile handle (``fid.name``). A buffer-backed archive
            # has no path and therefore no sidecar (like an in-memory mapping).
            filename = getattr(data, "filename", None)
            if not isinstance(filename, (str, Path)):
                candidate = getattr(getattr(data, "fid", None), "name", None)
                filename = candidate if isinstance(candidate, (str, Path)) else None
            provenance = (
                verify_archive_provenance(Path(filename)) if filename else None
            )
            return mapping, provenance
        if isinstance(data, Mapping):
            return dict(data), None
        if isinstance(data, (str, Path)):
            path = Path(data)
            if not path.is_file():
                raise FileNotFoundError(f"ROI time-course file not found: {path}")
            provenance = verify_archive_provenance(path)
            with np.load(path, allow_pickle=False) as npz:
                return {key: npz[key] for key in npz.files}, provenance
        raise TypeError(
            "input must be a Mapping, an np.lib.npyio.NpzFile or a path to an .npz "
            f"file; got {type(data).__name__}"
        )

    def validate_input(self, data: Any) -> bool:
        """Validate input is a source ROI time-course archive (with provenance)."""
        try:
            mapping, _provenance = self._as_mapping(data)
        except (TypeError, ValueError, OSError):
            return False
        return "roi_names" in mapping

    @staticmethod
    def _condition_names(data: Mapping[str, Any]) -> List[str]:
        """Conditions are the keys with a ``{key}_times`` partner (P2-F6)."""
        return [str(key) for key in data.keys() if f"{key}_times" in data]

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def process(
        self,
        roi_data: Union[Mapping[str, Any], np.lib.npyio.NpzFile, str, Path],
        subject: Optional[Any] = None,
        **kwargs
    ) -> ModuleResult:
        """
        Extract ERP features from source ROI time courses.

        Parameters
        ----------
        roi_data : Mapping | NpzFile | path
            Evoked ROI archive: ``{cond}: (n_rois, n_times)`` + ``{cond}_times`` +
            ``roi_names`` (+ optional ``{cond}_nave``).
        subject : object
            Subject object with id and session attributes
        **kwargs : dict
            Additional arguments (ignored; the time axis comes from ``{cond}_times``)

        Returns
        -------
        ModuleResult
            Results with extracted features DataFrame. Raises ``ValueError`` for
            structural input problems and when nothing matched — before writing
            any file.
        """
        start_time = time.time()

        data, provenance = self._as_mapping(roi_data)
        if "roi_names" not in data:
            raise ValueError(
                "input must contain 'roi_names' (the source ROI time-course archive "
                "written by SourceReconstructionModule)"
            )

        subject_id = subject.id if subject else "unknown"
        session_id = subject.session if hasattr(subject, "session") and subject.session else "01"

        all_roi_names = [str(x) for x in np.atleast_1d(data["roi_names"])]
        if self.rois != "all":
            roi_names = [r for r in all_roi_names if r in self.rois]
        else:
            roi_names = list(all_roi_names)
        if not roi_names:
            raise ValueError(
                f"no ROIs matched rois={self.rois!r}; available={all_roi_names}"
            )

        conditions = self._condition_names(data)
        if not conditions:
            raise ValueError(
                "no conditions found: expected '<cond>' arrays with a '<cond>_times' "
                f"partner; keys={sorted(str(k) for k in data.keys())}"
            )

        features: Dict[str, Any] = {
            "subject_id": subject_id,
            "session_id": session_id,
        }

        sfreq: Optional[float] = None
        nave_keys = set()
        for condition in conditions:
            tc = np.asarray(data[condition])
            times = np.asarray(data[f"{condition}_times"], dtype=float)
            if tc.ndim != 2:
                raise ValueError(
                    f"condition '{condition}': trial-level array with shape "
                    f"{tc.shape}; use the evoked *_roi_tc.npz"
                )
            if tc.shape[0] != len(all_roi_names):
                raise ValueError(
                    f"condition '{condition}': {tc.shape[0]} ROI rows do not match "
                    f"{len(all_roi_names)} roi_names"
                )
            if tc.shape[1] != len(times):
                raise ValueError(
                    f"condition '{condition}': {tc.shape[1]} samples do not match "
                    f"the {len(times)} entries of '{condition}_times'"
                )
            if len(times) < 2 or not np.all(np.diff(times) > 0):
                raise ValueError(
                    f"condition '{condition}': time axis must be strictly increasing "
                    "with at least two samples"
                )
            if sfreq is None:
                # sfreq is derived from the stored time axis (no 500 Hz default).
                sfreq = float(1.0 / np.mean(np.diff(times)))

            for roi_name in roi_names:
                roi_erp = tc[all_roi_names.index(roi_name)]
                for comp_name, comp_config in self.components.items():
                    features.update(
                        self._extract_component_features(
                            roi_erp, times, condition, roi_name, comp_name, comp_config
                        )
                    )
                if self.mean_amplitude_windows:
                    features.update(
                        self._extract_mean_amplitudes(
                            roi_erp, times, condition, roi_name,
                            self.mean_amplitude_windows,
                        )
                    )

            nave_key = f"{condition}_nave"
            if nave_key in data:
                features[nave_key] = int(np.asarray(data[nave_key]).reshape(-1)[0])
                nave_keys.add(nave_key)

        feature_columns = [
            k
            for k in features
            if k not in ("subject_id", "session_id") and k not in nave_keys
        ]
        if not feature_columns:
            raise ValueError(
                "no features produced (zero matched conditions/ROIs/metrics); "
                f"conditions={conditions}, rois={roi_names}, "
                f"components={list(self.components)}, metrics={self.metrics}; "
                "nothing was written"
            )

        # Convert to DataFrame (single row per subject/session; all columns are
        # per condition x ROI x component x metric).
        features_df = pd.DataFrame([features])

        # Save features. The CSV is written once, after all validation passed.
        output_files = []
        features_csv = self.output_dir / "features.csv"

        if features_csv.exists():
            existing = pd.read_csv(features_csv)
            combined = pd.concat([existing, features_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["subject_id", "session_id"], keep="last"
            )
            combined.to_csv(features_csv, index=False)
        else:
            features_df.to_csv(features_csv, index=False)

        output_files.append(features_csv)

        # Save config
        import yaml
        config_file = self.output_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        output_files.append(config_file)

        # Generate QC report if enabled
        if self.config.get("generate_qc", True):
            qc_html = self._generate_qc_report(
                data, features_df, subject_id, session_id, conditions, roi_names
            )
            if qc_html:
                output_files.append(qc_html)

        return ModuleResult(
            success=True,
            module_name=self.name,
            execution_time_seconds=time.time() - start_time,
            outputs={
                "features": features_df,
                "n_features": len(feature_columns),
                "n_rois": len(roi_names),
                "conditions": conditions,
                "components": list(self.components.keys()),
            },
            output_files=output_files,
            metadata={
                "variant": self.variant,
                "source_variant": self.source_variant,
                "n_components": len(self.components),
                "conditions": conditions,
                "n_rois": len(roi_names),
                "n_features": len(feature_columns),
                "sfreq": sfreq,
                # Provenance of the consumed archive (B5 task 8): the run identity
                # is carried into the feature metadata instead of being discarded.
                "source_config_hash": (
                    provenance.get("config_hash") if provenance else None
                ),
                # B5-N3: config hash + resolved condition -> event-code mapping.
                "source_run_identity_hash": (
                    provenance.get("run_identity_hash") if provenance else None
                ),
                "source_epochs_sha256": (
                    provenance.get("epochs_sha256") if provenance else None
                ),
                "source_epochs_path": (
                    provenance.get("epochs_path") if provenance else None
                ),
            },
        )

    def _extract_component_features(
        self,
        roi_erp: np.ndarray,
        times: np.ndarray,
        condition: str,
        roi_name: str,
        comp_name: str,
        comp_config: Dict[str, Any],
    ) -> Dict[str, float]:
        """Extract ERP component features for one condition x ROI."""

        features: Dict[str, float] = {}

        # Get component parameters
        search_window = comp_config.get("search_window", [0.0, 0.5])
        polarity = comp_config.get("polarity", "positive")

        # Find time indices in search window
        time_mask = (times >= search_window[0]) & (times <= search_window[1])
        win_times = times[time_mask]
        win_data = roi_erp[time_mask]

        if len(win_data) == 0:
            return {}

        # Find peak
        if polarity == "positive":
            peak_idx = np.argmax(win_data)
        else:
            peak_idx = np.argmin(win_data)

        peak_amplitude = win_data[peak_idx]
        peak_latency = win_times[peak_idx]

        # Per condition x ROI x component columns (never averaged across conditions)
        feature_prefix = f"{condition}_{roi_name}_{comp_name}"

        if "peak_amplitude" in self.metrics:
            features[f"{feature_prefix}_peak_amplitude"] = float(peak_amplitude)

        if "peak_latency" in self.metrics:
            features[f"{feature_prefix}_peak_latency"] = float(peak_latency)

        if "mean_amplitude" in self.metrics:
            features[f"{feature_prefix}_mean_amplitude"] = float(np.mean(win_data))

        if "area_under_curve" in self.metrics:
            dt = float(np.mean(np.diff(win_times))) if len(win_times) > 1 else 1.0
            auc = np.trapezoid(win_data, dx=dt)
            features[f"{feature_prefix}_area_under_curve"] = float(auc)

        return features

    def _extract_mean_amplitudes(
        self,
        roi_erp: np.ndarray,
        times: np.ndarray,
        condition: str,
        roi_name: str,
        windows: Dict[str, List[float]],
    ) -> Dict[str, float]:
        """Extract mean amplitudes in time windows for one condition x ROI."""

        features: Dict[str, float] = {}

        for window_name, window in windows.items():
            tmin, tmax = window

            # Find time indices
            time_mask = (times >= tmin) & (times <= tmax)
            win_data = roi_erp[time_mask]

            if len(win_data) > 0:
                features[f"{condition}_{roi_name}_mean_amp_{window_name}"] = float(
                    np.mean(win_data)
                )

        return features

    # ------------------------------------------------------------------
    # QC report
    # ------------------------------------------------------------------

    def _generate_qc_report(
        self,
        roi_data: Dict[str, Any],
        features_df: pd.DataFrame,
        subject_id: str,
        session_id: str,
        conditions: List[str],
        roi_names: List[str],
    ) -> Optional[Path]:
        """Generate source-level ERP QC report."""

        try:
            from eegcpm.modules.qc.html_report import HTMLReportBuilder

            # Create HTML report
            builder = HTMLReportBuilder(title=f"Source ERP Features: {subject_id}")

            builder.add_header("Overview", level=2)
            builder.add_text(f"Subject: {subject_id}")
            builder.add_text(f"Session: {session_id}")
            builder.add_text(f"Source Variant: {self.source_variant}")
            builder.add_text(f"Conditions: {', '.join(conditions)}")
            builder.add_text(f"Components: {', '.join(self.components.keys())}")

            # Plot ROI ERPs (one line per condition) with component markers
            plot_rois = roi_names[:12]  # Top 12 ROIs

            if plot_rois:
                fig = self._plot_roi_erps(roi_data, plot_rois, conditions)
                if fig:
                    from eegcpm.modules.qc.base import BaseQC
                    fig_bytes = BaseQC.fig_to_base64(fig)
                    builder.add_header("ROI Event-Related Potentials", level=2)
                    builder.add_figure(
                        "roi_erps", fig_bytes,
                        "ERP waveforms for the first 12 ROIs (one line per condition)",
                    )
                    plt.close(fig)

            # Save HTML
            qc_html = self.output_dir / f"{subject_id}_source_erp_qc.html"
            html = builder.build()
            with open(qc_html, 'w') as f:
                f.write(html)

            return qc_html

        except Exception as e:
            print(f"Warning: QC report generation failed: {e}")
            return None

    def _plot_roi_erps(
        self,
        roi_data: Dict[str, Any],
        plot_rois: List[str],
        conditions: List[str],
    ) -> Optional[plt.Figure]:
        """Plot ERP waveforms for selected ROIs, one line per condition."""

        all_roi_names = [str(x) for x in np.atleast_1d(roi_data["roi_names"])]

        n_rois = len(plot_rois)
        n_cols = 3
        n_rows = (n_rois + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        axes = axes.flatten() if n_rois > 1 else [axes]

        for idx, roi_name in enumerate(plot_rois):
            ax = axes[idx]
            roi_idx = all_roi_names.index(roi_name)

            for condition in conditions:
                times = np.asarray(roi_data[f"{condition}_times"], dtype=float)
                condition_data = np.asarray(roi_data[condition])
                ax.plot(
                    times, condition_data[roi_idx], linewidth=1.2, label=condition
                )

            ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
            ax.axvline(0, color='k', linestyle='--', linewidth=0.5)

            # Mark component windows
            for comp_name, comp_config in self.components.items():
                window = comp_config.get("search_window", [])
                if window:
                    ax.axvspan(window[0], window[1], alpha=0.2, label=comp_name)

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (a.u.)')
            ax.set_title(f'{roi_name}')
            ax.grid(True, alpha=0.3)

            if idx == 0:
                ax.legend(fontsize=8)

        # Hide unused subplots
        for idx in range(n_rois, len(axes)):
            axes[idx].axis('off')

        fig.suptitle('Source-Level Event-Related Potentials', fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig
