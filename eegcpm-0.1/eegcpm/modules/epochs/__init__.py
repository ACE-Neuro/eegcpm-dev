"""Epoch extraction module."""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import pandas as pd

from eegcpm.pipeline.base import BaseModule, ModuleResult, RawDataModule


def sanitize_name(name: str) -> str:
    """Replace invalid filename characters with underscores."""
    return re.sub(r'[/\\:*?"<>|\s]+', '_', name).strip('_')


class EpochExtractionModule(RawDataModule):
    """
    Extract epochs from continuous EEG data.

    Supports:
    - Event-based epoching
    - Baseline correction
    - Configurable artifact rejection (threshold, flat, autoreject)
    - ERP computation
    """

    name = "epoch_extraction"
    version = "0.1.0"
    description = "Extract epochs and compute ERPs"

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        super().__init__(config, output_dir)
        self.tmin = config.get("tmin", -0.5)
        self.tmax = config.get("tmax", 1.0)
        self.baseline = config.get("baseline", (-0.2, 0.0))
        self.event_id = config.get("event_id", None)
        self.detrend = config.get("detrend", None)
        self.decim = config.get("decim", 1)

        # Parse rejection config (supports nested dict or flat)
        # Nested: config has "rejection": {"strategy": ..., "reject": ..., "flat": ...}
        # Flat: config has "reject": {...}, "flat": {...} at top level (e.g. task config)
        rejection = config.get("rejection", None)
        if isinstance(rejection, dict) and ("reject" in rejection or "strategy" in rejection):
            self.reject = rejection.get("reject", None)
            self.flat = rejection.get("flat", None)
            self.reject_tmin = rejection.get("reject_tmin", None)
            self.reject_tmax = rejection.get("reject_tmax", None)
            self.reject_by_annotation = rejection.get("reject_by_annotation", True)
            self.use_autoreject = rejection.get("use_autoreject", False)
            self.autoreject_n_interpolate = rejection.get("autoreject_n_interpolate", [1, 4, 8, 16])
            self.rejection_strategy = rejection.get("strategy", "threshold")
        else:
            # Flat config or no rejection specified: read from top-level keys
            self.reject = config.get("reject_criteria", config.get("reject", None))
            self.flat = config.get("flat", None)
            self.reject_tmin = config.get("reject_tmin", None)
            self.reject_tmax = config.get("reject_tmax", None)
            self.reject_by_annotation = config.get("reject_by_annotation", True)
            self.use_autoreject = config.get("use_autoreject", False)
            self.autoreject_n_interpolate = config.get("autoreject_n_interpolate", [1, 4, 8, 16])
            self.rejection_strategy = config.get("rejection_strategy", "threshold")

    def process(
        self,
        data: mne.io.BaseRaw,
        subject: Optional[Any] = None,
        events: Optional[np.ndarray] = None,
        event_id: Optional[Dict[str, int]] = None,
        metadata: Optional[pd.DataFrame] = None,
        metadata_source: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> ModuleResult:
        """
        Extract epochs from raw data.

        Args:
            data: Raw EEG data
            subject: Subject info
            events: Optional events array (will find from raw if not provided)
            event_id: Optional event ID mapping
            metadata: Optional per-trial metadata with exactly one row per
                event, aligned to the ``events`` array (e.g. the DataFrame
                returned by ``load_events_from_bids_tsv(..., return_metadata=True)``).
                Attached to the epochs and saved as
                ``<subject>_epochs_metadata.csv`` next to the epochs FIF.
                Metadata is kept in lock-step with the saved epochs: events
                excluded by ``event_id`` and epochs dropped by rejection are
                removed together, so CSV row count == saved epoch count.
                ``ModuleResult.metadata`` records ``metadata_rows`` (rows in
                the DataFrame passed in, 0 when none), ``metadata_rows_saved``
                (rows written to the CSV / surviving epochs) and
                ``metadata_columns``.
            metadata_source: Optional path of the source events.tsv, recorded
                in the ``<subject>_epochs_metadata.json`` provenance sidecar.

        Returns:
            ModuleResult with Epochs
        """
        start_time = time.time()
        raw = data
        output_files = []
        warnings = []

        try:
            # Get events
            if events is None:
                events = mne.find_events(raw, verbose=False)

            n_events_input = int(len(events))
            # Rows of the metadata DataFrame the caller passed in (its length
            # before any MNE subsetting). The number of rows actually saved is
            # tracked separately as ``metadata_rows_saved``.
            metadata_rows_input = 0

            # Validate per-trial metadata before handing it to MNE: a length
            # mismatch would raise deep inside mne.Epochs with a confusing
            # message, and a silent misalignment is worse (no silent failures).
            if metadata is not None:
                if not isinstance(metadata, pd.DataFrame):
                    raise ValueError(
                        f"metadata must be a pandas.DataFrame, got {type(metadata).__name__}"
                    )
                if len(metadata) != len(events):
                    raise ValueError(
                        f"metadata has {len(metadata)} rows but events has "
                        f"{len(events)}; metadata must have exactly one row per "
                        "event, aligned to the events array"
                    )
                metadata_rows_input = int(len(metadata))

            if event_id is None:
                event_id = self.event_id
                if event_id is None:
                    # Create from unique event values
                    unique_events = np.unique(events[:, 2])
                    event_id = {str(e): e for e in unique_events}

            # MNE 1.13.2 alignment fix (verified by probe + source read): when
            # `event_id` filters events, mne.Epochs takes metadata rows
            # positionally from the start of the frame and then relabels them
            # with the surviving event indices — a silent misalignment
            # (epochs.py: `sub = np.where(np.isin(selection, self.selection))[0]`
            # then `metadata.iloc[sub]`). Pre-filter events and metadata
            # together so MNE's own event_id mask is a no-op and rows stay
            # aligned. Only applied when metadata is passed, so the existing
            # events-only path is untouched.
            if metadata is not None:
                wanted_codes = list(event_id.values())
                keep = np.isin(events[:, 2], wanted_codes)
                if not bool(keep.all()):
                    events = events[keep]
                    metadata = metadata.iloc[np.flatnonzero(keep)].reset_index(drop=True)

            # Determine rejection parameters based on strategy
            reject_param = None
            flat_param = None

            if self.rejection_strategy in ["threshold", "both"]:
                reject_param = self.reject
                flat_param = self.flat

            # Create epochs with rejection
            # Note: proj=False keeps projectors but doesn't apply them yet
            # This preserves average reference projector for source reconstruction
            epochs = mne.Epochs(
                raw,
                events=events,
                event_id=event_id,
                tmin=self.tmin,
                tmax=self.tmax,
                baseline=self.baseline,
                preload=True,
                proj=False,  # Keep projectors but don't apply yet
                reject=reject_param,
                flat=flat_param,
                reject_tmin=self.reject_tmin,
                reject_tmax=self.reject_tmax,
                reject_by_annotation=self.reject_by_annotation,
                detrend=self.detrend,
                decim=self.decim,
                metadata=metadata,
                verbose=False,
            )

            # Guard against metadata/epochs desynchronisation (covers MNE
            # regressions in the rejection path): lengths must always match.
            self._check_metadata_alignment(epochs, metadata)

            # Track rejection statistics
            n_before_autoreject = len(epochs)
            rejection_stats = {
                "n_original": len(events),
                "n_after_threshold": len(epochs),
                "n_rejected_threshold": len(events) - len(epochs),
            }

            # AutoReject if enabled (additional pass)
            if self.rejection_strategy in ["autoreject", "both"] and self.use_autoreject:
                epochs, ar_log = self._apply_autoreject(epochs)
                self._check_metadata_alignment(epochs, metadata)
                rejection_stats["n_after_autoreject"] = len(epochs)
                rejection_stats["n_rejected_autoreject"] = n_before_autoreject - len(epochs)
                rejection_stats["autoreject_log"] = ar_log

            # Compute ERPs
            erps = {}
            for condition in event_id:
                if condition in epochs.event_id:
                    erps[condition] = epochs[condition].average()

            # Save epochs
            subject_id = subject.id if subject else "unknown"
            epochs_path = self.output_dir / f"{subject_id}_epo.fif"
            epochs.save(epochs_path, overwrite=True, verbose=False)
            output_files.append(epochs_path)

            # Downstream consumers (page 9, trial sorting) read the metadata and
            # its provenance from the epochs object / this result dict.
            metadata_columns: List[str] = []
            metadata_rows_saved = 0
            if epochs.metadata is not None:
                metadata_columns = [str(col) for col in epochs.metadata.columns]
                metadata_rows_saved = len(epochs.metadata)
                metadata_csv_path = (
                    self.output_dir / f"{subject_id}_epochs_metadata.csv"
                )
                epochs.metadata.to_csv(metadata_csv_path, index=False)
                output_files.append(metadata_csv_path)

                # Provenance sidecar. A CSV cannot carry a header comment and
                # info['temp'] does not survive FIF save, so the source path
                # and row counts live here (AGENTS invariant 4).
                sidecar_path = (
                    self.output_dir / f"{subject_id}_epochs_metadata.json"
                )
                sidecar = {
                    "source_events_tsv": (
                        str(metadata_source) if metadata_source else None
                    ),
                    "n_rows": metadata_rows_saved,
                    "n_rows_input": metadata_rows_input,
                    "n_events_input": n_events_input,
                    "n_epochs": int(len(epochs)),
                    "columns": metadata_columns,
                    "epochs_file": str(epochs_path),
                    "written_at": datetime.now(timezone.utc).isoformat(),
                }
                with open(sidecar_path, "w") as fh:
                    json.dump(sidecar, fh, indent=2, default=str)
                output_files.append(sidecar_path)

            # Save ERPs
            for condition, erp in erps.items():
                # erp_path = self.output_dir / f"{subject_id}_{condition}_ave.fif"
                safe_condition = sanitize_name(condition)
                erp_path = self.output_dir / f"{subject_id}_{safe_condition}_ave.fif"
                erp.save(erp_path, overwrite=True, verbose=False)
                output_files.append(erp_path)

            return ModuleResult(
                success=True,
                module_name=self.name,
                execution_time_seconds=time.time() - start_time,
                outputs={
                    "data": epochs,
                    "epochs": epochs,
                    "erps": erps,
                },
                output_files=output_files,
                warnings=warnings,
                metadata={
                    "n_epochs": len(epochs),
                    "n_conditions": len(event_id),
                    "n_events_input": n_events_input,
                    "tmin": self.tmin,
                    "tmax": self.tmax,
                    "conditions": list(event_id.keys()),
                    "drop_log_summary": epochs.drop_log_stats(),
                    "metadata_columns": metadata_columns,
                    "metadata_rows": metadata_rows_input,
                    "metadata_rows_saved": metadata_rows_saved,
                    "metadata_source": (
                        str(metadata_source) if metadata_source else None
                    ),
                    "rejection": {
                        "strategy": self.rejection_strategy,
                        "reject_thresholds": self.reject,
                        "flat_thresholds": self.flat,
                        "stats": rejection_stats,
                    },
                },
            )

        except Exception as e:
            return ModuleResult(
                success=False,
                module_name=self.name,
                execution_time_seconds=time.time() - start_time,
                errors=[str(e)],
            )

    @staticmethod
    def _check_metadata_alignment(epochs: mne.Epochs, metadata) -> None:
        """Raise if ``epochs.metadata`` is not aligned 1:1 with the epochs."""
        if metadata is not None and (
            epochs.metadata is None or len(epochs.metadata) != len(epochs)
        ):
            n_meta = None if epochs.metadata is None else len(epochs.metadata)
            raise RuntimeError(
                "metadata/epochs alignment lost during epoching: "
                f"{n_meta} metadata rows for {len(epochs)} epochs"
            )

    def _apply_autoreject(self, epochs: mne.Epochs) -> Tuple[mne.Epochs, Optional[Dict]]:
        """
        Apply autoreject for automatic epoch rejection.

        Returns:
            Tuple of (cleaned epochs, rejection log dict)
        """
        try:
            from autoreject import AutoReject, get_rejection_threshold

            ar = AutoReject(
                n_interpolate=self.autoreject_n_interpolate,
                random_state=42,
                verbose=False,
            )
            epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

            # Build rejection log summary
            log_summary = {
                "n_bad_epochs": int(np.sum(reject_log.bad_epochs)),
                "labels": reject_log.labels.tolist() if hasattr(reject_log, 'labels') else None,
            }

            return epochs_clean, log_summary

        except ImportError:
            return epochs, {"error": "autoreject not installed"}

    def get_output_spec(self) -> Dict[str, str]:
        return {
            "epochs": "mne.Epochs object",
            "erps": "Dict of condition -> Evoked objects",
        }
