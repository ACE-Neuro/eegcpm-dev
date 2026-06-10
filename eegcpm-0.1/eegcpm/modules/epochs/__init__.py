"""Epoch extraction module."""

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mne
import numpy as np

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
        rejection = config.get("rejection", {})
        if isinstance(rejection, dict):
            self.reject = rejection.get("reject", None)
            self.flat = rejection.get("flat", None)
            self.reject_tmin = rejection.get("reject_tmin", None)
            self.reject_tmax = rejection.get("reject_tmax", None)
            self.reject_by_annotation = rejection.get("reject_by_annotation", True)
            self.use_autoreject = rejection.get("use_autoreject", False)
            self.autoreject_n_interpolate = rejection.get("autoreject_n_interpolate", [1, 4, 8, 16])
            self.rejection_strategy = rejection.get("strategy", "threshold")
        else:
            # Backwards compatibility with flat config
            self.reject = config.get("reject_criteria", config.get("reject", None))
            self.flat = config.get("flat", None)
            self.reject_tmin = None
            self.reject_tmax = None
            self.reject_by_annotation = config.get("reject_by_annotation", True)
            self.use_autoreject = config.get("use_autoreject", False)
            self.autoreject_n_interpolate = [1, 4, 8, 16]
            self.rejection_strategy = "threshold"

    def process(
        self,
        data: mne.io.BaseRaw,
        subject: Optional[Any] = None,
        events: Optional[np.ndarray] = None,
        event_id: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> ModuleResult:
        """
        Extract epochs from raw data.

        Args:
            data: Raw EEG data
            subject: Subject info
            events: Optional events array (will find from raw if not provided)
            event_id: Optional event ID mapping

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

            if event_id is None:
                event_id = self.event_id
                if event_id is None:
                    # Create from unique event values
                    unique_events = np.unique(events[:, 2])
                    event_id = {str(e): e for e in unique_events}

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
                verbose=False,
            )

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
                    "tmin": self.tmin,
                    "tmax": self.tmax,
                    "conditions": list(event_id.keys()),
                    "drop_log_summary": epochs.drop_log_stats(),
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
