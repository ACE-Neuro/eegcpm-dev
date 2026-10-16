"""
ERP Feature Extraction Module

Extracts Event-Related Potential features from epoched sensor-level data.
Features include:
- Peak amplitudes and latencies for defined components (P1, N1, P3, etc.)
- Mean amplitudes in time windows
- Area under curve
- Component-based QC reports

This is a FEATURE EXTRACTION stage, not just QC.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import time

import mne
import numpy as np
import pandas as pd

from eegcpm.pipeline.base import BaseModule, ModuleResult
from eegcpm.core.task_config import ERPComponentSpec


class ERPFeatureModule(BaseModule):
    """Extract ERP features from epoched sensor-level data."""

    name = "erp_features"
    version = "0.1.0"
    description = "ERP feature extraction (sensor-level)"

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        super().__init__(config, output_dir)

        # Feature extraction parameters
        self.components = config.get("components", {})
        self.mean_amplitude_windows = config.get("mean_amplitude_windows", {})
        self.metrics = config.get("metrics", ["peak_amplitude", "peak_latency"])

        # Dependencies
        depends_on = config.get("depends_on", {})
        self.preprocessing = depends_on.get("preprocessing", "unknown")
        self.task = depends_on.get("task", "unknown")
        self.epochs_variant = depends_on.get("epochs", "standard")

        # Variant name
        self.variant = config.get("variant", "default")

    def validate_input(self, data: mne.BaseEpochs) -> bool:
        """Validate input is epochs."""
        return isinstance(data, mne.BaseEpochs)

    def process(
        self,
        epochs: mne.BaseEpochs,
        subject: Optional[Any] = None,
        **kwargs
    ) -> ModuleResult:
        """
        Extract ERP features from epochs.

        Parameters
        ----------
        epochs : mne.BaseEpochs
            Epoched data
        subject : object
            Subject object with id and session attributes
        **kwargs : dict
            Additional arguments

        Returns
        -------
        ModuleResult
            Results with extracted features DataFrame
        """
        start_time = time.time()

        try:
            if not self.validate_input(epochs):
                return ModuleResult(
                    success=False,
                    module_name=self.name,
                    execution_time_seconds=time.time() - start_time,
                    errors=["Input must be MNE Epochs object"]
                )

            subject_id = subject.id if subject else "unknown"
            session_id = subject.session if hasattr(subject, "session") and subject.session else "01"

            # Initialize feature dictionary
            features = {
                "subject_id": subject_id,
                "session_id": session_id
            }

            # Extract features for each component
            for comp_name, comp_config in self.components.items():
                comp_features = self._extract_component_features(
                    epochs,
                    comp_name,
                    comp_config
                )
                features.update(comp_features)

            # Extract mean amplitude windows if specified
            if self.mean_amplitude_windows:
                mean_amp_features = self._extract_mean_amplitudes(
                    epochs,
                    self.mean_amplitude_windows
                )
                features.update(mean_amp_features)

            # Convert to DataFrame
            features_df = pd.DataFrame([features])

            # Save features
            output_files = []
            features_csv = self.output_dir / "features.csv"

            # Append or create
            if features_csv.exists():
                existing = pd.read_csv(features_csv)
                combined = pd.concat([existing, features_df], ignore_index=True)
                # Remove duplicates (same subject/session)
                combined = combined.drop_duplicates(subset=["subject_id", "session_id"], keep="last")
                combined.to_csv(features_csv, index=False)
            else:
                features_df.to_csv(features_csv, index=False)

            output_files.append(features_csv)

            # Save config for reproducibility
            import yaml
            config_file = self.output_dir / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            output_files.append(config_file)

            # Generate QC report if enabled
            if self.config.get("generate_qc", True):
                qc_html = self._generate_qc_report(
                    epochs,
                    features_df,
                    subject_id,
                    session_id
                )
                if qc_html:
                    output_files.append(qc_html)

            return ModuleResult(
                success=True,
                module_name=self.name,
                execution_time_seconds=time.time() - start_time,
                outputs={
                    "features": features_df,
                    "n_features": len(features) - 2,  # Exclude subject_id, session_id
                    "components": list(self.components.keys())
                },
                output_files=output_files,
                metadata={
                    "variant": self.variant,
                    "n_epochs": len(epochs),
                    "n_components": len(self.components)
                }
            )

        except Exception as e:
            return ModuleResult(
                success=False,
                module_name=self.name,
                execution_time_seconds=time.time() - start_time,
                errors=[str(e)]
            )

    def _extract_component_features(
        self,
        epochs: mne.BaseEpochs,
        comp_name: str,
        comp_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract features for one ERP component."""

        features = {}

        # Get component parameters
        search_window = comp_config.get("search_window", [0.0, 0.5])
        channels = comp_config.get("channels", "all")
        polarity = comp_config.get("polarity", "positive")

        # Average across epochs to get ERP
        evoked = epochs.average()

        # Pick channels
        if channels == "all":
            evoked_pick = evoked.copy()
        else:
            evoked_pick = evoked.copy().pick_channels(channels)

        # Get data and times
        data = evoked_pick.data  # Shape: (n_channels, n_times)
        times = evoked_pick.times

        # Find time indices in search window
        time_mask = (times >= search_window[0]) & (times <= search_window[1])
        win_times = times[time_mask]
        win_data = data[:, time_mask]

        # Average across channels
        avg_data = np.mean(win_data, axis=0)

        # Find peak
        if polarity == "positive":
            peak_idx = np.argmax(avg_data)
        else:
            peak_idx = np.argmin(avg_data)

        peak_amplitude = avg_data[peak_idx]
        peak_latency = win_times[peak_idx]

        # Store features
        if "peak_amplitude" in self.metrics:
            features[f"{comp_name}_amplitude"] = peak_amplitude

        if "peak_latency" in self.metrics:
            features[f"{comp_name}_latency"] = peak_latency

        if "mean_amplitude" in self.metrics:
            features[f"{comp_name}_mean"] = np.mean(avg_data)

        if "area_under_curve" in self.metrics:
            # Integrate over time window (np.trapezoid: `np.trapz` is deprecated
            # in numpy 2.x and removed in numpy 2.5)
            dt = np.mean(np.diff(win_times))
            auc = np.trapezoid(avg_data, dx=dt)
            features[f"{comp_name}_auc"] = auc

        return features

    def _extract_mean_amplitudes(
        self,
        epochs: mne.BaseEpochs,
        windows: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Extract mean amplitudes in specified time windows."""

        features = {}
        evoked = epochs.average()

        for window_name, window in windows.items():
            tmin, tmax = window

            # Crop to window
            evoked_crop = evoked.copy().crop(tmin=tmin, tmax=tmax)

            # Average across time and channels
            mean_amp = np.mean(evoked_crop.data)

            features[f"mean_amp_{window_name}"] = mean_amp

        return features

    def _generate_qc_report(
        self,
        epochs: mne.BaseEpochs,
        features_df: pd.DataFrame,
        subject_id: str,
        session_id: str
    ) -> Optional[Path]:
        """Generate ERP feature QC report using existing ERP QC module."""

        try:
            from eegcpm.modules.qc.erp_qc import ERPQC

            # Create evoked dict for each condition
            evokeds = {}
            for condition in epochs.event_id.keys():
                evokeds[condition] = epochs[condition].average()

            # Get channels for each component
            roi_channels = {}
            for comp_name, comp_config in self.components.items():
                channels = comp_config.get("channels", [])
                if channels and channels != "all":
                    roi_channels[comp_name] = channels

            # Define peak windows
            peak_windows = {}
            for comp_name, comp_config in self.components.items():
                window = comp_config.get("search_window")
                if window:
                    peak_windows[comp_name] = tuple(window)

            # Generate report
            qc = ERPQC(self.output_dir)
            qc_html = qc.generate_report(
                evokeds=evokeds,
                subject_id=subject_id,
                session_id=session_id,
                roi_channels=roi_channels if roi_channels else None,
                peak_windows=peak_windows if peak_windows else None
            )

            return qc_html

        except Exception as e:
            print(f"Warning: QC report generation failed: {e}")
            return None
