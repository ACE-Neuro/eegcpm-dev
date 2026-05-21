"""Task configuration for epoch extraction and trial sorting.

Defines experimental conditions, behavioral response mapping, and trial binning.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class ConditionSpec(BaseModel):
    """Specification for an experimental condition (independent variable)."""

    name: str = Field(..., description="Condition name (e.g., 'target', 'distractor')")
    event_codes: List[Union[int, str]] = Field(
        ...,
        description="Event codes for this condition (can be integers or string event names from events.tsv)"
    )
    description: Optional[str] = Field(None, description="Human-readable description")


class ResponseSpec(BaseModel):
    """Specification for behavioral response classification."""

    response_column: str = Field(..., description="Column name in events file with response data")
    categories: Dict[str, Any] = Field(
        ...,
        description="Response categories, e.g., {'correct': 1, 'incorrect': 0, 'too_fast': -1}"
    )
    rt_column: Optional[str] = Field(None, description="Column name for reaction time (if available)")
    rt_min: Optional[float] = Field(None, description="Minimum valid RT (ms)")
    rt_max: Optional[float] = Field(None, description="Maximum valid RT (ms)")


class BinSpec(BaseModel):
    """Specification for binning trials."""

    name: str = Field(..., description="Binning variable name (e.g., 'rt', 'difficulty')")
    method: str = Field(..., description="Binning method: 'quantile', 'fixed', 'custom'")
    n_bins: Optional[int] = Field(None, description="Number of bins (for quantile/fixed)")
    bin_edges: Optional[List[float]] = Field(None, description="Custom bin edges")
    labels: Optional[List[str]] = Field(None, description="Bin labels")
    column: Optional[str] = Field(None, description="Column to bin (for behavioral data)")


class ERPComponentSpec(BaseModel):
    """Specification for an ERP component to extract."""

    name: str = Field(..., description="Component name (e.g., 'P1', 'N1', 'P3')")
    search_window: tuple = Field(..., description="Time window (tmin, tmax) in seconds to search for peak")
    channels: List[str] = Field(..., description="Channels to average (e.g., ['Oz', 'O1', 'O2'])")
    polarity: str = Field(..., description="'positive' or 'negative'")
    description: Optional[str] = Field(None, description="Description of this component")


class TaskConfig(BaseModel):
    """Complete task configuration for epoch extraction and trial sorting."""

    task_name: str = Field(..., description="Task identifier (e.g., 'contdet', 'nback')")
    description: str = Field(..., description="Task description")

    # Task type - CRITICAL for QC and feature extraction
    task_type: str = Field(
        ...,
        description="Task type: 'event-related' or 'continuous' (resting/movie)"
    )

    # Primary event for basic QC/ERP
    primary_event: Optional[Dict[str, Any]] = Field(
        None,
        description="Primary event for basic ERP QC: {'name': 'target', 'codes': [1, 2]}"
    )

    # Event timing
    tmin: float = Field(-0.3, description="Epoch start time (s) before event")
    tmax: float = Field(0.8, description="Epoch end time (s) after event")
    baseline: tuple = Field((-0.2, 0.0), description="Baseline correction window (s)")

    # Experimental design
    conditions: List[ConditionSpec] = Field(..., description="Experimental conditions")
    response_mapping: Optional[ResponseSpec] = Field(None, description="Behavioral response specification")
    binning: Optional[List[BinSpec]] = Field(None, description="Trial binning specifications")

    # ERP components to extract (for ERP feature extraction stage)
    erp_components: Optional[List[ERPComponentSpec]] = Field(
        None,
        description="ERP components to extract (P1, N1, P3, etc.) - used in features stage"
    )

    # Rejection criteria
    reject: Optional[Dict[str, float]] = Field(None, description="Rejection thresholds by channel type")
    flat: Optional[Dict[str, float]] = Field(None, description="Flat detection thresholds")

    def get_event_codes_to_epoch(self) -> List[Union[int, str]]:
        """Get list of event codes to extract epochs for.

        Extracts all event codes from all conditions.
        This is used to filter which events to epoch (ignoring trial_start, triggers, etc.).

        Returns
        -------
        list
            Flat list of all event codes from all conditions
        """
        event_codes = []
        for condition in self.conditions:
            event_codes.extend(condition.event_codes)
        return event_codes

    # Additional metadata
    sampling_rate: Optional[float] = Field(None, description="Expected sampling rate (Hz)")
    notes: Optional[str] = Field(None, description="Additional notes about this task")

    @classmethod
    def from_yaml(cls, path: Path) -> "TaskConfig":
        """Load task config from YAML file."""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save task config to YAML file."""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.dict(exclude_none=True), f, default_flow_style=False)


class TrialSorter:
    """Sort and filter epochs based on task configuration."""

    def __init__(self, task_config: TaskConfig):
        self.config = task_config

    def sort_by_condition(self, epochs, metadata: Optional[Dict] = None):
        """Sort trials by experimental condition."""
        sorted_epochs = {}
        for condition in self.config.conditions:
            # Get epochs for this condition's event codes (legacy)
            # condition_epochs = epochs[condition.name]

            try:
                # First try direct name lookup (condition.name matches event name exactly)
                condition_epochs = epochs[condition.name]
            except KeyError:
                # Fall back to event_codes-based lookup.
                # Handles cases where condition names (e.g. "target") differ from
                # event names stored in the epochs (e.g. "Stimulus/S 1").
                matching_event_names = []
                for code in condition.event_codes:
                    if isinstance(code, int):
                        # Find event names whose numeric code matches
                        for event_name, event_code in epochs.event_id.items():
                            if event_code == code:
                                matching_event_names.append(event_name)
                    elif isinstance(code, str) and code in epochs.event_id:
                        matching_event_names.append(code)

                if not matching_event_names:
                    raise ValueError(
                        f"Condition '{condition.name}' with event_codes {condition.event_codes} "
                        f"could not be matched to any events in epochs. "
                        f"Available events: {list(epochs.event_id.keys())}"
                    )
                condition_epochs = epochs[matching_event_names]

            sorted_epochs[condition.name] = {
                'epochs': condition_epochs,
                'n_trials': len(condition_epochs),
                'description': condition.description
            }
        return sorted_epochs

    def sort_by_response(self, epochs, behavioral_data):
        """Sort trials by behavioral response (correct/incorrect/etc)."""
        if not self.config.response_mapping:
            raise ValueError("No response mapping defined in task config")

        response_col = self.config.response_mapping.response_column
        categories = self.config.response_mapping.categories

        sorted_epochs = {}
        for category_name, category_value in categories.items():
            # Filter epochs where response matches this category
            mask = behavioral_data[response_col] == category_value
            sorted_epochs[category_name] = {
                'epochs': epochs[mask],
                'n_trials': mask.sum()
            }

        return sorted_epochs

    def bin_trials(self, epochs, behavioral_data, bin_spec: BinSpec):
        """Bin trials by a variable (e.g., RT quartiles)."""
        import numpy as np
        import pandas as pd

        if bin_spec.method == 'quantile':
            # Bin by quantiles
            values = behavioral_data[bin_spec.column]
            bins = pd.qcut(values, q=bin_spec.n_bins, labels=bin_spec.labels, duplicates='drop')

        elif bin_spec.method == 'fixed':
            # Fixed bin edges
            values = behavioral_data[bin_spec.column]
            bins = pd.cut(values, bins=bin_spec.n_bins, labels=bin_spec.labels)

        elif bin_spec.method == 'custom':
            # Custom bin edges
            values = behavioral_data[bin_spec.column]
            bins = pd.cut(values, bins=bin_spec.bin_edges, labels=bin_spec.labels)

        else:
            raise ValueError(f"Unknown binning method: {bin_spec.method}")

        # Sort epochs into bins
        binned_epochs = {}
        for label in bins.unique():
            if pd.notna(label):
                mask = bins == label
                binned_epochs[str(label)] = {
                    'epochs': epochs[mask],
                    'n_trials': mask.sum()
                }

        return binned_epochs
