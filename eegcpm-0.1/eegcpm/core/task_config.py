"""Task configuration for epoch extraction and trial sorting.

Defines experimental conditions, behavioral response mapping, and trial binning.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    import mne


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


@dataclass
class SelectionRecord:
    """Provenance for one trial selection (subset, bin, split or sort)."""

    spec: Dict[str, Any]
    """Human-readable spec that produced the selection."""

    n_before: int
    """Number of trials in the input epochs."""

    n_after: int
    """Number of trials in the returned epochs."""

    mask_hash: str
    """SHA-256 of the boolean selection mask (empty for a pure re-order)."""

    source_epochs_hash: str
    """SHA-256 of the input epochs file content; for in-memory epochs, a
    fingerprint of the events matrix (no file exists to hash)."""

    timestamp: str
    """UTC ISO-8601 timestamp of when the selection was made."""


def _as_bool_mask(condition: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Convert a comparison result to a positional boolean ndarray."""
    if isinstance(condition, np.ndarray):
        return np.asarray(condition, dtype=bool)
    if condition.dtype == bool:
        return condition.to_numpy(dtype=bool)
    return condition.fillna(False).astype(bool).to_numpy(dtype=bool)


class TrialSorter:
    """Sort, filter and split epochs based on task configuration and metadata.

    Metadata is resolved at call time: the ``metadata`` argument defaults to
    ``epochs.metadata`` (epochs can receive metadata after construction) and
    must have exactly one row per trial.

    The UI-facing legacy methods (``sort_by_condition``, ``sort_by_response``,
    ``bin_trials``) keep their historical return shapes — each entry gains a
    ``'selection_record'`` key. The new methods (``select``, ``sort``,
    ``split``) return ``(epochs, SelectionRecord)`` tuples.
    """

    def __init__(self, task_config: TaskConfig):
        self.config = task_config

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_metadata(
        epochs: "mne.Epochs",
        metadata: Optional[pd.DataFrame],
        *,
        required: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Default ``metadata`` to ``epochs.metadata`` and validate alignment."""
        if metadata is None:
            metadata = getattr(epochs, "metadata", None)
        if metadata is None:
            if not required:
                return None
            raise ValueError(
                "No trial metadata available — attach it when saving epochs "
                "(EpochExtractionModule.process(..., metadata=<DataFrame>))."
            )
        if not isinstance(metadata, pd.DataFrame):
            raise ValueError(
                f"metadata must be a pandas.DataFrame, got {type(metadata).__name__}"
            )
        if len(metadata) != len(epochs):
            raise ValueError(
                f"metadata has {len(metadata)} rows but epochs has "
                f"{len(epochs)} trials; pass metadata aligned to the epochs "
                "(MNE subsets it on rejection)."
            )
        return metadata

    @staticmethod
    def _require_column(metadata: pd.DataFrame, column: str, context: str = "") -> None:
        if column not in metadata.columns:
            suffix = f" ({context})" if context else ""
            raise ValueError(
                f"Column {column!r}{suffix} not found in trial metadata. "
                f"Available columns: {list(metadata.columns)}"
            )

    @staticmethod
    def _source_epochs_hash(epochs: "mne.Epochs") -> str:
        """SHA-256 of the epochs file content (events fingerprint in memory)."""
        filenames = [
            str(f) for f in (getattr(epochs, "filenames", None) or []) if str(f)
        ]
        for filename in filenames:
            path = Path(filename)
            if path.is_file():
                digest = hashlib.sha256()
                with open(path, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()
        events = np.asarray(
            getattr(epochs, "events", np.empty((0, 3), dtype=int)), dtype=np.int64
        )
        return hashlib.sha256(
            np.ascontiguousarray(events).tobytes()
        ).hexdigest()

    @staticmethod
    def _mask_hash(mask: Optional[np.ndarray]) -> str:
        if mask is None:
            return ""
        mask_arr = np.ascontiguousarray(np.asarray(mask, dtype=bool))
        return hashlib.sha256(mask_arr.tobytes()).hexdigest()

    def _make_record(
        self,
        source_epochs: "mne.Epochs",
        selected_epochs: "mne.Epochs",
        spec: Dict[str, Any],
        *,
        mask: Optional[np.ndarray] = None,
    ) -> SelectionRecord:
        return SelectionRecord(
            spec=spec,
            n_before=len(source_epochs),
            n_after=len(selected_epochs),
            mask_hash=self._mask_hash(mask),
            source_epochs_hash=self._source_epochs_hash(source_epochs),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _compute_bins(values: pd.Series, bin_spec: BinSpec) -> pd.Series:
        """Bin ``values`` according to ``bin_spec`` (shared by select/bin_trials)."""
        if bin_spec.method == "quantile":
            if not bin_spec.n_bins:
                raise ValueError(
                    f"Binning spec '{bin_spec.name}' (quantile) needs n_bins"
                )
            return pd.qcut(
                values,
                q=bin_spec.n_bins,
                labels=bin_spec.labels,
                duplicates="drop",
            )
        if bin_spec.method == "fixed":
            if not bin_spec.n_bins:
                raise ValueError(
                    f"Binning spec '{bin_spec.name}' (fixed) needs n_bins"
                )
            return pd.cut(values, bins=bin_spec.n_bins, labels=bin_spec.labels)
        if bin_spec.method == "custom":
            if not bin_spec.bin_edges:
                raise ValueError(
                    f"Binning spec '{bin_spec.name}' (custom) needs bin_edges"
                )
            return pd.cut(values, bins=bin_spec.bin_edges, labels=bin_spec.labels)
        raise ValueError(f"Unknown binning method: {bin_spec.method}")

    def _bin_spec(self, name: str) -> BinSpec:
        specs = {spec.name: spec for spec in (self.config.binning or [])}
        if not specs:
            raise ValueError("No binning specifications defined in task config")
        if name not in specs:
            raise ValueError(
                f"Unknown binning spec {name!r}. Available bins: {list(specs)}"
            )
        return specs[name]

    # ------------------------------------------------------------------
    # new API
    # ------------------------------------------------------------------

    def select(
        self,
        epochs: "mne.Epochs",
        *,
        metadata: Optional[pd.DataFrame] = None,
        where: Optional[str] = None,
        conditions: Optional[List[str]] = None,
        response: Optional[str] = None,
        accuracy: Optional[int] = None,
        rt: tuple = (None, None),
        group: Optional[str] = None,
    ) -> Union[
        "tuple[mne.Epochs, SelectionRecord]",
        "Dict[str, tuple[mne.Epochs, SelectionRecord]]",
    ]:
        """Build a boolean mask from the predicates and return the subset.

        Predicates (AND-combined):
        - ``where``: pandas query string evaluated on the metadata.
        - ``conditions``: list of condition names (subset of
          ``self.config.conditions``); their ``event_codes`` are matched
          against the metadata ``trial_type`` column.
        - ``response``: category name from
          ``self.config.response_mapping.categories``.
        - ``accuracy``: 0 or 1, matched against
          ``self.config.response_mapping.response_column``.
        - ``rt``: ``(lo, hi)`` bounds in the units of
          ``self.config.response_mapping.rt_column``; ``None`` means open.
          Trials with missing RT are excluded when a bound is set.

        ``group`` changes the return signature: when it names a
        ``BinSpec`` from ``self.config.binning``, the (filtered) trials are
        binned and the method returns ``Dict[str, Tuple[Epochs,
        SelectionRecord]]`` keyed by bin label instead of a single epochs
        object. Trials with missing bin values are excluded from the bins
        (visible via the records' ``n_before``/``n_after`` counts).

        Returns:
            ``(epochs, SelectionRecord)``, or a dict of those tuples when
            ``group`` is given.

        Raises:
            ValueError: unknown column, unknown condition, unknown response
                category, invalid accuracy/rt, or a selection with zero
                trials.
        """
        from eegcpm.modules.epochs.trial_filter import build_mask, query_mask

        metadata = self._resolve_metadata(epochs, metadata)
        spec: Dict[str, Any] = {}
        mask = np.ones(len(metadata), dtype=bool)

        if where is not None:
            spec["where"] = where
            mask &= query_mask(metadata, where)

        if conditions is not None:
            if isinstance(conditions, str):
                raise ValueError(
                    "conditions must be a list of condition names, not a bare string"
                )
            conditions = list(conditions)
            available = {cond.name: cond for cond in self.config.conditions}
            unknown = [name for name in conditions if name not in available]
            if unknown:
                raise ValueError(
                    f"Unknown condition(s) {unknown}. "
                    f"Available conditions: {list(available)}"
                )
            codes: List[Any] = []
            for name in conditions:
                for code in available[name].event_codes:
                    if code not in codes:
                        codes.append(code)
            spec["conditions"] = conditions
            spec["condition_codes"] = list(codes)
            mask &= build_mask(metadata, conditions=codes)

        if response is not None:
            rm = self.config.response_mapping
            if rm is None:
                raise ValueError(
                    "No response_mapping defined in task config; "
                    "cannot select by response"
                )
            if response not in rm.categories:
                raise ValueError(
                    f"Unknown response category {response!r}. "
                    f"Available categories: {list(rm.categories)}"
                )
            spec["response"] = response
            mask &= build_mask(
                metadata,
                response=rm.categories[response],
                response_column=rm.response_column,
            )

        if accuracy is not None:
            if accuracy not in (0, 1):
                raise ValueError(f"accuracy must be 0, 1 or None, got {accuracy!r}")
            rm = self.config.response_mapping
            if rm is None:
                raise ValueError(
                    "No response_mapping defined in task config; "
                    "cannot select by accuracy"
                )
            spec["accuracy"] = int(accuracy)
            mask &= build_mask(
                metadata,
                accuracy=int(accuracy),
                accuracy_column=rm.response_column,
            )

        if rt is None:
            rt = (None, None)
        if not isinstance(rt, (tuple, list)) or len(rt) != 2:
            raise ValueError(
                f"rt must be a (lo, hi) tuple with None for open bounds, got {rt!r}"
            )
        rt_lo, rt_hi = rt
        if rt_lo is not None or rt_hi is not None:
            rm = self.config.response_mapping
            rt_column = rm.rt_column if rm is not None else None
            if rt_column is None:
                raise ValueError(
                    "rt bounds requested but the task config has no "
                    "response_mapping.rt_column"
                )
            spec["rt"] = [rt_lo, rt_hi]
            mask &= build_mask(metadata, rt=(rt_lo, rt_hi), rt_column=rt_column)

        if group is not None:
            bin_spec = self._bin_spec(group)
            if bin_spec.column is None:
                raise ValueError(
                    f"Binning spec '{group}' has no column to bin"
                )
            self._require_column(
                metadata, bin_spec.column, context=f"binning spec '{group}'"
            )
            bins = self._compute_bins(metadata[bin_spec.column], bin_spec)
            binned: Dict[str, Any] = {}
            for label in bins.cat.categories:
                combined = mask & _as_bool_mask(bins == label)
                n_after = int(combined.sum())
                if n_after == 0:
                    continue
                bin_epochs = epochs[combined]
                binned[str(label)] = (
                    bin_epochs,
                    self._make_record(
                        epochs,
                        bin_epochs,
                        {**spec, "group": group, "bin": str(label)},
                        mask=combined,
                    ),
                )
            if not binned:
                raise ValueError(
                    f"Selection for group {group!r} produced no trials "
                    f"(spec: {spec}, n_before={len(epochs)})"
                )
            return binned

        n_after = int(mask.sum())
        if n_after == 0:
            raise ValueError(
                f"Selection produced no trials (spec: {spec}, "
                f"n_before={len(epochs)})"
            )
        selected = epochs[mask]
        record = self._make_record(epochs, selected, spec, mask=mask)
        return selected, record

    def sort(
        self,
        epochs: "mne.Epochs",
        *,
        metadata: Optional[pd.DataFrame] = None,
        by: str,
        ascending: bool = True,
    ) -> "tuple[mne.Epochs, SelectionRecord]":
        """Return epochs reordered so that ``metadata[by]`` is sorted.

        Stable sort (ties keep their original relative order); the epoch data
        is reordered together with the metadata so the trial alignment is
        preserved.

        Raises:
            ValueError: unknown column.
        """
        metadata = self._resolve_metadata(epochs, metadata)
        self._require_column(metadata, by, context="sort key")

        frame = pd.DataFrame(
            {"_position": np.arange(len(metadata)), "_key": metadata[by].to_numpy()}
        )
        ordered = frame.sort_values("_key", ascending=ascending, kind="stable")
        positions = ordered["_position"].to_numpy(dtype=int)

        sorted_epochs = epochs[positions]
        record = self._make_record(
            epochs,
            sorted_epochs,
            {"sort_by": by, "ascending": bool(ascending), "n_trials": len(epochs)},
        )
        return sorted_epochs, record

    def split(
        self,
        epochs: "mne.Epochs",
        *,
        metadata: Optional[pd.DataFrame] = None,
        by: str = "position",
        parts: int = 2,
    ) -> "Dict[str, tuple[mne.Epochs, SelectionRecord]]":
        """Split epochs sequentially or by quantiles of a metadata column.

        - ``by='position'``: ``parts`` sequential chunks (first half, second
          half, ...), keyed ``'half_1'``, ``'half_2'``, ...
        - ``by=<column>``: ``parts`` quantile bins of that column, keyed
          ``'bin_1'``, ``'bin_2'``, ... in ascending order. Ties at a bin edge
          collapse the bin (pandas ``duplicates='drop'``), so fewer bins than
          requested is possible; trials with missing values are excluded
          (visible via the records' ``n_before``/``n_after`` counts).

        Returns:
            Dict mapping part name to ``(epochs, SelectionRecord)``.

        Raises:
            ValueError: unknown column, invalid ``parts``, or ``parts``
                exceeding the number of trials for a positional split.
        """
        metadata = self._resolve_metadata(epochs, metadata)
        if not isinstance(parts, (int, np.integer)) or parts < 1:
            raise ValueError(f"parts must be a positive integer, got {parts!r}")
        parts = int(parts)
        n_trials = len(epochs)
        if n_trials == 0:
            raise ValueError("Cannot split empty epochs")

        split_parts: Dict[str, Any] = {}
        if by == "position":
            if parts > n_trials:
                raise ValueError(
                    f"Cannot split {n_trials} trials into {parts} parts"
                )
            for index, chunk in enumerate(np.array_split(np.arange(n_trials), parts)):
                chunk = np.asarray(chunk, dtype=int)
                mask = np.zeros(n_trials, dtype=bool)
                mask[chunk] = True
                key = f"half_{index + 1}"
                part_epochs = epochs[chunk]
                split_parts[key] = (
                    part_epochs,
                    self._make_record(
                        epochs,
                        part_epochs,
                        {"split_by": "position", "parts": parts, "part": key},
                        mask=mask,
                    ),
                )
            return split_parts

        self._require_column(metadata, by, context="split key")
        values = metadata[by]
        bins = pd.qcut(values, q=parts, duplicates="drop")
        if bool(bins.isna().all()):
            raise ValueError(
                f"Cannot split by {by!r}: all values are missing/NaN"
            )
        for index, category in enumerate(bins.cat.categories, start=1):
            mask = _as_bool_mask(bins == category)
            key = f"bin_{index}"
            part_epochs = epochs[mask]
            split_parts[key] = (
                part_epochs,
                self._make_record(
                    epochs,
                    part_epochs,
                    {
                        "split_by": by,
                        "parts": parts,
                        "bin": key,
                        "bin_category": str(category),
                    },
                    mask=mask,
                ),
            )
        return split_parts

    # ------------------------------------------------------------------
    # legacy UI-facing API (return shapes unchanged)
    # ------------------------------------------------------------------

    def sort_by_condition(self, epochs, metadata: Optional[pd.DataFrame] = None):
        """Sort trials by experimental condition.

        Returns the legacy dict ``{condition_name: {'epochs', 'n_trials',
        'description'}}`` (UI page 9); each entry gains a
        ``'selection_record'``. Metadata is optional here because condition
        selection uses the epochs' event ids, but it is used for provenance
        when available.
        """
        metadata = self._resolve_metadata(epochs, metadata, required=False)
        sorted_epochs = {}
        for condition in self.config.conditions:
            # First try direct name lookup (condition.name matches event name exactly)
            try:
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

            selection = getattr(condition_epochs, "selection", None)
            mask = None
            if selection is not None:
                mask = np.zeros(len(epochs), dtype=bool)
                mask[np.asarray(selection, dtype=int)] = True

            sorted_epochs[condition.name] = {
                'epochs': condition_epochs,
                'n_trials': len(condition_epochs),
                'description': condition.description,
                'selection_record': self._make_record(
                    epochs,
                    condition_epochs,
                    {
                        "condition": condition.name,
                        "event_codes": list(condition.event_codes),
                    },
                    mask=mask,
                ),
            }
        return sorted_epochs

    def sort_by_response(self, epochs, behavioral_data=None):
        """Sort trials by behavioral response (correct/incorrect/etc).

        ``behavioral_data`` defaults to ``epochs.metadata`` and must have one
        row per trial. Returns the legacy dict shape (plus
        ``'selection_record'`` per entry).
        """
        if not self.config.response_mapping:
            raise ValueError("No response mapping defined in task config")

        behavioral_data = self._resolve_metadata(epochs, behavioral_data)
        response_col = self.config.response_mapping.response_column
        self._require_column(behavioral_data, response_col, context="response filter")
        categories = self.config.response_mapping.categories

        sorted_epochs = {}
        for category_name, category_value in categories.items():
            # Filter epochs where response matches this category
            mask = _as_bool_mask(behavioral_data[response_col] == category_value)
            category_epochs = epochs[mask]
            sorted_epochs[category_name] = {
                'epochs': category_epochs,
                'n_trials': int(mask.sum()),
                'selection_record': self._make_record(
                    epochs,
                    category_epochs,
                    {
                        "response": category_name,
                        "column": response_col,
                        "value": category_value,
                    },
                    mask=mask,
                ),
            }

        return sorted_epochs

    def bin_trials(self, epochs, behavioral_data=None, bin_spec: Optional[BinSpec] = None):
        """Bin trials by a variable (e.g., RT quartiles).

        ``behavioral_data`` defaults to ``epochs.metadata`` and must have one
        row per trial. Returns the legacy dict shape (plus
        ``'selection_record'`` per entry).
        """
        behavioral_data = self._resolve_metadata(epochs, behavioral_data)
        if bin_spec is None:
            available = [spec.name for spec in (self.config.binning or [])]
            raise ValueError(
                f"bin_spec is required. Available binning specs: {available}"
            )
        if not isinstance(bin_spec, BinSpec):
            raise ValueError(
                f"bin_spec must be a BinSpec, got {type(bin_spec).__name__}"
            )
        if bin_spec.column is None:
            raise ValueError(f"Binning spec '{bin_spec.name}' has no column to bin")
        self._require_column(
            behavioral_data, bin_spec.column, context=f"binning spec '{bin_spec.name}'"
        )

        bins = self._compute_bins(behavioral_data[bin_spec.column], bin_spec)

        # Sort epochs into bins. Iterate the categorical categories so bin
        # order is deterministic (label order for labelled specs), not the
        # first-appearance order of Series.unique().
        categories = (
            bins.cat.categories if hasattr(bins, "cat") else bins.unique()
        )
        binned_epochs = {}
        for label in categories:
            if pd.isna(label):
                continue
            mask = _as_bool_mask(bins == label)
            n_trials = int(mask.sum())
            if n_trials == 0:
                continue
            bin_epochs = epochs[mask]
            binned_epochs[str(label)] = {
                'epochs': bin_epochs,
                'n_trials': n_trials,
                'selection_record': self._make_record(
                    epochs,
                    bin_epochs,
                    {
                        "bin_spec": bin_spec.name,
                        "method": bin_spec.method,
                        "bin": str(label),
                    },
                    mask=mask,
                ),
            }

        return binned_epochs

