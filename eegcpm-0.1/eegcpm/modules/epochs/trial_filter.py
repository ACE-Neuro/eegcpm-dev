"""Trial filtering functions for epoch QC and selection.

Two layers:

- :func:`build_mask` / :func:`query_mask` compose a boolean ``numpy`` mask
  over an events/metadata DataFrame (used by
  :meth:`eegcpm.core.task_config.TrialSorter.select`). Masks are positional
  and therefore safe to apply to ``mne.Epochs`` whose metadata index has gaps
  (MNE keeps the original row labels after rejection).
- :func:`filter_by_rt`, :func:`filter_by_accuracy`,
  :func:`drop_missing_responses`, :func:`drop_sequential_positions` and
  :func:`apply_trial_filters` are the legacy DataFrame-level helpers. Their
  call signatures are unchanged.

Everything fails loudly: unknown columns, unknown ``keep`` values and
invalid bounds raise ``ValueError`` with the available options listed.
"""

from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = [
    "build_mask",
    "query_mask",
    "filter_by_rt",
    "filter_by_accuracy",
    "drop_missing_responses",
    "drop_sequential_positions",
    "apply_trial_filters",
]

#: Default columns used when the caller does not override them.
DEFAULT_RT_COLUMN = "response_time"
DEFAULT_ACCURACY_COLUMN = "accuracy"
DEFAULT_TRIAL_TYPE_COLUMN = "trial_type"


def _require_column(df: pd.DataFrame, column: str, context: str = "") -> None:
    """Raise ``ValueError`` when ``column`` is missing from ``df``."""
    if column not in df.columns:
        suffix = f" ({context})" if context else ""
        raise ValueError(
            f"Column {column!r}{suffix} not found in dataframe. "
            f"Available columns: {list(df.columns)}"
        )


def _canonical_label(value: Any) -> str:
    """String form of a condition/event label, normalising integer floats.

    Metadata loaded from BIDS events.tsv may store codes as ``"1"``, ``1`` or
    ``1.0`` depending on the writer; all three canonicalise to ``"1"``.
    """
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        as_float = float(value)
        if np.isfinite(as_float) and as_float.is_integer():
            return str(int(as_float))
        return str(as_float)
    return str(value)


def _match_any(values: pd.Series, candidates: Sequence[Any]) -> np.ndarray:
    """Boolean mask: row value canonicalises to one of ``candidates``."""
    wanted = {_canonical_label(c) for c in candidates}
    return np.asarray(
        [_canonical_label(v) in wanted for v in values.to_numpy()], dtype=bool
    )


def _bool_numpy(condition: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Convert a comparison result to a plain positional boolean ndarray.

    Handles pandas nullable dtypes (``pd.NA``) and non-boolean comparison
    results without ever producing an object-dtype mask.
    """
    if isinstance(condition, np.ndarray):
        return np.asarray(condition, dtype=bool)
    if condition.dtype == bool:
        return condition.to_numpy(dtype=bool)
    return condition.fillna(False).astype(bool).to_numpy(dtype=bool)


def query_mask(metadata_df: pd.DataFrame, where: str) -> np.ndarray:
    """Evaluate a pandas query string and return a positional boolean mask.

    Parameters
    ----------
    metadata_df : pandas.DataFrame
        Trial metadata (columns are referenced by name; use backticks for
        names with dots, e.g. ``where="`key_resp.rt` > 0.5"``).
    where : str
        A pandas ``DataFrame.query`` expression.

    Returns
    -------
    numpy.ndarray
        Boolean mask of length ``len(metadata_df)``.

    Raises
    ------
    ValueError
        For an empty/invalid query, an unknown column, or a DataFrame whose
        index is not unique (positional alignment would be ambiguous).
    """
    if not isinstance(where, str) or not where.strip():
        raise ValueError("where must be a non-empty pandas query string")

    try:
        selected = metadata_df.query(where)
    except pd.errors.UndefinedVariableError as exc:
        raise ValueError(
            f"Unknown column/variable in where clause {where!r}: {exc}. "
            f"Available columns: {list(metadata_df.columns)}"
        ) from exc
    except SyntaxError as exc:
        raise ValueError(
            f"Invalid where clause {where!r}: {exc}"
        ) from exc
    except Exception as exc:  # pandas query engine errors vary by version
        raise ValueError(
            f"where clause {where!r} could not be evaluated: {exc}. "
            f"Available columns: {list(metadata_df.columns)}"
        ) from exc

    if not metadata_df.index.is_unique:
        raise ValueError(
            "Cannot evaluate a where clause on metadata with a non-unique "
            "index: row alignment would be ambiguous. Reset the index first."
        )

    # Align on index labels (MNE metadata keeps its original labels with gaps
    # after rejection), never on positional offsets.
    return np.asarray(metadata_df.index.isin(selected.index), dtype=bool)


def build_mask(
    events_df: pd.DataFrame,
    *,
    where: Optional[str] = None,
    conditions: Optional[Sequence[Any]] = None,
    response: Optional[Any] = None,
    accuracy: Optional[Any] = None,
    rt: Sequence[Optional[float]] = (None, None),
    response_column: Optional[str] = None,
    rt_column: Optional[str] = None,
    accuracy_column: Optional[str] = None,
    trial_type_column: str = DEFAULT_TRIAL_TYPE_COLUMN,
) -> np.ndarray:
    """Build a boolean mask from trial-level predicates (AND-combined).

    Parameters
    ----------
    events_df : pandas.DataFrame
        Events/trial metadata table.
    where : str, optional
        pandas query string evaluated by :func:`query_mask`.
    conditions : sequence, optional
        Trial-type values; rows whose ``trial_type_column`` value matches any
        of them (canonically compared) are kept.
    response : optional
        Response value to match in ``response_column``.
    accuracy : optional
        Accuracy value (0/1) to match in ``accuracy_column`` (falling back to
        ``response_column`` when ``accuracy_column`` is None).
    rt : (lo, hi), optional
        Reaction-time bounds in the units of ``rt_column``; ``None`` means an
        open bound. Rows with missing RT are excluded when any bound is set.
    response_column, rt_column, accuracy_column : str, optional
        Column overrides.
    trial_type_column : str
        Column holding trial-type labels (default ``'trial_type'``).

    Returns
    -------
    numpy.ndarray
        Positional boolean mask of length ``len(events_df)``.

    Raises
    ------
    ValueError
        Unknown column, missing column for a requested predicate, or an
        invalid ``rt`` tuple.
    """
    mask = np.ones(len(events_df), dtype=bool)

    if where is not None:
        mask &= query_mask(events_df, where)

    if conditions is not None:
        _require_column(events_df, trial_type_column, "conditions filter")
        mask &= _match_any(events_df[trial_type_column], conditions)

    if response is not None:
        if response_column is None:
            raise ValueError(
                "response_column is required to filter by response value"
            )
        _require_column(events_df, response_column, "response filter")
        mask &= _bool_numpy(events_df[response_column] == response)

    if accuracy is not None:
        accuracy_col = accuracy_column or response_column
        if accuracy_col is None:
            raise ValueError(
                "accuracy_column (or response_column) is required to filter "
                "by accuracy"
            )
        _require_column(events_df, accuracy_col, "accuracy filter")
        mask &= _bool_numpy(events_df[accuracy_col] == accuracy)

    if rt is None:
        rt = (None, None)
    if not isinstance(rt, (tuple, list)) or len(rt) != 2:
        raise ValueError(
            f"rt must be a (lo, hi) tuple with None for open bounds, got {rt!r}"
        )
    rt_lo, rt_hi = rt
    if rt_lo is not None or rt_hi is not None:
        if rt_column is None:
            raise ValueError("rt_column is required to filter by reaction time")
        _require_column(events_df, rt_column, "reaction-time filter")
        values = events_df[rt_column]
        if rt_lo is not None:
            mask &= _bool_numpy(values >= rt_lo)
        if rt_hi is not None:
            mask &= _bool_numpy(values <= rt_hi)

    return np.asarray(mask, dtype=bool)


def filter_by_rt(
    events_df: pd.DataFrame,
    rt_column: str = DEFAULT_RT_COLUMN,
    rt_min: Optional[float] = None,
    rt_max: Optional[float] = None,
) -> pd.DataFrame:
    """
    Filter trials by reaction time bounds.

    Args:
        events_df: Events dataframe with RT column
        rt_column: Name of RT column (default: "response_time")
        rt_min: Minimum valid RT in seconds (reject faster)
        rt_max: Maximum valid RT in seconds (reject slower)

    Returns:
        Filtered dataframe
    """
    mask = build_mask(
        events_df, rt=(rt_min, rt_max), rt_column=rt_column
    )
    return events_df[mask].copy()


def filter_by_accuracy(
    events_df: pd.DataFrame,
    accuracy_column: str = DEFAULT_ACCURACY_COLUMN,
    keep: str = "correct",
) -> pd.DataFrame:
    """
    Filter trials by response accuracy.

    Args:
        events_df: Events dataframe with accuracy column
        accuracy_column: Name of accuracy column (default: "accuracy")
        keep: Which trials to keep - "correct" or "incorrect"

    Returns:
        Filtered dataframe
    """
    if keep == "correct":
        value = 1
    elif keep == "incorrect":
        value = 0
    else:
        raise ValueError(f"Invalid keep value: {keep}. Use 'correct' or 'incorrect'")

    mask = build_mask(
        events_df, accuracy=value, accuracy_column=accuracy_column
    )
    return events_df[mask].copy()


def drop_missing_responses(
    events_df: pd.DataFrame,
    rt_column: str = DEFAULT_RT_COLUMN,
) -> pd.DataFrame:
    """
    Drop trials with missing responses.

    Args:
        events_df: Events dataframe
        rt_column: Name of RT column (trials with NaN/0 RT are missing)

    Returns:
        Filtered dataframe

    Raises:
        ValueError: The RT column is missing (a configured "drop missing
            responses" step that silently does nothing would corrupt the
            analysis).
    """
    _require_column(events_df, rt_column, "drop missing responses")

    # Drop trials with NaN or 0 RT
    mask = _bool_numpy(events_df[rt_column].notna()) & _bool_numpy(
        events_df[rt_column] > 0
    )

    return events_df[mask].copy()


def drop_sequential_positions(
    events_df: pd.DataFrame,
    drop_first_n: int = 0,
    drop_last_n: int = 0,
) -> pd.DataFrame:
    """
    Drop trials from beginning or end of run.

    Useful for removing practice trials (beginning) or fatigue trials (end).

    Args:
        events_df: Events dataframe
        drop_first_n: Number of trials to drop from beginning
        drop_last_n: Number of trials to drop from end

    Returns:
        Filtered dataframe
    """
    df = events_df.copy()

    n_trials = len(df)
    if drop_first_n + drop_last_n >= n_trials:
        raise ValueError(
            f"Cannot drop {drop_first_n + drop_last_n} trials from {n_trials} total"
        )

    # Slice to keep middle trials
    start_idx = drop_first_n
    end_idx = n_trials - drop_last_n if drop_last_n > 0 else n_trials

    return df.iloc[start_idx:end_idx].reset_index(drop=True)


def apply_trial_filters(
    events_df: pd.DataFrame,
    filters_config: Any,
) -> pd.DataFrame:
    """
    Apply all trial filters from configuration.

    Args:
        events_df: Events dataframe
        filters_config: Configuration object with attributes
            ``drop_first_n``, ``drop_last_n``, ``drop_missing_responses``,
            ``rt_field``, ``rt_min``, ``rt_max``, ``accuracy_filter`` and
            ``accuracy_field`` (duck-typed; no schema class is enforced here).

    Returns:
        Filtered dataframe
    """
    df = events_df.copy()

    # 1. Sequential position first (practice/fatigue trials)
    if filters_config.drop_first_n > 0 or filters_config.drop_last_n > 0:
        df = drop_sequential_positions(
            df,
            drop_first_n=filters_config.drop_first_n,
            drop_last_n=filters_config.drop_last_n,
        )

    # 2. Missing responses
    if filters_config.drop_missing_responses:
        df = drop_missing_responses(df, rt_column=filters_config.rt_field)

    # 3. RT bounds
    if filters_config.rt_min is not None or filters_config.rt_max is not None:
        df = filter_by_rt(
            df,
            rt_column=filters_config.rt_field,
            rt_min=filters_config.rt_min,
            rt_max=filters_config.rt_max,
        )

    # 4. Accuracy filter
    if filters_config.accuracy_filter is not None:
        df = filter_by_accuracy(
            df,
            accuracy_column=filters_config.accuracy_field,
            keep=filters_config.accuracy_filter,
        )

    return df
