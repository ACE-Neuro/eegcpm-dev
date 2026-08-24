"""Prediction workflow helpers for Streamlit and batch scripts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class FeatureKey:
    """Parsed connectivity matrix key."""

    condition: str
    window: str
    method: str
    band: str
    statistic: str
    raw: str


def normalize_subject_id(value: Any) -> str:
    """Normalize participant identifiers to BIDS-style ``sub-XXX`` IDs."""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    if text.startswith("sub-"):
        return text

    match = re.search(r"(\d+)", text)
    if match:
        return f"sub-{int(match.group(1)):03d}"

    return f"sub-{text}"


def extract_numeric(value: Any) -> float:
    """Extract a numeric questionnaire response from raw Qualtrics-like values."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text:
        return np.nan

    parenthesized = re.search(r"\((-?\d+(?:\.\d+)?)\)", text)
    if parenthesized:
        return float(parenthesized.group(1))

    numeric = re.search(r"-?\d+(?:\.\d+)?", text)
    if numeric:
        return float(numeric.group(0))

    return np.nan


def load_behavior_table(path: Path) -> pd.DataFrame:
    """Load behavior/questionnaire data from CSV, TSV, or Excel."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported behavior file type: {path.suffix}")


def prepare_target_table(
    behavior: pd.DataFrame,
    subject_column: str,
    target_column: str | None = None,
    build_columns: list[str] | None = None,
    target_name: str = "target",
) -> pd.DataFrame:
    """Create a subject-target table from a behavior dataframe."""
    if subject_column not in behavior.columns:
        raise ValueError(f"Subject column not found: {subject_column}")

    out = pd.DataFrame()
    out["subject"] = behavior[subject_column].map(normalize_subject_id)

    if build_columns:
        missing = [col for col in build_columns if col not in behavior.columns]
        if missing:
            raise ValueError(f"Score columns not found: {missing}")
        numeric = behavior[build_columns].applymap(extract_numeric)
        out[target_name] = numeric.sum(axis=1, min_count=len(build_columns))
    elif target_column:
        if target_column not in behavior.columns:
            raise ValueError(f"Target column not found: {target_column}")
        out[target_name] = behavior[target_column].map(extract_numeric)
    else:
        raise ValueError("Select a target column or columns to build a score")

    out = out.replace({"subject": {"": np.nan}}).dropna(subset=["subject", target_name])
    out = out.drop_duplicates(subset=["subject"], keep="last")
    return out.reset_index(drop=True)


def parse_connectivity_key(key: str) -> FeatureKey | None:
    """Parse keys like ``target_baseline_plv_theta_mean``."""
    parts = key.split("_")
    if len(parts) < 5:
        return None
    statistic = parts[-1]
    band = parts[-2]
    method = parts[-3]
    window = parts[-4]
    condition = "_".join(parts[:-4])
    if statistic not in {"mean", "std", "variance"}:
        return None
    return FeatureKey(condition, window, method, band, statistic, key)


def scan_connectivity_options(project_root: Path, pipeline: str) -> dict[str, Any]:
    """Scan connectivity outputs and summarize available options."""
    base = project_root / "derivatives" / "connectivity" / pipeline
    options: dict[str, Any] = {
        "base": base,
        "tasks": [],
        "variants": [],
        "subjects": [],
        "conditions": [],
        "windows": [],
        "methods": [],
        "bands": [],
        "statistics": [],
        "keys": [],
    }
    if not base.exists():
        return options

    tasks: set[str] = set()
    variants: set[str] = set()
    subjects: set[str] = set()
    parsed_values: dict[str, set[str]] = {
        "conditions": set(),
        "windows": set(),
        "methods": set(),
        "bands": set(),
        "statistics": set(),
        "keys": set(),
    }

    for conn_file in base.glob("*/variant-*/*/*_connectivity.npz"):
        rel = conn_file.relative_to(base).parts
        if len(rel) < 4:
            continue
        tasks.add(rel[0])
        variants.add(rel[1].removeprefix("variant-"))
        subjects.add(rel[2])

    for task in sorted(tasks):
        for variant in sorted(variants):
            sample = next((base / task / f"variant-{variant}").glob("sub-*/*_connectivity.npz"), None)
            if sample is None:
                continue
            try:
                with np.load(sample, allow_pickle=True) as npz:
                    for key in npz.files:
                        parsed = parse_connectivity_key(key)
                        if parsed is None:
                            continue
                        parsed_values["conditions"].add(parsed.condition)
                        parsed_values["windows"].add(parsed.window)
                        parsed_values["methods"].add(parsed.method)
                        parsed_values["bands"].add(parsed.band)
                        parsed_values["statistics"].add(parsed.statistic)
                        parsed_values["keys"].add(parsed.raw)
            except Exception:
                continue

    options.update({
        "tasks": sorted(tasks),
        "variants": sorted(variants),
        "subjects": sorted(subjects),
        **{key: sorted(value) for key, value in parsed_values.items()},
    })
    return options


def _selected_keys(
    keys: list[str],
    conditions: list[str],
    windows: list[str],
    methods: list[str],
    bands: list[str],
    statistics: list[str],
) -> list[str]:
    selected = []
    for key in keys:
        parsed = parse_connectivity_key(key)
        if parsed is None:
            continue
        if conditions and parsed.condition not in conditions:
            continue
        if windows and parsed.window not in windows:
            continue
        if methods and parsed.method not in methods:
            continue
        if bands and parsed.band not in bands:
            continue
        if statistics and parsed.statistic not in statistics:
            continue
        selected.append(key)
    return sorted(selected)


def build_connectivity_feature_table(
    project_root: Path,
    pipeline: str,
    subjects: list[str],
    tasks: list[str],
    variants: list[str],
    conditions: list[str],
    windows: list[str],
    methods: list[str],
    bands: list[str],
    statistics: list[str],
    clip_negative_dwpli: bool = True,
) -> pd.DataFrame:
    """Build a subject-by-edge feature table from connectivity matrices."""
    base = project_root / "derivatives" / "connectivity" / pipeline
    rows: list[dict[str, float | str]] = []
    subjects = sorted({normalize_subject_id(subject) for subject in subjects if normalize_subject_id(subject)})

    for subject in subjects:
        row: dict[str, float | str] = {"subject": subject}
        for task in tasks:
            for variant in variants:
                conn_file = base / task / f"variant-{variant}" / subject / f"{subject}_connectivity.npz"
                if not conn_file.exists():
                    continue
                with np.load(conn_file, allow_pickle=True) as npz:
                    keys = _selected_keys(
                        list(npz.files), conditions, windows, methods, bands, statistics
                    )
                    for key in keys:
                        matrix = np.asarray(npz[key], dtype=float)
                        if matrix.shape[0] != matrix.shape[1]:
                            continue
                        parsed = parse_connectivity_key(key)
                        values = matrix.copy()
                        if clip_negative_dwpli and parsed and parsed.method == "dwpli":
                            values = np.maximum(values, 0.0)
                        tri_i, tri_j = np.triu_indices(values.shape[0], k=1)
                        prefix = f"{task}|{variant}|{key}"
                        for i, j, value in zip(tri_i, tri_j, values[tri_i, tri_j]):
                            row[f"{prefix}|roi{i:02d}-roi{j:02d}"] = float(value)
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df.columns) > 1:
        feature_cols = [col for col in df.columns if col != "subject"]
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    return df


def create_regression_model(model_name: str, random_state: int = 42) -> Any:
    """Create a scikit-learn-compatible regression model."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    if model_name == "elasticnet":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=random_state)),
        ])
    if model_name == "svr":
        # LinearSVR (liblinear) instead of libsvm SVR: libsvm's SMO solver can
        # fail to converge on certain training folds (observed hang on fold 74
        # of sstvis/eLORETA). LinearSVR is the standard choice for linear SVM
        # on high-dimensional data and always terminates.
        from sklearn.svm import LinearSVR
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LinearSVR(random_state=random_state, max_iter=20000)),
        ])
    if model_name == "ridge":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
    if model_name == "pls":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", PLSRegression(n_components=2)),
        ])
    if model_name == "spls_lite":
        return Pipeline([
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_regression, k=100)),
            ("model", PLSRegression(n_components=2)),
        ])
    if model_name == "random_forest":
        return RandomForestRegressor(n_estimators=200, max_depth=4, random_state=random_state)
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost is not installed in this environment") from exc
        return XGBRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            objective="reg:squarederror",
        )
    raise ValueError(f"Unknown model: {model_name}")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics for continuous questionnaire targets."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_rho, spearman_p = stats.spearmanr(y_true, y_pred)
    residual = y_true - y_pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def cross_validate_regression(
    X: pd.DataFrame,
    y: pd.Series,
    models: list[str],
    cv_method: str = "repeated_kfold",
    n_splits: int = 5,
    n_repeats: int = 20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run cross-validated regression and return metrics and predictions."""
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import KFold, LeaveOneOut, RepeatedKFold
    from sklearn.pipeline import Pipeline

    if cv_method == "leave_one_out":
        splitter = LeaveOneOut()
    elif cv_method == "kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        splitter = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )

    X_values = X.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for model_name in models:
        fold_predictions = np.full((len(y_values), 0), np.nan)
        repeat_pred: list[np.ndarray] = []
        split_number = 0
        for train_idx, test_idx in splitter.split(X_values):
            split_number += 1
            base_model = create_regression_model(model_name, random_state=random_state)
            model = Pipeline([("impute", SimpleImputer(strategy="median")), ("regressor", base_model)])
            model.fit(X_values[train_idx], y_values[train_idx])
            pred = np.asarray(model.predict(X_values[test_idx])).reshape(-1)
            split_vector = np.full(len(y_values), np.nan)
            split_vector[test_idx] = pred
            repeat_pred.append(split_vector)

            for idx, p in zip(test_idx, pred):
                prediction_rows.append({
                    "model": model_name,
                    "split": split_number,
                    "subject": X.index[idx],
                    "observed": y_values[idx],
                    "predicted": float(p),
                })

        if repeat_pred:
            fold_predictions = np.column_stack(repeat_pred)
            mean_pred = np.nanmean(fold_predictions, axis=1)
            row = {"model": model_name, **regression_metrics(y_values, mean_pred)}
            row["n_subjects"] = len(y_values)
            row["n_features"] = X.shape[1]
            metric_rows.append(row)

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def permutation_test_pearson(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> dict[str, float]:
    """Permutation p-value for Pearson correlation."""
    rng = np.random.default_rng(random_state)
    observed = stats.pearsonr(y_true, y_pred).statistic
    null = [stats.pearsonr(rng.permutation(y_true), y_pred).statistic for _ in range(n_permutations)]
    null_arr = np.asarray(null)
    return {
        "observed_pearson_r": float(observed),
        "permutation_p": float(np.mean(np.abs(null_arr) >= abs(observed))),
        "null_mean": float(np.mean(null_arr)),
        "null_std": float(np.std(null_arr)),
    }


def save_prediction_outputs(
    output_root: Path,
    config: dict[str, Any],
    feature_table: pd.DataFrame,
    target_table: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> Path:
    """Save a reproducible prediction run."""
    run_name = datetime.now().strftime("run-%Y%m%d-%H%M%S") + f"-{uuid4().hex[:8]}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "prediction_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    feature_table.to_csv(run_dir / "feature_table.csv", index=False)
    target_table.to_csv(run_dir / "target_table.csv", index=False)
    metrics.to_csv(run_dir / "model_metrics.csv", index=False)
    predictions.to_csv(run_dir / "predictions_long.csv", index=False)
    return run_dir
