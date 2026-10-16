"""Evaluation metrics for regression and classification."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "regression",
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        task: "regression" or "classification"

    Returns:
        Dict of metric name to value
    """
    if task == "regression":
        return compute_regression_metrics(y_true, y_pred)
    elif task == "classification":
        return compute_classification_metrics(y_true, y_pred)
    else:
        raise ValueError(f"Unknown task: {task}")


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute regression metrics."""
    metrics = {}

    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    metrics["r2"] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # MSE and RMSE
    metrics["mse"] = np.mean((y_true - y_pred) ** 2)
    metrics["rmse"] = np.sqrt(metrics["mse"])

    # MAE
    metrics["mae"] = np.mean(np.abs(y_true - y_pred))

    # Pearson correlation
    r, p = stats.pearsonr(y_true, y_pred)
    metrics["pearson_r"] = r
    metrics["pearson_p"] = p

    # Spearman correlation
    rho, p_spearman = stats.spearmanr(y_true, y_pred)
    metrics["spearman_rho"] = rho
    metrics["spearman_p"] = p_spearman

    return metrics


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
    )

    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # Handle multi-class
    unique_classes = np.unique(y_true)
    average = "binary" if len(unique_classes) == 2 else "macro"

    metrics["precision"] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, average=average, zero_division=0)

    # AUC if probabilities available. roc_auc_score raises ValueError when the
    # label set is degenerate (e.g. a single class in the fold); the metric is
    # then *not computable* and that is recorded explicitly instead of leaving
    # a silently missing key (P1-F11).
    if y_prob is not None:
        try:
            if len(unique_classes) == 2:
                metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except ValueError as exc:
            metrics["auc"] = None
            metrics["auc_error"] = str(exc)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def compare_models(
    results: Dict[str, Dict[str, float]],
    metric: str = "r2",
    n_permutations: int = 1000,
) -> Dict[str, Any]:
    """
    Compare multiple models statistically.

    Args:
        results: Dict mapping model name to its metrics
        metric: Metric to compare
        n_permutations: Number of permutations for significance testing

    Returns:
        Comparison results
    """
    model_names = list(results.keys())
    scores = {name: results[name].get(metric, 0) for name in model_names}

    # Rank models
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Pairwise comparisons (placeholder - would need predictions for proper tests)
    comparisons = {}
    for i, (name1, score1) in enumerate(ranked):
        for name2, score2 in ranked[i + 1:]:
            diff = score1 - score2
            comparisons[f"{name1}_vs_{name2}"] = {
                "difference": diff,
                "better": name1,
            }

    return {
        "ranking": ranked,
        "best_model": ranked[0][0],
        "best_score": ranked[0][1],
        "comparisons": comparisons,
    }


def permutation_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: callable,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Perform permutation test for significance.

    Args:
        y_true: True values
        y_pred: Predicted values
        metric_func: Function to compute metric (y_true, y_pred) -> float
        n_permutations: Number of permutations
        random_state: Random seed

    Returns:
        Dict with observed metric and p-value
    """
    np.random.seed(random_state)

    observed = metric_func(y_true, y_pred)

    null_distribution = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y_true)
        null_metric = metric_func(y_perm, y_pred)
        null_distribution.append(null_metric)

    null_distribution = np.array(null_distribution)

    # Two-tailed p-value
    p_value = np.mean(np.abs(null_distribution) >= np.abs(observed))

    return {
        "observed": observed,
        "p_value": p_value,
        "null_mean": np.mean(null_distribution),
        "null_std": np.std(null_distribution),
    }


def cross_validation_summary(
    cv_results: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Summarize cross-validation results.

    Args:
        cv_results: List of metric dicts from each fold

    Returns:
        Summary with mean and std for each metric
    """
    if not cv_results:
        return {}

    metrics = cv_results[0].keys()
    summary = {}

    for metric in metrics:
        values = [r[metric] for r in cv_results if metric in r]
        if values:
            summary[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }

    return summary
