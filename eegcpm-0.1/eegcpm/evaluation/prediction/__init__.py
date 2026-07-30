"""Connectome Predictive Modeling implementation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# Re-export the new ridge-CPM (Stage 5)
from .ridge_cpm import (
    ALPHA_GRID,
    ALPHA_GRID_RULE,
    CV_INNER_FOLDS,
    CV_N_REPEATS,
    CV_SEED,
    ESCALATION_P_HI,
    ESCALATION_P_LO,
    MC_CI_LEVEL,
    N_PERMUTATIONS_CONFIRMATORY,
    N_PERMUTATIONS_SCREENING,
    adaptive_permutation,
    fit_predict_outer_fold,
    repeated_cv_run,
)


class CPMPredictor:
    """
    Connectome Predictive Modeling (CPM) predictor.

    Implements the CPM approach:
    1. Edge selection based on correlation with target
    2. Summary score computation (positive/negative networks)
    3. Prediction using selected edges

    References:
        Shen et al. (2017) Nature Protocols
    """

    def __init__(
        self,
        threshold: float = 0.05,
        tail: str = "both",  # positive, negative, both
    ):
        """
        Initialize CPM predictor.

        Args:
            threshold: P-value threshold for edge selection
            tail: Which edges to use (positive, negative, both)
        """
        self.threshold = threshold
        self.tail = tail

        # Fitted parameters
        self.positive_mask: Optional[np.ndarray] = None
        self.negative_mask: Optional[np.ndarray] = None
        self.model_positive: Optional[Any] = None
        self.model_negative: Optional[Any] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "CPMPredictor":
        """
        Fit CPM model.

        Args:
            X: Feature matrix (n_subjects, n_edges)
            y: Target variable (n_subjects,)

        Returns:
            self
        """
        n_subjects, n_edges = X.shape

        # Edge selection: correlate each edge with target
        r_values = np.zeros(n_edges)
        p_values = np.zeros(n_edges)

        for i in range(n_edges):
            r, p = stats.pearsonr(X[:, i], y)
            r_values[i] = r
            p_values[i] = p

        # Create masks
        self.positive_mask = (p_values < self.threshold) & (r_values > 0)
        self.negative_mask = (p_values < self.threshold) & (r_values < 0)

        # Compute summary scores
        positive_sum = np.sum(X[:, self.positive_mask], axis=1) if self.positive_mask.any() else np.zeros(n_subjects)
        negative_sum = np.sum(X[:, self.negative_mask], axis=1) if self.negative_mask.any() else np.zeros(n_subjects)

        # Fit linear models
        from sklearn.linear_model import LinearRegression

        if self.tail in ["positive", "both"] and self.positive_mask.any():
            self.model_positive = LinearRegression()
            self.model_positive.fit(positive_sum.reshape(-1, 1), y)

        if self.tail in ["negative", "both"] and self.negative_mask.any():
            self.model_negative = LinearRegression()
            self.model_negative.fit(negative_sum.reshape(-1, 1), y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix (n_subjects, n_edges)

        Returns:
            Predictions (n_subjects,)
        """
        n_subjects = X.shape[0]
        predictions = np.zeros(n_subjects)
        n_models = 0

        if self.model_positive is not None and self.positive_mask.any():
            positive_sum = np.sum(X[:, self.positive_mask], axis=1)
            predictions += self.model_positive.predict(positive_sum.reshape(-1, 1))
            n_models += 1

        if self.model_negative is not None and self.negative_mask.any():
            negative_sum = np.sum(X[:, self.negative_mask], axis=1)
            predictions += self.model_negative.predict(negative_sum.reshape(-1, 1))
            n_models += 1

        if n_models > 1:
            predictions /= n_models

        return predictions

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        strategy: str = "kfold",
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            X: Feature matrix
            y: Target variable
            cv: Number of folds
            strategy: CV strategy (kfold, leave_one_out)

        Returns:
            CV results dict
        """
        from sklearn.model_selection import KFold, LeaveOneOut

        if strategy == "leave_one_out":
            splitter = LeaveOneOut()
        else:
            splitter = KFold(n_splits=cv, shuffle=True, random_state=42)

        predictions = np.zeros_like(y)

        for train_idx, test_idx in splitter.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]

            # Fit on training data
            temp_model = CPMPredictor(threshold=self.threshold, tail=self.tail)
            temp_model.fit(X_train, y_train)

            # Predict on test
            predictions[test_idx] = temp_model.predict(X_test)

        # Compute metrics
        r, p = stats.pearsonr(predictions, y)
        mse = np.mean((predictions - y) ** 2)
        r2 = 1 - np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)

        return {
            "predictions": predictions,
            "r": r,
            "p_value": p,
            "r2": r2,
            "mse": mse,
            "n_positive_edges": self.positive_mask.sum() if self.positive_mask is not None else 0,
            "n_negative_edges": self.negative_mask.sum() if self.negative_mask is not None else 0,
        }


class WithinSubjectPredictor:
    """
    Within-subject (trial-by-trial) prediction.

    Predicts trial-level outcomes from trial-level features.
    """

    def __init__(self, model_type: str = "ridge"):
        self.model_type = model_type
        self.model = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "WithinSubjectPredictor":
        """
        Fit within-subject model.

        Args:
            X: Trial features (n_trials, n_features)
            y: Trial outcomes (n_trials,)
        """
        from eegcpm.evaluation.models import ModelFactory

        self.model = ModelFactory.create(self.model_type)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict trial outcomes."""
        return self.model.predict(X)


class BetweenGroupPredictor:
    """
    Between-group prediction (classification).

    Classifies subjects into groups based on features.
    """

    def __init__(self, model_type: str = "svm"):
        self.model_type = model_type
        self.model = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "BetweenGroupPredictor":
        """
        Fit group classifier.

        Args:
            X: Subject features (n_subjects, n_features)
            y: Group labels (n_subjects,)
        """
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="linear", probability=True)),
        ])
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict group labels."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict group probabilities."""
        return self.model.predict_proba(X)
