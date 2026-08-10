"""
test_model.py
-------------
Unit tests for src/models/train.py and src/models/evaluate.py.

Tests use small synthetic datasets (a few hundred rows) so they run in seconds
without requiring the full pipeline to have been executed.  The tests verify:
  - make_estimator returns the correct sklearn class
  - compute_metrics returns values in expected ranges
  - a model can be trained on a tiny sample and returns a fitted estimator
    (i.e., calling predict_proba does not raise)
  - tune_threshold returns a threshold in [0, 1] and a metrics dict
    with all expected keys
  - the evaluation metrics (recall, precision, F1, ROC-AUC, accuracy) are all
    in [0, 1]
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.train import make_estimator, compute_metrics
from src.models.evaluate import tune_threshold


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_dataset():
    """Small balanced binary classification dataset for fast testing."""
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42,
        class_sep=1.5,
    )
    return X, y


@pytest.fixture
def fitted_lr(small_dataset):
    """Return a fitted LogisticRegression for metric/threshold tests."""
    X, y = small_dataset
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# Tests: make_estimator
# ---------------------------------------------------------------------------

class TestMakeEstimator:
    def test_logistic_regression_type(self):
        est = make_estimator("logistic_regression")
        assert isinstance(est, LogisticRegression)

    def test_random_forest_type(self):
        est = make_estimator("random_forest")
        assert isinstance(est, RandomForestClassifier)

    def test_gradient_boosting_type(self):
        est = make_estimator("gradient_boosting")
        assert isinstance(est, GradientBoostingClassifier)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            make_estimator("neural_network")


# ---------------------------------------------------------------------------
# Tests: compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_all_metric_keys_present(self, fitted_lr):
        model, X, y = fitted_lr
        metrics = compute_metrics(model, X, y)
        expected_keys = {"recall", "precision", "f1", "roc_auc", "accuracy"}
        assert expected_keys == set(metrics.keys())

    def test_all_metrics_in_valid_range(self, fitted_lr):
        model, X, y = fitted_lr
        metrics = compute_metrics(model, X, y)
        for name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{name}={value} is outside [0, 1]"

    def test_custom_threshold_affects_recall(self, fitted_lr):
        """Lower threshold should increase recall (more positives predicted)."""
        model, X, y = fitted_lr
        metrics_default = compute_metrics(model, X, y, threshold=0.5)
        metrics_low     = compute_metrics(model, X, y, threshold=0.2)
        # Recall at a lower threshold should be >= recall at default threshold
        assert metrics_low["recall"] >= metrics_default["recall"]

    def test_high_threshold_lowers_recall(self, fitted_lr):
        model, X, y = fitted_lr
        metrics_default = compute_metrics(model, X, y, threshold=0.5)
        metrics_high    = compute_metrics(model, X, y, threshold=0.9)
        # Very high threshold: fewer positives, recall should drop
        assert metrics_high["recall"] <= metrics_default["recall"]


# ---------------------------------------------------------------------------
# Tests: model training on small data
# ---------------------------------------------------------------------------

class TestModelTraining:
    @pytest.mark.parametrize("model_name", [
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    ])
    def test_estimator_fits_and_predicts(self, model_name, small_dataset):
        """Each estimator should fit on small data and return probabilities."""
        X, y = small_dataset
        model = make_estimator(model_name)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)
        # Probabilities must sum to 1 per row
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    @pytest.mark.parametrize("model_name", [
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    ])
    def test_fitted_model_recall_above_zero(self, model_name, small_dataset):
        """A fitted model on separable data should have recall > 0."""
        X, y = small_dataset
        model = make_estimator(model_name)
        model.fit(X, y)
        metrics = compute_metrics(model, X, y)
        assert metrics["recall"] > 0.0, (
            f"{model_name} has recall=0, which suggests something is wrong"
        )


# ---------------------------------------------------------------------------
# Tests: tune_threshold
# ---------------------------------------------------------------------------

class TestTuneThreshold:
    def test_returns_threshold_in_range(self, fitted_lr):
        model, X, y = fitted_lr
        y_prob = model.predict_proba(X)[:, 1]
        thresholds = np.arange(0.2, 0.8, 0.01)
        best_t, _ = tune_threshold(y_prob, y, thresholds, min_precision=0.3)
        assert 0.0 <= best_t <= 1.0

    def test_returns_all_expected_metric_keys(self, fitted_lr):
        model, X, y = fitted_lr
        y_prob = model.predict_proba(X)[:, 1]
        thresholds = np.arange(0.2, 0.8, 0.01)
        _, metrics = tune_threshold(y_prob, y, thresholds, min_precision=0.3)
        expected = {"threshold", "recall", "precision", "f1", "accuracy", "roc_auc"}
        assert expected == set(metrics.keys())

    def test_all_returned_metrics_in_range(self, fitted_lr):
        model, X, y = fitted_lr
        y_prob = model.predict_proba(X)[:, 1]
        thresholds = np.arange(0.2, 0.8, 0.01)
        _, metrics = tune_threshold(y_prob, y, thresholds, min_precision=0.3)
        for k, v in metrics.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} is outside [0, 1]"

    def test_selected_threshold_achieves_min_precision(self, fitted_lr):
        """The returned threshold should respect the minimum precision constraint."""
        model, X, y = fitted_lr
        y_prob = model.predict_proba(X)[:, 1]
        thresholds = np.arange(0.2, 0.8, 0.01)
        min_prec = 0.5
        best_t, metrics = tune_threshold(y_prob, y, thresholds, min_prec)
        # If precision is not NaN / zero, it should be >= min_prec
        # (unless fallback to 0.5 was triggered)
        if metrics["precision"] > 0.0:
            assert metrics["precision"] >= min_prec or best_t == 0.5

    def test_impossible_precision_falls_back_to_half(self, fitted_lr):
        """If min_precision=1.0 is impossible, fall back to threshold=0.5."""
        model, X, y = fitted_lr
        y_prob = model.predict_proba(X)[:, 1]
        thresholds = np.arange(0.2, 0.8, 0.01)
        best_t, _ = tune_threshold(y_prob, y, thresholds, min_precision=1.0)
        assert best_t == 0.5
