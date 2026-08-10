"""
train.py
--------
Model training stage for the customer churn pipeline.

Three classifiers are trained and compared:
  1. Logistic Regression  - linear baseline
  2. Random Forest        - non-linear ensemble, handles mixed feature types well
  3. Gradient Boosting    - boosted trees, often best on tabular data

Class imbalance is addressed with SMOTE during preprocessing (see preprocess.py).
The class_weight='balanced' flag is also set for Logistic Regression and Random
Forest as an extra guard in case any residual imbalance remains after SMOTE.

Hyperparameters are tuned with GridSearchCV scored on RECALL (primary metric)
with 5-fold cross-validation.

Each model's best estimator, parameters, and CV metrics are logged to MLflow.
The best model (highest recall on the validation set) is saved as a joblib
file and its path written to data/processed/best_model_path.txt so the
evaluate stage can pick it up without needing to re-scan mlruns/.
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, accuracy_score


ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = ROOT / "params.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "data" / "processed" / "models"


def load_params() -> dict:
    with open(PARAMS_FILE) as fh:
        return yaml.safe_load(fh)


def load_processed_data():
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    X_val   = np.load(PROCESSED_DIR / "X_val.npy")
    y_val   = np.load(PROCESSED_DIR / "y_val.npy")
    return X_train, y_train, X_val, y_val


def build_param_grid(model_name: str, params: dict) -> dict:
    """Return the sklearn-style param grid for a model name."""
    model_params = params["models"][model_name]
    grid = {}
    for k, v in model_params.items():
        # GridSearchCV uses estimator__param format for pipelines,
        # but here we pass the grid directly to the estimator
        grid[k] = v
    return grid


def make_estimator(model_name: str):
    """Return a base (unfitted) sklearn estimator for the given model name."""
    if model_name == "logistic_regression":
        return LogisticRegression(random_state=42)
    elif model_name == "random_forest":
        return RandomForestClassifier(random_state=42, n_jobs=-1)
    elif model_name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def compute_metrics(model, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    """Return a dict of classification metrics at a given decision threshold."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "recall":    recall_score(y, y_pred,    zero_division=0),
        "precision": precision_score(y, y_pred, zero_division=0),
        "f1":        f1_score(y, y_pred,        zero_division=0),
        "roc_auc":   roc_auc_score(y, y_prob),
        "accuracy":  accuracy_score(y, y_pred),
    }


def train_model(
    model_name: str,
    estimator,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    gs_params: dict,
    experiment_name: str,
) -> tuple:
    """
    Run GridSearchCV for one model, log everything to MLflow, return
    (best_estimator, val_metrics, run_id).
    """
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=gs_params["cv"],
        scoring=gs_params["scoring"],
        n_jobs=gs_params["n_jobs"],
        refit=gs_params["refit"],
        verbose=1,
        return_train_score=True,
    )

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("cv_folds",   gs_params["cv"])
        mlflow.log_param("cv_scoring", gs_params["scoring"])

        gs.fit(X_train, y_train)
        best = gs.best_estimator_

        # Log best hyperparameters
        for k, v in gs.best_params_.items():
            mlflow.log_param(f"best_{k}", v)

        mlflow.log_metric("cv_best_recall", gs.best_score_)

        # Evaluate on validation set at default threshold
        val_metrics = compute_metrics(best, X_val, y_val, threshold=0.5)
        for metric_name, value in val_metrics.items():
            mlflow.log_metric(f"val_{metric_name}", value)

        # Log the model artifact
        mlflow.sklearn.log_model(best, artifact_path=f"model_{model_name}")

        run_id = run.info.run_id

    print(f"  Best params:  {gs.best_params_}")
    print(f"  CV recall:    {gs.best_score_:.4f}")
    print(f"  Val metrics:  {val_metrics}")

    return best, val_metrics, run_id


def run(params: dict | None = None) -> None:
    if params is None:
        params = load_params()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mlruns_dir = ROOT / "mlruns"
    mlruns_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri((mlruns_dir).as_uri())
    experiment_name = "customer_churn_training"
    mlflow.set_experiment(experiment_name)

    X_train, y_train, X_val, y_val = load_processed_data()
    print(f"Loaded training data: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Loaded validation data: X_val={X_val.shape}, y_val={y_val.shape}")

    gs_params = params["gridsearch"]
    model_names = ["logistic_regression", "random_forest", "gradient_boosting"]

    results = {}
    for model_name in model_names:
        estimator  = make_estimator(model_name)
        param_grid = build_param_grid(model_name, params)
        best_model, val_metrics, run_id = train_model(
            model_name=model_name,
            estimator=estimator,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            gs_params=gs_params,
            experiment_name=experiment_name,
        )
        # Save each model
        model_path = MODELS_DIR / f"{model_name}.pkl"
        joblib.dump(best_model, model_path)
        results[model_name] = {
            "model":      best_model,
            "val_recall": val_metrics["recall"],
            "metrics":    val_metrics,
            "path":       model_path,
            "run_id":     run_id,
        }

    # ------------------------------------------------------------------
    # Select the best model based on validation recall
    # ------------------------------------------------------------------
    best_name = max(results, key=lambda k: results[k]["val_recall"])
    best_info = results[best_name]

    print(f"\n{'='*60}")
    print("Model comparison summary (validation recall):")
    for name, info in results.items():
        marker = " <-- SELECTED" if name == best_name else ""
        print(f"  {name:30s}  recall={info['val_recall']:.4f}{marker}")
    print(f"{'='*60}")

    # Write the best model path so the evaluate stage can load it
    best_model_path = best_info["path"]
    (PROCESSED_DIR / "best_model_path.txt").write_text(str(best_model_path))
    (PROCESSED_DIR / "best_model_name.txt").write_text(best_name)
    (PROCESSED_DIR / "best_run_id.txt").write_text(best_info["run_id"])

    print(f"\nBest model: {best_name}  (saved to {best_model_path})")
    print("Training stage complete.")


if __name__ == "__main__":
    run()
