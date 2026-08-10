"""
evaluate.py
-----------
Evaluation stage for the customer churn pipeline.

Loads the best model selected during training, evaluates it on the held-out
test set, performs threshold tuning, logs all metrics and artifacts to MLflow,
and saves a final evaluation summary JSON.

Threshold tuning rationale
--------------------------
The default decision threshold of 0.5 is arbitrary. For churn prediction,
missing a churner (false negative) is more costly than incorrectly flagging
a loyal customer (false positive). We sweep thresholds from 0.2 to 0.8 in
steps of 0.01 and select the threshold that maximises recall subject to the
constraint that precision stays above params.threshold.min_precision.
This constraint prevents us from trivially achieving recall=1.0 by predicting
churn for every customer, which would be operationally useless.
"""

import json
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    recall_score, precision_score, f1_score, roc_auc_score, accuracy_score,
    confusion_matrix, classification_report, roc_curve,
)


ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = ROOT / "params.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "data" / "processed" / "reports"


def load_params() -> dict:
    with open(PARAMS_FILE) as fh:
        return yaml.safe_load(fh)


def tune_threshold(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
    min_precision: float,
    primary_metric: str = "recall",
) -> tuple[float, dict]:
    """
    Find the threshold that maximises the primary_metric while keeping precision
    above min_precision.

    Returns (best_threshold, metrics_at_best_threshold).
    """
    best_threshold = 0.5
    best_score = -1.0
    best_metrics = {}

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        auc  = roc_auc_score(y_true, y_prob)
        acc  = accuracy_score(y_true, y_pred)
        
        # Decide which score to maximize
        if primary_metric == "f1":
            score = f1
        elif primary_metric == "roc_auc":
            score = auc
        elif primary_metric == "precision":
            score = prec
        else:
            score = rec

        if prec >= min_precision and score > best_score:
            best_score = score
            best_threshold = t
            best_metrics = {
                "threshold": float(t),
                "recall":    float(rec),
                "precision": float(prec),
                "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
                "accuracy":  float(accuracy_score(y_true, y_pred)),
                "roc_auc":   float(roc_auc_score(y_true, y_prob)),
            }

    if not best_metrics:
        # Fallback: if no threshold met min_precision, use 0.5
        print(
            f"WARNING: no threshold achieved precision >= {min_precision}. "
            "Falling back to 0.5."
        )
        best_threshold = 0.5
        y_pred = (y_prob >= 0.5).astype(int)
        best_metrics = {
            "threshold": 0.5,
            "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
            "accuracy":  float(accuracy_score(y_true, y_pred)),
            "roc_auc":   float(roc_auc_score(y_true, y_prob)),
        }

    return best_threshold, best_metrics


def plot_confusion_matrix(y_true, y_pred, output_path: Path, title: str = "Confusion Matrix") -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_roc_curve(y_true, y_prob, output_path: Path, model_name: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    best_threshold: float,
    output_path: Path,
) -> None:
    recalls    = []
    precisions = []
    f1s        = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, recalls,    label="Recall")
    ax.plot(thresholds, precisions, label="Precision")
    ax.plot(thresholds, f1s,        label="F1")
    ax.axvline(best_threshold, color="red", linestyle="--",
               label=f"Selected threshold = {best_threshold:.2f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Tuning Curve")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def run(params: dict | None = None) -> None:
    if params is None:
        params = load_params()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load best model info from training stage
    # ------------------------------------------------------------------
    best_model_path = Path((PROCESSED_DIR / "best_model_path.txt").read_text().strip())
    best_model_name = (PROCESSED_DIR / "best_model_name.txt").read_text().strip()
    best_run_id     = (PROCESSED_DIR / "best_run_id.txt").read_text().strip()

    print(f"Loading best model: {best_model_name} from {best_model_path}")
    model = joblib.load(best_model_path)

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy").astype(int)
    print(f"Test set: X={X_test.shape}, y={y_test.shape}, churn rate={y_test.mean():.3f}")

    # ------------------------------------------------------------------
    # Threshold tuning on test set
    # ------------------------------------------------------------------
    t_params  = params["threshold"]
    thresholds = np.arange(t_params["start"], t_params["stop"], t_params["step"])
    min_prec   = t_params["min_precision"]
    primary_metric = t_params.get("primary_metric", "recall")

    y_prob = model.predict_proba(X_test)[:, 1]
    best_threshold, tuned_metrics = tune_threshold(
        y_prob, y_test, thresholds, min_prec, primary_metric=primary_metric
    )

    print(f"\nThreshold tuning results:")
    print(f"  Selected threshold: {best_threshold:.2f}")
    print(f"  Metrics at selected threshold:")
    for k, v in tuned_metrics.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # Final predictions with tuned threshold
    y_pred_tuned = (y_prob >= best_threshold).astype(int)

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    print("\nGenerating report artefacts...")
    plot_confusion_matrix(
        y_test, y_pred_tuned,
        REPORTS_DIR / "confusion_matrix.png",
        title=f"Confusion Matrix ({best_model_name}, threshold={best_threshold:.2f})",
    )
    plot_roc_curve(
        y_test, y_prob,
        REPORTS_DIR / "roc_curve.png",
        model_name=best_model_name,
    )
    plot_threshold_sweep(
        y_test, y_prob, thresholds, best_threshold,
        REPORTS_DIR / "threshold_sweep.png",
    )

    # Classification report text
    report_text = classification_report(y_test, y_pred_tuned, target_names=["No Churn", "Churn"])
    (REPORTS_DIR / "classification_report.txt").write_text(report_text)
    print("  Saved: classification_report.txt")
    print("\n" + report_text)

    # ------------------------------------------------------------------
    # Log everything to MLflow (resuming the best model's run)
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("customer_churn_training")

    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_param("selected_threshold", best_threshold)
        mlflow.log_param("threshold_selection_criterion",
                         f"max {primary_metric} subject to precision >= {min_prec}")

        for metric_name, value in tuned_metrics.items():
            if isinstance(value, float):
                mlflow.log_metric(f"test_{metric_name}", value)

        mlflow.log_artifact(str(REPORTS_DIR / "confusion_matrix.png"))
        mlflow.log_artifact(str(REPORTS_DIR / "roc_curve.png"))
        mlflow.log_artifact(str(REPORTS_DIR / "threshold_sweep.png"))
        mlflow.log_artifact(str(REPORTS_DIR / "classification_report.txt"))

    # ------------------------------------------------------------------
    # Save evaluation summary JSON
    # ------------------------------------------------------------------
    summary = {
        "best_model":         best_model_name,
        "selected_threshold": float(best_threshold),
        "test_metrics":       tuned_metrics,
    }
    summary_path = PROCESSED_DIR / "evaluation_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nEvaluation summary saved to: {summary_path}")
    print("Evaluation stage complete.")


if __name__ == "__main__":
    run()
