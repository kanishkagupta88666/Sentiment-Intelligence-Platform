# ══════════════════════════════════════════════════════════════
#  utils.py
#  Shared helper functions used across all pipeline stages:
#    - evaluate()      → compute and print classification metrics
#    - log_mlflow()    → log a run's params, metrics, and artifacts
# ══════════════════════════════════════════════════════════════

import logging
from pathlib import Path

import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import LABEL_NAMES

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
def evaluate(true_labels, predicted_labels, model_name: str) -> dict:
    """
    Computes Accuracy and Macro F1, prints a full per-class
    classification report, and returns the metrics as a dict.

    Parameters
    ----------
    true_labels       : array-like of int  — ground truth (0 / 1 / 2)
    predicted_labels  : array-like of int  — model predictions
    model_name        : str                — label used in log output

    Returns
    -------
    dict with keys: model, accuracy, macro_f1
    """
    acc = accuracy_score(true_labels, predicted_labels)
    mf1 = f1_score(true_labels, predicted_labels, average="macro")

    log.info("\n─── %s ───", model_name)
    log.info(
        "\n%s",
        classification_report(
            true_labels, predicted_labels, target_names=LABEL_NAMES
        ),
    )
    log.info("  Accuracy : %.4f", acc)
    log.info("  Macro F1 : %.4f", mf1)

    return {"model": model_name, "accuracy": acc, "macro_f1": mf1}


# ──────────────────────────────────────────────────────────────
def log_mlflow(
    run_name:      str,
    params:        dict,
    metrics:       dict,
    artifact_path: str | None = None,
):
    """
    Opens an MLflow run, logs hyperparameters and evaluation metrics,
    and optionally uploads a local directory as artifacts (e.g. saved
    model weights).

    Parameters
    ----------
    run_name      : str        — display name for the MLflow run
    params        : dict       — hyperparameters to log
    metrics       : dict       — evaluation metrics to log
    artifact_path : str | None — local directory to upload as artifacts
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        # only log numeric values as metrics
        numeric_metrics = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        mlflow.log_metrics(numeric_metrics)

        if artifact_path and Path(artifact_path).exists():
            mlflow.log_artifacts(artifact_path, artifact_path="model")

    log.info(
        "MLflow run logged: %-30s  accuracy=%.4f  macro_f1=%.4f",
        run_name,
        metrics.get("accuracy", 0),
        metrics.get("macro_f1", 0),
    )
