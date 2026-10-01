from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, log_loss, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)


@dataclass(frozen=True)
class MetricBundle:
    task: str
    primary: str
    secondary: tuple[str, ...]
    scorers: dict[str, Callable[[Any, Any], float]]


def _classification_scores(y_true: Any, y_pred: Any, y_score: Any | None, *, multiclass: bool) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro" if multiclass else "binary", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro" if multiclass else "binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro" if multiclass else "binary", zero_division=0)),
    }
    if y_score is not None:
        try:
            if multiclass:
                out["roc_auc"] = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
                out["log_loss"] = float(log_loss(y_true, y_score))
            else:
                score = y_score[:, 1] if np.ndim(y_score) == 2 else y_score
                out["roc_auc"] = float(roc_auc_score(y_true, score))
                out["average_precision"] = float(average_precision_score(y_true, score))
                if np.ndim(y_score) == 2:
                    out["log_loss"] = float(log_loss(y_true, y_score))
        except ValueError:
            pass
    return out


def build_metric_bundle(task: str, y: Any) -> MetricBundle:
    """Choose the V1 primary metric before candidate results are seen."""
    task = task.lower()
    y_arr = np.asarray(y)
    if task == "classification":
        classes, counts = np.unique(y_arr, return_counts=True)
        if len(classes) < 2:
            raise ValueError("M5_METRIC_NEEDS_TWO_CLASSES")
        if len(classes) == 2:
            minority_fraction = float(counts.min() / counts.sum())
            primary = "average_precision" if minority_fraction < 0.10 else "roc_auc"
            secondary = ("roc_auc", "average_precision", "f1", "precision", "recall", "accuracy", "balanced_accuracy", "log_loss")
            return MetricBundle("classification", primary, secondary, {})
        return MetricBundle(
            "classification", "macro_f1",
            ("weighted_f1", "accuracy", "balanced_accuracy", "log_loss", "roc_auc"), {},
        )
    if task == "regression":
        y_num = y_arr.astype(float)
        q1, q3 = np.nanpercentile(y_num, [25, 75])
        iqr = q3 - q1
        outlier_fraction = float(np.mean((y_num < q1 - 1.5 * iqr) | (y_num > q3 + 1.5 * iqr))) if iqr > 0 else 0.0
        primary = "mae" if outlier_fraction > 0.05 else "rmse"
        return MetricBundle("regression", primary, ("rmse", "mae", "r2"), {})
    raise ValueError(f"M5_METRIC_UNSUPPORTED_TASK: {task}")


def calculate_metrics(task: str, y_true: Any, y_pred: Any, *, y_score: Any | None = None) -> dict[str, float]:
    if task == "regression":
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }
    multiclass = len(np.unique(y_true)) > 2
    scores = _classification_scores(y_true, y_pred, y_score, multiclass=multiclass)
    if multiclass:
        scores["macro_f1"] = scores.pop("f1")
        scores["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    return scores


def corrected_se(scores: Any, *, n_train: int, n_validation: int) -> float:
    """Nadeau-Bengio corrected resampled estimate of uncertainty."""
    values = np.asarray(scores, dtype=float)
    if values.size < 2:
        return 0.0
    variance = float(np.var(values, ddof=1))
    return float(np.sqrt((1.0 / values.size + n_validation / max(n_train, 1)) * variance))


def paired_corrected_se(differences: Any, *, n_train: int, n_validation: int) -> float:
    values = np.asarray(differences, dtype=float)
    if values.size < 2:
        return 0.0
    variance = float(np.var(values, ddof=1))
    return float(np.sqrt((1.0 / values.size + n_validation / max(n_train, 1)) * variance))
