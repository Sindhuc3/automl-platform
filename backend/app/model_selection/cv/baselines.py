from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor

from .cv_plan import CVPlan
from .metrics import build_metric_bundle, calculate_metrics


def evaluate_baseline(X: Any, y: Any, *, task: str, cv_plan: CVPlan) -> dict[str, Any]:
    """Evaluate one deterministic dummy baseline on the same frozen CV plan."""
    X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y_series = pd.Series(y).reset_index(drop=True)
    bundle = build_metric_bundle(task, y_series)

    if task == "classification":
        estimator = DummyClassifier(strategy="prior")
        baseline_strategy = "prior"
    elif task == "regression":
        # Mean is the V1 screening baseline; median is retained as metadata so
        # later work can compare it when the target distribution warrants it.
        estimator = DummyRegressor(strategy="mean")
        baseline_strategy = "mean"
    else:
        raise ValueError(f"M5_BASELINE_UNSUPPORTED_TASK: {task}")

    folds: list[dict[str, Any]] = []
    for fold in cv_plan.iter_folds():
        train_idx = np.asarray(fold.train_indices)
        val_idx = np.asarray(fold.validation_indices)
        try:
            model = estimator.__class__(**estimator.get_params())
            model.fit(X_df.iloc[train_idx], y_series.iloc[train_idx])
            pred = model.predict(X_df.iloc[val_idx])
            score_values = None
            if hasattr(model, "predict_proba"):
                score_values = model.predict_proba(X_df.iloc[val_idx])
            metrics = calculate_metrics(task, y_series.iloc[val_idx], pred, y_score=score_values)
            primary = float(metrics[bundle.primary]) if task != "regression" or bundle.primary not in {"rmse", "mae"} else -float(metrics[bundle.primary])
            folds.append({
                "repeat": fold.repeat,
                "fold": fold.fold,
                "status": "success",
                "primary": primary,
                "metrics": metrics,
                "n_train": fold.n_train,
                "n_validation": fold.n_validation,
                "error": None,
            })
        except Exception as exc:
            folds.append({
                "repeat": fold.repeat,
                "fold": fold.fold,
                "status": "failed",
                "primary": None,
                "metrics": {},
                "n_train": fold.n_train,
                "n_validation": fold.n_validation,
                "error": f"{type(exc).__name__}: {exc}",
            })

    successful = [f for f in folds if f["status"] == "success"]
    scores = [f["primary"] for f in successful]
    return {
        "baseline_id": f"dummy_{task}",
        "strategy": baseline_strategy,
        "task": task,
        "cv_plan_id": cv_plan.cv_plan_id,
        "primary_metric": bundle.primary,
        "mean_primary": float(np.mean(scores)) if scores else None,
        "std_primary": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "n_successful_folds": len(successful),
        "n_failed_folds": len(folds) - len(successful),
        "fold_results": folds,
        "status": "success" if successful else "failed",
    }
