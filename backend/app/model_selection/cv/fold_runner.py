from __future__ import annotations

import time
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..model_factory import build_estimator
from ..algorithm_registry import get_algorithm
from .cv_plan import CVPlan
from .metrics import calculate_metrics, build_metric_bundle


def _as_frame(X: Any) -> pd.DataFrame:
    return X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


def _apply_scaling(estimator: Any, scaling_profile: str) -> Any:
    if scaling_profile == "none":
        return estimator
    if scaling_profile == "standard":
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    raise ValueError(f"M5_UNKNOWN_SCALING_PROFILE: {scaling_profile}")


def _score_vector(task: str, metrics: Mapping[str, float], primary: str) -> float:
    if task == "regression" and primary in {"rmse", "mae"}:
        return -float(metrics[primary])
    return float(metrics[primary])


def _prepare_feature_set(
    feature_set: Any,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: Any,
):
    """Fit a feature-set recipe on the training fold and transform both folds.

    V1 supports static source/output columns, cloneable sklearn transformers,
    and the M4 Compact recipe. Any supervised recipe receives ``y_train`` only
    from the current CV training fold, preventing target leakage.
    """
    if feature_set is None:
        return X_train, X_val, int(X_train.shape[1]), [str(c) for c in X_train.columns]

    if isinstance(feature_set, Mapping) and feature_set.get("columns") is not None:
        columns = list(feature_set["columns"])
        missing = [c for c in columns if c not in X_train.columns]
        if missing:
            raise ValueError(f"M5_FEATURE_SET_COLUMNS_MISSING: {missing}")
        train_out = X_train.loc[:, columns].copy()
        val_out = X_val.loc[:, columns].copy()

        # M4-generated numeric features are already outside Module 3's fitted
        # source preprocessing. When requested, scale only those generated
        # columns here, leaving one-hot/frequency/boolean source features alone.
        scale_columns = [
            c for c in (feature_set.get("scale_columns") or [])
            if c in train_out.columns
        ]
        if scale_columns:
            scaler = StandardScaler()
            train_out.loc[:, scale_columns] = scaler.fit_transform(train_out[scale_columns])
            val_out.loc[:, scale_columns] = scaler.transform(val_out[scale_columns])
        return train_out, val_out, len(columns), [str(c) for c in columns]

    transformer = feature_set
    if isinstance(feature_set, Mapping) and callable(feature_set.get("transformer_factory")):
        transformer = feature_set["transformer_factory"]()
    elif isinstance(feature_set, Mapping) and feature_set.get("selector") == "compact":
        from .feature_sets import CompactFeatureSelector
        transformer = CompactFeatureSelector(
            task=str(context.get("task") or ""),
            correlation_threshold=float(feature_set.get("correlation_threshold", 0.98)),
            n_shuffles=int(feature_set.get("n_shuffles", 5)),
            null_quantile=float(feature_set.get("null_quantile", 0.95)),
            min_features=int(feature_set.get("min_features", 5)),
            random_state=int(feature_set.get("random_state", 42)),
        )
    elif callable(feature_set) and not hasattr(feature_set, "fit"):
        transformer = feature_set()

    if not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
        raise ValueError("M5_INVALID_FEATURE_SET_RECIPE")

    transformer = clone(transformer) if hasattr(transformer, "get_params") else transformer
    transformer.fit(X_train, y_train)
    train_out = transformer.transform(X_train)
    val_out = transformer.transform(X_val)
    names = None
    if hasattr(transformer, "get_feature_names_out"):
        try:
            names = [str(v) for v in transformer.get_feature_names_out(X_train.columns)]
        except Exception:
            try:
                names = [str(v) for v in transformer.get_feature_names_out()]
            except Exception:
                names = None
    train_out = pd.DataFrame(
        train_out,
        index=X_train.index,
        columns=names if names and len(names) == np.asarray(train_out).shape[1] else None,
    )
    val_out = pd.DataFrame(val_out, index=X_val.index, columns=train_out.columns)
    return train_out, val_out, int(train_out.shape[1]), [str(c) for c in train_out.columns]

def evaluate_candidate(
    X: Any,
    y: Any,
    *,
    algorithm_id: str,
    task: str,
    cv_plan: CVPlan,
    context: Mapping[str, Any] | None = None,
    estimator_overrides: Mapping[str, Any] | None = None,
    feature_set: Any | None = None,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate one algorithm + feature-set recipe on a frozen CV plan."""
    X_df = _as_frame(X)
    y_series = pd.Series(y).reset_index(drop=True)
    metric_bundle = build_metric_bundle(task, y_series)
    context = dict(context or {})
    context["task"] = task
    spec = get_algorithm(algorithm_id, task)
    if spec is None:
        raise ValueError(f"M5_UNKNOWN_ALGORITHM: {algorithm_id}")
    estimator = build_estimator(spec, context=context, overrides=estimator_overrides)
    estimator = _apply_scaling(estimator, context.get("scaling_profile", spec.scaling_profile))

    fold_results: list[dict[str, Any]] = []
    for fold in cv_plan.iter_folds():
        started = time.perf_counter()
        train_idx = np.asarray(fold.train_indices)
        val_idx = np.asarray(fold.validation_indices)
        try:
            X_train = X_df.iloc[train_idx]
            X_val = X_df.iloc[val_idx]
            X_train_fs, X_val_fs, n_out, feature_names = _prepare_feature_set(feature_set, X_train, X_val, y_series.iloc[train_idx])
            model = clone(estimator)
            model.fit(X_train_fs, y_series.iloc[train_idx])
            fit_seconds = time.perf_counter() - started
            predict_started = time.perf_counter()
            pred = model.predict(X_val_fs)
            predict_seconds = time.perf_counter() - predict_started
            score_values = None
            if hasattr(model, "predict_proba"):
                try:
                    score_values = model.predict_proba(X_val_fs)
                except Exception:
                    score_values = None
            elif hasattr(model, "decision_function"):
                try:
                    score_values = model.decision_function(X_val_fs)
                except Exception:
                    score_values = None
            metrics = calculate_metrics(task, y_series.iloc[val_idx], pred, y_score=score_values)
            fold_results.append({
                "repeat": fold.repeat,
                "fold": fold.fold,
                "status": "success",
                "primary": _score_vector(task, metrics, metric_bundle.primary),
                "metrics": metrics,
                "n_train": fold.n_train,
                "n_validation": fold.n_validation,
                "fit_seconds": round(fit_seconds, 6),
                "predict_seconds": round(predict_seconds, 6),
                "n_features_in": int(X_train.shape[1]),
                "n_features_out": n_out,
                "feature_names": feature_names,
                "error": None,
            })
        except Exception as exc:
            fold_results.append({
                "repeat": fold.repeat,
                "fold": fold.fold,
                "status": "failed",
                "primary": None,
                "metrics": {},
                "n_train": fold.n_train,
                "n_validation": fold.n_validation,
                "fit_seconds": round(time.perf_counter() - started, 6),
                "predict_seconds": None,
                "n_features_in": int(X_df.shape[1]),
                "n_features_out": None,
                "feature_names": [],
                "error": f"{type(exc).__name__}: {exc}",
            })

    successful = [r for r in fold_results if r["status"] == "success"]
    primary_scores = [r["primary"] for r in successful]
    mean_primary = float(np.mean(primary_scores)) if primary_scores else None
    std_primary = float(np.std(primary_scores, ddof=1)) if len(primary_scores) > 1 else 0.0
    return {
        "candidate_id": candidate_id or f"{algorithm_id}:{cv_plan.cv_plan_id}",
        "algorithm_id": algorithm_id,
        "task": task,
        "cv_plan_id": cv_plan.cv_plan_id,
        "primary_metric": metric_bundle.primary,
        "mean_primary": mean_primary,
        "std_primary": std_primary,
        "n_successful_folds": len(successful),
        "n_failed_folds": len(fold_results) - len(successful),
        "status": "success" if successful else "failed",
        "fold_results": fold_results,
    }
