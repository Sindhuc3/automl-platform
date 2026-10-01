from __future__ import annotations

from typing import Any, Mapping

from .algorithm_specs import AlgorithmSpec
from .algorithm_registry import get_algorithm


def _class_weight_kwargs(context: Mapping[str, Any]) -> dict[str, Any]:
    """Return class-balancing kwargs only when M5 explicitly requests them."""
    if context.get("class_weight_mode") == "balanced":
        return {"class_weight": "balanced"}
    return {}


def build_estimator(
    algorithm: str | AlgorithmSpec,
    *,
    context: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Any:
    """Build a fresh estimator for one M5 candidate.

    A fresh estimator is returned on every call so sklearn can safely clone it
    for every CV fold. `context` contains runtime information such as task or
    imbalance handling; `overrides` is reserved for M6/tightly controlled
    guided execution and never silently changes the registry defaults.
    """
    context = context or {}
    overrides = overrides or {}
    spec = algorithm if isinstance(algorithm, AlgorithmSpec) else get_algorithm(algorithm, context.get("task"))
    if spec is None:
        raise ValueError(f"M5_UNKNOWN_ALGORITHM: {algorithm}")

    params = dict(spec.default_parameters)
    params.update(overrides)
    if spec.task == "classification":
        params.update(_class_weight_kwargs(context))
    return spec.estimator_factory(params)
