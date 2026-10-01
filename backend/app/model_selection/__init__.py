"""Module 5 model-selection primitives.

This package contains the algorithm registry and model factory used by the
Module 5 screening engine. It deliberately does not run cross-validation or
make final model-selection decisions.
"""

from .algorithm_specs import AlgorithmSpec
from .algorithm_registry import get_algorithm, list_algorithms
from .model_factory import build_estimator

__all__ = ["AlgorithmSpec", "get_algorithm", "list_algorithms", "build_estimator"]

from .arbiter import ArbiterConfig, select_final_candidate

__all__ = ["ArbiterConfig", "select_final_candidate"]
