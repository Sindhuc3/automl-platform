from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from .algorithm_specs import AlgorithmSpec


@dataclass(frozen=True)
class EligibilityResult:
    algorithm_id: str
    eligible: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_id": self.algorithm_id,
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


def check_eligibility(
    spec: AlgorithmSpec,
    dataset: Mapping[str, Any],
    *,
    budget: Optional[Mapping[str, Any]] = None,
) -> EligibilityResult:
    """Apply conservative screening gates without looking at validation/test data."""
    budget = budget or {}
    n_rows = int(dataset.get("n_rows", 0))
    n_features = int(dataset.get("n_features", 0))
    sparse = bool(dataset.get("sparse", False))
    reasons: list[str] = []
    warnings: list[str] = []

    if n_rows <= 0:
        reasons.append("Dataset row count is unavailable or zero.")

    # These are screening-cost gates, not quality judgments.
    svm_max_rows = int(budget.get("svm_max_rows", 20000))
    knn_max_rows = int(budget.get("knn_max_rows", 50000))
    high_cost_max_rows = int(budget.get("high_cost_max_rows", 50000))
    dense_feature_cap = int(budget.get("dense_feature_cap", 5000))

    if spec.algorithm_id in {"svm", "svr"} and n_rows > svm_max_rows:
        reasons.append(f"Dataset has {n_rows} rows; {spec.display_name} screening limit is {svm_max_rows}.")
    if spec.algorithm_id == "knn" and n_rows > knn_max_rows:
        reasons.append(f"Dataset has {n_rows} rows; KNN screening limit is {knn_max_rows}.")
    if spec.cost_tier == "high" and n_rows > high_cost_max_rows:
        reasons.append(f"Dataset exceeds the high-cost screening limit of {high_cost_max_rows} rows.")
    if sparse and spec.algorithm_id == "naive_bayes":
        warnings.append("GaussianNB expects a dense numerical matrix; sparse support will be resolved by the preprocessing adapter.")
    if n_features > dense_feature_cap and spec.algorithm_id in {"svm", "svr", "knn"}:
        reasons.append(f"Feature count {n_features} exceeds dense distance/kernel screening cap {dense_feature_cap}.")

    return EligibilityResult(spec.algorithm_id, not reasons, tuple(reasons), tuple(warnings))
