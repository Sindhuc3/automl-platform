from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Mapping, Optional

Task = str
Factory = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True)
class AlgorithmSpec:
    """Immutable description of one M5 screening algorithm.

    The spec describes *how* M5 should construct and prepare an estimator.
    It intentionally contains no CV execution or winner-selection logic.
    """

    algorithm_id: str
    display_name: str
    task: Task
    family: str
    scaling_profile: str
    cost_tier: str
    estimator_factory: Factory
    default_parameters: Mapping[str, Any] = field(default_factory=dict)
    tunable_parameters: Mapping[str, Any] = field(default_factory=dict)
    supports_predict_proba: bool = False
    supports_decision_function: bool = False
    notes: str = ""
    tags: FrozenSet[str] = frozenset()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_id": self.algorithm_id,
            "display_name": self.display_name,
            "task": self.task,
            "family": self.family,
            "scaling_profile": self.scaling_profile,
            "cost_tier": self.cost_tier,
            "default_parameters": dict(self.default_parameters),
            "tunable_parameters": dict(self.tunable_parameters),
            "supports_predict_proba": self.supports_predict_proba,
            "supports_decision_function": self.supports_decision_function,
            "notes": self.notes,
            "tags": sorted(self.tags),
        }
