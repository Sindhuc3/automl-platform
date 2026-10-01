from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_METRIC_DIRECTIONS = {
    "roc_auc": "maximize",
    "average_precision": "maximize",
    "macro_f1": "maximize",
    "weighted_f1": "maximize",
    "accuracy": "maximize",
    "balanced_accuracy": "maximize",
    "precision": "maximize",
    "recall": "maximize",
    "r2": "maximize",
    "rmse": "minimize",
    "mae": "minimize",
    "log_loss": "minimize",
}

# Smaller values mean simpler/cheaper. This is deliberately a coarse policy,
# not a claim that one model family is universally superior.
_ALGORITHM_COMPLEXITY = {
    "linear_regression": 1,
    "logistic_regression": 1,
    "ridge": 1,
    "lasso": 1,
    "elastic_net": 2,
    "naive_bayes": 1,
    "decision_tree": 2,
    "knn": 2,
    "random_forest": 3,
    "extra_trees": 3,
    "gradient_boosting": 3,
    "xgboost": 4,
    "lightgbm": 4,
    "catboost": 4,
    "svm": 3,
    "svr": 3,
}


@dataclass(frozen=True)
class ArbiterConfig:
    """Deterministic M6.5 selection policy.

    M6 supplies tuned CV evidence. M6.5 turns that evidence into one
    FinalModelSpec for M7. The sealed test set is never accepted as input.
    """

    one_se_multiplier: float = 1.0
    require_guardrail_pass: bool = True
    require_baseline_improvement: bool = True
    improvement_multiplier: float = 1.0
    max_finalists: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "one_se_multiplier": self.one_se_multiplier,
            "require_guardrail_pass": self.require_guardrail_pass,
            "require_baseline_improvement": self.require_baseline_improvement,
            "improvement_multiplier": self.improvement_multiplier,
            "max_finalists": self.max_finalists,
        }


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _metric_direction(metric: str) -> str:
    if metric not in _METRIC_DIRECTIONS:
        raise ValueError(f"M65_UNSUPPORTED_PRIMARY_METRIC: {metric}")
    return _METRIC_DIRECTIONS[metric]


def _score(candidate: Mapping[str, Any]) -> float | None:
    for key in ("mean_primary", "tuned_mean_primary", "best_mean_primary"):
        value = candidate.get(key)
        if value is not None:
            return float(value)
    best_trial = candidate.get("best_trial")
    if isinstance(best_trial, Mapping):
        for key in ("mean_primary", "primary_mean", "score"):
            if best_trial.get(key) is not None:
                return float(best_trial[key])
    return None


def _se(candidate: Mapping[str, Any]) -> float:
    for key in ("corrected_se", "tuned_corrected_se", "best_corrected_se"):
        value = candidate.get(key)
        if value is not None:
            return max(0.0, float(value))
    trial = candidate.get("best_trial")
    if isinstance(trial, Mapping) and trial.get("corrected_se") is not None:
        return max(0.0, float(trial["corrected_se"]))
    return 0.0


def _guardrail_pass(candidate: Mapping[str, Any]) -> bool:
    value = candidate.get("guardrail_pass")
    if value is None:
        value = candidate.get("guardrails_pass")
    return bool(True if value is None else value)


def _baseline_evidence(candidate: Mapping[str, Any]) -> dict[str, Any]:
    evidence = candidate.get("baseline_evidence")
    if isinstance(evidence, Mapping):
        return dict(evidence)
    evidence = candidate.get("m5_baseline_evidence")
    if isinstance(evidence, Mapping):
        return dict(evidence)
    return {}


def _tuned_improvement(candidate: Mapping[str, Any]) -> tuple[float | None, float]:
    """Return tuned-vs-M5 improvement and its uncertainty when available."""
    for key in ("tuned_improvement", "improvement_over_m5", "m6_improvement"):
        if candidate.get(key) is not None:
            return float(candidate[key]), _se(candidate)
    m5_score = candidate.get("m5_mean_primary")
    score = _score(candidate)
    if m5_score is not None and score is not None:
        direction = _metric_direction(str(candidate.get("primary_metric")))
        delta = score - float(m5_score)
        return (delta if direction == "maximize" else -delta), _se(candidate)
    return None, _se(candidate)


def _feature_count(candidate: Mapping[str, Any]) -> int:
    for key in ("n_features", "n_features_out", "feature_count"):
        if candidate.get(key) is not None:
            try:
                return int(candidate[key])
            except (TypeError, ValueError):
                pass
    return 10**9


def _complexity(candidate: Mapping[str, Any]) -> int:
    algorithm_id = str(candidate.get("algorithm_id", ""))
    return _ALGORITHM_COMPLEXITY.get(algorithm_id, 99)


def _fit_seconds(candidate: Mapping[str, Any]) -> float:
    for key in ("fit_seconds", "elapsed_seconds", "best_fit_seconds"):
        if candidate.get(key) is not None:
            try:
                return float(candidate[key])
            except (TypeError, ValueError):
                pass
    return float("inf")


def _secondary_value(candidate: Mapping[str, Any], metric: str) -> float | None:
    metrics = candidate.get("mean_metrics") or candidate.get("metrics") or {}
    if isinstance(metrics, Mapping) and metrics.get(metric) is not None:
        return float(metrics[metric])
    trial = candidate.get("best_trial")
    if isinstance(trial, Mapping):
        metrics = trial.get("mean_metrics") or trial.get("metrics") or {}
        if isinstance(metrics, Mapping) and metrics.get(metric) is not None:
            return float(metrics[metric])
    return None


def _within_one_se(score: float, best: float, best_se: float, direction: str, multiplier: float) -> bool:
    band = max(0.0, multiplier) * best_se
    if direction == "maximize":
        return score >= best - band
    return score <= best + band


def _sort_key(candidate: Mapping[str, Any], direction: str) -> tuple[Any, ...]:
    score = _score(candidate)
    score_key = -(score if score is not None else float("-inf")) if direction == "maximize" else (score if score is not None else float("inf"))
    return (
        score_key,
        _complexity(candidate),
        _feature_count(candidate),
        _fit_seconds(candidate),
        str(candidate.get("algorithm_id", "")),
        str(candidate.get("feature_set_id", "")),
        str(candidate.get("candidate_id", "")),
    )


def select_final_candidate(
    hpo_output: Mapping[str, Any],
    *,
    task: str | None = None,
    config: ArbiterConfig | None = None,
) -> dict[str, Any]:
    """Select one M6 finalist for M7 using predefined CV-only evidence.

    Expected input is the M6 HPO output (or a compatible mapping containing
    ``candidates``). No test-set score is consulted or accepted as evidence.
    """
    cfg = config or ArbiterConfig()
    candidates = list(hpo_output.get("candidates") or hpo_output.get("finalists") or [])
    if not candidates:
        raise ValueError("M65_NO_HPO_CANDIDATES")

    metric = hpo_output.get("primary_metric") or (hpo_output.get("metric") or {}).get("primary_metric")
    if not metric:
        metric = candidates[0].get("primary_metric")
    if not metric:
        raise ValueError("M65_PRIMARY_METRIC_MISSING")
    metric = str(metric)
    direction = _metric_direction(metric)

    usable: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        score = _score(item)
        if score is None:
            rejected.append({"candidate_id": item.get("candidate_id"), "reason": "missing_primary_score"})
            continue
        if cfg.require_guardrail_pass and not _guardrail_pass(item):
            rejected.append({"candidate_id": item.get("candidate_id"), "reason": "guardrail_failed"})
            continue
        evidence = _baseline_evidence(item)
        if cfg.require_baseline_improvement:
            beats = evidence.get("beats_baseline")
            if beats is False:
                rejected.append({"candidate_id": item.get("candidate_id"), "reason": "does_not_beat_baseline"})
                continue
            if beats is None:
                # If M6 did not carry baseline evidence, retain the candidate but
                # make the missing evidence explicit rather than inventing it.
                item["baseline_evidence_status"] = "not_provided"
        improvement, improvement_se = _tuned_improvement({**item, "primary_metric": metric})
        item["tuned_improvement"] = improvement
        item["tuned_improvement_corrected_se"] = improvement_se
        if improvement is not None and cfg.improvement_multiplier > 0:
            item["tuned_improvement_pass"] = bool(improvement > cfg.improvement_multiplier * improvement_se)
        else:
            item["tuned_improvement_pass"] = None
        item["corrected_se"] = _se(item)
        usable.append(item)

    if not usable:
        return {
            "status": "no_eligible_candidate",
            "task": task,
            "primary_metric": metric,
            "metric_direction": direction,
            "selected": None,
            "finalists": [],
            "rejected": rejected,
            "decision_log": [{"event": "selection_failed", "reason": "no_candidate_passed_guardrails_and_baseline"}],
            "m7_handoff": None,
            "config": cfg.to_dict(),
        }

    best = sorted(usable, key=lambda c: _sort_key(c, direction))[0]
    best_score = _score(best)
    best_se = _se(best)
    assert best_score is not None

    for item in usable:
        score = _score(item)
        assert score is not None
        item["within_one_se_of_best"] = _within_one_se(score, best_score, best_se, direction, cfg.one_se_multiplier)

    tie_zone = [c for c in usable if c["within_one_se_of_best"]]

    # Selection rule: the best score establishes the tie zone; within that zone
    # prefer the simpler configuration, then fewer features, lower fit cost,
    # and finally deterministic IDs.
    selected = sorted(tie_zone, key=lambda c: (
        _complexity(c),
        _feature_count(c),
        _fit_seconds(c),
        str(c.get("algorithm_id", "")),
        str(c.get("feature_set_id", "")),
        str(c.get("candidate_id", "")),
    ))[0]

    finalists = sorted(tie_zone, key=lambda c: _sort_key(c, direction))[: max(1, cfg.max_finalists)]

    selected_params = selected.get("best_parameters") or selected.get("parameters")
    best_trial = selected.get("best_trial")
    if selected_params is None and isinstance(best_trial, Mapping):
        selected_params = best_trial.get("parameters")
    selected_params = dict(selected_params or {})

    final_model_spec = {
        "algorithm_id": selected.get("algorithm_id"),
        "feature_set_id": selected.get("feature_set_id"),
        "task": task or selected.get("task"),
        "primary_metric": metric,
        "metric_direction": direction,
        "hyperparameters": selected_params,
        "cv_score": _score(selected),
        "corrected_se": _se(selected),
        "n_features": None if _feature_count(selected) >= 10**9 else _feature_count(selected),
        "selection_basis": {
            "guardrail_pass": _guardrail_pass(selected),
            "within_one_se_of_best": selected["within_one_se_of_best"],
            "tuned_improvement": selected.get("tuned_improvement"),
            "tuned_improvement_corrected_se": selected.get("tuned_improvement_corrected_se"),
            "complexity_tier": _complexity(selected),
        },
    }

    selection_id = f"m65_{_hash(final_model_spec)}"
    return {
        "status": "selected",
        "selection_id": selection_id,
        "task": task or selected.get("task"),
        "primary_metric": metric,
        "metric_direction": direction,
        "selected": final_model_spec,
        "finalists": [
            {
                "candidate_id": c.get("candidate_id"),
                "algorithm_id": c.get("algorithm_id"),
                "feature_set_id": c.get("feature_set_id"),
                "cv_score": _score(c),
                "corrected_se": _se(c),
                "within_one_se_of_best": c["within_one_se_of_best"],
                "complexity_tier": _complexity(c),
                "n_features": None if _feature_count(c) >= 10**9 else _feature_count(c),
            }
            for c in finalists
        ],
        "rejected": rejected,
        "decision_log": [
            {"event": "best_cv_candidate_identified", "candidate_id": best.get("candidate_id")},
            {"event": "one_se_tie_zone_built", "candidate_count": len(tie_zone), "best_score": best_score, "best_corrected_se": best_se},
            {"event": "complexity_tiebreak_applied", "selected_candidate_id": selected.get("candidate_id")},
        ],
        "m4_feedback": {
            "status": "winner_selected",
            "feature_set_id": selected.get("feature_set_id"),
            "algorithm_id": selected.get("algorithm_id"),
        },
        "m7_handoff": {
            "status": "ready",
            "selection_id": selection_id,
            "final_model_spec": final_model_spec,
            "test_set_used": False,
        },
        "reproducibility": {
            "source_screening_id": hpo_output.get("screening_id"),
            "source_hpo_id": hpo_output.get("hpo_id"),
            "policy_hash": _hash(cfg.to_dict()),
        },
        "config": cfg.to_dict(),
    }
