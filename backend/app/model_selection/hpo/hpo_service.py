from __future__ import annotations

import hashlib
import itertools
import json
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..algorithm_registry import get_algorithm
from ..cv.cv_plan import CVPlan, build_cv_plan
from ..cv.fold_runner import evaluate_candidate
from ..cv.metrics import corrected_se


@dataclass(frozen=True)
class HPOConfig:
    """Bounded M6 HPO policy.

    M6 tunes only candidates promoted by M5. It reuses M5's frozen CVPlan
    whenever one is supplied, evaluates every trial with the same folds, and
    never touches a sealed test set.
    """

    max_trials_per_candidate: int = 25
    max_promoted_candidates: int = 4
    global_budget_seconds: float = 600.0
    candidate_budget_seconds: float = 180.0
    max_fits: int = 600
    min_success_fraction: float = 0.80
    random_seed: int = 42
    n_jobs: int = 1

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _direction(primary_metric: str) -> str:
    return "minimize" if primary_metric in {"rmse", "mae", "log_loss"} else "maximize"


def _score(result: Mapping[str, Any]) -> float | None:
    return None if result.get("mean_primary") is None else float(result["mean_primary"])


def _trial_better(a: Mapping[str, Any], b: Mapping[str, Any], direction: str) -> bool:
    sa, sb = _score(a), _score(b)
    if sa is None:
        return False
    if sb is None:
        return True
    return sa < sb if direction == "minimize" else sa > sb


def _sample_params(
    space: Mapping[str, Sequence[Any]],
    *,
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Return deterministic unique configurations from a JSON-friendly space."""
    if not space:
        return [{}]

    keys = list(space)
    values = [list(space[k]) for k in keys]
    all_count = 1
    for vals in values:
        all_count *= max(1, len(vals))

    rng = np.random.default_rng(seed)
    samples: list[dict[str, Any]] = []

    # Enumerate small spaces exactly; sample without replacement for larger ones.
    if all_count <= n:
        for combo in itertools.product(*values):
            samples.append(dict(zip(keys, combo)))
        return samples

    seen: set[str] = set()
    while len(samples) < n:
        combo = [vals[int(rng.integers(0, len(vals)))] for vals in values]
        params = dict(zip(keys, combo))
        token = json.dumps(params, sort_keys=True, default=str)
        if token not in seen:
            seen.add(token)
            samples.append(params)
    return samples


def _feature_set_for(candidate: Mapping[str, Any], feature_sets: Sequence[Any]) -> Any:
    fsid = candidate.get("feature_set_id")
    for i, fs in enumerate(feature_sets):
        if isinstance(fs, Mapping):
            current = str(fs.get("feature_set_id", f"feature_set_{i + 1}"))
        else:
            current = f"feature_set_{i + 1}"
        if fsid is None or current == str(fsid):
            return fs
    raise ValueError(f"M6_FEATURE_SET_NOT_FOUND: {fsid}")


def _candidate_lookup(screening_output: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return only the candidates explicitly promoted by M5.

    M6 must never widen the M5 handoff back to every guardrail-passing
    candidate. The M5 contract is the source of truth: eligible_for_hpo is a
    bounded shortlist produced from baseline evidence + one-SE tie-band +
    guardrails.
    """
    all_candidates = [
        dict(c) for c in (screening_output.get("candidates") or [])
        if c.get("status") == "success" and c.get("mean_primary") is not None
    ]
    if not all_candidates:
        raise ValueError("M6_NO_SUCCESSFUL_M5_CANDIDATES")

    handoff = screening_output.get("m6_handoff") or {}
    eligible_ids = list(
        screening_output.get("eligible_for_hpo")
        or handoff.get("eligible_for_hpo")
        or []
    )
    if not eligible_ids:
        raise ValueError("M6_NO_ELIGIBLE_M5_CANDIDATES")

    by_id = {str(c.get("candidate_id")): c for c in all_candidates}
    missing = [cid for cid in eligible_ids if str(cid) not in by_id]
    if missing:
        raise ValueError(f"M6_HANDOFF_CANDIDATES_MISSING: {missing}")

    candidates = [by_id[str(cid)] for cid in eligible_ids]
    invalid = [
        c.get("candidate_id") for c in candidates
        if c.get("guardrail_pass") is not True
        or (c.get("baseline_evidence") or {}).get("beats_baseline") is not True
        or c.get("within_one_se_of_best") is not True
    ]
    if invalid:
        raise ValueError(f"M6_INVALID_M5_PROMOTION: {invalid}")

    if len(candidates) > 4:
        raise ValueError("M6_PROMOTION_CAP_EXCEEDED: M5 may promote at most 4 candidates")

    return candidates


def _paired_baseline(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    c = {
        (f["repeat"], f["fold"]): f
        for f in candidate.get("fold_results", [])
        if f.get("status") == "success" and f.get("primary") is not None
    }
    b = {
        (f["repeat"], f["fold"]): f
        for f in baseline.get("fold_results", [])
        if f.get("status") == "success" and f.get("primary") is not None
    }
    keys = sorted(set(c) & set(b))
    if not keys:
        return {"n_paired": 0, "mean_improvement": None, "corrected_se": None, "beats_baseline": False}
    diffs = np.asarray([float(c[k]["primary"]) - float(b[k]["primary"]) for k in keys])
    n_train = int(np.median([c[k]["n_train"] for k in keys]))
    n_val = int(np.median([c[k]["n_validation"] for k in keys]))
    se = corrected_se(diffs, n_train=n_train, n_validation=n_val)
    mean = float(np.mean(diffs))
    return {
        "n_paired": len(keys),
        "mean_improvement": mean,
        "corrected_se": float(se),
        "beats_baseline": bool(mean - se > 0),
    }


def _corrected_result(result: dict[str, Any]) -> dict[str, Any]:
    scores = [
        f["primary"] for f in result.get("fold_results", [])
        if f.get("status") == "success" and f.get("primary") is not None
    ]
    if len(scores) > 1:
        n_train = int(np.median([
            f["n_train"] for f in result["fold_results"] if f.get("status") == "success"
        ]))
        n_val = int(np.median([
            f["n_validation"] for f in result["fold_results"] if f.get("status") == "success"
        ]))
        result["corrected_se"] = float(
            corrected_se(scores, n_train=n_train, n_validation=n_val)
        )
    else:
        result["corrected_se"] = 0.0
    return result


def run_hpo(
    X_train: Any,
    y_train: Any,
    *,
    task: str,
    screening_output: Mapping[str, Any],
    feature_sets: Sequence[Any] | None = None,
    cv_plan: CVPlan | None = None,
    config: HPOConfig | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run bounded randomized HPO on M5-promoted candidates.

    Inputs are training data only. ``screening_output`` provides the M5
    evidence and, when possible, its exact frozen CV plan. If a CV plan is
    not supplied directly, this function reconstructs the plan from the M5
    serialized fold indices, so M5 and M6 evaluate on identical folds.
    """
    cfg = config or HPOConfig()
    X = X_train.copy() if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
    y = pd.Series(y_train).reset_index(drop=True)
    if len(X) != len(y):
        raise ValueError("M6_INPUT_LENGTH_MISMATCH")

    candidates = _candidate_lookup(screening_output)
    candidates = candidates[: cfg.max_promoted_candidates]

    metric = str(
        screening_output.get("primary_metric")
        or (screening_output.get("metric") or {}).get("primary_metric")
        or candidates[0]["primary_metric"]
    )

    # Prefer the exact M5 plan object. The CVPlan dataclass is immutable, so
    # reconstructing it from the serialized plan preserves fold identity.
    if cv_plan is None:
        serialized = screening_output.get("cv_plan")
        if serialized and serialized.get("folds"):
            cv_plan = CVPlan(
                cv_plan_id=str(serialized["cv_plan_id"]),
                task=str(serialized["task"]),
                n_splits=int(serialized["n_splits"]),
                n_repeats=int(serialized["n_repeats"]),
                shuffle=bool(serialized.get("shuffle", True)),
                random_seed=int(serialized["random_seed"]),
                strategy=str(serialized["strategy"]),
                folds=tuple(
                    __import__("app.model_selection.cv.cv_plan", fromlist=["Fold"]).Fold(
                        int(f["repeat"]), int(f["fold"]),
                        tuple(f["train_indices"]), tuple(f["validation_indices"])
                    )
                    for f in serialized["folds"]
                ),
                notes=tuple(serialized.get("notes", [])),
            )
        else:
            cv_plan = build_cv_plan(
                X, y, task=task, requested_splits=5,
                requested_repeats=1, random_seed=cfg.random_seed,
            )

    feature_sets = list(feature_sets or [{"feature_set_id": "all_features", "columns": list(X.columns)}])
    baseline = screening_output.get("baseline") or {}

    started = time.perf_counter()
    fit_count = 0
    trials: list[dict[str, Any]] = []
    finalists: list[dict[str, Any]] = []
    decision_log: list[dict[str, Any]] = []

    for candidate_index, m5_candidate in enumerate(candidates):
        if time.perf_counter() - started >= cfg.global_budget_seconds:
            decision_log.append({"event": "global_budget_reached"})
            break

        algorithm_id = str(m5_candidate["algorithm_id"])
        spec = get_algorithm(algorithm_id, task)
        if spec is None:
            decision_log.append({
                "event": "unknown_algorithm_skipped",
                "algorithm_id": algorithm_id,
            })
            continue

        fs = _feature_set_for(m5_candidate, feature_sets)
        # Trial 0 is always the M5/default configuration. The remaining
        # budget is filled with deterministic unique samples from the
        # algorithm-specific M6 search space. This gives HPO a free baseline
        # point and makes the tuned-vs-screening comparison meaningful.
        space = dict(spec.tunable_parameters)
        random_samples = _sample_params(
            space,
            n=max(0, cfg.max_trials_per_candidate - 1),
            seed=cfg.random_seed + candidate_index * 1009,
        )
        default_params = dict(spec.default_parameters)
        samples = [default_params]
        seen = {json.dumps(default_params, sort_keys=True, default=str)}
        for sample in random_samples:
            merged = dict(default_params)
            merged.update(sample)
            token = json.dumps(merged, sort_keys=True, default=str)
            if token not in seen:
                seen.add(token)
                samples.append(merged)
            if len(samples) >= max(1, cfg.max_trials_per_candidate):
                break

        candidate_started = time.perf_counter()
        best: dict[str, Any] | None = None
        candidate_trials: list[str] = []

        for trial_index, params in enumerate(samples):
            if fit_count + len(cv_plan.folds) > cfg.max_fits:
                decision_log.append({"event": "fit_budget_reached", "algorithm_id": algorithm_id})
                break
            if time.perf_counter() - candidate_started >= cfg.candidate_budget_seconds:
                decision_log.append({"event": "candidate_budget_reached", "algorithm_id": algorithm_id})
                break
            if time.perf_counter() - started >= cfg.global_budget_seconds:
                decision_log.append({"event": "global_budget_reached", "algorithm_id": algorithm_id})
                break

            # Merge defaults with the sampled override. The model factory applies
            # these values after its algorithm-specific defaults.
            result = evaluate_candidate(
                X, y,
                algorithm_id=algorithm_id,
                task=task,
                cv_plan=cv_plan,
                context=context,
                estimator_overrides=params,
                feature_set=fs,
                candidate_id=f"m6:{algorithm_id}:{m5_candidate.get('feature_set_id')}:{trial_index}:{cv_plan.cv_plan_id}",
            )
            result = _corrected_result(result)
            result["feature_set_id"] = m5_candidate.get("feature_set_id")
            result["parameters"] = dict(params)
            result["trial_index"] = trial_index
            result["m5_candidate_id"] = m5_candidate.get("candidate_id")
            result["elapsed_seconds"] = round(time.perf_counter() - candidate_started, 6)
            candidate_trials.append(result["candidate_id"])
            trials.append(result)
            fit_count += len(cv_plan.folds)

            if best is None or _trial_better(result, best, _direction(metric)):
                best = result

        if best is None:
            continue

        baseline_evidence = _paired_baseline(best, baseline) if baseline else dict(
            m5_candidate.get("baseline_evidence") or {}
        )
        m5_score = m5_candidate.get("mean_primary")
        best_score = best.get("mean_primary")
        if m5_score is not None and best_score is not None:
            raw_delta = float(best_score) - float(m5_score)
            tuned_improvement = raw_delta if _direction(metric) == "maximize" else -raw_delta
        else:
            tuned_improvement = None

        success_fraction = (
            best["n_successful_folds"] /
            max(1, best["n_successful_folds"] + best["n_failed_folds"])
        )
        promoted = dict(best)
        promoted.update({
            "candidate_id": f"m6_final:{algorithm_id}:{m5_candidate.get('feature_set_id')}",
            "primary_metric": metric,
            "best_trial": {
                "candidate_id": best["candidate_id"],
                "parameters": dict(best.get("parameters") or {}),
                "trial_index": best.get("trial_index"),
                "mean_primary": best.get("mean_primary"),
                "corrected_se": best.get("corrected_se"),
                "fold_results": best.get("fold_results", []),
            },
            "best_parameters": dict(best.get("parameters") or {}),
            "m5_mean_primary": m5_score,
            "tuned_improvement": tuned_improvement,
            "baseline_evidence": baseline_evidence,
            "guardrail_pass": bool(success_fraction >= cfg.min_success_fraction),
            "n_features": max(
                [f["n_features_out"] for f in best.get("fold_results", [])
                 if f.get("status") == "success" and f.get("n_features_out") is not None] or [X.shape[1]]
            ),
            "trials_evaluated": len(candidate_trials),
            "trial_ids": candidate_trials,
            "hpo_elapsed_seconds": round(time.perf_counter() - candidate_started, 6),
        })
        finalists.append(promoted)

    direction = _direction(metric)
    finalists.sort(
        key=lambda c: float(c["mean_primary"]),
        reverse=(direction == "maximize"),
    )

    fingerprint = _hash_payload({
        'task': task, 'metric': metric, 'cv_plan_id': cv_plan.cv_plan_id,
        'candidates': [c['candidate_id'] for c in finalists],
        'config': cfg.to_dict(),
    })
    hpo_id = f"m6_{int(time.time() * 1000)}_{hashlib.sha256(fingerprint.encode()).hexdigest()[:8]}"

    return {
        "hpo_id": hpo_id,
        "hpo_fingerprint": fingerprint,
        "task": task,
        "primary_metric": metric,
        "metric_direction": direction,
        "cv_plan_id": cv_plan.cv_plan_id,
        "cv_plan": cv_plan.to_dict(),
        "source_screening_id": screening_output.get("screening_id"),
        "candidates": finalists,
        "trials": trials,
        "finalists": [
            {
                "candidate_id": c["candidate_id"],
                "algorithm_id": c["algorithm_id"],
                "feature_set_id": c["feature_set_id"],
                "mean_primary": c["mean_primary"],
                "corrected_se": c["corrected_se"],
                "best_parameters": c["best_parameters"],
                "tuned_improvement": c["tuned_improvement"],
                "trials_evaluated": c["trials_evaluated"],
            }
            for c in finalists
        ],
        "guardrails": {
            "max_trials_per_candidate": cfg.max_trials_per_candidate,
            "max_promoted_candidates": cfg.max_promoted_candidates,
            "max_fits": cfg.max_fits,
            "fits_used": fit_count,
            "global_budget_seconds": cfg.global_budget_seconds,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        },
        "m4_feedback": {
            "status": "hpo_evidence_only",
            "feature_sets": [c.get("feature_set_id") for c in finalists],
        },
        "m7_handoff": {
            "status": "ready_for_m65",
            "candidate_count": len(finalists),
            "test_set_used": False,
        },
        "decision_log": decision_log,
        "reproducibility": {
            "random_seed": cfg.random_seed,
            "cv_plan_id": cv_plan.cv_plan_id,
            "config": cfg.to_dict(),
        },
    }
