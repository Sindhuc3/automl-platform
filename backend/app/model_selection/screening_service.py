from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .algorithm_registry import get_algorithm, list_algorithms
from .eligibility import check_eligibility
from .cv.baselines import evaluate_baseline
from .cv.cv_plan import build_cv_plan
from .cv.fold_runner import evaluate_candidate
from .cv.metrics import paired_corrected_se
from .defaults import SEARCH_SPACES


@dataclass(frozen=True)
class ScreeningConfig:
    max_rows: int = 50_000
    max_candidates: int = 100
    max_feature_sets: int = 4
    max_hpo_candidates: int = 4
    global_budget_seconds: float = 600.0
    candidate_budget_seconds: float = 120.0
    max_fits: int = 400
    min_success_fraction: float = 0.80
    random_seed: int = 42
    requested_splits: int = 5
    requested_repeats: int | None = None
    n_jobs: int = 1

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _hash_payload(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _feature_set_id(feature_set: Any, index: int) -> str:
    if isinstance(feature_set, Mapping) and feature_set.get("feature_set_id"):
        return str(feature_set["feature_set_id"])
    return f"feature_set_{index + 1}"


def _feature_set_payload(feature_set: Any, index: int) -> dict[str, Any]:
    fsid = _feature_set_id(feature_set, index)
    if isinstance(feature_set, Mapping):
        return {"feature_set_id": fsid, "columns": feature_set.get("columns"), "description": feature_set.get("description", "")}
    return {"feature_set_id": fsid, "description": "Feature-set transformer recipe"}


def _paired_evidence(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    c_by_key = {(f["repeat"], f["fold"]): f for f in candidate.get("fold_results", []) if f["status"] == "success" and f["primary"] is not None}
    b_by_key = {(f["repeat"], f["fold"]): f for f in baseline.get("fold_results", []) if f["status"] == "success" and f["primary"] is not None}
    keys = sorted(set(c_by_key) & set(b_by_key))
    diffs = [float(c_by_key[k]["primary"] - b_by_key[k]["primary"]) for k in keys]
    if not diffs:
        return {"n_paired": 0, "mean_improvement": None, "corrected_se": None, "beats_baseline": False}
    n_train = int(np.median([c_by_key[k]["n_train"] for k in keys]))
    n_val = int(np.median([c_by_key[k]["n_validation"] for k in keys]))
    se = paired_corrected_se(diffs, n_train=n_train, n_validation=n_val)
    mean = float(np.mean(diffs))
    return {
        "n_paired": len(diffs),
        "mean_improvement": mean,
        "corrected_se": float(se),
        "beats_baseline": bool(mean - se > 0),
    }


def _rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [c for c in candidates if c.get("status") == "success" and c.get("mean_primary") is not None]
    if not valid:
        return []
    task = str(valid[0].get("task", "classification"))
    primary = str(valid[0].get("primary_metric", ""))
    minimize = task == "regression" and primary in {"rmse", "mae"}
    ordered = sorted(valid, key=lambda c: float(c["mean_primary"]), reverse=not minimize)
    best = float(ordered[0]["mean_primary"])
    # Candidate-level corrected SE is a conservative screening tie-band proxy.
    for rank, c in enumerate(ordered, 1):
        c["rank"] = rank
        se = float(c.get("corrected_se", 0.0))
        c["within_one_se_of_best"] = (
            bool(float(c["mean_primary"]) - best <= se)
            if minimize
            else bool(best - float(c["mean_primary"]) <= se)
        )
    return ordered


def run_screening(
    X_train: Any,
    y_train: Any,
    *,
    task: str,
    feature_sets: Sequence[Any] | None = None,
    algorithm_ids: Sequence[str] | None = None,
    dataset_metadata: Mapping[str, Any] | None = None,
    config: ScreeningConfig | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run M5 screening: frozen CV + dummy baseline + algorithm × feature-set candidates."""
    cfg = config or ScreeningConfig()
    X = X_train.copy() if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
    y = pd.Series(y_train).reset_index(drop=True)
    if len(X) != len(y):
        raise ValueError("M5_INPUT_LENGTH_MISMATCH")
    if len(X) > cfg.max_rows:
        raise ValueError(f"M5_SCREENING_ROW_CAP_EXCEEDED: {len(X)} > {cfg.max_rows}")

    feature_sets = list(feature_sets or [{"feature_set_id": "all_features", "columns": list(X.columns)}])
    if not feature_sets or len(feature_sets) > cfg.max_feature_sets:
        raise ValueError(f"M5_FEATURE_SET_LIMIT: expected 1..{cfg.max_feature_sets}")

    requested = list(algorithm_ids) if algorithm_ids is not None else [s.algorithm_id for s in list_algorithms(task)]
    metadata = dict(dataset_metadata or {})
    metadata.setdefault("n_rows", len(X))
    metadata.setdefault("n_features", X.shape[1])
    plan = build_cv_plan(
        X,
        y,
        task=task,
        requested_splits=cfg.requested_splits,
        requested_repeats=cfg.requested_repeats,
        random_seed=cfg.random_seed,
    )
    started = time.perf_counter()
    baseline = evaluate_baseline(X, y, task=task, cv_plan=plan)
    baseline_time = time.perf_counter() - started
    candidates: list[dict[str, Any]] = []
    eligibility: list[dict[str, Any]] = []
    decision_log: list[dict[str, Any]] = []
    fit_count = 0

    for alg_id in requested:
        spec = get_algorithm(alg_id, task)
        if spec is None:
            eligibility.append({"algorithm_id": alg_id, "eligible": False, "reasons": ["Unknown algorithm for task."], "warnings": []})
            continue
        result = check_eligibility(spec, metadata, budget={"high_cost_max_rows": cfg.max_rows, "dense_feature_cap": 5000})
        eligibility.append(result.to_dict())
        if not result.eligible:
            decision_log.append({"event": "algorithm_skipped", "algorithm_id": alg_id, "reasons": list(result.reasons)})
            continue

        for idx, fs in enumerate(feature_sets):
            if time.perf_counter() - started > cfg.global_budget_seconds:
                decision_log.append({"event": "global_budget_reached", "algorithm_id": alg_id})
                break
            if len(candidates) >= cfg.max_candidates:
                decision_log.append({"event": "candidate_budget_reached", "max_candidates": cfg.max_candidates})
                break
            if fit_count + len(plan.folds) > cfg.max_fits:
                decision_log.append({"event": "fit_budget_reached", "algorithm_id": alg_id})
                break
            fsid = _feature_set_id(fs, idx)
            candidate_started = time.perf_counter()
            candidate = evaluate_candidate(
                X, y,
                algorithm_id=alg_id,
                task=task,
                cv_plan=plan,
                context=context,
                feature_set=fs,
                candidate_id=f"{alg_id}:{fsid}:{plan.cv_plan_id}",
            )
            elapsed = time.perf_counter() - candidate_started
            fit_count += len(plan.folds)
            candidate["feature_set_id"] = fsid
            candidate["elapsed_seconds"] = round(elapsed, 6)
            candidate["timeout_warning"] = elapsed > cfg.candidate_budget_seconds
            candidate["baseline_evidence"] = _paired_evidence(candidate, baseline)
            successful = candidate["n_successful_folds"]
            total = candidate["n_successful_folds"] + candidate["n_failed_folds"]
            candidate["guardrail_pass"] = bool(total and successful / total >= cfg.min_success_fraction)
            candidates.append(candidate)
            if elapsed > cfg.candidate_budget_seconds:
                decision_log.append({"event": "candidate_over_budget", "candidate_id": candidate["candidate_id"], "seconds": elapsed})

    # Add corrected uncertainty per candidate and rank only as evidence; M6/Arbiter
    # remains responsible for promotion/final selection.
    for c in candidates:
        scores = [f["primary"] for f in c["fold_results"] if f["status"] == "success" and f["primary"] is not None]
        if len(scores) > 1:
            n_train = int(np.median([f["n_train"] for f in c["fold_results"] if f["status"] == "success"]))
            n_val = int(np.median([f["n_validation"] for f in c["fold_results"] if f["status"] == "success"]))
            from .cv.metrics import corrected_se
            c["corrected_se"] = float(corrected_se(scores, n_train=n_train, n_validation=n_val))
        else:
            c["corrected_se"] = 0.0
    ranked = _rank_candidates(candidates)

    # M5 produces evidence; it does not choose the final model. However, the
    # M6 contract deliberately limits how many screening candidates may enter
    # HPO. Keep only candidates that (a) passed the fold-success guardrail,
    # (b) beat the dummy baseline with paired corrected-SE evidence, and
    # (c) are inside the M5 one-SE tie band. Rank remains the deterministic
    # ordering; M6 will make the actual tuning/final-selection decision.
    promotion_pool = [
        c for c in ranked
        if c.get("guardrail_pass")
        and c.get("baseline_evidence", {}).get("beats_baseline")
        and c.get("within_one_se_of_best", False)
    ]
    eligible_for_hpo = promotion_pool[: cfg.max_hpo_candidates]
    tie_band = [
        {
            "candidate_id": c["candidate_id"],
            "algorithm_id": c["algorithm_id"],
            "feature_set_id": c["feature_set_id"],
            "mean_primary": c["mean_primary"],
            "corrected_se": c.get("corrected_se"),
            "rank": c.get("rank"),
        }
        for c in promotion_pool
    ]
    promoted_ids = {c["candidate_id"] for c in eligible_for_hpo}
    if promotion_pool:
        for c in ranked:
            if c["candidate_id"] in promoted_ids:
                c["eligible_for_hpo"] = True
            else:
                c["eligible_for_hpo"] = False
    no_candidate_beats_baseline = not any(
        c.get("guardrail_pass") and c.get("baseline_evidence", {}).get("beats_baseline")
        for c in candidates
    )

    feature_evidence: dict[str, Any] = {}
    for alg_id in sorted({c["algorithm_id"] for c in candidates}):
        group = [c for c in candidates if c["algorithm_id"] == alg_id and c.get("mean_primary") is not None]
        minimize = task == "regression" and str(group[0].get("primary_metric")) in {"rmse", "mae"}
        group = sorted(group, key=lambda c: c["mean_primary"], reverse=not minimize)
        if group:
            ref = group[0]
            feature_evidence[alg_id] = {
                "best_feature_set_id": ref["feature_set_id"],
                "feature_set_scores": [
                    {"feature_set_id": c["feature_set_id"], "mean_primary": c["mean_primary"], "rank_within_algorithm": i + 1}
                    for i, c in enumerate(group)
                ],
            }

    screening_id = f"m5_{_hash_payload({'task': task, 'n': len(X), 'features': list(X.columns), 'plan': plan.cv_plan_id})}"
    return {
        "screening_id": screening_id,
        "task": task,
        "cv_plan": plan.to_dict(),
        "metric": {"primary_metric": baseline["primary_metric"]},
        "baseline": baseline,
        "candidates": candidates,
        "ranking": [
            {
                "rank": c.get("rank"),
                "candidate_id": c["candidate_id"],
                "algorithm_id": c["algorithm_id"],
                "feature_set_id": c["feature_set_id"],
                "mean_primary": c["mean_primary"],
                "corrected_se": c.get("corrected_se"),
                "within_one_se_of_best": c.get("within_one_se_of_best", False),
                "beats_baseline": c["baseline_evidence"]["beats_baseline"],
            }
            for c in ranked
        ],
        "eligible_algorithms": eligibility,
        "feature_set_evidence": feature_evidence,
        "tie_band": tie_band,
        "eligible_for_hpo": [c["candidate_id"] for c in eligible_for_hpo],
        "guardrails": {
            "row_cap": cfg.max_rows,
            "candidate_budget_seconds": cfg.candidate_budget_seconds,
            "global_budget_seconds": cfg.global_budget_seconds,
            "max_fits": cfg.max_fits,
            "max_hpo_candidates": cfg.max_hpo_candidates,
            "fits_used": fit_count,
            "baseline_seconds": round(baseline_time, 6),
        },
        "decision_log": decision_log + ([{
            "event": "m6_hpo_promotion_cap",
            "max_hpo_candidates": cfg.max_hpo_candidates,
            "eligible_candidate_ids": [c["candidate_id"] for c in eligible_for_hpo],
            "reason": "guardrail + beats-baseline + one-SE tie-band; M6 retains final decision authority",
        }] if promotion_pool else [{
            "event": "no_candidate_beats_baseline",
            "reason": "No guardrail-passing candidate beat the dummy baseline with paired corrected-SE evidence.",
        }]),
        "m4_feedback": {"status": "evidence_only", "candidate_feature_sets_evaluated": [_feature_set_payload(fs, i) for i, fs in enumerate(feature_sets)]},
        "m6_handoff": {
            "status": "ready_for_promotion" if eligible_for_hpo else "no_eligible_candidates",
            "primary_metric": baseline["primary_metric"],
            "cv_plan_ref": plan.cv_plan_id,
            "eligible_for_hpo": [c["candidate_id"] for c in eligible_for_hpo],
            "candidates": [c["candidate_id"] for c in eligible_for_hpo],
            "tie_band": [c["candidate_id"] for c in promotion_pool],
            "no_candidate_beats_baseline": no_candidate_beats_baseline,
            "hpo_space_ids": {c["algorithm_id"]: c["algorithm_id"] for c in eligible_for_hpo if c["algorithm_id"] in SEARCH_SPACES},
        },
        "reproducibility": {"random_seed": cfg.random_seed, "cv_plan_id": plan.cv_plan_id},
    }
