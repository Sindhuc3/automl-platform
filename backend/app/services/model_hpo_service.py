from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd

from app.database.mongodb import model_hpo_runs_collection, model_screening_runs_collection
from app.model_selection.hpo.hpo_service import HPOConfig, run_hpo
from app.services.model_screening_service import _load_training_frame


def _now():
    return datetime.now(timezone.utc)


def _safe(value: Any):
    return json.loads(json.dumps(value, default=str))


def _profile_for_candidate(candidate: Mapping[str, Any], screening: Mapping[str, Any]) -> str:
    cid = str(candidate.get("candidate_id"))
    for profile, output in (screening.get("profile_outputs") or {}).items():
        for c in output.get("candidates", []):
            if str(c.get("candidate_id")) == cid:
                return str(profile)
    return "plain"


def _build_cv_plan_from_profile(profile_output: Mapping[str, Any]):
    # M5 now exposes the complete materialized plan in each profile output.
    # run_hpo accepts this serialized form and reconstructs the exact folds.
    return profile_output.get("cv_plan")


def run_model_hpo(
    dataset_id: str,
    *,
    screening_id: str | None = None,
    config: Mapping[str, Any] | None = None,
):
    screening = None
    if screening_id:
        screening = model_screening_runs_collection.find_one(
            {"dataset_id": dataset_id, "screening_id": screening_id}, {"_id": 0}
        )
    else:
        docs = list(
            model_screening_runs_collection.find(
                {"dataset_id": dataset_id}, {"_id": 0}
            ).sort("created_at", -1).limit(1)
        )
        screening = docs[0] if docs else None
    if not screening:
        raise ValueError("M6_SCREENING_NOT_FOUND: Run Module 5 first.")
    if screening.get("status") != "completed":
        raise ValueError("M6_SCREENING_NOT_COMPLETE: Module 5 must complete successfully.")

    eligible_ids = list(
        screening.get("eligible_for_hpo")
        or (screening.get("m6_handoff") or {}).get("eligible_for_hpo")
        or []
    )
    if not eligible_ids:
        raise ValueError("M6_NO_ELIGIBLE_CANDIDATES: Module 5 did not promote candidates to HPO.")
    if len(eligible_ids) > 4:
        raise ValueError("M6_PROMOTION_CAP_EXCEEDED: Module 5 handed off more than 4 candidates.")

    repro = screening.get("reproducibility") or {}
    dataset, pre, fe, plain, scaled, mappings, feature_sets, generated_names = _load_training_frame(
        dataset_id, repro.get("preprocessing_run_id"), repro.get("feature_engineering_run_id")
    )
    task = str((pre.get("inputs") or {}).get("problem_type", "")).lower()
    if task not in {"classification", "regression"}:
        raise ValueError("M6_PROBLEM_TYPE_MISSING: Module 2/3 did not provide a supported problem type.")

    candidates_by_id = {
        str(c.get("candidate_id")): c for c in screening.get("candidates", [])
    }
    selected = [candidates_by_id[cid] for cid in eligible_ids if cid in candidates_by_id]
    if len(selected) != len(eligible_ids):
        missing = [cid for cid in eligible_ids if cid not in candidates_by_id]
        raise ValueError(f"M6_HANDOFF_CANDIDATES_MISSING: {missing}")

    cfg = HPOConfig(**dict(config or {}))
    if cfg.max_promoted_candidates > 4:
        cfg = HPOConfig(**{**cfg.to_dict(), "max_promoted_candidates": 4})

    started = time.perf_counter()
    profile_results: dict[str, Any] = {}
    all_candidates: list[dict[str, Any]] = []
    all_trials: list[dict[str, Any]] = []
    all_logs: list[dict[str, Any]] = []

    # HPO is grouped by M3 profile so a scaled algorithm receives exactly the
    # same representation it saw during M5. Each group gets the exact M5
    # materialized CV plan from its profile output.
    groups: dict[str, list[dict[str, Any]]] = {}
    for candidate in selected:
        groups.setdefault(_profile_for_candidate(candidate, screening), []).append(candidate)

    for profile, group in groups.items():
        elapsed = time.perf_counter() - started
        remaining_global = cfg.global_budget_seconds - elapsed
        if remaining_global <= 0:
            all_logs.append({"event": "global_budget_reached", "profile": profile})
            break
        group_cfg = HPOConfig(**{**cfg.to_dict(), "global_budget_seconds": remaining_global})
        frame = scaled if profile == "scaled" else plain
        X = frame.drop(columns=["__target__"])
        y = frame["__target__"]
        profile_output = (screening.get("profile_outputs") or {}).get(profile) or {}
        serialized_plan = _build_cv_plan_from_profile(profile_output)
        if not serialized_plan:
            raise ValueError(f"M6_CV_PLAN_MISSING: M5 profile '{profile}' did not persist its frozen plan.")

        group_ids = {str(c["candidate_id"]) for c in group}
        group_screening = dict(screening)
        group_screening["candidates"] = [c for c in screening.get("candidates", []) if str(c.get("candidate_id")) in group_ids]
        group_screening["eligible_for_hpo"] = [c["candidate_id"] for c in group]
        group_screening["m6_handoff"] = {"eligible_for_hpo": group_screening["eligible_for_hpo"]}
        group_screening["cv_plan"] = serialized_plan
        group_screening["primary_metric"] = screening.get("primary_metric") or (screening.get("metric") or {}).get("primary_metric")

        # Use the translated M5 feature recipes for this profile. They are
        # target-independent recipes; any supervised selector is fitted inside
        # run_hpo's CV fold, not globally.
        group_feature_sets = []
        seen_fs: set[str] = set()
        for fs in screening.get("feature_sets", []):
            fsid = str(fs.get("feature_set_id"))
            # The integrated M5 service emits one translated recipe per M3
            # profile. Prefer recipes whose scaling metadata matches this
            # profile; this matters when M4-generated numeric features exist.
            scale_columns = list(fs.get("scale_columns") or [])
            matches_profile = (profile == "scaled" and bool(scale_columns)) or (profile == "plain" and not scale_columns)
            if matches_profile and fsid not in seen_fs:
                group_feature_sets.append(fs)
                seen_fs.add(fsid)
        if not group_feature_sets:
            for fs in screening.get("feature_sets", []):
                fsid = str(fs.get("feature_set_id"))
                if fsid not in seen_fs:
                    group_feature_sets.append(fs)
                    seen_fs.add(fsid)

        result = run_hpo(
            X,
            y,
            task=task,
            screening_output=group_screening,
            feature_sets=group_feature_sets,
            cv_plan=None,
            config=group_cfg,
            context={"profile": profile, "scaling_profile": "none"},
        )
        profile_results[profile] = result
        all_candidates.extend(result.get("candidates", []))
        all_trials.extend(result.get("trials", []))
        all_logs.extend(result.get("decision_log", []))

    if not all_candidates:
        raise ValueError("M6_NO_HPO_RESULTS: No promoted candidate completed HPO.")

    direction = profile_results[next(iter(profile_results))]["metric_direction"]
    all_candidates.sort(
        key=lambda c: float(c["mean_primary"]), reverse=(direction == "maximize")
    )

    fingerprint = f"m6f_{uuid.uuid4().hex[:16]}"
    hpo_id = f"m6_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    result = {
        "hpo_id": hpo_id,
        "hpo_fingerprint": fingerprint,
        "dataset_id": dataset_id,
        "source_screening_id": screening.get("screening_id"),
        "task": task,
        "status": "completed",
        "primary_metric": profile_results[next(iter(profile_results))]["primary_metric"],
        "metric_direction": direction,
        "cv_plan_ids": sorted({r["cv_plan_id"] for r in profile_results.values()}),
        "candidates": all_candidates,
        "trials": all_trials,
        "finalists": [
            {
                "candidate_id": c["candidate_id"],
                "algorithm_id": c["algorithm_id"],
                "feature_set_id": c.get("feature_set_id"),
                "mean_primary": c.get("mean_primary"),
                "corrected_se": c.get("corrected_se"),
                "best_parameters": c.get("best_parameters"),
                "tuned_improvement": c.get("tuned_improvement"),
                "trials_evaluated": c.get("trials_evaluated"),
            }
            for c in all_candidates
        ],
        "profile_results": profile_results,
        "guardrails": {
            "max_trials_per_candidate": cfg.max_trials_per_candidate,
            "max_promoted_candidates": 4,
            "max_fits": cfg.max_fits,
            "fits_used": sum(
                len(r.get("trials", [])) * len(r.get("cv_plan", {}).get("folds", []))
                for r in profile_results.values()
            ),
            "global_budget_seconds": cfg.global_budget_seconds,
            "candidate_budget_seconds": cfg.candidate_budget_seconds,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "test_set_used": False,
        },
        "m4_feedback": {"status": "hpo_evidence_only"},
        "m7_handoff": {
            "status": "ready_for_m65",
            "candidate_count": len(all_candidates),
            "test_set_used": False,
            "source_screening_id": screening.get("screening_id"),
        },
        "decision_log": all_logs,
        "reproducibility": {
            "random_seed": cfg.random_seed,
            "source_screening_id": screening.get("screening_id"),
            "cv_plan_ids": sorted({r["cv_plan_id"] for r in profile_results.values()}),
            "config": cfg.to_dict(),
        },
        "created_at": _now(),
    }
    model_hpo_runs_collection.insert_one(_safe(result))
    return _safe(result)


def get_model_hpo(dataset_id: str, hpo_id: str | None = None):
    query = {"dataset_id": dataset_id}
    if hpo_id:
        query["hpo_id"] = hpo_id
        return model_hpo_runs_collection.find_one(query, {"_id": 0})
    return list(
        model_hpo_runs_collection.find(query, {"_id": 0})
        .sort("created_at", -1)
        .limit(20)
    )
