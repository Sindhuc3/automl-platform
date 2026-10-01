from __future__ import annotations

import json
import hashlib
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from app.database.mongodb import (
    datasets_collection,
    preprocessing_runs_collection,
    feature_engineering_runs_collection,
    model_screening_runs_collection,
)
from app.model_selection.algorithm_registry import get_algorithm, list_algorithms
from app.model_selection.screening_service import ScreeningConfig, run_screening


def _now():
    return datetime.now(timezone.utc)


def _safe(value: Any):
    return json.loads(json.dumps(value, default=str))


def _latest(collection, dataset_id: str, run_id: str | None, label: str):
    query = {"dataset_id": dataset_id}
    if run_id:
        query["run_id"] = run_id
        doc = collection.find_one(query, {"_id": 0})
    else:
        docs = list(collection.find(query, {"_id": 0}).sort("created_at", -1).limit(1))
        doc = docs[0] if docs else None
    if not doc:
        raise ValueError(f"M5_{label}_NOT_FOUND: Run Module {label} first.")
    return doc


def _load_training_frame(
    dataset_id: str,
    preprocessing_run_id: str | None,
    feature_engineering_run_id: str | None,
):
    dataset = datasets_collection.find_one({"dataset_id": dataset_id}, {"_id": 0})
    if not dataset:
        raise ValueError("M5_DATASET_NOT_FOUND: Dataset not found.")

    pre = _latest(
        preprocessing_runs_collection,
        dataset_id,
        preprocessing_run_id,
        "PREPROCESSING",
    )
    if pre.get("status") != "completed":
        raise ValueError("M5_PREPROCESSING_NOT_COMPLETE: Module 3 must complete successfully.")

    fe = _latest(
        feature_engineering_runs_collection,
        dataset_id,
        feature_engineering_run_id,
        "FEATURE_ENGINEERING",
    )
    if fe.get("status") != "completed":
        raise ValueError("M5_FEATURE_ENGINEERING_NOT_COMPLETE: Module 4 must complete successfully.")

    pre_paths = pre.get("artifact_paths") or {}
    plain_path = Path(pre_paths.get("processed_train_plain", ""))
    scaled_path = Path(pre_paths.get("processed_train_scaled", ""))
    if not plain_path.exists() or not scaled_path.exists():
        raise ValueError("M5_PROCESSED_TRAIN_MISSING: Module 3 processed training artifacts are missing.")

    plain = pd.read_csv(plain_path)
    scaled = pd.read_csv(scaled_path)
    target_col = "__target__"
    if target_col not in plain.columns or target_col not in scaled.columns:
        raise ValueError("M5_TARGET_ARTIFACT_MISSING: Processed training artifacts do not contain __target__.")
    if len(plain) != len(scaled):
        raise ValueError("M5_PROFILE_ROW_MISMATCH: Plain and scaled training artifacts have different row counts.")

    # M4's training semantic frame is kept in the same training-row order as M3.
    fe_paths = fe.get("artifact_paths") or {}
    engineered_path = Path(fe_paths.get("engineered_training_features", ""))
    if not engineered_path.exists():
        raise ValueError("M5_ENGINEERED_ARTIFACT_MISSING: Module 4 engineered training artifact is missing.")
    engineered = pd.read_csv(engineered_path)
    if len(engineered) != len(plain):
        raise ValueError(
            f"M5_TRAINING_ROW_MISMATCH: Module 3 has {len(plain)} rows but Module 4 has {len(engineered)}."
        )

    report = pre.get("report") or {}
    mappings = report.get("feature_mapping") or {}
    fe_report = fe.get("report") or {}
    feature_sets = fe_report.get("feature_sets") or []
    generated_records = (fe_report.get("engineering") or {}).get("created") or []
    generated_names = [
        str(item.get("name"))
        for item in generated_records
        if item.get("name") in engineered.columns
    ]

    # Add only Module 4-generated columns to each M3 profile. Source columns are
    # already represented by M3's fitted transformations.
    generated_values = engineered.loc[:, generated_names].copy() if generated_names else pd.DataFrame(index=plain.index)
    for frame in (plain, scaled):
        for col in generated_names:
            frame[col] = pd.to_numeric(generated_values[col], errors="coerce")

    return dataset, pre, fe, plain, scaled, mappings, feature_sets, generated_names


def _source_to_transformed(mapping: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for item in mapping:
        source = item.get("source_column")
        name = item.get("feature_name")
        if source and name:
            out.setdefault(str(source), []).append(str(name))
    return out


def _translate_feature_set(
    feature_set: Mapping[str, Any],
    *,
    task: str,
    source_map: Mapping[str, list[str]],
    generated_names: list[str],
    all_columns: list[str],
    algorithm_scaling: str,
) -> dict[str, Any]:
    source_features = list(feature_set.get("features") or [])
    selected: list[str] = []
    unresolved: list[str] = []

    for source in source_features:
        if source in generated_names:
            selected.append(source)
            continue
        mapped = source_map.get(str(source), [])
        if mapped:
            selected.extend(mapped)
        elif source in all_columns:
            selected.append(str(source))
        else:
            unresolved.append(str(source))

    # A selector recipe is evaluated in-fold. For Compact, all Engineered
    # candidate columns form the input to the target-dependent selector.
    selection = feature_set.get("selection")
    recipe: dict[str, Any] = {}
    if isinstance(selection, Mapping) and selection.get("enabled"):
        recipe = {
            "selector": "compact",
            "task": task,
            "correlation_threshold": float(
                ((selection.get("pipeline_spec") or [{}])[0].get("params") or {}).get("threshold", 0.98)
            ),
            "n_shuffles": int(
                ((selection.get("pipeline_spec") or [{}, {}])[1].get("params") or {}).get("n_shuffles", 5)
            ),
            "null_quantile": float(
                ((selection.get("pipeline_spec") or [{}, {}])[1].get("params") or {}).get("quantile", 0.95)
            ),
            "min_features": int(
                ((selection.get("pipeline_spec") or [{}, {}])[1].get("params") or {}).get("min_features", 5)
            ),
            "random_state": 42,
        }
        selected = [c for c in all_columns if c != "__target__"]

    if unresolved:
        raise ValueError(
            f"M5_FEATURE_SET_TRANSLATION_FAILED: Could not map M4 source features {unresolved} to Module 3 transformed features."
        )

    selected = list(dict.fromkeys(c for c in selected if c in all_columns and c != "__target__"))
    if not selected:
        raise ValueError(f"M5_EMPTY_FEATURE_SET: '{feature_set.get('name', feature_set.get('set_id'))}' maps to no transformed features.")

    scale_columns = [c for c in generated_names if c in selected] if algorithm_scaling == "standard" else []
    recipe.update(
        {
            "feature_set_id": str(feature_set.get("set_id") or feature_set.get("feature_set_id") or "feature_set"),
            "columns": selected,
            "scale_columns": scale_columns,
            "description": feature_set.get("description", ""),
            "m4_source_features": source_features,
            "m4_recipe": selection,
        }
    )
    return recipe


def run_model_screening(
    dataset_id: str,
    *,
    preprocessing_run_id: str | None = None,
    feature_engineering_run_id: str | None = None,
    algorithm_ids: list[str] | None = None,
    config: Mapping[str, Any] | None = None,
):
    dataset, pre, fe, plain, scaled, mappings, feature_sets, generated_names = _load_training_frame(
        dataset_id, preprocessing_run_id, feature_engineering_run_id
    )

    task = str((pre.get("inputs") or {}).get("problem_type", "")).lower()
    if task not in {"classification", "regression"}:
        raise ValueError("M5_PROBLEM_TYPE_MISSING: Module 2/3 did not provide a supported problem type.")

    cfg = ScreeningConfig(**dict(config or {}))
    requested_algorithms = list(algorithm_ids or [s.algorithm_id for s in list_algorithms(task)])

    # Build one frozen fold plan over row positions. Each algorithm can use a
    # different M3 profile, but all see identical CV indices.
    # run_screening itself materializes the plan from the supplied frame.
    # We run one profile at a time only when the profile changes, then merge
    # evidence under one screening manifest.
    profile_groups: dict[str, list[str]] = {"plain": [], "scaled": []}
    for alg_id in requested_algorithms:
        spec = get_algorithm(alg_id, task)
        if spec is None:
            continue
        profile_groups["scaled" if spec.scaling_profile == "standard" else "plain"].append(alg_id)

    start = time.perf_counter()
    combined_candidates = []
    combined_eligibility = []
    combined_logs = []
    profile_outputs = {}
    baseline_reference = None
    common_feature_set_records = []

    for profile, algs in profile_groups.items():
        if not algs:
            continue
        frame = scaled if profile == "scaled" else plain
        X = frame.drop(columns=["__target__"])
        y = frame["__target__"]

        mapping = mappings.get(profile) or []
        source_map = _source_to_transformed(mapping)
        translated_sets = []
        for fs in feature_sets:
            translated = _translate_feature_set(
                fs,
                task=task,
                source_map=source_map,
                generated_names=generated_names,
                all_columns=list(X.columns),
                algorithm_scaling="standard" if profile == "scaled" else "none",
            )
            translated_sets.append(translated)

        output = run_screening(
            X,
            y,
            task=task,
            algorithm_ids=algs,
            feature_sets=translated_sets,
            dataset_metadata={
                "n_rows": len(X),
                "n_features": X.shape[1],
                "dataset_id": dataset_id,
                "profile": profile,
            },
            config=cfg,
            context={"profile": profile, "scaling_profile": "none"},
        )
        profile_outputs[profile] = output
        if baseline_reference is None:
            baseline_reference = output["baseline"]
        else:
            # Baseline is expected to be deterministic across profiles. Keep
            # the first as the canonical evidence and record the second.
            pass
        combined_candidates.extend(output["candidates"])
        combined_eligibility.extend(output["eligible_algorithms"])
        combined_logs.extend(output["decision_log"])
        common_feature_set_records.extend(translated_sets)

    if not combined_candidates:
        raise ValueError("M5_NO_CANDIDATES_COMPLETED: No requested algorithm passed the screening gates.")

    # Recompute a global evidence ranking from the merged candidate evidence.
    primary_metric = profile_outputs[next(iter(profile_outputs))]["metric"]["primary_metric"]
    minimize = task == "regression" and primary_metric in {"rmse", "mae"}
    ranked = sorted(
        [c for c in combined_candidates if c.get("status") == "success" and c.get("mean_primary") is not None],
        key=lambda c: float(c["mean_primary"]),
        reverse=not minimize,
    )
    if ranked:
        best_score = float(ranked[0]["mean_primary"])
    else:
        best_score = None
    for rank, candidate in enumerate(ranked, 1):
        candidate["rank"] = rank
        score = float(candidate["mean_primary"])
        se = float(candidate.get("corrected_se") or 0.0)
        candidate["within_one_se_of_best"] = (
            bool(score - best_score <= se)
            if minimize and best_score is not None
            else bool(best_score - score <= se) if best_score is not None else False
        )

    # Global M5 -> M6 promotion must happen after plain/scaled profile outputs
    # are merged. Otherwise each profile can independently promote candidates
    # and the final service response can accidentally hand every screened
    # candidate to M6. M5 only forwards a bounded evidence-based shortlist;
    # M6 remains responsible for HPO and the final decision.
    promotion_pool = [
        c for c in ranked
        if c.get("guardrail_pass")
        and (c.get("baseline_evidence") or {}).get("beats_baseline")
        and c.get("within_one_se_of_best", False)
    ]
    eligible_for_hpo = promotion_pool[: cfg.max_hpo_candidates]
    promoted_ids = {c["candidate_id"] for c in eligible_for_hpo}
    for candidate in ranked:
        candidate["eligible_for_hpo"] = candidate["candidate_id"] in promoted_ids

    # A screening execution is a persisted run, so its MongoDB id must be
    # unique even when the exact same dataset/config is screened again. Keep
    # the deterministic content hash separately for reproducibility/caching.
    screening_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "preprocessing_run_id": pre.get("run_id"),
                "feature_engineering_run_id": fe.get("run_id"),
                "algorithms": requested_algorithms,
                "seed": cfg.random_seed,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    screening_id = f"m5_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

    result = {
        "screening_id": screening_id,
        "screening_fingerprint": screening_fingerprint,
        "dataset_id": dataset_id,
        "task": task,
        "status": "completed",
        "primary_metric": primary_metric,
        "cv_plan_ids": sorted({c.get("cv_plan_id") for c in combined_candidates}),
        "baseline": baseline_reference,
        "candidates": combined_candidates,
        "ranking": [
            {
                "rank": c.get("rank"),
                "candidate_id": c["candidate_id"],
                "algorithm_id": c["algorithm_id"],
                "feature_set_id": c.get("feature_set_id"),
                "mean_primary": c.get("mean_primary"),
                "corrected_se": c.get("corrected_se"),
                "beats_baseline": (c.get("baseline_evidence") or {}).get("beats_baseline"),
                "profile": next(
                    (p for p, out in profile_outputs.items() if any(x is c for x in out.get("candidates", []))),
                    None,
                ),
            }
            for c in ranked
        ],
        "eligible_algorithms": combined_eligibility,
        "feature_sets": common_feature_set_records,
        "profile_outputs": profile_outputs,
        "guardrails": {
            "config": cfg.to_dict(),
            "elapsed_seconds": round(time.perf_counter() - start, 6),
            "test_set_used": False,
        },
        "decision_log": combined_logs,
        "m4_feedback": {
            "status": "evidence_only",
            "feature_sets_evaluated": [fs.get("feature_set_id") for fs in common_feature_set_records],
            "feature_selection_execution": "target-dependent Compact selection, when present, is fitted inside each CV training fold",
        },
        "tie_band": [
            {
                "candidate_id": c["candidate_id"],
                "algorithm_id": c["algorithm_id"],
                "feature_set_id": c.get("feature_set_id"),
                "mean_primary": c.get("mean_primary"),
                "corrected_se": c.get("corrected_se"),
                "rank": c.get("rank"),
            }
            for c in promotion_pool
        ],
        "eligible_for_hpo": [c["candidate_id"] for c in eligible_for_hpo],
        "m6_handoff": {
            "status": "ready_for_promotion" if eligible_for_hpo else "no_eligible_candidates",
            "primary_metric": primary_metric,
            "cv_plan_ref": sorted({c.get("cv_plan_id") for c in combined_candidates}),
            "candidate_ids": [c["candidate_id"] for c in eligible_for_hpo],
            "eligible_for_hpo": [c["candidate_id"] for c in eligible_for_hpo],
            "tie_band": [c["candidate_id"] for c in promotion_pool],
            "no_candidate_beats_baseline": not any(
                c.get("guardrail_pass") and (c.get("baseline_evidence") or {}).get("beats_baseline")
                for c in combined_candidates
            ),
        },
        "reproducibility": {
            "dataset_id": dataset_id,
            "preprocessing_run_id": pre.get("run_id"),
            "feature_engineering_run_id": fe.get("run_id"),
            "random_seed": cfg.random_seed,
        },
        "created_at": _now(),
    }

    model_screening_runs_collection.insert_one(_safe(result))
    return _safe(result)


def get_model_screening(dataset_id: str, screening_id: str | None = None):
    query = {"dataset_id": dataset_id}
    if screening_id:
        query["screening_id"] = screening_id
        return model_screening_runs_collection.find_one(query, {"_id": 0})
    return list(
        model_screening_runs_collection.find(query, {"_id": 0})
        .sort("created_at", -1)
        .limit(20)
    )
