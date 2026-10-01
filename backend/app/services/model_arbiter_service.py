from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

from app.database.mongodb import model_hpo_runs_collection, model_selection_runs_collection
from app.model_selection.arbiter.selection_arbiter import ArbiterConfig, select_final_candidate


def _now():
    return datetime.now(timezone.utc)


def _safe(value: Any):
    return json.loads(json.dumps(value, default=str))


def run_model_arbiter(
    dataset_id: str,
    *,
    hpo_id: str | None = None,
    config: Mapping[str, Any] | None = None,
):
    """Run M6.5 selection only; this stage does no model fitting.

    It consumes persisted M6 CV/HPO evidence, applies the deterministic
    selection policy, and writes a single FinalModelSpec for M7.
    The sealed test set is never loaded.
    """
    if hpo_id:
        hpo = model_hpo_runs_collection.find_one(
            {"dataset_id": dataset_id, "hpo_id": hpo_id}, {"_id": 0}
        )
    else:
        hpo = model_hpo_runs_collection.find_one(
            {"dataset_id": dataset_id}, {"_id": 0}, sort=[("created_at", -1)]
        )

    if not hpo:
        raise ValueError("M65_HPO_NOT_FOUND: Run Module 6 first.")
    if hpo.get("status") != "completed":
        raise ValueError("M65_HPO_NOT_COMPLETE: Module 6 must complete successfully.")
    if (hpo.get("m7_handoff") or {}).get("test_set_used") is True:
        raise ValueError("M65_TEST_SET_FORBIDDEN: M6.5 cannot consume test-set evidence.")

    candidates = list(hpo.get("candidates") or [])
    if not candidates:
        raise ValueError("M65_NO_HPO_CANDIDATES: Module 6 returned no candidates.")

    # Keep this stage intentionally cheap: selection is O(number of candidates)
    # and does not refit or cross-validate anything.
    cfg = ArbiterConfig(**dict(config or {}))
    selection = select_final_candidate(
        hpo,
        task=hpo.get("task"),
        config=cfg,
    )

    selection_id = selection.get("selection_id") or f"m65_{uuid.uuid4().hex[:12]}"
    result = {
        "selection_id": selection_id,
        "dataset_id": dataset_id,
        "source_hpo_id": hpo.get("hpo_id"),
        "source_screening_id": hpo.get("source_screening_id"),
        "task": hpo.get("task"),
        **selection,
        "created_at": _now(),
    }
    model_selection_runs_collection.insert_one(_safe(result))
    return _safe(result)


def get_model_arbiter(dataset_id: str, selection_id: str | None = None):
    query = {"dataset_id": dataset_id}
    if selection_id:
        query["selection_id"] = selection_id
        return model_selection_runs_collection.find_one(query, {"_id": 0})
    return list(
        model_selection_runs_collection.find(query, {"_id": 0})
        .sort("created_at", -1)
        .limit(20)
    )
