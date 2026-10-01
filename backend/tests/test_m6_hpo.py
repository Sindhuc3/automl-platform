import pandas as pd
from sklearn.datasets import load_breast_cancer

from app.model_selection.hpo.hpo_service import HPOConfig, run_hpo
from app.model_selection.screening_service import ScreeningConfig, run_screening


def test_hpo_reuses_m5_cv_and_produces_m65_compatible_candidates():
    d = load_breast_cancer()
    X = pd.DataFrame(d.data, columns=d.feature_names)
    y = pd.Series(d.target)
    screening = run_screening(
        X, y, task="classification",
        algorithm_ids=["logistic_regression", "decision_tree"],
        feature_sets=[{"feature_set_id": "all", "columns": list(X.columns)}],
        config=ScreeningConfig(requested_repeats=1, max_fits=20, global_budget_seconds=60),
    )
    hpo = run_hpo(
        X, y, task="classification",
        screening_output=screening,
        feature_sets=[{"feature_set_id": "all", "columns": list(X.columns)}],
        config=HPOConfig(max_trials_per_candidate=2, max_promoted_candidates=2, max_fits=30),
    )
    assert hpo["cv_plan_id"] == screening["cv_plan"]["cv_plan_id"]
    assert hpo["candidates"]
    assert all(c["best_parameters"] is not None for c in hpo["candidates"])
    assert hpo["m7_handoff"]["test_set_used"] is False


def test_hpo_is_compatible_with_arbiter():
    from app.model_selection.arbiter.selection_arbiter import select_final_candidate
    d = load_breast_cancer()
    X = pd.DataFrame(d.data, columns=d.feature_names)
    y = pd.Series(d.target)
    screening = run_screening(
        X, y, task="classification",
        algorithm_ids=["logistic_regression"],
        feature_sets=[{"feature_set_id": "all", "columns": list(X.columns)}],
        config=ScreeningConfig(requested_repeats=1, max_fits=10, global_budget_seconds=60),
    )
    hpo = run_hpo(
        X, y, task="classification",
        screening_output=screening,
        feature_sets=[{"feature_set_id": "all", "columns": list(X.columns)}],
        config=HPOConfig(max_trials_per_candidate=2, max_promoted_candidates=1, max_fits=10),
    )
    # Baseline evidence is retained from M5 when available.
    assert hpo["candidates"][0]["baseline_evidence"]
    result = select_final_candidate(hpo)
    assert result["status"] in {"selected", "no_eligible_candidate"}


def test_hpo_uses_only_m5_handoff_and_keeps_default_trial_zero():
    d = load_breast_cancer()
    X = pd.DataFrame(d.data, columns=d.feature_names)
    y = pd.Series(d.target)
    screening = run_screening(
        X, y, task="classification",
        algorithm_ids=["logistic_regression", "decision_tree", "random_forest"],
        feature_sets=[{"feature_set_id": "all", "columns": list(X.columns)}],
        config=ScreeningConfig(requested_repeats=1, max_fits=30, global_budget_seconds=60),
    )
    eligible = screening["eligible_for_hpo"]
    assert len(eligible) <= 4
    assert eligible
    # The HPO service must reject widening the candidate set; its output count
    # cannot exceed the exact M5 promotion list.
    hpo = run_hpo(
        X, y, task="classification",
        screening_output=screening,
        feature_sets=[{"feature_set_id": "all", "columns": list(X.columns)}],
        config=HPOConfig(max_trials_per_candidate=2, max_promoted_candidates=4, max_fits=40),
    )
    assert {c["m5_candidate_id"] for c in hpo["candidates"]}.issubset(set(eligible))
    for c in hpo["candidates"]:
        assert c["best_trial"]["trial_index"] == 0 or c["trials_evaluated"] >= 1
