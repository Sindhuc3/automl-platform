import pandas as pd
from sklearn.datasets import load_breast_cancer

from app.model_selection.screening_service import ScreeningConfig, run_screening


def test_screening_runs_baseline_and_candidates():
    d = load_breast_cancer()
    X = pd.DataFrame(d.data, columns=d.feature_names)
    y = pd.Series(d.target)
    result = run_screening(
        X, y,
        task="classification",
        algorithm_ids=["logistic_regression", "decision_tree"],
        feature_sets=[{"feature_set_id": "all", "columns": list(X.columns)}],
        config=ScreeningConfig(requested_repeats=1, max_fits=20, global_budget_seconds=60),
    )
    assert result["baseline"]["status"] == "success"
    assert len(result["candidates"]) == 2
    assert result["ranking"]
    assert all("beats_baseline" in row for row in result["ranking"])


def test_static_feature_set_is_used_per_fold():
    d = load_breast_cancer()
    X = pd.DataFrame(d.data, columns=d.feature_names)
    y = pd.Series(d.target)
    result = run_screening(
        X, y,
        task="classification",
        algorithm_ids=["logistic_regression"],
        feature_sets=[{"feature_set_id": "small", "columns": list(X.columns[:3])}],
        config=ScreeningConfig(requested_repeats=1, max_fits=20, global_budget_seconds=60),
    )
    candidate = result["candidates"][0]
    assert candidate["status"] == "success"
    assert all(f["n_features_out"] == 3 for f in candidate["fold_results"] if f["status"] == "success")


def test_m6_handoff_is_capped_at_four_and_uses_promotion_guardrails():
    d = load_breast_cancer()
    X = pd.DataFrame(d.data, columns=d.feature_names)
    y = pd.Series(d.target)
    result = run_screening(
        X, y,
        task="classification",
        algorithm_ids=["logistic_regression", "decision_tree", "random_forest", "extra_trees"],
        feature_sets=[
            {"feature_set_id": "A", "columns": list(X.columns)},
            {"feature_set_id": "B", "columns": list(X.columns[:10])},
        ],
        config=ScreeningConfig(requested_repeats=1, max_fits=40, global_budget_seconds=60),
    )
    eligible = result["eligible_for_hpo"]
    handoff = result["m6_handoff"]["eligible_for_hpo"]
    assert len(eligible) <= 4
    assert handoff == eligible
    assert len(result["tie_band"]) >= len(eligible)
    for candidate_id in eligible:
        candidate = next(c for c in result["candidates"] if c["candidate_id"] == candidate_id)
        assert candidate["guardrail_pass"] is True
        assert candidate["baseline_evidence"]["beats_baseline"] is True
        assert candidate["within_one_se_of_best"] is True
