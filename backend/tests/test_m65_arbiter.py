from app.model_selection.arbiter.selection_arbiter import ArbiterConfig, select_final_candidate


def _candidate(cid, alg, score, se, *, complexity=None, features=10, beats=True, improvement=0.0, guardrail=True):
    return {
        "candidate_id": cid,
        "algorithm_id": alg,
        "feature_set_id": "fs1",
        "mean_primary": score,
        "corrected_se": se,
        "n_features": features,
        "guardrail_pass": guardrail,
        "baseline_evidence": {"beats_baseline": beats},
        "tuned_improvement": improvement,
        "primary_metric": "roc_auc",
    }


def test_arbiter_selects_simpler_model_inside_one_se_zone():
    result = select_final_candidate({
        "hpo_id": "h1",
        "primary_metric": "roc_auc",
        "candidates": [
            _candidate("rf", "random_forest", .992, .003, features=20, improvement=.010),
            _candidate("lr", "logistic_regression", .9905, .003, features=10, improvement=.008),
        ],
    })
    assert result["status"] == "selected"
    assert result["selected"]["algorithm_id"] == "logistic_regression"
    assert result["m7_handoff"]["test_set_used"] is False


def test_arbiter_selects_clear_best_score():
    result = select_final_candidate({
        "primary_metric": "roc_auc",
        "candidates": [
            _candidate("lr", "logistic_regression", .90, .01, improvement=.01),
            _candidate("rf", "random_forest", .96, .01, features=20, improvement=.02),
        ],
    })
    assert result["selected"]["algorithm_id"] == "random_forest"


def test_regression_uses_minimization():
    candidates = [
        {"candidate_id": "a", "algorithm_id": "ridge", "feature_set_id": "fs1", "mean_primary": 1.5, "corrected_se": .05, "n_features": 10, "guardrail_pass": True, "baseline_evidence": {"beats_baseline": True}, "primary_metric": "rmse"},
        {"candidate_id": "b", "algorithm_id": "random_forest", "feature_set_id": "fs1", "mean_primary": 1.2, "corrected_se": .05, "n_features": 15, "guardrail_pass": True, "baseline_evidence": {"beats_baseline": True}, "primary_metric": "rmse"},
    ]
    result = select_final_candidate({"primary_metric": "rmse", "candidates": candidates})
    assert result["selected"]["algorithm_id"] == "random_forest"
    assert result["metric_direction"] == "minimize"


def test_failed_guardrails_are_rejected():
    result = select_final_candidate({
        "primary_metric": "roc_auc",
        "candidates": [_candidate("x", "random_forest", .99, .01, guardrail=False)],
    })
    assert result["status"] == "no_eligible_candidate"
    assert result["rejected"][0]["reason"] == "guardrail_failed"
