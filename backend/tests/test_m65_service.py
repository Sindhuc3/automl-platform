from app.model_selection.arbiter.selection_arbiter import select_final_candidate


def candidate(cid, alg, score, se, features, *, beats=True):
    return {
        "candidate_id": cid,
        "algorithm_id": alg,
        "feature_set_id": "A",
        "mean_primary": score,
        "corrected_se": se,
        "n_features": features,
        "guardrail_pass": True,
        "baseline_evidence": {"beats_baseline": beats},
        "tuned_improvement": 0.01,
        "best_parameters": {"random_state": 42},
        "primary_metric": "roc_auc",
    }


def test_titanic_style_m65_selection_is_fast_policy_only():
    hpo = {
        "hpo_id": "m6-test",
        "task": "classification",
        "primary_metric": "roc_auc",
        "candidates": [
            candidate("gba", "gradient_boosting", .8932173, .0144397, 19),
            candidate("gbb", "gradient_boosting", .8905389, .0143366, 20),
            candidate("cba", "catboost", .8891818, .0169235, 19),
            candidate("cbb", "catboost", .8856110, .0160278, 20),
        ],
        "m7_handoff": {"test_set_used": False},
    }
    result = select_final_candidate(hpo, task="classification")
    assert result["status"] == "selected"
    assert result["selected"]["algorithm_id"] == "gradient_boosting"
    assert result["selected"]["feature_set_id"] == "A"
    assert result["m7_handoff"]["test_set_used"] is False
