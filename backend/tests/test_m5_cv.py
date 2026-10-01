import numpy as np
import pandas as pd

from app.model_selection.cv.cv_plan import build_cv_plan
from app.model_selection.cv.metrics import build_metric_bundle, paired_corrected_se


def test_classification_plan_is_stratified_and_reusable():
    y = np.array([0] * 80 + [1] * 20)
    X = pd.DataFrame({"x": np.arange(100)})
    plan = build_cv_plan(X, y, task="classification", requested_splits=5, requested_repeats=1)
    assert plan.strategy == "stratified_kfold"
    assert len(plan.folds) == 5
    assert all(len(f.train_indices) + len(f.validation_indices) == 100 for f in plan.folds)


def test_small_regression_uses_quantile_stratification_when_possible():
    rng = np.random.default_rng(42)
    y = np.exp(rng.normal(size=120))
    X = pd.DataFrame({"x": rng.normal(size=120)})
    plan = build_cv_plan(X, y, task="regression", requested_splits=5, requested_repeats=1)
    assert len(plan.folds) == 5
    assert "quantile" in plan.strategy


def test_metric_primary_is_frozen_before_candidate_results():
    y_balanced = np.array([0] * 90 + [1] * 10)
    bundle = build_metric_bundle("classification", y_balanced)
    assert bundle.primary == "roc_auc"

    y_rare = np.array([0] * 95 + [1] * 5)
    bundle_rare = build_metric_bundle("classification", y_rare)
    assert bundle_rare.primary == "average_precision"


def test_paired_corrected_se_zero_for_identical_scores():
    assert paired_corrected_se([0.0, 0.0, 0.0, 0.0, 0.0], n_train=80, n_validation=20) == 0.0
