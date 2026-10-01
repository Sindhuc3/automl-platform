from app.model_selection.algorithm_registry import list_algorithms
from app.model_selection.model_factory import build_estimator


def test_classification_registry_has_expected_algorithms():
    ids = [s.algorithm_id for s in list_algorithms("classification")]
    assert ids == [
        "logistic_regression", "naive_bayes", "decision_tree", "random_forest",
        "extra_trees", "gradient_boosting", "xgboost", "lightgbm", "catboost", "svm", "knn",
    ]


def test_regression_registry_has_expected_algorithms():
    ids = [s.algorithm_id for s in list_algorithms("regression")]
    assert ids == [
        "linear_regression", "ridge", "lasso", "elastic_net", "decision_tree",
        "random_forest", "extra_trees", "gradient_boosting", "xgboost", "lightgbm",
        "catboost", "svr", "knn",
    ]


def test_factories_build_fresh_estimators():
    for spec in list_algorithms("classification") + list_algorithms("regression"):
        a = build_estimator(spec)
        b = build_estimator(spec)
        assert type(a) is type(b)
        assert a is not b
