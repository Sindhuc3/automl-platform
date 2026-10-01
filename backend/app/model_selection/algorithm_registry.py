from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .algorithm_specs import AlgorithmSpec
from .defaults import CLASSIFICATION_DEFAULTS, REGRESSION_DEFAULTS, SEARCH_SPACES


def _ctor(cls):
    def factory(params: Mapping[str, Any]):
        return cls(**dict(params))
    return factory


def _optional_ctor(module_name: str, class_name: str):
    def factory(params: Mapping[str, Any]):
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
        except ImportError as exc:
            raise RuntimeError(
                f"M5_OPTIONAL_DEPENDENCY_MISSING: install {module_name} to use this algorithm."
            ) from exc
        return cls(**dict(params))
    return factory


def _spec(
    algorithm_id: str,
    display_name: str,
    task: str,
    family: str,
    scaling: str,
    cost: str,
    factory,
    *,
    supports_predict_proba=False,
    supports_decision_function=False,
    notes="",
    tags=(),
) -> AlgorithmSpec:
    defaults = CLASSIFICATION_DEFAULTS if task == "classification" else REGRESSION_DEFAULTS
    return AlgorithmSpec(
        algorithm_id=algorithm_id,
        display_name=display_name,
        task=task,
        family=family,
        scaling_profile=scaling,
        cost_tier=cost,
        estimator_factory=factory,
        default_parameters=defaults.get(algorithm_id, {}),
        tunable_parameters=SEARCH_SPACES.get(algorithm_id, {}),
        supports_predict_proba=supports_predict_proba,
        supports_decision_function=supports_decision_function,
        notes=notes,
        tags=frozenset(tags),
    )


_CLASSIFICATION: tuple[AlgorithmSpec, ...] = (
    _spec("logistic_regression", "Logistic Regression", "classification", "linear", "standard", "low", _ctor(LogisticRegression), supports_predict_proba=True, tags=("linear", "probabilistic")),
    _spec("naive_bayes", "Naive Bayes (Gaussian)", "classification", "probabilistic", "none", "low", _ctor(GaussianNB), supports_predict_proba=True, notes="V1 uses GaussianNB because M3 supplies a numerical feature matrix.", tags=("probabilistic",)),
    _spec("decision_tree", "Decision Tree", "classification", "tree", "none", "low", _ctor(DecisionTreeClassifier), supports_predict_proba=True, tags=("tree", "nonlinear")),
    _spec("random_forest", "Random Forest", "classification", "bagging", "none", "medium", _ctor(RandomForestClassifier), supports_predict_proba=True, tags=("ensemble", "bagging", "tree")),
    _spec("extra_trees", "Extra Trees", "classification", "bagging", "none", "medium", _ctor(ExtraTreesClassifier), supports_predict_proba=True, tags=("ensemble", "bagging", "tree")),
    _spec("gradient_boosting", "Gradient Boosting", "classification", "boosting", "none", "medium", _ctor(GradientBoostingClassifier), supports_predict_proba=True, tags=("ensemble", "boosting", "tree")),
    _spec("xgboost", "XGBoost", "classification", "boosting", "none", "high", _optional_ctor("xgboost", "XGBClassifier"), supports_predict_proba=True, tags=("ensemble", "boosting", "external")),
    _spec("lightgbm", "LightGBM", "classification", "boosting", "none", "high", _optional_ctor("lightgbm", "LGBMClassifier"), supports_predict_proba=True, tags=("ensemble", "boosting", "external")),
    _spec("catboost", "CatBoost", "classification", "boosting", "none", "high", _optional_ctor("catboost", "CatBoostClassifier"), supports_predict_proba=True, tags=("ensemble", "boosting", "external", "categorical")),
    _spec("svm", "Support Vector Machine (RBF SVC)", "classification", "kernel", "standard", "high", _ctor(SVC), supports_decision_function=True, tags=("kernel", "nonlinear")),
    _spec("knn", "K-Nearest Neighbors", "classification", "distance", "standard", "medium", _ctor(KNeighborsClassifier), supports_predict_proba=True, tags=("distance", "nonlinear")),
)

_REGRESSION: tuple[AlgorithmSpec, ...] = (
    _spec("linear_regression", "Linear Regression", "regression", "linear", "none", "low", _ctor(LinearRegression), tags=("linear",)),
    _spec("ridge", "Ridge Regression", "regression", "linear", "standard", "low", _ctor(Ridge), tags=("linear", "regularized")),
    _spec("lasso", "Lasso Regression", "regression", "linear", "standard", "low", _ctor(Lasso), tags=("linear", "regularized", "sparse")),
    _spec("elastic_net", "ElasticNet Regression", "regression", "linear", "standard", "medium", _ctor(ElasticNet), tags=("linear", "regularized", "sparse")),
    _spec("decision_tree", "Decision Tree Regressor", "regression", "tree", "none", "low", _ctor(DecisionTreeRegressor), tags=("tree", "nonlinear")),
    _spec("random_forest", "Random Forest Regressor", "regression", "bagging", "none", "medium", _ctor(RandomForestRegressor), tags=("ensemble", "bagging", "tree")),
    _spec("extra_trees", "Extra Trees Regressor", "regression", "bagging", "none", "medium", _ctor(ExtraTreesRegressor), tags=("ensemble", "bagging", "tree")),
    _spec("gradient_boosting", "Gradient Boosting Regressor", "regression", "boosting", "none", "medium", _ctor(GradientBoostingRegressor), tags=("ensemble", "boosting", "tree")),
    _spec("xgboost", "XGBoost Regressor", "regression", "boosting", "none", "high", _optional_ctor("xgboost", "XGBRegressor"), tags=("ensemble", "boosting", "external")),
    _spec("lightgbm", "LightGBM Regressor", "regression", "boosting", "none", "high", _optional_ctor("lightgbm", "LGBMRegressor"), tags=("ensemble", "boosting", "external")),
    _spec("catboost", "CatBoost Regressor", "regression", "boosting", "none", "high", _optional_ctor("catboost", "CatBoostRegressor"), tags=("ensemble", "boosting", "external", "categorical")),
    _spec("svr", "Support Vector Regression (RBF)", "regression", "kernel", "standard", "high", _ctor(SVR), supports_decision_function=True, tags=("kernel", "nonlinear")),
    _spec("knn", "K-Nearest Neighbors Regressor", "regression", "distance", "standard", "medium", _ctor(KNeighborsRegressor), tags=("distance", "nonlinear")),
)

_BY_TASK: Dict[str, Dict[str, AlgorithmSpec]] = {
    "classification": {s.algorithm_id: s for s in _CLASSIFICATION},
    "regression": {s.algorithm_id: s for s in _REGRESSION},
}


def list_algorithms(task: Optional[str] = None) -> List[AlgorithmSpec]:
    if task is None:
        return [*_CLASSIFICATION, *_REGRESSION]
    task = task.lower()
    if task not in _BY_TASK:
        raise ValueError(f"M5_INVALID_TASK: {task}")
    return list(_BY_TASK[task].values())


def get_algorithm(algorithm_id: str, task: Optional[str] = None) -> Optional[AlgorithmSpec]:
    if task:
        return _BY_TASK.get(task.lower(), {}).get(algorithm_id)
    matches = [spec for specs in _BY_TASK.values() for spec in specs.values() if spec.algorithm_id == algorithm_id]
    return matches[0] if matches else None


def registry_report(task: Optional[str] = None) -> List[Dict[str, Any]]:
    return [spec.to_dict() for spec in list_algorithms(task)]
