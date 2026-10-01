"""Near-default configurations used by Module 5 screening.

M5 is a screening stage. These values are intentionally bounded; broad
hyperparameter optimization belongs to Module 6.
"""

RANDOM_SEED = 42

CLASSIFICATION_DEFAULTS = {
    "logistic_regression": {"max_iter": 1000, "solver": "lbfgs"},
    "naive_bayes": {},
    "decision_tree": {"random_state": RANDOM_SEED},
    "random_forest": {"n_estimators": 200, "random_state": RANDOM_SEED, "n_jobs": 1},
    "extra_trees": {"n_estimators": 200, "random_state": RANDOM_SEED, "n_jobs": 1},
    "gradient_boosting": {"random_state": RANDOM_SEED},
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "random_state": RANDOM_SEED,
        "n_jobs": 1,
        "tree_method": "hist",
        "verbosity": 0,
    },
    "lightgbm": {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "random_state": RANDOM_SEED,
        "n_jobs": 1,
        "verbosity": -1,
    },
    "catboost": {
        "iterations": 200,
        "depth": 6,
        "learning_rate": 0.1,
        "random_seed": RANDOM_SEED,
        "verbose": False,
        "thread_count": 1,
    },
    "svm": {"kernel": "rbf", "probability": False},
    "knn": {"n_neighbors": 5},
}

REGRESSION_DEFAULTS = {
    "linear_regression": {},
    "ridge": {"alpha": 1.0},
    "lasso": {"alpha": 1.0, "max_iter": 5000},
    "elastic_net": {"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 5000},
    "decision_tree": {"random_state": RANDOM_SEED},
    "random_forest": {"n_estimators": 200, "random_state": RANDOM_SEED, "n_jobs": 1},
    "extra_trees": {"n_estimators": 200, "random_state": RANDOM_SEED, "n_jobs": 1},
    "gradient_boosting": {"random_state": RANDOM_SEED},
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "random_state": RANDOM_SEED,
        "n_jobs": 1,
        "tree_method": "hist",
        "verbosity": 0,
    },
    "lightgbm": {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "random_state": RANDOM_SEED,
        "n_jobs": 1,
        "verbosity": -1,
    },
    "catboost": {
        "iterations": 200,
        "depth": 6,
        "learning_rate": 0.1,
        "random_seed": RANDOM_SEED,
        "verbose": False,
        "thread_count": 1,
    },
    "svr": {"kernel": "rbf"},
    "knn": {"n_neighbors": 5},
}

# M6 consumes these as candidate search spaces. Values are deliberately
# represented as simple JSON-friendly lists/ranges rather than sklearn
# distributions so they can later be serialized into run manifests.
SEARCH_SPACES = {
    "logistic_regression": {"C": [0.01, 0.1, 1.0, 10.0]},
    "naive_bayes": {"var_smoothing": [1e-11, 1e-9, 1e-7]},
    "decision_tree": {"max_depth": [None, 3, 5, 10, 20], "min_samples_leaf": [1, 2, 5, 10]},
    "random_forest": {"n_estimators": [100, 200, 400], "max_depth": [None, 5, 10, 20], "min_samples_leaf": [1, 2, 5]},
    "extra_trees": {"n_estimators": [100, 200, 400], "max_depth": [None, 5, 10, 20], "min_samples_leaf": [1, 2, 5]},
    "gradient_boosting": {"n_estimators": [100, 200, 400], "learning_rate": [0.03, 0.05, 0.1], "max_depth": [2, 3, 5]},
    "xgboost": {"n_estimators": [100, 200, 400], "learning_rate": [0.03, 0.05, 0.1], "max_depth": [3, 5, 7]},
    "lightgbm": {"n_estimators": [100, 200, 400], "learning_rate": [0.03, 0.05, 0.1], "num_leaves": [15, 31, 63]},
    "catboost": {"iterations": [100, 200, 400], "learning_rate": [0.03, 0.05, 0.1], "depth": [4, 6, 8]},
    "svm": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01, 0.1]},
    "knn": {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"]},
    "linear_regression": {},
    "ridge": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "lasso": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
    "elastic_net": {"alpha": [0.001, 0.01, 0.1, 1.0], "l1_ratio": [0.1, 0.5, 0.9]},
    "svr": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01, 0.1], "epsilon": [0.01, 0.1, 0.2]},
}
