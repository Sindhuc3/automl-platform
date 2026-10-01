from __future__ import annotations
from typing import Any, Dict, List

METHODS = [
    {"id": "variance", "name": "Variance Threshold", "family": "filter", "execution": "module4-forward-or-fold", "automatic": "target-blind only when a genuinely constant feature is found"},
    {"id": "univariate", "name": "Univariate Tests (ANOVA-F / F-regression)", "family": "filter", "execution": "module5-cv", "automatic": True},
    {"id": "mutual_information", "name": "Mutual Information", "family": "filter", "execution": "module5-cv", "automatic": True},
    {"id": "correlation", "name": "Correlation / Redundancy Filtering", "family": "filter", "execution": "module5-cv", "automatic": True},
    {"id": "l1", "name": "L1 / Lasso", "family": "embedded", "execution": "module5-cv", "automatic": True},
    {"id": "tree_importance", "name": "Tree-based Importance", "family": "embedded", "execution": "module5-cv", "automatic": True},
    {"id": "rfe", "name": "Recursive Feature Elimination", "family": "wrapper", "execution": "module5-cv", "automatic": True},
    {"id": "rfecv", "name": "RFECV", "family": "wrapper", "execution": "module5-cv", "automatic": True},
    {"id": "sequential", "name": "Sequential Forward / Backward Selection", "family": "wrapper", "execution": "module5-cv", "automatic": True},
]


def selector_capability_report() -> List[Dict[str, Any]]:
    return METHODS


def build_compact_spec(features: List[str], n_rows: int) -> Dict[str, Any]:
    # Compact is useful when there is enough feature space for a selection
    # problem.  The recipe is resolved independently inside each CV fold.
    enabled = len(features) > 15 and n_rows >= 300
    return {
        "kind": "selector_based",
        "enabled": enabled,
        "fit_scope": "inside_cross_validation",
        "pipeline_spec": [
            {"step": "redundancy_prune", "params": {"threshold": 0.98, "measure": "spearman|cramers_v", "representative": "fold_training_evidence"}},
            {"step": "mi_null_selector", "params": {"n_shuffles": 5, "quantile": 0.95, "min_features": min(len(features), 5), "group_aware": True}},
        ] if enabled else [],
        "n_features_upper_bound": len(features),
        "reason": "Compact selection is model/target dependent, so the recipe is refit inside each Module 5/6 CV fold." if enabled else "Dataset is small or feature count is low; Baseline/Engineered candidates are retained so the model-selection stage can compare them directly.",
    }
