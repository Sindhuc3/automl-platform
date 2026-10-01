from __future__ import annotations
from typing import Any, Dict, List, Tuple
import hashlib
import pandas as pd
import numpy as np
from .constants import *


def _is_id_like(name: str, s: pd.Series) -> bool:
    n = name.lower().replace("_", "").replace("-", "")
    hints = ("id", "identifier", "uuid", "guid", "serial", "index", "registration", "roll", "usn")
    ratio = s.nunique(dropna=True) / max(1, s.notna().sum())
    return ratio >= 0.95 and (any(h in n for h in hints) or pd.api.types.is_integer_dtype(s))


def _feature_signature(s: pd.Series) -> str:
    vals = s.astype(str).fillna("<NA>").tolist()
    return hashlib.sha256("\x1f".join(vals).encode("utf-8")).hexdigest()


def safe_filter(df: pd.DataFrame, relevance: List[Dict[str, Any]] | None = None, correlation_prune: float = CORRELATION_PRUNE) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
    decisions: List[Dict[str, Any]] = []
    keep: List[str] = []
    signatures: Dict[str, str] = {}

    for col in df.columns:
        s = df[col]
        nunique = int(s.nunique(dropna=True))
        ratio = nunique / max(1, s.notna().sum())
        evidence = {"n_rows": len(df), "n_unique": nunique, "unique_ratio": round(ratio, 4), "dtype": str(s.dtype)}
        if nunique <= 1:
            decisions.append({"feature": col, "decision": "EXCLUDE", "rule_id": "CONSTANT", "mode": "AUTO", "evidence": evidence,
                              "threshold": {"value": 1, "name": "minimum_unique_values", "kind": "rule", "user_adjustable": False},
                              "effect": {"baseline": "removed"}, "fit_scope": {"learned_from": "training_data_only", "n_rows_used": len(df), "sampled": False, "seed": 42},
                              "explanation": f"{col} has {nunique} observed unique value(s), so it carries no variation for a predictive model. Keeping a constant feature would add a column without providing information that can distinguish observations.",
                              "what_happens_next": f"{col} is excluded from all candidate feature sets.", "candidate_sets": {"Baseline": "excluded", "Engineered": "excluded", "Compact": "excluded"}, "reversible": True})
            continue
        counts = s.dropna().value_counts()
        if len(counts) and counts.iloc[0] / max(1, s.notna().sum()) >= NEAR_CONSTANT_DOMINANCE and (len(counts) - 1) > 0 and int(s.notna().sum() - counts.iloc[0]) < NEAR_CONSTANT_RARE_COUNT:
            # Rare-event predictors can be genuinely useful.  Flag them instead
            # of deleting them; Module 5/6 can decide using CV evidence.
            near_constant_flag = {
                "feature": col, "decision": "FLAG", "rule_id": "NEAR_CONSTANT", "mode": "AUTO",
                "evidence": {**evidence, "dominant_ratio": round(float(counts.iloc[0] / max(1, s.notna().sum())), 4)},
                "threshold": {"dominance": NEAR_CONSTANT_DOMINANCE, "rare_count": NEAR_CONSTANT_RARE_COUNT, "kind": "heuristic", "user_adjustable": True},
                "effect": {"baseline": "retained", "compact": "eligible_for_selection"},
                "fit_scope": {"learned_from": "training_data_only", "n_rows_used": len(df), "sampled": False, "seed": 42},
                "explanation": f"{col} is highly concentrated in one value. It is flagged rather than automatically deleted because a rare minority pattern can still carry predictive signal.",
                "what_happens_next": f"{col} remains available for candidate-set evaluation; later CV-based selection may remove it.",
                "candidate_sets": {"Baseline": "retained", "Engineered": "retained", "Compact": "eligible"}, "reversible": True
            }
            decisions.append(near_constant_flag)
        if _is_id_like(col, s):
            decisions.append({"feature": col, "decision": "EXCLUDE", "rule_id": "ID_LIKE", "mode": "AUTO", "evidence": evidence,
                              "threshold": {"value": 0.95, "name": "unique_ratio", "kind": "heuristic", "user_adjustable": True},
                              "effect": {"baseline": "removed"}, "fit_scope": {"learned_from": "training_data_only", "n_rows_used": len(df), "sampled": False, "seed": 42},
                              "explanation": f"{col} is almost unique per row and its name/type is consistent with an identifier. Identifier-like values usually represent row identity rather than a reusable predictive measurement, so the platform excludes the column to reduce noise and leakage risk.",
                              "what_happens_next": f"{col} is excluded from all automatic candidate sets, while its decision remains recorded in the lineage.", "candidate_sets": {"Baseline": "excluded", "Engineered": "excluded", "Compact": "excluded"}, "reversible": True})
            continue
        sig = _feature_signature(s)
        if sig in signatures:
            original = signatures[sig]
            decisions.append({"feature": col, "decision": "EXCLUDE", "rule_id": "DUPLICATE_FEATURE", "mode": "AUTO", "evidence": {**evidence, "duplicate_of": original},
                              "threshold": {"value": 1.0, "name": "exact_feature_signature", "kind": "rule", "user_adjustable": False},
                              "effect": {"baseline": "removed", "duplicate_of": original}, "fit_scope": {"learned_from": "training_data_only", "n_rows_used": len(df), "sampled": False, "seed": 42},
                              "explanation": f"{col} contains exactly the same observed values as {original}. Keeping both would duplicate the same information and could make later feature selection less stable, so the duplicate is removed while the original is retained.",
                              "what_happens_next": f"{col} is excluded and {original} remains the representative feature.", "candidate_sets": {"Baseline": "excluded", "Engineered": "excluded", "Compact": "excluded"}, "reversible": True})
            continue
        signatures[sig] = col
        keep.append(col)
        decisions.append({"feature": col, "decision": "KEEP", "rule_id": "SAFE_FILTER_PASS", "mode": "AUTO", "evidence": evidence,
                          "effect": {"baseline": "retained"}, "fit_scope": {"learned_from": "training_data_only", "n_rows_used": len(df), "sampled": False, "seed": 42},
                          "explanation": f"{col} passed the target-blind safety filters. It has usable variation and does not match the automatic constant, near-constant, identifier-like or exact-duplicate exclusion rules, so it remains available for candidate feature sets.",
                          "what_happens_next": f"{col} is retained in Baseline and carried into Engineered unless a later generated-feature or redundancy rule changes its membership.", "candidate_sets": {"Baseline": "retained", "Engineered": "retained", "Compact": "eligible"}, "reversible": True})

    current = df[keep].copy()
    # Target-free redundancy pruning for the compact candidate set; baseline keeps all safe features.
    compact = current.copy()
    removed_redundant = []
    numeric = compact.select_dtypes(include=[np.number])
    if numeric.shape[1] > 1:
        corr = numeric.corr(method="spearman").abs()
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i + 1:]:
                if a not in compact.columns or b not in compact.columns:
                    continue
                v = corr.loc[a, b]
                if pd.notna(v) and v >= correlation_prune:
                    # Target-blind representative rule.  Predictive relevance is
                    # deliberately not used here; supervised selection belongs in
                    # Module 5/6 inside CV.
                    ma = float(compact[a].isna().mean())
                    mb = float(compact[b].isna().mean())
                    if ma != mb:
                        drop = a if ma > mb else b
                    else:
                        drop = max(a, b)
                    if drop in compact.columns:
                        compact = compact.drop(columns=[drop])
                        removed_redundant.append((drop, a if drop == b else b, float(v)))
    for drop, other, v in removed_redundant:
        decisions.append({
            "feature": drop,
            "decision": "FLAG",
            "rule_id": "REDUNDANT_CORRELATION",
            "mode": "AUTO",
            "evidence": {"paired_with": other, "spearman_abs": round(v, 4), "measure": "Spearman correlation"},
            "threshold": {"value": correlation_prune, "name": "correlation_prune_threshold", "kind": "heuristic", "user_adjustable": True},
            "effect": {
                "baseline": "retained",
                "compact": "candidate member removed",
                "paired_feature": other
            },
            "fit_scope": {"learned_from": "training_data_only", "n_rows_used": len(df), "sampled": False, "seed": 42},
            "explanation": f"{drop} shows a very strong monotonic relationship with {other} in the training data. Because highly correlated numeric features can carry overlapping information, the feature is flagged for redundancy analysis rather than removed globally. The Baseline candidate keeps {drop}; the Compact candidate can retain the stronger representative and remove the redundant member.",
            "what_happens_next": f"{drop} remains in Baseline. Compact removes this member of the redundant pair while retaining {other} as the representative.",
            "candidate_sets": {"Baseline": "retained", "Engineered": "retained if present", "Compact": "removed from candidate"},
            "reversible": True
        })
    return current, decisions, removed_redundant
