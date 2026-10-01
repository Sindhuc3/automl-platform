from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from .constants import MAX_GENERATED_FEATURES, MAX_DATE_PARTS, MAX_INTERACTION_FEATURES


def _numeric(df, cols):
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def _safe_num(s):
    x = pd.to_numeric(s, errors="coerce")
    return x.fillna(x.median() if x.notna().any() else 0.0)


def _add(out, records, name, values, parents, transform, reason):
    if name in out.columns or len(records) >= MAX_GENERATED_FEATURES:
        return False
    v = pd.Series(values, index=out.index).replace([np.inf, -np.inf], np.nan)
    if v.notna().sum() == 0 or v.nunique(dropna=True) <= 1:
        return False
    out[name] = v
    records.append({"feature_id": f"f4_{len(records)+1:04d}", "name": name, "origin": "engineered", "parents": parents,
                    "transform": transform, "decision": "CREATE", "mode": "AUTO",
                    "operation": transform, "effect": {"added": [name], "parents": parents},
                    "fit_scope": {"learned_from": "training_data_only", "n_rows_used": len(out), "sampled": False, "seed": 42},
                    "explanation": reason, "what_happens_next": f"{name} is added to the Engineered candidate and remains eligible for Compact if it passes later redundancy and selection rules.",
                    "candidate_sets": {"Baseline": "not added", "Engineered": "added", "Compact": "eligible"}, "reversible": True})
    return True


def engineer_features(df: pd.DataFrame, source_stats: List[Dict[str, Any]], max_generated: int = MAX_GENERATED_FEATURES) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    out = df.copy()
    records: List[Dict[str, Any]] = []
    numeric = _numeric(out, list(out.columns))

    # Datetime parts: use datetime-like columns even when Module 3 excluded the raw datetime.
    for col in list(out.columns):
        if len(records) >= max_generated:
            break
        s = out[col]
        dt = None
        if pd.api.types.is_datetime64_any_dtype(s):
            dt = pd.to_datetime(s, errors="coerce", format="mixed")
        elif s.dtype == object:
            parsed = pd.to_datetime(s, errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.8 and parsed.notna().sum() >= 3:
                dt = parsed
        if dt is None:
            continue
        parts = [("year", dt.dt.year), ("month", dt.dt.month), ("day", dt.dt.day),
                 ("dayofweek", dt.dt.dayofweek), ("is_month_end", dt.dt.is_month_end.astype(int)),
                 ("is_month_start", dt.dt.is_month_start.astype(int))]
        for part, vals in parts[:MAX_DATE_PARTS]:
            _add(out, records, f"{col}__{part}", vals, [col], f"datetime_{part}",
                 f"{col} is date-like, so {part} extracts a reusable calendar component without exposing the raw datetime object to downstream models.")

    # Missingness count preserves information about row-level incompleteness.
    missing_cols = [c for c in out.columns if out[c].isna().any()]
    if missing_cols:
        _add(out, records, "__missing_count", out[missing_cols].isna().sum(axis=1), missing_cols, "row_missing_count",
             f"A row-level missing-count feature summarizes how many source fields are incomplete for each observation; it is target-blind and uses no test-set statistics.")

    # Learned encodings such as frequency encoding are NOT materialized here.
    # Their statistics must be learned separately inside each CV training fold.
    # Module 4 records such recipes; Module 5 executes them in-fold.

    # High-confidence semantic rule: quantity x price / amount-like pair.
    lower = {c: c.lower() for c in out.columns}
    qty = [c for c in numeric if any(k in lower[c] for k in ("qty", "quantity", "units"))]
    price = [c for c in numeric if any(k in lower[c] for k in ("price", "unitprice", "unit_price", "rate"))]
    for q in qty:
        for p in price:
            if q == p or len(records) >= max_generated:
                continue
            name = f"{q}__x__{p}"
            _add(out, records, name, _safe_num(out[q]) * _safe_num(out[p]), [q, p], "quantity_times_price",
                 f"The names {q} and {p} match common quantity/price semantics, so their product is a high-confidence total-value feature rather than a blind pairwise interaction.")

    # Safe numeric pair features: only a small deterministic sample, never all pairs.
    numeric = _numeric(out, list(df.columns))
    if len(numeric) >= 2 and len(records) < max_generated:
        for i, a in enumerate(numeric):
            for b in numeric[i + 1:]:
                if len(records) >= min(max_generated, MAX_GENERATED_FEATURES + 0):
                    break
                an, bn = a.lower(), b.lower()
                if any(k in an + bn for k in ("id", "index", "identifier", "uuid")):
                    continue
                # Only create a ratio when names suggest a meaningful rate/proportion relationship.
                if any(k in an for k in ("amount", "total", "cost", "price")) and any(k in bn for k in ("count", "qty", "quantity", "units")):
                    denom = _safe_num(out[b]).replace(0, np.nan)
                    _add(out, records, f"{a}__per__{b}", _safe_num(out[a]) / denom, [a, b], "safe_ratio",
                         f"The pair {a} and {b} has amount/count semantics, so a per-unit ratio is a bounded, interpretable derived feature.")
                    break
            if len(records) >= max_generated:
                break

    return out, records
