from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, f_classif, f_regression


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a.astype(str).fillna("<NA>"), b.astype(str).fillna("<NA>"))
    if table.empty or min(table.shape) < 2:
        return 0.0
    observed = table.to_numpy(dtype=float)
    n = observed.sum()
    if n <= 0:
        return 0.0
    rows = observed.sum(axis=1, keepdims=True)
    cols = observed.sum(axis=0, keepdims=True)
    expected = rows @ cols / n
    mask = expected > 0
    chi2 = ((observed - expected) ** 2 / np.where(mask, expected, 1)).sum()
    phi2 = chi2 / n
    r, k = observed.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(1.0, n - 1))
    rcorr = r - ((r - 1) ** 2) / max(1.0, n - 1)
    kcorr = k - ((k - 1) ** 2) / max(1.0, n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def _safe_numeric_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.isna().all():
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return x.fillna(x.median())


def feature_statistics(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        unique = int(non_null.nunique())
        missing = int(s.isna().sum())
        item = {
            "feature": col,
            "dtype": str(s.dtype),
            "rows": int(len(s)),
            "missing_count": missing,
            "missing_percent": round(100 * missing / max(1, len(s)), 2),
            "unique_count": unique,
            "unique_ratio": round(unique / max(1, len(non_null)), 4),
            "constant": unique <= 1,
            "variance": None,
            "skewness": None,
        }
        if pd.api.types.is_numeric_dtype(s):
            x = pd.to_numeric(s, errors="coerce")
            item["variance"] = float(x.var()) if x.notna().sum() > 1 else 0.0
            item["skewness"] = float(x.skew()) if x.notna().sum() > 2 else 0.0
        rows.append(item)
    return rows


def redundancy_pairs(df: pd.DataFrame, threshold: float = 0.95) -> List[Dict[str, Any]]:
    result = []
    numeric = _numeric_frame(df)
    if numeric.shape[1] > 1:
        corr = numeric.corr(method="spearman").abs()
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i + 1:]:
                value = corr.loc[a, b]
                if pd.notna(value) and value >= threshold:
                    result.append({"left": a, "right": b, "measure": "spearman", "value": round(float(value), 4)})
    cats = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for i, a in enumerate(cats):
        for b in cats[i + 1:]:
            if df[a].nunique(dropna=True) > 50 or df[b].nunique(dropna=True) > 50:
                continue
            value = _cramers_v(df[a], df[b])
            if value >= threshold:
                result.append({"left": a, "right": b, "measure": "cramers_v", "value": round(value, 4)})
    return result


def target_relevance(df: pd.DataFrame, y: pd.Series, problem_type: str, seed: int = 42) -> List[Dict[str, Any]]:
    features = [c for c in df.columns if c != y.name]
    if not features or len(df) < 3:
        return []
    X = df[features].copy()
    numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in features if c not in numeric_cols]
    # MI in sklearn needs a numeric matrix. Categorical values are frequency-coded for diagnostics only.
    for c in numeric_cols:
        X[c] = _safe_numeric_series(X[c])
    for c in cat_cols:
        counts = X[c].astype(str).fillna("<NA>").value_counts(dropna=False)
        X[c] = X[c].astype(str).fillna("<NA>").map(counts).astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if problem_type == "classification":
        yy = pd.Series(pd.factorize(y.astype(str))[0], index=y.index)
        scores = mutual_info_classif(X, yy, random_state=seed)
        try:
            f_scores, pvals = f_classif(X, yy)
        except Exception:
            f_scores = np.full(len(features), np.nan)
            pvals = np.full(len(features), np.nan)
    else:
        yy = pd.to_numeric(y, errors="coerce").fillna(pd.to_numeric(y, errors="coerce").median())
        scores = mutual_info_regression(X, yy, random_state=seed)
        try:
            f_scores, pvals = f_regression(X, yy)
        except Exception:
            f_scores = np.full(len(features), np.nan)
            pvals = np.full(len(features), np.nan)
    return [
        {"feature": c, "mutual_information": round(float(mi), 6),
         "univariate_f": None if not np.isfinite(fs) else round(float(fs), 6),
         "univariate_p": None if not np.isfinite(pv) else round(float(pv), 6)}
        for c, mi, fs, pv in zip(features, scores, f_scores, pvals)
    ]
