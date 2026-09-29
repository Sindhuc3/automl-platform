from typing import Dict, List
import numpy as np
import pandas as pd
from app.preprocessing.constants import IQR_MULTIPLIER, OUTLIER_MIN_ROWS


def fit_iqr_caps(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
    if len(df) < OUTLIER_MIN_ROWS:
        return {}
    result = {}
    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < OUTLIER_MIN_ROWS or s.nunique() <= 2:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 0:
            continue
        result[col] = {"lower": float(q1 - IQR_MULTIPLIER * iqr), "upper": float(q3 + IQR_MULTIPLIER * iqr)}
    return result


def cap_outliers(df: pd.DataFrame, caps: Dict[str, Dict[str, float]]) -> tuple[pd.DataFrame, Dict[str, int]]:
    out = df.copy()
    counts = {}
    for col, bounds in caps.items():
        if col not in out.columns:
            continue
        before = pd.to_numeric(out[col], errors="coerce")
        clipped = before.clip(bounds["lower"], bounds["upper"])
        counts[col] = int((before != clipped).fillna(False).sum())
        out[col] = clipped
    return out, counts
