from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.preprocessing.constants import MISSING_INDICATOR_MIN, MISSING_EXCLUDE_MIN

VALID_NUMERIC_METHODS = {"mean", "median"}
VALID_CATEGORICAL_METHODS = {"most_frequent", "constant:__MISSING__"}


def _numeric_default(series: pd.Series) -> tuple[str, str]:
    """Choose mean for roughly symmetric numeric data, median for skew/outliers."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 3:
        return "median", "SMALL_SAMPLE_ROBUST_DEFAULT"
    skew = float(values.skew()) if values.nunique() > 2 else 0.0
    if not np.isfinite(skew) or abs(skew) > 1.0:
        return "median", "SKEWED_NUMERIC_MEDIAN"
    return "mean", "SYMMETRIC_NUMERIC_MEAN"


def missing_plan_for_column(
    df: pd.DataFrame,
    column: str,
    data_type: str,
    override: Optional[str] = None,
) -> dict:
    s = df[column]
    missing_count = int(s.isna().sum())
    total = len(s)
    ratio = missing_count / max(1, total)

    if ratio >= MISSING_EXCLUDE_MIN:
        return {"action": "exclude", "method": None, "indicator": False,
                "reason_code": "HIGH_MISSINGNESS", "missing_count": missing_count,
                "missing_percent": round(ratio * 100, 2)}

    if missing_count == 0:
        return {"action": "retain", "method": None, "indicator": False,
                "reason_code": "NO_MISSING_VALUES", "missing_count": 0, "missing_percent": 0.0}

    if data_type == "numeric":
        method, reason = _numeric_default(s)
        if override is not None:
            if override not in VALID_NUMERIC_METHODS:
                raise ValueError(f"PRE_INVALID_MISSING_OVERRIDE: '{override}' is not valid for numeric column '{column}'.")
            method = override
            reason = "GUIDED_OVERRIDE"
        return {"action": "impute", "method": method,
                "indicator": ratio >= MISSING_INDICATOR_MIN,
                "reason_code": reason, "missing_count": missing_count,
                "missing_percent": round(ratio * 100, 2)}

    if data_type == "boolean":
        method = override or "most_frequent"
        if method != "most_frequent":
            raise ValueError(f"PRE_INVALID_MISSING_OVERRIDE: Boolean column '{column}' supports most_frequent imputation only.")
        return {"action": "impute", "method": method,
                "indicator": ratio >= MISSING_INDICATOR_MIN,
                "reason_code": "BOOLEAN_MISSINGNESS", "missing_count": missing_count,
                "missing_percent": round(ratio * 100, 2)}

    if data_type in {"categorical", "numeric_categorical"}:
        method = override
        if method is None:
            method = "most_frequent" if ratio < MISSING_INDICATOR_MIN else "constant:__MISSING__"
        if method not in VALID_CATEGORICAL_METHODS:
            raise ValueError(f"PRE_INVALID_MISSING_OVERRIDE: '{method}' is not valid for categorical column '{column}'.")
        return {"action": "impute", "method": method, "indicator": False,
                "reason_code": "CATEGORICAL_MISSINGNESS" if override is None else "GUIDED_OVERRIDE",
                "missing_count": missing_count, "missing_percent": round(ratio * 100, 2)}

    return {"action": "exclude", "method": None, "indicator": False,
            "reason_code": "UNSUPPORTED_MISSING_TYPE", "missing_count": missing_count,
            "missing_percent": round(ratio * 100, 2)}


def choose_imputation(df: pd.DataFrame, numeric: List[str], categorical: List[str]) -> Dict[str, str]:
    result = {}
    for col in numeric:
        result[col] = _numeric_default(df[col])[0]
    for col in categorical:
        result[col] = "most_frequent"
    return result


def missing_indicator_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    return [c for c in columns if float(df[c].isna().mean()) >= MISSING_INDICATOR_MIN]
