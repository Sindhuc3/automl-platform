
from typing import Dict, List
import pandas as pd
from app.preprocessing.constants import MISSING_INDICATOR_MIN, MISSING_EXCLUDE_MIN


def missing_plan_for_column(df: pd.DataFrame, column: str, data_type: str) -> dict:
    s = df[column]
    missing_count = int(s.isna().sum())
    total = len(s)
    ratio = missing_count / max(1, total)

    if ratio >= MISSING_EXCLUDE_MIN:
        return {
            "action": "exclude",
            "method": None,
            "indicator": False,
            "reason_code": "HIGH_MISSINGNESS",
            "missing_count": missing_count,
            "missing_percent": round(ratio * 100, 2),
        }

    if missing_count == 0:
        return {
            "action": "retain",
            "method": None,
            "indicator": False,
            "reason_code": "NO_MISSING_VALUES",
            "missing_count": 0,
            "missing_percent": 0.0,
        }

    if data_type == "numeric":
        indicator = ratio >= MISSING_INDICATOR_MIN
        return {
            "action": "impute",
            "method": "median",
            "indicator": indicator,
            "reason_code": "NUMERIC_MISSINGNESS",
            "missing_count": missing_count,
            "missing_percent": round(ratio * 100, 2),
        }

    if data_type == "boolean":
        return {
            "action": "impute",
            "method": "most_frequent",
            "indicator": ratio >= MISSING_INDICATOR_MIN,
            "reason_code": "BOOLEAN_MISSINGNESS",
            "missing_count": missing_count,
            "missing_percent": round(ratio * 100, 2),
        }

    if data_type in {"categorical", "numeric_categorical"}:
        if ratio < MISSING_INDICATOR_MIN:
            method = "most_frequent"
            indicator = False
        else:
            method = "constant:__MISSING__"
            indicator = False
        return {
            "action": "impute",
            "method": method,
            "indicator": indicator,
            "reason_code": "CATEGORICAL_MISSINGNESS",
            "missing_count": missing_count,
            "missing_percent": round(ratio * 100, 2),
        }

    return {
        "action": "exclude",
        "method": None,
        "indicator": False,
        "reason_code": "UNSUPPORTED_MISSING_TYPE",
        "missing_count": missing_count,
        "missing_percent": round(ratio * 100, 2),
    }


def choose_imputation(df: pd.DataFrame, numeric: List[str], categorical: List[str]) -> Dict[str, str]:
    result = {}
    for col in numeric:
        result[col] = "median"
    for col in categorical:
        result[col] = "most_frequent"
    return result


def missing_indicator_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    return [
        c for c in columns
        if float(df[c].isna().mean()) >= MISSING_INDICATOR_MIN
    ]
