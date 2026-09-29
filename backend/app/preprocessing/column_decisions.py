
from typing import Any, Dict, List
import pandas as pd
from app.preprocessing.constants import (
    MISSING_EXCLUDE_MIN,
    HIGH_CARD_UNIQUE_RATIO,
    OHE_MAX_DISTINCT,
)
from app.preprocessing.missing_values import missing_plan_for_column
from app.preprocessing.encoding import automatic_encoding_decision


def _type_map_from_quality(quality: Dict[str, Any]) -> Dict[str, str]:
    result = {}
    for item in quality.get("column_profiles", []) or []:
        if isinstance(item, dict) and item.get("column"):
            result[item["column"]] = item.get("type", "unknown")
    raw = quality.get("column_types") or {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                result[k] = v.get("type", "unknown")
            else:
                result[k] = v
    return result


def build_column_decisions(
    df: pd.DataFrame,
    target: str,
    validation: Dict[str, Any],
) -> List[Dict[str, Any]]:
    type_map = _type_map_from_quality(validation)
    excluded_from_m2 = {
        x.get("column"): x.get("reason")
        for x in (validation.get("excluded_features") or [])
        if isinstance(x, dict) and x.get("column")
    }

    decisions = []

    for col in df.columns:
        if col == target:
            decisions.append({
                "column": col,
                "action": "exclude",
                "category": "target",
                "reason_code": "TARGET",
                "reason": "The selected target is kept as y and is never included among input features.",
            })
            continue

        missing_ratio = float(df[col].isna().mean())
        unique = int(df[col].nunique(dropna=True))
        valid_count = int(df[col].notna().sum())
        unique_ratio = unique / max(1, valid_count)
        ctype = type_map.get(col, "")

        m2_reason = excluded_from_m2.get(col)
        if m2_reason in {"identifier", "identifier_like", "contact", "constant", "free_text", "datetime"}:
            decisions.append({
                "column": col,
                "action": "exclude",
                "category": "exclusion",
                "reason_code": f"EXCL_{str(m2_reason).upper()}",
                "reason": f"Module 2 classified this column as {m2_reason}; it is not a safe predictive feature in the default automatic pipeline.",
                "observed": {
                    "row_count": len(df),
                    "missing_count": int(df[col].isna().sum()),
                    "missing_percent": round(missing_ratio * 100, 2),
                    "unique_count": unique,
                    "unique_ratio": round(unique_ratio, 4),
                },
            })
            continue

        if missing_ratio >= MISSING_EXCLUDE_MIN:
            decisions.append({
                "column": col,
                "action": "exclude",
                "category": "exclusion",
                "reason_code": "EXCL_HIGH_MISSING",
                "reason": (
                    f"{missing_ratio:.1%} of the observed rows are missing in this column, "
                    f"which exceeds the automatic {MISSING_EXCLUDE_MIN:.0%} exclusion threshold."
                ),
                "observed": {
                    "row_count": len(df),
                    "missing_count": int(df[col].isna().sum()),
                    "missing_percent": round(missing_ratio * 100, 2),
                    "unique_count": unique,
                    "unique_ratio": round(unique_ratio, 4),
                },
            })
            continue

        if ctype in {"numeric", "float", "integer"}:
            data_type = "numeric"
        elif ctype in {"numeric_categorical"}:
            data_type = "numeric_categorical"
        elif ctype in {"boolean", "bool"}:
            data_type = "boolean"
        elif ctype in {"categorical", "object", "string", "text"}:
            data_type = "categorical"
        elif pd.api.types.is_bool_dtype(df[col]):
            data_type = "boolean"
        elif pd.api.types.is_numeric_dtype(df[col]):
            data_type = "numeric"
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            data_type = "categorical"
        else:
            decisions.append({
                "column": col,
                "action": "exclude",
                "category": "exclusion",
                "reason_code": "EXCL_UNSUPPORTED",
                "reason": "The detected feature type is not supported by the Module 3 automatic preprocessing recipe.",
            })
            continue

        if data_type in {"categorical", "numeric_categorical", "boolean"}:
            enc = automatic_encoding_decision(
                col, data_type, unique, unique_ratio, missing_ratio
            )
            if enc["encoding"] == "excluded_high_cardinality":
                decisions.append({
                    "column": col,
                    "action": "exclude",
                    "category": "exclusion",
                    "reason_code": "EXCL_HIGH_CARDINALITY",
                    "data_type": data_type,
                    "reason": (
                        f"{unique} distinct values across {valid_count} non-missing rows "
                        f"would create a high-dimensional categorical representation. "
                        "Automatic V1 avoids target encoding because of leakage risk and does not "
                        "invent an ordinal relationship."
                    ),
                    "observed": {
                        "row_count": len(df),
                        "missing_count": int(df[col].isna().sum()),
                        "missing_percent": round(missing_ratio * 100, 2),
                        "unique_count": unique,
                        "unique_ratio": round(unique_ratio, 4),
                    },
                })
                continue

        missing_plan = missing_plan_for_column(df, col, data_type)

        decisions.append({
            "column": col,
            "action": "use",
            "category": "feature",
            "data_type": data_type,
            "missing_ratio": missing_ratio,
            "unique": unique,
            "unique_ratio": unique_ratio,
            "missing_plan": missing_plan,
            "encoding": (
                "one_hot"
                if data_type in {"categorical", "numeric_categorical", "boolean"}
                else None
            ),
            "outlier_treatment": (
                "iqr_cap"
                if data_type == "numeric"
                else None
            ),
        })

    return decisions
