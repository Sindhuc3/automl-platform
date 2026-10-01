from typing import Any, Dict, List
import pandas as pd
from app.preprocessing.constants import MISSING_EXCLUDE_MIN
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
            result[k] = v.get("type", "unknown") if isinstance(v, dict) else v
    return result


def build_column_decisions(df: pd.DataFrame, target: str, validation: Dict[str, Any], overrides: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    overrides = overrides or {}
    column_overrides = overrides.get("columns", {}) or {}
    type_map = _type_map_from_quality(validation)
    excluded_from_m2 = {x.get("column"): x.get("reason") for x in (validation.get("excluded_features") or []) if isinstance(x, dict) and x.get("column")}
    decisions = []

    for col in df.columns:
        if col == target:
            decisions.append({"column": col, "action": "exclude", "category": "target", "reason_code": "TARGET",
                              "reason": "The selected target is kept as y and is never included among input features."})
            continue

        missing_ratio = float(df[col].isna().mean())
        unique = int(df[col].nunique(dropna=True))
        valid_count = int(df[col].notna().sum())
        unique_ratio = unique / max(1, valid_count)
        ctype = type_map.get(col, "")

        m2_reason = excluded_from_m2.get(col)
        if m2_reason in {"identifier", "identifier_like", "contact", "constant", "free_text", "datetime"}:
            decisions.append({"column": col, "action": "exclude", "category": "exclusion", "reason_code": f"EXCL_{str(m2_reason).upper()}",
                              "reason": f"Module 2 classified this column as {m2_reason}; it is not a safe predictive feature in the default automatic pipeline.",
                              "observed": {"row_count": len(df), "missing_count": int(df[col].isna().sum()), "missing_percent": round(missing_ratio * 100, 2), "unique_count": unique, "unique_ratio": round(unique_ratio, 4)}})
            continue

        if missing_ratio >= MISSING_EXCLUDE_MIN:
            decisions.append({"column": col, "action": "exclude", "category": "exclusion", "reason_code": "EXCL_HIGH_MISSING",
                              "reason": f"{missing_ratio:.1%} of values are missing, exceeding the automatic {MISSING_EXCLUDE_MIN:.0%} threshold.",
                              "observed": {"row_count": len(df), "missing_count": int(df[col].isna().sum()), "missing_percent": round(missing_ratio * 100, 2), "unique_count": unique, "unique_ratio": round(unique_ratio, 4)}})
            continue

        if ctype in {"numeric", "float", "integer"}: data_type = "numeric"
        elif ctype == "numeric_categorical": data_type = "numeric_categorical"
        elif ctype in {"boolean", "bool"}: data_type = "boolean"
        elif ctype in {"categorical", "object", "string", "text"}: data_type = "categorical"
        elif pd.api.types.is_bool_dtype(df[col]): data_type = "boolean"
        elif pd.api.types.is_numeric_dtype(df[col]): data_type = "numeric"
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]): data_type = "categorical"
        else:
            decisions.append({"column": col, "action": "exclude", "category": "exclusion", "reason_code": "EXCL_UNSUPPORTED",
                              "reason": "The detected feature type is not supported by the automatic preprocessing recipe."})
            continue

        override = column_overrides.get(col, {}) or {}
        missing_override = override.get("imputation")
        missing_plan = missing_plan_for_column(df, col, data_type, missing_override)
        if missing_plan.get("action") == "exclude":
            decisions.append({"column": col, "action": "exclude", "category": "exclusion", "reason_code": "EXCL_HIGH_MISSING",
                              "data_type": data_type, "missing_plan": missing_plan, "reason": "The feature has too much missingness for a stable automatic representation."})
            continue

        encoding = None
        if data_type in {"categorical", "numeric_categorical", "boolean"}:
            encoding = automatic_encoding_decision(col, data_type, unique, unique_ratio, missing_ratio, df[col])
            requested_encoding = override.get("encoding")
            if requested_encoding:
                encoding = dict(encoding)
                if requested_encoding == "ordinal":
                    from app.preprocessing.encoding import detect_ordinal_order
                    order = override.get("categories") or detect_ordinal_order(col, df[col], data_type)
                    if not order:
                        raise ValueError(f"PRE_INVALID_ENCODING_OVERRIDE: '{col}' requires an explicit or detectable category order for ordinal encoding.")
                    encoding.update({"encoding": "ordinal", "method": "OrdinalEncoder", "categories": order, "reason_code": "GUIDED_OVERRIDE"})
                elif requested_encoding in {"one_hot", "frequency", "boolean"}:
                    if data_type == "boolean" and requested_encoding != "boolean":
                        raise ValueError(f"PRE_INVALID_ENCODING_OVERRIDE: Boolean column '{col}' must use 0/1 encoding.")
                    encoding.update({"encoding": requested_encoding, "method": {"one_hot": "OneHotEncoder", "frequency": "FrequencyEncoder", "boolean": "0_1"}[requested_encoding], "reason_code": "GUIDED_OVERRIDE"})
                else:
                    raise ValueError(f"PRE_INVALID_ENCODING_OVERRIDE: Unsupported encoding '{requested_encoding}' for '{col}'.")

        decisions.append({
            "column": col, "action": "use", "category": "feature", "data_type": data_type,
            "missing_ratio": missing_ratio, "unique": unique, "unique_ratio": unique_ratio,
            "missing_plan": missing_plan,
            "encoding": encoding.get("encoding") if encoding else None,
            "encoding_plan": encoding,
            "outlier_treatment": "iqr_cap" if data_type == "numeric" else None,
        })

    return decisions
