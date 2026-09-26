from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MISSING_TARGET_WARNING_PERCENT = 20.0
MISSING_TARGET_SEVERE_PERCENT = 50.0

HIGH_CARDINALITY_RATIO = 0.50
HIGH_CARDINALITY_ABSOLUTE = 50

MIN_VALID_ROWS = 2

# For classification, a class with fewer than this many
# observations is reported as a warning.
SMALL_CLASS_COUNT = 20


# ============================================================
# BASIC HELPERS
# ============================================================

def _is_missing(value: Any) -> bool:
    """
    Safely determine whether a value is missing.
    """
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _safe_value(value: Any) -> Any:
    """
    Convert NumPy/Pandas values into JSON-safe values.
    """
    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def _safe_dict(data: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Convert dictionary keys/values into JSON-safe values.
    """
    result = {}

    for key, value in data.items():
        result[str(_safe_value(key))] = _safe_value(value)

    return result


def _normalize_name(value: str) -> str:
    """
    Normalize a column name for semantic checks.
    """
    return (
        str(value)
        .strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
    )


# ============================================================
# SEMANTIC ROLE DETECTION
# ============================================================

def detect_target_semantic_role(
    target_column: str,
    df: pd.DataFrame,
    id_candidates: List[Dict[str, Any]],
    column_type: str,
) -> Dict[str, Any]:
    """
    Determine whether the selected target looks like:
    - normal target
    - identifier
    - contact information
    - free text
    - datetime
    - constant

    This function does NOT choose the target.
    The user has already selected it.
    """

    normalized_name = _normalize_name(target_column)

    # --------------------------------------------------------
    # CONSTANT
    # --------------------------------------------------------

    unique_count = int(
        df[target_column].dropna().nunique()
    )

    if unique_count <= 1:
        return {
            "role": "constant",
            "valid": False,
            "reason": (
                "The selected column contains only one "
                "unique value, so it cannot provide a "
                "meaningful prediction target."
            ),
        }

    # --------------------------------------------------------
    # IDENTIFIER / CONTACT FROM MODULE 1
    # --------------------------------------------------------

    for candidate in id_candidates:

        candidate_column = str(
            candidate.get("column", "")
        )

        if candidate_column != target_column:
            continue

        role = candidate.get(
            "role",
            "identifier_like"
        )

        if role == "contact":
            return {
                "role": "contact",
                "valid": False,
                "reason": (
                    "The selected column appears to contain "
                    "contact information rather than a prediction outcome."
                ),
            }

        if role in {
            "identifier",
            "identifier_like",
        }:
            return {
                "role": "identifier",
                "valid": False,
                "reason": (
                    "The selected column appears to be an "
                    "identifier rather than a meaningful prediction outcome."
                ),
            }

    # --------------------------------------------------------
    # NAME-BASED CONTACT DETECTION
    # --------------------------------------------------------

    contact_words = [
        "email",
        "e mail",
        "mail",
        "phone",
        "mobile",
        "telephone",
        "contact",
    ]

    if any(
        word in normalized_name
        for word in contact_words
    ):
        return {
            "role": "contact",
            "valid": False,
            "reason": (
                "The selected column appears to contain "
                "contact information and should not be used "
                "as a prediction target."
            ),
        }

    # --------------------------------------------------------
    # DATETIME
    # --------------------------------------------------------

    if column_type == "datetime":

        return {
            "role": "datetime",
            "valid": False,
            "reason": (
                "Datetime columns are not accepted as targets "
                "in the current supervised-learning version."
            ),
        }

    # --------------------------------------------------------
    # FREE TEXT
    # --------------------------------------------------------

    if column_type == "text":

        return {
            "role": "free_text",
            "valid": False,
            "reason": (
                "The selected column appears to contain free text. "
                "Free-text prediction targets are outside the scope "
                "of the current tabular AutoML pipeline."
            ),
        }

    # --------------------------------------------------------
    # OTHERWISE NORMAL TARGET
    # --------------------------------------------------------

    return {
        "role": "target",
        "valid": True,
        "reason": (
            "The selected column has a usable target-like structure."
        ),
    }


# ============================================================
# PROBLEM TYPE DETECTION
# ============================================================

def detect_problem_type(
    series: pd.Series,
    column_type: str,
) -> Dict[str, Any]:
    """
    Determine the ML problem type AFTER the user has selected
    a valid target.

    Possible results:
    - binary classification
    - multiclass classification
    - regression
    - ambiguous
    """

    valid_series = series.dropna()

    unique_values = valid_series.nunique()

    if unique_values < 2:

        return {
            "problem_type": None,
            "sub_problem_type": None,
            "confidence": "high",
            "valid": False,
            "reason": (
                "The target has fewer than two unique values."
            ),
        }

    # --------------------------------------------------------
    # BOOLEAN
    # --------------------------------------------------------

    if (
        pd.api.types.is_bool_dtype(
            valid_series
        )
        or column_type == "boolean"
    ):

        return {
            "problem_type": "classification",
            "sub_problem_type": "binary",
            "confidence": "high",
            "valid": True,
            "reason": (
                "The target contains boolean values."
            ),
        }

    # --------------------------------------------------------
    # EXACTLY TWO VALUES
    # --------------------------------------------------------

    if unique_values == 2:

        return {
            "problem_type": "classification",
            "sub_problem_type": "binary",
            "confidence": "high",
            "valid": True,
            "reason": (
                "The target contains exactly two unique values."
            ),
        }

    # --------------------------------------------------------
    # DATETIME / TEXT
    # --------------------------------------------------------

    if column_type in {
        "datetime",
        "text",
    }:

        return {
            "problem_type": None,
            "sub_problem_type": None,
            "confidence": "high",
            "valid": False,
            "reason": (
                "This target type is not supported "
                "for the current tabular pipeline."
            ),
        }

    # --------------------------------------------------------
    # CATEGORICAL
    # --------------------------------------------------------

    if column_type in {
        "categorical",
        "numeric_categorical",
    }:

        return {
            "problem_type": "classification",
            "sub_problem_type": "multiclass",
            "confidence": "high",
            "valid": True,
            "reason": (
                "The target is categorical or represents "
                "a small set of discrete categories."
            ),
        }

    # --------------------------------------------------------
    # NUMERIC
    # --------------------------------------------------------

    if (
        pd.api.types.is_numeric_dtype(
            valid_series
        )
        or column_type == "numeric"
    ):

        # Integer-like values with relatively few categories
        # are treated as classification.
        try:

            numeric_values = pd.to_numeric(
                valid_series,
                errors="coerce"
            )

            integer_like = bool(
                np.all(
                    np.isclose(
                        numeric_values.dropna()
                        .astype(float),
                        np.round(
                            numeric_values.dropna()
                            .astype(float)
                        )
                    )
                )
            )

        except Exception:

            integer_like = False

        unique_ratio = (
            unique_values /
            max(len(valid_series), 1)
        )

        if (
            integer_like
            and unique_values <= 20
            and unique_ratio <= 0.20
        ):

            return {
                "problem_type": "classification",
                "sub_problem_type": "multiclass",
                "confidence": "medium",
                "valid": True,
                "reason": (
                    "The numeric target contains a small "
                    "number of discrete integer-like values."
                ),
            }

        # Otherwise treat a continuous numeric target
        # as regression.
        return {
            "problem_type": "regression",
            "sub_problem_type": "continuous",
            "confidence": "high",
            "valid": True,
            "reason": (
                "The target contains continuous numeric values."
            ),
        }

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    return {
        "problem_type": None,
        "sub_problem_type": None,
        "confidence": "low",
        "valid": False,
        "reason": (
            "The system could not determine a supported "
            "problem type for this target."
        ),
    }


# ============================================================
# TARGET STATISTICS
# ============================================================

def calculate_target_statistics(
    series: pd.Series,
) -> Dict[str, Any]:

    total_rows = len(series)

    missing_count = int(
        series.isna().sum()
    )

    valid_row_count = int(
        total_rows - missing_count
    )

    unique_count = int(
        series.dropna().nunique()
    )

    missing_percentage = (
        (missing_count / total_rows) * 100
        if total_rows > 0
        else 0.0
    )

    unique_ratio = (
        unique_count / valid_row_count
        if valid_row_count > 0
        else 0.0
    )

    return {
        "total_row_count": total_rows,
        "valid_row_count": valid_row_count,
        "missing_count": missing_count,
        "missing_percentage": round(
            missing_percentage,
            2
        ),
        "unique_count": unique_count,
        "unique_ratio": round(
            unique_ratio,
            4
        ),
    }


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def calculate_class_distribution(
    series: pd.Series,
) -> Dict[str, Any]:

    valid_series = series.dropna()

    counts = valid_series.value_counts(
        dropna=False
    )

    total = len(valid_series)

    distribution = {}

    for value, count in counts.items():

        percentage = (
            (int(count) / total) * 100
            if total > 0
            else 0
        )

        distribution[str(
            _safe_value(value)
        )] = {
            "count": int(count),
            "percentage": round(
                percentage,
                2
            ),
        }

    imbalance_ratio = None

    if len(counts) >= 2:

        largest = int(
            counts.iloc[0]
        )

        smallest = int(
            counts.iloc[-1]
        )

        if smallest > 0:
            imbalance_ratio = round(
                largest / smallest,
                3
            )

    return {
        "classes": distribution,
        "class_count": int(
            len(counts)
        ),
        "imbalance_ratio": imbalance_ratio,
    }


# ============================================================
# TARGET VALIDATION
# ============================================================

def validate_target(
    df: pd.DataFrame,
    target_column: str,
    column_types: Dict[str, str],
    id_candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Main Module 2 function.

    The user MUST provide target_column.

    This function does not recommend or select a target.
    """

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    information: List[Dict[str, Any]] = []

    # ========================================================
    # 1. TARGET EXISTS
    # ========================================================

    if target_column not in df.columns:

        errors.append({
            "code": "TARGET_NOT_FOUND",
            "message": (
                f"Target column '{target_column}' "
                "does not exist in the dataset."
            ),
        })

        return {
            "status": "invalid",
            "target_column": target_column,
            "errors": errors,
            "warnings": warnings,
            "information": information,
        }

    series = df[target_column]

    column_type = column_types.get(
        target_column,
        "unknown"
    )

    # ========================================================
    # 2. TARGET STATISTICS
    # ========================================================

    statistics = calculate_target_statistics(
        series
    )

    information.append({
        "code": "TARGET_SELECTED",
        "message": (
            f"User selected '{target_column}' as the target."
        ),
    })

    information.append({
        "code": "TARGET_ROWS",
        "message": (
            f"{statistics['valid_row_count']} valid "
            f"target values out of "
            f"{statistics['total_row_count']} rows."
        ),
    })

    # ========================================================
    # 3. COMPLETELY MISSING
    # ========================================================

    if statistics["valid_row_count"] == 0:

        errors.append({
            "code": "TARGET_ALL_MISSING",
            "message": (
                "The selected target contains no valid values."
            ),
        })

        return {
            "status": "invalid",
            "target_column": target_column,
            "target_data_type": str(
                series.dtype
            ),
            "target_semantic_role": "target",
            "target_statistics": statistics,
            "errors": errors,
            "warnings": warnings,
            "information": information,
        }

    # ========================================================
    # 4. CONSTANT TARGET
    # ========================================================

    if statistics["unique_count"] <= 1:

        errors.append({
            "code": "CONSTANT_TARGET",
            "message": (
                "The selected target contains only one "
                "unique value and cannot define a useful "
                "supervised-learning problem."
            ),
        })

        return {
            "status": "invalid",
            "target_column": target_column,
            "target_data_type": str(
                series.dtype
            ),
            "target_semantic_role": "constant",
            "target_statistics": statistics,
            "errors": errors,
            "warnings": warnings,
            "information": information,
        }

    # ========================================================
    # 5. MISSING TARGET WARNING
    # ========================================================

    missing_percentage = (
        statistics["missing_percentage"]
    )

    if (
        missing_percentage >=
        MISSING_TARGET_SEVERE_PERCENT
    ):

        warnings.append({
            "code": "SEVERE_TARGET_MISSINGNESS",
            "message": (
                f"{missing_percentage}% of target values "
                "are missing. These rows cannot be used "
                "for supervised training."
            ),
        })

    elif (
        missing_percentage >=
        MISSING_TARGET_WARNING_PERCENT
    ):

        warnings.append({
            "code": "TARGET_MISSINGNESS",
            "message": (
                f"{missing_percentage}% of target values "
                "are missing. Rows with missing targets "
                "cannot be used for supervised training."
            ),
        })

    # ========================================================
    # 6. SEMANTIC ROLE
    # ========================================================

    semantic = detect_target_semantic_role(
        target_column=target_column,
        df=df,
        id_candidates=id_candidates,
        column_type=column_type,
    )

    semantic_role = semantic["role"]

    if not semantic["valid"]:

        errors.append({
            "code": "INVALID_TARGET_ROLE",
            "message": semantic["reason"],
        })

        return {
            "status": "invalid",
            "target_column": target_column,
            "target_data_type": str(
                series.dtype
            ),
            "target_semantic_role": semantic_role,
            "target_statistics": statistics,
            "errors": errors,
            "warnings": warnings,
            "information": information,
        }

    # ========================================================
    # 7. PROBLEM TYPE
    # ========================================================

    problem = detect_problem_type(
        series=series,
        column_type=column_type,
    )

    if not problem["valid"]:

        errors.append({
            "code": "UNSUPPORTED_TARGET_TYPE",
            "message": problem["reason"],
        })

        return {
            "status": "invalid",
            "target_column": target_column,
            "target_data_type": str(
                series.dtype
            ),
            "target_semantic_role": semantic_role,
            "target_statistics": statistics,
            "errors": errors,
            "warnings": warnings,
            "information": information,
        }

    problem_type = problem[
        "problem_type"
    ]

    sub_problem_type = problem[
        "sub_problem_type"
    ]

    information.append({
        "code": "PROBLEM_TYPE_DETECTED",
        "message": (
            f"Detected {sub_problem_type} "
            f"{problem_type}."
        ),
    })

    # ========================================================
    # 8. CLASSIFICATION ANALYSIS
    # ========================================================

    class_distribution = None

    if problem_type == "classification":

        class_distribution = (
            calculate_class_distribution(
                series
            )
        )

        class_count = (
            class_distribution[
                "class_count"
            ]
        )

        if class_count < 2:

            errors.append({
                "code": "INSUFFICIENT_CLASSES",
                "message": (
                    "Classification requires at least "
                    "two target classes."
                ),
            })

        imbalance_ratio = (
            class_distribution[
                "imbalance_ratio"
            ]
        )

        if (
            imbalance_ratio is not None
            and imbalance_ratio >= 4
        ):

            warnings.append({
                "code": "CLASS_IMBALANCE",
                "message": (
                    f"Class imbalance detected. "
                    f"Majority/minority ratio is "
                    f"{imbalance_ratio}:1."
                ),
            })

        # Report very small classes.
        for class_name, class_data in (
            class_distribution[
                "classes"
            ].items()
        ):

            if (
                class_data["count"]
                < SMALL_CLASS_COUNT
            ):

                warnings.append({
                    "code": "SMALL_CLASS",
                    "message": (
                        f"Class '{class_name}' contains "
                        f"only {class_data['count']} "
                        "observations."
                    ),
                })

    # ========================================================
    # 9. HIGH CARDINALITY
    # ========================================================

    unique_ratio = (
        statistics["unique_ratio"]
    )

    if (
        unique_ratio >= HIGH_CARDINALITY_RATIO
        and statistics["unique_count"]
        >= HIGH_CARDINALITY_ABSOLUTE
    ):

        warnings.append({
            "code": "HIGH_TARGET_CARDINALITY",
            "message": (
                "The selected target has many unique "
                "values relative to the number of rows. "
                "Review whether this represents a meaningful "
                "prediction outcome."
            ),
        })

    # ========================================================
    # 10. VALID ROW COUNT
    # ========================================================

    if (
        statistics["valid_row_count"]
        < MIN_VALID_ROWS
    ):

        errors.append({
            "code": "INSUFFICIENT_TARGET_ROWS",
            "message": (
                "There are not enough valid target rows "
                "to define a supervised-learning problem."
            ),
        })

    # ========================================================
    # 11. FINAL STATUS
    # ========================================================

    if errors:

        status = "invalid"

    elif warnings:

        status = "valid_with_warnings"

    else:

        status = "valid"

    # ========================================================
    # 12. RESULT
    # ========================================================

    result = {
        "status": status,

        "target_column": target_column,

        "target_data_type": str(
            series.dtype
        ),

        "target_semantic_role": semantic_role,

        "target_statistics": statistics,

        "problem_definition": {
            "task": (
                f"{sub_problem_type}_"
                f"{problem_type}"
            ),
            "target": target_column,
        },

        "problem_type": problem_type,

        "sub_problem_type": sub_problem_type,

        "detection_confidence": problem[
            "confidence"
        ],

        "class_distribution": class_distribution,

        "errors": errors,

        "warnings": warnings,

        "information": information,
    }

    return result