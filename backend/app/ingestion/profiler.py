from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _safe_value(value):
    """
    Convert NumPy/Pandas values into JSON-safe values.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        return float(value)

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    return value


def scan_missing_values(
    df: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:

    result = {}

    for column in df.columns:

        missing_count = int(
            df[column].isna().sum()
        )

        total = len(df)

        percentage = (
            (missing_count / total) * 100
            if total > 0
            else 0
        )

        result[str(column)] = {
            "count": missing_count,
            "percentage": round(
                percentage,
                2,
            ),
        }

    return result


def scan_duplicates(
    df: pd.DataFrame,
) -> Dict[str, Any]:

    duplicate_count = int(
        df.duplicated().sum()
    )

    total_rows = len(df)

    duplicate_ratio = (
        duplicate_count / total_rows
        if total_rows > 0
        else 0
    )

    return {
        "count": duplicate_count,
        "ratio": round(
            duplicate_ratio,
            4,
        ),
        "percentage": round(
            duplicate_ratio * 100,
            2,
        ),
    }


def detect_constant_columns(
    df: pd.DataFrame,
) -> List[str]:

    constant_columns = []

    for column in df.columns:

        unique_count = int(
            df[column].nunique(
                dropna=False
            )
        )

        if unique_count <= 1:
            constant_columns.append(
                str(column)
            )

    return constant_columns


def _categorical_distribution(
    series: pd.Series,
    limit: int = 5,
) -> List[Dict[str, Any]]:

    value_counts = (
        series
        .dropna()
        .astype(str)
        .value_counts()
        .head(limit)
    )

    total = int(
        series.dropna().shape[0]
    )

    result = []

    for value, count in value_counts.items():

        percentage = (
            (int(count) / total) * 100
            if total > 0
            else 0
        )

        result.append(
            {
                "value": str(value),
                "count": int(count),
                "percentage": round(
                    percentage,
                    2,
                ),
            }
        )

    return result


def _datetime_statistics(
    series: pd.Series,
) -> Dict[str, Any]:

    parsed = pd.to_datetime(
        series,
        errors="coerce",
    ).dropna()

    if parsed.empty:
        return {}

    return {
        "earliest": _safe_value(
            parsed.min()
        ),
        "latest": _safe_value(
            parsed.max()
        ),
    }


def _text_statistics(
    series: pd.Series,
) -> Dict[str, Any]:

    text_series = (
        series
        .dropna()
        .astype(str)
    )

    if text_series.empty:
        return {}

    lengths = text_series.str.len()

    return {
        "average_length": round(
            float(lengths.mean()),
            2,
        ),
        "minimum_length": int(
            lengths.min()
        ),
        "maximum_length": int(
            lengths.max()
        ),
    }


def compute_descriptive_stats(
    df: pd.DataFrame,
    column_types: Dict[str, str],
    id_candidates: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:

    result = {}

    # ---------------------------------------------------------
    # Create semantic-role lookup
    # ---------------------------------------------------------

    id_roles = {
        item["column"]: item.get(
            "role",
            "identifier_like",
        )
        for item in id_candidates
    }

    # ---------------------------------------------------------
    # Detect constants once
    # ---------------------------------------------------------

    constant_columns = set(
        detect_constant_columns(df)
    )

    for column in df.columns:

        column_name = str(column)

        series = df[column]

        detected_type = column_types.get(
            column_name,
            "unknown",
        )

        unique_count = int(
            series.nunique(
                dropna=True
            )
        )

        stats = {
            "type": detected_type,
            "count": int(
                series.count()
            ),
            "unique": unique_count,
        }

        # -----------------------------------------------------
        # CONSTANT
        # -----------------------------------------------------

        if column_name in constant_columns:

            stats["role"] = "constant"

            stats["recommended_action"] = (
                "exclude_from_modeling"
            )

            stats["reason"] = (
                "Column contains no variation."
            )

            result[column_name] = stats

            continue

        # -----------------------------------------------------
        # IDENTIFIER / CONTACT
        # -----------------------------------------------------

        if column_name in id_roles:

            semantic_role = id_roles[
                column_name
            ]

            stats["role"] = semantic_role

            stats["recommended_action"] = (
                "exclude_from_modeling"
            )

            if semantic_role == "contact":
                stats["reason"] = (
                    "Column contains contact "
                    "information and should not "
                    "be used as a predictive feature."
                )

            else:
                stats["reason"] = (
                    "Column appears to be an "
                    "identifier and should not "
                    "be used as a predictive feature."
                )

            result[column_name] = stats

            continue

        # -----------------------------------------------------
        # NUMERICAL
        # -----------------------------------------------------

        if detected_type == "numeric":

            numeric_series = pd.to_numeric(
                series,
                errors="coerce",
            ).dropna()

            if not numeric_series.empty:

                stats["role"] = "feature"

                stats.update(
                    {
                        "mean": _safe_value(
                            numeric_series.mean()
                        ),
                        "std": _safe_value(
                            numeric_series.std()
                        ),
                        "min": _safe_value(
                            numeric_series.min()
                        ),
                        "median": _safe_value(
                            numeric_series.median()
                        ),
                        "max": _safe_value(
                            numeric_series.max()
                        ),
                    }
                )

            result[column_name] = stats

            continue

        # -----------------------------------------------------
        # NUMERICAL-CATEGORICAL
        # -----------------------------------------------------

        if detected_type == (
            "numeric_categorical"
        ):

            stats["role"] = (
                "categorical_feature"
            )

            stats["top_values"] = (
                _categorical_distribution(
                    series
                )
            )

            result[column_name] = stats

            continue

        # -----------------------------------------------------
        # CATEGORICAL
        # -----------------------------------------------------

        if detected_type == "categorical":

            stats["role"] = (
                "categorical_feature"
            )

            stats["top_values"] = (
                _categorical_distribution(
                    series
                )
            )

            result[column_name] = stats

            continue

        # -----------------------------------------------------
        # BOOLEAN
        # -----------------------------------------------------

        if detected_type == "boolean":

            stats["role"] = (
                "categorical_feature"
            )

            stats["value_distribution"] = (
                _categorical_distribution(
                    series,
                    limit=2,
                )
            )

            result[column_name] = stats

            continue

        # -----------------------------------------------------
        # DATETIME
        # -----------------------------------------------------

        if detected_type == "datetime":

            stats["role"] = (
                "datetime_feature"
            )

            stats.update(
                _datetime_statistics(
                    series
                )
            )

            result[column_name] = stats

            continue

        # -----------------------------------------------------
        # TEXT
        # -----------------------------------------------------

        if detected_type == "text":

            stats["role"] = "text_feature"

            stats.update(
                _text_statistics(
                    series
                )
            )

            result[column_name] = stats

            continue

        # -----------------------------------------------------
        # UNKNOWN
        # -----------------------------------------------------

        stats["role"] = "unknown"

        result[column_name] = stats

    return result


def get_preview(
    df: pd.DataFrame,
    number_of_rows: int = 10,
) -> List[Dict[str, Any]]:

    preview_df = df.head(
        number_of_rows
    ).copy()

    preview_df = preview_df.replace(
        {
            np.nan: None,
            pd.NaT: None,
        }
    )

    records = preview_df.to_dict(
        orient="records"
    )

    safe_records = []

    for record in records:

        safe_record = {}

        for key, value in record.items():

            safe_record[str(key)] = _safe_value(
                value
            )

        safe_records.append(
            safe_record
        )

    return safe_records


def profile_dataset(
    df: pd.DataFrame,
    column_types: Dict[str, str],
    id_candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:

    memory_bytes = int(
        df.memory_usage(
            deep=True
        ).sum()
    )

    memory_mb = (
        memory_bytes
        / (1024 * 1024)
    )

    missing = scan_missing_values(
        df
    )

    duplicate_info = scan_duplicates(
        df
    )

    constant_columns = (
        detect_constant_columns(df)
    )

    unique_values = {
        str(column): int(
            df[column].nunique(
                dropna=True
            )
        )
        for column in df.columns
    }

    descriptive_stats = (
        compute_descriptive_stats(
            df,
            column_types,
            id_candidates,
        )
    )

    preview = get_preview(df)

    return {
        "rows": int(
            df.shape[0]
        ),

        "columns": int(
            df.shape[1]
        ),

        "memory": {
            "bytes": memory_bytes,
            "mb": round(
                memory_mb,
                4,
            ),
        },

        "column_names": [
            str(column)
            for column in df.columns
        ],

        "missing_values": missing,

        "duplicate_rows": duplicate_info,

        "constant_columns": (
            constant_columns
        ),

        "unique_values": unique_values,

        "descriptive_statistics": (
            descriptive_stats
        ),

        "preview": preview,
    }