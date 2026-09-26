from typing import Dict

import pandas as pd


BOOLEAN_VALUES = {
    "true",
    "false",
    "yes",
    "no",
    "y",
    "n",
    "0",
    "1",
}


DATE_NAME_HINTS = {
    "date",
    "time",
    "datetime",
    "timestamp",
    "dob",
    "created_at",
    "updated_at",
}


def _is_boolean_like(series: pd.Series) -> bool:
    non_null = series.dropna()

    if non_null.empty:
        return False

    if pd.api.types.is_bool_dtype(series):
        return True

    values = {
        str(value).strip().lower()
        for value in non_null.unique()
    }

    return len(values) == 2 and values.issubset(BOOLEAN_VALUES)


def _looks_like_datetime(
    series: pd.Series,
    column_name: str,
) -> bool:

    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    name = column_name.lower()

    name_has_date_hint = any(
        token in name
        for token in DATE_NAME_HINTS
    )

    if not name_has_date_hint:
        return False

    non_null = series.dropna()

    if non_null.empty:
        return False

    sample = non_null.astype(str).head(100)

    parsed = pd.to_datetime(
        sample,
        errors="coerce",
    )

    valid_ratio = parsed.notna().mean()

    return valid_ratio >= 0.8


def detect_column_type(
    series: pd.Series,
    column_name: str,
) -> str:

    non_null = series.dropna()

    if non_null.empty:
        return "unknown"

    unique_count = int(
        non_null.nunique(dropna=True)
    )

    total_count = len(non_null)

    unique_ratio = (
        unique_count / total_count
        if total_count
        else 0
    )

    # ---------------------------------------------------------
    # 1. Constant
    # ---------------------------------------------------------

    # Constant detection comes first.
    # A constant column should not be classified as
    # numeric-categorical or categorical.
    if unique_count <= 1:
        return "constant"

    # ---------------------------------------------------------
    # 2. Boolean
    # ---------------------------------------------------------

    if _is_boolean_like(series):
        return "boolean"

    # ---------------------------------------------------------
    # 3. Datetime
    # ---------------------------------------------------------

    if _looks_like_datetime(
        series,
        column_name,
    ):
        return "datetime"

    # ---------------------------------------------------------
    # 4. Numeric
    # ---------------------------------------------------------

    if pd.api.types.is_numeric_dtype(series):

        # A numeric column with a small number of
        # distinct values is treated as categorical.
        if (
            unique_count <= 30
            or unique_ratio < 0.05
        ):
            return "numeric_categorical"

        return "numeric"

    # ---------------------------------------------------------
    # 5. Pandas categorical dtype
    # ---------------------------------------------------------

    if isinstance(
        series.dtype,
        pd.CategoricalDtype,
    ):
        return "categorical"

    # ---------------------------------------------------------
    # 6. Strings / object
    # ---------------------------------------------------------

    if (
        pd.api.types.is_string_dtype(series)
        or series.dtype == object
    ):

        text_series = (
            non_null
            .astype(str)
        )

        average_length = float(
            text_series
            .str.len()
            .mean()
        )

        # Low-cardinality strings are categorical.
        if (
            unique_count <= 30
            or unique_ratio < 0.05
        ):
            return "categorical"

        # Longer high-cardinality strings are
        # more likely to represent free text.
        if average_length >= 30:
            return "text"

        return "categorical"

    # ---------------------------------------------------------
    # 7. Unknown
    # ---------------------------------------------------------

    return "unknown"


def detect_column_types(
    df: pd.DataFrame,
) -> Dict[str, str]:

    result = {}

    for column in df.columns:

        column_name = str(column)

        result[column_name] = detect_column_type(
            df[column],
            column_name,
        )

    return result