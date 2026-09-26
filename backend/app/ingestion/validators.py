from typing import Any, Dict, List

import pandas as pd


def create_issue(
    severity: str,
    code: str,
    message: str,
    column: str | None = None,
    suggested_action: str | None = None,
) -> Dict[str, Any]:

    return {
        "severity": severity,
        "code": code,
        "message": message,
        "column": column,
        "suggested_action": suggested_action,
    }


def validate_structure(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """
    Perform structural validation before profiling.
    """

    errors = []
    warnings = []
    information = []

    # No rows
    if df.shape[0] == 0:
        errors.append(
            create_issue(
                "error",
                "NO_ROWS",
                "The dataset contains no data rows.",
                suggested_action="Upload a dataset containing at least one data row.",
            )
        )

    # No columns
    if df.shape[1] == 0:
        errors.append(
            create_issue(
                "error",
                "NO_COLUMNS",
                "The dataset contains no columns.",
                suggested_action="Upload a tabular dataset containing columns.",
            )
        )

    # One column
    if df.shape[1] == 1:
        errors.append(
            create_issue(
                "error",
                "SINGLE_COLUMN_DATASET",
                "The dataset contains only one column. A feature/target split is not possible.",
                suggested_action="Check the delimiter or upload a dataset with multiple columns.",
            )
        )

    # Check duplicate column names.
    column_names = [str(column) for column in df.columns]

    exact_duplicates = []
    seen = set()

    for column in column_names:
        if column in seen:
            exact_duplicates.append(column)
        else:
            seen.add(column)

    if exact_duplicates:
        warnings.append(
            create_issue(
                "warning",
                "DUPLICATE_COLUMN_NAMES",
                (
                    "Duplicate column names were detected. "
                    "They will be automatically renamed to unique names."
                ),
                suggested_action="Review the renamed columns before modeling.",
            )
        )

    # Check names that differ only by case/whitespace.
    normalized = {}

    for column in column_names:
        normalized_name = " ".join(column.strip().lower().split())

        normalized.setdefault(
            normalized_name,
            []
        ).append(column)

    for normalized_name, original_names in normalized.items():

        if len(original_names) > 1:

            warnings.append(
                create_issue(
                    "warning",
                    "SIMILAR_COLUMN_NAMES",
                    (
                        f"Columns {original_names} have the same normalized name "
                        f"'{normalized_name}'."
                    ),
                    suggested_action="Review the automatically cleaned column names.",
                )
            )

    # Empty / invalid column names
    for column in column_names:

        if not column.strip():

            warnings.append(
                create_issue(
                    "warning",
                    "EMPTY_COLUMN_NAME",
                    "A column has an empty name.",
                    column=column,
                    suggested_action="A generated column name will be assigned.",
                )
            )

    # General information
    information.append(
        create_issue(
            "info",
            "DATASET_SHAPE",
            f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.",
        )
    )

    return {
        "errors": errors,
        "warnings": warnings,
        "information": information,
    }


def clean_column_names(df: pd.DataFrame):
    """
    Clean column names and make them unique.

    The original uploaded file is never modified.
    """

    cleaned_names = []
    used_names = set()

    for index, original_name in enumerate(df.columns):

        name = str(original_name).strip()

        if not name:
            name = f"Column_{index + 1}"

        # Normalize repeated whitespace.
        name = " ".join(name.split())

        base_name = name
        counter = 1

        while name.lower() in used_names:
            name = f"{base_name}_{counter}"
            counter += 1

        used_names.add(name.lower())
        cleaned_names.append(name)

    cleaned_df = df.copy()
    cleaned_df.columns = cleaned_names

    return cleaned_df, cleaned_names