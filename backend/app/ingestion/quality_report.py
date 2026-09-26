from typing import Any, Dict, List, Optional


def _issue(
    severity: str,
    code: str,
    message: str,
    column: Optional[str] = None,
    suggested_action: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Creates one standardized quality-report issue.
    """

    return {
        "severity": severity,
        "code": code,
        "message": message,
        "column": column,
        "suggested_action": suggested_action,
    }


def build_quality_report(
    df,
    validation_result: Dict[str, List[Dict[str, Any]]],
    profile: Dict[str, Any],
    column_types: Dict[str, str],
    id_candidates: List[Dict[str, Any]],
    loading_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Builds the unified Dataset Quality Report.

    This report contains:
    - errors
    - warnings
    - information
    - loading information
    - column type detection
    - ID candidates
    - dataset profile
    """

    # ---------------------------------------------------------
    # 1. Get validation issues
    # ---------------------------------------------------------

    errors = list(
        validation_result.get("errors", [])
    )

    warnings = list(
        validation_result.get("warnings", [])
    )

    information = list(
        validation_result.get("information", [])
    )

    # ---------------------------------------------------------
    # 2. Loading information
    # ---------------------------------------------------------

    encoding = loading_metadata.get("encoding")

    delimiter = loading_metadata.get("delimiter")

    skipped_rows = loading_metadata.get(
        "skipped_rows",
        []
    )

    # Encoding information
    if encoding:
        information.append(
            _issue(
                severity="info",
                code="ENCODING_USED",
                message=(
                    f"Dataset loaded using encoding: {encoding}."
                ),
            )
        )

    # Delimiter information
    if delimiter:
        information.append(
            _issue(
                severity="info",
                code="DELIMITER_USED",
                message=(
                    f"CSV delimiter detected/used: {delimiter}."
                ),
            )
        )

    # Malformed rows
    if skipped_rows:
        warnings.append(
            _issue(
                severity="warning",
                code="MALFORMED_ROWS_SKIPPED",
                message=(
                    f"{len(skipped_rows)} malformed CSV row(s) "
                    "were skipped during loading."
                ),
                suggested_action=(
                    "Inspect the original file and correct "
                    "malformed rows if the skipped records "
                    "are important."
                ),
            )
        )

    # ---------------------------------------------------------
    # 3. Missing values
    # ---------------------------------------------------------

    missing_values = profile.get(
        "missing_values",
        {}
    )

    for column, details in missing_values.items():

        percentage = details.get(
            "percentage",
            0
        )

        count = details.get(
            "count",
            0
        )

        # Completely empty column
        if percentage >= 100:

            warnings.append(
                _issue(
                    severity="warning",
                    code="COMPLETELY_EMPTY_COLUMN",
                    message=(
                        f"Column '{column}' contains "
                        "100% missing values."
                    ),
                    column=column,
                    suggested_action=(
                        "This column will likely be removed "
                        "during preprocessing."
                    ),
                )
            )

        # More than 50% missing
        elif percentage > 50:

            warnings.append(
                _issue(
                    severity="warning",
                    code="HIGH_MISSING_VALUES",
                    message=(
                        f"Column '{column}' has "
                        f"{percentage}% missing values "
                        f"({count} missing cells)."
                    ),
                    column=column,
                    suggested_action=(
                        "Review whether this column should "
                        "be retained during preprocessing."
                    ),
                )
            )

        # Some missing values
        elif count > 0:

            information.append(
                _issue(
                    severity="info",
                    code="MISSING_VALUES",
                    message=(
                        f"Column '{column}' has "
                        f"{count} missing value(s) "
                        f"({percentage}%)."
                    ),
                    column=column,
                )
            )

    # ---------------------------------------------------------
    # 4. Constant columns
    # ---------------------------------------------------------

    constant_columns = profile.get(
        "constant_columns",
        []
    )

    for column in constant_columns:

        warnings.append(
            _issue(
                severity="warning",
                code="CONSTANT_COLUMN",
                message=(
                    f"Column '{column}' contains "
                    "only one unique value."
                ),
                column=column,
                suggested_action=(
                    "Exclude this column from modeling "
                    "because it contains no useful variation."
                ),
            )
        )

    # ---------------------------------------------------------
    # 5. Duplicate rows
    # ---------------------------------------------------------

    duplicate_rows = profile.get(
        "duplicate_rows",
        {}
    )

    duplicate_percentage = duplicate_rows.get(
        "percentage",
        0
    )

    duplicate_count = duplicate_rows.get(
        "count",
        0
    )

    if duplicate_count > 0:

        # High duplicate ratio
        if duplicate_percentage >= 5:

            warnings.append(
                _issue(
                    severity="warning",
                    code="HIGH_DUPLICATE_ROWS",
                    message=(
                        f"{duplicate_count} duplicate rows "
                        f"were detected "
                        f"({duplicate_percentage}%)."
                    ),
                    suggested_action=(
                        "Review duplicates because they may "
                        "affect model evaluation."
                    ),
                )
            )

        # Low duplicate ratio
        else:

            information.append(
                _issue(
                    severity="info",
                    code="DUPLICATE_ROWS",
                    message=(
                        f"{duplicate_count} duplicate row(s) "
                        f"detected "
                        f"({duplicate_percentage}%)."
                    ),
                )
            )

    # ---------------------------------------------------------
    # 6. ID column candidates
    # ---------------------------------------------------------

    for candidate in id_candidates:

        column = candidate.get(
            "column"
        )

        signals = candidate.get(
            "signals",
            []
        )

        # Convert signals into readable text
        if isinstance(signals, list):
            signal_text = ", ".join(
                str(signal)
                for signal in signals
            )
        else:
            signal_text = str(signals)

        warnings.append(
            _issue(
                severity="warning",
                code="ID_COLUMN_CANDIDATE",
                message=(
                    f"Column '{column}' appears to be "
                    f"an identifier ({signal_text})."
                ),
                column=column,
                suggested_action=(
                    "Exclude this column from the default "
                    "feature set. The column is not deleted "
                    "and can be reviewed later."
                ),
            )
        )

    # ---------------------------------------------------------
    # 7. Column type information
    # ---------------------------------------------------------

    for column, detected_type in column_types.items():

        information.append(
            _issue(
                severity="info",
                code="COLUMN_TYPE",
                message=(
                    f"Column '{column}' detected as "
                    f"'{detected_type}'."
                ),
                column=column,
            )
        )

    # ---------------------------------------------------------
    # 8. Overall dataset status
    # ---------------------------------------------------------

    if errors:

        status = "blocked"

    elif warnings:

        status = "usable_with_warnings"

    else:

        status = "clean"

    # ---------------------------------------------------------
    # 9. Recommended actions
    # ---------------------------------------------------------

    recommended_actions = []

    if warnings:

        recommended_actions.append(
            {
                "type": "review",
                "message": (
                    "Review warnings before continuing "
                    "to the machine-learning pipeline."
                ),
            }
        )

    if errors:

        recommended_actions.append(
            {
                "type": "fix",
                "message": (
                    "Fix the blocking errors before "
                    "continuing to the machine-learning pipeline."
                ),
            }
        )

    # ---------------------------------------------------------
    # 10. Final Quality Report
    # ---------------------------------------------------------

    return {
        "status": status,

        "blocking": bool(errors),

        "summary": {
            "errors": len(errors),
            "warnings": len(warnings),
            "information": len(information),
        },

        "errors": errors,

        "warnings": warnings,

        "information": information,

        "recommended_actions": recommended_actions,

        "loading": {
            "encoding": encoding,
            "delimiter": delimiter,
            "malformed_rows_skipped": len(
                skipped_rows
            ),
        },

        "column_types": column_types,

        "id_candidates": id_candidates,

        "profile": profile,
    }