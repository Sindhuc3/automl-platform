from typing import Any, Dict, List

import pandas as pd


ID_NAME_HINTS = [
    "id",
    "_id",
    "identifier",
    "uuid",
    "guid",
    "serial",
    "index",
    "key",
    "code",
    "number",
    "no",
    "usn",
    "roll",
    "registration",
    "registration_no",
    "registration_number",
    "candidate",
    "applicant",
]


CONTACT_NAME_HINTS = [
    "email",
    "e-mail",
    "mail",
    "mobile",
    "phone",
    "telephone",
    "contact",
]


def _normalise_name(column_name: str) -> str:
    return (
        column_name
        .lower()
        .strip()
        .replace("-", "_")
        .replace(" ", "_")
    )


def _name_contains_hint(
    column_name: str,
    hints: List[str],
) -> bool:

    name = _normalise_name(column_name)

    tokens = {
        token
        for token in name.split("_")
        if token
    }

    for hint in hints:

        normalized_hint = (
            hint
            .lower()
            .strip()
            .replace("-", "_")
            .replace(" ", "_")
        )

        if normalized_hint in tokens:
            return True

        if name == normalized_hint:
            return True

        if name.startswith(
            f"{normalized_hint}_"
        ):
            return True

        if name.endswith(
            f"_{normalized_hint}"
        ):
            return True

        if f"_{normalized_hint}_" in name:
            return True

    return False


def _name_is_id_like(
    column_name: str,
) -> bool:

    return _name_contains_hint(
        column_name,
        ID_NAME_HINTS,
    )


def _name_is_contact_like(
    column_name: str,
) -> bool:

    return _name_contains_hint(
        column_name,
        CONTACT_NAME_HINTS,
    )


def _is_sequential_integer(
    series: pd.Series,
) -> bool:

    numeric = pd.to_numeric(
        series.dropna(),
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return False

    if not all(
        float(value).is_integer()
        for value in numeric
    ):
        return False

    values = sorted(
        set(
            numeric
            .astype(int)
            .tolist()
        )
    )

    if len(values) < 2:
        return False

    differences = [
        values[index + 1] - values[index]
        for index in range(
            len(values) - 1
        )
    ]

    return all(
        difference == 1
        for difference in differences
    )


def detect_id_candidates(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> List[Dict[str, Any]]:

    candidates = []

    for column in df.columns:

        column_name = str(column)

        # Never flag the target as an ID candidate.
        if (
            target_column
            and column_name == target_column
        ):
            continue

        series = df[column]

        non_null = series.dropna()

        if non_null.empty:
            continue

        unique_count = int(
            non_null.nunique(
                dropna=True
            )
        )

        total_count = len(non_null)

        uniqueness_ratio = (
            unique_count / total_count
            if total_count
            else 0
        )

        # Constant columns are handled separately.
        if unique_count <= 1:
            continue

        high_uniqueness = (
            uniqueness_ratio >= 0.95
        )

        name_id_signal = _name_is_id_like(
            column_name
        )

        contact_signal = _name_is_contact_like(
            column_name
        )

        sequential_signal = False

        if pd.api.types.is_numeric_dtype(
            series
        ):
            sequential_signal = (
                _is_sequential_integer(series)
            )

        # -----------------------------------------------------
        # Determine semantic role
        # -----------------------------------------------------

        role = None

        if contact_signal and high_uniqueness:
            role = "contact"

        elif name_id_signal and (
            high_uniqueness
            or sequential_signal
        ):
            role = "identifier"

        elif (
            high_uniqueness
            and sequential_signal
        ):
            role = "identifier"

        elif (
            high_uniqueness
            and pd.api.types.is_string_dtype(
                series
            )
        ):
            role = "identifier_like"

        if role is None:
            continue

        # -----------------------------------------------------
        # Signals
        # -----------------------------------------------------

        signals = []

        if high_uniqueness:
            signals.append(
                f"uniqueness ratio "
                f"{uniqueness_ratio:.2f}"
            )

        if sequential_signal:
            signals.append(
                "sequential integer pattern"
            )

        if name_id_signal:
            signals.append(
                "ID-like column name"
            )

        if contact_signal:
            signals.append(
                "contact-information column name"
            )

        candidates.append(
            {
                "column": column_name,
                "role": role,
                "uniqueness_ratio": round(
                    uniqueness_ratio,
                    4,
                ),
                "unique_values": unique_count,
                "signals": signals,
                "recommended_action": (
                    "exclude_by_default"
                ),
            }
        )

    return candidates