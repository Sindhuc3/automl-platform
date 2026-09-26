from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from app.config import STORAGE_DIR
from app.database.mongodb import datasets_collection
from app.ingestion.id_detection import detect_id_candidates
from app.ingestion.type_detection import detect_column_types
from app.target.target_validation import validate_target


# ============================================================
# STORAGE
# ============================================================

BASE_STORAGE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / STORAGE_DIR
)


# ============================================================
# HELPERS
# ============================================================

def _utc_now():
    return datetime.now(timezone.utc)


def _load_validated_dataset(
    document: Dict[str, Any],
) -> pd.DataFrame:

    validated_path = document.get("validated_storage_path")

    if not validated_path:
        raise ValueError(
            "Validated dataset path is not available."
        )

    path = Path(validated_path)

    if not path.exists():
        raise ValueError(
            "Validated dataset file could not be found."
        )

    try:
        return pd.read_csv(path)

    except Exception as error:
        raise ValueError(
            f"Unable to load validated dataset: {error}"
        )


# ============================================================
# TARGET VALIDATION
# ============================================================

def validate_dataset_target(
    dataset_id: str,
    target_column: str,
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # 1. Find dataset
    # --------------------------------------------------------

    document = datasets_collection.find_one(
        {
            "dataset_id": dataset_id
        },
        {
            "_id": 0
        }
    )

    if not document:
        raise ValueError(
            "Dataset not found."
        )

    # --------------------------------------------------------
    # 2. Load validated dataset
    # --------------------------------------------------------

    df = _load_validated_dataset(document)

    # --------------------------------------------------------
    # 3. Detect column types
    # --------------------------------------------------------

    column_types = detect_column_types(df)

    # --------------------------------------------------------
    # 4. Detect ID candidates
    # --------------------------------------------------------

    id_candidates = detect_id_candidates(df)

    # --------------------------------------------------------
    # 5. Validate selected target
    # --------------------------------------------------------

    result = validate_target(
        df=df,
        target_column=target_column,
        column_types=column_types,
        id_candidates=id_candidates,
    )

    # ========================================================
    # IMPORTANT:
    # If target itself is already invalid,
    # DO NOT continue building a problem definition.
    # ========================================================

    if result["status"] == "invalid":

        # Remove any previously saved target information.
        datasets_collection.update_one(
            {
                "dataset_id": dataset_id
            },
            {
                "$unset": {
                    "target_column": "",
                    "problem_definition": "",
                    "target_validation": "",
                    "target_validation_updated_at": "",
                }
            }
        )

        return result

    # ========================================================
    # 6. Determine usable/excluded features
    # ========================================================

    excluded_features = []
    usable_features = []

    for column in df.columns:

        # Target itself is not a feature.
        if column == target_column:
            continue

        column_type = column_types.get(
            column,
            "unknown"
        )

        # ----------------------------------------------------
        # Constant columns
        # ----------------------------------------------------

        if column_type == "constant":

            excluded_features.append({
                "column": column,
                "reason": "constant",
            })

            continue

        # ----------------------------------------------------
        # ID/contact columns
        # ----------------------------------------------------

        candidate = next(
            (
                item
                for item in id_candidates
                if item.get("column") == column
            ),
            None,
        )

        if candidate:

            role = candidate.get(
                "role",
                "identifier_like"
            )

            if role in {
                "identifier",
                "identifier_like",
                "contact",
            }:

                excluded_features.append({
                    "column": column,
                    "reason": role,
                })

                continue

        # ----------------------------------------------------
        # Datetime
        # ----------------------------------------------------

        if column_type == "datetime":

            excluded_features.append({
                "column": column,
                "reason": "datetime",
            })

            continue

        # ----------------------------------------------------
        # Free text
        # ----------------------------------------------------

        if column_type == "text":

            excluded_features.append({
                "column": column,
                "reason": "free_text",
            })

            continue

        # ----------------------------------------------------
        # Otherwise usable
        # ----------------------------------------------------

        usable_features.append(column)

    # ========================================================
    # 7. Store feature information
    # ========================================================

    result["usable_features"] = usable_features

    result["usable_feature_count"] = len(
        usable_features
    )

    result["excluded_features"] = excluded_features

    result["excluded_feature_count"] = len(
        excluded_features
    )

    # ========================================================
    # 8. No usable feature check
    # ========================================================

    if len(usable_features) == 0:

        result["status"] = "invalid"

        result["errors"].append({
            "code": "NO_USABLE_FEATURES",
            "message": (
                "The selected target is valid, but no "
                "usable feature columns remain for prediction."
            ),
        })

    elif len(usable_features) == 1:

        result["warnings"].append({
            "code": "ONE_USABLE_FEATURE",
            "message": (
                "Only one usable feature is available "
                "for the selected target."
            ),
        })

        result["status"] = "valid_with_warnings"

    # ========================================================
    # 9. Problem definition
    # ========================================================

    result["problem_definition"]["features"] = (
        usable_features
    )

    result["problem_definition"]["excluded_features"] = (
        excluded_features
    )

    # ========================================================
    # 10. Save ONLY if valid
    # ========================================================

    if result["status"] in {
        "valid",
        "valid_with_warnings",
    }:

        update_fields = {
            "target_column": target_column,

            "target_validation": result,

            "target_validation_updated_at": (
                _utc_now()
            ),

            "problem_definition": (
                result["problem_definition"]
            ),
        }

        datasets_collection.update_one(
            {
                "dataset_id": dataset_id
            },
            {
                "$set": update_fields
            }
        )

    else:

        # If feature validation made it invalid,
        # don't leave a selected target saved.
        datasets_collection.update_one(
            {
                "dataset_id": dataset_id
            },
            {
                "$unset": {
                    "target_column": "",
                    "problem_definition": "",
                    "target_validation": "",
                    "target_validation_updated_at": "",
                }
            }
        )

    return result


# ============================================================
# GET SAVED TARGET
# ============================================================

def get_dataset_target(
    dataset_id: str,
):

    document = datasets_collection.find_one(
        {
            "dataset_id": dataset_id
        },
        {
            "_id": 0,
            "dataset_id": 1,
            "target_column": 1,
            "target_validation": 1,
            "problem_definition": 1,
        }
    )

    if not document:

        raise ValueError(
            "Dataset not found."
        )

    return document