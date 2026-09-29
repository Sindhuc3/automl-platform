from typing import Any, Dict

REQUIRED_DATASET_FIELDS = [
    "dataset_id",
    "validated_storage_path",
    "column_names",
]

REQUIRED_TARGET_FIELDS = [
    "target_column",
    "target_validation",
    "problem_definition",
]


def validate_contract(dataset: Dict[str, Any]) -> Dict[str, Any]:
    missing = [k for k in REQUIRED_DATASET_FIELDS if not dataset.get(k)]
    target = dataset.get("target_validation") or {}
    missing.extend(k for k in REQUIRED_TARGET_FIELDS if not dataset.get(k))

    status = target.get("status")
    if status not in {"valid", "valid_with_warnings"}:
        missing.append("target_validation.status=valid")

    if missing:
        raise ValueError(
            "PRE_CONTRACT_MISSING_FIELD: " + ", ".join(missing)
        )

    return {
        "dataset_id": dataset["dataset_id"],
        "validated_storage_path": dataset["validated_storage_path"],
        "target_column": dataset["target_column"],
        "target_validation": target,
        "problem_definition": dataset["problem_definition"],
    }
