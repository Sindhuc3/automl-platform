from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.target_service import (
    get_dataset_target,
    validate_dataset_target,
)


router = APIRouter(
    prefix="/api/datasets",
    tags=["Target Validation"],
)


class TargetSelectionRequest(BaseModel):
    target_column: str


@router.post("/{dataset_id}/target")
def validate_target_endpoint(
    dataset_id: str,
    request: TargetSelectionRequest
):
    try:
        target_column = request.target_column.strip()

        if not target_column:
            raise ValueError(
                "Please select a target column."
            )

        result = validate_dataset_target(
            dataset_id=dataset_id,
            target_column=target_column,
        )

        return {
            "message": "Target validation completed.",
            "dataset_id": dataset_id,
            "target": result,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Target validation failed: {error}"
        )


@router.get("/{dataset_id}/target")
def get_target_endpoint(dataset_id: str):

    try:

        return get_dataset_target(dataset_id)

    except ValueError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to retrieve target information: {error}"
        )