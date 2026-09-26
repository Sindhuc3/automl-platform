import json

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from app.database.mongodb import (
    check_database_connection,
)

from app.services.dataset_service import (
    get_dataset,
    get_quality_report,
    list_datasets,
    process_uploaded_dataset,
)


router = APIRouter(
    prefix="/api/datasets",
    tags=["Datasets"],
)


# ------------------------------------------
# Upload Dataset
# ------------------------------------------

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
):

    if not check_database_connection():

        raise HTTPException(
            status_code=503,
            detail=(
                "MongoDB is not available. "
                "Please start MongoDB and try again."
            ),
        )

    try:

        result = await process_uploaded_dataset(
            file
        )

        return result

    except ValueError as exc:

        # Some validation errors contain a
        # complete Quality Report.
        try:

            parsed = json.loads(
                str(exc)
            )

            if (
                isinstance(parsed, dict)
                and "quality_report" in parsed
            ):

                raise HTTPException(
                    status_code=400,
                    detail=parsed,
                )

        except json.JSONDecodeError:
            pass

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"An unexpected error occurred: {str(exc)}"
            ),
        )


# ------------------------------------------
# List Datasets
# ------------------------------------------

@router.get("")
def get_all_datasets():

    if not check_database_connection():

        raise HTTPException(
            status_code=503,
            detail="MongoDB is not available.",
        )

    return {
        "datasets": list_datasets()
    }


# ------------------------------------------
# Get Single Dataset
# ------------------------------------------

@router.get("/{dataset_id}")
def get_single_dataset(
    dataset_id: str,
):

    if not check_database_connection():

        raise HTTPException(
            status_code=503,
            detail="MongoDB is not available.",
        )

    dataset = get_dataset(
        dataset_id
    )

    if not dataset:

        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        )

    return dataset


# ------------------------------------------
# Get Quality Report
# ------------------------------------------

@router.get("/{dataset_id}/quality-report")
def get_dataset_quality_report(
    dataset_id: str,
):

    if not check_database_connection():

        raise HTTPException(
            status_code=503,
            detail="MongoDB is not available.",
        )

    report = get_quality_report(
        dataset_id
    )

    if not report:

        raise HTTPException(
            status_code=404,
            detail="Dataset not found.",
        )

    return report