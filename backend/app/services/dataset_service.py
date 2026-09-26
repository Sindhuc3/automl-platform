import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import UploadFile

from app.config import (
    MAX_FILE_SIZE_BYTES,
    STORAGE_DIR,
    SUPPORTED_EXTENSIONS,
)
from app.database.mongodb import datasets_collection
from app.ingestion.id_detection import (
    detect_id_candidates,
)
from app.ingestion.loader import (
    DatasetLoadError,
    load_dataset,
)
from app.ingestion.profiler import (
    profile_dataset,
)
from app.ingestion.quality_report import (
    build_quality_report,
)
from app.ingestion.type_detection import (
    detect_column_types,
)
from app.ingestion.validators import (
    clean_column_names,
    validate_structure,
)


BASE_STORAGE_PATH = Path(__file__).resolve().parent.parent.parent / STORAGE_DIR

DATASETS_PATH = BASE_STORAGE_PATH


def _utc_now():
    return datetime.now(
        timezone.utc
    )


def _safe_filename(filename: str) -> str:

    filename = Path(filename).name

    filename = re.sub(
        r"[^A-Za-z0-9._-]",
        "_",
        filename
    )

    return filename


async def _save_uploaded_file(
    upload_file: UploadFile,
    destination: Path,
) -> int:

    total_size = 0

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with destination.open(
        "wb"
    ) as output_file:

        while True:

            chunk = await upload_file.read(
                1024 * 1024
            )

            if not chunk:
                break

            total_size += len(chunk)

            if total_size > MAX_FILE_SIZE_BYTES:

                output_file.close()

                if destination.exists():
                    destination.unlink()

                raise ValueError(
                    (
                        "File is too large. "
                        f"Maximum allowed size is "
                        f"{MAX_FILE_SIZE_BYTES / (1024 * 1024):.0f} MB."
                    )
                )

            output_file.write(chunk)

    return total_size


def _create_dataset_id() -> str:

    timestamp = datetime.now().strftime(
        "%Y%m%d"
    )

    short_uuid = uuid.uuid4().hex[:8].upper()

    return f"DS-{timestamp}-{short_uuid}"


def _serialize_dataframe_for_storage(
    df: pd.DataFrame,
    destination: Path,
):
    """
    Store the validated/cleaned version separately
    from the immutable original upload.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        destination,
        index=False
    )


def _json_safe_document(document: Dict[str, Any]):
    """
    Ensure nested values are JSON serializable.
    """

    return json.loads(
        json.dumps(
            document,
            default=str
        )
    )


async def process_uploaded_dataset(
    upload_file: UploadFile,
) -> Dict[str, Any]:

    original_filename = (
        upload_file.filename
        or "dataset"
    )

    safe_filename = _safe_filename(
        original_filename
    )

    extension = Path(
        safe_filename
    ).suffix.lower()

    # ------------------------------------------
    # 1. File validation
    # ------------------------------------------

    if extension not in SUPPORTED_EXTENSIONS:

        raise ValueError(
            (
                "Unsupported file format. "
                "Please upload a CSV or XLSX file."
            )
        )

    dataset_id = _create_dataset_id()

    dataset_folder = (
        DATASETS_PATH / dataset_id
    )

    dataset_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    raw_filename = (
        f"raw_{safe_filename}"
    )

    raw_path = (
        dataset_folder / raw_filename
    )

    # ------------------------------------------
    # 2. Save original file
    # ------------------------------------------

    file_size = await _save_uploaded_file(
        upload_file,
        raw_path
    )

    if file_size == 0:

        if raw_path.exists():
            raw_path.unlink()

        raise ValueError(
            "The uploaded file is empty."
        )

    # ------------------------------------------
    # 3. Load dataset
    # ------------------------------------------

    try:

        df, loading_metadata = load_dataset(
            str(raw_path)
        )

    except DatasetLoadError:

        # Preserve the original upload even
        # when loading fails.
        raise

    # ------------------------------------------
    # 4. Structural validation
    # ------------------------------------------

    validation_result = (
        validate_structure(df)
    )

    # If there are blocking structural errors,
    # return the information without profiling.
    if validation_result["errors"]:

        quality_report = build_quality_report(
            df=df,
            validation_result=validation_result,
            profile={
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "memory": {
                    "bytes": int(
                        df.memory_usage(
                            deep=True
                        ).sum()
                    ),
                    "mb": 0,
                },
                "column_names": [
                    str(column)
                    for column in df.columns
                ],
                "missing_values": {},
                "duplicate_rows": {
                    "count": 0,
                    "ratio": 0,
                    "percentage": 0,
                },
                "constant_columns": [],
                "unique_values": {},
                "descriptive_statistics": {},
                "preview": [],
            },
            column_types={},
            id_candidates=[],
            loading_metadata=loading_metadata,
        )

        raise ValueError(
            json.dumps(
                {
                    "message": (
                        "Dataset validation failed."
                    ),
                    "dataset_id": dataset_id,
                    "quality_report": quality_report,
                }
            )
        )

    # ------------------------------------------
    # 5. Clean column names
    # ------------------------------------------

    cleaned_df, cleaned_names = (
        clean_column_names(df)
    )

    # ------------------------------------------
    # 6. Detect column types
    # ------------------------------------------

    column_types = detect_column_types(
        cleaned_df
    )

    # ------------------------------------------
    # 7. Detect possible ID columns
    # ------------------------------------------

    id_candidates = detect_id_candidates(
        cleaned_df
    )

    # ------------------------------------------
    # 8. Profile dataset
    # ------------------------------------------

    profile = profile_dataset(
        cleaned_df,
        column_types,
        id_candidates
    )


    # ------------------------------------------
    # 9. Build unified Quality Report
    # ------------------------------------------

    quality_report = build_quality_report(
        df=cleaned_df,
        validation_result=validation_result,
        profile=profile,
        column_types=column_types,
        id_candidates=id_candidates,
        loading_metadata=loading_metadata,
    )

    # ------------------------------------------
    # 10. Store validated version separately
    # ------------------------------------------

    validated_filename = (
        "validated_dataset.csv"
    )

    validated_path = (
        dataset_folder /
        validated_filename
    )

    _serialize_dataframe_for_storage(
        cleaned_df,
        validated_path
    )

    # ------------------------------------------
    # 11. Store metadata in MongoDB
    # ------------------------------------------

    created_at = _utc_now()

    dataset_document = {
        "dataset_id": dataset_id,
        "original_filename": original_filename,
        "stored_filename": raw_filename,
        "validated_filename": validated_filename,
        "file_type": extension.replace(
            ".",
            ""
        ).upper(),
        "file_size": file_size,
        "rows": int(cleaned_df.shape[0]),
        "columns": int(cleaned_df.shape[1]),
        "column_names": cleaned_names,
        "quality_status": quality_report[
            "status"
        ],
        "quality_report": quality_report,
        "target_column": None,
        "created_at": created_at,
        "storage_path": str(
            raw_path
        ),
        "validated_storage_path": str(
            validated_path
        ),
    }

    dataset_document = (
        _json_safe_document(
            dataset_document
        )
    )

    datasets_collection.insert_one(
        dataset_document
    )

    # ------------------------------------------
    # 12. Return response
    # ------------------------------------------

    return {
        "message": (
            "Dataset uploaded and analyzed successfully."
        ),
        "dataset_id": dataset_id,
        "dataset": {
            "dataset_id": dataset_id,
            "original_filename": original_filename,
            "file_type": extension.replace(
                ".",
                ""
            ).upper(),
            "file_size": file_size,
            "rows": int(
                cleaned_df.shape[0]
            ),
            "columns": int(
                cleaned_df.shape[1]
            ),
            "column_names": cleaned_names,
            "quality_status": quality_report[
                "status"
            ],
            "quality_report": quality_report,
            "created_at": created_at.isoformat(),
        },
    }


def get_dataset(dataset_id: str):

    document = datasets_collection.find_one(
        {
            "dataset_id": dataset_id
        },
        {
            "_id": 0
        }
    )

    return document


def get_quality_report(
    dataset_id: str,
):

    document = datasets_collection.find_one(
        {
            "dataset_id": dataset_id
        },
        {
            "_id": 0,
            "quality_report": 1,
            "dataset_id": 1,
        }
    )

    return document


def list_datasets(
    limit: int = 50,
):

    documents = (
        datasets_collection
        .find(
            {},
            {
                "_id": 0,
                "dataset_id": 1,
                "original_filename": 1,
                "file_type": 1,
                "file_size": 1,
                "rows": 1,
                "columns": 1,
                "quality_status": 1,
                "created_at": 1,
                "target_column": 1,
            },
        )
        .sort(
            "created_at",
            -1
        )
        .limit(limit)
    )

    return list(documents)


def set_target_column(
    dataset_id: str,
    target_column: str,
):

    document = get_dataset(
        dataset_id
    )

    if not document:
        raise ValueError(
            "Dataset not found."
        )

    column_names = document.get(
        "column_names",
        []
    )

    if target_column not in column_names:

        raise ValueError(
            (
                f"Target column '{target_column}' "
                "does not exist in the dataset."
            )
        )

    datasets_collection.update_one(
        {
            "dataset_id": dataset_id
        },
        {
            "$set": {
                "target_column": target_column
            }
        }
    )

    return {
        "message": "Target column saved successfully.",
        "dataset_id": dataset_id,
        "target_column": target_column,
    }