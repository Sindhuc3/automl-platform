
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.schemas.preprocessing_schema import PreprocessingRunRequest
from app.services.preprocessing_service import run_preprocessing, get_preprocessing

router = APIRouter(prefix="/api/datasets", tags=["Preprocessing"])


@router.post("/{dataset_id}/preprocessing")
def start(dataset_id: str, request: PreprocessingRunRequest):
    if request.mode not in {"automatic", "guided"}:
        raise HTTPException(400, "mode must be automatic or guided")

    try:
        result = run_preprocessing(
            dataset_id,
            request.mode,
            request.version,
            request.overrides,
        )
        return {
            "message": "Preprocessing completed successfully.",
            "preprocessing": result,
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Preprocessing failed: {e}")


@router.get("/{dataset_id}/preprocessing")
def listing(dataset_id: str):
    return {"preprocessing": get_preprocessing(dataset_id)}


@router.get("/{dataset_id}/preprocessing/{run_id}")
def read(dataset_id: str, run_id: str):
    result = get_preprocessing(dataset_id, run_id)
    if not result:
        raise HTTPException(404, "Preprocessing run not found.")
    return {"preprocessing": result}


@router.get("/{dataset_id}/preprocessing/{run_id}/download/{artifact}")
def download_artifact(dataset_id: str, run_id: str, artifact: str):
    allowed = {
        "report": "report",
        "config": "config",
        "feature_mapping": "feature_mapping",
        "processed_train_plain": "processed_train_plain",
        "processed_test_plain": "processed_test_plain",
        "processed_train_scaled": "processed_train_scaled",
        "processed_test_scaled": "processed_test_scaled",
    }

    key = allowed.get(artifact)
    if not key:
        raise HTTPException(400, "Unsupported preprocessing artifact.")

    result = get_preprocessing(dataset_id, run_id)
    if not result:
        raise HTTPException(404, "Preprocessing run not found.")

    raw_path = (result.get("artifact_paths") or {}).get(key)
    if not raw_path:
        raise HTTPException(404, "Requested artifact is not available.")

    path = Path(raw_path).resolve()
    artifact_root = Path(raw_path).resolve().parent

    if path.parent != artifact_root or not path.exists():
        raise HTTPException(404, "Requested artifact could not be found.")

    return FileResponse(
        path=str(path),
        filename=path.name,
        media_type="application/octet-stream",
    )
