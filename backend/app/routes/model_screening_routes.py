from fastapi import APIRouter, HTTPException

from app.services.model_screening_service import (
    run_model_screening,
    get_model_screening,
)

router = APIRouter(
    prefix="/api/datasets",
    tags=["Model Screening"],
)


@router.post("/{dataset_id}/model-screening")
def start_model_screening(dataset_id: str, request: dict):
    try:
        result = run_model_screening(
            dataset_id,
            preprocessing_run_id=request.get("preprocessing_run_id"),
            feature_engineering_run_id=request.get("feature_engineering_run_id"),
            algorithm_ids=request.get("algorithm_ids"),
            config=request.get("config") or {},
        )
        return {"model_screening": result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model screening failed: {exc}")


@router.get("/{dataset_id}/model-screening")
def list_model_screening(dataset_id: str):
    return {"model_screening": get_model_screening(dataset_id)}


@router.get("/{dataset_id}/model-screening/{screening_id}")
def read_model_screening(dataset_id: str, screening_id: str):
    result = get_model_screening(dataset_id, screening_id)
    if not result:
        raise HTTPException(status_code=404, detail="Model screening run not found.")
    return {"model_screening": result}


@router.get("/{dataset_id}/model-screening/{screening_id}/candidates")
def list_candidates(dataset_id: str, screening_id: str):
    result = get_model_screening(dataset_id, screening_id)
    if not result:
        raise HTTPException(status_code=404, detail="Model screening run not found.")
    return {
        "screening_id": screening_id,
        "candidates": result.get("candidates", []),
        "ranking": result.get("ranking", []),
    }
