from fastapi import APIRouter, HTTPException

from app.services.model_hpo_service import run_model_hpo, get_model_hpo

router = APIRouter(
    prefix="/api/datasets",
    tags=["Hyperparameter Optimization"],
)


@router.post("/{dataset_id}/model-hpo")
def start_model_hpo(dataset_id: str, request: dict):
    try:
        result = run_model_hpo(
            dataset_id,
            screening_id=request.get("screening_id"),
            config=request.get("config") or {},
        )
        return {"model_hpo": result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model HPO failed: {exc}")


@router.get("/{dataset_id}/model-hpo")
def list_model_hpo(dataset_id: str):
    return {"model_hpo": get_model_hpo(dataset_id)}


@router.get("/{dataset_id}/model-hpo/{hpo_id}")
def read_model_hpo(dataset_id: str, hpo_id: str):
    result = get_model_hpo(dataset_id, hpo_id)
    if not result:
        raise HTTPException(status_code=404, detail="Model HPO run not found.")
    return {"model_hpo": result}
