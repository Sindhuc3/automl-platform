from fastapi import APIRouter, HTTPException

from app.services.model_arbiter_service import run_model_arbiter, get_model_arbiter

router = APIRouter(
    prefix="/api/datasets",
    tags=["Selection Arbiter (M6.5)"],
)


@router.post("/{dataset_id}/model-selection")
def start_model_selection(dataset_id: str, request: dict):
    try:
        result = run_model_arbiter(
            dataset_id,
            hpo_id=request.get("hpo_id"),
            config=request.get("config") or {},
        )
        return {"model_selection": result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model selection failed: {exc}")


@router.get("/{dataset_id}/model-selection")
def list_model_selection(dataset_id: str):
    return {"model_selection": get_model_arbiter(dataset_id)}


@router.get("/{dataset_id}/model-selection/{selection_id}")
def read_model_selection(dataset_id: str, selection_id: str):
    result = get_model_arbiter(dataset_id, selection_id)
    if not result:
        raise HTTPException(status_code=404, detail="Model selection run not found.")
    return {"model_selection": result}
