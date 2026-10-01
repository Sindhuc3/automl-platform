from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.services.feature_engineering_service import run_feature_engineering, get_feature_engineering

router=APIRouter(prefix="/api/datasets",tags=["Feature Engineering"])

@router.post("/{dataset_id}/feature-engineering")
def start(dataset_id:str, request:dict):
    try:
        return {"feature_engineering": run_feature_engineering(dataset_id, request.get("mode","automatic"), request.get("version",1), request.get("preprocessing_run_id"), request.get("overrides") or {})}
    except ValueError as e: raise HTTPException(400,str(e))
    except Exception as e: raise HTTPException(500,f"Feature engineering failed: {e}")

@router.get("/{dataset_id}/feature-engineering")
def listing(dataset_id:str): return {"feature_engineering":get_feature_engineering(dataset_id)}

@router.get("/{dataset_id}/feature-engineering/{run_id}")
def read(dataset_id:str,run_id:str):
    r=get_feature_engineering(dataset_id,run_id)
    if not r: raise HTTPException(404,"Feature engineering run not found.")
    return {"feature_engineering":r}

@router.get("/{dataset_id}/feature-engineering/{run_id}/download/{artifact}")
def download(dataset_id:str,run_id:str,artifact:str):
    allowed={"report":"report","feature_registry":"feature_registry","candidate_feature_sets":"candidate_feature_sets","engineered_training_features":"engineered_training_features"}
    key=allowed.get(artifact)
    if not key: raise HTTPException(400,"Unsupported feature engineering artifact.")
    r=get_feature_engineering(dataset_id,run_id)
    if not r: raise HTTPException(404,"Feature engineering run not found.")
    raw=(r.get("artifact_paths") or {}).get(key)
    if not raw: raise HTTPException(404,"Artifact not available.")
    p=Path(raw).resolve()
    if not p.exists() or p.parent != Path(raw).resolve().parent: raise HTTPException(404,"Artifact could not be found.")
    return FileResponse(str(p),filename=p.name,media_type="application/octet-stream")
