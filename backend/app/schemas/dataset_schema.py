from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class DatasetSummary(BaseModel):
    dataset_id: str
    original_filename: str
    file_type: str
    file_size: int
    rows: int
    columns: int
    quality_status: str
    created_at: str


class TargetSelection(BaseModel):
    target_column: str


class DatasetUploadResponse(BaseModel):
    message: str
    dataset_id: str
    dataset: Dict[str, Any]


class DatasetListResponse(BaseModel):
    datasets: List[Dict[str, Any]]


class TargetSelectionResponse(BaseModel):
    message: str
    dataset_id: str
    target_column: str