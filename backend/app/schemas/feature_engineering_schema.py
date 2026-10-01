from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class FeatureEngineeringRunRequest(BaseModel):
    mode: str = Field(default="automatic")
    version: int = Field(default=1, ge=1)
    preprocessing_run_id: Optional[str] = None
    overrides: Dict[str, Any] = Field(default_factory=dict)
