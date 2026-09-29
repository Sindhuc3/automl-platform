from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

class PreprocessingRunRequest(BaseModel):
    mode: str = Field(default="automatic")
    version: int = Field(default=1, ge=1)
    overrides: Dict[str, Any] = Field(default_factory=dict)

class PreprocessingResponse(BaseModel):
    message: str
    preprocessing: Dict[str, Any]
