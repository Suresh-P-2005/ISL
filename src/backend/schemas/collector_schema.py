from typing import List, Optional
from pydantic import BaseModel, Field

class CollectImageRequest(BaseModel):
    landmarks: List[float] = Field(..., description="Image landmarks vector")
    label: str = Field(..., description="Class label for dataset")
    mode: Optional[str] = Field("alphabet", description="Dataset mode: alphabet, number, static_word")

class CollectVideoRequest(BaseModel):
    frames: List[List[float]] = Field(..., description="Sequence of landmark frames")
    label: str = Field(..., description="Dynamic word label")
    mode: Optional[str] = Field("word", description="Dataset mode: word")

class DeleteLabelRequest(BaseModel):
    mode: Optional[str] = None
    label: Optional[str] = None

class DeleteRealRequest(BaseModel):
    mode: Optional[str] = None
