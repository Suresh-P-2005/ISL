from typing import List, Optional
from pydantic import BaseModel, Field

class InferenceRequest(BaseModel):
    landmarks: List[float] = Field(..., description="126-length normalized landmark array")
    mode: Optional[str] = Field("alphabet", description="Classification mode: alphabet, number, static_word")
    engine: Optional[str] = Field("auto", description="Inference engine preference: auto, rf, cnn")

class SequenceInferenceRequest(BaseModel):
    frames: List[List[float]] = Field(..., description="List of keyframe landmark vectors")
    num_hands: Optional[int] = Field(1, description="Number of hands active")
