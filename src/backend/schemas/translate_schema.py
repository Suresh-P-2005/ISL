from typing import Optional
from pydantic import BaseModel, Field

class TranslateRequest(BaseModel):
    word: str = Field(..., description="Word to translate")
    lang: Optional[str] = Field("en-US", description="Target language code (e.g., hi-IN, ta-IN)")

class SentenceRequest(BaseModel):
    words: str = Field(..., description="Space-separated keywords")
    lang: Optional[str] = Field("en-US", description="Target language code")

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize to speech")
    lang: Optional[str] = Field("en-US", description="Language code for speech synthesis")
