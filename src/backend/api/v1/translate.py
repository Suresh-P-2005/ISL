import io
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from src.backend.schemas.translate_schema import TranslateRequest, SentenceRequest, TTSRequest

translate_router = APIRouter(tags=["Translation & Speech"])

@translate_router.post("/make_sentence")
async def make_sentence(payload: SentenceRequest, request: Request):
    translation_service = request.app.state.translation_service
    res = translation_service.construct_sentence(payload.words, payload.lang or "en-US")
    return res

@translate_router.post("/translate")
async def translate(payload: TranslateRequest, request: Request):
    translation_service = request.app.state.translation_service
    translated = translation_service.translate_word(payload.word, payload.lang or "en-US")
    return {"original": payload.word, "translated": translated}

@translate_router.post("/tts")
async def tts_route(payload: TTSRequest):
    if not payload.text:
        raise HTTPException(status_code=400, detail="No text provided for TTS")

    gtts_lang = (payload.lang or "en-US").split('-')[0]

    try:
        from gtts import gTTS
        tts = gTTS(text=payload.text, lang=gtts_lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return StreamingResponse(fp, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Generation failed: {str(e)}")
