import asyncio
from fastapi import APIRouter, Request, HTTPException
from src.backend.schemas.inference_schema import InferenceRequest, SequenceInferenceRequest

inference_router = APIRouter(tags=["Inference"])

@inference_router.post("/predict")
async def predict(payload: InferenceRequest, request: Request):
    inference_service = request.app.state.inference_service
    res = await asyncio.to_thread(
        inference_service.predict_static,
        landmarks=payload.landmarks,
        mode=payload.mode or "alphabet",
        engine=payload.engine or "auto"
    )
    return res

@inference_router.post("/predict_sequence")
async def predict_sequence(payload: SequenceInferenceRequest, request: Request):
    inference_service = request.app.state.inference_service
    res = await asyncio.to_thread(
        inference_service.predict_sequence,
        frames=payload.frames,
        num_hands=payload.num_hands or 1
    )
    return res
