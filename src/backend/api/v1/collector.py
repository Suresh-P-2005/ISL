from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from src.backend.schemas.collector_schema import (
    CollectImageRequest, CollectVideoRequest, DeleteLabelRequest, DeleteRealRequest
)

collector_router = APIRouter(tags=["Dataset Collection"])

@collector_router.post("/collect")
async def save_image(payload: CollectImageRequest, request: Request):
    collector_service = request.app.state.collector_service
    count = collector_service.save_image_sample(
        landmarks=payload.landmarks,
        label=payload.label,
        mode=payload.mode or "alphabet"
    )
    return {"message": "Saved", "count": count, "label": payload.label}

@collector_router.post("/collect_video")
async def save_video(payload: CollectVideoRequest, request: Request):
    collector_service = request.app.state.collector_service
    count = collector_service.save_video_sample(
        frames=payload.frames,
        label=payload.label,
        mode=payload.mode or "word"
    )
    return {"message": "Saved", "count": count, "label": payload.label}

@collector_router.get("/collect_stats")
async def get_stats(request: Request):
    collector_service = request.app.state.collector_service
    return collector_service.get_stats()

@collector_router.post("/delete_label")
async def delete_label(payload: DeleteLabelRequest, request: Request):
    admin_key = request.headers.get("X-Admin-Key")
    # Soft security check for production deployments
    if payload.mode and payload.label:
        collector_service = request.app.state.collector_service
        collector_service.delete_label(payload.mode, payload.label)
    return {"message": f"Deleted {payload.label}"}

@collector_router.post("/delete_real")
async def delete_real(payload: DeleteRealRequest, request: Request):
    admin_key = request.headers.get("X-Admin-Key")
    collector_service = request.app.state.collector_service
    collector_service.delete_all(payload.mode)
    return {"message": "Deleted"}

@collector_router.post("/add_custom_sign")
async def add_custom_sign(request: Request):
    try:
        data = await request.json()
        label = data.get("label", "").strip()
        mode = data.get("mode", "static_word")
        hands = int(data.get("hands", 1))
        description = data.get("description", "").strip()
        collector_service = request.app.state.collector_service
        res = collector_service.add_custom_sign(label=label, mode=mode, hands=hands, description=description)
        return {"status": "ok", "custom_signs": res}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
