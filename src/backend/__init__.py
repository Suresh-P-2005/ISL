import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, Response

from config import get_config
from src.backend.services.inference_service import InferenceService
from src.backend.services.translation_service import TranslationService
from src.backend.services.collector_service import CollectorService
from src.backend.services.auth_service import AuthService
from src.backend.core.logging_config import setup_logging

def create_app(config_object=None):
    if config_object is None:
        config_object = get_config()

    setup_logging(config_object.BASE_DIR)

    app = FastAPI(
        title="ISL Translator API",
        description="Production-Ready Indian Sign Language Recognition & Translation API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    app.state.config = config_object

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Security Response Header Middleware
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "public, max-age=86400"
        return response

    # Initialize Services
    models_path = config_object.MODELS_DIR if os.path.exists(config_object.MODELS_DIR) else config_object.LEGACY_MODELS_DIR
    app.state.inference_service = InferenceService(models_dir=models_path, config=vars(config_object))
    app.state.translation_service = TranslationService()
    app.state.collector_service = CollectorService(
        data_dir=config_object.DATA_DIR,
        video_dir=config_object.VIDEO_DIR,
        keyframes=config_object.KEYFRAMES
    )
    app.state.auth_service = AuthService(db_dir=config_object.DATA_DIR, secret_key=config_object.SECRET_KEY)

    # Include APIRouters
    from src.backend.api.v1.inference import inference_router
    from src.backend.api.v1.collector import collector_router
    from src.backend.api.v1.translate import translate_router
    from src.backend.api.v1.health import health_router
    from src.backend.api.v1.auth import auth_router

    app.include_router(inference_router)
    app.include_router(collector_router)
    app.include_router(translate_router)
    app.include_router(health_router)
    app.include_router(auth_router)

    # Mount Static Files (CSS, JS, assets)
    if hasattr(config_object, 'STATIC_DIR') and os.path.exists(config_object.STATIC_DIR):
        app.mount("/static", StaticFiles(directory=config_object.STATIC_DIR), name="static")

    # Web Page UI Routes (Serve HTML templates)
    @app.get("/", response_class=HTMLResponse)
    async def route_realtime():
        realtime_path = os.path.join(config_object.TEMPLATES_DIR, "realtime.html")
        if os.path.exists(realtime_path):
            with open(realtime_path, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>ISL Translator - Realtime Page</h1>"

    @app.get("/upload", response_class=HTMLResponse)
    async def route_upload():
        upload_path = os.path.join(config_object.TEMPLATES_DIR, "upload.html")
        if os.path.exists(upload_path):
            with open(upload_path, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>ISL Translator - Upload Page</h1>"

    @app.get("/collect", response_class=HTMLResponse)
    async def route_collect_page():
        collect_path = os.path.join(config_object.TEMPLATES_DIR, "collect.html")
        if os.path.exists(collect_path):
            with open(collect_path, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>ISL Translator - Collect Page</h1>"

    @app.get("/tutorial", response_class=HTMLResponse)
    async def route_tutorial():
        tutorial_path = os.path.join(config_object.TEMPLATES_DIR, "tutorial.html")
        if os.path.exists(tutorial_path):
            with open(tutorial_path, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>ISL Translator - Tutorial Page</h1>"

    @app.get("/login", response_class=HTMLResponse)
    async def route_login():
        login_path = os.path.join(config_object.TEMPLATES_DIR, "login.html")
        if os.path.exists(login_path):
            with open(login_path, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>ISL Translator - Login Page</h1>"

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        # Serve an inline SVG favicon to prevent 404 log spam
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
               '<rect width="32" height="32" rx="8" fill="#38bdf8"/>'
               '<text x="16" y="23" text-anchor="middle" font-size="18" font-family="sans-serif" fill="#fff">🤟</text>'
               '</svg>')
        return Response(content=svg, media_type="image/svg+xml")

    return app

app = create_app()
