import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import settings
from routes.upscale import router as upscale_router
from routes.translate import router as translate_router
from services.upscale_service import UpscaleService
from services.ocr_service import OCRService
from services.translation_service import TranslationService
from services.render_service import RenderService
from utils.logging_utils import logger

# Ensure directories exist
os.makedirs(settings.OUTPUT_DIRECTORY, exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("weights", exist_ok=True)
os.makedirs("fonts", exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the AI Models once
    logger.info("Initializing application resources...")
    
    # 1. Image Upscaler Service
    upscale_service = UpscaleService(
        model_path=settings.MODEL_PATH,
        device=settings.DEVICE
    )
    try:
        upscale_service.load_model()
    except Exception as e:
        logger.exception("Failed to load upscale model during startup.")
        
    # 2. OCR Service
    ocr_service = OCRService(
        engine_name=settings.OCR_ENGINE,
        device=settings.DEVICE
    )
    
    # 3. Translation Service
    translation_service = TranslationService(
        model_name=settings.TRANSLATION_MODEL,
        device=settings.DEVICE
    )
    try:
        translation_service.load_model()
    except Exception as e:
        logger.exception("Failed to load translation model during startup.")
        
    # 4. Render Service (Downloads Noto Devanagari font if missing)
    render_service = RenderService(font_dir="fonts")
    
    app.state.upscale_service = upscale_service
    app.state.ocr_service = ocr_service
    app.state.translation_service = translation_service
    app.state.render_service = render_service
    
    yield
    # Shutdown: Clean up resources if needed
    logger.info("Tearing down application resources...")

app = FastAPI(
    title="AI Image Processing Dashboard",
    description="FastAPI service for intelligent AI Super Resolution upscaling and Image Text Translation.",
    version="1.1.0",
    lifespan=lifespan
)

# CORS middleware for cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount upscale output files directory
app.mount("/output", StaticFiles(directory=settings.OUTPUT_DIRECTORY), name="output")

# Include api routes
app.include_router(upscale_router)
app.include_router(translate_router)

# Mount frontend files (index.html, styles.css)
# Serve index.html as the landing page
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Read environment port or default to 8000
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
