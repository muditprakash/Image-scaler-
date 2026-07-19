import time
import uuid
from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List

from config import settings
from services.image_service import ImageService
from services.ocr_service import OCRService
from services.translation_service import TranslationService
from services.inpaint_service import InpaintService
from services.render_service import RenderService
from utils.logging_utils import logger, request_id_ctx

router = APIRouter()

SUPPORTED_LANGUAGES = ["English", "Hindi", "Marathi"]

# Dependency providers
def get_ocr_service(request: Request) -> OCRService:
    return request.app.state.ocr_service

def get_translation_service(request: Request) -> TranslationService:
    return request.app.state.translation_service

def get_render_service(request: Request) -> RenderService:
    return request.app.state.render_service

class TranslateResponse(BaseModel):
    status: str
    detected_regions: int
    translated_regions: int
    target_language: str
    processing_time: str
    ocr_engine: str
    translation_model: str
    output_url: str

class NoTextResponse(BaseModel):
    status: str
    message: str

@router.post("/image/translate")
async def translate_image(
    request: Request,
    image: UploadFile = File(...),
    target_language: str = Form("English"),
    ocr_service: OCRService = Depends(get_ocr_service),
    translation_service: TranslationService = Depends(get_translation_service),
    render_service: RenderService = Depends(get_render_service)
) -> Any:
    
    # Establish Request ID context
    req_id = str(uuid.uuid4())
    request_id_ctx.set(req_id)
    
    start_time = time.time()
    
    # Validate target language
    lang = target_language.strip().capitalize()
    if lang not in SUPPORTED_LANGUAGES:
        logger.error(f"Unsupported target language requested: {target_language}", extra={"request_id": req_id})
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{target_language}'. Supported: {SUPPORTED_LANGUAGES}"
        )
        
    # Validate file format
    ext = ImageService.validate_image(image)
    
    # Read image contents
    contents = await image.read()
    if len(contents) > 15 * 1024 * 1024:
        logger.error("Uploaded image exceeds 15MB size limit", extra={"request_id": req_id})
        raise HTTPException(status_code=400, detail="Image size exceeds the 15MB limit.")
        
    # Read metadata
    metadata = ImageService.read_metadata(contents)
    width, height = metadata["width"], metadata["height"]
    
    logger.info(
        f"Received image translation request: {width}x{height} image to target_language={lang}",
        extra={"request_id": req_id, "input_width": width, "input_height": height, "file_size": len(contents)}
    )

    try:
        # Convert bytes to OpenCV image
        cv2_img = ImageService.bytes_to_cv2(contents)
        
        # 1. OCR Stage
        ocr_start = time.time()
        regions = ocr_service.detect_text(cv2_img)
        ocr_elapsed = time.time() - ocr_start
        logger.info(f"OCR completed in {ocr_elapsed:.2f}s. Regions found: {len(regions)}", extra={"request_id": req_id})
        
        # Return custom message if no text detected (without crashing or failing)
        if not regions:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": "No text detected in image."
                }
            )

        # 2. Translation Stage
        trans_start = time.time()
        for idx, region in enumerate(regions):
            orig_text = region["text"]
            # Translate text
            translated = translation_service.translate_text(orig_text, lang)
            region["translated_text"] = translated
            logger.info(f"Region {idx + 1}: '{orig_text}' -> '{translated}'", extra={"request_id": req_id})
        trans_elapsed = time.time() - trans_start
        logger.info(f"Translation completed in {trans_elapsed:.2f}s", extra={"request_id": req_id})

        # 3. Inpainting Stage
        inpainted_img = InpaintService.inpaint_text(cv2_img, regions)
        
        # 4. Rendering Stage
        render_start = time.time()
        final_img = render_service.render_translations(inpainted_img, regions)
        render_elapsed = time.time() - render_start
        logger.info(f"Rendering completed in {render_elapsed:.2f}s", extra={"request_id": req_id})

        # Save output image
        filename, filepath = ImageService.save_image(
            image=final_img,
            output_dir=settings.OUTPUT_DIRECTORY,
            ext=ext
        )
        
        total_time = time.time() - start_time
        logger.info(
            f"Image translation completed in {total_time:.2f}s.",
            extra={
                "request_id": req_id,
                "ocr_time": f"{ocr_elapsed:.2f} sec",
                "translation_time": f"{trans_elapsed:.2f} sec",
                "rendering_time": f"{render_elapsed:.2f} sec",
                "total_time": f"{total_time:.2f} sec"
            }
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "detected_regions": len(regions),
                "translated_regions": len(regions),
                "target_language": lang,
                "processing_time": f"{total_time:.2f} sec",
                "ocr_engine": ocr_service.active_engine,
                "translation_model": settings.TRANSLATION_MODEL,
                "output_url": f"/output/{filename}"
            }
        )
        
    except Exception as exc:
        logger.exception("Unexpected error during image translation pipeline", extra={"request_id": req_id})
        raise HTTPException(status_code=500, detail="Internal server error during image translation pipeline.")

@router.get("/languages")
async def list_languages() -> List[str]:
    return SUPPORTED_LANGUAGES

@router.get("/ocr/health")
async def ocr_health(request: Request) -> Dict[str, str]:
    ocr_service = request.app.state.ocr_service
    return {
        "status": "active",
        "ocr_engine": ocr_service.active_engine,
        "device": ocr_service.device
    }
