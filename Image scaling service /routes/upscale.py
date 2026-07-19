import time
import uuid
from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List

from config import settings
from services.image_service import ImageService
from services.resolution_service import ResolutionService
from services.upscale_service import UpscaleService
from utils.logging_utils import logger, request_id_ctx

router = APIRouter()

# Dependency provider for UpscaleService (instantiated in app state)
def get_upscale_service(request: Request) -> UpscaleService:
    return request.app.state.upscale_service

class UpscaleResponse(BaseModel):
    status: str
    original_resolution: str
    output_resolution: str
    processing_time: str
    model: str
    output_url: str

@router.post("/upscale", response_model=UpscaleResponse)
async def upscale_image(
    request: Request,
    image: UploadFile = File(...),
    profile: str = Form("desktop"),
    sharpen: bool = Form(True),
    denoise: bool = Form(False),
    upscale_service: UpscaleService = Depends(get_upscale_service)
) -> JSONResponse:
    
    # Establish Request ID context
    req_id = str(uuid.uuid4())
    request_id_ctx.set(req_id)
    
    start_time = time.time()
    
    # 1. Validate format
    ext = ImageService.validate_image(image)
    
    # 2. Read image content
    contents = await image.read()
    
    # Validate file size
    if len(contents) > 15 * 1024 * 1024:
        logger.error(f"File size too large: {len(contents)} bytes")
        raise HTTPException(status_code=400, detail="Image size exceeds the 15MB limit.")
        
    # 3. Read metadata
    metadata = ImageService.read_metadata(contents)
    width, height = metadata["width"], metadata["height"]
    
    # Log request details
    extra_log_info = {
        "request_id": req_id,
        "input_width": width,
        "input_height": height,
        "profile": profile,
        "file_size": len(contents),
        "format": ext
    }
    logger.info(f"Received upscale request: {width}x{height} image", extra=extra_log_info)
    
    try:
        # 4. Determine target resolution and required scaling
        target_info = ResolutionService.calculate_target(width, height, profile)
        
        # 5. Convert bytes to OpenCV image
        cv2_img = ImageService.bytes_to_cv2(contents)
        
        # 6. Execute upscaling pipeline
        result = upscale_service.process(
            img_bgr=cv2_img,
            target_w=target_info["target_width"],
            target_h=target_info["target_height"],
            passes=target_info["passes"],
            run_sharpen=sharpen,
            run_denoise=denoise
        )
        
        # 7. Save output
        filename, filepath = ImageService.save_image(
            image=result["image"],
            output_dir=settings.OUTPUT_DIRECTORY,
            ext=ext
        )
        
        elapsed = time.time() - start_time
        logger.info(
            f"Upscale request completed successfully in {elapsed:.2f}s. Output: {target_info['target_width']}x{target_info['target_height']}",
            extra={"request_id": req_id, "inference_time": result["processing_time"]}
        )
        
        # 8. Return response
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "original_resolution": f"{width}x{height}",
                "output_resolution": f"{target_info['target_width']}x{target_info['target_height']}",
                "processing_time": result["processing_time"],
                "model": result["model_used"],
                "output_url": f"/output/{filename}"
            }
        )
        
    except ValueError as val_err:
        logger.error(f"Validation error: {str(val_err)}", extra={"request_id": req_id})
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        logger.exception("Unexpected error during upscale processing", extra={"request_id": req_id})
        raise HTTPException(status_code=500, detail="Internal server error during upscale processing.")

@router.get("/health")
async def health_check(request: Request) -> Dict[str, str]:
    # Check if model is initialized
    service = request.app.state.upscale_service
    model_loaded = service.model is not None
    return {
        "status": "healthy",
        "model_loaded": str(model_loaded),
        "device": service.device
    }

@router.get("/models")
async def list_models() -> Dict[str, Any]:
    return {
        "active_model": "RealESRGAN_x4plus",
        "available_models": ["RealESRGAN_x4plus"],
        "architecture": "RRDBNet (Residual-in-Residual Dense Block Network)"
    }

@router.get("/profiles")
async def list_profiles() -> List[Dict[str, Any]]:
    return [
        {"name": "mobile", "description": "Max resolution 1080x1920, suitable for smartphones"},
        {"name": "desktop", "description": "Max resolution 2560x1440, suitable for monitors"},
        {"name": "hoarding", "description": "Target resolution 12000x8000, suitable for billboard and print"}
    ]
