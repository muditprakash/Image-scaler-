import os
import uuid
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any
from fastapi import UploadFile, HTTPException
from utils.logging_utils import logger

SUPPORTED_FORMATS = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15MB max file upload size

class ImageService:
    @staticmethod
    def validate_image(file: UploadFile) -> str:
        """
        Validates content type and basic constraints.
        Returns the clean extension.
        """
        if file.content_type not in SUPPORTED_FORMATS:
            logger.error(f"Unsupported content type: {file.content_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{file.content_type}'. Supported: PNG, JPEG, WEBP."
            )
        
        ext = "png"
        if file.content_type:
            ext = file.content_type.split("/")[-1]
            if ext == "jpeg":
                ext = "jpg"
        return ext

    @staticmethod
    def read_metadata(image_bytes: bytes) -> Dict[str, Any]:
        """
        Reads image metadata using Pillow without loading full image into PyTorch.
        """
        from io import BytesIO
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                return {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                }
        except Exception as e:
            logger.exception("Failed to read image metadata")
            raise HTTPException(status_code=400, detail="Invalid or corrupt image file.")

    @staticmethod
    def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
        """
        Converts raw bytes to OpenCV BGR image.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image.")
        return img

    @staticmethod
    def cv2_to_bytes(image: np.ndarray, ext: str) -> bytes:
        """
        Converts OpenCV image to raw bytes.
        """
        success, encoded_img = cv2.imencode(f".{ext}", image)
        if not success:
            raise ValueError("Failed to encode image.")
        return encoded_img.tobytes()

    @staticmethod
    def sharpen_image(image: np.ndarray) -> np.ndarray:
        """
        Applies a high-quality sharpening kernel.
        """
        logger.info("Applying sharpening filter...")
        # Unsharp masking or classic sharpening kernel
        # Kernel options: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
        kernel = np.array([
            [ 0, -0.5,  0],
            [-0.5,  3.0, -0.5],
            [ 0, -0.5,  0]
        ], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def denoise_image(image: np.ndarray) -> np.ndarray:
        """
        Applies bilateral filtering (preserves edges while removing noise).
        """
        logger.info("Applying bilateral denoising filter...")
        return cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)

    @staticmethod
    def save_image(image: np.ndarray, output_dir: str, ext: str) -> Tuple[str, str]:
        """
        Saves image to output folder and returns filename and absolute path.
        """
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{uuid.uuid4()}.{ext}"
        filepath = os.path.join(output_dir, filename)
        
        cv2.imwrite(filepath, image)
        logger.info(f"Image saved successfully to {filepath}")
        return filename, filepath
