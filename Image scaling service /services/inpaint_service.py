import cv2
import numpy as np
from typing import List, Dict, Any
from utils.logging_utils import logger

class InpaintService:
    @staticmethod
    def inpaint_text(img_bgr: np.ndarray, regions: List[Dict[str, Any]]) -> np.ndarray:
        """
        Removes original text from the image using OpenCV Telea Inpainting.
        """
        if not regions:
            logger.info("No regions to inpaint. Returning original image.")
            return img_bgr.copy()

        logger.info(f"Preparing mask for inpainting {len(regions)} text regions...")
        h, w, c = img_bgr.shape
        
        # Create a single channel binary mask (0 = background, 255 = region to inpaint)
        mask = np.zeros((h, w), dtype=np.uint8)

        for region in regions:
            box = region["box"]
            # Convert float points to integer numpy array
            pts = np.array(box, dtype=np.int32)
            # Fill the polygon inside the mask
            cv2.fillPoly(mask, [pts], 255)

        # Dilate mask slightly to cover edges/anti-aliasing of text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)

        logger.info("Executing OpenCV Telea Inpainting...")
        inpainted = cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        logger.info("Inpainting completed.")
        
        return inpainted
