import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from utils.logging_utils import logger
from config import settings

class OCRService:
    def __init__(self, engine_name: str = "PaddleOCR", device: str = "cpu"):
        self.engine_name = engine_name
        self.device = device
        self.paddle_ocr = None
        self.easy_ocr_reader = None
        self.active_engine = None
        
        self.initialize_engine()

    def initialize_engine(self) -> None:
        """
        Attempts to load the selected OCR engine, with automatic fallback.
        """
        use_gpu = self.device in ("cuda", "mps")
        
        # 1. Attempt PaddleOCR if preferred
        if self.engine_name.lower() == "paddleocr":
            try:
                from paddleocr import PaddleOCR
                logger.info("Initializing PaddleOCR engine...")
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu, show_log=False)
                self.active_engine = "PaddleOCR"
                logger.info("PaddleOCR engine loaded successfully.")
                return
            except Exception as e:
                logger.warning(f"Failed to load PaddleOCR ({str(e)}). Falling back to EasyOCR.")
                
        # 2. Fallback to EasyOCR
        try:
            import easyocr
            logger.info("Initializing EasyOCR reader...")
            self.easy_ocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
            self.active_engine = "EasyOCR"
            logger.info("EasyOCR engine loaded successfully.")
        except Exception as e:
            logger.exception("Failed to initialize any OCR engine.")
            raise RuntimeError("No OCR engine (PaddleOCR or EasyOCR) could be initialized.") from e

    def detect_text(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects text inside the image.
        Returns a list of regions:
        {
            "box": [[x, y], ...], # 4 corner points
            "text": str,
            "confidence": float,
            "angle": float # Rotation angle in degrees
        }
        """
        if self.active_engine == "PaddleOCR" and self.paddle_ocr:
            try:
                return self._run_paddle_ocr(img_bgr)
            except Exception as e:
                logger.exception("PaddleOCR run failed. Attempting quick EasyOCR fallback run.")
                if not self.easy_ocr_reader:
                    import easyocr
                    use_gpu = self.device in ("cuda", "mps")
                    self.easy_ocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
                    self.active_engine = "EasyOCR"
                return self._run_easy_ocr(img_bgr)
        elif self.easy_ocr_reader:
            return self._run_easy_ocr(img_bgr)
        else:
            raise RuntimeError("No active OCR engine loaded.")

    def _run_paddle_ocr(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        results = self.paddle_ocr.ocr(img_bgr, cls=True) # type: ignore
        regions = []
        if not results or not results[0]:
            return regions
            
        for line in results[0]:
            box = [[float(pt[0]), float(pt[1])] for pt in line[0]]
            text = line[1][0]
            confidence = float(line[1][1])
            angle = self._calculate_rotation_angle(box)
            
            regions.append({
                "box": box,
                "text": text,
                "confidence": confidence,
                "angle": angle
            })
        return regions

    def _run_easy_ocr(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        results = self.easy_ocr_reader.readtext(img_bgr) # type: ignore
        regions = []
        for res in results:
            # res: (bbox, text, prob)
            box = [[float(pt[0]), float(pt[1])] for pt in res[0]]
            text = res[1]
            confidence = float(res[2])
            angle = self._calculate_rotation_angle(box)
            
            regions.append({
                "box": box,
                "text": text,
                "confidence": confidence,
                "angle": angle
            })
        return regions

    def _calculate_rotation_angle(self, box: List[List[float]]) -> float:
        """
        Calculates rotation angle in degrees from the bounding box orientation.
        """
        try:
            # Top-left (p1) to top-right (p2) vector
            x1, y1 = box[0]
            x2, y2 = box[1]
            dx = x2 - x1
            dy = y2 - y1
            angle_rad = math.atan2(dy, dx)
            return math.degrees(angle_rad)
        except Exception:
            return 0.0
