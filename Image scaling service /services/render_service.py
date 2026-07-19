import os
import requests
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from typing import List, Dict, Any, Tuple
from utils.logging_utils import logger

class RenderService:
    def __init__(self, font_dir: str = "fonts"):
        self.font_dir = font_dir
        self.font_path = os.path.join(self.font_dir, "NotoSansDevanagari-Regular.ttf")
        os.makedirs(self.font_dir, exist_ok=True)
        self.ensure_font_exists()

    def ensure_font_exists(self) -> None:
        """
        Downloads Noto Sans Devanagari font from Google Fonts repo if it's missing.
        Supports English, Hindi, and Marathi rendering seamlessly.
        """
        if not os.path.exists(self.font_path):
            url = "https://github.com/google/fonts/raw/main/ofl/notosansdevanagari/NotoSansDevanagari-Regular.ttf"
            logger.info(f"Downloading NotoSansDevanagari font from {url}...")
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                with open(self.font_path, "wb") as f:
                    f.write(response.content)
                logger.info("Font downloaded successfully.")
            except Exception as e:
                logger.exception("Failed to download Devanagari font. Falling back to default system font.")
                # We will fall back to PIL default font if loading fails

    def _determine_text_color(self, img_bgr: np.ndarray, box: List[List[float]]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Determines text color and outline color based on the background brightness of the bounding box.
        Returns: (text_rgb, outline_rgb)
        """
        try:
            # Crop the bounding box region to compute average brightness
            pts = np.array(box, dtype=np.int32)
            rect = cv2.boundingRect(pts) # type: ignore
            x, y, w, h = rect
            
            # Ensure crop coordinates are valid
            h_img, w_img = img_bgr.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_img, x + w), min(h_img, y + h)
            
            if x2 > x1 and y2 > y1:
                crop = img_bgr[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) # type: ignore
                avg_brightness = np.mean(gray)
            else:
                avg_brightness = 128
        except Exception:
            import cv2
            avg_brightness = 128

        # If background is bright, render black text with white outline.
        # If background is dark, render white text with black outline.
        if avg_brightness > 127:
            return (0, 0, 0), (255, 255, 255)
        else:
            return (255, 255, 255), (0, 0, 0)

    def render_translations(self, img_bgr: np.ndarray, regions: List[Dict[str, Any]]) -> np.ndarray:
        """
        Renders translated texts inside the bounding boxes of the inpainted BGR image.
        """
        if not regions:
            return img_bgr.copy()

        logger.info(f"Rendering {len(regions)} translated text blocks...")
        
        # Convert BGR image to PIL image (RGBA to allow alpha blending)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # type: ignore
        pil_img = Image.fromarray(img_rgb).convert("RGBA")
        
        for region in regions:
            box = region["box"]
            translated_text = region.get("translated_text", "")
            if not translated_text.strip():
                continue

            angle = region.get("angle", 0.0)

            # Get box dimensions (width and height)
            pts = np.array(box, dtype=np.int32)
            # Find min area bounding box coordinates
            rect = cv2.minAreaRect(pts) # type: ignore
            box_w = int(rect[1][0])
            box_h = int(rect[1][1])
            
            # minAreaRect can swap width and height depending on the angle
            # We enforce that width is the horizontal-like span
            # If the angle is steep, rect[1] might have height > width.
            # Let's approximate box_w and box_h using corner points
            p0, p1, p2, p3 = box
            w_est = math.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
            h_est = math.sqrt((p3[0] - p0[0])**2 + (p3[1] - p0[1])**2)
            box_w, box_h = int(max(4, w_est)), int(max(4, h_est))

            # Determine colors
            text_color, outline_color = self._determine_text_color(img_bgr, box)

            # Create a transparent scratch pad for the text rendering
            # We render horizontal text first, then rotate it
            pad_w = int(box_w * 1.5)
            pad_h = int(box_h * 1.5)
            
            text_canvas = Image.new("RGBA", (pad_w, pad_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(text_canvas)

            # Determine suitable font size
            font_size = max(8, int(box_h * 0.8))
            font = None
            
            # Loop to find font size that fits the bounding box width
            while font_size > 6:
                try:
                    font = ImageFont.truetype(self.font_path, font_size)
                except Exception:
                    font = ImageFont.load_default()
                    break

                # Get text size
                try:
                    left, top, right, bottom = draw.textbbox((0, 0), translated_text, font=font)
                    text_w = right - left
                    text_h = bottom - top
                except AttributeError:
                    # Legacy fallback
                    text_w, text_h = draw.textsize(translated_text, font=font) # type: ignore

                # If the text width fits within our box width, we stop
                if text_w <= box_w:
                    break
                font_size -= 1

            # Render text centered on the text canvas
            text_x = (pad_w - text_w) // 2
            text_y = (pad_h - text_h) // 2
            
            # Draw text with outline
            draw.text(
                (text_x, text_y),
                translated_text,
                font=font,
                fill=text_color,
                stroke_width=2,
                stroke_fill=outline_color
            )

            # Rotate the text canvas if needed
            # Negative angle for Pillow rotate because it rotates counter-clockwise
            if abs(angle) > 0.5:
                # Rotate around center of canvas
                rotated_canvas = text_canvas.rotate(-angle, resample=Image.Resampling.BILINEAR, expand=False)
            else:
                rotated_canvas = text_canvas

            # Paste the rotated text back onto the original image
            # Find the center of the bounding box to align
            center_x = sum([pt[0] for pt in box]) / 4
            center_y = sum([pt[1] for pt in box]) / 4
            
            paste_x = int(center_x - pad_w / 2)
            paste_y = int(center_y - pad_h / 2)
            
            # Composite using alpha channel as mask
            pil_img.alpha_composite(rotated_canvas, (paste_x, paste_y))

        # Convert back to OpenCV BGR
        final_rgb = np.array(pil_img.convert("RGB"))
        return cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR) # type: ignore
