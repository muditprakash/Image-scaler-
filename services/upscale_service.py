import time
import math
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import Dict, Any, Optional
from models.rrdbnet import get_rrdbnet_model, RRDBNet
from utils.logging_utils import logger

class UpscaleService:
    def __init__(self, model_path: str, device: str = "auto", tile_size: int = 512, tile_pad: int = 32):
        self.model_path = model_path
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.device = self._select_device(device)
        self.model: Optional[RRDBNet] = None

    def _select_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                selected = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                selected = "mps"
            else:
                selected = "cpu"
        else:
            selected = device
        logger.info(f"Target execution device selected: {selected}")
        return selected

    def load_model(self) -> None:
        """
        Loads the model once and caches it.
        """
        if self.model is None:
            self.model = get_rrdbnet_model(self.model_path, self.device)
            logger.info("Real-ESRGAN Model initialized and cached successfully.")

    def face_enhancement_hook(self, image: np.ndarray) -> np.ndarray:
        """
        Placeholder hook for future GFPGAN / CodeFormer face restoration.
        """
        logger.info("Face enhancement hook triggered (currently a no-op placeholder).")
        # In a real setup:
        # face_enhancer = GFPGANer(...)
        # _, _, restored_img = face_enhancer.enhance(image, ...)
        # return restored_img
        return image

    def upscale_single_pass(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Runs one pass of 4x upscaling using the cached model.
        Splits image into tiles if it exceeds tile_size to prevent OOM.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        h, w, c = img_bgr.shape
        # Check if we can run direct inference or if we need tiling
        if h <= self.tile_size and w <= self.tile_size:
            return self._direct_inference(img_bgr)
        else:
            return self._tiled_inference(img_bgr)

    def _direct_inference(self, img_bgr: np.ndarray) -> np.ndarray:
        # Convert BGR to RGB and float tensor
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            output_tensor = self.model(tensor) # type: ignore
            
        output = output_tensor.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
        output = (output * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    def _tiled_inference(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Processes image in overlapping tiles to prevent memory issues.
        """
        logger.info(f"Using tiled inference. Image shape: {img_bgr.shape}")
        h, w, c = img_bgr.shape
        scale = 4
        
        # Output canvas
        output_h, output_w = h * scale, w * scale
        output_img = np.zeros((output_h, output_w, c), dtype=np.uint8)
        
        # Calculate tiles
        stride = self.tile_size - 2 * self.tile_pad
        h_tiles = math.ceil((h - 2 * self.tile_pad) / stride) + 1
        w_tiles = math.ceil((w - 2 * self.tile_pad) / stride) + 1

        for i in range(h_tiles):
            for j in range(w_tiles):
                # Input tile coordinates in original image
                y1 = i * stride
                y2 = min(y1 + self.tile_size, h)
                y1 = max(0, y2 - self.tile_size) # Adjust start if close to boundary
                
                x1 = j * stride
                x2 = min(x1 + self.tile_size, w)
                x1 = max(0, x2 - self.tile_size)
                
                tile = img_bgr[y1:y2, x1:x2]
                
                # Run inference on tile
                upscaled_tile = self._direct_inference(tile)
                
                # Output tile coordinates
                out_y1, out_y2 = y1 * scale, y2 * scale
                out_x1, out_x2 = x1 * scale, x2 * scale
                
                # Determine cropping bounds for the tile to prevent seam artifacts
                crop_top = (y1 - (i * stride)) * scale if y1 > 0 else 0
                crop_left = (x1 - (j * stride)) * scale if x1 > 0 else 0
                
                # We blend or copy the tile to output
                # Simply copy the non-overlapping inner part
                # For simplicity and speed, crop the pad boundaries
                pad_top = scale * self.tile_pad if y1 > 0 else 0
                pad_bottom = scale * self.tile_pad if y2 < h else 0
                pad_left = scale * self.tile_pad if x1 > 0 else 0
                pad_right = scale * self.tile_pad if x2 < w else 0
                
                # Output slice
                slice_y1 = out_y1 + pad_top
                slice_y2 = out_y2 - pad_bottom
                slice_x1 = out_x1 + pad_left
                slice_x2 = out_x2 - pad_right
                
                # Tile slice
                tile_y1 = pad_top
                tile_y2 = upscaled_tile.shape[0] - pad_bottom
                tile_x1 = pad_left
                tile_x2 = upscaled_tile.shape[1] - pad_right
                
                output_img[slice_y1:slice_y2, slice_x1:slice_x2] = upscaled_tile[tile_y1:tile_y2, tile_x1:tile_x2]
                
        return output_img

    def process(self, img_bgr: np.ndarray, target_w: int, target_h: int, passes: int, run_sharpen: bool = True, run_denoise: bool = False) -> Dict[str, Any]:
        """
        Executes the full pipeline: multiple passes of AI upscaling,
        post-processing (sharpen, denoise, face enhancement), and final bicubic fit.
        """
        start_time = time.time()
        logger.info(f"Starting upscale processing pipeline. Passes requested: {passes}")
        
        # Load model if not cached
        self.load_model()
        
        # Separate alpha channel if PNG has transparency
        # Check if 4 channels, but we receive BGR (3 channels) from cv2_to_bytes decoder by default.
        # If we need alpha support, the caller passes a 4-channel image.
        has_alpha = img_bgr.shape[2] == 4
        if has_alpha:
            logger.info("Image has alpha channel. Upscaling RGB and Alpha separately...")
            alpha = img_bgr[:, :, 3]
            rgb = img_bgr[:, :, :3]
        else:
            rgb = img_bgr

        # Perform AI upscaling passes (each pass scales by 4x)
        current_img = rgb
        for p in range(passes):
            logger.info(f"Running AI super-resolution pass {p + 1}...")
            current_img = self.upscale_single_pass(current_img)

        # Upscale Alpha channel using bicubic interpolation if present
        if has_alpha:
            alpha_target_h, alpha_target_w = current_img.shape[0], current_img.shape[1]
            alpha_upscaled = cv2.resize(alpha, (alpha_target_w, alpha_target_h), interpolation=cv2.INTER_CUBIC)
            # Combine back
            current_img = np.dstack((current_img, alpha_upscaled))

        # Perform final resizing to target dimensions if AI output differs from target
        h_out, w_out = current_img.shape[:2]
        if w_out != target_w or h_out != target_h:
            logger.info(f"Resizing upscaled image from {w_out}x{h_out} to target {target_w}x{target_h} using bicubic interpolation...")
            current_img = cv2.resize(current_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # Image post-processing filters
        # Face enhancement (optional hook)
        current_img = self.face_enhancement_hook(current_img)
        
        # Denoise (run before sharpening if active)
        if run_denoise:
            # If 4 channels, apply only to RGB
            if current_img.shape[2] == 4:
                rgb_part = current_img[:, :, :3]
                denoised_rgb = cv2.bilateralFilter(rgb_part, d=5, sigmaColor=50, sigmaSpace=50)
                current_img[:, :, :3] = denoised_rgb
            else:
                current_img = cv2.bilateralFilter(current_img, d=5, sigmaColor=50, sigmaSpace=50)
                
        # Sharpen
        if run_sharpen:
            kernel = np.array([[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]], dtype=np.float32)
            if current_img.shape[2] == 4:
                rgb_part = current_img[:, :, :3]
                sharpened_rgb = cv2.filter2D(rgb_part, -1, kernel)
                current_img[:, :, :3] = sharpened_rgb
            else:
                current_img = cv2.filter2D(current_img, -1, kernel)

        elapsed = time.time() - start_time
        logger.info(f"Upscale pipeline finished in {elapsed:.3f} seconds.")
        
        return {
            "image": current_img,
            "processing_time": f"{elapsed:.2f} sec",
            "model_used": "RealESRGAN_x4plus"
        }
