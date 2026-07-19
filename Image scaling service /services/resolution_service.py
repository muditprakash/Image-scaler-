from typing import Tuple, Dict, Any
from dataclasses import dataclass
from utils.logging_utils import logger

@dataclass
class ResolutionProfile:
    name: str
    max_width: int
    max_height: int
    is_target_exact: bool  # True for hoarding (exact target), False for mobile/desktop (max bounds)

PROFILES: Dict[str, ResolutionProfile] = {
    "mobile": ResolutionProfile(name="mobile", max_width=1080, max_height=1920, is_target_exact=False),
    "desktop": ResolutionProfile(name="desktop", max_width=2560, max_height=1440, is_target_exact=False),
    "hoarding": ResolutionProfile(name="hoarding", max_width=12000, max_height=8000, is_target_exact=bool(True)),
}

class ResolutionService:
    @staticmethod
    def get_profile(profile_name: str) -> ResolutionProfile:
        name = profile_name.lower().strip()
        if name not in PROFILES:
            raise ValueError(f"Unsupported profile: '{profile_name}'. Supported: {list(PROFILES.keys())}")
        return PROFILES[name]

    @classmethod
    def calculate_target(cls, width: int, height: int, profile_name: str) -> Dict[str, Any]:
        """
        Calculates the target resolution, scale factor, and number of AI passes needed.
        """
        profile = cls.get_profile(profile_name)
        
        # Adjust profile max width/height based on input aspect ratio (portrait vs landscape)
        is_input_portrait = height > width
        
        if is_input_portrait:
            # Swap max bounds to match orientation
            max_w = min(profile.max_width, profile.max_height)
            max_h = max(profile.max_width, profile.max_height)
        else:
            max_w = max(profile.max_width, profile.max_height)
            max_h = min(profile.max_width, profile.max_height)
            
        if profile.is_target_exact:
            # We want to upscale so that the image is exactly filled to target bounds
            # Preserving aspect ratio: we match the larger scale required
            scale_w = max_w / width
            scale_h = max_h / height
            required_scale = max(scale_w, scale_h)
            
            target_w = int(round(width * required_scale))
            target_h = int(round(height * required_scale))
        else:
            # For mobile/desktop, we scale up to fit within the max bounds
            scale_w = max_w / width
            scale_h = max_h / height
            required_scale = min(scale_w, scale_h)
            
            # If the image is already larger than the bounds, we still run AI super-resolution (4x)
            # to enhance quality, and then resize down to fit the bounds.
            if required_scale < 1.0:
                required_scale = 1.0 # Minimum 1x upscale before resizing
                
            target_w = int(round(width * required_scale))
            target_h = int(round(height * required_scale))

        # Determine number of 4x passes required
        # Each model pass scales by 4x.
        passes = 1
        current_ai_scale = 4.0
        while current_ai_scale < required_scale:
            passes += 1
            current_ai_scale *= 4.0

        logger.info(
            f"Resolution calculated for profile={profile_name}. "
            f"Original: {width}x{height}, Target: {target_w}x{target_h}, "
            f"Required Scale: {required_scale:.2f}x, AI Passes: {passes} (AI Scale: {current_ai_scale}x)"
        )

        return {
            "target_width": target_w,
            "target_height": target_h,
            "required_scale": required_scale,
            "passes": passes,
            "ai_scale": current_ai_scale,
        }
