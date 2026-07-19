import os
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    MODEL_PATH: str = Field(
        default="weights/RealESRGAN_x4plus.pth",
        description="Path to the model weights file"
    )
    OUTPUT_DIRECTORY: str = Field(
        default="output",
        description="Directory to save upscaled images"
    )
    DEVICE: str = Field(
        default="auto",
        description="Device to run inference on: 'cuda', 'cpu', 'mps', or 'auto'"
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    OCR_ENGINE: str = Field(
        default="PaddleOCR",
        description="OCR Engine choice: 'PaddleOCR' or 'EasyOCR'"
    )
    TRANSLATION_MODEL: str = Field(
        default="facebook/nllb-200-distilled-600M",
        description="Hugging Face translation model name"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
