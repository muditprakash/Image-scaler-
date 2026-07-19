# AI Image Processing Suite (Upscaling & Text Translation)

A high-performance production-ready FastAPI backend designed to process images using open-source deep learning models. 

This repository exposes two primary services:
1. **AI Image Upscaler**: Scales images using open-source deep learning Super-Resolution (Real-ESRGAN), dynamically calculating dimensions and applying memory-safe tiled PyTorch inference.
2. **AI Image Text Translator**: Translates visible text in images into English, Hindi, or Marathi using PaddleOCR/EasyOCR and the `facebook/nllb-200-distilled-600M` model, cleaning the original text using OpenCV Telea Inpainting and rendering translations back in place.

---

## Features

### AI Image Upscaler
- **Dynamic Scale Detection**: Adapts target dimensions dynamically based on input aspect ratio and selected resolution profiles (`mobile`, `desktop`, `hoarding`).
- **Tiled Inference**: Splits large images into overlapping tiles to prevent OOM errors on GPUs/CPUs.
- **Multi-pass Super-Resolution**: Runs multiple 4x upscaling passes automatically.
- **Advanced Post-Processing**: Custom sharpening and bilateral denoising filters.

### AI Image Text Translator
- **OCR Engine Auto-fallback**: Runs PaddleOCR and automatically falls back to EasyOCR if dependencies are missing, capturing boxes, confidence, and rotation angle.
- **NLLB Translation**: Translates text using the local `facebook/nllb-200-distilled-600M` Hugging Face model without external API requests.
- **Background Inpainting**: Cleans background text using OpenCV Telea inpainting before rendering.
- **Rotated Unicode Rendering**: Automatically downloads and utilizes `Noto Sans Devanagari` to render translated text centered, fitted, and correctly rotated back on the image with custom contrast-aware coloring.

---

## Project Architecture

```text
image-upscaler/
├── app.py                # FastAPI bootstrapper, static mounts, and startup lifecycle warming
├── config.py             # Pydantic Settings management
├── requirements.txt      # Python dependencies
├── routes/
│   ├── upscale.py        # Upscaler API routes (/upscale, /health, /models, /profiles)
│   └── translate.py      # Translator API routes (/image/translate, /languages, /ocr/health)
├── services/
│   ├── image_service.py  # OpenCV validations and post-processing filters (sharpen, denoise)
│   ├── upscale_service.py# PyTorch pipeline logic, tiling, and pass orchestration
│   ├── resolution_service.py # Dynamic target dimensions and passes calculations
│   ├── ocr_service.py    # PaddleOCR detecting with EasyOCR fallback
│   ├── translation_service.py # NLLB-200 model loading and text translation logic
│   ├── inpaint_service.py# OpenCV Telea background inpainting
│   └── render_service.py # Rotated Devanagari text fitting and canvas compositing
├── models/
│   └── rrdbnet.py        # Pure PyTorch implementation of the Real-ESRGAN generator
├── utils/
│   └── logging_utils.py  # Structured JSON logs with Request ID Context tracking
├── static/
│   └── index.html        # Interactive client UI dashboard (dual-mode)
├── weights/              # Caches downloaded upscaler weights (RealESRGAN_x4plus.pth)
├── fonts/                # Caches downloaded Unicode fonts (NotoSansDevanagari-Regular.ttf)
├── output/               # Stores generated upscale and translation outputs
├── Dockerfile            # Container definition
└── docker-compose.yml    # Multi-container local orchestration script
```

---

## Local Installation

### 1. Set Up Virtual Environment

Ensure you have Python 3.12 installed:

```bash
# Create virtual environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Model Weights & Fonts
- **Upscaler Weights (`RealESRGAN_x4plus.pth`)**: Automatically downloaded to the `weights/` directory on application startup.
- **Devanagari Font (`NotoSansDevanagari-Regular.ttf`)**: Automatically downloaded to the `fonts/` directory on application startup.
- **NLLB Translation Model (`facebook/nllb-200-distilled-600M`)**: Tokenizer and model weights are automatically downloaded and cached by the Hugging Face `transformers` library on startup.

---

## Run the Application

```bash
python app.py
```
The server will boot on `http://localhost:8000`. You can visit `http://localhost:8000` in your web browser to open the dual dashboard, or visit `http://localhost:8000/docs` to view the Swagger/OpenAPI documentation.

---

## Running with Docker

### Run with docker-compose (Recommended)

This mounts the `weights/`, `fonts/`, and `output/` folders into the container to prevent re-downloading model weights on restarts.

```bash
# Start the container
docker-compose up --build
```
Access the application at `http://localhost:8000`.

---

## API Examples

### POST `/upscale`
```bash
curl -X 'POST' \
  'http://localhost:8000/upscale' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@demo_photo.jpg;type=image/jpeg' \
  -F 'profile=desktop' \
  -F 'sharpen=true' \
  -F 'denoise=false'
```

### POST `/image/translate`
```bash
curl -X 'POST' \
  'http://localhost:8000/image/translate' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@poster_english.jpg;type=image/jpeg' \
  -F 'target_language=Hindi'
```

#### Response
```json
{
  "status": "success",
  "detected_regions": 12,
  "translated_regions": 12,
  "target_language": "Hindi",
  "processing_time": "1.80 sec",
  "ocr_engine": "PaddleOCR",
  "translation_model": "facebook/nllb-200-distilled-600M",
  "output_url": "/output/6a8c77b2-f8c6-47b2-bd74-e35b0dbe422c.jpg"
}
```

### GET `/languages`
```bash
curl http://localhost:8000/languages
```
```json
[
  "English",
  "Hindi",
  "Marathi"
]
```

