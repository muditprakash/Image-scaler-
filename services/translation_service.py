import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Dict, Optional
from utils.logging_utils import logger

# NLLB language code mapping
LANGUAGE_MAP = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Marathi": "mar_Deva"
}

class TranslationService:
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSeq2SeqLM] = None

    def load_model(self) -> None:
        """
        Loads and caches tokenizer and translation model on target device.
        """
        if self.model is None:
            logger.info(f"Loading translation model '{self.model_name}' on {self.device}...")
            
            # AutoTokenizer and AutoModelForSeq2SeqLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Map device (mps is fully supported by torch, fallback to cpu if not GPU)
            model_device = self.device
            if model_device == "mps":
                # Some operations in transformers are not fully implemented on MPS,
                # CPU is safer and fast enough for 600M param model if CUDA is not present.
                # But let's allow it or fallback.
                model_device = "cpu"
                
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(model_device) # type: ignore
            logger.info("Translation model loaded successfully.")

    def translate_text(self, text: str, target_lang: str) -> str:
        """
        Translates a single block of text into target language using NLLB-200.
        """
        if not text.strip():
            return ""

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Translation model is not loaded. Call load_model() first.")
            
        target_lang_code = LANGUAGE_MAP.get(target_lang)
        if not target_lang_code:
            raise ValueError(f"Unsupported target language: {target_lang}")

        # Set source to English as primary OCR source is English
        # NLLB requires setting src_lang and target_lang
        # We assume source text is in English (eng_Latn)
        self.tokenizer.src_lang = "eng_Latn"
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # Get target language token ID
        forced_bos_token_id = self.tokenizer.lang_code_to_id[target_lang_code]
        
        with torch.no_grad():
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=128
            )
            
        translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated_text
