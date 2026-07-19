import logging
import json
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="system")

class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "request_id": request_id_ctx.get(),
        }
        
        # Include extra attributes passed to the logger
        if hasattr(record, "extra_info") and isinstance(record.extra_info, dict): # type: ignore
            log_data.update(record.extra_info)
        elif hasattr(record, "extra") and isinstance(record.extra, dict):
            log_data.update(record.extra)

        # Include standard exception details if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    root_logger = logging.getLogger("upscaler")
    root_logger.setLevel(log_level.upper())
    
    # Remove existing handlers
    if root_logger.handlers:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    formatter = StructuredFormatter()
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Prevent propagation to root to avoid duplicate outputs
    root_logger.propagate = False
    return root_logger

logger = setup_logging()
