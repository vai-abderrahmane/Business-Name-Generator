import logging
import os
from typing import Optional

LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv(
    "APP_LOG_FORMAT",
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

_logger_configured = False

def configure_logging(level: Optional[str] = None) -> None:
    global _logger_configured
    if _logger_configured:
        return
    lvl = getattr(logging, (level or LOG_LEVEL), logging.INFO)
    logging.basicConfig(level=lvl, format=LOG_FORMAT)
    _logger_configured = True

def get_logger(name: str) -> logging.Logger:
    if not _logger_configured:
        configure_logging()
    return logging.getLogger(name)
