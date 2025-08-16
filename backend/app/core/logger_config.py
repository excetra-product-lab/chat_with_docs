"""Central logging configuration for the backend application.

This module sets up a sensible default logging configuration that writes
logs both to the console and to a rotating file (logs/app.log). Importing
this module anywhere in the codebase will automatically configure the
root logger – there is no need to call any function explicitly.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Log directory / file setup
# ---------------------------------------------------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

# ---------------------------------------------------------------------------
# Formatter & Handlers
# ---------------------------------------------------------------------------

LOG_LEVEL = logging.INFO

_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

_file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setLevel(LOG_LEVEL)
_file_handler.setFormatter(_formatter)

_stream_handler = logging.StreamHandler()
_stream_handler.setLevel(LOG_LEVEL)
_stream_handler.setFormatter(_formatter)

# ---------------------------------------------------------------------------
# Root logger configuration – ensure idempotency
# ---------------------------------------------------------------------------

_root_logger = logging.getLogger()
if not _root_logger.handlers:  # Avoid adding duplicate handlers in reloads
    _root_logger.setLevel(LOG_LEVEL)
    _root_logger.addHandler(_file_handler)
    _root_logger.addHandler(_stream_handler)
