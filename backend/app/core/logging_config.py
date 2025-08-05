"""
Structured logging configuration for the RAG pipeline.

This module sets up comprehensive logging with structured JSON output
for better monitoring, debugging, and alerting capabilities.
"""

import json
import logging
import logging.config
import os
from collections.abc import MutableMapping
from datetime import datetime
from typing import Any


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured JSON logs.

    This formatter ensures all log messages are output in a consistent
    JSON format that can be easily parsed by log aggregation systems.
    """

    def __init__(self, include_extra: bool = True):
        """
        Initialize the structured formatter.

        Args:
            include_extra: Whether to include extra fields from log records
        """
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as structured JSON.

        Args:
            record: The log record to format

        Returns:
            JSON-formatted log message
        """
        # Base log structure
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add process/thread info for debugging
        log_entry.update({"process_id": record.process, "thread_id": record.thread})

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
                if record.exc_info
                else None,
            }

        # Include extra fields if enabled
        if self.include_extra and hasattr(record, "__dict__"):
            # Get extra fields (excluding standard logging attributes)
            standard_attrs = {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
            }

            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in standard_attrs and not key.startswith("_"):
                    # Ensure the value is JSON serializable
                    try:
                        json.dumps(value)
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)

            if extra_fields:
                log_entry["extra"] = extra_fields

        # Convert to JSON string
        try:
            return json.dumps(log_entry, default=str, separators=(",", ":"))
        except Exception:
            # Fallback to basic format if JSON serialization fails
            return f"{datetime.utcnow().isoformat()}Z - {record.levelname} - {record.name} - {record.getMessage()}"


class RAGPipelineLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds RAG pipeline context to all log messages.

    This adapter automatically includes request context and pipeline
    stage information in log messages.
    """

    def __init__(self, logger: logging.Logger, extra: dict[str, Any] | None = None):
        """
        Initialize the logger adapter.

        Args:
            logger: The base logger instance
            extra: Additional context to include in all log messages
        """
        super().__init__(logger, extra or {})

    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> tuple:
        """
        Process log message to add extra context.

        Args:
            msg: The log message
            kwargs: Additional keyword arguments

        Returns:
            Tuple of (message, kwargs) with extra context added
        """
        # Merge adapter extra with kwargs extra
        if "extra" not in kwargs:
            kwargs["extra"] = {}

        kwargs["extra"].update(self.extra)

        # Add pipeline context if available
        kwargs["extra"]["component"] = "rag_pipeline"

        return msg, kwargs


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "structured",
    log_file: str | None = None,
    enable_console: bool = True,
) -> None:
    """
    Set up structured logging configuration for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ("structured" for JSON, "simple" for plain text)
        log_file: Optional file path for log output
        enable_console: Whether to enable console logging
    """
    # Validate log level (ensure it exists)
    getattr(logging, log_level.upper(), logging.INFO)

    # Configure handlers
    handlers = {}

    if enable_console:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "level": log_level.upper(),
            "stream": "ext://sys.stdout",
        }

    if log_file:
        handlers["file"] = {
            "class": "logging.FileHandler",
            "level": log_level.upper(),
            "filename": log_file,
            "mode": "a",
        }

    # Configure formatters
    formatters = {}

    if log_format.lower() == "structured":
        formatters["structured"] = {"()": StructuredFormatter, "include_extra": True}
        formatter_name = "structured"
    else:
        formatters["simple"] = {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
        formatter_name = "simple"

    # Apply formatter to all handlers
    for handler_config in handlers.values():
        handler_config["formatter"] = formatter_name

    # Logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": {
            "": {  # Root logger
                "level": log_level.upper(),
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
            "app": {  # Application logger
                "level": log_level.upper(),
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
            "uvicorn": {  # Uvicorn logger
                "level": log_level.upper(),
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
            "uvicorn.access": {  # Uvicorn access logger
                "level": log_level.upper(),
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
        },
    }

    # Apply the configuration
    logging.config.dictConfig(config)


def get_rag_logger(
    name: str, extra_context: dict[str, Any] | None = None
) -> RAGPipelineLoggerAdapter:
    """
    Get a RAG pipeline logger with structured context.

    Args:
        name: Logger name (usually __name__)
        extra_context: Additional context to include in all log messages

    Returns:
        RAG pipeline logger adapter
    """
    base_logger = logging.getLogger(name)
    return RAGPipelineLoggerAdapter(base_logger, extra_context)


def configure_application_logging():
    """
    Configure logging for the entire application.

    This function should be called during application startup to ensure
    proper logging configuration across all modules.
    """
    # Get configuration from environment variables
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_format = os.getenv("LOG_FORMAT", "structured")
    log_file = os.getenv("LOG_FILE")
    enable_console = os.getenv("ENABLE_CONSOLE_LOGGING", "true").lower() == "true"

    # Set up logging
    setup_logging(
        log_level=log_level,
        log_format=log_format,
        log_file=log_file,
        enable_console=enable_console,
    )

    # Log configuration startup
    logger = get_rag_logger(__name__)
    logger.info(
        "Logging configuration initialized",
        extra={
            "log_level": log_level,
            "log_format": log_format,
            "log_file": log_file,
            "console_enabled": enable_console,
        },
    )
