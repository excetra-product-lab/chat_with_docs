"""
Error reporting utilities for structured error logging and monitoring.

This module provides utilities to log domain exceptions with structured context,
making it easier to debug issues and monitor system health.
"""

import logging
import traceback
from datetime import datetime
from typing import Any

from .exceptions import DocumentProcessingError


class ErrorReporter:
    """Simple error reporter for structured error logging."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    def report_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        operation: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """
        Report an error with structured context.

        Args:
            error: The exception that occurred
            context: Additional context (file info, request data, etc.)
            operation: Description of what operation was being performed
            user_id: Optional user identifier for correlation
        """
        error_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "operation": operation,
            "user_id": user_id,
        }

        # Add error code if it's a domain exception
        if isinstance(error, DocumentProcessingError):
            error_data["error_code"] = error.error_code

        # Add context if provided
        if context:
            error_data["context"] = context

        # Determine log level based on error type
        if isinstance(error, DocumentProcessingError):
            # Domain errors are expected, log at WARNING level
            self.logger.warning(
                f"Domain error in {operation}: {error}",
                extra={"error_data": error_data},
            )
        else:
            # Unexpected errors get ERROR level with stack trace
            error_data["stack_trace"] = traceback.format_exc()
            self.logger.error(
                f"Unexpected error in {operation}: {error}",
                extra={"error_data": error_data},
            )

    def report_validation_error(
        self,
        error: Exception,
        filename: str | None = None,
        file_size: int | None = None,
        user_id: str | None = None,
    ) -> None:
        """Report a file validation error with file context."""
        context = {}
        if filename:
            context["filename"] = filename
        if file_size:
            context["file_size"] = file_size

        self.report_error(
            error,
            context=context,
            operation="file_validation",
            user_id=user_id,
        )

    def report_processing_error(
        self,
        error: Exception,
        filename: str | None = None,
        file_type: str | None = None,
        processing_step: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Report a document processing error with processing context."""
        context = {}
        if filename:
            context["filename"] = filename
        if file_type:
            context["file_type"] = file_type
        if processing_step:
            context["processing_step"] = processing_step

        self.report_error(
            error,
            context=context,
            operation="document_processing",
            user_id=user_id,
        )


# Global error reporter instance
error_reporter = ErrorReporter()


# Convenience functions for common use cases
def report_error(
    error: Exception,
    context: dict[str, Any] | None = None,
    operation: str | None = None,
    user_id: str | None = None,
) -> None:
    """Report an error using the global error reporter."""
    error_reporter.report_error(error, context, operation, user_id)


def report_validation_error(
    error: Exception,
    filename: str | None = None,
    file_size: int | None = None,
    user_id: str | None = None,
) -> None:
    """Report a validation error using the global error reporter."""
    error_reporter.report_validation_error(error, filename, file_size, user_id)


def report_processing_error(
    error: Exception,
    filename: str | None = None,
    file_type: str | None = None,
    processing_step: str | None = None,
    user_id: str | None = None,
) -> None:
    """Report a processing error using the global error reporter."""
    error_reporter.report_processing_error(
        error, filename, file_type, processing_step, user_id
    )
