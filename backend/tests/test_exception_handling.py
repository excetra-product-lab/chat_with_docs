"""Tests for the exception handling system."""

from unittest.mock import patch

from fastapi import HTTPException

from app.core.error_reporter import ErrorReporter, report_error
from app.core.exceptions import (
    CorruptedDocumentError,
    DocumentProcessingError,
    DocumentValidationError,
    EmptyDocumentError,
    FileTooLargeError,
    UnsupportedFormatError,
    to_http_exception,
)


class TestExceptionHierarchy:
    """Test the custom exception hierarchy."""

    def test_base_exception(self):
        """Test base DocumentProcessingError."""
        error = DocumentProcessingError("Test error", "TEST_001")
        assert str(error) == "Test error"
        assert error.error_code == "TEST_001"

    def test_validation_error(self):
        """Test DocumentValidationError inherits correctly."""
        error = DocumentValidationError("Validation failed")
        assert isinstance(error, DocumentProcessingError)
        assert error.error_code == "DocumentValidationError"

    def test_unsupported_format_error(self):
        """Test UnsupportedFormatError."""
        error = UnsupportedFormatError("Format not supported")
        assert isinstance(error, DocumentProcessingError)
        assert str(error) == "Format not supported"


class TestExceptionConversion:
    """Test conversion of domain exceptions to HTTP exceptions."""

    def test_validation_error_to_http(self):
        """Test validation error converts to 422."""
        error = DocumentValidationError("Invalid file")
        http_exc = to_http_exception(error)
        assert isinstance(http_exc, HTTPException)
        assert http_exc.status_code == 422
        assert http_exc.detail == "Invalid file"

    def test_unsupported_format_to_http(self):
        """Test unsupported format error converts to 400."""
        error = UnsupportedFormatError("Bad format")
        http_exc = to_http_exception(error)
        assert http_exc.status_code == 400
        assert http_exc.detail == "Bad format"

    def test_processing_error_to_http(self):
        """Test general processing error converts to 500."""
        error = DocumentProcessingError("Something went wrong")
        http_exc = to_http_exception(error)
        assert http_exc.status_code == 500
        assert "Document processing failed" in http_exc.detail


class TestErrorReporter:
    """Test the error reporting functionality."""

    def test_error_reporter_basic(self):
        """Test basic error reporting."""
        with patch("app.core.error_reporter.logging.getLogger") as mock_logger:
            reporter = ErrorReporter()
            error = DocumentValidationError("Test error")

            reporter.report_error(error, operation="test_operation")

            # Verify logger was called
            mock_logger.return_value.warning.assert_called_once()

    def test_convenience_function(self):
        """Test the convenience report_error function."""
        with patch(
            "app.core.error_reporter.error_reporter.report_error"
        ) as mock_report:
            error = DocumentProcessingError("Test")
            report_error(error, operation="test")

            mock_report.assert_called_once_with(error, None, "test", None)


class TestIntegration:
    """Test integration of exceptions with existing code patterns."""

    def test_specific_exception_types(self):
        """Test that specific exception types are properly categorized."""
        # File validation errors
        file_too_large = FileTooLargeError("File exceeds limit")
        assert isinstance(file_too_large, DocumentValidationError)

        # Document content errors
        empty_doc = EmptyDocumentError("No content")
        assert isinstance(empty_doc, DocumentValidationError)

        corrupted_doc = CorruptedDocumentError("Cannot decode")
        assert isinstance(corrupted_doc, DocumentProcessingError)

        # Check they convert to appropriate HTTP status codes
        assert to_http_exception(file_too_large).status_code == 422
        assert to_http_exception(empty_doc).status_code == 422
