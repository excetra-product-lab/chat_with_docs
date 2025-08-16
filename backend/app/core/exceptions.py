"""
Simple exception hierarchy for the document processing system.

This module provides domain-specific exceptions that integrate with the existing
HTTPException-based API error handling while providing clear error categorization.
"""


class DocumentProcessingError(Exception):
    """Base exception for all document processing errors."""

    def __init__(self, message: str, error_code: str | None = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        super().__init__(self.message)


class DocumentValidationError(DocumentProcessingError):
    """Raised when document validation fails (maps to HTTP 422)."""

    pass


class DocumentParsingError(DocumentProcessingError):
    """Raised when document parsing fails."""

    pass


class DocumentChunkingError(DocumentProcessingError):
    """Raised when document chunking fails."""

    pass


class UnsupportedFormatError(DocumentProcessingError):
    """Raised when file format is not supported (maps to HTTP 400)."""

    pass


class FileTooLargeError(DocumentValidationError):
    """Raised when file exceeds size limits."""

    pass


class EmptyDocumentError(DocumentValidationError):
    """Raised when document contains no content."""

    pass


class CorruptedDocumentError(DocumentParsingError):
    """Raised when document is corrupted or unreadable."""

    pass


class EmbeddingError(DocumentProcessingError):
    """Raised when embedding generation fails."""

    pass


class VectorStoreError(DocumentProcessingError):
    """Raised when vector store operations fail."""

    pass


# Utility function to convert domain exceptions to HTTP exceptions
def to_http_exception(error: DocumentProcessingError):
    """Convert domain exceptions to FastAPI HTTPException."""
    from fastapi import HTTPException

    # Validation errors -> 422 (Unprocessable Entity)
    if isinstance(error, DocumentValidationError):
        return HTTPException(status_code=422, detail=error.message)

    # Unsupported format -> 400 (Bad Request)
    if isinstance(error, UnsupportedFormatError):
        return HTTPException(status_code=400, detail=error.message)

    # All other processing errors -> 500 (Internal Server Error)
    return HTTPException(
        status_code=500, detail=f"Document processing failed: {error.message}"
    )
