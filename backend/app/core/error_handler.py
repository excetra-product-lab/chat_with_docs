"""
Global error handler for the RAG pipeline with HTTP status code mapping
and user-friendly error message formatting.

This module provides centralized error handling, ensuring consistent
API responses and proper logging for all RAG pipeline failures.
"""

from typing import Dict, Any, Optional, Tuple
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import logging
import uuid
from datetime import datetime

from .exceptions import (
    RAGPipelineError,
    DocumentNotFoundError,
    AzureAPIError,
    CitationParsingError, 
    EmptyContextError,
    VectorSearchError,
    TokenLimitExceededError,
    LLMGenerationError,
    AuthenticationError,
    ConfigurationError
)


class ErrorHandler:
    """
    Central error handler for the RAG pipeline.
    
    Maps custom exceptions to appropriate HTTP status codes and
    generates consistent, user-friendly error responses.
    """
    
    # HTTP status code mapping for custom exceptions
    STATUS_CODE_MAPPING: Dict[type, int] = {
        DocumentNotFoundError: 404,  # Not Found
        AzureAPIError: 502,  # Bad Gateway (external service issue)
        CitationParsingError: 500,  # Internal Server Error
        EmptyContextError: 422,  # Unprocessable Entity (input valid but insufficient)
        VectorSearchError: 503,  # Service Unavailable (database/search issue)
        TokenLimitExceededError: 413,  # Payload Too Large
        LLMGenerationError: 500,  # Internal Server Error
        AuthenticationError: 401,  # Unauthorized
        ConfigurationError: 500,  # Internal Server Error
        RAGPipelineError: 500,  # Internal Server Error (fallback)
    }
    
    # User-friendly error messages that don't expose internal details
    USER_FRIENDLY_MESSAGES: Dict[type, str] = {
        DocumentNotFoundError: "I couldn't find any relevant information in your documents to answer this question. Please try rephrasing your question or uploading additional documents.",
        AzureAPIError: "I'm experiencing issues connecting to the AI service. Please try again in a few moments.",
        CitationParsingError: "I encountered an issue while linking my response to your source documents. The answer may still be accurate, but citations might be incomplete.",
        EmptyContextError: "I don't have enough information from your documents to provide a comprehensive answer. Please try uploading more relevant documents or asking a more specific question.",
        VectorSearchError: "I'm having trouble searching through your documents right now. Please try again later.",
        TokenLimitExceededError: "Your question or the relevant documents are too long for me to process. Please try asking a shorter question or work with fewer documents.",
        LLMGenerationError: "I'm unable to generate a response right now due to a technical issue. Please try again.",
        AuthenticationError: "You don't have permission to perform this action. Please check your authentication.",
        ConfigurationError: "There's a configuration issue preventing me from processing your request. Please contact support.",
        RAGPipelineError: "I encountered an unexpected issue while processing your request. Please try again later."
    }
    
    # Error codes for client-side handling
    ERROR_CODES: Dict[type, str] = {
        DocumentNotFoundError: "DOCUMENT_NOT_FOUND",
        AzureAPIError: "AI_SERVICE_ERROR",
        CitationParsingError: "CITATION_ERROR",
        EmptyContextError: "INSUFFICIENT_CONTEXT",
        VectorSearchError: "SEARCH_ERROR",
        TokenLimitExceededError: "INPUT_TOO_LONG",
        LLMGenerationError: "GENERATION_ERROR",
        AuthenticationError: "AUTHENTICATION_ERROR",
        ConfigurationError: "CONFIGURATION_ERROR",
        RAGPipelineError: "UNKNOWN_ERROR"
    }
    
    @classmethod
    def get_error_response(
        cls, 
        error: Exception, 
        request_id: Optional[str] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Generate standardized error response for any exception.
        
        Args:
            error: The exception that occurred
            request_id: Optional unique request identifier
            
        Returns:
            Tuple of (status_code, response_body)
        """
        logger = logging.getLogger(__name__)
        
        # Generate unique request ID if not provided
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Determine error type and get appropriate mappings
        error_type = type(error)
        status_code = cls._get_status_code(error_type)
        error_code = cls._get_error_code(error_type)
        user_message = cls._get_user_message(error_type, error)
        
        # Extract additional details if available
        details = {}
        if isinstance(error, RAGPipelineError):
            details = error.details.copy()
            # Remove sensitive information
            details.pop('original_error', None)
        
        # Structure the error response
        error_response = {
            "error": {
                "code": error_code,
                "message": user_message,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
        
        # Add debug information in development (based on details)
        if details.get("include_debug", False):
            error_response["debug"] = {
                "error_type": error_type.__name__,
                "details": details
            }
        
        # Log the error for monitoring
        cls._log_error(error, request_id, status_code, logger)
        
        return status_code, error_response
    
    @classmethod
    def _get_status_code(cls, error_type: type) -> int:
        """Get HTTP status code for error type."""
        # Check for exact match first
        if error_type in cls.STATUS_CODE_MAPPING:
            return cls.STATUS_CODE_MAPPING[error_type]
        
        # Check inheritance hierarchy
        for exception_class, status_code in cls.STATUS_CODE_MAPPING.items():
            if issubclass(error_type, exception_class):
                return status_code
        
        # Default fallback
        return 500
    
    @classmethod
    def _get_error_code(cls, error_type: type) -> str:
        """Get error code for error type."""
        # Check for exact match first
        if error_type in cls.ERROR_CODES:
            return cls.ERROR_CODES[error_type]
        
        # Check inheritance hierarchy
        for exception_class, error_code in cls.ERROR_CODES.items():
            if issubclass(error_type, exception_class):
                return error_code
        
        # Default fallback
        return "UNKNOWN_ERROR"
    
    @classmethod
    def _get_user_message(cls, error_type: type, error: Exception) -> str:
        """Get user-friendly message for error type."""
        # Check for exact match first
        if error_type in cls.USER_FRIENDLY_MESSAGES:
            return cls.USER_FRIENDLY_MESSAGES[error_type]
        
        # Check inheritance hierarchy
        for exception_class, message in cls.USER_FRIENDLY_MESSAGES.items():
            if issubclass(error_type, exception_class):
                return message
        
        # Use error message if it's user-friendly (for RAGPipelineError)
        if isinstance(error, RAGPipelineError):
            return error.message
        
        # Default fallback
        return "An unexpected error occurred. Please try again later."
    
    @classmethod
    def _log_error(
        cls, 
        error: Exception, 
        request_id: str, 
        status_code: int, 
        logger: logging.Logger
    ) -> None:
        """Log error with structured information for monitoring."""
        log_data = {
            "request_id": request_id,
            "error_type": type(error).__name__,
            "status_code": status_code,
            "error_message": str(error)
        }
        
        # Add additional context for RAG pipeline errors
        if isinstance(error, RAGPipelineError):
            log_data.update(error.details)
        
        # Log at appropriate level based on status code
        if status_code >= 500:
            logger.error(f"Server error in RAG pipeline: {str(error)}", extra=log_data)
        elif status_code >= 400:
            logger.warning(f"Client error in RAG pipeline: {str(error)}", extra=log_data)
        else:
            logger.info(f"RAG pipeline response: {str(error)}", extra=log_data)


class FallbackResponseManager:
    """
    Manages fallback responses for recoverable error scenarios.
    
    Provides appropriate fallback messages when the system can still
    provide some value to the user despite encountering issues.
    """
    
    @classmethod
    def get_fallback_response(cls, error: Exception) -> Optional[Dict[str, Any]]:
        """
        Generate fallback response for recoverable errors.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Fallback response dict or None if no fallback available
        """
        if isinstance(error, DocumentNotFoundError):
            return {
                "answer": "I couldn't find specific information in your documents to answer this question. This might be because:\n\n1. The relevant information isn't in your uploaded documents\n2. The question might need to be more specific\n3. The information might be phrased differently in your documents\n\nTry rephrasing your question or uploading additional relevant documents.",
                "citations": [],
                "confidence": 0.0,
                "fallback_reason": "no_documents_found"
            }
        
        elif isinstance(error, EmptyContextError):
            return {
                "answer": "I found some potentially relevant documents, but I couldn't extract enough clear information to provide a confident answer. Please try:\n\n1. Asking a more specific question\n2. Uploading documents with clearer, more detailed information\n3. Checking if your documents contain the information you're looking for",
                "citations": [],
                "confidence": 0.0,
                "fallback_reason": "insufficient_context"
            }
        
        elif isinstance(error, CitationParsingError):
            # For citation errors, we might still have a generated answer
            # This would need to be handled in the calling code
            return None
        
        elif isinstance(error, TokenLimitExceededError):
            return {
                "answer": "Your request is too large for me to process. Please try:\n\n1. Asking a shorter, more focused question\n2. Working with fewer documents at once\n3. Breaking down complex questions into smaller parts",
                "citations": [],
                "confidence": 0.0,
                "fallback_reason": "input_too_large"
            }
        
        # No fallback available for other error types
        return None


def create_error_response(error: Exception, request_id: Optional[str] = None) -> JSONResponse:
    """
    Create a FastAPI JSONResponse for any exception.
    
    Args:
        error: The exception that occurred
        request_id: Optional unique request identifier
        
    Returns:
        FastAPI JSONResponse with appropriate status code and error message
    """
    status_code, response_body = ErrorHandler.get_error_response(error, request_id)
    return JSONResponse(
        status_code=status_code,
        content=response_body
    )