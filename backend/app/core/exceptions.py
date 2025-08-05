"""
Custom exception hierarchy for the RAG (Retrieval-Augmented Generation) pipeline.

This module defines a comprehensive set of custom exceptions to handle various
failure scenarios within the RAG system, enabling precise error handling and
appropriate HTTP response mapping.
"""

import logging
from typing import Any


class RAGPipelineError(Exception):
    """
    Base exception class for all RAG pipeline errors.

    This serves as the parent class for all custom exceptions in the RAG system,
    allowing for broad exception catching when needed.
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize the RAG pipeline error.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error context
            original_error: Optional original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.original_error = original_error

        # Log the error for monitoring and debugging
        logger = logging.getLogger(__name__)
        logger.error(
            f"RAGPipelineError: {message}",
            extra={
                "error_details": self.details,
                "original_error": str(original_error) if original_error else None,
            },
        )


class DocumentNotFoundError(RAGPipelineError):
    """
    Exception raised when no relevant documents are found for a query.

    This error occurs when the vector similarity search returns no results
    or when all retrieved chunks are below the relevance threshold.
    """

    def __init__(
        self,
        query: str,
        user_id: int | None = None,
        threshold: float | None = None,
    ):
        message = f"No relevant documents found for query: '{query[:50]}...'"
        details = {
            "query": query,
            "user_id": user_id,
            "similarity_threshold": threshold,
            "error_code": "DOCUMENT_NOT_FOUND",
        }
        super().__init__(message, details)


class AzureAPIError(RAGPipelineError):
    """
    Exception raised for Azure OpenAI API-related failures.

    This includes rate limits, authentication errors, service unavailability,
    and other API-specific issues.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_type: str | None = None,
        original_error: Exception | None = None,
    ):
        details = {
            "status_code": status_code,
            "error_type": error_type,
            "error_code": "AZURE_API_ERROR",
        }
        super().__init__(message, details, original_error)


class CitationParsingError(RAGPipelineError):
    """
    Exception raised when citation extraction or parsing fails.

    This occurs when the LLM response contains malformed citations that
    cannot be mapped back to source documents.
    """

    def __init__(
        self, citation_text: str, reason: str, chunks_count: int | None = None
    ):
        message = f"Failed to parse citation '{citation_text}': {reason}"
        details = {
            "citation_text": citation_text,
            "parsing_failure_reason": reason,
            "available_chunks": chunks_count,
            "error_code": "CITATION_PARSING_ERROR",
        }
        super().__init__(message, details)


class EmptyContextError(RAGPipelineError):
    """
    Exception raised when no valid context can be built from retrieved chunks.

    This occurs when retrieved chunks are empty, corrupted, or contain
    insufficient information to generate a meaningful response.
    """

    def __init__(self, chunks_retrieved: int = 0, reason: str | None = None):
        message = f"Cannot build context from {chunks_retrieved} chunks"
        if reason:
            message += f": {reason}"

        details = {
            "chunks_retrieved": chunks_retrieved,
            "context_failure_reason": reason,
            "error_code": "EMPTY_CONTEXT_ERROR",
        }
        super().__init__(message, details)


class VectorSearchError(RAGPipelineError):
    """
    Exception raised when vector similarity search fails.

    This includes database connection issues, index problems, or
    embedding-related failures during the search process.
    """

    def __init__(
        self,
        query: str,
        reason: str,
        user_id: int | None = None,
        original_error: Exception | None = None,
    ):
        message = f"Vector search failed for query '{query[:50]}...': {reason}"
        details = {
            "query": query,
            "user_id": user_id,
            "search_failure_reason": reason,
            "error_code": "VECTOR_SEARCH_ERROR",
        }
        super().__init__(message, details, original_error)


class TokenLimitExceededError(RAGPipelineError):
    """
    Exception raised when input exceeds model token limits.

    This occurs when the combined question, context, and system prompt
    exceed the model's maximum context window.
    """

    def __init__(
        self, token_count: int, token_limit: int, component: str | None = None
    ):
        message = f"Token limit exceeded: {token_count} tokens (limit: {token_limit})"
        if component:
            message += f" in {component}"

        details = {
            "token_count": token_count,
            "token_limit": token_limit,
            "exceeding_component": component,
            "error_code": "TOKEN_LIMIT_EXCEEDED",
        }
        super().__init__(message, details)


class LLMGenerationError(RAGPipelineError):
    """
    Exception raised when LLM fails to generate a valid response.

    This includes empty responses, malformed outputs, or other
    generation-specific failures not covered by Azure API errors.
    """

    def __init__(
        self,
        reason: str,
        question: str | None = None,
        attempts: int | None = None,
        original_error: Exception | None = None,
    ):
        message = f"LLM generation failed: {reason}"
        details = {
            "generation_failure_reason": reason,
            "question": question[:100] if question else None,
            "retry_attempts": attempts,
            "error_code": "LLM_GENERATION_ERROR",
        }
        super().__init__(message, details, original_error)


class AuthenticationError(RAGPipelineError):
    """
    Exception raised for authentication and authorization failures.

    This includes invalid API keys, expired tokens, or insufficient
    permissions for accessing resources.
    """

    def __init__(
        self,
        message: str,
        user_id: int | None = None,
        resource: str | None = None,
    ):
        details = {
            "user_id": user_id,
            "protected_resource": resource,
            "error_code": "AUTHENTICATION_ERROR",
        }
        super().__init__(message, details)


class ConfigurationError(RAGPipelineError):
    """
    Exception raised for configuration-related issues.

    This includes missing environment variables, invalid settings,
    or misconfigured services.
    """

    def __init__(
        self, setting_name: str, reason: str, current_value: str | None = None
    ):
        message = f"Configuration error for '{setting_name}': {reason}"
        details = {
            "setting_name": setting_name,
            "configuration_issue": reason,
            "current_value": current_value,
            "error_code": "CONFIGURATION_ERROR",
        }
        super().__init__(message, details)
