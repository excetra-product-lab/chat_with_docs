"""
Langchain configuration and initialization module.

This module provides centralized configuration for Langchain components,
including LLM setup, embedding models, and tracing configuration.
"""

import logging
import os
from typing import Optional, Union

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.llms import FakeListLLM
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from pydantic import SecretStr

from .settings import settings

logger = logging.getLogger(__name__)


class LangchainConfig:
    """Centralized Langchain configuration class."""

    def __init__(self) -> None:
        self._llm: Optional[Union[AzureChatOpenAI, FakeListLLM]] = None
        self._embeddings: Optional[Union[AzureOpenAIEmbeddings, FakeEmbeddings]] = None
        self._setup_langchain_environment()

    def _setup_langchain_environment(self) -> None:
        """Set up Langchain environment variables and logging."""
        # Configure LangSmith tracing if enabled
        if settings.LANGCHAIN_TRACING_V2:
            os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2

        if settings.LANGCHAIN_API_KEY:
            os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY

        if settings.LANGCHAIN_PROJECT:
            os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT

        # Configure verbose logging
        if settings.LANGCHAIN_VERBOSE:
            logging.getLogger("langchain").setLevel(logging.DEBUG)
            logging.getLogger("langchain.llms").setLevel(logging.DEBUG)
            logging.getLogger("langchain.chains").setLevel(logging.DEBUG)

    @property
    def llm(self) -> Union[AzureChatOpenAI, FakeListLLM]:
        """Get configured Azure LLM instance."""
        if self._llm is None:
            if settings.AZURE_OPENAI_API_KEY and settings.AZURE_OPENAI_ENDPOINT:
                api_key = (
                    SecretStr(settings.AZURE_OPENAI_API_KEY)
                    if settings.AZURE_OPENAI_API_KEY
                    else None
                )
                self._llm = AzureChatOpenAI(
                    api_key=api_key,
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                    azure_deployment=(
                        settings.AZURE_OPENAI_DEPLOYMENT_NAME or settings.OPENAI_MODEL
                    ),
                    api_version="2024-02-01",
                    temperature=settings.OPENAI_TEMPERATURE,
                )
                deployment = settings.AZURE_OPENAI_DEPLOYMENT_NAME or settings.OPENAI_MODEL
                logger.info(f"Initialized Azure OpenAI LLM with deployment: {deployment}")
            else:
                # Use fake LLM for testing when no Azure API key/endpoint is provided
                self._llm = FakeListLLM(responses=["This is a test response"])
                logger.warning(
                    "No Azure OpenAI API key/endpoint provided, using fake LLM for testing"
                )

        return self._llm

    @property
    def embeddings(self) -> Union[AzureOpenAIEmbeddings, FakeEmbeddings]:
        """Get configured Azure embeddings instance."""
        if self._embeddings is None:
            if settings.AZURE_OPENAI_API_KEY and settings.AZURE_OPENAI_ENDPOINT:
                api_key = (
                    SecretStr(settings.AZURE_OPENAI_API_KEY)
                    if settings.AZURE_OPENAI_API_KEY
                    else None
                )
                self._embeddings = AzureOpenAIEmbeddings(
                    api_key=api_key,
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                    azure_deployment=(
                        settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or settings.EMBEDDING_MODEL
                    ),
                    api_version="2024-02-01",
                    dimensions=settings.EMBEDDING_DIMENSION,
                )
                deployment = settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or settings.EMBEDDING_MODEL
                logger.info(f"Initialized Azure OpenAI embeddings with deployment: {deployment}")
            else:
                # Use fake embeddings for testing when no Azure API key/endpoint is provided
                self._embeddings = FakeEmbeddings(size=settings.EMBEDDING_DIMENSION)
                logger.warning(
                    "No Azure OpenAI API key/endpoint provided, using fake embeddings for testing"
                )

        return self._embeddings

    def get_chunk_config(self) -> dict:
        """Get document chunking configuration."""
        return {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "max_tokens_per_chunk": settings.MAX_TOKENS_PER_CHUNK,
        }

    def is_tracing_enabled(self) -> bool:
        """Check if LangSmith tracing is enabled."""
        return bool(settings.LANGCHAIN_TRACING_V2 and settings.LANGCHAIN_API_KEY)


# Global instance
langchain_config = LangchainConfig()
