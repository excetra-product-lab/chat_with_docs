"""Azure OpenAI embedding service for generating vector embeddings from text chunks."""

import logging

from fastapi import HTTPException
from openai import AsyncAzureOpenAI

from app.core.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using Azure OpenAI."""

    def __init__(self) -> None:
        """Initialize the embedding service with Azure OpenAI client."""
        if (
            not settings.AZURE_OPENAI_API_KEY
            or not settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        ):
            raise ValueError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_EMBEDDING_DEPLOYMENT must be configured"
            )

        self.client = AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_version=settings.AZURE_OPENAI_EMBEDDING_API_VERSION,
        )

        # Use embedding deployment name or fallback to embedding model name
        # todo: change this to env variable
        self.embedding_deployment = "text-embedding-3-small"

        logger.info(
            f"Initialized Azure OpenAI embedding service with deployment: {self.embedding_deployment}"
        )

    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for a single text string.

        Args:
            text: The text to generate embedding for

        Returns:
            list[float]: The embedding vector

        Raises:
            HTTPException: If embedding generation fails
        """
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_deployment,
                input=text,
            )

            embedding = response.data[0].embedding
            logger.debug(
                f"Generated embedding of length {len(embedding)} for text of length {len(text)}"
            )

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to generate embedding: {str(e)}"
            ) from e

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple text strings in batch.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            list[list[float]]: List of embedding vectors

        Raises:
            HTTPException: If embedding generation fails
        """
        if not texts:
            return []

        try:
            # Azure OpenAI supports batch embedding requests
            response = await self.client.embeddings.create(
                model=self.embedding_deployment,
                input=texts,
            )

            embeddings = [item.embedding for item in response.data]
            logger.info(
                f"Generated {len(embeddings)} embeddings for batch of {len(texts)} texts"
            )

            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to generate batch embeddings: {str(e)}"
            ) from e


# Global embedding service instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """
    Get or create the global embedding service instance.

    Returns:
        EmbeddingService: The embedding service instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
