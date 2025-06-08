"""
Pytest configuration and fixtures for Langchain testing.

This module provides shared fixtures and utilities for testing Langchain components
with proper fallbacks to fake implementations when Azure OpenAI credentials are not available.
"""

from typing import Any, Generator, List
from unittest.mock import Mock, patch

import pytest
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from langchain.schema.document import Document
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.llms import FakeListLLM
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from app.core.langchain_config import langchain_config


@pytest.fixture
def mock_langchain_config() -> Generator[Mock, None, None]:
    """
    Fixture that provides a mocked LangchainConfig for testing.

    This fixture ensures tests always use fake implementations
    regardless of whether Azure OpenAI credentials are available.
    """
    with patch("app.core.langchain_config.langchain_config") as mock_config:
        mock_config.llm = FakeListLLM(responses=["Test response"])
        mock_config.embeddings = FakeEmbeddings(size=1536)
        mock_config.get_chunk_config.return_value = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "max_tokens_per_chunk": 1000,
        }
        mock_config.is_tracing_enabled.return_value = False
        yield mock_config


@pytest.fixture
def fake_llm() -> FakeListLLM:
    """
    Fixture that provides a FakeListLLM for testing.

    Returns a FakeListLLM with predefined responses suitable for testing.
    """
    return FakeListLLM(
        responses=[
            "This is a test response for document analysis.",
            "Here is another test response for Q&A.",
            "Final test response for completion.",
        ]
    )


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    """
    Fixture that provides fake embeddings for testing.

    Returns FakeEmbeddings with the same dimension as the production embeddings.
    """
    return FakeEmbeddings(size=1536)


@pytest.fixture
def sample_documents() -> List[Document]:
    """
    Fixture that provides sample Langchain Document objects for testing.

    Returns a list of Document objects with realistic content and metadata.
    """
    return [
        Document(
            page_content="This is the first document about artificial intelligence and machine learning.",
            metadata={"source": "doc1.txt", "page": 1, "chunk_id": 0},
        ),
        Document(
            page_content="This is the second document discussing natural language processing techniques.",
            metadata={"source": "doc2.txt", "page": 1, "chunk_id": 1},
        ),
        Document(
            page_content="The third document covers deep learning architectures and neural networks.",
            metadata={"source": "doc3.txt", "page": 2, "chunk_id": 2},
        ),
    ]


@pytest.fixture
def sample_chat_messages() -> List[BaseMessage]:
    """
    Fixture that provides sample chat messages for testing conversational chains.

    Returns a list of alternating human and AI messages for testing chat functionality.
    """
    return [
        HumanMessage(content="What is artificial intelligence?"),
        AIMessage(content="Artificial intelligence is a field of computer science..."),
        HumanMessage(content="How does machine learning work?"),
        AIMessage(content="Machine learning works by training models on data..."),
    ]


@pytest.fixture
def mock_azure_credentials():
    """
    Fixture that temporarily sets Azure OpenAI environment variables for testing.

    This can be used when you need to test the Azure OpenAI configuration path
    without actually making API calls.
    """
    with patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_API_KEY": "test-api-key",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_DEPLOYMENT_NAME": "test-deployment",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "test-embedding-deployment",
        },
    ):
        yield


class LangchainTestHelper:
    """
    Helper class for Langchain-specific test utilities and assertions.

    This class provides methods for common testing patterns with Langchain components.
    """

    @staticmethod
    def assert_is_fake_llm(llm: Any) -> None:
        """Assert that the provided LLM is a fake implementation."""
        assert isinstance(llm, FakeListLLM), f"Expected FakeListLLM, got {type(llm)}"

    @staticmethod
    def assert_is_azure_llm(llm: Any) -> None:
        """Assert that the provided LLM is an Azure implementation."""
        assert isinstance(llm, AzureChatOpenAI), f"Expected AzureChatOpenAI, got {type(llm)}"

    @staticmethod
    def assert_is_fake_embeddings(embeddings: Any) -> None:
        """Assert that the provided embeddings are fake implementation."""
        assert isinstance(
            embeddings, FakeEmbeddings
        ), f"Expected FakeEmbeddings, got {type(embeddings)}"

    @staticmethod
    def assert_is_azure_embeddings(embeddings: Any) -> None:
        """Assert that the provided embeddings are Azure implementation."""
        assert isinstance(
            embeddings, AzureOpenAIEmbeddings
        ), f"Expected AzureOpenAIEmbeddings, got {type(embeddings)}"

    @staticmethod
    def assert_document_structure(doc: Document) -> None:
        """Assert that a Document has the expected structure."""
        assert hasattr(doc, "page_content"), "Document missing page_content"
        assert hasattr(doc, "metadata"), "Document missing metadata"
        assert isinstance(doc.page_content, str), "page_content must be string"
        assert isinstance(doc.metadata, dict), "metadata must be dict"

    @staticmethod
    def assert_message_structure(message: BaseMessage) -> None:
        """Assert that a message has the expected structure."""
        assert hasattr(message, "content"), "Message missing content"
        assert isinstance(message.content, str), "Message content must be string"

    @staticmethod
    def create_test_document(content: str, source: str = "test.txt", **metadata) -> Document:
        """Create a test Document with specified content and metadata."""
        default_metadata = {"source": source, "page": 1, "chunk_id": 0}
        default_metadata.update(metadata)
        return Document(page_content=content, metadata=default_metadata)

    @staticmethod
    def create_test_embedding_vector(size: int = 1536) -> List[float]:
        """Create a test embedding vector of specified size."""
        return [0.1] * size

    @staticmethod
    def mock_llm_response(llm: FakeListLLM, responses: List[str]) -> None:
        """Update a FakeListLLM with new responses."""
        llm.responses = responses
        llm.i = 0  # Reset response index


@pytest.fixture
def langchain_helper() -> LangchainTestHelper:
    """
    Fixture that provides the LangchainTestHelper utility class.

    This fixture gives access to all the testing utilities in the helper class.
    """
    return LangchainTestHelper()


# Async fixtures for testing async Langchain operations
@pytest.fixture
async def async_fake_llm() -> FakeListLLM:
    """
    Async fixture for FakeListLLM.

    Useful when testing async chains that require an LLM.
    """
    return FakeListLLM(responses=["Async test response"])


@pytest.fixture
async def async_fake_embeddings() -> FakeEmbeddings:
    """
    Async fixture for FakeEmbeddings.

    Useful when testing async embedding operations.
    """
    return FakeEmbeddings(size=1536)


# Integration fixtures that work with the actual LangchainConfig
@pytest.fixture
def ensure_fake_config():
    """
    Fixture that ensures the global langchain_config uses fake implementations.

    This fixture temporarily modifies the global config to use fake implementations
    and restores the original state after the test.
    """
    # Store original instances
    original_llm = langchain_config._llm
    original_embeddings = langchain_config._embeddings

    # Force fake implementations
    langchain_config._llm = FakeListLLM(responses=["Test response"])
    langchain_config._embeddings = FakeEmbeddings(size=1536)

    try:
        yield langchain_config
    finally:
        # Restore original instances
        langchain_config._llm = original_llm
        langchain_config._embeddings = original_embeddings
