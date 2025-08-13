"""
Pytest configuration and fixtures for Langchain testing.

This module provides shared fixtures and utilities for testing Langchain components
with proper fallbacks to fake implementations when Azure OpenAI credentials are not available.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import Mock, patch

import pytest
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from langchain.schema.document import Document
from langchain_community.embeddings import FakeEmbeddings
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from app.core.langchain_config import langchain_config


# Patch the embedding service before any imports that might use it
@pytest.fixture(scope="session", autouse=True)
def mock_embedding_service_for_collection():
    """
    Session-scoped fixture that mocks the embedding service during test collection.

    This prevents the embedding service from being initialized with real Azure credentials
    during test collection, which would fail in environments without those credentials.
    """
    # Mock the embedding service to return a fake implementation
    with patch(
        "app.services.embedding_service.get_embedding_service"
    ) as mock_get_service:
        mock_service = Mock()
        mock_service.generate_embedding.return_value = [0.1] * 1536
        mock_service.generate_embeddings_batch.return_value = [
            [0.1] * 1536 for _ in range(10)
        ]
        mock_get_service.return_value = mock_service

        # Also patch the EmbeddingService class to prevent direct instantiation issues
        with patch("app.services.embedding_service.EmbeddingService") as mock_class:
            mock_class.return_value = mock_service
            yield


# Fix LangChain compatibility issues by creating simple mock replacements
class SimpleFakeLLM:
    """Simple mock LLM that avoids Pydantic validation issues."""

    def __init__(self, responses=None):
        self.responses = responses or ["Test response"]
        self.i = 0

    def __call__(self, *args, **kwargs):
        response = self.responses[self.i % len(self.responses)]
        self.i += 1
        return response

    def invoke(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)


# Try to use the real FakeListLLM, fall back to our simple version
try:
    from langchain_community.llms import FakeListLLM

    # Try to create a test instance to see if it works
    test_llm = FakeListLLM(responses=["test"])

    # If we get here, FakeListLLM works
    USE_REAL_FAKE_LLM = True
except Exception:
    # FakeListLLM has issues, use our simple version
    FakeListLLM = SimpleFakeLLM
    USE_REAL_FAKE_LLM = False

# Similar approach for other models if needed
try:
    FakeEmbeddings.model_rebuild()
    AzureChatOpenAI.model_rebuild()
except Exception:
    pass  # These might work without rebuilding


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
def sample_documents() -> list[Document]:
    """
    Fixture that provides sample Langchain Document objects for testing.

    Returns a list of Document objects with realistic content and metadata.
    """
    return [
        Document(
            page_content=(
                "This is the first document about artificial intelligence and machine learning."
            ),
            metadata={"source": "doc1.txt", "page": 1, "chunk_id": 0},
        ),
        Document(
            page_content=(
                "This is the second document discussing natural language processing techniques."
            ),
            metadata={"source": "doc2.txt", "page": 1, "chunk_id": 1},
        ),
        Document(
            page_content=(
                "The third document covers deep learning architectures and neural networks."
            ),
            metadata={"source": "doc3.txt", "page": 2, "chunk_id": 2},
        ),
    ]


@pytest.fixture
def sample_chat_messages() -> list[BaseMessage]:
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
            "AZURE_OPENAI_API_KEY": "test-api-key",  # pragma: allowlist secret
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_API_VERSION": "2024-08-01-preview",
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
        if USE_REAL_FAKE_LLM:
            assert isinstance(llm, FakeListLLM), (
                f"Expected FakeListLLM, got {type(llm)}"
            )
        else:
            assert isinstance(llm, SimpleFakeLLM), (
                f"Expected SimpleFakeLLM, got {type(llm)}"
            )

    @staticmethod
    def assert_is_azure_llm(llm: Any) -> None:
        """Assert that the provided LLM is an Azure implementation."""
        assert isinstance(llm, AzureChatOpenAI), (
            f"Expected AzureChatOpenAI, got {type(llm)}"
        )

    @staticmethod
    def assert_is_fake_embeddings(embeddings: Any) -> None:
        """Assert that the provided embeddings are fake implementation."""
        assert isinstance(embeddings, FakeEmbeddings), (
            f"Expected FakeEmbeddings, got {type(embeddings)}"
        )

    @staticmethod
    def assert_is_azure_embeddings(embeddings: Any) -> None:
        """Assert that the provided embeddings are Azure implementation."""
        assert isinstance(embeddings, AzureOpenAIEmbeddings), (
            f"Expected AzureOpenAIEmbeddings, got {type(embeddings)}"
        )

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
    def create_test_document(
        content: str, source: str = "test.txt", **metadata
    ) -> Document:
        """Create a test Document with specified content and metadata."""
        default_metadata = {"source": source, "page": 1, "chunk_id": 0}
        default_metadata.update(metadata)
        return Document(page_content=content, metadata=default_metadata)

    @staticmethod
    def create_test_embedding_vector(size: int = 1536) -> list[float]:
        """Create a test embedding vector of specified size."""
        return [0.1] * size

    @staticmethod
    def mock_llm_response(llm: FakeListLLM, responses: list[str]) -> None:
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


# =============================================================================
# Database Testing Fixtures
# =============================================================================


@pytest.fixture(scope="function")
def test_db():
    """Create an in-memory SQLite database for testing."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.models.database import Base

    # Create in-memory SQLite database
    engine = create_engine("sqlite:///:memory:", echo=False)
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Create all tables
    Base.metadata.create_all(bind=engine)

    try:
        yield TestSessionLocal
    finally:
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def test_db_session(test_db):
    """Provide a database session for testing."""
    session = test_db()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function", autouse=True)
def mock_database_settings(monkeypatch):
    """Mock database settings to use in-memory SQLite for testing."""
    # Override the DATABASE_URL environment variable
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")

    # Also patch the settings object directly to ensure tests use SQLite
    from app.core.settings import settings

    monkeypatch.setattr(settings, "DATABASE_URL", "sqlite:///:memory:")

    # Patch the enhanced_vectorstore module's engine and session
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.models.database import Base

    # Create test engine and session
    test_engine = create_engine("sqlite:///:memory:", echo=False)
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    # Create all tables in the test database
    Base.metadata.create_all(bind=test_engine)

    # Patch the enhanced_vectorstore module
    try:
        import app.services.enhanced_vectorstore

        monkeypatch.setattr(app.services.enhanced_vectorstore, "engine", test_engine)
        monkeypatch.setattr(
            app.services.enhanced_vectorstore, "SessionLocal", TestSessionLocal
        )
    except ImportError:
        pass  # Module not imported yet

    yield

    # Clean up
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def mock_enhanced_vectorstore(test_db_session):
    """Mock EnhancedVectorStore to use test database."""
    from unittest.mock import AsyncMock, Mock, patch

    from app.services.enhanced_vectorstore import EnhancedVectorStore
    # EnhancedCitation import removed

    with patch("app.services.enhanced_vectorstore.SessionLocal") as mock_session:
        mock_session.return_value = test_db_session

        # Mock the embedding service to avoid API calls
        mock_embedding_service = Mock()
        mock_embedding_service.generate_embeddings_batch.return_value = [
            [0.1] * 1536
            for _ in range(10)  # Return fake embeddings
        ]

        with patch(
            "app.services.enhanced_vectorstore.get_embedding_service"
        ) as mock_get_embedding:
            mock_get_embedding.return_value = mock_embedding_service

            # Create the vector store instance
            vector_store = EnhancedVectorStore(embedding_service=mock_embedding_service)

            # Mock the vector search methods to avoid PostgreSQL-specific operations
            async def mock_enhanced_similarity_search(
                query,
                user_id=None,
                k=5,
                include_hierarchy=True,
                include_relationships=True,
                confidence_threshold=0.7,
                **kwargs,
            ):
                """Mock enhanced similarity search - citations removed."""
                # Return empty list since citations were removed
                return []

            async def mock_store_enhanced_document(enhanced_doc):
                """Mock document storage to avoid database operations."""
                return {"document_id": "test-doc-123", "chunks_stored": 5}

            # Patch the methods
            vector_store.enhanced_similarity_search = AsyncMock(
                side_effect=mock_enhanced_similarity_search
            )
            vector_store.store_enhanced_document = AsyncMock(
                side_effect=mock_store_enhanced_document
            )

            yield vector_store


@pytest.fixture(scope="function")
def mock_vector_operations():
    """Mock PostgreSQL vector operations for SQLite compatibility."""
    # This fixture is available if needed, but the enhanced_vectorstore fixture
    # handles all the vector operation mocking we need
    yield


@pytest.fixture(scope="function")
def performance_embedding_service():
    """
    Fixture that provides a mock embedding service for performance testing.

    This fixture provides a mock embedding service that simulates realistic
    performance characteristics for benchmarking tests.
    """
    import asyncio
    import random
    from unittest.mock import AsyncMock, Mock

    mock_service = Mock()

    async def mock_generate_embedding(text: str) -> list[float]:
        """Mock embedding generation with simulated delay."""
        # Simulate realistic API delay
        await asyncio.sleep(random.uniform(0.01, 0.05))
        return [random.random() for _ in range(1536)]

    async def mock_generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
        """Mock batch embedding generation with simulated delay."""
        # Simulate batch processing delay
        await asyncio.sleep(random.uniform(0.1, 0.3))
        return [[random.random() for _ in range(1536)] for _ in texts]

    mock_service.generate_embedding = AsyncMock(side_effect=mock_generate_embedding)
    mock_service.generate_embeddings_batch = AsyncMock(
        side_effect=mock_generate_embeddings_batch
    )

    return mock_service


@pytest.fixture(scope="function")
def mock_encoding_detector():
    """
    Fixture that provides a mock text encoding detector for testing.

    This addresses the skipped tests for text encoding detection architecture.
    """
    from unittest.mock import Mock

    mock_detector = Mock()
    mock_detector.detect_encoding.return_value = {
        "encoding": "utf-8",
        "confidence": 0.99,
        "language": "en",
    }
    mock_detector.is_valid_encoding.return_value = True
    mock_detector.convert_encoding.return_value = "converted text content"

    return mock_detector


@pytest.fixture(scope="function")
def mock_pdf_layout_analyzer():
    """
    Fixture that provides a mock PDF layout preservation analyzer for testing.

    This addresses the skipped tests for PDF layout preservation architecture.
    """
    from unittest.mock import Mock

    mock_analyzer = Mock()
    mock_analyzer.analyze_layout.return_value = {
        "columns": 2,
        "headers": ["Header 1", "Header 2"],
        "tables": [{"rows": 5, "cols": 3, "position": (10, 20, 100, 200)}],
        "images": [{"bbox": (50, 60, 150, 160), "alt_text": "Chart showing data"}],
        "text_blocks": [
            {"text": "Sample text block", "bbox": (0, 0, 200, 50), "font_size": 12}
        ],
    }
    mock_analyzer.preserve_layout.return_value = "Layout-preserved text content"
    mock_analyzer.extract_structured_data.return_value = {
        "paragraphs": ["Paragraph 1", "Paragraph 2"],
        "lists": ["Item 1", "Item 2"],
        "headings": {"h1": ["Main Title"], "h2": ["Subtitle"]},
    }

    return mock_analyzer
