"""
Tests for Langchain testing utilities and configuration.

This module validates that the Langchain testing utilities work correctly
and demonstrates proper usage patterns for testing Langchain components.
"""

from unittest.mock import patch

import pytest
from langchain.schema import AIMessage, Document, HumanMessage
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.llms import FakeListLLM

from app.core.langchain_config import langchain_config


class TestLangchainTestingUtilities:
    """Test cases for Langchain testing utilities and fixtures."""

    def test_fake_llm_fixture(self, fake_llm):
        """Test that the fake_llm fixture provides a working FakeListLLM."""
        assert isinstance(fake_llm, FakeListLLM)
        assert len(fake_llm.responses) == 3

        # Test that we can get responses
        response = fake_llm.invoke("Test prompt")
        assert response in fake_llm.responses

    def test_fake_embeddings_fixture(self, fake_embeddings):
        """Test that the fake_embeddings fixture provides working FakeEmbeddings."""
        assert isinstance(fake_embeddings, FakeEmbeddings)

        # Test embedding a single document
        embedding = fake_embeddings.embed_query("Test query")
        assert isinstance(embedding, list)
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)

        # Test embedding multiple documents
        embeddings = fake_embeddings.embed_documents(["Doc 1", "Doc 2"])
        assert len(embeddings) == 2
        assert all(len(emb) == 1536 for emb in embeddings)

    def test_sample_documents_fixture(self, sample_documents):
        """Test that the sample_documents fixture provides valid Document objects."""
        assert isinstance(sample_documents, list)
        assert len(sample_documents) == 3

        for doc in sample_documents:
            assert isinstance(doc, Document)
            assert isinstance(doc.page_content, str)
            assert isinstance(doc.metadata, dict)
            assert "source" in doc.metadata
            assert "page" in doc.metadata
            assert "chunk_id" in doc.metadata

    def test_sample_chat_messages_fixture(self, sample_chat_messages):
        """Test that the sample_chat_messages fixture provides valid message objects."""
        assert isinstance(sample_chat_messages, list)
        assert len(sample_chat_messages) == 4

        # Check alternating pattern of human/AI messages
        assert isinstance(sample_chat_messages[0], HumanMessage)
        assert isinstance(sample_chat_messages[1], AIMessage)
        assert isinstance(sample_chat_messages[2], HumanMessage)
        assert isinstance(sample_chat_messages[3], AIMessage)

        for message in sample_chat_messages:
            assert hasattr(message, "content")
            assert isinstance(message.content, str)
            assert len(message.content) > 0

    def test_langchain_helper_assertions(
        self, langchain_helper, fake_llm, fake_embeddings
    ):
        """Test the assertion methods in LangchainTestHelper."""
        # Test LLM assertions
        langchain_helper.assert_is_fake_llm(fake_llm)

        with pytest.raises(AssertionError):
            langchain_helper.assert_is_azure_llm(fake_llm)

        # Test embeddings assertions
        langchain_helper.assert_is_fake_embeddings(fake_embeddings)

        with pytest.raises(AssertionError):
            langchain_helper.assert_is_azure_embeddings(fake_embeddings)

    def test_langchain_helper_document_assertions(
        self, langchain_helper, sample_documents
    ):
        """Test the document assertion methods in LangchainTestHelper."""
        for doc in sample_documents:
            langchain_helper.assert_document_structure(doc)

        # Test with invalid document structure
        invalid_doc = type("MockDoc", (), {})()
        with pytest.raises(AssertionError):
            langchain_helper.assert_document_structure(invalid_doc)

    def test_langchain_helper_message_assertions(
        self, langchain_helper, sample_chat_messages
    ):
        """Test the message assertion methods in LangchainTestHelper."""
        for message in sample_chat_messages:
            langchain_helper.assert_message_structure(message)

        # Test with invalid message structure
        invalid_message = type("MockMessage", (), {})()
        with pytest.raises(AssertionError):
            langchain_helper.assert_message_structure(invalid_message)

    def test_langchain_helper_create_test_document(self, langchain_helper):
        """Test the create_test_document method in LangchainTestHelper."""
        content = "This is test content for a document."
        doc = langchain_helper.create_test_document(content)

        assert isinstance(doc, Document)
        assert doc.page_content == content
        assert doc.metadata["source"] == "test.txt"
        assert doc.metadata["page"] == 1
        assert doc.metadata["chunk_id"] == 0

        # Test with custom metadata
        custom_doc = langchain_helper.create_test_document(
            content, source="custom.pdf", page=5, custom_field="test_value"
        )
        assert custom_doc.metadata["source"] == "custom.pdf"
        assert custom_doc.metadata["page"] == 5
        assert custom_doc.metadata["custom_field"] == "test_value"

    def test_langchain_helper_create_test_embedding_vector(self, langchain_helper):
        """Test the create_test_embedding_vector method in LangchainTestHelper."""
        # Test default size
        vector = langchain_helper.create_test_embedding_vector()
        assert isinstance(vector, list)
        assert len(vector) == 1536
        assert all(x == 0.1 for x in vector)

        # Test custom size
        custom_vector = langchain_helper.create_test_embedding_vector(size=100)
        assert len(custom_vector) == 100
        assert all(x == 0.1 for x in custom_vector)

    def test_langchain_helper_mock_llm_response(self, langchain_helper, fake_llm):
        """Test the mock_llm_response method in LangchainTestHelper."""
        new_responses = ["New response 1", "New response 2"]
        langchain_helper.mock_llm_response(fake_llm, new_responses)

        assert fake_llm.responses == new_responses
        assert fake_llm.i == 0

        # Test that we get the new responses
        response1 = fake_llm.invoke("Test")
        response2 = fake_llm.invoke("Test")
        assert response1 == "New response 1"
        assert response2 == "New response 2"

    def test_mock_langchain_config_fixture(self, mock_langchain_config):
        """Test that the mock_langchain_config fixture works correctly."""
        # The mock should provide fake implementations
        llm = mock_langchain_config.llm
        embeddings = mock_langchain_config.embeddings

        assert isinstance(llm, FakeListLLM)
        assert isinstance(embeddings, FakeEmbeddings)

        # Test configuration methods
        chunk_config = mock_langchain_config.get_chunk_config()
        assert chunk_config["chunk_size"] == 1000
        assert chunk_config["chunk_overlap"] == 200
        assert chunk_config["max_tokens_per_chunk"] == 1000

        assert mock_langchain_config.is_tracing_enabled() is False

    def test_ensure_fake_config_fixture(self, ensure_fake_config):
        """Test that the ensure_fake_config fixture forces fake implementations."""
        # The global config should now use fake implementations
        llm = ensure_fake_config.llm
        embeddings = ensure_fake_config.embeddings

        assert isinstance(llm, FakeListLLM)
        assert isinstance(embeddings, FakeEmbeddings)

        # Test that the configuration methods still work
        chunk_config = ensure_fake_config.get_chunk_config()
        assert isinstance(chunk_config, dict)
        assert "chunk_size" in chunk_config

    @pytest.mark.asyncio
    async def test_async_fake_llm_fixture(self, async_fake_llm):
        """Test that the async_fake_llm fixture works for async operations."""
        assert isinstance(async_fake_llm, FakeListLLM)

        # Test async invocation
        response = await async_fake_llm.ainvoke("Async test prompt")
        assert response == "Async test response"

    @pytest.mark.asyncio
    async def test_async_fake_embeddings_fixture(self, async_fake_embeddings):
        """Test that the async_fake_embeddings fixture works for async operations."""
        assert isinstance(async_fake_embeddings, FakeEmbeddings)

        # Test async embedding
        embedding = await async_fake_embeddings.aembed_query("Async test query")
        assert isinstance(embedding, list)
        assert len(embedding) == 1536

    def test_mock_azure_credentials_fixture(self, mock_azure_credentials):
        """Test that the mock_azure_credentials fixture sets environment variables."""
        import os

        assert os.environ.get("AZURE_OPENAI_API_KEY") == "test-api-key"
        assert (
            os.environ.get("AZURE_OPENAI_ENDPOINT") == "https://test.openai.azure.com/"
        )
        assert os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME") == "test-deployment"
        assert (
            os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            == "test-embedding-deployment"
        )


class TestLangchainConfigIntegration:
    """Test cases for integration with the actual LangchainConfig."""

    def test_config_methods_work_regardless_of_implementation(self):
        """Test that configuration methods work with both real and fake implementations."""
        # These methods should work regardless of whether we're using real or fake implementations
        chunk_config = langchain_config.get_chunk_config()
        assert isinstance(chunk_config, dict)
        assert "chunk_size" in chunk_config
        assert "chunk_overlap" in chunk_config
        assert "max_tokens_per_chunk" in chunk_config

        # Tracing status should be a boolean
        tracing_enabled = langchain_config.is_tracing_enabled()
        assert isinstance(tracing_enabled, bool)

    def test_config_with_mocked_azure_credentials(self, mock_azure_credentials):
        """Test LangchainConfig behavior when Azure credentials are available."""
        # Create a new config instance to pick up the mocked environment variables
        from app.core.langchain_config import LangchainConfig

        with (
            patch("app.core.settings.settings.AZURE_OPENAI_API_KEY", "test-api-key"),
            patch(
                "app.core.settings.settings.AZURE_OPENAI_ENDPOINT",
                "https://test.openai.azure.com/",
            ),
        ):
            test_config = LangchainConfig()

            # With mocked credentials, the config should try to create Azure instances
            # But since we're not making real API calls, we can just verify the types
            # This test verifies the configuration logic works correctly
            assert test_config is not None
            assert hasattr(test_config, "llm")
            assert hasattr(test_config, "embeddings")


class TestLangchainTestingPatterns:
    """Examples of common testing patterns with Langchain components."""

    def test_document_processing_pattern(
        self, sample_documents, fake_embeddings, langchain_helper
    ):
        """Example test pattern for document processing workflows."""
        # Verify documents have expected structure
        for doc in sample_documents:
            langchain_helper.assert_document_structure(doc)

        # Test embedding the documents
        texts = [doc.page_content for doc in sample_documents]
        embeddings = fake_embeddings.embed_documents(texts)

        assert len(embeddings) == len(sample_documents)
        for embedding in embeddings:
            assert len(embedding) == 1536

    def test_chat_conversation_pattern(
        self, sample_chat_messages, fake_llm, langchain_helper
    ):
        """Example test pattern for chat/conversation workflows."""
        # Verify messages have expected structure
        for message in sample_chat_messages:
            langchain_helper.assert_message_structure(message)

        # Set up custom responses for the conversation
        langchain_helper.mock_llm_response(
            fake_llm,
            [
                "I understand your question about AI.",
                "Machine learning involves training algorithms on data.",
            ],
        )

        # Simulate conversation
        response1 = fake_llm.invoke("What is AI?")
        response2 = fake_llm.invoke("How does ML work?")

        assert "understand" in response1
        assert "training" in response2

    def test_qa_chain_pattern(self, sample_documents, fake_llm, langchain_helper):
        """Example test pattern for Q&A chain workflows."""
        # Set up a realistic Q&A response
        langchain_helper.mock_llm_response(
            fake_llm,
            [
                (
                    "Based on the documents, artificial intelligence is a field that focuses on "
                    "creating intelligent systems."
                )
            ],
        )

        # Simulate a Q&A interaction
        question = "What is artificial intelligence?"
        context = "\n".join([doc.page_content for doc in sample_documents])

        # This simulates what a real Q&A chain would do
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        response = fake_llm.invoke(prompt)

        assert "artificial intelligence" in response.lower()
        assert "field" in response.lower()
