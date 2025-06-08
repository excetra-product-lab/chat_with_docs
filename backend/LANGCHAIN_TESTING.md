# Langchain Testing Utilities Guide

This document explains how to use the Langchain testing utilities that have been integrated into the chat_with_docs project's testing framework.

## Overview

The Langchain testing utilities provide fixtures, helpers, and patterns for testing Langchain components with proper fallbacks to fake implementations when Azure OpenAI credentials are not available. This ensures tests can run in any environment without requiring API access.

## Key Components

### 1. Testing Fixtures (`conftest.py`)

The `conftest.py` file provides several pytest fixtures for consistent Langchain testing:

#### Core Fixtures

- **`fake_llm`** - Provides a `FakeListLLM` with predefined responses
- **`fake_embeddings`** - Provides `FakeEmbeddings` with 1536-dimensional vectors
- **`sample_documents`** - Provides realistic Langchain Document objects
- **`sample_chat_messages`** - Provides alternating Human/AI message sequences
- **`langchain_helper`** - Provides the `LangchainTestHelper` utility class

#### Configuration Fixtures

- **`mock_langchain_config`** - Mocks the global langchain_config to use fake implementations
- **`ensure_fake_config`** - Forces the real config to temporarily use fake implementations
- **`mock_azure_credentials`** - Temporarily sets Azure environment variables for testing

#### Async Fixtures

- **`async_fake_llm`** - Async version of fake_llm for testing async operations
- **`async_fake_embeddings`** - Async version of fake_embeddings

## Usage Examples

### Basic LLM Testing

```python
def test_llm_functionality(fake_llm, langchain_helper):
    """Test basic LLM functionality."""
    # Verify it's using fake implementation
    langchain_helper.assert_is_fake_llm(fake_llm)

    # Test getting a response
    response = fake_llm.invoke("Test prompt")
    assert response in fake_llm.responses

    # Update responses for specific test
    langchain_helper.mock_llm_response(fake_llm, ["Custom response"])
    response = fake_llm.invoke("Another prompt")
    assert response == "Custom response"
```

### Embeddings Testing

```python
def test_embeddings_functionality(fake_embeddings, langchain_helper):
    """Test embeddings functionality."""
    # Verify it's using fake implementation
    langchain_helper.assert_is_fake_embeddings(fake_embeddings)

    # Test single query embedding
    embedding = fake_embeddings.embed_query("Test query")
    assert len(embedding) == 1536

    # Test multiple document embeddings
    embeddings = fake_embeddings.embed_documents(["Doc 1", "Doc 2"])
    assert len(embeddings) == 2
```

### Document Processing Testing

```python
def test_document_processing(sample_documents, langchain_helper):
    """Test document processing workflows."""
    # Validate document structure
    for doc in sample_documents:
        langchain_helper.assert_document_structure(doc)

    # Create custom test documents
    custom_doc = langchain_helper.create_test_document(
        "Custom content",
        source="custom.pdf",
        page=2
    )
    assert custom_doc.metadata["source"] == "custom.pdf"
```

### Chat Message Testing

```python
def test_chat_messages(sample_chat_messages, langchain_helper):
    """Test chat message handling."""
    # Validate message structure
    for message in sample_chat_messages:
        langchain_helper.assert_message_structure(message)

    # Check message types
    assert isinstance(sample_chat_messages[0], HumanMessage)
    assert isinstance(sample_chat_messages[1], AIMessage)
```

### Async Testing

```python
@pytest.mark.asyncio
async def test_async_operations(async_fake_llm, async_fake_embeddings):
    """Test async Langchain operations."""
    # Test async LLM
    response = await async_fake_llm.ainvoke("Async prompt")
    assert response == "Async test response"

    # Test async embeddings
    embedding = await async_fake_embeddings.aembed_query("Async query")
    assert len(embedding) == 1536
```

### Configuration Testing

```python
def test_with_mocked_config(mock_langchain_config):
    """Test using mocked Langchain configuration."""
    # The mock provides controlled fake implementations
    llm = mock_langchain_config.llm
    embeddings = mock_langchain_config.embeddings

    assert isinstance(llm, FakeListLLM)
    assert isinstance(embeddings, FakeEmbeddings)

    # Test configuration methods
    config = mock_langchain_config.get_chunk_config()
    assert config["chunk_size"] == 1000
```

### Integration Testing

```python
def test_with_real_config(ensure_fake_config):
    """Test using the real config but forced to fake implementations."""
    # This uses the actual langchain_config but with fake implementations
    llm = ensure_fake_config.llm
    embeddings = ensure_fake_config.embeddings

    # These will be fake implementations for testing
    assert isinstance(llm, FakeListLLM)
    assert isinstance(embeddings, FakeEmbeddings)
```

## Testing Patterns

### 1. Q&A Chain Testing Pattern

```python
def test_qa_chain_pattern(sample_documents, fake_llm, langchain_helper):
    """Example pattern for testing Q&A chains."""
    # Set up realistic Q&A response
    langchain_helper.mock_llm_response(fake_llm, [
        "Based on the documents, the answer is..."
    ])

    # Simulate Q&A workflow
    question = "What is the main topic?"
    context = "\n".join([doc.page_content for doc in sample_documents])
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

    response = fake_llm.invoke(prompt)
    assert "answer" in response.lower()
```

### 2. Document Processing Pipeline Pattern

```python
def test_document_pipeline(sample_documents, fake_embeddings, langchain_helper):
    """Example pattern for testing document processing pipelines."""
    # Validate input documents
    for doc in sample_documents:
        langchain_helper.assert_document_structure(doc)

    # Process documents through pipeline
    texts = [doc.page_content for doc in sample_documents]
    embeddings = fake_embeddings.embed_documents(texts)

    # Validate output
    assert len(embeddings) == len(sample_documents)
    for embedding in embeddings:
        assert len(embedding) == 1536
```

### 3. Conversational AI Pattern

```python
def test_conversation_pattern(sample_chat_messages, fake_llm, langchain_helper):
    """Example pattern for testing conversational workflows."""
    # Set up conversation responses
    langchain_helper.mock_llm_response(fake_llm, [
        "I understand your question.",
        "Let me provide more details.",
    ])

    # Simulate conversation
    for i, message in enumerate(sample_chat_messages[::2]):  # Human messages
        langchain_helper.assert_message_structure(message)
        response = fake_llm.invoke(message.content)
        assert len(response) > 0
```

## Helper Class Methods

The `LangchainTestHelper` class provides several utility methods:

### Assertion Methods

- `assert_is_fake_llm(llm)` - Assert LLM is fake implementation
- `assert_is_azure_llm(llm)` - Assert LLM is Azure implementation
- `assert_is_fake_embeddings(embeddings)` - Assert embeddings are fake
- `assert_is_azure_embeddings(embeddings)` - Assert embeddings are Azure
- `assert_document_structure(doc)` - Assert Document has required fields
- `assert_message_structure(message)` - Assert message has required fields

### Creation Methods

- `create_test_document(content, source, **metadata)` - Create test Document
- `create_test_embedding_vector(size)` - Create test embedding vector
- `mock_llm_response(llm, responses)` - Update FakeListLLM responses

## Environment Considerations

### Testing Without Azure Credentials

When Azure OpenAI credentials are not available (default state):

```python
def test_default_behavior():
    """Test behavior without Azure credentials."""
    from app.core.langchain_config import langchain_config

    # Should automatically use fake implementations
    llm = langchain_config.llm
    embeddings = langchain_config.embeddings

    # In environments without Azure setup, these will be fake
    assert isinstance(llm, FakeListLLM)
    assert isinstance(embeddings, FakeEmbeddings)
```

### Testing With Mocked Azure Credentials

```python
def test_with_azure_credentials(mock_azure_credentials):
    """Test behavior with mocked Azure credentials."""
    # Environment variables are temporarily set
    import os
    assert os.environ.get('AZURE_OPENAI_API_KEY') == 'test-api-key'

    # Can test configuration logic without real API calls
    from app.core.langchain_config import LangchainConfig
    test_config = LangchainConfig()
    # Configuration object created successfully
```

## Best Practices

1. **Always use fixtures** - Don't create Langchain objects directly in tests
2. **Validate inputs and outputs** - Use helper assertions to verify structure
3. **Test both sync and async** - Use appropriate fixtures for async operations
4. **Mock responses appropriately** - Set realistic responses for your test scenarios
5. **Test configuration paths** - Verify both fake and real configuration scenarios
6. **Use appropriate patterns** - Follow established patterns for common workflows

## Integration with Existing Tests

These utilities integrate seamlessly with the existing test framework:

```python
class TestMyLangchainFeature:
    """Example integration with existing test patterns."""

    def setup_method(self):
        """Setup method compatible with existing patterns."""
        self.test_data = "sample data"

    def test_my_feature(self, fake_llm, sample_documents, langchain_helper):
        """Test that follows existing patterns but uses Langchain utilities."""
        # Existing setup
        assert self.test_data == "sample data"

        # Langchain-specific testing
        langchain_helper.mock_llm_response(fake_llm, ["Test response"])
        result = fake_llm.invoke("Test")
        assert result == "Test response"

        # Validate documents
        for doc in sample_documents:
            langchain_helper.assert_document_structure(doc)
```

## Running the Tests

To run the Langchain testing utilities tests:

```bash
# Run all Langchain utility tests
pytest tests/test_langchain_utilities.py -v

# Run with coverage
pytest tests/test_langchain_utilities.py --cov=app.core.langchain_config

# Run specific test patterns
pytest tests/test_langchain_utilities.py::TestLangchainTestingPatterns -v
```

The utilities are designed to work in any environment and will automatically use fake implementations when Azure OpenAI credentials are not available, ensuring your tests can run consistently across development, CI/CD, and production environments.
