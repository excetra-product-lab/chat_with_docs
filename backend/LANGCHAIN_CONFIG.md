# Langchain Configuration

This document outlines the Langchain-specific configuration options added to the chat_with_docs backend.

## Environment Variables

Add these variables to your `.env` file to configure Langchain integration:

### Required for Full Functionality

```bash
# Azure OpenAI API Key - Required for production use
AZURE_OPENAI_API_KEY=your-azure-openai-api-key-here

# Azure OpenAI Endpoint - Required for production use
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Azure OpenAI Deployment Names
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o  # Your chat model deployment
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002  # Your embedding deployment

# Optional: Default deployment name fallback (default: gpt-4o-mini)
OPENAI_MODEL=gpt-4o-mini

# Optional: Temperature for LLM responses (default: 0.1)
OPENAI_TEMPERATURE=0.1

# Optional: Embedding model fallback (default: text-embedding-3-small)
EMBEDDING_MODEL=text-embedding-3-small

# Optional: Embedding dimensions (default: 1536)
EMBEDDING_DIMENSION=1536
```

### Optional - LangSmith Tracing

LangSmith provides observability and debugging for Langchain applications:

```bash
# Enable LangSmith tracing
LANGCHAIN_TRACING_V2=true

# LangSmith API key (get from https://smith.langchain.com/)
LANGCHAIN_API_KEY=your-langsmith-api-key

# LangSmith project name
LANGCHAIN_PROJECT=chat-with-docs

# Enable verbose Langchain logging
LANGCHAIN_VERBOSE=true
```

### Document Processing Configuration

```bash
# Document chunking settings (already in settings.py)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_TOKENS_PER_CHUNK=8000
```

## Testing without API Keys

The system includes fallback configurations for testing:

- **FakeListLLM**: Used when no Azure OpenAI API key/endpoint is provided
- **FakeEmbeddings**: Used for embedding operations in test environments

## Usage

The configuration is automatically loaded through the `LangchainConfig` class:

```python
from app.core.langchain_config import langchain_config

# Get configured LLM
llm = langchain_config.llm

# Get configured embeddings
embeddings = langchain_config.embeddings

# Get chunking configuration
chunk_config = langchain_config.get_chunk_config()

# Check if tracing is enabled
if langchain_config.is_tracing_enabled():
    print("LangSmith tracing is active")
```

## Integration Points

The Langchain configuration integrates with:

1. **Document Processing**: Text splitting and chunking
2. **Q&A Chains**: Language model for question answering
3. **Embeddings**: Vector representations for document search
4. **Logging**: Comprehensive logging for debugging
5. **Tracing**: Optional LangSmith integration for observability
