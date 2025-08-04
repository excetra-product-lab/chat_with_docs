import pytest
from unittest.mock import patch, MagicMock
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.llms import FakeListLLM
from langchain_community.embeddings import FakeEmbeddings

from app.core.langchain_config import LangchainConfig


class TestLangchainConfig:
    """Tests for Task 3.1: Azure OpenAI Client Configuration"""
    
    @patch('app.core.langchain_config.settings')
    def test_llm_azure_configuration(self, mock_settings):
        """Test proper Azure OpenAI LLM configuration"""
        # Mock settings for Azure OpenAI
        mock_settings.AZURE_OPENAI_API_KEY = "test-api-key"
        mock_settings.AZURE_OPENAI_ENDPOINT = "https://test.openai.azure.com/"
        mock_settings.AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o-mini"
        mock_settings.OPENAI_MODEL = "gpt-4o-mini"
        mock_settings.OPENAI_TEMPERATURE = 0.1
        mock_settings.LANGCHAIN_TRACING_V2 = "false"
        mock_settings.LANGCHAIN_API_KEY = None
        mock_settings.LANGCHAIN_PROJECT = None
        mock_settings.LANGCHAIN_VERBOSE = False
        
        config = LangchainConfig()
        llm = config.llm
        
        # Verify LLM is AzureChatOpenAI instance
        assert isinstance(llm, AzureChatOpenAI)
        
        # Note: We can't directly access private attributes in tests,
        # but we can verify the instance was created properly
        assert llm is not None
    
    @patch('app.core.langchain_config.settings')
    def test_llm_fallback_to_fake(self, mock_settings):
        """Test fallback to FakeListLLM when no Azure credentials"""
        # Mock settings without Azure OpenAI credentials
        mock_settings.AZURE_OPENAI_API_KEY = None
        mock_settings.AZURE_OPENAI_ENDPOINT = None
        mock_settings.LANGCHAIN_TRACING_V2 = "false"
        mock_settings.LANGCHAIN_API_KEY = None
        mock_settings.LANGCHAIN_PROJECT = None
        mock_settings.LANGCHAIN_VERBOSE = False
        
        config = LangchainConfig()
        llm = config.llm
        
        # Verify fallback to FakeListLLM
        assert isinstance(llm, FakeListLLM)
    
    @patch('app.core.langchain_config.settings')
    def test_embeddings_azure_configuration(self, mock_settings):
        """Test proper Azure OpenAI embeddings configuration"""
        # Mock settings for Azure OpenAI
        mock_settings.AZURE_OPENAI_API_KEY = "test-api-key"
        mock_settings.AZURE_OPENAI_ENDPOINT = "https://test.openai.azure.com/"
        mock_settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
        mock_settings.EMBEDDING_MODEL = "text-embedding-3-small"
        mock_settings.EMBEDDING_DIMENSION = 1536
        mock_settings.LANGCHAIN_TRACING_V2 = "false"
        mock_settings.LANGCHAIN_API_KEY = None
        mock_settings.LANGCHAIN_PROJECT = None
        mock_settings.LANGCHAIN_VERBOSE = False
        
        config = LangchainConfig()
        embeddings = config.embeddings
        
        # Verify embeddings is AzureOpenAIEmbeddings instance
        assert isinstance(embeddings, AzureOpenAIEmbeddings)
    
    @patch('app.core.langchain_config.settings')
    def test_embeddings_fallback_to_fake(self, mock_settings):
        """Test fallback to FakeEmbeddings when no Azure credentials"""
        # Mock settings without Azure OpenAI credentials
        mock_settings.AZURE_OPENAI_API_KEY = None
        mock_settings.AZURE_OPENAI_ENDPOINT = None
        mock_settings.EMBEDDING_DIMENSION = 1536
        mock_settings.LANGCHAIN_TRACING_V2 = "false"
        mock_settings.LANGCHAIN_API_KEY = None
        mock_settings.LANGCHAIN_PROJECT = None
        mock_settings.LANGCHAIN_VERBOSE = False
        
        config = LangchainConfig()
        embeddings = config.embeddings
        
        # Verify fallback to FakeEmbeddings
        assert isinstance(embeddings, FakeEmbeddings)
    
    @patch('app.core.langchain_config.settings')
    def test_chunk_config(self, mock_settings):
        """Test chunk configuration retrieval"""
        mock_settings.CHUNK_SIZE = 1000
        mock_settings.CHUNK_OVERLAP = 200
        mock_settings.MAX_TOKENS_PER_CHUNK = 8000
        mock_settings.LANGCHAIN_TRACING_V2 = "false"
        mock_settings.LANGCHAIN_API_KEY = None
        mock_settings.LANGCHAIN_PROJECT = None
        mock_settings.LANGCHAIN_VERBOSE = False
        
        config = LangchainConfig()
        chunk_config = config.get_chunk_config()
        
        assert chunk_config["chunk_size"] == 1000
        assert chunk_config["chunk_overlap"] == 200
        assert chunk_config["max_tokens_per_chunk"] == 8000
    
    @patch('app.core.langchain_config.settings')
    def test_tracing_enabled_check(self, mock_settings):
        """Test LangSmith tracing detection"""
        # Test with tracing enabled
        mock_settings.LANGCHAIN_TRACING_V2 = "true"
        mock_settings.LANGCHAIN_API_KEY = "test-api-key"
        mock_settings.LANGCHAIN_PROJECT = None
        mock_settings.LANGCHAIN_VERBOSE = False
        
        config = LangchainConfig()
        assert config.is_tracing_enabled() is True
        
        # Test with tracing disabled
        mock_settings.LANGCHAIN_TRACING_V2 = "false"
        mock_settings.LANGCHAIN_API_KEY = None
        
        config = LangchainConfig()
        assert config.is_tracing_enabled() is False
    
    @patch('app.core.langchain_config.settings')
    @patch('app.core.langchain_config.os.environ')
    def test_environment_setup(self, mock_environ, mock_settings):
        """Test environment variable setup for LangChain"""
        mock_settings.LANGCHAIN_TRACING_V2 = "true"
        mock_settings.LANGCHAIN_API_KEY = "test-langchain-key"
        mock_settings.LANGCHAIN_PROJECT = "test-project"
        mock_settings.LANGCHAIN_VERBOSE = True
        mock_settings.AZURE_OPENAI_API_KEY = None
        mock_settings.AZURE_OPENAI_ENDPOINT = None
        
        # Initialize config (triggers environment setup)
        config = LangchainConfig()
        
        # Verify environment variables were attempted to be set
        # Note: We can't easily test the actual environment variable setting
        # but we can verify the config was initialized without errors
        assert config is not None
    
    @patch('app.core.langchain_config.settings')
    def test_singleton_behavior(self, mock_settings):
        """Test that the same LLM instance is returned on multiple calls"""
        mock_settings.AZURE_OPENAI_API_KEY = "test-api-key"
        mock_settings.AZURE_OPENAI_ENDPOINT = "https://test.openai.azure.com/"
        mock_settings.AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o-mini"
        mock_settings.OPENAI_MODEL = "gpt-4o-mini"
        mock_settings.OPENAI_TEMPERATURE = 0.1
        mock_settings.LANGCHAIN_TRACING_V2 = "false"
        mock_settings.LANGCHAIN_API_KEY = None
        mock_settings.LANGCHAIN_PROJECT = None
        mock_settings.LANGCHAIN_VERBOSE = False
        
        config = LangchainConfig()
        
        # Get LLM multiple times
        llm1 = config.llm
        llm2 = config.llm
        
        # Should be the same instance (singleton pattern)
        assert llm1 is llm2
        
        # Same test for embeddings
        embeddings1 = config.embeddings
        embeddings2 = config.embeddings
        
        assert embeddings1 is embeddings2