from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://localhost/chatwithdocs"  # Default for testing

    # Azure OpenAI - Required for production use
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_DEPLOYMENT_NAME: Optional[str] = None
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: Optional[str] = None

    # Azure OpenAI Model Configuration
    OPENAI_MODEL: str = "gpt-4o-mini"  # Default deployment name
    OPENAI_TEMPERATURE: float = 0.1

    # Langchain Configuration
    LANGCHAIN_TRACING_V2: str = "false"  # For LangSmith tracing
    LANGCHAIN_API_KEY: Optional[str] = None  # For LangSmith
    LANGCHAIN_PROJECT: Optional[str] = None  # For LangSmith project
    LANGCHAIN_VERBOSE: bool = False  # Enable verbose logging

    # Embedding Configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536

    # Authentication
    SECRET_KEY: str = "dev-secret-key-change-in-production"  # pragma: allowlist secret
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]

    # Vector store & Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_TOKENS_PER_CHUNK: int = 8000

    USE_OPENAI_EMBEDDINGS: bool = False  # Gate expensive network calls in processing

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
