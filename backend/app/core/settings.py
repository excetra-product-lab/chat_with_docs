from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://localhost/chatwithdocs"  # Default for testing

    # Azure OpenAI - Required for production use
    AZURE_OPENAI_API_KEY: str | None = None
    AZURE_OPENAI_ENDPOINT: str | None = None
    AZURE_OPENAI_DEPLOYMENT_NAME: str | None = None
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str | None = None
    AZURE_OPENAI_API_VERSION: str | None = None

    # Azure OpenAI Model Configuration
    OPENAI_MODEL: str = "gpt-4o-mini"  # Default deployment name
    OPENAI_TEMPERATURE: float = 0.1

    # Langchain Configuration
    LANGCHAIN_TRACING_V2: str = "false"  # For LangSmith tracing
    LANGCHAIN_API_KEY: str | None = None  # For LangSmith
    LANGCHAIN_PROJECT: str | None = None  # For LangSmith project
    LANGCHAIN_VERBOSE: bool = False  # Enable verbose logging

    # Embedding Configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536

    # Authentication
    SECRET_KEY: str = "dev-secret-key-change-in-production"  # pragma: allowlist secret
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # CORS
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000"]

    # Vector store & Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_TOKENS_PER_CHUNK: int = 8000

    USE_OPENAI_EMBEDDINGS: bool = False  # Gate expensive network calls in processing

    # File upload settings
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_FILE_EXTENSIONS: list[str] = [".pdf", ".docx", ".txt", ".md"]

    # Supabase settings
    SUPABASE_URL: str | None = None
    SUPABASE_KEY: str | None = None
    SUPABASE_BUCKET_NAME: str = "documents"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
