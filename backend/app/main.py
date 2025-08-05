import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import auth, chat, documents
from app.core.error_handler import create_error_response
from app.core.exceptions import RAGPipelineError
from app.core.logging_config import configure_application_logging
from app.core.settings import settings

# Initialize logging configuration
configure_application_logging()

app = FastAPI(
    title="Chat With Docs API",
    description="RAG-powered document Q&A system",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handlers
@app.exception_handler(RAGPipelineError)
async def rag_pipeline_exception_handler(request: Request, exc: RAGPipelineError):
    """Handle all RAG pipeline custom exceptions."""
    request_id = str(uuid.uuid4())

    # Add request context to error details
    if hasattr(exc, "details"):
        exc.details.update(
            {
                "request_url": str(request.url),
                "request_method": request.method,
                "request_id": request_id,
            }
        )

    return create_error_response(exc, request_id)


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other unhandled exceptions."""
    logger = logging.getLogger(__name__)
    request_id = str(uuid.uuid4())

    # Log the unexpected error
    logger.error(
        f"Unhandled exception in request {request_id}: {str(exc)}",
        exc_info=True,
        extra={
            "request_id": request_id,
            "request_url": str(request.url),
            "request_method": request.method,
            "error_type": type(exc).__name__,
        },
    )

    # Create a generic RAG pipeline error for consistent handling
    generic_error = RAGPipelineError(
        "An unexpected error occurred while processing your request.",
        details={
            "request_id": request_id,
            "request_url": str(request.url),
            "request_method": request.method,
            "original_error_type": type(exc).__name__,
        },
        original_error=exc,
    )

    return create_error_response(generic_error, request_id)


# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])


@app.get("/")
async def read_root():
    return {"message": "Chat With Docs API", "version": "1.0.0"}
