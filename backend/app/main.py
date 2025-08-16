from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import auth, chat, documents

# Initialize global logging configuration before anything else
from app.core import logger_config  # noqa: F401  # pylint: disable=unused-import
from app.core.settings import settings

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

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])


@app.get("/")
async def read_root():
    return {"message": "Chat With Docs API", "version": "1.0.0"}
