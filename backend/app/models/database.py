from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from app.core.settings import settings

# Import pgvector properly for both type checking and runtime
if TYPE_CHECKING:
    from pgvector.sqlalchemy import Vector
else:
    try:
        from pgvector.sqlalchemy import Vector
    except ImportError:
        # Fallback for environments without pgvector
        def Vector(dimensions: int):  # type: ignore
            """Fallback Vector type when pgvector is not available."""
            return Text


Base: Any = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    documents = relationship("Document", back_populates="user")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String, default="processing")
    storage_key = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    content = Column(Text)
    embedding: Any = Column(
        Vector(settings.EMBEDDING_DIMENSION)
    )  # Use pgvector's Vector type
    page = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")
