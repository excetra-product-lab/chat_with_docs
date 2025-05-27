from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

if TYPE_CHECKING:
    from sqlalchemy.dialects.postgresql import VECTOR  # type: ignore
else:
    # For runtime, use a placeholder that won't cause import errors
    try:
        from sqlalchemy.dialects.postgresql import VECTOR  # type: ignore
    except ImportError:
        # Fallback for environments without postgresql dialect
        def VECTOR(x):
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
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    content = Column(Text)
    embedding: Any = Column(VECTOR(1536))
    page = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")
