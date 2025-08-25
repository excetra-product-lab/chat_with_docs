from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

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

    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(String, default="true")
    created_at = Column(DateTime, default=datetime.utcnow)

    documents = relationship("Document", back_populates="user")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    user_id = Column(String, ForeignKey("users.id"))
    status = Column(String, default="processing")
    storage_key = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("Chunk", back_populates="document")
    user = relationship("User", back_populates="documents")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    content = Column(Text)
    embedding: Any = Column(
        Vector(settings.EMBEDDING_DIMENSION)
    )  # Use pgvector's Vector type
    page = Column(Integer, nullable=True)
    # Enhanced metadata fields
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    chunk_hash = Column(String, nullable=True)
    chunk_type = Column(String, nullable=True)
    hierarchical_level = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    quality_score = Column(Float, nullable=True)
    enhanced_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")


class DocumentHierarchy(Base):
    __tablename__ = "document_hierarchies"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    hierarchy_id = Column(String, unique=True, index=True)
    document_filename = Column(String)
    total_elements = Column(Integer, default=0)
    hierarchy_data = Column(JSON)  # Store the full hierarchy as JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document")
    elements = relationship("DocumentElement", back_populates="hierarchy")


class DocumentElement(Base):
    __tablename__ = "document_elements"

    id = Column(Integer, primary_key=True, index=True)
    hierarchy_id = Column(Integer, ForeignKey("document_hierarchies.id"))
    element_id = Column(String, index=True)
    element_type = Column(String)
    semantic_role = Column(String, nullable=True)
    importance_score = Column(Float, nullable=True)
    level = Column(Integer, nullable=True)
    content = Column(Text, nullable=True)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    page_number = Column(Integer, nullable=True)
    element_metadata = Column(JSON, nullable=True)  # Renamed from 'metadata'
    created_at = Column(DateTime, default=datetime.utcnow)

    hierarchy = relationship("DocumentHierarchy", back_populates="elements")
    chunk_references = relationship("ChunkElementReference", back_populates="element")


class ChunkElementReference(Base):
    __tablename__ = "chunk_element_references"

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(Integer, ForeignKey("chunks.id"))
    element_id = Column(Integer, ForeignKey("document_elements.id"))
    reference_type = Column(
        String, default="contains"
    )  # contains, references, relates_to
    confidence_score = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    chunk = relationship("Chunk")
    element = relationship("DocumentElement", back_populates="chunk_references")


class ElementRelationship(Base):
    __tablename__ = "element_relationships"

    id = Column(Integer, primary_key=True, index=True)
    source_element_id = Column(Integer, ForeignKey("document_elements.id"))
    target_element_id = Column(Integer, ForeignKey("document_elements.id"))
    relationship_type = Column(String)  # parent-child, sibling, reference, etc.
    confidence_score = Column(Float, default=1.0)
    relationship_metadata = Column(JSON, nullable=True)  # Renamed from 'metadata'
    created_at = Column(DateTime, default=datetime.utcnow)

    source_element = relationship("DocumentElement", foreign_keys=[source_element_id])
    target_element = relationship("DocumentElement", foreign_keys=[target_element_id])
