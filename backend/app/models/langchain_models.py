"""
Enhanced Langchain-based data models for documents, chunks, and metadata.

This module provides Langchain-compatible models that extend the existing schemas
with proper BaseModel integration, enhanced validation, and serialization capabilities.
"""

import hashlib
import logging
from datetime import datetime
from typing import Any

from langchain_core.documents import Document as LangchainDocument
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.schemas import DocumentChunk as BaseDocumentChunk
from app.models.schemas import DocumentMetadata as BaseDocumentMetadata

# Citation import removed

logger = logging.getLogger(__name__)


class EnhancedDocumentMetadata(BaseModel):
    """Enhanced document metadata using Langchain's BaseModel with validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    filename: str = Field(..., description="Original filename of the document")
    file_type: str = Field(..., description="Type of the document (pdf, word, text)")
    total_pages: int | None = Field(None, ge=0, description="Total number of pages")
    total_chars: int = Field(..., ge=0, description="Total character count")
    total_tokens: int = Field(0, ge=0, description="Total token count")
    sections: list[str] = Field(
        default_factory=list, description="Document section titles"
    )

    # Enhanced fields for Langchain integration
    document_hash: str | None = Field(
        None, description="Hash of document content for deduplication"
    )
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow)
    langchain_source: bool = Field(
        False, description="Whether processed with Langchain"
    )
    structure_detected: bool = Field(
        False, description="Whether document structure was detected"
    )

    # Additional metadata for enhanced processing
    content_language: str | None = Field(None, description="Detected content language")
    extraction_quality: float | None = Field(
        None, ge=0.0, le=1.0, description="Quality score of extraction"
    )
    additional_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("file_type")
    @classmethod
    def validate_file_type(cls, v):
        """Validate file type is supported."""
        allowed_types = {"pdf", "word", "text", "markdown", "html"}
        if v.lower() not in allowed_types:
            logger.warning(f"Unsupported file type: {v}")
        return v.lower()

    @field_validator("sections")
    @classmethod
    def validate_sections(cls, v):
        """Ensure sections are non-empty strings."""
        return [section.strip() for section in v if section.strip()]

    def generate_content_hash(self, content: str) -> str:
        """Generate a hash of the document content for deduplication."""
        self.document_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return self.document_hash

    def to_base_metadata(self) -> BaseDocumentMetadata:
        """Convert to base DocumentMetadata for backward compatibility."""
        return BaseDocumentMetadata(
            filename=self.filename,
            file_type=self.file_type,
            total_pages=self.total_pages,
            total_chars=self.total_chars,
            total_tokens=self.total_tokens,
            sections=self.sections,
        )

    @classmethod
    def from_base_metadata(
        cls, base_metadata: BaseDocumentMetadata, **kwargs
    ) -> "EnhancedDocumentMetadata":
        """Create enhanced metadata from base metadata."""
        return cls(
            filename=base_metadata.filename,
            file_type=base_metadata.file_type,
            total_pages=base_metadata.total_pages,
            total_chars=base_metadata.total_chars,
            total_tokens=base_metadata.total_tokens,
            sections=base_metadata.sections,
            **kwargs,
        )


class EnhancedDocumentChunk(BaseModel):
    """Enhanced document chunk using Langchain's BaseModel with validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str = Field(..., min_length=1, description="Chunk text content")
    chunk_index: int = Field(..., ge=0, description="Index of chunk in document")
    document_filename: str = Field(..., description="Source document filename")
    page_number: int | None = Field(None, ge=1, description="Page number if applicable")
    section_title: str | None = Field(None, description="Section title if applicable")
    start_char: int = Field(..., ge=0, description="Start character position")
    end_char: int = Field(..., gt=0, description="End character position")
    char_count: int = Field(..., ge=1, description="Character count")
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Enhanced fields for Langchain integration
    chunk_hash: str | None = Field(None, description="Hash of chunk content")
    langchain_source: bool = Field(
        False, description="Whether processed with Langchain"
    )
    embedding_model: str | None = Field(None, description="Model used for embeddings")
    token_count: int = Field(0, ge=0, description="Token count for this chunk")
    quality_score: float | None = Field(
        None, ge=0.0, le=1.0, description="Quality score"
    )

    # Structural information
    hierarchical_level: int = Field(
        0, ge=0, description="Hierarchical level in document"
    )
    parent_section: str | None = Field(None, description="Parent section identifier")
    chunk_type: str = Field(
        "content", description="Type of chunk (content, header, footer, etc.)"
    )
    chunk_references: list[str] = Field(
        default_factory=list, description="References to document elements"
    )

    @field_validator("end_char")
    @classmethod
    def validate_char_positions(cls, v, info):
        """Validate that end_char > start_char."""
        if (
            hasattr(info, "data")
            and "start_char" in info.data
            and v <= info.data["start_char"]
        ):
            raise ValueError("end_char must be greater than start_char")
        return v

    @field_validator("char_count")
    @classmethod
    def validate_char_count_consistency(cls, v, info):
        """Validate char_count matches position difference."""
        if (
            hasattr(info, "data")
            and "start_char" in info.data
            and "end_char" in info.data
        ):
            expected_count = info.data["end_char"] - info.data["start_char"]
            if v != expected_count:
                logger.warning(
                    f"char_count {v} doesn't match positions {expected_count}"
                )
        return v

    @field_validator("chunk_type")
    @classmethod
    def validate_chunk_type(cls, v):
        """Validate chunk type."""
        allowed_types = {
            "content",
            "header",
            "footer",
            "title",
            "metadata",
            "table",
            "list",
            "code",
            # Add chunk types from document splitter
            "basic_split",
            "token_split",
            "paragraph_based",
            "heading_section",
            "heading_section_split",
            "semantic_sentence",
        }
        if v.lower() not in allowed_types:
            logger.warning(f"Unknown chunk type: {v}")
        return v.lower()

    def generate_chunk_hash(self) -> str:
        """Generate a hash of the chunk content."""
        content = f"{self.text}:{self.chunk_index}:{self.document_filename}"
        self.chunk_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return self.chunk_hash

    def to_langchain_document(self) -> LangchainDocument:
        """Convert to Langchain Document format."""
        metadata = {
            "source": self.document_filename,
            "chunk_index": self.chunk_index,
            "page": self.page_number,
            "section": self.section_title,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "char_count": self.char_count,
            "chunk_type": self.chunk_type,
            "hierarchical_level": self.hierarchical_level,
            "langchain_source": self.langchain_source,
            **self.metadata,
        }

        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return LangchainDocument(page_content=self.text, metadata=metadata)

    @classmethod
    def from_langchain_document(
        cls, doc: LangchainDocument, chunk_index: int, document_filename: str
    ) -> "EnhancedDocumentChunk":
        """Create enhanced chunk from Langchain Document."""
        metadata = doc.metadata.copy()

        # Handle empty documents by providing minimum valid content
        content = doc.page_content.strip()
        if not content:
            content = f"[Empty chunk {chunk_index}]"
            logger.warning(f"Empty content in chunk {chunk_index}, using placeholder")

        content_length = len(content)
        start_char = metadata.pop("start_char", 0)
        end_char = metadata.pop("end_char", start_char + content_length)

        # Ensure end_char is greater than start_char
        if end_char <= start_char:
            end_char = start_char + content_length

        return cls(
            text=content,
            chunk_index=chunk_index,
            document_filename=document_filename,
            page_number=metadata.pop("page", None),
            section_title=metadata.pop("section", None),
            start_char=start_char,
            end_char=end_char,
            char_count=content_length,
            chunk_type=metadata.pop("chunk_type", "content"),
            hierarchical_level=metadata.pop("hierarchical_level", 0),
            langchain_source=metadata.pop("langchain_source", True),
            token_count=metadata.pop("token_count", 0),
            metadata=metadata,
        )

    def to_base_chunk(self) -> BaseDocumentChunk:
        """Convert to base DocumentChunk for backward compatibility."""
        return BaseDocumentChunk(
            text=self.text,
            chunk_index=self.chunk_index,
            document_filename=self.document_filename,
            page_number=self.page_number,
            section_title=self.section_title,
            start_char=self.start_char,
            end_char=self.end_char,
            char_count=self.char_count,
            metadata=self.metadata,
        )

    @classmethod
    def from_base_chunk(
        cls, base_chunk: BaseDocumentChunk, **kwargs
    ) -> "EnhancedDocumentChunk":
        """Create enhanced chunk from base chunk."""
        return cls(
            text=base_chunk.text,
            chunk_index=base_chunk.chunk_index,
            document_filename=base_chunk.document_filename,
            page_number=base_chunk.page_number,
            section_title=base_chunk.section_title,
            start_char=base_chunk.start_char,
            end_char=base_chunk.end_char,
            char_count=base_chunk.char_count,
            metadata=base_chunk.metadata,
            **kwargs,
        )


class EnhancedDocument(BaseModel):
    """Enhanced document model using Langchain's BaseModel."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str | None = Field(None, description="Document identifier")
    filename: str = Field(..., description="Document filename")
    user_id: int | None = Field(None, description="User ID who owns the document")
    status: str = Field("processing", description="Processing status")
    storage_key: str | None = Field(None, description="Storage service key")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Enhanced document content and metadata
    content: str | None = Field(None, description="Full document content")
    metadata: EnhancedDocumentMetadata = Field(..., description="Document metadata")
    chunks: list[EnhancedDocumentChunk] = Field(
        default_factory=list, description="Document chunks"
    )

    # Processing information
    processing_stats: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str | None = Field(None, description="Embedding model used")
    vector_store_id: str | None = Field(None, description="Vector store identifier")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        """Validate document status."""
        allowed_statuses = {"processing", "completed", "failed", "pending", "archived"}
        if v.lower() not in allowed_statuses:
            logger.warning(f"Unknown status: {v}")
        return v.lower()

    @field_validator("chunks")
    @classmethod
    def validate_chunks_consistency(cls, v, info):
        """Validate chunks are consistent with document."""
        if hasattr(info, "data") and "filename" in info.data:
            for chunk in v:
                if chunk.document_filename != info.data["filename"]:
                    raise ValueError(
                        f"Chunk filename mismatch: {chunk.document_filename} != {info.data['filename']}"
                    )
        return v

    def add_chunk(self, chunk: EnhancedDocumentChunk) -> None:
        """Add a chunk to the document with validation."""
        if chunk.document_filename != self.filename:
            chunk.document_filename = self.filename

        # Set chunk index if not already set
        if chunk.chunk_index == 0 and self.chunks:
            chunk.chunk_index = max(c.chunk_index for c in self.chunks) + 1

        self.chunks.append(chunk)
        self.updated_at = datetime.utcnow()

    def get_chunk_by_index(self, index: int) -> EnhancedDocumentChunk | None:
        """Get chunk by index."""
        for chunk in self.chunks:
            if chunk.chunk_index == index:
                return chunk
        return None

    def get_total_tokens(self) -> int:
        """Calculate total tokens across all chunks."""
        return sum(chunk.token_count for chunk in self.chunks)

    def get_total_chars(self) -> int:
        """Calculate total characters across all chunks."""
        return sum(chunk.char_count for chunk in self.chunks)

    def to_langchain_documents(self) -> list[LangchainDocument]:
        """Convert all chunks to Langchain Documents."""
        return [chunk.to_langchain_document() for chunk in self.chunks]

    @classmethod
    def from_langchain_documents(
        cls, documents: list[LangchainDocument], filename: str, **kwargs
    ) -> "EnhancedDocument":
        """Create enhanced document from Langchain Documents."""
        chunks = [
            EnhancedDocumentChunk.from_langchain_document(doc, i, filename)
            for i, doc in enumerate(documents)
        ]

        # Combine all document content for the main content field
        combined_content = "\n\n".join(
            doc.page_content for doc in documents if doc.page_content.strip()
        )

        # Create metadata from documents
        total_chars = sum(len(doc.page_content) for doc in documents)
        metadata = EnhancedDocumentMetadata(
            filename=filename,
            file_type=cls._detect_file_type(filename),
            total_chars=total_chars,
            langchain_source=True,
            structure_detected=any("section" in doc.metadata for doc in documents),
        )

        return cls(
            filename=filename,
            content=combined_content,  # Add the combined content here
            metadata=metadata,
            chunks=chunks,
            **kwargs,
        )

    @staticmethod
    def _detect_file_type(filename: str) -> str:
        """Detect file type from filename."""
        extension = filename.lower().split(".")[-1] if "." in filename else ""
        type_mapping = {
            "pdf": "pdf",
            "doc": "word",
            "docx": "word",
            "txt": "text",
            "md": "markdown",
            "html": "html",
            "htm": "html",
        }
        return type_mapping.get(extension, "unknown")

    def calculate_processing_stats(self) -> dict[str, Any]:
        """Calculate comprehensive processing statistics."""
        stats = {
            "document": {
                "filename": self.filename,
                "total_chunks": len(self.chunks),
                "total_characters": self.get_total_chars(),
                "total_tokens": self.get_total_tokens(),
                "status": self.status,
            },
            "chunks": {
                "avg_chunk_size": self.get_total_chars() / len(self.chunks)
                if self.chunks
                else 0,
                "min_chunk_size": min(c.char_count for c in self.chunks)
                if self.chunks
                else 0,
                "max_chunk_size": max(c.char_count for c in self.chunks)
                if self.chunks
                else 0,
                "chunks_with_sections": sum(1 for c in self.chunks if c.section_title),
                "chunks_by_type": {},
            },
            "processing": {
                "langchain_used": any(c.langchain_source for c in self.chunks),
                "embedding_model": self.embedding_model,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
            },
        }

        # Calculate chunk type distribution
        chunk_types = {}
        for chunk in self.chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
        stats["chunks"]["chunks_by_type"] = chunk_types

        self.processing_stats = stats
        return stats


# EnhancedCitation class removed


# Utility functions for model conversion and integration


def convert_to_enhanced_document(
    base_metadata: BaseDocumentMetadata, base_chunks: list[BaseDocumentChunk], **kwargs
) -> EnhancedDocument:
    """Convert base models to enhanced document."""
    enhanced_metadata = EnhancedDocumentMetadata.from_base_metadata(base_metadata)
    enhanced_chunks = [
        EnhancedDocumentChunk.from_base_chunk(chunk) for chunk in base_chunks
    ]

    return EnhancedDocument(
        filename=base_metadata.filename,
        metadata=enhanced_metadata,
        chunks=enhanced_chunks,
        **kwargs,
    )


def convert_from_enhanced_document(
    enhanced_doc: EnhancedDocument,
) -> tuple[BaseDocumentMetadata, list[BaseDocumentChunk]]:
    """Convert enhanced document back to base models."""
    base_metadata = enhanced_doc.metadata.to_base_metadata()
    base_chunks = [chunk.to_base_chunk() for chunk in enhanced_doc.chunks]

    return base_metadata, base_chunks


def integrate_with_langchain_pipeline(
    langchain_docs: list[LangchainDocument], filename: str, **kwargs
) -> EnhancedDocument:
    """Create enhanced document from Langchain pipeline output."""
    return EnhancedDocument.from_langchain_documents(langchain_docs, filename, **kwargs)
