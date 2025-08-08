"""
Tests for enhanced Langchain-based data models.

This module tests the enhanced models for proper validation, serialization,
and integration with the existing document processing pipeline.
"""

from datetime import datetime

import pytest
from langchain_core.documents import Document as LangchainDocument

from app.models.langchain_models import (
    EnhancedDocument,
    EnhancedDocumentChunk,
    EnhancedDocumentMetadata,
    convert_from_enhanced_document,
    # EnhancedCitation removed,
    convert_to_enhanced_document,
    integrate_with_langchain_pipeline,
)
from app.models.schemas import DocumentChunk, DocumentMetadata

# Citation import removed


class TestEnhancedDocumentMetadata:
    """Test cases for EnhancedDocumentMetadata model."""

    def test_create_enhanced_metadata(self):
        """Test creating enhanced metadata with validation."""
        metadata = EnhancedDocumentMetadata(
            filename="test.pdf",
            file_type="pdf",
            total_chars=1000,
            total_tokens=200,
            sections=["Introduction", "Conclusion"],
            langchain_source=True,
            structure_detected=True,
        )

        assert metadata.filename == "test.pdf"
        assert metadata.file_type == "pdf"
        assert metadata.total_chars == 1000
        assert metadata.total_tokens == 200
        assert metadata.sections == ["Introduction", "Conclusion"]
        assert metadata.langchain_source is True
        assert metadata.structure_detected is True
        assert isinstance(metadata.processing_timestamp, datetime)

    def test_file_type_validation(self):
        """Test file type validation and normalization."""
        metadata = EnhancedDocumentMetadata(
            filename="test.PDF", file_type="PDF", total_chars=1000
        )
        assert metadata.file_type == "pdf"

    def test_sections_validation(self):
        """Test sections are cleaned and validated."""
        metadata = EnhancedDocumentMetadata(
            filename="test.pdf",
            file_type="pdf",
            total_chars=1000,
            sections=["  Section 1  ", "", "Section 2", "   "],
        )
        assert metadata.sections == ["Section 1", "Section 2"]

    def test_generate_content_hash(self):
        """Test content hash generation."""
        metadata = EnhancedDocumentMetadata(
            filename="test.pdf", file_type="pdf", total_chars=1000
        )

        content = "Sample document content"
        hash_value = metadata.generate_content_hash(content)

        assert metadata.document_hash == hash_value
        assert len(hash_value) == 64  # SHA256 hex length

    def test_conversion_to_base_metadata(self):
        """Test conversion to base DocumentMetadata."""
        enhanced = EnhancedDocumentMetadata(
            filename="test.pdf",
            file_type="pdf",
            total_pages=10,
            total_chars=1000,
            total_tokens=200,
            sections=["Section 1", "Section 2"],
        )

        base = enhanced.to_base_metadata()

        assert isinstance(base, DocumentMetadata)
        assert base.filename == enhanced.filename
        assert base.file_type == enhanced.file_type
        assert base.total_pages == enhanced.total_pages
        assert base.total_chars == enhanced.total_chars
        assert base.total_tokens == enhanced.total_tokens
        assert base.sections == enhanced.sections

    def test_conversion_from_base_metadata(self):
        """Test creation from base DocumentMetadata."""
        base = DocumentMetadata(
            filename="test.pdf",
            file_type="pdf",
            total_pages=10,
            total_chars=1000,
            total_tokens=200,
            sections=["Section 1"],
        )

        enhanced = EnhancedDocumentMetadata.from_base_metadata(
            base, langchain_source=True, structure_detected=True
        )

        assert enhanced.filename == base.filename
        assert enhanced.file_type == base.file_type
        assert enhanced.total_pages == base.total_pages
        assert enhanced.total_chars == base.total_chars
        assert enhanced.total_tokens == base.total_tokens
        assert enhanced.sections == base.sections
        assert enhanced.langchain_source is True
        assert enhanced.structure_detected is True


class TestEnhancedDocumentChunk:
    """Test cases for EnhancedDocumentChunk model."""

    def test_create_enhanced_chunk(self):
        """Test creating enhanced chunk with validation."""
        chunk = EnhancedDocumentChunk(
            text="Sample chunk content",
            chunk_index=0,
            document_filename="test.pdf",
            page_number=1,
            section_title="Introduction",
            start_char=0,
            end_char=20,
            char_count=20,
            chunk_type="content",
            langchain_source=True,
        )

        assert chunk.text == "Sample chunk content"
        assert chunk.chunk_index == 0
        assert chunk.document_filename == "test.pdf"
        assert chunk.page_number == 1
        assert chunk.section_title == "Introduction"
        assert chunk.start_char == 0
        assert chunk.end_char == 20
        assert chunk.char_count == 20
        assert chunk.chunk_type == "content"
        assert chunk.langchain_source is True

    def test_char_position_validation(self):
        """Test character position validation."""
        with pytest.raises(
            ValueError, match="end_char must be greater than start_char"
        ):
            EnhancedDocumentChunk(
                text="Sample text",
                chunk_index=0,
                document_filename="test.pdf",
                start_char=10,
                end_char=5,  # Invalid: end < start
                char_count=11,
            )

    def test_chunk_type_validation(self):
        """Test chunk type validation and normalization."""
        chunk = EnhancedDocumentChunk(
            text="Sample text",
            chunk_index=0,
            document_filename="test.pdf",
            start_char=0,
            end_char=11,
            char_count=11,
            chunk_type="CONTENT",
        )
        assert chunk.chunk_type == "content"

    def test_generate_chunk_hash(self):
        """Test chunk hash generation."""
        chunk = EnhancedDocumentChunk(
            text="Sample chunk content",
            chunk_index=0,
            document_filename="test.pdf",
            start_char=0,
            end_char=20,
            char_count=20,
        )

        hash_value = chunk.generate_chunk_hash()

        assert chunk.chunk_hash == hash_value
        assert len(hash_value) == 64  # SHA256 hex length

    def test_to_langchain_document(self):
        """Test conversion to Langchain Document."""
        chunk = EnhancedDocumentChunk(
            text="Sample chunk content",
            chunk_index=0,
            document_filename="test.pdf",
            page_number=1,
            section_title="Introduction",
            start_char=0,
            end_char=20,
            char_count=20,
            chunk_type="content",
            hierarchical_level=1,
        )

        langchain_doc = chunk.to_langchain_document()

        assert isinstance(langchain_doc, LangchainDocument)
        assert langchain_doc.page_content == "Sample chunk content"
        assert langchain_doc.metadata["source"] == "test.pdf"
        assert langchain_doc.metadata["chunk_index"] == 0
        assert langchain_doc.metadata["page"] == 1
        assert langchain_doc.metadata["section"] == "Introduction"
        assert langchain_doc.metadata["chunk_type"] == "content"
        assert langchain_doc.metadata["hierarchical_level"] == 1

    def test_from_langchain_document(self):
        """Test creation from Langchain Document."""
        langchain_doc = LangchainDocument(
            page_content="Sample content",
            metadata={
                "page": 1,
                "section": "Introduction",
                "start_char": 0,
                "end_char": 14,
                "chunk_type": "content",
                "hierarchical_level": 1,
                "langchain_source": True,
            },
        )

        chunk = EnhancedDocumentChunk.from_langchain_document(
            langchain_doc, chunk_index=0, document_filename="test.pdf"
        )

        assert chunk.text == "Sample content"
        assert chunk.chunk_index == 0
        assert chunk.document_filename == "test.pdf"
        assert chunk.page_number == 1
        assert chunk.section_title == "Introduction"
        assert chunk.start_char == 0
        assert chunk.end_char == 14
        assert chunk.chunk_type == "content"
        assert chunk.hierarchical_level == 1
        assert chunk.langchain_source is True

    def test_conversion_to_base_chunk(self):
        """Test conversion to base DocumentChunk."""
        enhanced = EnhancedDocumentChunk(
            text="Sample content",
            chunk_index=0,
            document_filename="test.pdf",
            page_number=1,
            section_title="Introduction",
            start_char=0,
            end_char=14,
            char_count=14,
            metadata={"extra": "data"},
        )

        base = enhanced.to_base_chunk()

        assert isinstance(base, DocumentChunk)
        assert base.text == enhanced.text
        assert base.chunk_index == enhanced.chunk_index
        assert base.document_filename == enhanced.document_filename
        assert base.page_number == enhanced.page_number
        assert base.section_title == enhanced.section_title
        assert base.start_char == enhanced.start_char
        assert base.end_char == enhanced.end_char
        assert base.char_count == enhanced.char_count
        assert base.metadata == enhanced.metadata

    def test_conversion_from_base_chunk(self):
        """Test creation from base DocumentChunk."""
        base = DocumentChunk(
            text="Sample content",
            chunk_index=0,
            document_filename="test.pdf",
            page_number=1,
            section_title="Introduction",
            start_char=0,
            end_char=14,
            char_count=14,
            metadata={"extra": "data"},
        )

        enhanced = EnhancedDocumentChunk.from_base_chunk(
            base, langchain_source=True, chunk_type="content"
        )

        assert enhanced.text == base.text
        assert enhanced.chunk_index == base.chunk_index
        assert enhanced.document_filename == base.document_filename
        assert enhanced.page_number == base.page_number
        assert enhanced.section_title == base.section_title
        assert enhanced.start_char == base.start_char
        assert enhanced.end_char == base.end_char
        assert enhanced.char_count == base.char_count
        assert enhanced.metadata == base.metadata
        assert enhanced.langchain_source is True
        assert enhanced.chunk_type == "content"


class TestEnhancedDocument:
    """Test cases for EnhancedDocument model."""

    def test_create_enhanced_document(self):
        """Test creating enhanced document with validation."""
        metadata = EnhancedDocumentMetadata(
            filename="test.pdf", file_type="pdf", total_chars=1000
        )

        chunks = [
            EnhancedDocumentChunk(
                text="First chunk",
                chunk_index=0,
                document_filename="test.pdf",
                start_char=0,
                end_char=11,
                char_count=11,
            ),
            EnhancedDocumentChunk(
                text="Second chunk",
                chunk_index=1,
                document_filename="test.pdf",
                start_char=12,
                end_char=24,
                char_count=12,
            ),
        ]

        document = EnhancedDocument(
            filename="test.pdf", metadata=metadata, chunks=chunks, status="completed"
        )

        assert document.filename == "test.pdf"
        assert document.status == "completed"
        assert len(document.chunks) == 2
        assert isinstance(document.created_at, datetime)

    def test_chunks_consistency_validation(self):
        """Test chunk consistency validation."""
        metadata = EnhancedDocumentMetadata(
            filename="test.pdf", file_type="pdf", total_chars=1000
        )

        chunks = [
            EnhancedDocumentChunk(
                text="Content",
                chunk_index=0,
                document_filename="wrong.pdf",  # Wrong filename
                start_char=0,
                end_char=7,
                char_count=7,
            )
        ]

        with pytest.raises(ValueError, match="Chunk filename mismatch"):
            EnhancedDocument(filename="test.pdf", metadata=metadata, chunks=chunks)

    def test_add_chunk(self):
        """Test adding chunks to document."""
        metadata = EnhancedDocumentMetadata(
            filename="test.pdf", file_type="pdf", total_chars=1000
        )

        document = EnhancedDocument(filename="test.pdf", metadata=metadata)

        chunk = EnhancedDocumentChunk(
            text="New chunk",
            chunk_index=0,
            document_filename="different.pdf",  # Will be corrected
            start_char=0,
            end_char=9,
            char_count=9,
        )

        document.add_chunk(chunk)

        assert len(document.chunks) == 1
        assert document.chunks[0].document_filename == "test.pdf"

    def test_get_chunk_by_index(self):
        """Test retrieving chunk by index."""
        metadata = EnhancedDocumentMetadata(
            filename="test.pdf", file_type="pdf", total_chars=1000
        )

        chunks = [
            EnhancedDocumentChunk(
                text="First chunk",
                chunk_index=0,
                document_filename="test.pdf",
                start_char=0,
                end_char=11,
                char_count=11,
            ),
            EnhancedDocumentChunk(
                text="Second chunk",
                chunk_index=1,
                document_filename="test.pdf",
                start_char=12,
                end_char=24,
                char_count=12,
            ),
        ]

        document = EnhancedDocument(
            filename="test.pdf", metadata=metadata, chunks=chunks
        )

        chunk = document.get_chunk_by_index(1)
        assert chunk is not None
        assert chunk.text == "Second chunk"

        missing_chunk = document.get_chunk_by_index(5)
        assert missing_chunk is None

    def test_calculate_totals(self):
        """Test calculating total tokens and characters."""
        metadata = EnhancedDocumentMetadata(
            filename="test.pdf", file_type="pdf", total_chars=1000
        )

        chunks = [
            EnhancedDocumentChunk(
                text="First chunk",
                chunk_index=0,
                document_filename="test.pdf",
                start_char=0,
                end_char=11,
                char_count=11,
                token_count=3,
            ),
            EnhancedDocumentChunk(
                text="Second chunk",
                chunk_index=1,
                document_filename="test.pdf",
                start_char=12,
                end_char=24,
                char_count=12,
                token_count=4,
            ),
        ]

        document = EnhancedDocument(
            filename="test.pdf", metadata=metadata, chunks=chunks
        )

        assert document.get_total_tokens() == 7
        assert document.get_total_chars() == 23

    def test_to_langchain_documents(self):
        """Test conversion to Langchain Documents."""
        metadata = EnhancedDocumentMetadata(
            filename="test.pdf", file_type="pdf", total_chars=23
        )

        chunks = [
            EnhancedDocumentChunk(
                text="First chunk",
                chunk_index=0,
                document_filename="test.pdf",
                start_char=0,
                end_char=11,
                char_count=11,
            ),
            EnhancedDocumentChunk(
                text="Second chunk",
                chunk_index=1,
                document_filename="test.pdf",
                start_char=12,
                end_char=24,
                char_count=12,
            ),
        ]

        document = EnhancedDocument(
            filename="test.pdf", metadata=metadata, chunks=chunks
        )

        langchain_docs = document.to_langchain_documents()

        assert len(langchain_docs) == 2
        assert all(isinstance(doc, LangchainDocument) for doc in langchain_docs)
        assert langchain_docs[0].page_content == "First chunk"
        assert langchain_docs[1].page_content == "Second chunk"

    def test_from_langchain_documents(self):
        """Test creation from Langchain Documents."""
        langchain_docs = [
            LangchainDocument(
                page_content="First chunk", metadata={"page": 1, "section": "Intro"}
            ),
            LangchainDocument(
                page_content="Second chunk", metadata={"page": 2, "section": "Body"}
            ),
        ]

        document = EnhancedDocument.from_langchain_documents(
            langchain_docs, filename="test.pdf", status="completed"
        )

        assert document.filename == "test.pdf"
        assert document.status == "completed"
        assert len(document.chunks) == 2
        assert document.chunks[0].text == "First chunk"
        assert document.chunks[1].text == "Second chunk"
        assert document.metadata.langchain_source is True

    def test_calculate_processing_stats(self):
        """Test processing statistics calculation."""
        metadata = EnhancedDocumentMetadata(
            filename="test.pdf", file_type="pdf", total_chars=23
        )

        chunks = [
            EnhancedDocumentChunk(
                text="First chunk",
                chunk_index=0,
                document_filename="test.pdf",
                start_char=0,
                end_char=11,
                char_count=11,
                chunk_type="content",
                langchain_source=True,
            ),
            EnhancedDocumentChunk(
                text="Header",
                chunk_index=1,
                document_filename="test.pdf",
                start_char=12,
                end_char=18,
                char_count=6,
                chunk_type="header",
                langchain_source=True,
            ),
        ]

        document = EnhancedDocument(
            filename="test.pdf", metadata=metadata, chunks=chunks, status="completed"
        )

        stats = document.calculate_processing_stats()

        assert stats["document"]["filename"] == "test.pdf"
        assert stats["document"]["total_chunks"] == 2
        assert stats["document"]["total_characters"] == 17
        assert stats["document"]["status"] == "completed"
        assert stats["chunks"]["chunks_by_type"]["content"] == 1
        assert stats["chunks"]["chunks_by_type"]["header"] == 1
        assert stats["processing"]["langchain_used"] is True


# TestEnhancedCitation class removed (was citation-related)


class TestModelConversions:
    """Test cases for model conversion utilities."""

    def test_convert_to_enhanced_document(self):
        """Test converting base models to enhanced document."""
        base_metadata = DocumentMetadata(
            filename="test.pdf",
            file_type="pdf",
            total_chars=100,
            total_tokens=20,
            sections=["Section 1"],
        )

        base_chunks = [
            DocumentChunk(
                text="Sample chunk",
                chunk_index=0,
                document_filename="test.pdf",
                start_char=0,
                end_char=12,
                char_count=12,
            )
        ]

        enhanced_doc = convert_to_enhanced_document(
            base_metadata, base_chunks, status="completed"
        )

        assert isinstance(enhanced_doc, EnhancedDocument)
        assert enhanced_doc.filename == "test.pdf"
        assert enhanced_doc.status == "completed"
        assert len(enhanced_doc.chunks) == 1
        assert enhanced_doc.chunks[0].text == "Sample chunk"

    def test_convert_from_enhanced_document(self):
        """Test converting enhanced document back to base models."""
        metadata = EnhancedDocumentMetadata(
            filename="test.pdf", file_type="pdf", total_chars=100
        )

        chunks = [
            EnhancedDocumentChunk(
                text="Sample chunk",
                chunk_index=0,
                document_filename="test.pdf",
                start_char=0,
                end_char=12,
                char_count=12,
            )
        ]

        enhanced_doc = EnhancedDocument(
            filename="test.pdf", metadata=metadata, chunks=chunks
        )

        base_metadata, base_chunks = convert_from_enhanced_document(enhanced_doc)

        assert isinstance(base_metadata, DocumentMetadata)
        assert len(base_chunks) == 1
        assert isinstance(base_chunks[0], DocumentChunk)
        assert base_metadata.filename == "test.pdf"
        assert base_chunks[0].text == "Sample chunk"

    def test_integrate_with_langchain_pipeline(self):
        """Test integration with Langchain pipeline output."""
        langchain_docs = [
            LangchainDocument(
                page_content="First document",
                metadata={"source": "test.pdf", "page": 1},
            ),
            LangchainDocument(
                page_content="Second document",
                metadata={"source": "test.pdf", "page": 2},
            ),
        ]

        enhanced_doc = integrate_with_langchain_pipeline(
            langchain_docs, filename="test.pdf", status="completed"
        )

        assert isinstance(enhanced_doc, EnhancedDocument)
        assert enhanced_doc.filename == "test.pdf"
        assert enhanced_doc.status == "completed"
        assert len(enhanced_doc.chunks) == 2
        assert enhanced_doc.metadata.langchain_source is True


if __name__ == "__main__":
    pytest.main([__file__])
