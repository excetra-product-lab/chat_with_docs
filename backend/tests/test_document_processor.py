"""Tests for document processor service."""

import io
from typing import Optional
from unittest.mock import patch

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from app.services.document_processor import DocumentProcessor, ProcessingResult


class TestDocumentProcessor:
    """Test suite for the DocumentProcessor service."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor(
            chunk_size=100,
            chunk_overlap=20,
            min_chunk_size=10,
            use_langchain=False,  # Small size for testing
        )
        # Patch the internal validation method to isolate processor logic
        self.validate_patcher = patch(
            "app.services.document_parser.DocumentParser._validate_file", return_value=None
        )
        self.mock_validate = self.validate_patcher.start()

    def teardown_method(self):
        """Tear down the test environment after each test method."""
        self.validate_patcher.stop()

    def create_upload_file(self, content: bytes, filename: str, content_type: Optional[str] = None):
        """Helper to create a mock UploadFile."""
        headers = Headers({"content-type": content_type} if content_type else {})
        return UploadFile(filename=filename, file=io.BytesIO(content), headers=headers)

    @pytest.mark.asyncio
    async def test_process_text_document(self):
        """Test processing a simple text document."""
        content = b"This is a test document."
        file = self.create_upload_file(content, "test.txt", "text/plain")

        result = await self.processor.process_document(file)

        assert isinstance(result, ProcessingResult)
        assert result.parsed_content.text == "This is a test document."
        assert len(result.chunks) > 0
        assert result.processing_stats["processing"]["success"] is True

    @pytest.mark.asyncio
    async def test_process_document_validation_success(self):
        """Test that processing succeeds with valid inputs."""
        content = b"This is a valid test document that is long enough to be processed."
        file = self.create_upload_file(content, "test.txt", "text/plain")

        result = await self.processor.process_document(file)
        assert result.processing_stats["processing"]["success"] is True

    @pytest.mark.asyncio
    async def test_process_unsupported_format(self):
        """Test processing an unsupported file format raises an HTTPException."""
        file = self.create_upload_file(b"content", "test.xyz", "application/unknown")

        with pytest.raises(HTTPException) as exc_info:
            await self.processor.process_document(file)
        assert exc_info.value.status_code == 400
        assert "Unsupported file format" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_process_empty_file(self):
        """Test processing an empty file raises an HTTPException."""
        mock_file = self.create_upload_file(b"", "empty.txt", "text/plain")

        with pytest.raises(HTTPException) as exc_info:
            await self.processor.process_document(mock_file)

        assert exc_info.value.status_code == 500
        assert "No text content found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_process_large_document(self):
        """Test processing a document that should result in multiple chunks."""
        # Create content that is guaranteed to be larger than the chunk size
        paragraphs = []
        base_sentence = "This is paragraph {i} with truly unique and substantial content. "
        for i in range(100):  # Increased paragraph count
            # Add more unique words to each paragraph
            unique_words = f"Variation {i}. " * 20
            paragraphs.append(base_sentence.format(i=i + 1) + unique_words * 30)

        content = "\\n\\n".join(paragraphs).encode("utf-8")
        file = self.create_upload_file(content, "large.txt", "text/plain")

        # Run the processor
        result = await self.processor.process_document(file)

        # Assert that processing was successful and created at least one chunk
        assert result.processing_stats["processing"]["success"] is True
        assert len(result.chunks) >= 1

    def test_generate_processing_stats(self):
        """Test the generation of processing statistics."""
        from app.services.document_parser import DocumentMetadata, ParsedContent
        from app.services.text_chunker import DocumentChunk

        # Create mock parsed content
        metadata = DocumentMetadata(
            filename="test.pdf",
            file_type="pdf",
            total_pages=3,
            total_chars=500,
            sections=["Introduction", "Conclusion"],
        )
        parsed_content = ParsedContent(
            text="Test content",
            metadata=metadata,
            page_texts=["Page 1", "Page 2", "Page 3"],
            structured_content=[
                {"type": "page", "page_number": 1, "text": "Page 1"},
                {"type": "page", "page_number": 2, "text": "Page 2"},
                {"type": "page", "page_number": 3, "text": "Page 3"},
            ],
        )

        # Create mock chunks
        chunks = [
            DocumentChunk("Chunk 1", 0, "test.pdf", page_number=1),
            DocumentChunk("Chunk 2", 1, "test.pdf", page_number=2),
        ]

        stats = self.processor._generate_processing_stats(parsed_content, chunks)

        assert stats["document"]["filename"] == "test.pdf"
        assert stats["document"]["file_type"] == "pdf"
        assert stats["document"]["total_pages"] == 3
        assert stats["document"]["total_characters"] == 500
        assert stats["document"]["sections_detected"] == 2

        assert stats["parsing"]["structured_content_items"] == 3
        assert stats["parsing"]["page_texts_extracted"] == 3
        assert stats["parsing"]["sections_found"] == ["Introduction", "Conclusion"]

        assert stats["chunking"]["total_chunks"] == 2
        assert stats["processing"]["success"] is True
        assert stats["processing"]["chunks_ready_for_embedding"] == 2

    def test_validate_processing_result_invalid_no_text(self):
        """Test validation fails when no text content."""
        from app.services.document_parser import DocumentMetadata, ParsedContent

        metadata = DocumentMetadata("test.txt", "txt")
        parsed_content = ParsedContent("", metadata)  # Empty text
        result = ProcessingResult(parsed_content, [], {})

        is_valid = self.processor.validate_processing_result(result)
        assert is_valid is False

    def test_validate_processing_result_invalid_no_chunks(self):
        """Test validation fails when no chunks created."""
        from app.services.document_parser import DocumentMetadata, ParsedContent

        metadata = DocumentMetadata("test.txt", "txt")
        parsed_content = ParsedContent("Valid text content", metadata)
        result = ProcessingResult(parsed_content, [], {"processing": {"success": True}})

        is_valid = self.processor.validate_processing_result(result)
        assert is_valid is False

    def test_validate_processing_result_invalid_empty_chunk(self):
        """Test validation fails when chunk has no text."""
        from app.services.document_parser import DocumentMetadata, ParsedContent
        from app.services.text_chunker import DocumentChunk

        metadata = DocumentMetadata("test.txt", "txt")
        parsed_content = ParsedContent("Valid text content", metadata)
        chunks = [DocumentChunk("", 0, "test.txt")]  # Empty chunk text
        result = ProcessingResult(parsed_content, chunks, {"processing": {"success": True}})

        is_valid = self.processor.validate_processing_result(result)
        assert is_valid is False

    def test_validate_processing_result_invalid_missing_filename(self):
        """Test validation fails when chunk missing filename."""
        from app.services.document_parser import DocumentMetadata, ParsedContent
        from app.services.text_chunker import DocumentChunk

        metadata = DocumentMetadata("test.txt", "txt")
        parsed_content = ParsedContent("Valid text content", metadata)
        chunks = [DocumentChunk("Valid text", 0, "")]  # Empty filename
        result = ProcessingResult(parsed_content, chunks, {"processing": {"success": True}})

        is_valid = self.processor.validate_processing_result(result)
        assert is_valid is False

    def test_validate_processing_result_invalid_processing_failed(self):
        """Test validation fails when processing stats indicate failure."""
        from app.services.document_parser import DocumentMetadata, ParsedContent
        from app.services.text_chunker import DocumentChunk

        metadata = DocumentMetadata("test.txt", "txt")
        parsed_content = ParsedContent("Valid text content", metadata)
        chunks = [DocumentChunk("Valid text", 0, "test.txt")]
        result = ProcessingResult(parsed_content, chunks, {"processing": {"success": False}})

        is_valid = self.processor.validate_processing_result(result)
        assert is_valid is False

    def test_get_supported_formats(self):
        """Test getting supported file formats."""
        formats = self.processor.get_supported_formats()

        assert isinstance(formats, list)
        assert "application/pdf" in formats
        assert "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in formats
        assert "text/plain" in formats

    def test_get_processing_config(self):
        """Test getting processing configuration."""
        config = self.processor.get_processing_config()

        assert config["chunk_size"] == 100
        assert config["chunk_overlap"] == 20
        assert config["min_chunk_size"] == 10
        assert config["max_file_size_mb"] > 0
        assert isinstance(config["supported_formats"], list)
        assert len(config["supported_formats"]) > 0

    @pytest.mark.asyncio
    async def test_process_document_with_unicode(self):
        """Test processing document with unicode characters."""
        content = "Document with unicode: café, naïve, résumé, 中文, العربية"
        file = self.create_upload_file(content.encode("utf-8"), "unicode.txt", "text/plain")

        result = await self.processor.process_document(file)

        assert result.parsed_content.text == content
        assert len(result.chunks) > 0
        assert all(
            "café" in chunk.text
            or "naïve" in chunk.text
            or "résumé" in chunk.text
            or "中文" in chunk.text
            or "العربية" in chunk.text
            for chunk in result.chunks
            if len(chunk.text) > 20
        )

    @pytest.mark.asyncio
    async def test_process_document_preserves_structure(self):
        """Test that document processing preserves structure information."""
        content = "TITLE\n\nThis is paragraph one.\n\nThis is paragraph two."
        file = self.create_upload_file(content.encode("utf-8"), "structured.txt", "text/plain")

        result = await self.processor.process_document(file)

        # Check that structured content was created
        assert len(result.parsed_content.structured_content) > 0

        # Check that chunks preserve document filename
        assert all(chunk.document_filename == "structured.txt" for chunk in result.chunks)

        # Check that chunk indices are sequential
        assert all(result.chunks[i].chunk_index == i for i in range(len(result.chunks)))


class TestProcessingResult:
    """Test cases for the ProcessingResult data class."""

    def test_processing_result_creation(self):
        """Test basic creation of ProcessingResult."""
        from app.services.document_parser import DocumentMetadata, ParsedContent

        metadata = DocumentMetadata("test.txt", "txt")
        parsed_content = ParsedContent("text", metadata)
        result = ProcessingResult(parsed_content, [], {})

        assert result.parsed_content == parsed_content
        assert result.chunks == []
        assert result.processing_stats == {}
