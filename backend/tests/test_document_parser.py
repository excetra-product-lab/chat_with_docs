"""Tests for document parser service."""

import io

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from app.services.document_parser import DocumentMetadata, DocumentParser, ParsedContent


class TestDocumentParser:
    """Test cases for DocumentParser."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = DocumentParser()

    def create_upload_file(
        self, content: bytes, filename: str, content_type: str | None = None
    ):
        """Helper to create UploadFile for testing."""
        file_obj = io.BytesIO(content)
        headers = Headers({"content-type": content_type} if content_type else {})
        return UploadFile(
            file=file_obj, filename=filename, size=len(content), headers=headers
        )

    def test_validate_file_success(self):
        """Test successful file validation."""
        file = self.create_upload_file(b"test content", "test.txt")
        # Should not raise exception
        self.parser._validate_file(file)

    def test_validate_file_no_filename(self):
        """Test validation fails with no filename."""
        file = self.create_upload_file(b"test content", "")
        with pytest.raises(HTTPException) as exc_info:
            self.parser._validate_file(file)
        assert exc_info.value.status_code == 400
        assert "No filename provided" in str(exc_info.value.detail)

    def test_validate_file_too_large(self):
        """Test validation fails with oversized file."""
        # Create a file larger than MAX_FILE_SIZE
        large_content = b"x" * (self.parser.MAX_FILE_SIZE + 1)
        file = self.create_upload_file(large_content, "large.txt")
        with pytest.raises(HTTPException) as exc_info:
            self.parser._validate_file(file)
        assert exc_info.value.status_code == 400
        assert "File too large" in str(exc_info.value.detail)

    def test_get_file_type_from_content_type(self):
        """Test file type detection from content type."""
        assert self.parser._get_file_type("application/pdf", "test.pdf") == "pdf"
        assert (
            self.parser._get_file_type(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "test.docx",
            )
            == "docx"
        )
        assert self.parser._get_file_type("text/plain", "test.txt") == "txt"

    def test_get_file_type_from_extension(self):
        """Test file type detection from file extension."""
        assert self.parser._get_file_type(None, "test.pdf") == "pdf"
        assert self.parser._get_file_type(None, "test.docx") == "docx"
        assert self.parser._get_file_type(None, "test.txt") == "txt"
        assert self.parser._get_file_type(None, "TEST.PDF") == "pdf"  # Case insensitive

    def test_get_file_type_unsupported(self):
        """Test unsupported file type raises exception."""
        with pytest.raises(HTTPException) as exc_info:
            self.parser._get_file_type("application/unknown", "test.unknown")
        assert exc_info.value.status_code == 400
        assert "Unsupported file format" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_parse_text_file_utf8(self):
        """Test parsing UTF-8 text file."""
        content = "This is a test document.\n\nIt has multiple paragraphs.\n\nAnd some unicode: café"
        self.create_upload_file(content.encode("utf-8"), "test.txt", "text/plain")

        result = await self.parser._parse_text(content.encode("utf-8"), "test.txt")

        assert isinstance(result, ParsedContent)
        assert result.text == content
        assert result.metadata.filename == "test.txt"
        assert result.metadata.file_type == "txt"
        assert result.metadata.total_chars == len(content)
        assert len(result.structured_content) > 0

    @pytest.mark.asyncio
    async def test_parse_text_file_different_encodings(self):
        """Test parsing text files with different encodings."""
        content = "Test with special chars: café, naïve, résumé"

        # Test UTF-16
        utf16_content = content.encode("utf-16")
        result = await self.parser._parse_text(utf16_content, "test_utf16.txt")
        assert content in result.text

        # Test Latin-1
        latin1_content = "Test with latin chars: café".encode("latin-1")
        result = await self.parser._parse_text(latin1_content, "test_latin1.txt")
        assert "café" in result.text

    @pytest.mark.asyncio
    async def test_parse_text_file_empty(self):
        """Test parsing empty text file raises exception."""
        with pytest.raises(HTTPException) as exc_info:
            await self.parser._parse_text(b"", "empty.txt")
        assert exc_info.value.status_code == 400
        assert "No text content found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_parse_text_file_with_fallback_encoding(self):
        """Test parsing text file that requires fallback encoding."""
        # Create content that will fail UTF-8 but succeed with latin-1
        # This tests that our encoding fallback mechanism works
        latin1_content = "Test with special chars: café".encode("latin-1")
        result = await self.parser._parse_text(latin1_content, "latin1.txt")

        assert isinstance(result, ParsedContent)
        assert "café" in result.text
        assert result.metadata.filename == "latin1.txt"
        assert result.metadata.file_type == "txt"

    @pytest.mark.asyncio
    async def test_parse_document_text_file(self):
        """Test full document parsing for text file."""
        content = "This is a test document.\n\nIt has multiple paragraphs."
        file = self.create_upload_file(
            content.encode("utf-8"), "test.txt", "text/plain"
        )

        result = await self.parser.parse_document(file)

        assert isinstance(result, ParsedContent)
        assert result.text == content
        assert result.metadata.filename == "test.txt"
        assert result.metadata.file_type == "txt"
        assert result.metadata.total_chars == len(content)
        assert result.metadata.total_tokens > 0  # Should have token count
        assert isinstance(result.metadata.total_tokens, int)
        # Token count should be reasonable for this text (roughly 10-15 tokens)
        assert 5 <= result.metadata.total_tokens <= 20
        assert len(result.structured_content) > 0

    @pytest.mark.asyncio
    async def test_token_counting_consistency(self):
        """Test that token counting is consistent and reasonable."""
        content = "Hello world! This is a test document with multiple sentences."
        file = self.create_upload_file(
            content.encode("utf-8"), "test.txt", "text/plain"
        )

        result = await self.parser.parse_document(file)

        # Verify token count is present and reasonable
        assert result.metadata.total_tokens > 0
        assert isinstance(result.metadata.total_tokens, int)
        # Should be more than 1 token but less than character count
        assert 1 < result.metadata.total_tokens < len(content)

        # Parse the same content again to ensure consistency
        file2 = self.create_upload_file(
            content.encode("utf-8"), "test2.txt", "text/plain"
        )
        result2 = await self.parser.parse_document(file2)

        # Token counts should be identical for identical content
        assert result.metadata.total_tokens == result2.metadata.total_tokens

    @pytest.mark.asyncio
    async def test_parse_document_unsupported_format(self):
        """Test parsing unsupported file format."""
        file = self.create_upload_file(b"test", "test.xyz", "application/unknown")

        with pytest.raises(HTTPException) as exc_info:
            await self.parser.parse_document(file)
        assert exc_info.value.status_code == 400
        assert "Unsupported file format" in str(exc_info.value.detail)

    def test_supported_formats(self):
        """Test that supported formats are correctly defined."""
        formats = self.parser.SUPPORTED_FORMATS
        assert "application/pdf" in formats
        assert (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            in formats
        )
        assert "text/plain" in formats

        assert formats["application/pdf"] == "pdf"
        assert (
            formats[
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ]
            == "docx"
        )
        assert formats["text/plain"] == "txt"


class TestDocumentMetadata:
    """Test cases for DocumentMetadata."""

    def test_document_metadata_creation(self):
        """Test creating DocumentMetadata."""
        metadata = DocumentMetadata(
            filename="test.pdf",
            file_type="pdf",
            total_pages=5,
            total_chars=1000,
            total_tokens=250,
            sections=["Introduction", "Conclusion"],
        )

        assert metadata.filename == "test.pdf"
        assert metadata.file_type == "pdf"
        assert metadata.total_pages == 5
        assert metadata.total_chars == 1000
        assert metadata.total_tokens == 250
        assert metadata.sections == ["Introduction", "Conclusion"]

    def test_document_metadata_defaults(self):
        """Test DocumentMetadata with default values."""
        metadata = DocumentMetadata(filename="test.txt", file_type="txt")

        assert metadata.filename == "test.txt"
        assert metadata.file_type == "txt"
        assert metadata.total_pages is None
        assert metadata.total_chars == 0
        assert metadata.total_tokens == 0
        assert metadata.sections == []


class TestParsedContent:
    """Test cases for ParsedContent."""

    def test_parsed_content_creation(self):
        """Test creating ParsedContent."""
        metadata = DocumentMetadata("test.txt", "txt", total_chars=100)
        content = ParsedContent(
            text="Test content",
            metadata=metadata,
            page_texts=["Page 1", "Page 2"],
            structured_content=[{"type": "paragraph", "text": "Test"}],
        )

        assert content.text == "Test content"
        assert content.metadata == metadata
        assert content.page_texts == ["Page 1", "Page 2"]
        assert content.structured_content == [{"type": "paragraph", "text": "Test"}]

    def test_parsed_content_defaults(self):
        """Test ParsedContent with default values."""
        metadata = DocumentMetadata("test.txt", "txt")
        content = ParsedContent(text="Test", metadata=metadata)

        assert content.text == "Test"
        assert content.metadata == metadata
        assert content.page_texts == []
        assert content.structured_content == []
