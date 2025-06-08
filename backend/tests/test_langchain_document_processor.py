"""Tests for the Langchain document processor service."""

import os
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException
from langchain_core.documents import Document

from app.services.document_parser import DocumentMetadata, ParsedContent
from app.services.langchain_document_processor import LangchainDocumentProcessor


class TestLangchainDocumentProcessor:
    """Test cases for LangchainDocumentProcessor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = LangchainDocumentProcessor()

    def create_upload_file(self, content: bytes, filename: str, content_type: str):
        """Helper to create mock UploadFile objects."""
        mock_file = Mock()
        mock_file.filename = filename
        mock_file.content_type = content_type
        mock_file.size = len(content)
        mock_file.read = AsyncMock(return_value=content)
        mock_file.seek = AsyncMock()
        return mock_file

    def test_init(self):
        """Test processor initialization."""
        processor = LangchainDocumentProcessor("cl100k_base")
        assert processor.token_counter is not None
        assert processor.chunk_config is not None
        assert processor.MAX_FILE_SIZE == 50 * 1024 * 1024

    def test_get_supported_formats(self):
        """Test getting supported file formats."""
        formats = self.processor.get_supported_formats()
        expected_formats = [
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
            "text/plain",
        ]
        assert all(fmt in formats for fmt in expected_formats)

    def test_get_processing_config(self):
        """Test getting processing configuration."""
        config = self.processor.get_processing_config()
        assert all(k in config for k in ["chunk_size", "chunk_overlap", "max_tokens_per_chunk"])

    def test_validate_file_success(self):
        """Test successful file validation."""
        file = self.create_upload_file(b"test", "test.pdf", "application/pdf")
        self.processor._validate_file(file)

    def test_validate_file_no_filename(self):
        """Test file validation with no filename."""
        file = self.create_upload_file(b"test", "", "application/pdf")
        with pytest.raises(HTTPException, match="No filename provided"):
            self.processor._validate_file(file)

    def test_validate_file_too_large(self):
        """Test file validation with oversized file."""
        large_content = b"x" * (self.processor.MAX_FILE_SIZE + 1)
        file = self.create_upload_file(large_content, "large.pdf", "application/pdf")
        with pytest.raises(HTTPException, match="File too large"):
            self.processor._validate_file(file)

    def test_get_file_type(self):
        """Test file type detection."""
        assert self.processor._get_file_type("application/pdf", "test.pdf") == "pdf"
        assert self.processor._get_file_type(None, "test.docx") == "docx"
        with pytest.raises(HTTPException, match="Unsupported file format"):
            self.processor._get_file_type(None, "test.xyz")

    def test_convert_to_parsed_content(self):
        """Test converting Langchain documents to ParsedContent."""
        docs = [Document(page_content="Page 1", metadata={"page": 0})]
        result = self.processor._convert_to_parsed_content(docs, "test.pdf", "pdf")
        assert isinstance(result, ParsedContent)
        assert "--- Page 1 ---" in result.text

    def test_create_metadata_with_tokens(self):
        """Test metadata creation with token counting."""
        metadata = self.processor._create_metadata_with_tokens("f.txt", "txt", "text", 1, [])
        assert isinstance(metadata, DocumentMetadata)
        assert metadata.total_tokens > 0

    def test_create_langchain_text_splitter(self):
        """Test creating a text splitter."""
        splitter = self.processor.create_langchain_text_splitter(chunk_size=500, chunk_overlap=50)
        assert splitter._chunk_size == 500
        assert splitter._chunk_overlap == 50

    def test_split_documents_with_langchain(self):
        """Test splitting documents."""
        docs = [Document(page_content="a" * 1000)]
        split_docs = self.processor.split_documents_with_langchain(
            docs, chunk_size=100, chunk_overlap=10
        )
        assert len(split_docs) > 1

    @patch("app.services.langchain_document_processor.PyPDFLoader")
    @pytest.mark.asyncio
    async def test_load_pdf_with_langchain(self, mock_loader):
        """Test PDF loading."""
        mock_loader.return_value.load.return_value = [Document(page_content="...")]
        await self.processor._load_pdf_with_langchain("dummy.pdf")
        mock_loader.assert_called_with("dummy.pdf")

    @patch("app.services.langchain_document_processor.UnstructuredWordDocumentLoader")
    @pytest.mark.asyncio
    async def test_load_word_with_langchain(self, mock_loader):
        """Test Word loading."""
        mock_loader.return_value.load.return_value = [Document(page_content="...")]
        await self.processor._load_word_with_langchain("dummy.docx")
        mock_loader.assert_called_with("dummy.docx")

    @patch("app.services.langchain_document_processor.TextLoader")
    @pytest.mark.asyncio
    async def test_load_text_with_langchain(self, mock_loader):
        """Test text loading with fallback."""
        mock_loader.side_effect = [
            Mock(load=Mock(side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, ""))),
            Mock(load=Mock(return_value=[Document(page_content="...")])),
        ]
        await self.processor._load_text_with_langchain("dummy.txt")
        assert mock_loader.call_count == 2

    @patch("app.services.langchain_document_processor.TextLoader")
    @pytest.mark.asyncio
    async def test_load_text_all_encodings_fail(self, mock_loader):
        """Test text loading failure."""
        mock_loader.return_value.load.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "")
        with pytest.raises(ValueError, match="Could not decode text file"):
            await self.processor._load_text_with_langchain("dummy.txt")


@pytest.mark.integration
class TestLangchainDocumentProcessorIntegration:
    """Integration tests for LangchainDocumentProcessor."""

    def setup_method(self):
        self.processor = LangchainDocumentProcessor()

    @pytest.mark.asyncio
    async def test_text_file_processing_integration(self):
        """Test processing a real text file."""
        content = "This is a test."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            with open(path, "rb") as file_bytes_obj:
                upload_file = Mock()
                upload_file.filename = "test.txt"
                upload_file.content_type = "text/plain"
                upload_file.size = len(content)
                upload_file.read = AsyncMock(return_value=file_bytes_obj.read())

                result = await self.processor.process_document_with_langchain(upload_file)
                assert content in result.text
        finally:
            os.unlink(path)

    def test_splitter_integration(self):
        """Test splitter with real documents."""
        docs = [Document(page_content="a" * 1000)]
        splitter = self.processor.create_langchain_text_splitter(chunk_size=100, chunk_overlap=10)
        split_docs = splitter.split_documents(docs)
        assert len(split_docs) > 9
