"""Integration tests for documents API endpoints."""

import io
from unittest.mock import patch, MagicMock
from datetime import datetime

from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.api.routes.documents import get_storage_service
from app.main import app

# Mock user for testing


def mock_get_current_user():
    return {"id": 1, "email": "test@example.com"}


# Mock storage service for testing
class MockStorageService:
    """Mock implementation of SupabaseFileService for testing."""
    
    async def upload_file(self, file, document_id: str) -> str:
        """Mock file upload that returns a fake storage key."""
        # Return a fake storage key based on document ID and file extension
        file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'txt'
        return f"test-bucket/{document_id}.{file_extension}"


# Create mock instances
mock_storage = MockStorageService()

def mock_get_storage_service():
    return mock_storage


# Override the dependencies
app.dependency_overrides[get_current_user] = mock_get_current_user
app.dependency_overrides[get_storage_service] = mock_get_storage_service

client = TestClient(app)


class TestDocumentsAPI:
    """Test cases for documents API endpoints."""

    def create_test_file(self, content: bytes, filename: str, content_type: str = "text/plain"):
        """Helper to create test file for upload."""
        return ("file", (filename, io.BytesIO(content), content_type))

    def test_get_processing_config(self):
        """Test getting processing configuration."""

        response = client.get("/api/documents/processing-config")

        assert response.status_code == 200
        data = response.json()

        assert "chunk_size" in data
        assert "chunk_overlap" in data
        assert "min_chunk_size" in data
        assert "max_file_size_mb" in data
        assert "supported_formats" in data
        assert isinstance(data["supported_formats"], list)
        assert len(data["supported_formats"]) > 0

    def test_process_text_document_success(self):
        """Test successful text document processing."""

        content = "This is a test document for processing. " * 10
        test_file = self.create_test_file(content.encode("utf-8"), "test.txt", "text/plain")

        response = client.post("/api/documents/process", files=[test_file])

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "Successfully processed document" in data["message"]
        assert data["document_metadata"]["filename"] == "test.txt"
        assert data["document_metadata"]["file_type"] == "txt"
        assert len(data["chunks"]) > 0

        # Verify chunk structure
        chunk = data["chunks"][0]
        assert "text" in chunk
        assert "chunk_index" in chunk
        assert "document_filename" in chunk
        assert "char_count" in chunk
        assert chunk["document_filename"] == "test.txt"

        # Verify processing stats
        assert "document" in data["processing_stats"]
        assert "parsing" in data["processing_stats"]
        assert "chunking" in data["processing_stats"]
        assert "processing" in data["processing_stats"]
        assert data["processing_stats"]["processing"]["success"] is True

    def test_process_document_unsupported_format(self):
        """Test processing unsupported file format."""

        test_file = self.create_test_file(b"test content", "test.xyz", "application/unknown")

        response = client.post("/api/documents/process", files=[test_file])

        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]

    def test_process_document_empty_file(self):
        """Test processing empty file."""

        test_file = self.create_test_file(b"", "empty.txt", "text/plain")

        response = client.post("/api/documents/process", files=[test_file])

        # The document processor wraps the original HTTPException in a 500 error
        assert response.status_code == 500
        assert "No text content found" in response.json()["detail"]

    def test_process_document_no_filename(self):
        """Test processing file without filename."""

        test_file = self.create_test_file(b"test content", "", "text/plain")

        response = client.post("/api/documents/process", files=[test_file])

        # FastAPI returns 422 for validation errors
        assert response.status_code == 422

    def test_process_document_large_file(self):
        """Test processing large file within limits."""

        # Create a moderately large file (but within limits)
        content = "This is a large test document. " * 1000  # About 32KB
        test_file = self.create_test_file(content.encode("utf-8"), "large.txt", "text/plain")

        response = client.post("/api/documents/process", files=[test_file])

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        # With default chunk size of 1000, this might create only 1 chunk due to repetitive content
        assert len(data["chunks"]) >= 1  # At least one chunk should be created
        assert data["processing_stats"]["document"]["total_characters"] == len(content)

    def test_process_document_unicode_content(self):
        """Test processing document with unicode content."""

        # Make content longer to meet minimum chunk size requirements
        content = (
            "Document with unicode: café, naïve, résumé, 中文, العربية, 🚀. " * 5
            + "This document contains various unicode characters "
            + "from different languages and scripts. " * 3
        )
        test_file = self.create_test_file(content.encode("utf-8"), "unicode.txt", "text/plain")

        response = client.post("/api/documents/process", files=[test_file])

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        # Check that unicode characters are preserved
        full_text = " ".join(chunk["text"] for chunk in data["chunks"])
        assert "café" in full_text
        assert "中文" in full_text
        assert "العربية" in full_text
        assert "🚀" in full_text

    def test_process_document_structured_content(self):
        """Test processing document with structured content."""

        content = """INTRODUCTION

This is the introduction section of the document.

MAIN CONTENT

This is the main content section with detailed information.

CONCLUSION

This is the conclusion section that summarizes everything."""

        test_file = self.create_test_file(content.encode("utf-8"), "structured.txt", "text/plain")

        response = client.post("/api/documents/process", files=[test_file])

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["chunks"]) > 0

        # Check that structured content was detected
        parsing_stats = data["processing_stats"]["parsing"]
        assert parsing_stats["structured_content_items"] > 0

    def test_process_document_unauthorized(self):
        """Test processing document without authentication."""
        # Temporarily remove the dependency override to test unauthorized access
        original_override = app.dependency_overrides.get(get_current_user)
        if get_current_user in app.dependency_overrides:
            del app.dependency_overrides[get_current_user]

        try:
            test_file = self.create_test_file(
                b"test content that is long enough to meet minimum requirements for processing",
                "test.txt",
                "text/plain",
            )

            response = client.post("/api/documents/process", files=[test_file])

            # Should return 401 or 403 depending on auth implementation
            assert response.status_code in [401, 403]
        finally:
            # Restore the dependency override
            if original_override:
                app.dependency_overrides[get_current_user] = original_override

    def test_process_multiple_chunks(self):
        """Test processing document that creates multiple chunks."""

        # Create content that will definitely create multiple chunks
        paragraphs = []
        for i in range(10):
            paragraphs.append(f"This is paragraph {i+1} with substantial content. " * 20)

        content = "\n\n".join(paragraphs)
        test_file = self.create_test_file(content.encode("utf-8"), "multi_chunk.txt", "text/plain")

        response = client.post("/api/documents/process", files=[test_file])

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert len(data["chunks"]) > 1

        # Verify chunk indices are sequential
        chunk_indices = [chunk["chunk_index"] for chunk in data["chunks"]]
        assert chunk_indices == list(range(len(chunk_indices)))

        # Verify all chunks have the same document filename
        assert all(chunk["document_filename"] == "multi_chunk.txt" for chunk in data["chunks"])

        # Verify chunk statistics
        chunking_stats = data["processing_stats"]["chunking"]
        assert chunking_stats["total_chunks"] == len(data["chunks"])
        assert chunking_stats["total_characters"] > 0
        assert chunking_stats["average_chunk_size"] > 0

    def test_process_document_response_format(self):
        """Test that response format matches expected schema."""

        # Make content longer to meet minimum chunk size requirements
        content = "Test document for response format validation. " * 10
        test_file = self.create_test_file(content.encode("utf-8"), "format_test.txt", "text/plain")

        response = client.post("/api/documents/process", files=[test_file])

        assert response.status_code == 200
        data = response.json()

        # Verify top-level structure
        required_fields = ["success", "message", "document_metadata", "chunks", "processing_stats"]
        for field in required_fields:
            assert field in data

        # Verify document_metadata structure
        metadata = data["document_metadata"]
        metadata_fields = ["filename", "file_type", "total_chars"]
        for field in metadata_fields:
            assert field in metadata

        # Verify chunk structure
        if data["chunks"]:
            chunk = data["chunks"][0]
            chunk_fields = [
                "text",
                "chunk_index",
                "document_filename",
                "char_count",
                "start_char",
                "end_char",
                "metadata",
            ]
            for field in chunk_fields:
                assert field in chunk

        # Verify processing_stats structure
        stats = data["processing_stats"]
        stats_sections = ["document", "parsing", "chunking", "processing"]
        for section in stats_sections:
            assert section in stats

    # Upload endpoint tests
    @patch('app.api.routes.documents.datetime')
    @patch('app.api.routes.documents.uuid.uuid4')
    def test_upload_document_success(self, mock_uuid4, mock_datetime):
        """Test successful document upload with mocked UUID and datetime."""
        
        # Mock UUID generation
        mock_uuid4.return_value = "test-doc-id-123"
        
        # Mock datetime
        fixed_datetime = datetime(2024, 1, 15, 10, 30, 0)
        mock_datetime.now.return_value = fixed_datetime
        
        # Create test file
        test_content = b"Test document content for upload"
        test_file = self.create_test_file(test_content, "test.txt", "text/plain")
        
        # Make request
        response = client.post("/api/documents/upload", files=[test_file])
        
        # Test the API response
        assert response.status_code == 200
        data = response.json()
        
        assert data["id"] == "test-doc-id-123"
        assert data["filename"] == "test.txt"
        assert data["user_id"] == 1
        assert data["status"] == "uploaded"
        assert data["storage_key"] == "test-bucket/test-doc-id-123.txt"
        assert data["created_at"] == "2024-01-15T10:30:00"

    def test_upload_storage_service_error(self):
        """Test handling of storage service errors."""
        # Create a mock that raises an exception
        error_storage = MagicMock()
        error_storage.upload_file.side_effect = Exception("Storage service error")
        
        # Temporarily override the storage service
        original_override = app.dependency_overrides.get(get_storage_service)
        app.dependency_overrides[get_storage_service] = lambda: error_storage
        
        try:
            test_file = self.create_test_file(b"Test content", "test.txt")
            response = client.post("/api/documents/upload", files=[test_file])
            
            assert response.status_code == 500
            assert "Failed to upload file" in response.json()["detail"]
            
        finally:
            # Restore the original override
            if original_override:
                app.dependency_overrides[get_storage_service] = original_override
