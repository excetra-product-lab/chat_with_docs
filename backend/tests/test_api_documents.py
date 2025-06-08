"""Integration tests for documents API endpoints."""

import io

from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.main import app

# Mock user for testing


def mock_get_current_user():
    return {"id": 1, "email": "test@example.com"}


# Override the dependency
app.dependency_overrides[get_current_user] = mock_get_current_user

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
        assert "Document processing validation failed" in response.json()["detail"]

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
        assert data["processing_stats"]["document"]["total_characters"] == 31001

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
        base_sentence = "This is paragraph {i} with truly unique and substantial content. "
        for i in range(20):  # Increased paragraph count
            # Add more unique words to each paragraph
            unique_words = f"Variation {i}. " * 5
            paragraphs.append(base_sentence.format(i=i + 1) + unique_words * 15)

        content = "\\n\\n".join(paragraphs)
        test_file = self.create_test_file(
            content.encode("utf-8"), "multi_chunk.txt", "text/plain"
        )

        response = client.post("/api/documents/process", files=[test_file])

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["chunks"]) > 1

    def test_process_document_response_format(self):
        """Test the format of the response for a successful document processing."""

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
