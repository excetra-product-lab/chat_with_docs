"""Tests for SupabaseFileService."""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException, UploadFile

from app.services.supabase_file_service import SupabaseFileService

if TYPE_CHECKING:
    from pytest_mock.plugin import MockerFixture


class TestSupabaseFileService:
    """Test suite for SupabaseFileService."""

    @pytest.fixture
    def mock_settings(self, mocker: "MockerFixture") -> Mock:
        """Mock settings with required configuration."""
        mock_settings = mocker.patch("app.services.supabase_file_service.settings")
        mock_settings.SUPABASE_URL = "https://test.supabase.co"
        mock_settings.SUPABASE_KEY = "test-key"
        mock_settings.SUPABASE_BUCKET_NAME = "test-bucket"
        mock_settings.MAX_FILE_SIZE_MB = 10
        mock_settings.ALLOWED_FILE_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"]
        return mock_settings

    @pytest.fixture
    def mock_supabase_client(self, mocker: "MockerFixture") -> Mock:
        """Mock Supabase client."""
        mock_client = Mock()
        mock_storage = Mock()
        mock_bucket = Mock()

        mock_client.storage = mock_storage
        mock_storage.get_bucket.return_value = {"name": "test-bucket"}
        mock_storage.create_bucket.return_value = {"name": "test-bucket"}
        mock_storage.from_.return_value = mock_bucket
        mock_bucket.upload.return_value = {"path": "test-path"}

        mocker.patch(
            "app.services.supabase_file_service.create_client", return_value=mock_client
        )
        return mock_client

    @pytest.fixture
    def mock_upload_file(self) -> Mock:
        """Create a mock UploadFile object."""
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test_document.pdf"
        mock_file.content_type = "application/pdf"
        mock_file.read = AsyncMock(return_value=b"test file content")
        return mock_file

    def test_init_success(
        self, mock_settings: Mock, mock_supabase_client: Mock
    ) -> None:
        """Test successful initialization of SupabaseFileService."""
        service = SupabaseFileService()

        assert service.supabase == mock_supabase_client
        assert service.bucket_name == "test-bucket"
        assert service.max_file_size_bytes == 10 * 1024 * 1024

    def test_init_missing_url(self, mocker: "MockerFixture") -> None:
        """Test initialization fails when SUPABASE_URL is missing."""
        mock_settings = mocker.patch("app.services.supabase_file_service.settings")
        mock_settings.SUPABASE_URL = None
        mock_settings.SUPABASE_KEY = "test-key"

        with pytest.raises(
            ValueError, match="SUPABASE_URL and SUPABASE_KEY must be set"
        ):
            SupabaseFileService()

    def test_init_missing_key(self, mocker: "MockerFixture") -> None:
        """Test initialization fails when SUPABASE_KEY is missing."""
        mock_settings = mocker.patch("app.services.supabase_file_service.settings")
        mock_settings.SUPABASE_URL = "https://test.supabase.co"
        mock_settings.SUPABASE_KEY = None

        with pytest.raises(
            ValueError, match="SUPABASE_URL and SUPABASE_KEY must be set"
        ):
            SupabaseFileService()

    def test_init_missing_both_credentials(self, mocker: "MockerFixture") -> None:
        """Test initialization fails when both credentials are missing."""
        mock_settings = mocker.patch("app.services.supabase_file_service.settings")
        mock_settings.SUPABASE_URL = None
        mock_settings.SUPABASE_KEY = None

        with pytest.raises(
            ValueError, match="SUPABASE_URL and SUPABASE_KEY must be set"
        ):
            SupabaseFileService()

    def test_ensure_bucket_exists_bucket_exists(
        self, mock_settings: Mock, mock_supabase_client: Mock
    ) -> None:
        """Test _ensure_bucket_exists when bucket already exists."""
        SupabaseFileService()

        # Should not raise any exception
        mock_supabase_client.storage.get_bucket.assert_called_once_with("test-bucket")
        mock_supabase_client.storage.create_bucket.assert_not_called()

    def test_ensure_bucket_exists_creates_bucket(
        self, mock_settings: Mock, mock_supabase_client: Mock
    ) -> None:
        """Test _ensure_bucket_exists creates bucket when it doesn't exist."""
        # Make get_bucket raise an exception to simulate bucket not existing
        mock_supabase_client.storage.get_bucket.side_effect = Exception(
            "Bucket not found"
        )

        SupabaseFileService()

        mock_supabase_client.storage.get_bucket.assert_called_once_with("test-bucket")
        mock_supabase_client.storage.create_bucket.assert_called_once_with(
            "test-bucket", options={"public": False, "fileSizeLimit": 10 * 1024 * 1024}
        )

    def test_ensure_bucket_exists_create_fails(
        self, mock_settings: Mock, mock_supabase_client: Mock
    ) -> None:
        """Test _ensure_bucket_exists raises HTTPException when bucket creation fails."""
        # Make both operations fail
        mock_supabase_client.storage.get_bucket.side_effect = Exception(
            "Bucket not found"
        )
        mock_supabase_client.storage.create_bucket.side_effect = Exception(
            "Create failed"
        )

        with pytest.raises(HTTPException) as exc_info:
            SupabaseFileService()

        assert exc_info.value.status_code == 500
        assert "Failed to create storage bucket" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_validate_file_metadata_valid_pdf(
        self, mock_settings: Mock, mock_supabase_client: Mock
    ) -> None:
        """Test file validation with valid PDF file."""
        service = SupabaseFileService()
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "document.pdf"

        # Should not raise any exception
        await service._validate_file_metadata(mock_file)

    @pytest.mark.asyncio
    async def test_validate_file_metadata_valid_docx(
        self, mock_settings: Mock, mock_supabase_client: Mock
    ) -> None:
        """Test file validation with valid DOCX file."""
        service = SupabaseFileService()
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "document.docx"

        # Should not raise any exception
        await service._validate_file_metadata(mock_file)

    @pytest.mark.asyncio
    async def test_validate_file_metadata_case_insensitive(
        self, mock_settings: Mock, mock_supabase_client: Mock
    ) -> None:
        """Test file validation is case insensitive."""
        service = SupabaseFileService()
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "document.PDF"

        # Should not raise any exception
        await service._validate_file_metadata(mock_file)

    @pytest.mark.asyncio
    async def test_validate_file_metadata_invalid_extension(
        self, mock_settings: Mock, mock_supabase_client: Mock
    ) -> None:
        """Test file validation with invalid file extension."""
        service = SupabaseFileService()
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "document.exe"

        with pytest.raises(HTTPException) as exc_info:
            await service._validate_file_metadata(mock_file)

        assert exc_info.value.status_code == 400
        assert "File type '.exe' is not supported" in str(exc_info.value.detail)
        assert ".pdf, .docx, .txt, .md" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_validate_file_metadata_no_extension(
        self, mock_settings: Mock, mock_supabase_client: Mock
    ) -> None:
        """Test file validation with file that has no extension."""
        service = SupabaseFileService()
        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "document"

        with pytest.raises(HTTPException) as exc_info:
            await service._validate_file_metadata(mock_file)

        assert exc_info.value.status_code == 400
        assert "File type '' is not supported" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_upload_file_success(
        self, mock_settings: Mock, mock_supabase_client: Mock, mock_upload_file: Mock
    ) -> None:
        """Test successful file upload."""
        service = SupabaseFileService()
        document_id = "doc123"

        result = await service.upload_file(mock_upload_file, document_id)

        # Check that file was read
        mock_upload_file.read.assert_called_once()

        # Check that upload was called with correct parameters
        mock_bucket = mock_supabase_client.storage.from_.return_value
        mock_bucket.upload.assert_called_once_with(
            path="test-bucket/doc123.pdf",
            file=b"test file content",
            file_options={"content-type": "application/pdf", "upsert": False},
        )

        # Check return value
        assert result == "test-bucket/doc123.pdf"

    @pytest.mark.asyncio
    async def test_upload_file_no_content_type(
        self, mock_settings: Mock, mock_supabase_client: Mock
    ) -> None:
        """Test file upload when content_type is None."""
        service = SupabaseFileService()
        document_id = "doc123"

        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.pdf"
        mock_file.content_type = None
        mock_file.read = AsyncMock(return_value=b"test content")

        await service.upload_file(mock_file, document_id)

        mock_bucket = mock_supabase_client.storage.from_.return_value
        call_args = mock_bucket.upload.call_args
        assert (
            call_args[1]["file_options"]["content-type"] == "application/octet-stream"
        )

    @pytest.mark.asyncio
    async def test_upload_file_invalid_type(
        self, mock_settings: Mock, mock_supabase_client: Mock
    ) -> None:
        """Test file upload with invalid file type."""
        service = SupabaseFileService()
        document_id = "doc123"

        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "malware.exe"

        with pytest.raises(HTTPException) as exc_info:
            await service.upload_file(mock_file, document_id)

        assert exc_info.value.status_code == 400
        assert "File type '.exe' is not supported" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_upload_file_upload_fails(
        self, mock_settings: Mock, mock_supabase_client: Mock, mock_upload_file: Mock
    ) -> None:
        """Test file upload when Supabase upload fails."""
        service = SupabaseFileService()
        document_id = "doc123"

        # Make upload fail
        mock_bucket = mock_supabase_client.storage.from_.return_value
        mock_bucket.upload.side_effect = Exception("Upload failed")

        with pytest.raises(HTTPException) as exc_info:
            await service.upload_file(mock_upload_file, document_id)

        assert exc_info.value.status_code == 500
        assert "Failed to upload file: Upload failed" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_upload_file_read_fails(
        self, mock_settings: Mock, mock_supabase_client: Mock
    ) -> None:
        """Test file upload when file read fails."""
        service = SupabaseFileService()
        document_id = "doc123"

        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "test.pdf"
        mock_file.content_type = "application/pdf"
        mock_file.read = AsyncMock(side_effect=Exception("Read failed"))

        with pytest.raises(HTTPException) as exc_info:
            await service.upload_file(mock_file, document_id)

        assert exc_info.value.status_code == 500
        assert "Failed to upload file: Read failed" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_upload_file_generates_correct_path(
        self, mock_settings: Mock, mock_supabase_client: Mock
    ) -> None:
        """Test that upload_file generates correct file path."""
        service = SupabaseFileService()
        document_id = "doc-456"

        mock_file = Mock(spec=UploadFile)
        mock_file.filename = "My Document.PDF"  # Test with spaces and uppercase
        mock_file.content_type = "application/pdf"
        mock_file.read = AsyncMock(return_value=b"content")

        result = await service.upload_file(mock_file, document_id)

        # Should convert to lowercase and use document ID
        assert result == "test-bucket/doc-456.pdf"

        # Check upload was called with correct path
        mock_bucket = mock_supabase_client.storage.from_.return_value
        call_args = mock_bucket.upload.call_args
        assert call_args[1]["path"] == "test-bucket/doc-456.pdf"

    @pytest.mark.asyncio
    async def test_upload_file_different_extensions(
        self, mock_settings: Mock, mock_supabase_client: Mock
    ) -> None:
        """Test upload with different valid file extensions."""
        service = SupabaseFileService()
        document_id = "doc123"

        test_cases = [
            ("document.txt", ".txt"),
            ("document.md", ".md"),
            ("document.docx", ".docx"),
            ("DOCUMENT.TXT", ".txt"),  # Test case conversion
        ]

        for filename, expected_ext in test_cases:
            mock_file = Mock(spec=UploadFile)
            mock_file.filename = filename
            mock_file.content_type = "text/plain"
            mock_file.read = AsyncMock(return_value=b"content")

            result = await service.upload_file(mock_file, document_id)

            assert result == f"test-bucket/doc123{expected_ext}"

            # Reset mock for next iteration
            mock_supabase_client.storage.from_.return_value.upload.reset_mock()
