from pathlib import Path

from fastapi import HTTPException, UploadFile
from supabase import Client, create_client

from app.core.settings import settings


class SupabaseFileService:
    """Service to handle file upload to Supabase Storage."""

    def __init__(self) -> None:
        """Initialize the Supabase file service."""
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY must be set in environment variables"
            )

        self.supabase: Client = create_client(
            settings.SUPABASE_URL, settings.SUPABASE_KEY
        )
        self.bucket_name = settings.SUPABASE_BUCKET_NAME
        self.max_file_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024

        # Ensure bucket exists
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self) -> None:
        """Ensure the storage bucket exists."""
        try:
            # Try to get bucket info
            self.supabase.storage.get_bucket(self.bucket_name)
        except Exception:
            # If bucket doesn't exist, create it
            try:
                self.supabase.storage.create_bucket(
                    self.bucket_name,
                    options={
                        "public": False,
                        "fileSizeLimit": self.max_file_size_bytes,
                    },
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to create storage bucket: {str(e)}"
                ) from e

    async def _validate_file_metadata(self, file: UploadFile) -> None:
        # Validate file type

        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in settings.ALLOWED_FILE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"File type '{file_extension}' is not supported. "
                    f"Please upload one of these file types: "
                    f"{', '.join(settings.ALLOWED_FILE_EXTENSIONS)}. "
                    f"Make sure your file has the correct extension."
                ),
            )

    async def upload_file(self, file: UploadFile, document_id: str) -> str:
        """
        Upload file to Supabase Storage with validation.
        Returns storage_key
        """
        # Validate file type
        await self._validate_file_metadata(file)

        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        # Generate filename using document ID and file extension
        safe_filename = f"{document_id}{Path(file.filename).suffix.lower()}"

        # Create document-specific path
        file_path = f"{self.bucket_name}/{safe_filename}"

        try:
            # Read file content as bytes to fix SpooledTemporaryFile issue
            file_content = await file.read()

            # Upload file content to Supabase
            self.supabase.storage.from_(self.bucket_name).upload(
                path=file_path,
                file=file_content,
                file_options={
                    "content-type": file.content_type or "application/octet-stream",
                    "upsert": False,  # Don't overwrite existing files
                },
            )

            # Return the storage key
            return file_path

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to upload file: {str(e)}"
            ) from e
