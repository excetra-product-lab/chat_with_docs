import threading
from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.dependencies import get_current_user
from app.models.schemas import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    DocumentProcessingResult,
    ProcessingConfig,
    ProcessingStats,
)
from app.services.document_processor import DocumentProcessor, ProcessingResult
from app.services.supabase_file_service import SupabaseFileService

router = APIRouter()

# Initialize document processor
document_processor = DocumentProcessor()

# Storage service instance (lazily initialized with thread safety)
_storage_service_instance: SupabaseFileService | None = None
_storage_service_lock = threading.Lock()


def get_storage_service() -> SupabaseFileService:
    """
    Dependency to provide storage service with thread-safe lazy initialization.
    Creates the instance on first access and reuses it for subsequent calls.
    Uses double-checked locking pattern to avoid race conditions.
    """
    global _storage_service_instance

    # First check without lock (optimization for already initialized case)
    if _storage_service_instance is not None:
        return _storage_service_instance

    # Acquire lock for initialization
    with _storage_service_lock:
        # Second check with lock to handle race conditions
        if _storage_service_instance is None:
            _storage_service_instance = SupabaseFileService()

    return _storage_service_instance


async def _process_document_internal(file: UploadFile) -> ProcessingResult:
    """
    Internal function to process document and return raw result with validation.
    Used by both process_document endpoint and upload_document endpoint.

    Returns:
        ProcessingResult: The validated processing result
    """
    # Process the document
    result = await document_processor.process_document(file)

    # Validate the processing result
    if not document_processor.validate_processing_result(result):
        raise HTTPException(
            status_code=500, detail="Document processing validation failed"
        )

    return result


@router.post("/process", response_model=DocumentProcessingResult)
async def process_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Process an uploaded document: parse text and create chunks.
    This endpoint handles the document parsing and text extraction.
    """
    try:
        # Use shared processing function
        result = await _process_document_internal(file)

        # Convert to response format with proper Pydantic models
        document_metadata = DocumentMetadata(
            filename=result.parsed_content.metadata.filename,
            file_type=result.parsed_content.metadata.file_type,
            total_pages=result.parsed_content.metadata.total_pages,
            total_chars=result.parsed_content.metadata.total_chars,
            total_tokens=result.parsed_content.metadata.total_tokens,
            sections=result.parsed_content.metadata.sections,
        )

        chunks = [
            DocumentChunk(
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                document_filename=chunk.document_filename,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                char_count=chunk.char_count,
                metadata=chunk.metadata,
            )
            for chunk in result.chunks
        ]

        processing_stats = ProcessingStats(
            document=result.processing_stats["document"],
            parsing=result.processing_stats["parsing"],
            chunking=result.processing_stats["chunking"],
            processing=result.processing_stats["processing"],
        )

        return DocumentProcessingResult(
            success=True,
            message=f"Successfully processed document: {file.filename}",
            document_metadata=document_metadata,
            chunks=chunks,
            processing_stats=processing_stats,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Unexpected error processing document: {str(e)}"
        ) from e


@router.get("/processing-config", response_model=ProcessingConfig)
async def get_processing_config():
    """Get the current document processing configuration."""
    return document_processor.get_processing_config()


@router.post("/upload", response_model=Document)
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    storage_service: SupabaseFileService = Depends(get_storage_service),
):
    """
    Upload and save a document file, then process it (parse and chunk).
    This endpoint handles both file storage and document processing in one step.
    """
    try:
        user_id = current_user["id"]
        # user_id = 2

        if not file.filename:
            raise HTTPException(
                status_code=400, detail="Uploaded file must have a filename"
            )

        # Create document record in database first to get the ID
        from app.core.vectorstore import SessionLocal
        from app.models.database import Document as DBDocument

        db = SessionLocal()
        try:
            db_document = DBDocument(
                filename=file.filename,
                user_id=user_id,
                status="processing",
            )
            db.add(db_document)
            db.commit()
            db.refresh(db_document)

            # Use this single ID for everything
            document_id = db_document.id

        finally:
            db.close()

        # Upload the file using the same document ID
        storage_key = await storage_service.upload_file(file, str(document_id))

        # Reset file pointer for processing
        await file.seek(0)

        # Process the document and capture the results
        processing_result = await _process_document_internal(file)

        # Store chunks with embeddings using the same document ID
        from app.core.vectorstore import store_chunks_with_embeddings
        from app.models.schemas import DocumentChunk

        # Convert text_chunker.DocumentChunk to schemas.DocumentChunk
        schema_chunks = [
            DocumentChunk(
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                document_filename=chunk.document_filename,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                char_count=chunk.char_count,
                metadata=chunk.metadata,
            )
            for chunk in processing_result.chunks
        ]

        chunk_count = await store_chunks_with_embeddings(
            schema_chunks,
            str(document_id),  # Same ID used everywhere
        )

        # Update document with storage key and status
        db = SessionLocal()
        try:
            document_to_update: DBDocument | None = (
                db.query(DBDocument).filter(DBDocument.id == document_id).first()
            )
            if document_to_update is not None:
                document_to_update.storage_key = storage_key  # type: ignore[assignment]
                document_to_update.status = "processed"  # type: ignore[assignment]
                db.commit()
        finally:
            db.close()

        # Create response document
        document = Document(
            id=str(document_id),  # Same ID used everywhere
            filename=file.filename,
            user_id=user_id,
            status="processed",
            storage_key=storage_key,
            created_at=datetime.now(),
            chunk_count=chunk_count,
        )

        return document

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to upload and process document: {str(e)}"
        ) from e


@router.get("/", response_model=list[Document])
async def list_documents(current_user: dict = Depends(get_current_user)):
    # TODO: Fetch user's documents from database
    return []


@router.delete("/{document_id}")
async def delete_document(
    document_id: int, current_user: dict = Depends(get_current_user)
):
    # TODO: Verify ownership and delete document
    return {"message": "Document deleted successfully"}
