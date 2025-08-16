import logging
import threading
from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.dependencies import get_current_user
from app.models.langchain_models import convert_to_enhanced_document
from app.models.schemas import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    DocumentProcessingResult,
    ProcessingConfig,
    ProcessingStats,
)

# EnhancedCitation import removed
from app.services.document_processor import DocumentProcessor, ProcessingResult
from app.services.enhanced_document_service import EnhancedDocumentService

# Enhanced vector store integration
from app.services.enhanced_vectorstore import (
    EnhancedVectorStore,
)
from app.services.supabase_file_service import SupabaseFileService

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize document processor and enhanced services
document_processor = DocumentProcessor()
enhanced_document_service = EnhancedDocumentService(document_processor)

# Enhanced vector store instance (lazily initialized with thread safety)
_enhanced_vector_store_instance: EnhancedVectorStore | None = None
_enhanced_vector_store_lock = threading.Lock()

# Storage service instance (lazily initialized with thread safety)
_storage_service_instance: SupabaseFileService | None = None
_storage_service_lock = threading.Lock()


def get_enhanced_vector_store() -> EnhancedVectorStore:
    """
    Dependency to provide enhanced vector store with thread-safe lazy initialization.
    Creates the instance on first access and reuses it for subsequent calls.
    Uses double-checked locking pattern to avoid race conditions.
    """
    global _enhanced_vector_store_instance

    # First check without lock (optimization for already initialized case)
    if _enhanced_vector_store_instance is not None:
        return _enhanced_vector_store_instance

    # Acquire lock for initialization
    with _enhanced_vector_store_lock:
        # Double-check inside lock in case another thread initialized it
        if _enhanced_vector_store_instance is None:
            _enhanced_vector_store_instance = EnhancedVectorStore()
        return _enhanced_vector_store_instance


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
            status_code=400,
            detail="Unsupported file format or no content could be extracted",
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

        # Process document with enhanced services
        try:
            # Use enhanced document service for processing
            enhanced_doc = await enhanced_document_service.process_document_enhanced(
                file, use_enhanced_models=True, preserve_structure=True
            )

            # Set the document ID and user ID for the enhanced document
            enhanced_doc.id = str(document_id)
            enhanced_doc.user_id = user_id
            enhanced_doc.storage_key = storage_key

            # Store with enhanced vector store
            chunk_count = await get_enhanced_vector_store().store_enhanced_document(
                enhanced_doc
            )

        except Exception as e:
            # Fallback to legacy processing if enhanced processing fails
            logger.warning(
                f"Enhanced processing failed, falling back to legacy: {str(e)}"
            )

            # Process the document and capture the results with legacy method
            processing_result = await _process_document_internal(file)

            # Convert to enhanced document for storage
            # Convert service types to schema types
            from app.models.schemas import DocumentChunk as BaseDocumentChunk
            from app.models.schemas import DocumentMetadata as BaseDocumentMetadata

            base_metadata = BaseDocumentMetadata(
                filename=processing_result.parsed_content.metadata.filename,
                file_type=processing_result.parsed_content.metadata.file_type,
                total_pages=processing_result.parsed_content.metadata.total_pages,
                total_chars=processing_result.parsed_content.metadata.total_chars,
                total_tokens=processing_result.parsed_content.metadata.total_tokens,
                sections=processing_result.parsed_content.metadata.sections,
            )

            base_chunks = [
                BaseDocumentChunk(
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

            enhanced_doc = convert_to_enhanced_document(
                base_metadata,
                base_chunks,
                status="completed",
                content=processing_result.parsed_content.text,
                processing_stats=processing_result.processing_stats,
            )
            enhanced_doc.id = str(document_id)
            enhanced_doc.user_id = user_id
            enhanced_doc.storage_key = storage_key

            # Store with enhanced vector store
            chunk_count = await get_enhanced_vector_store().store_enhanced_document(
                enhanced_doc
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
                document_to_update.updated_at = datetime.utcnow()  # type: ignore[assignment]
                db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update document status: {str(e)}")
            # Don't raise here as the main processing succeeded
        finally:
            db.close()

        # Create response document
        document = Document(
            id=int(document_id),  # Explicitly cast to int to match schema
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
    """List all documents for the current user."""
    from app.core.vectorstore import SessionLocal
    from app.models.database import Document as DBDocument

    user_id = current_user["id"]

    db = SessionLocal()
    try:
        db_documents = db.query(DBDocument).filter(DBDocument.user_id == user_id).all()

        documents = []
        for db_doc in db_documents:
            # Count chunks for this document
            chunk_count = len(db_doc.chunks) if db_doc.chunks else 0

            # Note: db_doc attributes are already the values, not Column objects
            document = Document(
                id=int(db_doc.id),  # Explicitly cast to int to match schema
                filename=str(db_doc.filename),
                user_id=int(db_doc.user_id),
                status=str(db_doc.status),
                storage_key=str(db_doc.storage_key) if db_doc.storage_key else None,
                created_at=db_doc.created_at,  # type: ignore[arg-type]
                chunk_count=chunk_count,
            )
            documents.append(document)

        return documents
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch documents: {str(e)}"
        ) from e
    finally:
        db.close()


@router.delete("/{document_id}")
async def delete_document(
    document_id: int, current_user: dict = Depends(get_current_user)
):
    """Delete a document and all its chunks."""
    from app.core.vectorstore import SessionLocal
    from app.models.database import Chunk
    from app.models.database import Document as DBDocument

    user_id = current_user["id"]

    db = SessionLocal()
    try:
        # Find the document and verify ownership
        document = (
            db.query(DBDocument)
            .filter(DBDocument.id == document_id, DBDocument.user_id == user_id)
            .first()
        )

        if not document:
            raise HTTPException(
                status_code=404,
                detail="Document not found or you don't have permission to delete it",
            )

        # Delete all chunks first (cascade)
        db.query(Chunk).filter(Chunk.document_id == document_id).delete()

        # Delete the document
        db.delete(document)
        db.commit()

        # TODO: Also delete from file storage if needed
        # if document.storage_key:
        #     storage_service = get_storage_service()
        #     await storage_service.delete_file(document.storage_key)

        return {"message": f"Document '{document.filename}' deleted successfully"}

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Failed to delete document: {str(e)}"
        ) from e
    finally:
        db.close()


# Enhanced search function removed (was citation-related)


@router.get("/chunks/{chunk_id}/related")
async def get_related_chunks(
    chunk_id: str,
    relation_types: str | None = None,  # comma-separated list
    max_distance: int = 2,
    current_user: dict = Depends(get_current_user),
) -> list[dict]:
    """
    Get chunks related to a specific chunk through hierarchical relationships.

    Args:
        chunk_id: ID of the source chunk
        relation_types: Optional comma-separated list of relationship types to include
        max_distance: Maximum relationship distance to traverse (default: 2)

    Returns:
        List of related enhanced chunks
    """
    try:
        # Parse relation types if provided
        relation_type_list = None
        if relation_types:
            relation_type_list = [rt.strip() for rt in relation_types.split(",")]

        related_chunks = await get_enhanced_vector_store().get_related_chunks(
            chunk_id=chunk_id,
            relation_types=relation_type_list,
            max_distance=max_distance,
        )

        # Convert to dict format for JSON response
        result = []
        for chunk in related_chunks:
            result.append(
                {
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "chunk_type": chunk.chunk_type,
                    "hierarchical_level": chunk.hierarchical_level,
                    "quality_score": chunk.quality_score,
                    "chunk_references": chunk.chunk_references,
                }
            )

        logger.info(f"Found {len(result)} related chunks for chunk {chunk_id}")
        return result

    except Exception as e:
        logger.error(f"Failed to get related chunks: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get related chunks: {str(e)}"
        ) from e


@router.post("/hierarchy/store")
async def store_document_hierarchy(
    hierarchy_data: dict,
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Store document hierarchy information for enhanced retrieval.

    Args:
        hierarchy_data: Dictionary containing hierarchy information

    Returns:
        Dictionary with hierarchy_id and status
    """
    try:
        from app.models.hierarchy_models import DocumentHierarchy

        # Create hierarchy from input data
        hierarchy = DocumentHierarchy.from_dict(hierarchy_data)

        # Store hierarchy
        hierarchy_id = await get_enhanced_vector_store().store_hierarchy(hierarchy)

        logger.info(
            f"Stored hierarchy {hierarchy_id} with {hierarchy.total_elements} elements"
        )

        return {
            "hierarchy_id": hierarchy_id,
            "total_elements": hierarchy.total_elements,
            "status": "stored",
        }

    except Exception as e:
        logger.error(f"Failed to store hierarchy: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to store hierarchy: {str(e)}"
        ) from e
