"""
Enhanced document service that integrates Langchain models with the existing processing pipeline.

This service demonstrates how to use the enhanced Langchain models while maintaining
compatibility with the existing document processing workflow.
"""

import logging

from fastapi import UploadFile

from app.models.langchain_models import (
    EnhancedDocument,
    convert_to_enhanced_document,
    integrate_with_langchain_pipeline,
)
from app.services.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class EnhancedDocumentService:
    """
    Enhanced document service that bridges existing processing with Langchain models.

    This service provides enhanced functionality while maintaining backward compatibility
    with the existing document processing pipeline.
    """

    def __init__(self, document_processor: DocumentProcessor | None = None):
        """
        Initialize the enhanced document service.

        Args:
            document_processor: Optional existing document processor instance
        """
        self.document_processor = document_processor or DocumentProcessor(
            use_langchain=True
        )
        self.langchain_processor = self.document_processor.get_langchain_processor()
        self.logger = logging.getLogger(__name__)

    async def process_document_enhanced(
        self,
        file: UploadFile,
        use_enhanced_models: bool = True,
        preserve_structure: bool = True,
    ) -> EnhancedDocument:
        """
        Process a document and return enhanced Langchain-compatible models.

        Args:
            file: The uploaded file to process
            use_enhanced_models: Whether to use enhanced models or convert from base
            preserve_structure: Whether to preserve document structure

        Returns:
            EnhancedDocument: Processed document with enhanced models
        """
        self.logger.info(f"Processing document with enhanced models: {file.filename}")

        if use_enhanced_models and self.langchain_processor:
            # Direct Langchain processing with enhanced models
            return await self._process_with_enhanced_langchain(file, preserve_structure)
        else:
            # Use existing processor and convert to enhanced models
            return await self._process_with_conversion(file)

    async def _process_with_enhanced_langchain(
        self, file: UploadFile, preserve_structure: bool
    ) -> EnhancedDocument:
        """Process document directly with Langchain and create enhanced models."""
        import tempfile
        from pathlib import Path

        # Create temporary file for Langchain processing
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename or "").suffix
        ) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()

            try:
                # Determine file type and appropriate HTML handling
                file_suffix = Path(file.filename or "").suffix.lower()
                is_markdown = file_suffix in [".md", ".markdown"]

                # Process with Langchain pipeline
                # Now using improved HTML processing that's markdown-aware
                langchain_docs = await self.langchain_processor.process_single_file(
                    temp_file.name,
                    preserve_structure=preserve_structure,
                    remove_html=True,  # Use smart HTML processing (now markdown-aware)
                    clean_text=True,  # Apply text cleaning
                )

                # Create enhanced document from Langchain output
                enhanced_doc = integrate_with_langchain_pipeline(
                    langchain_docs,
                    file.filename or "unknown",
                    status="completed",
                    langchain_source=True,
                )

                # Generate content hash and additional metadata
                if enhanced_doc.content:
                    enhanced_doc.metadata.generate_content_hash(enhanced_doc.content)

                # Update chunk hashes and token counts
                for chunk in enhanced_doc.chunks:
                    chunk.generate_chunk_hash()
                    if hasattr(self.langchain_processor, "token_counter"):
                        chunk.token_count = (
                            self.langchain_processor.token_counter.count_tokens(
                                chunk.text
                            )
                        )

                # Calculate processing stats
                enhanced_doc.calculate_processing_stats()

                self.logger.info(
                    f"Enhanced Langchain processing completed: {len(enhanced_doc.chunks)} chunks created"
                )

                return enhanced_doc

            finally:
                # Clean up temp file
                Path(temp_file.name).unlink(missing_ok=True)

    async def _process_with_conversion(self, file: UploadFile) -> EnhancedDocument:
        """Process with existing pipeline and convert to enhanced models."""
        # Use existing document processor
        processing_result = await self.document_processor.process_document(file)

        # Convert to enhanced models
        enhanced_doc = convert_to_enhanced_document(
            processing_result.parsed_content.metadata,
            processing_result.chunks,
            status="completed",
            content=processing_result.parsed_content.text,
            processing_stats=processing_result.processing_stats,
        )

        # Generate additional metadata
        enhanced_doc.metadata.generate_content_hash(enhanced_doc.content or "")
        enhanced_doc.metadata.langchain_source = any(
            content.get("langchain_source", False)
            for content in processing_result.parsed_content.structured_content
        )

        # Update chunk information
        for chunk in enhanced_doc.chunks:
            chunk.generate_chunk_hash()
            chunk.langchain_source = enhanced_doc.metadata.langchain_source

        enhanced_doc.calculate_processing_stats()

        self.logger.info(
            f"Conversion to enhanced models completed: {len(enhanced_doc.chunks)} chunks"
        )

        return enhanced_doc

    def validate_enhanced_document(self, document: EnhancedDocument) -> bool:
        """
        Validate an enhanced document for completeness and consistency.

        Args:
            document: The enhanced document to validate

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Basic validation
            if not document.filename or not document.metadata:
                self.logger.error("Document missing filename or metadata")
                return False

            if not document.chunks:
                self.logger.error("Document has no chunks")
                return False

            # Validate metadata consistency
            calculated_chars = sum(chunk.char_count for chunk in document.chunks)
            if (
                abs(document.metadata.total_chars - calculated_chars) > 100
            ):  # Allow small discrepancy
                self.logger.warning(
                    f"Character count mismatch: metadata={document.metadata.total_chars}, "
                    f"calculated={calculated_chars}"
                )

            # Validate chunk consistency
            for i, chunk in enumerate(document.chunks):
                if chunk.document_filename != document.filename:
                    self.logger.error(f"Chunk {i} filename mismatch")
                    return False

                if chunk.char_count != len(chunk.text):
                    self.logger.warning(f"Chunk {i} character count mismatch")

                if chunk.end_char <= chunk.start_char:
                    self.logger.error(f"Chunk {i} invalid character positions")
                    return False

            # Validate chunk ordering
            for i in range(1, len(document.chunks)):
                if document.chunks[i].chunk_index <= document.chunks[i - 1].chunk_index:
                    self.logger.warning("Chunks may not be in correct order")

            self.logger.info("Enhanced document validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Error validating enhanced document: {e}")
            return False

    def get_document_statistics(self, document: EnhancedDocument) -> dict:
        """Get comprehensive statistics for an enhanced document."""
        stats = document.calculate_processing_stats()

        # Add validation statistics
        validation_stats = {
            "valid_chunks": sum(1 for chunk in document.chunks if chunk.text.strip()),
            "chunks_with_sections": sum(
                1 for chunk in document.chunks if chunk.section_title
            ),
            "chunks_with_pages": sum(
                1 for chunk in document.chunks if chunk.page_number
            ),
            "unique_chunk_types": len(
                set(chunk.chunk_type for chunk in document.chunks)
            ),
            "has_content_hash": document.metadata.document_hash is not None,
            "processing_timestamp": document.metadata.processing_timestamp.isoformat(),
        }

        stats["validation"] = validation_stats
        return stats

    # create_citations_from_chunks method removed (was citation-related)

    def export_to_langchain_documents(self, document: EnhancedDocument) -> list:
        """Export enhanced document as Langchain Documents for downstream processing."""
        return document.to_langchain_documents()

    def get_processing_config(self) -> dict:
        """Get processing configuration including enhanced model options."""
        config = self.document_processor.get_processing_config()

        config.update(
            {
                "enhanced_models_available": True,
                "enhanced_validation": True,
                "content_hashing": True,
                "structure_preservation": True,
                "citation_enhancement": True,
            }
        )

        return config


# Utility functions for integration


async def process_document_with_enhanced_models(
    file: UploadFile, document_processor: DocumentProcessor | None = None
) -> EnhancedDocument:
    """
    Convenience function to process a document with enhanced models.

    Args:
        file: The uploaded file to process
        document_processor: Optional existing document processor

    Returns:
        EnhancedDocument: Processed document with enhanced models
    """
    service = EnhancedDocumentService(document_processor)
    return await service.process_document_enhanced(file)


def create_enhanced_service() -> EnhancedDocumentService:
    """Create a new enhanced document service with default configuration."""
    return EnhancedDocumentService()
