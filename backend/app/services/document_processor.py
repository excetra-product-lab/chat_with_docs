"""Document processing service that orchestrates parsing and chunking."""

import logging
from typing import Dict, List

from fastapi import HTTPException, UploadFile

from app.services.document_parser import DocumentParser, ParsedContent
from app.services.text_chunker import DocumentChunk, TextChunker

logger = logging.getLogger(__name__)


class ProcessingResult:
    """Container for document processing results."""

    def __init__(
        self,
        parsed_content: ParsedContent,
        chunks: List[DocumentChunk],
        processing_stats: Dict,
    ):
        self.parsed_content = parsed_content
        self.chunks = chunks
        self.processing_stats = processing_stats


class DocumentProcessor:
    """Main service for processing uploaded documents."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        min_chunk_size: int = 100,
    ):
        """
        Initialize the document processor.

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum size for a chunk to be considered valid
        """
        self.parser = DocumentParser()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )
        self.logger = logging.getLogger(__name__)

    async def process_document(self, file: UploadFile) -> ProcessingResult:
        """
        Process an uploaded document: parse and chunk it.

        Args:
            file: The uploaded file to process

        Returns:
            ProcessingResult: Complete processing results

        Raises:
            HTTPException: If processing fails
        """
        try:
            self.logger.info(f"Starting processing of document: {file.filename}")

            # Step 1: Parse the document
            self.logger.info("Parsing document...")
            parsed_content = await self.parser.parse_document(file)

            # Step 2: Chunk the parsed content
            self.logger.info("Chunking document...")
            chunks = self.chunker.chunk_document(parsed_content)

            # Step 3: Generate processing statistics
            processing_stats = self._generate_processing_stats(parsed_content, chunks)

            self.logger.info(
                f"Successfully processed document {file.filename}: " f"{len(chunks)} chunks created"
            )

            return ProcessingResult(
                parsed_content=parsed_content,
                chunks=chunks,
                processing_stats=processing_stats,
            )

        except HTTPException:
            # Re-raise HTTP exceptions (validation errors, etc.)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error processing document {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

    def _generate_processing_stats(
        self, parsed_content: ParsedContent, chunks: List[DocumentChunk]
    ) -> Dict:
        """Generate comprehensive processing statistics."""
        chunk_stats = self.chunker.get_chunk_summary(chunks)

        return {
            "document": {
                "filename": parsed_content.metadata.filename,
                "file_type": parsed_content.metadata.file_type,
                "total_pages": parsed_content.metadata.total_pages,
                "total_characters": parsed_content.metadata.total_chars,
                "sections_detected": len(parsed_content.metadata.sections),
            },
            "parsing": {
                "structured_content_items": len(parsed_content.structured_content),
                "page_texts_extracted": len(parsed_content.page_texts),
                "sections_found": parsed_content.metadata.sections,
            },
            "chunking": chunk_stats,
            "processing": {
                "success": True,
                "chunks_ready_for_embedding": len(chunks),
            },
        }

    def validate_processing_result(self, result: ProcessingResult) -> bool:
        """
        Validate that processing was successful and results are usable.

        Args:
            result: The processing result to validate

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check that we have parsed content
            if not result.parsed_content or not result.parsed_content.text.strip():
                self.logger.error("No text content extracted from document")
                return False

            # Check that we have chunks
            if not result.chunks:
                self.logger.error("No chunks created from document")
                return False

            # Check that chunks have required metadata
            for i, chunk in enumerate(result.chunks):
                if not chunk.text.strip():
                    self.logger.error(f"Chunk {i} has no text content")
                    return False

                if not chunk.document_filename:
                    self.logger.error(f"Chunk {i} missing document filename")
                    return False

            # Check processing stats
            if not result.processing_stats.get("processing", {}).get("success"):
                self.logger.error("Processing stats indicate failure")
                return False

            self.logger.info("Processing result validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Error validating processing result: {str(e)}")
            return False

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return list(self.parser.SUPPORTED_FORMATS.keys())

    def get_processing_config(self) -> Dict:
        """Get current processing configuration."""
        return {
            "chunk_size": self.chunker.chunk_size,
            "chunk_overlap": self.chunker.chunk_overlap,
            "min_chunk_size": self.chunker.min_chunk_size,
            "max_file_size_mb": self.parser.MAX_FILE_SIZE / 1024 / 1024,
            "supported_formats": self.get_supported_formats(),
        }
