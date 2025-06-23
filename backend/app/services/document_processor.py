"""Document processing service that orchestrates parsing and chunking."""

import logging

from fastapi import HTTPException, UploadFile

from app.services.document_parser import DocumentParser, ParsedContent
from app.services.langchain_document_processor import LangchainDocumentProcessor
from app.services.text_chunker import DocumentChunk, TextChunker

logger = logging.getLogger(__name__)


class ProcessingResult:
    """Container for document processing results."""

    def __init__(
        self,
        parsed_content: ParsedContent,
        chunks: list[DocumentChunk],
        processing_stats: dict,
    ):
        self.parsed_content = parsed_content
        self.chunks = chunks
        self.processing_stats = processing_stats


class DocumentProcessor:
    """Main service for processing uploaded documents with Langchain integration."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        min_chunk_size: int = 100,
        use_langchain: bool = True,
    ):
        """
        Initialize the document processor.

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum size for a chunk to be considered valid
            use_langchain: Whether to use Langchain processors when available
        """
        self.parser = DocumentParser()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )
        self.use_langchain = use_langchain
        self.langchain_processor = (
            LangchainDocumentProcessor() if use_langchain else None
        )
        self.logger = logging.getLogger(__name__)

    async def process_document(
        self, file: UploadFile, prefer_langchain: bool | None = None
    ) -> ProcessingResult:
        """
        Process a document using standard or Langchain processors.

        Args:
            file: The uploaded file to process
            prefer_langchain: Override the default Langchain usage behavior

        Returns:
            ProcessingResult: Complete processing results

        Raises:
            HTTPException: If processing fails
        """
        try:
            self.logger.info(f"Starting processing of document: {file.filename}")

            # Step 0: Validate file before processing
            self.parser._validate_file(file)
            await file.seek(0)  # Reset file pointer after validation

            # Determine processing method
            use_langchain_for_this = (
                prefer_langchain if prefer_langchain is not None else self.use_langchain
            )

            # Step 1: Parse the document
            if use_langchain_for_this and self.langchain_processor:
                self.logger.info("Parsing document with Langchain...")
                try:
                    parsed_content = (
                        await self.langchain_processor.process_document_with_langchain(
                            file
                        )
                    )
                    self.logger.info("Successfully parsed document with Langchain")
                except Exception as e:
                    self.logger.warning(
                        f"Langchain parsing failed, falling back to standard parser: {str(e)}"
                    )
                    # Reset file pointer and try standard parsing
                    await file.seek(0)
                    parsed_content = await self.parser.parse_document(file)
            else:
                self.logger.info("Parsing document with standard parser...")
                parsed_content = await self.parser.parse_document(file)

            # Step 2: Chunk the parsed content
            self.logger.info("Chunking document...")
            chunks = self.chunker.chunk_document(parsed_content)

            # Step 3: Generate processing statistics
            processing_stats = self._generate_processing_stats(parsed_content, chunks)

            # Add Langchain processing information to stats
            processing_stats["processing"]["langchain_used"] = (
                use_langchain_for_this
                and any(
                    content.get("langchain_source", False)
                    for content in parsed_content.structured_content
                )
            )

            self.logger.info(
                f"Successfully processed document {file.filename}: "
                f"{len(chunks)} chunks created "
                f"(Langchain: {processing_stats['processing']['langchain_used']})"
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
            self.logger.error(
                f"Unexpected error processing document {file.filename}: {str(e)}"
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to process document: {str(e)}"
            ) from e

    async def process_with_langchain_only(self, file: UploadFile) -> ProcessingResult:
        """
        Process a document using only Langchain processors.

        Args:
            file: The uploaded file to process

        Returns:
            ProcessingResult: Complete processing results

        Raises:
            HTTPException: If Langchain processing fails or is not available
        """
        if not self.langchain_processor:
            raise HTTPException(
                status_code=500, detail="Langchain processor not available"
            )

        return await self.process_document(file, prefer_langchain=True)

    def _generate_processing_stats(
        self, parsed_content: ParsedContent, chunks: list[DocumentChunk]
    ) -> dict:
        """Generate comprehensive processing statistics."""
        chunk_stats = self.chunker.get_chunk_summary(chunks)

        # Check if Langchain was used
        langchain_used = any(
            content.get("langchain_source", False)
            for content in parsed_content.structured_content
        )

        return {
            "document": {
                "filename": parsed_content.metadata.filename,
                "file_type": parsed_content.metadata.file_type,
                "total_pages": parsed_content.metadata.total_pages,
                "total_characters": parsed_content.metadata.total_chars,
                "total_tokens": parsed_content.metadata.total_tokens,
                "sections_detected": len(parsed_content.metadata.sections),
            },
            "parsing": {
                "structured_content_items": len(parsed_content.structured_content),
                "page_texts_extracted": len(parsed_content.page_texts),
                "sections_found": parsed_content.metadata.sections,
                "langchain_processor_used": langchain_used,
            },
            "chunking": chunk_stats,
            "processing": {
                "success": True,
                "chunks_ready_for_embedding": len(chunks),
                "langchain_used": langchain_used,
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

    def get_supported_formats(self) -> list[str]:
        """Get list of supported file formats."""
        return list(self.parser.SUPPORTED_FORMATS.keys())

    def get_processing_config(self) -> dict:
        """Get current processing configuration."""
        config = {
            "chunk_size": self.chunker.chunk_size,
            "chunk_overlap": self.chunker.chunk_overlap,
            "min_chunk_size": self.chunker.min_chunk_size,
            "max_file_size_mb": self.parser.MAX_FILE_SIZE / 1024 / 1024,
            "supported_formats": self.get_supported_formats(),
            "langchain_enabled": self.use_langchain,
        }

        # Add Langchain-specific config if available
        if self.langchain_processor:
            langchain_config = self.langchain_processor.get_processing_config()
            config["langchain_config"] = langchain_config

        return config

    def set_langchain_usage(self, use_langchain: bool) -> None:
        """
        Enable or disable Langchain usage.

        Args:
            use_langchain: Whether to use Langchain processors
        """
        self.use_langchain = use_langchain
        if use_langchain and self.langchain_processor is None:
            self.langchain_processor = LangchainDocumentProcessor()

        self.logger.info(
            f"Langchain usage {'enabled' if use_langchain else 'disabled'}"
        )

    def get_langchain_processor(self) -> LangchainDocumentProcessor | None:
        """Get the Langchain processor instance if available."""
        return self.langchain_processor
