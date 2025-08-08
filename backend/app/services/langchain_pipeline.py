"""Document processing pipeline using modular services."""

import asyncio
import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.services.document_loaders import (
    BaseDocumentLoader,
    PDFDocumentLoader,
    TextDocumentLoader,
    WordDocumentLoader,
)
from app.services.document_transformers import DocumentTransformer
from app.services.hierarchical_chunker import HierarchicalChunker
from app.utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """Document processing pipeline using modular services."""

    def __init__(self, token_counter: TokenCounter | None = None):
        """Initialize the document processor with modular services.

        Args:
            token_counter: Optional token counter for precise token-based operations
        """
        self.logger = logging.getLogger(__name__)
        self.token_counter = token_counter or TokenCounter()

        # Initialize services
        self.transformer = DocumentTransformer()
        # Use HierarchicalChunker for tree-based chunking (default for legal documents)
        self.splitter = HierarchicalChunker(
            token_counter=self.token_counter,
            legal_specific=True,  # Optimized for legal documents
            chunk_size=600,  # Optimal token size for RAG
            chunk_overlap=100,  # Good context preservation
        )

        # Initialize loaders
        self.loaders: dict[str, BaseDocumentLoader] = {
            "pdf": PDFDocumentLoader(),
            "word": WordDocumentLoader(),
            "text": TextDocumentLoader(),
        }

        # File extension mappings
        self.extension_mappings = {
            ".pdf": "pdf",
            ".doc": "word",
            ".docx": "word",
            ".txt": "text",
            ".md": "text",
            ".rst": "text",
            ".py": "text",
            ".js": "text",
            ".html": "text",
            ".xml": "text",
            ".json": "text",
            ".csv": "text",
        }

    async def process_documents(
        self,
        file_paths: list[str | Path],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        remove_html: bool = True,
        remove_redundant: bool = True,
        clean_text: bool = True,
        merge_short_documents: bool = False,
        min_document_length: int = 100,
        splitting_strategy: str = "recursive",
        preserve_structure: bool = True,
        use_token_counting: bool = False,
        batch_size: int = 10,
    ) -> list[Document]:
        """Process multiple documents through the complete pipeline.

        Args:
            file_paths: List of file paths to process
            chunk_size: Target size for document chunks
            chunk_overlap: Overlap between consecutive chunks
            remove_html: Whether to remove HTML tags
            remove_redundant: Whether to remove redundant documents
            clean_text: Whether to apply text cleaning
            merge_short_documents: Whether to merge short documents
            min_document_length: Minimum length for document merging
            splitting_strategy: Strategy for text splitting
            preserve_structure: Whether to preserve document structure
            use_token_counting: Whether to use token-based counting
            batch_size: Number of files to process concurrently

        Returns:
            List of processed document chunks
        """
        self.logger.info(
            f"Starting document processing pipeline for {len(file_paths)} files"
        )

        # Step 1: Load documents
        documents = await self._load_documents_batch(file_paths, batch_size)

        if not documents:
            self.logger.warning("No documents were successfully loaded")
            return []

        self.logger.info(f"Loaded {len(documents)} documents")

        # Step 2: Transform documents
        transformed_documents = await self.transformer.transform_documents(
            documents=documents,
            remove_html=remove_html,
            remove_redundant=remove_redundant,
            clean_text=clean_text,
            merge_short_documents=merge_short_documents,
            min_document_length=min_document_length,
        )

        self.logger.info(f"Transformed to {len(transformed_documents)} documents")

        # Step 3: Split documents into chunks
        # Note: HierarchicalChunker uses its initialization parameters, not call-time parameters
        if hasattr(self.splitter, "chunk_document"):
            # Use HierarchicalChunker's specialized method
            chunks = []
            for doc in transformed_documents:
                doc_chunks = self.splitter.chunk_document(doc)
                chunks.extend(doc_chunks)
        else:
            # Fallback for standard LangChain splitters
            chunks = await self.splitter.split_documents(
                documents=transformed_documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strategy=splitting_strategy,
                use_token_counting=use_token_counting,
            )

        self.logger.info(f"Split into {len(chunks)} chunks")

        # Step 4: Analyze results
        if hasattr(self.splitter, "analyze_chunk_distribution"):
            chunk_stats = await self.splitter.analyze_chunk_distribution(chunks)
        else:
            # Manual chunk analysis for HierarchicalChunker
            chunk_lengths = [len(chunk.page_content) for chunk in chunks]
            chunk_stats = {
                "total_chunks": len(chunks),
                "avg_chunk_length": sum(chunk_lengths) / len(chunks) if chunks else 0,
                "min_chunk_length": min(chunk_lengths) if chunks else 0,
                "max_chunk_length": max(chunk_lengths) if chunks else 0,
                "chunking_strategy": "hierarchical_tree_based",
            }
        self.logger.info(f"Chunk distribution: {chunk_stats}")

        return chunks

    async def _load_documents_batch(
        self, file_paths: list[str | Path], batch_size: int
    ) -> list[Document]:
        """Load documents in batches to manage memory and concurrency."""
        all_documents = []

        # Process files in batches
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i : i + batch_size]
            self.logger.info(
                f"Processing batch {i // batch_size + 1}: {len(batch)} files"
            )

            # Load batch concurrently
            batch_tasks = [self._load_single_document(file_path) for file_path in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Collect successful results
            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Error in batch processing: {result}")
                elif isinstance(result, list):
                    all_documents.extend(result)

        return all_documents

    async def _load_single_document(self, file_path: str | Path) -> list[Document]:
        """Load a single document using the appropriate loader."""
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                self.logger.error(f"File not found: {file_path}")
                return []

            # Determine loader type
            loader_type = self._get_loader_type(file_path)
            if not loader_type:
                self.logger.warning(
                    f"No loader available for file type: {file_path.suffix}"
                )
                return []

            # Load document
            loader = self.loaders[loader_type]
            documents = await loader.load_document(file_path)

            self.logger.debug(f"Loaded {len(documents)} documents from {file_path}")
            return documents

        except Exception as e:
            self.logger.error(f"Error loading document {file_path}: {e}")
            return []

    def _get_loader_type(self, file_path: Path) -> str | None:
        """Determine the appropriate loader type for a file."""
        extension = file_path.suffix.lower()
        return self.extension_mappings.get(extension)

    async def process_single_file(
        self, file_path: str | Path, **kwargs
    ) -> list[Document]:
        """Process a single file through the complete pipeline."""
        return await self.process_documents([file_path], **kwargs)

    async def load_documents_only(
        self, file_paths: list[str | Path], batch_size: int = 10
    ) -> list[Document]:
        """Load documents without transformation or splitting."""
        return await self._load_documents_batch(file_paths, batch_size)

    async def transform_documents_only(
        self, documents: list[Document], **transform_kwargs
    ) -> list[Document]:
        """Apply transformations to documents without loading or splitting."""
        return await self.transformer.transform_documents(documents, **transform_kwargs)

    async def split_documents_only(
        self, documents: list[Document], **split_kwargs
    ) -> list[Document]:
        """Split documents without loading or transformation."""
        return await self.splitter.split_documents(documents, **split_kwargs)

    async def get_supported_file_types(self) -> dict[str, list[str]]:
        """Get information about supported file types."""
        return {
            loader_type: [
                ext for ext, lt in self.extension_mappings.items() if lt == loader_type
            ]
            for loader_type in self.loaders.keys()
        }

    async def analyze_documents(self, documents: list[Document]) -> dict[str, Any]:
        """Analyze document characteristics and provide processing recommendations."""
        if not documents:
            return {}

        # Basic statistics
        doc_lengths = [len(doc.page_content) for doc in documents]
        total_length = sum(doc_lengths)

        # Analyze content types
        content_analysis = {
            "total_documents": len(documents),
            "total_characters": total_length,
            "avg_document_length": total_length / len(documents),
            "min_document_length": min(doc_lengths),
            "max_document_length": max(doc_lengths),
            "document_sources": list(
                {doc.metadata.get("source", "unknown") for doc in documents}
            ),
        }

        # Recommend optimal chunk size
        optimal_chunk_size = await self.splitter.calculate_optimal_chunk_size(documents)
        content_analysis["recommended_chunk_size"] = optimal_chunk_size

        # Analyze structure
        structure_info = []
        for doc in documents[:5]:  # Sample first 5 documents
            structure = self.splitter._analyze_document_structure(doc.page_content)
            structure_info.append(structure)

        if structure_info:
            content_analysis["structure_analysis"] = {
                "avg_headings": sum(s["heading_count"] for s in structure_info)
                / len(structure_info),
                "avg_paragraphs": sum(s["paragraph_count"] for s in structure_info)
                / len(structure_info),
                "has_structure": any(
                    s["has_headings"] or s["has_paragraphs"] for s in structure_info
                ),
            }

        return content_analysis

    async def estimate_processing_time(
        self, file_paths: list[str | Path]
    ) -> dict[str, Any]:
        """Estimate processing time and resource requirements."""
        total_size = 0
        file_types = {}

        for file_path in file_paths:
            try:
                path = Path(file_path)
                if path.exists():
                    size = path.stat().st_size
                    total_size += size

                    ext = path.suffix.lower()
                    file_types[ext] = file_types.get(ext, 0) + 1
            except Exception:
                continue

        # Rough estimates based on file size and type
        base_time_per_mb = 2  # seconds per MB
        estimated_time = (total_size / (1024 * 1024)) * base_time_per_mb

        # Adjust for file types (PDFs are slower)
        pdf_count = file_types.get(".pdf", 0)
        if pdf_count > 0:
            estimated_time *= 1.5  # PDFs take longer

        return {
            "total_files": len(file_paths),
            "total_size_mb": total_size / (1024 * 1024),
            "file_types": file_types,
            "estimated_time_seconds": estimated_time,
            "estimated_time_minutes": estimated_time / 60,
        }

    async def validate_files(self, file_paths: list[str | Path]) -> dict[str, Any]:
        """Validate files before processing."""
        results = {
            "valid_files": [],
            "invalid_files": [],
            "unsupported_files": [],
            "missing_files": [],
        }

        for file_path in file_paths:
            try:
                path = Path(file_path)

                if not path.exists():
                    results["missing_files"].append(str(path))
                    continue

                loader_type = self._get_loader_type(path)
                if not loader_type:
                    results["unsupported_files"].append(str(path))
                    continue

                # Try to validate with the appropriate loader
                loader = self.loaders[loader_type]
                if hasattr(loader, "validate_file"):
                    is_valid = await loader.validate_file(path)
                    if is_valid:
                        results["valid_files"].append(str(path))
                    else:
                        results["invalid_files"].append(str(path))
                else:
                    # Assume valid if no validation method
                    results["valid_files"].append(str(path))

            except Exception as e:
                self.logger.error(f"Error validating file {file_path}: {e}")
                results["invalid_files"].append(str(file_path))

        return results

    def get_processing_config(self) -> dict:
        """Get current processing configuration."""
        return {
            "chunk_size": 1000,  # Default chunk size
            "chunk_overlap": 200,  # Default chunk overlap
            "max_tokens_per_chunk": 1024,  # Default max tokens per chunk
            "max_file_size_mb": 100,  # Default max file size in MB
            "supported_formats": list(self.extension_mappings.keys()),
            "langchain_enabled": True,
        }

    async def get_processing_stats(self) -> dict[str, Any]:
        """Get statistics about the processor and its services."""
        return {
            "supported_loaders": list(self.loaders.keys()),
            "supported_extensions": list(self.extension_mappings.keys()),
            "transformer_available": self.transformer is not None,
            "splitter_available": self.splitter is not None,
            "token_counter_available": self.token_counter is not None,
        }
