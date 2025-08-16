"""Document transformation service for cleaning and processing documents."""

import logging
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentTransformer:
    """Service for transforming and cleaning documents."""

    def __init__(self):
        """Initialize the document transformer."""
        self.logger = logging.getLogger(__name__)

    async def transform_documents(
        self,
        documents: list[Document],
        remove_html: bool = True,
        remove_redundant: bool = True,
        clean_text: bool = True,
        merge_short_documents: bool = False,
        min_document_length: int = 100,
    ) -> list[Document]:
        """Apply a series of transformations to the documents.

        Args:
            documents: A list of documents to transform
            remove_html: Whether to remove HTML tags
            remove_redundant: Whether to remove redundant documents using embeddings
            clean_text: Whether to apply text cleaning operations
            merge_short_documents: Whether to merge documents shorter than min_document_length
            min_document_length: Minimum length for documents when merging

        Returns:
            A list of transformed documents
        """
        if not documents:
            return documents

        transformed_documents = documents.copy()

        # Apply text cleaning
        if clean_text:
            transformed_documents = await self._clean_document_text(
                transformed_documents
            )

        # HTML removal
        if remove_html:
            transformed_documents = await self._remove_html_tags(transformed_documents)

        # Merge short documents if requested
        if merge_short_documents:
            transformed_documents = await self._merge_short_documents(
                transformed_documents, min_document_length
            )

        # Redundancy removal (expensive operation, do last)
        if remove_redundant:
            transformed_documents = await self._remove_redundant_documents(
                transformed_documents
            )

        self.logger.info(
            f"Document transformation complete: {len(documents)} -> {len(transformed_documents)} documents"
        )

        return transformed_documents

    async def _remove_html_tags(self, documents: list[Document]) -> list[Document]:
        """Remove HTML tags from documents using appropriate method based on content type."""
        if not documents:
            return documents

        # Check if any documents are markdown-based or contain HTML
        markdown_docs = []
        html_docs = []
        text_docs = []

        for doc in documents:
            source = str(doc.metadata.get("source", "")).lower()
            has_md_extension = source.endswith((".md", ".markdown"))
            has_md_markers = any(
                marker in doc.page_content for marker in ["#", "**", "__", "```", "---"]
            )
            is_markdown = has_md_extension or has_md_markers

            # Check if document contains HTML tags
            has_html_tags = any(
                tag in doc.page_content.lower()
                for tag in [
                    "<html",
                    "<div",
                    "<p>",
                    "<span",
                    "<table",
                    "<script",
                    "<style",
                ]
            )

            # Debug logging
            self.logger.info(f"Document source: {source}")
            self.logger.info(f"Has .md extension: {has_md_extension}")
            self.logger.info(f"Has markdown markers: {has_md_markers}")
            self.logger.info(f"Classified as markdown: {is_markdown}")
            self.logger.info(f"Has HTML tags: {has_html_tags}")

            if is_markdown:
                markdown_docs.append(doc)
            elif has_html_tags:
                html_docs.append(doc)
            else:
                text_docs.append(doc)

        transformed_documents = []

        # Process markdown documents with markdown-aware cleaner
        if markdown_docs:
            try:
                from .markdown_html_cleaner import create_markdown_html_cleaner

                markdown_cleaner = create_markdown_html_cleaner()
                cleaned_markdown = await markdown_cleaner.transform_documents(
                    markdown_docs
                )
                transformed_documents.extend(cleaned_markdown)
                self.logger.info(
                    f"Applied markdown-aware HTML cleaning to {len(markdown_docs)} documents"
                )
            except ImportError:
                self.logger.warning(
                    "Markdown HTML cleaner not available, skipping HTML removal for markdown"
                )
                transformed_documents.extend(markdown_docs)

        # Process HTML documents with BeautifulSoup transformer
        if html_docs:
            try:
                # Import inside the try-block so we can gracefully skip if the
                # optional BeautifulSoup dependency is not available
                from langchain_community.document_transformers import (
                    BeautifulSoupTransformer,
                )

                self.logger.info(
                    f"Applying BeautifulSoupTransformer to {len(html_docs)} HTML documents"
                )
                bs_transformer = BeautifulSoupTransformer()
                html_cleaned = bs_transformer.transform_documents(
                    html_docs,
                    unwanted_tags=[
                        "script",
                        "style",
                        "nav",
                        "footer",
                        "header",
                    ],  # Removed "a" which was too aggressive
                    remove_lines=False,  # Changed to preserve content structure
                    remove_new_lines=False,  # Changed to preserve content structure
                )
                transformed_documents.extend(html_cleaned)

            except ImportError:
                # BeautifulSoup4 not installed – skip HTML removal for HTML docs
                self.logger.warning(
                    "BeautifulSoup4 not available – skipping HTML tag removal for HTML documents"
                )
                transformed_documents.extend(html_docs)

            except Exception as e:
                self.logger.error(f"Error during HTML removal: {str(e)}")
                transformed_documents.extend(html_docs)

        # Process plain text documents (no HTML cleaning needed)
        if text_docs:
            self.logger.info(
                f"Preserving {len(text_docs)} plain text documents without HTML processing"
            )
            transformed_documents.extend(text_docs)

        self.logger.info(f"HTML removal complete for {len(documents)} documents")
        return transformed_documents

    async def _remove_redundant_documents(
        self, documents: list[Document]
    ) -> list[Document]:
        """Remove redundant documents using embeddings-based similarity."""
        # DISABLED: LangChain API has changed and DocumentCompressorPipeline is unstable
        # For MVP purposes, we'll skip redundancy removal to avoid API compatibility issues
        self.logger.info(
            "Redundancy removal skipped - using all documents for better recall"
        )
        return documents

        # TODO: Re-implement with stable LangChain API in future version
        # The following code has API compatibility issues:
        """
        try:
            from langchain.retrievers.document_compressors import (
                DocumentCompressorPipeline,
                EmbeddingsFilter,
            )
            from langchain_openai import OpenAIEmbeddings

            self.logger.info("Applying EmbeddingsFilter to remove redundant documents")

            # Create a splitter for preprocessing
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator=". ")

            # Create embeddings and filter
            embeddings = OpenAIEmbeddings()
            redundant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.95)

            # Create pipeline
            pipeline: Any = DocumentCompressorPipeline(transformers=[splitter, redundant_filter])

            # Try async first, fallback to sync if not available
            try:
                if hasattr(pipeline, 'atransform_documents'):
                    transformed_seq = await pipeline.atransform_documents(documents)
                else:
                    # Fallback to sync method
                    import asyncio
                    transformed_seq = await asyncio.get_event_loop().run_in_executor(
                        None, pipeline.transform_documents, documents
                    )
                transformed_documents = list(transformed_seq)
            except AttributeError:
                # Final fallback to direct sync call
                transformed_documents = pipeline.transform_documents(documents)

            self.logger.info(
                f"Redundancy removal complete: {len(documents)} -> {len(transformed_documents)} documents"
            )
            return transformed_documents

        except Exception as e:
            # Most commonly triggered when OPENAI_API_KEY is not available
            self.logger.warning(f"Embeddings-based redundancy filter skipped due to error: {e}")
            return documents
        """

    async def _clean_document_text(self, documents: list[Document]) -> list[Document]:
        """Apply text cleaning operations to documents."""
        cleaned_documents = []

        for doc in documents:
            cleaned_content = self._clean_text_content(doc.page_content)

            # Create new document with cleaned content
            cleaned_doc = Document(
                page_content=cleaned_content, metadata=doc.metadata.copy()
            )

            # Add cleaning metadata
            cleaned_doc.metadata["text_cleaned"] = True
            cleaned_doc.metadata["original_length"] = len(doc.page_content)
            cleaned_doc.metadata["cleaned_length"] = len(cleaned_content)

            cleaned_documents.append(cleaned_doc)

        self.logger.info(f"Text cleaning complete for {len(documents)} documents")
        return cleaned_documents

    def _clean_text_content(self, text: str) -> str:
        """Clean text content by removing unwanted characters and normalizing."""
        if not text:
            return text

        # Remove excessive whitespace
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            # Strip whitespace from each line
            cleaned_line = line.strip()

            # Skip empty lines that are just whitespace
            if cleaned_line:
                # Replace multiple spaces with single space
                cleaned_line = " ".join(cleaned_line.split())
                cleaned_lines.append(cleaned_line)
            elif (
                cleaned_lines and cleaned_lines[-1]
            ):  # Keep single empty lines between content
                cleaned_lines.append("")

        # Join lines back together
        cleaned_text = "\n".join(cleaned_lines)

        # Remove excessive newlines (more than 2 consecutive)
        while "\n\n\n" in cleaned_text:
            cleaned_text = cleaned_text.replace("\n\n\n", "\n\n")

        # Remove common unwanted characters
        unwanted_chars = [
            "\r",
            "\x00",
            "\ufffd",
        ]  # carriage return, null, replacement char
        for char in unwanted_chars:
            cleaned_text = cleaned_text.replace(char, "")

        return cleaned_text.strip()

    async def _merge_short_documents(
        self, documents: list[Document], min_length: int = 100
    ) -> list[Document]:
        """Merge documents that are shorter than the minimum length."""
        if not documents:
            return documents

        merged_documents = []
        current_merge_group: list[Document] = []
        current_merge_length = 0

        for doc in documents:
            content_length = len(doc.page_content)

            # If document is long enough, process any pending merge group and add this doc
            if content_length >= min_length:
                # Process any pending merge group
                if current_merge_group:
                    merged_doc = self._create_merged_document(current_merge_group)
                    merged_documents.append(merged_doc)
                    current_merge_group = []
                    current_merge_length = 0

                # Add the long document as-is
                merged_documents.append(doc)

            else:
                # Add to merge group
                current_merge_group.append(doc)
                current_merge_length += content_length

                # If merge group is now long enough, merge and add
                if current_merge_length >= min_length:
                    merged_doc = self._create_merged_document(current_merge_group)
                    merged_documents.append(merged_doc)
                    current_merge_group = []
                    current_merge_length = 0

        # Handle any remaining documents in merge group
        if current_merge_group:
            merged_doc = self._create_merged_document(current_merge_group)
            merged_documents.append(merged_doc)

        self.logger.info(
            f"Document merging complete: {len(documents)} -> {len(merged_documents)} documents"
        )
        return merged_documents

    def _create_merged_document(self, documents: list[Document]) -> Document:
        """Create a single document by merging multiple documents."""
        if len(documents) == 1:
            return documents[0]

        # Combine content
        content_parts = [doc.page_content for doc in documents]
        merged_content = "\n\n".join(content_parts)

        # Combine metadata
        merged_metadata = documents[0].metadata.copy()
        merged_metadata.update(
            {
                "merged_from_count": len(documents),
                "merged_sources": [
                    doc.metadata.get("source", "unknown") for doc in documents
                ],
                "merged_pages": [doc.metadata.get("page", 0) for doc in documents],
                "original_lengths": [len(doc.page_content) for doc in documents],
                "merged_length": len(merged_content),
            }
        )

        # If all documents have the same source, keep it; otherwise mark as merged
        sources = {doc.metadata.get("source", "unknown") for doc in documents}
        if len(sources) == 1:
            merged_metadata["source"] = list(sources)[0]
        else:
            merged_metadata["source"] = "merged_documents"

        return Document(page_content=merged_content, metadata=merged_metadata)

    async def clean_document_metadata(
        self, documents: list[Document]
    ) -> list[Document]:
        """Clean and normalize document metadata."""
        cleaned_documents = []

        for doc in documents:
            cleaned_metadata = self._clean_metadata_dict(doc.metadata)

            cleaned_doc = Document(
                page_content=doc.page_content, metadata=cleaned_metadata
            )

            cleaned_documents.append(cleaned_doc)

        self.logger.info(f"Metadata cleaning complete for {len(documents)} documents")
        return cleaned_documents

    def _clean_metadata_dict(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Clean and normalize a metadata dictionary."""
        cleaned: dict[str, Any] = {}

        for key, value in metadata.items():
            # Normalize key names
            clean_key = key.strip().lower().replace(" ", "_")

            # Clean values
            if isinstance(value, str):
                clean_value = value.strip()
                # Remove empty strings
                if clean_value:
                    cleaned[clean_key] = clean_value
            elif isinstance(value, int | float | bool):
                cleaned[clean_key] = value
            elif isinstance(value, list | dict):
                # Keep complex types as-is for now
                cleaned[clean_key] = value
            elif value is not None:
                # Convert other types to string
                cleaned[clean_key] = str(value).strip()

        return cleaned

    async def split_large_documents(
        self, documents: list[Document], max_length: int = 10000, overlap: int = 200
    ) -> list[Document]:
        """Split documents that exceed the maximum length."""
        split_documents = []

        for doc in documents:
            if len(doc.page_content) <= max_length:
                split_documents.append(doc)
            else:
                # Split the document
                parts = self._split_document_content(
                    doc.page_content, max_length, overlap
                )

                for i, part in enumerate(parts):
                    split_metadata = doc.metadata.copy()
                    split_metadata.update(
                        {
                            "split_index": i,
                            "split_total": len(parts),
                            "original_length": len(doc.page_content),
                            "split_length": len(part),
                        }
                    )

                    split_doc = Document(page_content=part, metadata=split_metadata)
                    split_documents.append(split_doc)

        self.logger.info(
            f"Document splitting complete: {len(documents)} -> {len(split_documents)} documents"
        )
        return split_documents

    def _split_document_content(
        self, content: str, max_length: int, overlap: int
    ) -> list[str]:
        """Split content into chunks with overlap."""
        if len(content) <= max_length:
            return [content]

        parts = []
        start = 0

        while start < len(content):
            end = start + max_length

            if end >= len(content):
                # Last chunk
                parts.append(content[start:])
                break

            # Try to find a good break point (sentence or paragraph boundary)
            chunk = content[start:end]

            # Look for sentence boundaries near the end
            for boundary in [". ", ".\n", "\n\n"]:
                last_boundary = chunk.rfind(boundary)
                if last_boundary > max_length * 0.7:  # Don't break too early
                    end = start + last_boundary + len(boundary)
                    break

            parts.append(content[start:end])
            start = max(start + 1, end - overlap)  # Ensure progress with overlap

        return parts
