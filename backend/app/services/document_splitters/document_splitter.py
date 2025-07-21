"""Document splitting service for intelligent text chunking."""

import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from app.utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class DocumentSplitter:
    """Service for splitting documents into optimal chunks."""

    def __init__(self, token_counter: Optional[TokenCounter] = None):
        """Initialize the document splitter.

        Args:
            token_counter: Optional token counter for precise token-based splitting
        """
        self.logger = logging.getLogger(__name__)
        self.token_counter = token_counter or TokenCounter()

    async def split_documents(
        self,
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: str = "recursive",
        preserve_structure: bool = True,
        use_token_counting: bool = False,
    ) -> List[Document]:
        """Split documents into chunks using the specified strategy.

        Args:
            documents: List of documents to split
            chunk_size: Target size for each chunk (characters or tokens)
            chunk_overlap: Overlap between consecutive chunks
            strategy: Splitting strategy ('recursive', 'character', 'token', 'semantic')
            preserve_structure: Whether to preserve document structure (headings, etc.)
            use_token_counting: Whether to use token-based counting instead of characters

        Returns:
            List of document chunks
        """
        if not documents:
            return documents

        all_chunks = []

        for doc in documents:
            if preserve_structure:
                chunks = await self._split_with_structure_preservation(
                    doc, chunk_size, chunk_overlap, strategy, use_token_counting
                )
            else:
                chunks = await self._split_document_basic(
                    doc, chunk_size, chunk_overlap, strategy, use_token_counting
                )

            all_chunks.extend(chunks)

        self.logger.info(
            f"Document splitting complete: {len(documents)} documents -> {len(all_chunks)} chunks"
        )
        return all_chunks

    async def _split_with_structure_preservation(
        self,
        document: Document,
        chunk_size: int,
        chunk_overlap: int,
        strategy: str,
        use_token_counting: bool,
    ) -> List[Document]:
        """Split document while preserving structural elements."""
        content = document.page_content

        # Detect structural elements
        structure_info = self._analyze_document_structure(content)

        if structure_info["has_headings"]:
            return await self._split_by_headings(
                document, chunk_size, chunk_overlap, strategy, use_token_counting
            )
        elif structure_info["has_paragraphs"]:
            return await self._split_by_paragraphs(
                document, chunk_size, chunk_overlap, strategy, use_token_counting
            )
        else:
            return await self._split_document_basic(
                document, chunk_size, chunk_overlap, strategy, use_token_counting
            )

    def _analyze_document_structure(self, content: str) -> Dict[str, Any]:
        """Analyze the structure of document content."""
        # Check for markdown-style headings
        heading_pattern = r"^#{1,6}\s+.+$"
        headings = re.findall(heading_pattern, content, re.MULTILINE)

        # Check for paragraph breaks
        paragraphs = content.split("\n\n")
        meaningful_paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 50]

        # Check for list items
        list_pattern = r"^\s*[-*+]\s+.+$|^\s*\d+\.\s+.+$"
        list_items = re.findall(list_pattern, content, re.MULTILINE)

        return {
            "has_headings": len(headings) > 0,
            "heading_count": len(headings),
            "has_paragraphs": len(meaningful_paragraphs) > 2,
            "paragraph_count": len(meaningful_paragraphs),
            "has_lists": len(list_items) > 0,
            "list_item_count": len(list_items),
            "avg_paragraph_length": sum(len(p) for p in meaningful_paragraphs)
            / max(1, len(meaningful_paragraphs)),
        }

    async def _split_by_headings(
        self,
        document: Document,
        chunk_size: int,
        chunk_overlap: int,
        strategy: str,
        use_token_counting: bool,
    ) -> List[Document]:
        """Split document by heading structure."""
        content = document.page_content

        # Find all headings with their positions
        heading_pattern = r"^(#{1,6})\s+(.+)$"
        headings = []

        for match in re.finditer(heading_pattern, content, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            start_pos = match.start()
            headings.append(
                {
                    "level": level,
                    "title": title,
                    "start_pos": start_pos,
                    "line_start": content[:start_pos].count("\n"),
                }
            )

        if not headings:
            return await self._split_document_basic(
                document, chunk_size, chunk_overlap, strategy, use_token_counting
            )

        # Create sections based on headings
        sections = []
        for i, heading in enumerate(headings):
            start_pos = heading["start_pos"]
            end_pos = headings[i + 1]["start_pos"] if i + 1 < len(headings) else len(content)

            section_content = content[start_pos:end_pos].strip()
            if section_content:
                sections.append(
                    {
                        "content": section_content,
                        "heading": heading,
                        "start_pos": start_pos,
                        "end_pos": end_pos,
                    }
                )

        # Split each section if it's too large
        chunks = []
        for section in sections:
            section_length = self._get_content_length(section["content"], use_token_counting)

            if section_length <= chunk_size:
                # Section fits in one chunk
                chunk_metadata = document.metadata.copy()
                chunk_metadata.update(
                    {
                        "heading_level": section["heading"]["level"],
                        "heading_title": section["heading"]["title"],
                        "section_start": section["start_pos"],
                        "section_end": section["end_pos"],
                        "chunk_type": "heading_section",
                    }
                )

                chunk = Document(page_content=section["content"], metadata=chunk_metadata)
                chunks.append(chunk)
            else:
                # Split large section
                section_doc = Document(
                    page_content=section["content"], metadata=document.metadata.copy()
                )

                section_chunks = await self._split_document_basic(
                    section_doc, chunk_size, chunk_overlap, strategy, use_token_counting
                )

                # Add heading context to each chunk
                for j, chunk in enumerate(section_chunks):
                    chunk.metadata.update(
                        {
                            "heading_level": section["heading"]["level"],
                            "heading_title": section["heading"]["title"],
                            "section_start": section["start_pos"],
                            "section_end": section["end_pos"],
                            "chunk_type": "heading_section_split",
                            "section_chunk_index": j,
                            "section_chunk_total": len(section_chunks),
                        }
                    )

                chunks.extend(section_chunks)

        return chunks

    async def _split_by_paragraphs(
        self,
        document: Document,
        chunk_size: int,
        chunk_overlap: int,
        strategy: str,
        use_token_counting: bool,
    ) -> List[Document]:
        """Split document by paragraph boundaries."""
        content = document.page_content
        paragraphs = content.split("\n\n")

        # Filter out empty paragraphs
        meaningful_paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk_parts = []
        current_chunk_length = 0

        for paragraph in meaningful_paragraphs:
            paragraph_length = self._get_content_length(paragraph, use_token_counting)

            # If adding this paragraph would exceed chunk size
            if current_chunk_parts and (current_chunk_length + paragraph_length > chunk_size):
                # Create chunk from current parts
                chunk_content = "\n\n".join(current_chunk_parts)
                chunk_metadata = document.metadata.copy()
                chunk_metadata.update(
                    {
                        "chunk_type": "paragraph_based",
                        "paragraph_count": len(current_chunk_parts),
                        "chunk_length": current_chunk_length,
                    }
                )

                chunk = Document(page_content=chunk_content, metadata=chunk_metadata)
                chunks.append(chunk)

                # Handle overlap by keeping last paragraph if it fits in overlap
                if chunk_overlap > 0 and current_chunk_parts:
                    last_paragraph = current_chunk_parts[-1]
                    last_paragraph_length = self._get_content_length(
                        last_paragraph, use_token_counting
                    )

                    if last_paragraph_length <= chunk_overlap:
                        current_chunk_parts = [last_paragraph]
                        current_chunk_length = last_paragraph_length
                    else:
                        current_chunk_parts = []
                        current_chunk_length = 0
                else:
                    current_chunk_parts = []
                    current_chunk_length = 0

            # Add current paragraph
            current_chunk_parts.append(paragraph)
            current_chunk_length += paragraph_length

            # If single paragraph is too large, split it further
            if paragraph_length > chunk_size:
                # Remove the large paragraph and process it separately
                current_chunk_parts.pop()
                current_chunk_length -= paragraph_length

                # Create chunk from parts before large paragraph
                if current_chunk_parts:
                    chunk_content = "\n\n".join(current_chunk_parts)
                    chunk_metadata = document.metadata.copy()
                    chunk_metadata.update(
                        {
                            "chunk_type": "paragraph_based",
                            "paragraph_count": len(current_chunk_parts),
                            "chunk_length": current_chunk_length,
                        }
                    )

                    chunk = Document(page_content=chunk_content, metadata=chunk_metadata)
                    chunks.append(chunk)

                # Split the large paragraph
                large_paragraph_doc = Document(
                    page_content=paragraph, metadata=document.metadata.copy()
                )

                paragraph_chunks = await self._split_document_basic(
                    large_paragraph_doc, chunk_size, chunk_overlap, strategy, use_token_counting
                )

                for chunk in paragraph_chunks:
                    chunk.metadata["chunk_type"] = "large_paragraph_split"

                chunks.extend(paragraph_chunks)

                # Reset for next paragraphs
                current_chunk_parts = []
                current_chunk_length = 0

        # Handle remaining paragraphs
        if current_chunk_parts:
            chunk_content = "\n\n".join(current_chunk_parts)
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_type": "paragraph_based",
                    "paragraph_count": len(current_chunk_parts),
                    "chunk_length": current_chunk_length,
                }
            )

            chunk = Document(page_content=chunk_content, metadata=chunk_metadata)
            chunks.append(chunk)

        return chunks

    async def _split_document_basic(
        self,
        document: Document,
        chunk_size: int,
        chunk_overlap: int,
        strategy: str,
        use_token_counting: bool,
    ) -> List[Document]:
        """Split document using basic text splitting strategies."""
        content = document.page_content

        if use_token_counting:
            return await self._split_by_tokens(document, chunk_size, chunk_overlap)

        if strategy == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        elif strategy == "character":
            splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n\n",
            )
        elif strategy == "semantic":
            return await self._split_semantically(document, chunk_size, chunk_overlap)
        else:
            # Default to recursive
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        # Split the document
        texts = splitter.split_text(content)

        # Create Document objects with metadata
        chunks = []
        for i, text in enumerate(texts):
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_index": i,
                    "chunk_total": len(texts),
                    "chunk_strategy": strategy,
                    "chunk_size": len(text),
                    "chunk_type": "basic_split",
                }
            )

            chunk = Document(page_content=text, metadata=chunk_metadata)
            chunks.append(chunk)

        return chunks

    async def _split_by_tokens(
        self,
        document: Document,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[Document]:
        """Split document by token count."""
        try:
            splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                model_name="gpt-4.1",
                chunk_overlap=chunk_overlap,
            )

            texts = splitter.split_text(document.page_content)

            chunks = []
            for i, text in enumerate(texts):
                token_count = await self.token_counter.count_tokens(text)

                chunk_metadata = document.metadata.copy()
                chunk_metadata.update(
                    {
                        "chunk_index": i,
                        "chunk_total": len(texts),
                        "chunk_strategy": "token",
                        "chunk_tokens": token_count,
                        "chunk_size": len(text),
                        "chunk_type": "token_split",
                    }
                )

                chunk = Document(page_content=text, metadata=chunk_metadata)
                chunks.append(chunk)

            return chunks

        except Exception as e:
            self.logger.warning(
                f"Token-based splitting failed: {e}. Falling back to character splitting."
            )
            return await self._split_document_basic(
                document, chunk_size, chunk_overlap, "recursive", False
            )

    async def _split_semantically(
        self,
        document: Document,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[Document]:
        """Split document based on semantic boundaries."""
        try:
            # This would require embeddings and semantic similarity
            # For now, fall back to sentence-based splitting
            content = document.page_content

            # Split by sentences first
            sentences = self._split_into_sentences(content)

            chunks = []
            current_chunk_sentences = []
            current_chunk_length = 0

            for sentence in sentences:
                sentence_length = len(sentence)

                # If adding this sentence would exceed chunk size
                if current_chunk_sentences and (
                    current_chunk_length + sentence_length > chunk_size
                ):
                    # Create chunk from current sentences
                    chunk_content = " ".join(current_chunk_sentences)
                    chunk_metadata = document.metadata.copy()
                    chunk_metadata.update(
                        {
                            "chunk_type": "semantic_sentence",
                            "sentence_count": len(current_chunk_sentences),
                            "chunk_length": current_chunk_length,
                        }
                    )

                    chunk = Document(page_content=chunk_content, metadata=chunk_metadata)
                    chunks.append(chunk)

                    # Handle overlap
                    if chunk_overlap > 0 and current_chunk_sentences:
                        # Keep last few sentences for overlap
                        overlap_sentences = []
                        overlap_length = 0

                        for sent in reversed(current_chunk_sentences):
                            if overlap_length + len(sent) <= chunk_overlap:
                                overlap_sentences.insert(0, sent)
                                overlap_length += len(sent)
                            else:
                                break

                        current_chunk_sentences = overlap_sentences
                        current_chunk_length = overlap_length
                    else:
                        current_chunk_sentences = []
                        current_chunk_length = 0

                current_chunk_sentences.append(sentence)
                current_chunk_length += sentence_length

            # Handle remaining sentences
            if current_chunk_sentences:
                chunk_content = " ".join(current_chunk_sentences)
                chunk_metadata = document.metadata.copy()
                chunk_metadata.update(
                    {
                        "chunk_type": "semantic_sentence",
                        "sentence_count": len(current_chunk_sentences),
                        "chunk_length": current_chunk_length,
                    }
                )

                chunk = Document(page_content=chunk_content, metadata=chunk_metadata)
                chunks.append(chunk)

            return chunks

        except Exception as e:
            self.logger.warning(
                f"Semantic splitting failed: {e}. Falling back to recursive splitting."
            )
            return await self._split_document_basic(
                document, chunk_size, chunk_overlap, "recursive", False
            )

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        # Simple sentence splitting - could be improved with NLTK or spaCy
        sentence_pattern = r"(?<=[.!?])\s+"
        sentences = re.split(sentence_pattern, text)

        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter out very short fragments
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def _get_content_length(self, content: str, use_token_counting: bool) -> int:
        """Get the length of content (characters or tokens)."""
        if use_token_counting:
            # For async token counting, we'd need to await this
            # For now, use a simple approximation: ~4 chars per token
            return len(content) // 4
        else:
            return len(content)

    async def calculate_optimal_chunk_size(
        self,
        documents: List[Document],
        target_chunks_per_document: int = 5,
        min_chunk_size: int = 500,
        max_chunk_size: int = 2000,
    ) -> int:
        """Calculate optimal chunk size based on document characteristics."""
        if not documents:
            return 1000  # Default

        # Calculate average document length
        total_length = sum(len(doc.page_content) for doc in documents)
        avg_length = total_length / len(documents)

        # Calculate optimal chunk size
        optimal_size = int(avg_length / target_chunks_per_document)

        # Clamp to min/max bounds
        optimal_size = max(min_chunk_size, min(max_chunk_size, optimal_size))

        self.logger.info(
            f"Calculated optimal chunk size: {optimal_size} "
            f"(avg doc length: {avg_length:.0f}, target chunks: {target_chunks_per_document})"
        )

        return optimal_size

    async def analyze_chunk_distribution(self, chunks: List[Document]) -> Dict[str, Any]:
        """Analyze the distribution of chunk sizes and characteristics."""
        if not chunks:
            return {}

        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        chunk_types = [chunk.metadata.get("chunk_type", "unknown") for chunk in chunks]

        # Calculate statistics
        stats = {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "chunk_types": {},
            "size_distribution": {
                "small": len([s for s in chunk_sizes if s < 500]),
                "medium": len([s for s in chunk_sizes if 500 <= s < 1500]),
                "large": len([s for s in chunk_sizes if s >= 1500]),
            },
        }

        # Count chunk types
        for chunk_type in chunk_types:
            stats["chunk_types"][chunk_type] = stats["chunk_types"].get(chunk_type, 0) + 1

        return stats
