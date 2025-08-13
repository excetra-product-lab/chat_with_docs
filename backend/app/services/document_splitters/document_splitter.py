"""Document splitting service for intelligent text chunking."""

import logging
import re
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class DocumentSplitter:
    """Service for splitting documents into optimal chunks."""

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        default_model: str = "gpt-4",
        legal_specific: bool = False,
    ):
        """Initialize the document splitter.

        Args:
            token_counter: Optional token counter for precise token-based splitting
            default_model: Default model for token counting and splitting
            legal_specific: Whether to use legal-specific tokenization patterns
        """
        self.logger = logging.getLogger(__name__)
        self.default_model = default_model
        self.legal_specific = legal_specific

        # Create token counter with appropriate configuration
        if token_counter:
            self.token_counter = token_counter
        else:
            self.token_counter = TokenCounter.create_for_model(
                model_name=default_model, legal_specific=legal_specific
            )

    async def split_document(
        self,
        document: Document,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: str = "recursive",
        use_token_counting: bool = False,
        model_name: str | None = None,
    ) -> list[Document]:
        """
        Split a document into chunks using various strategies.

        Args:
            document: The document to split
            chunk_size: Maximum size of each chunk (chars or tokens)
            chunk_overlap: Overlap between chunks (chars or tokens)
            strategy: Splitting strategy ('recursive', 'character', 'token', 'semantic', 'paragraph')
            use_token_counting: Whether to use token-based counting
            model_name: Optional model name for token counting

        Returns:
            List of document chunks
        """
        try:
            self.logger.info(
                f"Splitting document with strategy '{strategy}', chunk_size={chunk_size}, "
                f"overlap={chunk_overlap}, token_counting={use_token_counting}"
            )

            effective_model = model_name or self.default_model

            # Choose splitting strategy
            if strategy == "token":
                return await self._split_by_tokens(
                    document, chunk_size, chunk_overlap, effective_model
                )
            elif strategy == "paragraph":
                return await self._split_by_paragraphs(
                    document,
                    chunk_size,
                    chunk_overlap,
                    strategy,
                    use_token_counting,
                    effective_model,
                )
            elif strategy == "semantic":
                return await self._split_semantically(
                    document, chunk_size, chunk_overlap, effective_model
                )
            else:
                return await self._split_document_basic(
                    document,
                    chunk_size,
                    chunk_overlap,
                    strategy,
                    use_token_counting,
                    effective_model,
                )

        except Exception as e:
            self.logger.error(f"Error splitting document: {e}")
            # Return original document as single chunk on error
            return [document]

    async def split_documents(
        self,
        documents: list[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: str = "recursive",
        use_token_counting: bool = False,
        model_name: str | None = None,
    ) -> list[Document]:
        """Split multiple documents into chunks."""
        all_chunks = []

        for doc in documents:
            chunks = await self.split_document(
                doc, chunk_size, chunk_overlap, strategy, use_token_counting, model_name
            )
            all_chunks.extend(chunks)

        return all_chunks

    async def _split_with_structure_preservation(
        self,
        document: Document,
        chunk_size: int,
        chunk_overlap: int,
        strategy: str,
        use_token_counting: bool,
        model_name: str,
    ) -> list[Document]:
        """Split document while preserving structural elements."""
        content = document.page_content

        # Detect structural elements
        structure_info = self._analyze_document_structure(content)

        if structure_info["has_headings"]:
            return await self._split_by_headings(
                document,
                chunk_size,
                chunk_overlap,
                strategy,
                use_token_counting,
                model_name,
            )
        elif structure_info["has_paragraphs"]:
            return await self._split_by_paragraphs(
                document,
                chunk_size,
                chunk_overlap,
                strategy,
                use_token_counting,
                model_name,
            )
        else:
            return await self._split_document_basic(
                document,
                chunk_size,
                chunk_overlap,
                strategy,
                use_token_counting,
                model_name,
            )

    def _analyze_document_structure(self, content: str) -> dict[str, Any]:
        """Analyze the structure of document content."""
        # Check for markdown-style headings
        heading_pattern = r"^#{1,6}\s+.+$"
        headings = re.findall(heading_pattern, content, re.MULTILINE)

        # Check for paragraph breaks
        paragraphs = content.split("\n\n")
        meaningful_paragraphs = [
            p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 50
        ]

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
        model_name: str,
    ) -> list[Document]:
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
                document,
                chunk_size,
                chunk_overlap,
                strategy,
                use_token_counting,
                model_name,
            )

        # Create sections based on headings
        sections = []
        for i, heading in enumerate(headings):
            start_pos = int(heading["start_pos"])
            end_pos = (
                int(headings[i + 1]["start_pos"])
                if i + 1 < len(headings)
                else len(content)
            )

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
            section_length = self._get_content_length(
                str(section["content"]), use_token_counting
            )

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

                chunk = Document(
                    page_content=str(section["content"]), metadata=chunk_metadata
                )
                chunks.append(chunk)
            else:
                # Split large section
                section_doc = Document(
                    page_content=str(section["content"]),
                    metadata=document.metadata.copy(),
                )

                section_chunks = await self._split_document_basic(
                    section_doc,
                    chunk_size,
                    chunk_overlap,
                    strategy,
                    use_token_counting,
                    model_name,
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
        model_name: str,
    ) -> list[Document]:
        """Split document by paragraphs with token awareness."""
        try:
            content = document.page_content
            paragraphs = content.split("\n\n")

            chunks = []
            current_chunk_parts: list[str] = []
            current_chunk_length = 0

            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue

                # Calculate paragraph length (tokens or characters)
                if use_token_counting:
                    paragraph_length = (
                        await self.token_counter.count_tokens_for_model_async(
                            paragraph, model_name
                        )
                    )
                else:
                    paragraph_length = len(paragraph)

                # Check if adding this paragraph would exceed chunk size
                projected_length = current_chunk_length + paragraph_length
                if current_chunk_parts and projected_length > chunk_size:
                    # Create chunk from current parts
                    chunk_content = "\n\n".join(current_chunk_parts)
                    chunk_metadata = document.metadata.copy()

                    # Add token count if using token counting
                    if use_token_counting:
                        chunk_tokens = (
                            await self.token_counter.count_tokens_for_model_async(
                                chunk_content, model_name
                            )
                        )
                        chunk_metadata["chunk_tokens"] = chunk_tokens

                    chunk_metadata.update(
                        {
                            "chunk_type": "paragraph_based",
                            "paragraph_count": len(current_chunk_parts),
                            "chunk_length": current_chunk_length,
                            "model_name": model_name,
                            "use_token_counting": use_token_counting,
                        }
                    )

                    chunk = Document(
                        page_content=chunk_content, metadata=chunk_metadata
                    )
                    chunks.append(chunk)

                    # Start new chunk with overlap if specified
                    if chunk_overlap > 0 and current_chunk_parts:
                        # Keep some paragraphs for overlap
                        overlap_parts = current_chunk_parts[-1:]  # Keep last paragraph
                        current_chunk_parts = overlap_parts + [paragraph]
                        current_chunk_length = paragraph_length
                        if overlap_parts:
                            if use_token_counting:
                                overlap_length = await self.token_counter.count_tokens_for_model_async(
                                    overlap_parts[0], model_name
                                )
                            else:
                                overlap_length = len(overlap_parts[0])
                            current_chunk_length += overlap_length
                    else:
                        current_chunk_parts = [paragraph]
                        current_chunk_length = paragraph_length
                else:
                    current_chunk_parts.append(paragraph)
                    current_chunk_length = projected_length

            # Handle remaining paragraphs
            if current_chunk_parts:
                chunk_content = "\n\n".join(current_chunk_parts)
                chunk_metadata = document.metadata.copy()

                # Add token count if using token counting
                if use_token_counting:
                    chunk_tokens = (
                        await self.token_counter.count_tokens_for_model_async(
                            chunk_content, model_name
                        )
                    )
                    chunk_metadata["chunk_tokens"] = chunk_tokens

                chunk_metadata.update(
                    {
                        "chunk_type": "paragraph_based",
                        "paragraph_count": len(current_chunk_parts),
                        "chunk_length": current_chunk_length,
                        "model_name": model_name,
                        "use_token_counting": use_token_counting,
                    }
                )

                chunk = Document(page_content=chunk_content, metadata=chunk_metadata)
                chunks.append(chunk)

            self.logger.info(
                f"Split document into {len(chunks)} paragraph-based chunks"
            )
            return chunks

        except Exception as e:
            self.logger.warning(
                f"Paragraph splitting failed: {e}. Falling back to basic splitting."
            )
            return await self._split_document_basic(
                document,
                chunk_size,
                chunk_overlap,
                "recursive",
                use_token_counting,
                model_name,
            )

    async def _split_document_basic(
        self,
        document: Document,
        chunk_size: int,
        chunk_overlap: int,
        strategy: str,
        use_token_counting: bool,
        model_name: str,
    ) -> list[Document]:
        """Split document using basic text splitting strategies."""
        content = document.page_content

        if use_token_counting:
            return await self._split_by_tokens(
                document, chunk_size, chunk_overlap, model_name
            )

        # Create appropriate splitter based on strategy
        splitter: RecursiveCharacterTextSplitter | CharacterTextSplitter
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
            return await self._split_semantically(
                document, chunk_size, chunk_overlap, model_name
            )
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
                    "model_name": model_name,
                    "use_token_counting": use_token_counting,
                }
            )

            chunk = Document(page_content=text, metadata=chunk_metadata)
            chunks.append(chunk)

        self.logger.info(
            f"Split document into {len(chunks)} chunks using {strategy} strategy"
        )
        return chunks

    async def _split_by_tokens(
        self,
        document: Document,
        chunk_size: int,
        chunk_overlap: int,
        model_name: str,
    ) -> list[Document]:
        """Split document by token count using enhanced TokenCounter."""
        try:
            # Use the enhanced TokenCounter's Langchain integration
            texts = self.token_counter.split_text_by_tokens(
                document.page_content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                model_name=model_name,
            )

            chunks = []
            for i, text in enumerate(texts):
                # Use async token counting
                token_count = await self.token_counter.count_tokens_for_model_async(
                    text, model_name
                )

                chunk_metadata = document.metadata.copy()
                chunk_metadata.update(
                    {
                        "chunk_index": i,
                        "chunk_total": len(texts),
                        "chunk_strategy": "token",
                        "chunk_tokens": token_count,
                        "chunk_size": len(text),
                        "chunk_type": "token_split",
                        "model_name": model_name,
                        "legal_specific": self.legal_specific,
                    }
                )

                chunk = Document(page_content=text, metadata=chunk_metadata)
                chunks.append(chunk)

            self.logger.info(
                f"Split document into {len(chunks)} token-based chunks using {model_name}"
            )
            return chunks

        except Exception as e:
            self.logger.warning(
                f"Token-based splitting failed: {e}. Falling back to character splitting."
            )
            return await self._split_document_basic(
                document, chunk_size, chunk_overlap, "recursive", False, model_name
            )

    async def _split_semantically(
        self,
        document: Document,
        chunk_size: int,
        chunk_overlap: int,
        model_name: str,
    ) -> list[Document]:
        """Split document based on semantic boundaries."""
        try:
            content = document.page_content

            # Split into sentences first
            sentences = self._split_into_sentences(content)

            chunks = []
            current_chunk_sentences: list[str] = []
            current_chunk_length = 0

            for sentence in sentences:
                # Use token counting for semantic splitting for better accuracy
                sentence_length = await self.token_counter.count_tokens_for_model_async(
                    sentence, model_name
                )

                # Check if adding this sentence would exceed chunk size
                if (
                    current_chunk_sentences
                    and (current_chunk_length + sentence_length) > chunk_size
                ):
                    # Create chunk from current sentences
                    chunk_content = " ".join(current_chunk_sentences)
                    chunk_tokens = (
                        await self.token_counter.count_tokens_for_model_async(
                            chunk_content, model_name
                        )
                    )

                    chunk_metadata = document.metadata.copy()
                    chunk_metadata.update(
                        {
                            "chunk_type": "semantic_sentence",
                            "sentence_count": len(current_chunk_sentences),
                            "chunk_length": current_chunk_length,
                            "chunk_tokens": chunk_tokens,
                            "model_name": model_name,
                        }
                    )

                    chunk = Document(
                        page_content=chunk_content, metadata=chunk_metadata
                    )
                    chunks.append(chunk)

                    # Start new chunk with overlap if specified
                    if chunk_overlap > 0 and current_chunk_sentences:
                        # Keep some sentences for overlap
                        overlap_sentences = current_chunk_sentences[
                            -2:
                        ]  # Keep last 2 sentences
                        overlap_length = sum(
                            [
                                await self.token_counter.count_tokens_for_model_async(
                                    s, model_name
                                )
                                for s in overlap_sentences
                            ]
                        )
                        current_chunk_sentences = overlap_sentences + [sentence]
                        current_chunk_length = overlap_length + sentence_length
                    else:
                        current_chunk_sentences = [sentence]
                        current_chunk_length = sentence_length
                else:
                    current_chunk_sentences.append(sentence)
                    current_chunk_length += sentence_length

            # Handle remaining sentences
            if current_chunk_sentences:
                chunk_content = " ".join(current_chunk_sentences)
                chunk_tokens = await self.token_counter.count_tokens_for_model_async(
                    chunk_content, model_name
                )

                chunk_metadata = document.metadata.copy()
                chunk_metadata.update(
                    {
                        "chunk_type": "semantic_sentence",
                        "sentence_count": len(current_chunk_sentences),
                        "chunk_length": current_chunk_length,
                        "chunk_tokens": chunk_tokens,
                        "model_name": model_name,
                    }
                )

                chunk = Document(page_content=chunk_content, metadata=chunk_metadata)
                chunks.append(chunk)

            self.logger.info(f"Split document into {len(chunks)} semantic chunks")
            return chunks

        except Exception as e:
            self.logger.warning(
                f"Semantic splitting failed: {e}. Falling back to recursive splitting."
            )
            return await self._split_document_basic(
                document, chunk_size, chunk_overlap, "recursive", False, model_name
            )

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex patterns."""
        # Enhanced sentence splitting for legal documents
        if self.legal_specific:
            # Legal-specific sentence patterns
            sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])|(?<=\.)\s+(?=\d+\.)|(?<=\.)\s+(?=§)|(?<=\.)\s+(?=Section)"
        else:
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

    def _get_content_length(
        self, content: str, use_token_counting: bool, model_name: str | None = None
    ) -> int:
        """Get the length of content (characters or tokens)."""
        if use_token_counting and model_name:
            # Note: This would need to be async in a real implementation
            # For now, use the sync version with a warning
            try:
                return self.token_counter.count_tokens_for_model(content, model_name)
            except Exception as e:
                self.logger.warning(f"Failed to count tokens: {e}")
                return len(content) // 4  # Fallback approximation
        else:
            return len(content)

    def get_optimal_chunk_size(
        self, document: Document, target_chunks: int = 10, model_name: str | None = None
    ) -> int:
        """
        Calculate optimal chunk size for a document to achieve target number of chunks.

        Args:
            document: Document to analyze
            target_chunks: Desired number of chunks
            model_name: Model for token counting

        Returns:
            Recommended chunk size
        """
        effective_model = model_name or self.default_model

        try:
            total_tokens = self.token_counter.count_tokens_for_model(
                document.page_content, effective_model
            )
            optimal_size = max(100, total_tokens // target_chunks)  # Minimum 100 tokens

            self.logger.info(
                f"Calculated optimal chunk size: {optimal_size} tokens for {target_chunks} chunks"
            )
            return optimal_size

        except Exception as e:
            self.logger.error(f"Error calculating optimal chunk size: {e}")
            return 1000  # Default fallback

    async def analyze_chunk_distribution(
        self, chunks: list[Document]
    ) -> dict[str, Any]:
        """
        Analyze the distribution and characteristics of document chunks.

        Args:
            chunks: List of document chunks to analyze

        Returns:
            Dictionary containing chunk distribution statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "chunk_sizes": [],
                "total_characters": 0,
                "avg_tokens_per_chunk": 0,
                "chunk_types": {},
                "metadata_analysis": {},
            }

        try:
            # Calculate basic statistics
            chunk_sizes = [len(chunk.page_content) for chunk in chunks]
            total_chunks = len(chunks)
            total_characters = sum(chunk_sizes)

            # Calculate chunk type distribution
            chunk_types: dict[str, int] = {}
            chunk_strategies: dict[str, int] = {}

            # Estimate tokens per chunk (using character-based estimation if token count not available)
            total_estimated_tokens = 0
            chunks_with_tokens = 0

            for chunk in chunks:
                # Analyze chunk type from metadata
                chunk_type = chunk.metadata.get("chunk_type", "unknown")
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

                # Analyze chunk strategy
                strategy = chunk.metadata.get("chunk_strategy", "unknown")
                chunk_strategies[strategy] = chunk_strategies.get(strategy, 0) + 1

                # Get token count if available, otherwise estimate
                chunk_tokens = chunk.metadata.get("chunk_tokens")
                if chunk_tokens is not None:
                    total_estimated_tokens += chunk_tokens
                    chunks_with_tokens += 1
                else:
                    # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
                    estimated_tokens = len(chunk.page_content) // 4
                    total_estimated_tokens += estimated_tokens

            avg_tokens_per_chunk = (
                total_estimated_tokens / total_chunks if total_chunks > 0 else 0
            )

            # Analyze metadata patterns
            metadata_keys: set[str] = set()
            for chunk in chunks:
                metadata_keys.update(chunk.metadata.keys())

            metadata_analysis = {
                "common_metadata_fields": list(metadata_keys),
                "chunks_with_token_counts": chunks_with_tokens,
                "token_counting_coverage": chunks_with_tokens / total_chunks
                if total_chunks > 0
                else 0,
            }

            stats = {
                "total_chunks": total_chunks,
                "avg_chunk_size": total_characters / total_chunks
                if total_chunks > 0
                else 0,
                "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
                "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
                "chunk_sizes": chunk_sizes,
                "total_characters": total_characters,
                "avg_tokens_per_chunk": avg_tokens_per_chunk,
                "total_estimated_tokens": total_estimated_tokens,
                "chunk_types": chunk_types,
                "chunk_strategies": chunk_strategies,
                "metadata_analysis": metadata_analysis,
                "size_distribution": {
                    "small_chunks": len([s for s in chunk_sizes if s < 500]),
                    "medium_chunks": len([s for s in chunk_sizes if 500 <= s < 1500]),
                    "large_chunks": len([s for s in chunk_sizes if s >= 1500]),
                },
            }

            self.logger.info(
                f"Chunk analysis complete: {total_chunks} chunks, avg size: {stats['avg_chunk_size']:.1f} chars"
            )
            return stats

        except Exception as e:
            self.logger.error(f"Error analyzing chunk distribution: {e}")
            return {
                "total_chunks": len(chunks),
                "error": str(e),
                "avg_chunk_size": sum(len(chunk.page_content) for chunk in chunks)
                / len(chunks)
                if chunks
                else 0,
            }
