"""Text chunking service for splitting documents into manageable chunks for RAG."""

import logging
import re
from typing import Dict, List, Optional

from app.services.document_parser import ParsedContent

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Container for a document chunk with metadata."""

    def __init__(
        self,
        text: str,
        chunk_index: int,
        document_filename: str,
        page_number: Optional[int] = None,
        section_title: Optional[str] = None,
        start_char: int = 0,
        end_char: int = 0,
        metadata: Optional[Dict] = None,
    ):
        self.text = text
        self.chunk_index = chunk_index
        self.document_filename = document_filename
        self.page_number = page_number
        self.section_title = section_title
        self.start_char = start_char
        self.end_char = end_char
        self.metadata = metadata or {}
        self.char_count = len(text)


class TextChunker:
    """Service for chunking text content into manageable pieces for RAG."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        min_chunk_size: int = 100,
    ):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum size for a chunk to be considered valid
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.logger = logging.getLogger(__name__)

    def chunk_document(self, parsed_content: ParsedContent) -> List[DocumentChunk]:
        """
        Chunk a parsed document into smaller pieces.

        Args:
            parsed_content: The parsed document content

        Returns:
            List[DocumentChunk]: List of document chunks
        """
        chunks = []

        # Use structured content if available for better chunking
        if parsed_content.structured_content:
            chunks = self._chunk_structured_content(parsed_content)
        else:
            # Fallback to simple text chunking
            chunks = self._chunk_plain_text(parsed_content)

        # Filter out chunks that are too small
        valid_chunks = [chunk for chunk in chunks if len(chunk.text.strip()) >= self.min_chunk_size]

        self.logger.info(
            f"Created {len(valid_chunks)} chunks from document {parsed_content.metadata.filename}"
        )

        return valid_chunks

    def _chunk_structured_content(self, parsed_content: ParsedContent) -> List[DocumentChunk]:
        """Chunk document using structured content information."""
        chunks: List[DocumentChunk] = []
        current_chunk = ""
        current_page = None
        current_section = None
        chunk_start_char = 0

        for item in parsed_content.structured_content:
            item_text = item.get("text", "")
            item_type = item.get("type", "")

            # Track page numbers for PDFs
            if item_type == "page":
                current_page = item.get("page_number")

            # Track section headers for structured documents
            if item.get("is_potential_header", False):
                current_section = item_text

            # Check if adding this item would exceed chunk size
            if len(current_chunk) + len(item_text) > self.chunk_size and current_chunk:
                # Create chunk from current content
                chunk = self._create_chunk(
                    text=current_chunk.strip(),
                    chunk_index=len(chunks),
                    filename=parsed_content.metadata.filename,
                    page_number=current_page,
                    section_title=current_section,
                    start_char=chunk_start_char,
                    end_char=chunk_start_char + len(current_chunk),
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + "\n\n" + item_text
                chunk_start_char += len(current_chunk) - len(overlap_text) - len(item_text) - 2
            else:
                # Add item to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + item_text
                else:
                    current_chunk = item_text
                    chunk_start_char = 0

        # Add final chunk if there's remaining content
        if current_chunk.strip():
            chunk = self._create_chunk(
                text=current_chunk.strip(),
                chunk_index=len(chunks),
                filename=parsed_content.metadata.filename,
                page_number=current_page,
                section_title=current_section,
                start_char=chunk_start_char,
                end_char=chunk_start_char + len(current_chunk),
            )
            chunks.append(chunk)

        return chunks

    def _chunk_plain_text(self, parsed_content: ParsedContent) -> List[DocumentChunk]:
        """Chunk document using simple text splitting."""
        text = parsed_content.text
        chunks: List[DocumentChunk] = []

        # Split text into sentences for better chunk boundaries
        sentences = self._split_into_sentences(text)

        current_chunk = ""
        current_start = 0

        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Create chunk from current content
                chunk = self._create_chunk(
                    text=current_chunk.strip(),
                    chunk_index=len(chunks),
                    filename=parsed_content.metadata.filename,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_start += len(current_chunk) - len(overlap_text) - len(sentence) - 1
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                    current_start = 0

        # Add final chunk if there's remaining content
        if current_chunk.strip():
            chunk = self._create_chunk(
                text=current_chunk.strip(),
                chunk_index=len(chunks),
                filename=parsed_content.metadata.filename,
                start_char=current_start,
                end_char=current_start + len(current_chunk),
            )
            chunks.append(chunk)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        # Pattern for sentence boundaries (periods, exclamation marks, question marks)
        # followed by whitespace and capital letter
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"

        sentences = re.split(sentence_pattern, text)

        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of a chunk."""
        if len(text) <= self.chunk_overlap:
            return text

        # Try to find a good break point (sentence boundary) within overlap range
        overlap_start = len(text) - self.chunk_overlap
        overlap_text = text[overlap_start:]

        # Look for sentence boundaries in the overlap
        sentence_boundaries = [m.start() for m in re.finditer(r"[.!?]\s+", overlap_text)]

        if sentence_boundaries:
            # Use the last sentence boundary as the start of overlap
            last_boundary = sentence_boundaries[-1] + 2  # +2 to include the punctuation and space
            return overlap_text[last_boundary:]

        # If no sentence boundary found, use the full overlap
        return overlap_text

    def _create_chunk(
        self,
        text: str,
        chunk_index: int,
        filename: str,
        page_number: Optional[int] = None,
        section_title: Optional[str] = None,
        start_char: int = 0,
        end_char: int = 0,
    ) -> DocumentChunk:
        """Create a DocumentChunk with metadata."""
        return DocumentChunk(
            text=text,
            chunk_index=chunk_index,
            document_filename=filename,
            page_number=page_number,
            section_title=section_title,
            start_char=start_char,
            end_char=end_char,
            metadata={
                "chunk_size": len(text),
                "has_page_number": page_number is not None,
                "has_section_title": section_title is not None,
            },
        )

    def get_chunk_summary(self, chunks: List[DocumentChunk]) -> Dict:
        """Get summary statistics about the chunks."""
        if not chunks:
            return {"total_chunks": 0}

        total_chars = sum(chunk.char_count for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks)

        chunks_with_pages = sum(1 for chunk in chunks if chunk.page_number is not None)
        chunks_with_sections = sum(1 for chunk in chunks if chunk.section_title is not None)

        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "average_chunk_size": round(avg_chunk_size, 2),
            "chunks_with_page_numbers": chunks_with_pages,
            "chunks_with_section_titles": chunks_with_sections,
            "min_chunk_size": min(chunk.char_count for chunk in chunks),
            "max_chunk_size": max(chunk.char_count for chunk in chunks),
        }
