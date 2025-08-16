"""Text chunking service for splitting documents into manageable chunks for RAG."""

import logging
import re
from typing import Generator

from app.services.document_parser import ParsedContent

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Container for a document chunk with metadata and validation."""

    def __init__(
        self,
        text: str,
        chunk_index: int,
        document_filename: str,
        page_number: int | None = None,
        section_title: str | None = None,
        start_char: int = 0,
        end_char: int = 0,
        metadata: dict | None = None,
    ):
        # Validation
        if not text or not text.strip():
            raise ValueError("Chunk text cannot be empty or whitespace-only")
        if chunk_index < 0:
            raise ValueError("Chunk index must be non-negative")
        if start_char < 0:
            raise ValueError("Start character position must be non-negative")
        if end_char < start_char:
            raise ValueError("End character position must be >= start character position")
        if not document_filename or not document_filename.strip():
            raise ValueError("Document filename cannot be empty")

        self.text = text.strip()  # Always store trimmed text
        self.chunk_index = chunk_index
        self.document_filename = document_filename.strip()
        self.page_number = page_number
        self.section_title = section_title.strip() if section_title else None
        self.start_char = start_char
        self.end_char = end_char
        self.metadata = metadata or {}
        # Calculate char_count from the actual stored (trimmed) text
        self.char_count = len(self.text)


class TextChunker:
    """Service for chunking text content into manageable pieces for RAG."""

    # Pre-compiled regex patterns for performance
    SENTENCE_PATTERN = re.compile(
        r'(?<![A-Z][a-z])\. (?=[A-Z])|(?<=[.!?])\s+(?=[A-Z])',
        re.MULTILINE
    )
    
    # More sophisticated sentence boundary detection
    ADVANCED_SENTENCE_PATTERN = re.compile(
        r'''
        (?<!\w\.\w.)        # Not preceded by abbreviation like Dr. or U.S.
        (?<![A-Z][a-z]\.)   # Not preceded by Name. (like Mr.)
        (?<![0-9]\.)        # Not preceded by number.
        (?<=\.|\!|\?)       # Must be preceded by sentence ender
        \s+                 # One or more whitespace
        (?=[A-Z])           # Must be followed by capital letter
        ''',
        re.VERBOSE | re.MULTILINE
    )
    
    OVERLAP_BOUNDARY_PATTERN = re.compile(r'[.!?]\s+')

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
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if chunk_overlap < 0:
            raise ValueError("Chunk overlap must be non-negative")
        if min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be positive")
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.logger = logging.getLogger(__name__)

    def chunk_document(self, parsed_content: ParsedContent) -> list[DocumentChunk]:
        """
        Chunk a parsed document into smaller pieces.

        Args:
            parsed_content: The parsed document content

        Returns:
            List[DocumentChunk]: List of document chunks
        """
        if not parsed_content or not parsed_content.text.strip():
            self.logger.warning(f"Empty document content for {parsed_content.metadata.filename}")
            return []

        # Use structured content if available for better chunking
        if parsed_content.structured_content:
            chunks = list(self._chunk_structured_content(parsed_content))
        else:
            # Fallback to simple text chunking
            chunks = list(self._chunk_plain_text(parsed_content))

        # Filter out chunks that are too small and reassign indices
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            if len(chunk.text.strip()) >= self.min_chunk_size:
                # Create new chunk with corrected index
                valid_chunk = DocumentChunk(
                    text=chunk.text,
                    chunk_index=i,  # Sequential indexing after filtering
                    document_filename=chunk.document_filename,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata=chunk.metadata,
                )
                valid_chunks.append(valid_chunk)

        self.logger.info(
            f"Created {len(valid_chunks)} valid chunks from document {parsed_content.metadata.filename} "
            f"(filtered {len(chunks) - len(valid_chunks)} chunks below minimum size)"
        )

        return valid_chunks

    def _chunk_structured_content(
        self, parsed_content: ParsedContent
    ) -> Generator[DocumentChunk, None, None]:
        """Chunk document using structured content information."""
        current_parts = []  # Use list for efficient string building
        current_page = None
        current_section = None
        text_position = 0  # Track position in original text
        chunk_start_pos = 0

        for item in parsed_content.structured_content:
            item_text = item.get("text", "").strip()
            if not item_text:
                continue
                
            item_type = item.get("type", "")

            # Track page numbers for PDFs
            if item_type == "page":
                current_page = item.get("page_number")

            # Track section headers for structured documents
            if item.get("is_potential_header", False):
                current_section = item_text

            # Calculate what the new chunk size would be
            separator = "\n\n" if current_parts else ""
            potential_addition = separator + item_text
            current_text = "".join(current_parts)
            
            # Check if adding this item would exceed chunk size
            if len(current_text) + len(potential_addition) > self.chunk_size and current_parts:
                # Create chunk from current content
                chunk_text = current_text.strip()
                if chunk_text:  # Only create non-empty chunks
                    yield self._create_chunk(
                        text=chunk_text,
                        chunk_index=0,  # Will be reassigned later
                        filename=parsed_content.metadata.filename,
                        page_number=current_page,
                        section_title=current_section,
                        start_char=chunk_start_pos,
                        end_char=chunk_start_pos + len(chunk_text),
                    )

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_text)
                current_parts = [overlap_text, potential_addition] if overlap_text else [item_text]
                
                # Update position tracking
                chunk_start_pos = text_position - len(overlap_text) if overlap_text else text_position
            else:
                # Add item to current chunk
                current_parts.append(potential_addition)
                if not current_parts or len(current_parts) == 1:  # First item
                    chunk_start_pos = text_position

            # Update text position
            text_position += len(item_text) + (2 if item_type != "page" else 1)  # Account for separators

        # Add final chunk if there's remaining content
        if current_parts:
            final_text = "".join(current_parts).strip()
            if final_text:
                yield self._create_chunk(
                    text=final_text,
                    chunk_index=0,  # Will be reassigned later
                    filename=parsed_content.metadata.filename,
                    page_number=current_page,
                    section_title=current_section,
                    start_char=chunk_start_pos,
                    end_char=chunk_start_pos + len(final_text),
                )

    def _chunk_plain_text(self, parsed_content: ParsedContent) -> Generator[DocumentChunk, None, None]:
        """Chunk document using simple text splitting with proper position tracking."""
        text = parsed_content.text.strip()
        if not text:
            return

        # Split text into sentences for better chunk boundaries
        sentences = self._split_into_sentences(text)
        if not sentences:
            return

        current_parts = []  # Use list for efficient string building
        current_start_pos = 0
        text_position = 0  # Track position in original text

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Calculate potential new chunk size
            separator = " " if current_parts else ""
            potential_addition = separator + sentence
            current_text = "".join(current_parts)
            
            # Check if adding this sentence would exceed chunk size
            if len(current_text) + len(potential_addition) > self.chunk_size and current_parts:
                # Create chunk from current content
                chunk_text = current_text.strip()
                if chunk_text:  # Only create non-empty chunks
                    yield self._create_chunk(
                        text=chunk_text,
                        chunk_index=0,  # Will be reassigned later
                        filename=parsed_content.metadata.filename,
                        start_char=current_start_pos,
                        end_char=current_start_pos + len(chunk_text),
                    )

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_text)
                if overlap_text:
                    current_parts = [overlap_text, " " + sentence]
                    # Find where overlap starts in original text
                    overlap_start_in_chunk = len(current_text) - len(overlap_text)
                    current_start_pos += overlap_start_in_chunk
                else:
                    current_parts = [sentence]
                    current_start_pos = text_position
            else:
                # Add sentence to current chunk
                current_parts.append(potential_addition)
                if len(current_parts) == 1:  # First sentence
                    current_start_pos = text_position

            # Update text position (find sentence in original text)
            sentence_start = text.find(sentence, text_position)
            if sentence_start != -1:
                text_position = sentence_start + len(sentence)

        # Add final chunk if there's remaining content
        if current_parts:
            final_text = "".join(current_parts).strip()
            if final_text:
                yield self._create_chunk(
                    text=final_text,
                    chunk_index=0,  # Will be reassigned later
                    filename=parsed_content.metadata.filename,
                    start_char=current_start_pos,
                    end_char=current_start_pos + len(final_text),
                )

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using improved regex patterns."""
        if not text.strip():
            return []

        # Try advanced pattern first
        try:
            sentences = self.ADVANCED_SENTENCE_PATTERN.split(text)
        except Exception:
            # Fallback to simple pattern
            sentences = self.SENTENCE_PATTERN.split(text)

        # Clean up sentences and remove empty ones
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 1:  # Avoid single character "sentences"
                cleaned_sentences.append(sentence)

        # If no sentences found, return the whole text
        if not cleaned_sentences and text.strip():
            cleaned_sentences = [text.strip()]

        return cleaned_sentences

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of a chunk with improved boundary detection."""
        if not text or len(text) <= self.chunk_overlap:
            return text

        # Get the overlap region
        overlap_start = len(text) - self.chunk_overlap
        overlap_text = text[overlap_start:]

        # Look for sentence boundaries in the overlap using pre-compiled pattern
        boundaries = list(self.OVERLAP_BOUNDARY_PATTERN.finditer(overlap_text))

        if boundaries:
            # Use the last sentence boundary as the start of overlap
            last_boundary = boundaries[-1]
            boundary_end = last_boundary.end()
            result = overlap_text[boundary_end:].strip()
            if result:  # Only return non-empty overlap
                return result

        # If no good boundary found, try to avoid breaking words
        words = overlap_text.split()
        if len(words) > 1:
            # Remove the first partial word to avoid mid-word breaks
            return " ".join(words[1:])

        # Last resort: use the full overlap
        return overlap_text

    def _create_chunk(
        self,
        text: str,
        chunk_index: int,
        filename: str,
        page_number: int | None = None,
        section_title: str | None = None,
        start_char: int = 0,
        end_char: int = 0,
    ) -> DocumentChunk:
        """Create a DocumentChunk with metadata."""
        # Ensure end_char matches the actual text length after stripping
        cleaned_text = text.strip()
        if end_char == start_char + len(text):
            # Adjust end_char for stripped text
            end_char = start_char + len(cleaned_text)
        
        return DocumentChunk(
            text=cleaned_text,
            chunk_index=chunk_index,
            document_filename=filename,
            page_number=page_number,
            section_title=section_title,
            start_char=start_char,
            end_char=end_char,
            metadata={
                "chunk_size": len(cleaned_text),
                "has_page_number": page_number is not None,
                "has_section_title": section_title is not None,
                "chunk_method": "structured" if section_title else "plain_text",
            },
        )

    def get_chunk_summary(self, chunks: list[DocumentChunk]) -> dict:
        """Get summary statistics about the chunks."""
        if not chunks:
            return {"total_chunks": 0}

        total_chars = sum(chunk.char_count for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks)

        chunks_with_pages = sum(1 for chunk in chunks if chunk.page_number is not None)
        chunks_with_sections = sum(
            1 for chunk in chunks if chunk.section_title is not None
        )

        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "average_chunk_size": round(avg_chunk_size, 2),
            "chunks_with_page_numbers": chunks_with_pages,
            "chunks_with_section_titles": chunks_with_sections,
            "min_chunk_size": min(chunk.char_count for chunk in chunks),
            "max_chunk_size": max(chunk.char_count for chunk in chunks),
        }
