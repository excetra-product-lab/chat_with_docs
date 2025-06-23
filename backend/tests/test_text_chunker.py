"""Tests for text chunker service."""

from app.services.document_parser import DocumentMetadata, ParsedContent
from app.services.text_chunker import DocumentChunk, TextChunker


class TestTextChunker:
    """Test cases for TextChunker."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = TextChunker(
            chunk_size=100,
            chunk_overlap=20,
            min_chunk_size=10,  # Small size for testing
        )

    def create_parsed_content(
        self, text: str, filename: str = "test.txt", structured_content=None
    ):
        """Helper to create ParsedContent for testing."""
        metadata = DocumentMetadata(
            filename=filename, file_type="txt", total_chars=len(text)
        )
        return ParsedContent(
            text=text, metadata=metadata, structured_content=structured_content or []
        )

    def test_chunk_simple_text(self):
        """Test chunking simple text."""
        text = "This is a test document. " * 10  # Create text longer than chunk_size
        parsed_content = self.create_parsed_content(text)

        chunks = self.chunker.chunk_document(parsed_content)

        assert len(chunks) > 1  # Should create multiple chunks
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.document_filename == "test.txt" for chunk in chunks)
        assert all(len(chunk.text) >= self.chunker.min_chunk_size for chunk in chunks)

    def test_chunk_with_structured_content(self):
        """Test chunking with structured content."""
        structured_content = [
            {"type": "paragraph", "text": "First paragraph. " * 5},
            {"type": "paragraph", "text": "Second paragraph. " * 5},
            {"type": "paragraph", "text": "Third paragraph. " * 5},
        ]
        text = " ".join(item["text"] for item in structured_content)
        parsed_content = self.create_parsed_content(
            text, structured_content=structured_content
        )

        chunks = self.chunker.chunk_document(parsed_content)

        assert len(chunks) > 0
        assert all(
            chunk.char_count <= self.chunker.chunk_size + self.chunker.chunk_overlap
            for chunk in chunks
        )

    def test_chunk_with_page_numbers(self):
        """Test chunking PDF content with page numbers."""
        structured_content = [
            {"type": "page", "page_number": 1, "text": "Page 1 content. " * 10},
            {"type": "page", "page_number": 2, "text": "Page 2 content. " * 10},
        ]
        text = " ".join(item["text"] for item in structured_content)
        parsed_content = self.create_parsed_content(
            text, "test.pdf", structured_content
        )

        chunks = self.chunker.chunk_document(parsed_content)

        assert len(chunks) > 0
        # Check that page numbers are preserved
        page_numbers = [
            chunk.page_number for chunk in chunks if chunk.page_number is not None
        ]
        assert len(page_numbers) > 0

    def test_chunk_with_section_headers(self):
        """Test chunking with section headers."""
        structured_content = [
            {"type": "paragraph", "text": "INTRODUCTION", "is_potential_header": True},
            {"type": "paragraph", "text": "This is the introduction content. " * 5},
            {"type": "paragraph", "text": "CONCLUSION", "is_potential_header": True},
            {"type": "paragraph", "text": "This is the conclusion content. " * 5},
        ]
        text = " ".join(item["text"] for item in structured_content)
        parsed_content = self.create_parsed_content(
            text, structured_content=structured_content
        )

        chunks = self.chunker.chunk_document(parsed_content)

        assert len(chunks) > 0
        # Check that section titles are preserved
        section_titles = [
            chunk.section_title for chunk in chunks if chunk.section_title is not None
        ]
        assert len(section_titles) > 0
        assert "INTRODUCTION" in section_titles or "CONCLUSION" in section_titles

    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        text = (
            "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. "
            * 3
        )
        parsed_content = self.create_parsed_content(text)

        chunks = self.chunker.chunk_document(parsed_content)

        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                current_chunk = chunks[i]
                next_chunk = chunks[i + 1]

                # Get the end of current chunk and start of next chunk
                current_end = current_chunk.text[-self.chunker.chunk_overlap :]
                next_start = next_chunk.text[: self.chunker.chunk_overlap]

                # There should be some common words (simple overlap check)
                current_words = set(current_end.split())
                next_words = set(next_start.split())
                overlap_words = current_words.intersection(next_words)

                # Should have at least some overlapping words
                assert (
                    len(overlap_words) > 0
                    or len(current_chunk.text) < self.chunker.chunk_overlap
                )

    def test_chunk_minimum_size_filter(self):
        """Test that chunks below minimum size are filtered out."""
        # Create text with very short segments
        text = "A. B. C. D. E."  # Very short sentences
        parsed_content = self.create_parsed_content(text)

        chunks = self.chunker.chunk_document(parsed_content)

        # All chunks should meet minimum size requirement
        assert all(len(chunk.text) >= self.chunker.min_chunk_size for chunk in chunks)

    def test_split_into_sentences(self):
        """Test sentence splitting functionality."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        sentences = self.chunker._split_into_sentences(text)

        assert len(sentences) == 4
        assert "First sentence." in sentences[0]
        assert "Second sentence!" in sentences[1]
        assert "Third sentence?" in sentences[2]
        assert "Fourth sentence." in sentences[3]

    def test_get_overlap_text(self):
        """Test overlap text extraction."""
        text = (
            "This is a long sentence that should be used for overlap testing. "
            "Another sentence here."
        )

        overlap = self.chunker._get_overlap_text(text)

        assert len(overlap) <= self.chunker.chunk_overlap
        assert overlap in text
        # Should try to end at sentence boundary if possible
        assert (
            overlap.endswith(".")
            or overlap.endswith("!")
            or overlap.endswith("?")
            or len(text) <= self.chunker.chunk_overlap
        )

    def test_chunk_summary(self):
        """Test chunk summary statistics."""
        text = "This is a test document. " * 20
        parsed_content = self.create_parsed_content(text)
        chunks = self.chunker.chunk_document(parsed_content)

        summary = self.chunker.get_chunk_summary(chunks)

        assert summary["total_chunks"] == len(chunks)
        assert summary["total_characters"] > 0
        assert summary["average_chunk_size"] > 0
        assert summary["min_chunk_size"] >= self.chunker.min_chunk_size
        assert summary["max_chunk_size"] > 0
        assert "chunks_with_page_numbers" in summary
        assert "chunks_with_section_titles" in summary

    def test_chunk_summary_empty(self):
        """Test chunk summary with empty chunks list."""
        summary = self.chunker.get_chunk_summary([])
        assert summary["total_chunks"] == 0

    def test_create_chunk(self):
        """Test chunk creation with metadata."""
        chunk = self.chunker._create_chunk(
            text="Test chunk content",
            chunk_index=0,
            filename="test.pdf",
            page_number=1,
            section_title="Introduction",
            start_char=0,
            end_char=18,
        )

        assert chunk.text == "Test chunk content"
        assert chunk.chunk_index == 0
        assert chunk.document_filename == "test.pdf"
        assert chunk.page_number == 1
        assert chunk.section_title == "Introduction"
        assert chunk.start_char == 0
        assert chunk.end_char == 18
        assert chunk.char_count == len("Test chunk content")
        assert chunk.metadata["chunk_size"] == len("Test chunk content")
        assert chunk.metadata["has_page_number"] is True
        assert chunk.metadata["has_section_title"] is True

    def test_chunk_very_long_text(self):
        """Test chunking very long text."""
        # Create text much longer than chunk size
        text = "This is a very long document. " * 100
        parsed_content = self.create_parsed_content(text)

        chunks = self.chunker.chunk_document(parsed_content)

        assert len(chunks) > 5  # Should create many chunks
        assert all(
            len(chunk.text) <= self.chunker.chunk_size + self.chunker.chunk_overlap
            for chunk in chunks
        )

        # Verify all text is preserved (approximately, accounting for overlap)
        total_unique_chars = sum(len(chunk.text) for chunk in chunks)
        # Should be close to original length (with some extra due to overlap)
        assert total_unique_chars >= len(text)

    def test_chunk_short_text(self):
        """Test chunking text shorter than chunk size."""
        text = "Short text."
        parsed_content = self.create_parsed_content(text)

        chunks = self.chunker.chunk_document(parsed_content)

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_index == 0


class TestDocumentChunk:
    """Test cases for DocumentChunk."""

    def test_document_chunk_creation(self):
        """Test creating DocumentChunk."""
        chunk = DocumentChunk(
            text="Test chunk content",
            chunk_index=0,
            document_filename="test.pdf",
            page_number=1,
            section_title="Introduction",
            start_char=0,
            end_char=18,
            metadata={"custom": "value"},
        )

        assert chunk.text == "Test chunk content"
        assert chunk.chunk_index == 0
        assert chunk.document_filename == "test.pdf"
        assert chunk.page_number == 1
        assert chunk.section_title == "Introduction"
        assert chunk.start_char == 0
        assert chunk.end_char == 18
        assert chunk.char_count == len("Test chunk content")
        assert chunk.metadata["custom"] == "value"

    def test_document_chunk_defaults(self):
        """Test DocumentChunk with default values."""
        chunk = DocumentChunk(
            text="Test content", chunk_index=0, document_filename="test.txt"
        )

        assert chunk.text == "Test content"
        assert chunk.chunk_index == 0
        assert chunk.document_filename == "test.txt"
        assert chunk.page_number is None
        assert chunk.section_title is None
        assert chunk.start_char == 0
        assert chunk.end_char == 0
        assert chunk.char_count == len("Test content")
        assert chunk.metadata == {}

    def test_char_count_calculation(self):
        """Test that char_count is calculated correctly."""
        text = "This is a test with unicode: café"
        chunk = DocumentChunk(text=text, chunk_index=0, document_filename="test.txt")

        assert chunk.char_count == len(text)
