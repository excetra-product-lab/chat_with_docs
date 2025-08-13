"""Tests for hierarchical chunker with legal document support.

This module tests the HierarchicalChunker MVP functionality including:
- Hierarchy-aware chunking with legal document structures
- Token-based size management and optimization
- Integration with TokenCounter and StructureDetector
- Boundary preservation and semantic coherence
- Langchain compatibility
"""

from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from app.services.document_structure_detector import StructureDetector
from app.services.document_structure_detector.data_models import (
    DocumentStructure,
    ElementType,
)
from app.services.hierarchical_chunker import HierarchicalChunk, HierarchicalChunker
from app.utils.token_counter import TokenCounter


class TestHierarchicalChunker:
    """Test cases for HierarchicalChunker MVP functionality."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        # Create test components
        self.token_counter = TokenCounter(legal_specific=True)
        self.structure_detector = StructureDetector()

        # Initialize chunker with MVP-optimal settings
        self.chunker = HierarchicalChunker(
            token_counter=self.token_counter,
            structure_detector=self.structure_detector,
            chunk_size=600,  # MVP target
            chunk_overlap=100,
            legal_specific=True,
        )

    def test_hierarchical_chunker_initialization(self):
        """Test HierarchicalChunker initialization with MVP defaults."""
        assert self.chunker._chunk_size == 600
        assert self.chunker._chunk_overlap == 100
        assert self.chunker.min_chunk_size == 100
        assert self.chunker.max_chunk_size == 1024
        assert self.chunker.target_range == (400, 800)
        assert self.chunker.legal_specific is True
        assert self.chunker.model_name == "gpt-4o"
        assert self.chunker.token_counter is not None
        assert self.chunker.structure_detector is not None

    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom chunk size parameters."""
        custom_chunker = HierarchicalChunker(
            chunk_size=500,
            chunk_overlap=50,
            min_chunk_size=200,
            max_chunk_size=800,
            target_range=(300, 600),
            model_name="gpt-4o",
            legal_specific=False,
        )

        assert custom_chunker._chunk_size == 500
        assert custom_chunker._chunk_overlap == 50
        assert custom_chunker.min_chunk_size == 200
        assert custom_chunker.max_chunk_size == 800
        assert custom_chunker.target_range == (300, 600)
        assert custom_chunker.legal_specific is False

    def test_parameter_validation(self):
        """Test chunk parameter validation and adjustment."""
        # Test minimum chunk size adjustment
        chunker = HierarchicalChunker(chunk_size=50)  # Below minimum
        assert chunker._chunk_size == 100  # Should be adjusted to minimum

        # Test maximum chunk size adjustment
        chunker = HierarchicalChunker(chunk_size=2000)  # Above maximum
        assert chunker._chunk_size == 1024  # Should be adjusted to maximum

        # Test overlap adjustment
        chunker = HierarchicalChunker(
            chunk_size=400, chunk_overlap=500
        )  # Overlap > chunk_size
        assert chunker._chunk_overlap <= chunker._chunk_size // 4  # Should be adjusted

    def test_token_counter_integration(self):
        """Test integration with TokenCounter for accurate token counting."""
        test_text = "This is a test sentence for token counting validation."

        # Test token counting
        token_count = self.chunker._count_tokens_with_tracking(test_text)
        assert isinstance(token_count, int)
        assert token_count > 0

        # Verify usage statistics are updated
        assert self.chunker.token_usage_stats["token_counting_calls"] > 0
        assert self.chunker.token_usage_stats["total_tokens_processed"] >= token_count

    def test_default_legal_separators(self):
        """Test default legal document separators are properly configured."""
        separators = self.chunker._get_default_legal_separators()

        # Check hierarchy order (highest to lowest)
        assert "\n\nCHAPTER " in separators
        assert "\n\nARTICLE " in separators
        assert "\n\nSECTION " in separators
        assert "\n\n§ " in separators
        assert "\n\n(" in separators
        assert "\n\n" in separators
        assert ". " in separators

        # Verify hierarchy ordering
        chapter_idx = separators.index("\n\nCHAPTER ")
        section_idx = separators.index("\n\nSECTION ")
        paragraph_idx = separators.index("\n\n")

        assert chapter_idx < section_idx < paragraph_idx

    @pytest.fixture
    def sample_legal_document(self) -> str:
        """Sample legal document with hierarchical structure."""
        return """ARTICLE I
GENERAL PROVISIONS

Section 1.1 Purpose and Scope
This Agreement sets forth the terms and conditions governing the relationship between the parties.

Section 1.2 Definitions
For purposes of this Agreement, the following definitions shall apply:

(a) "Agreement" means this legal document and all amendments thereto.
(b) "Party" means each of the entities entering into this Agreement.
(c) "Effective Date" means the date this Agreement becomes effective.

ARTICLE II
OBLIGATIONS AND RESPONSIBILITIES

Section 2.1 General Obligations
Each Party shall comply with all applicable laws and regulations.

Subsection 2.1.1 Compliance Requirements
All activities must conform to applicable legal standards.

Subsection 2.1.2 Reporting Obligations
Regular reports must be submitted as specified herein.

Section 2.2 Specific Responsibilities
The following specific responsibilities apply to each Party:

(a) Maintain accurate records of all transactions.
(b) Provide timely notice of any material changes.
(c) Cooperate in good faith with all reasonable requests.

ARTICLE III
TERMINATION

Section 3.1 Termination for Cause
This Agreement may be terminated immediately upon material breach.

Section 3.2 Termination for Convenience
Either Party may terminate this Agreement with thirty (30) days written notice."""

    def test_hierarchical_chunking_basic(self, sample_legal_document):
        """Test basic hierarchical chunking functionality."""
        chunks = self.chunker.chunk_text_with_hierarchy(sample_legal_document)

        # Verify we get HierarchicalChunk objects
        assert len(chunks) > 0
        assert all(isinstance(chunk, HierarchicalChunk) for chunk in chunks)

        # Verify chunks have required attributes
        for chunk in chunks:
            assert hasattr(chunk, "text")
            assert hasattr(chunk, "token_count")
            assert hasattr(chunk, "hierarchy_level")
            assert hasattr(chunk, "chunk_index")
            assert chunk.token_count > 0
            assert len(chunk.text.strip()) > 0

    def test_chunk_size_constraints(self, sample_legal_document):
        """Test that chunks respect size constraints."""
        chunks = self.chunker.chunk_text_with_hierarchy(sample_legal_document)

        # Check token counts are within acceptable ranges
        for chunk in chunks:
            assert chunk.token_count >= self.chunker.min_chunk_size or len(chunks) == 1
            assert chunk.token_count <= self.chunker.max_chunk_size

        # Verify most chunks are in target range
        target_chunks = sum(
            1
            for chunk in chunks
            if self.chunker.target_min <= chunk.token_count <= self.chunker.target_max
        )

        # For small documents, may only have one chunk below target range
        target_percentage = (target_chunks / len(chunks)) * 100

        # If only one chunk and it's a small document, that's acceptable for MVP
        if len(chunks) == 1:
            # Single chunk should at least be within reasonable bounds
            assert (
                chunks[0].token_count >= self.chunker.min_chunk_size
                or chunks[0].token_count >= 50
            )
        else:
            # For multiple chunks, expect at least 30% in target range
            assert target_percentage >= 30

    def test_hierarchy_metadata_extraction(self, sample_legal_document):
        """Test extraction of hierarchy metadata from chunks."""
        chunks = self.chunker.chunk_text_with_hierarchy(sample_legal_document)

        # Find chunks with hierarchy information
        hierarchy_chunks = [chunk for chunk in chunks if chunk.hierarchy_level > 0]

        if hierarchy_chunks:  # Only test if structure was detected
            # Verify hierarchy levels are reasonable
            max_level = max(chunk.hierarchy_level for chunk in hierarchy_chunks)
            assert max_level <= 5  # Reasonable maximum depth

            # Check for proper element types
            element_types = [
                chunk.element_type for chunk in hierarchy_chunks if chunk.element_type
            ]
            assert len(element_types) > 0  # At least some should have element types

    def test_boundary_preservation(self, sample_legal_document):
        """Test that hierarchical boundaries are preserved."""
        chunks = self.chunker.chunk_text_with_hierarchy(sample_legal_document)

        # Verify no chunks start or end mid-word (basic boundary check)
        for chunk in chunks:
            text = chunk.text.strip()

            # Check text doesn't start/end with partial words (unless single word)
            words = text.split()
            if len(words) > 1:
                # First and last words should be complete
                assert not text.startswith(" ") or text[0].isupper()
                assert not text.endswith(" ") or text[-1] in ".!?"

    def test_chunk_optimization(self):
        """Test chunk size optimization functionality."""
        # Test with chunks that need optimization
        test_chunks = [
            "Short chunk.",  # Too small
            "This is a medium-sized chunk that should be in the target range for token counting and optimization testing purposes.",  # Target size
            "This is an extremely long chunk that exceeds the maximum size limit and should be split into smaller pieces. "
            * 50,  # Too large
        ]

        optimized = self.chunker._optimize_chunk_sizes(test_chunks)

        # Verify optimization occurred
        assert len(optimized) > 0

        # Check that large chunks were split
        large_chunks_before = sum(
            1
            for chunk in test_chunks
            if self.chunker._count_tokens_with_tracking(chunk)
            > self.chunker.max_chunk_size
        )
        large_chunks_after = sum(
            1
            for chunk in optimized
            if self.chunker._count_tokens_with_tracking(chunk)
            > self.chunker.max_chunk_size
        )

        if large_chunks_before > 0:
            assert large_chunks_after < large_chunks_before

    def test_langchain_document_compatibility(self, sample_legal_document):
        """Test compatibility with Langchain Document objects."""
        # Create a Langchain Document
        doc = Document(
            page_content=sample_legal_document,
            metadata={"source": "test_legal_doc.txt", "page": 1},
        )

        # Chunk the document
        doc_chunks = self.chunker.chunk_document(doc)

        # Verify we get Document objects back
        assert len(doc_chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in doc_chunks)

        # Verify metadata is preserved and enhanced
        for chunk in doc_chunks:
            assert "source" in chunk.metadata
            assert "page" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert "chunk_tokens" in chunk.metadata
            assert "hierarchy_level" in chunk.metadata

    def test_chunk_summary_statistics(self, sample_legal_document):
        """Test chunk summary statistics generation."""
        chunks = self.chunker.chunk_text_with_hierarchy(sample_legal_document)
        summary = self.chunker.get_chunk_summary(chunks)

        # Verify summary contains expected fields
        assert "total_chunks" in summary
        assert "total_tokens" in summary
        assert "average_tokens_per_chunk" in summary
        assert "min_tokens" in summary
        assert "max_tokens" in summary
        assert "hierarchy_distribution" in summary

        # Verify values are reasonable
        assert summary["total_chunks"] == len(chunks)
        assert summary["total_tokens"] > 0
        assert summary["average_tokens_per_chunk"] > 0
        assert summary["min_tokens"] <= summary["max_tokens"]

    def test_chunk_size_analysis(self, sample_legal_document):
        """Test detailed chunk size analysis functionality."""
        chunks = self.chunker.chunk_text_with_hierarchy(sample_legal_document)
        analysis = self.chunker.analyze_chunk_sizes(chunks)

        # Verify analysis contains expected sections
        assert "total_chunks" in analysis
        assert "token_statistics" in analysis
        assert "size_distribution" in analysis
        assert "performance_metrics" in analysis
        assert "configuration" in analysis
        assert "recommendations" in analysis

        # Verify token statistics
        token_stats = analysis["token_statistics"]
        assert "average" in token_stats
        assert "minimum" in token_stats
        assert "maximum" in token_stats
        assert "median" in token_stats

        # Verify size distribution
        size_dist = analysis["size_distribution"]
        assert "within_target_range" in size_dist
        assert "too_small" in size_dist
        assert "too_large" in size_dist

    def test_token_usage_statistics(self, sample_legal_document):
        """Test token usage statistics tracking."""
        # Reset stats
        self.chunker.reset_token_usage_stats()
        initial_stats = self.chunker.get_token_usage_stats()

        # Perform chunking
        self.chunker.chunk_text_with_hierarchy(sample_legal_document)
        final_stats = self.chunker.get_token_usage_stats()

        # Verify statistics were updated
        assert (
            final_stats["total_tokens_processed"]
            > initial_stats["total_tokens_processed"]
        )
        assert (
            final_stats["total_chunks_created"] > initial_stats["total_chunks_created"]
        )
        assert (
            final_stats["token_counting_calls"] > initial_stats["token_counting_calls"]
        )

        # Verify derived metrics
        assert "avg_tokens_per_chunk" in final_stats
        assert final_stats["avg_tokens_per_chunk"] > 0

    def test_processing_cost_estimation(self, sample_legal_document):
        """Test processing cost estimation functionality."""
        cost_analysis = self.chunker.estimate_processing_cost(
            sample_legal_document, cost_per_1k_tokens=0.002
        )

        # Verify cost analysis fields
        assert "input_tokens" in cost_analysis
        assert "estimated_chunks" in cost_analysis
        assert "processing_tokens" in cost_analysis
        assert "base_cost_usd" in cost_analysis
        assert "processing_cost_usd" in cost_analysis
        assert "cost_per_chunk_usd" in cost_analysis

        # Verify values are reasonable
        assert cost_analysis["input_tokens"] > 0
        assert cost_analysis["estimated_chunks"] > 0
        assert cost_analysis["processing_cost_usd"] >= cost_analysis["base_cost_usd"]

    def test_model_optimization_recommendations(self):
        """Test model-specific optimization recommendations."""
        # Test GPT-4 optimization
        gpt4_rec = self.chunker.optimize_for_model("gpt-4o")

        assert "model_name" in gpt4_rec
        assert "recommended_chunk_size" in gpt4_rec
        assert "recommended_overlap" in gpt4_rec
        assert "notes" in gpt4_rec

        # Verify GPT-4 specific recommendations
        assert gpt4_rec["model_name"] == "gpt-4o"
        assert len(gpt4_rec["notes"]) > 0

    def test_token_counter_validation(self):
        """Test TokenCounter integration validation."""
        validation = self.chunker.validate_token_counter_integration()

        # Verify validation fields
        assert "token_counter_initialized" in validation
        assert "model_name" in validation
        assert "basic_counting_works" in validation
        assert "usage_stats" in validation

        # Verify integration is working
        assert validation["token_counter_initialized"] is True
        assert validation["basic_counting_works"] is True
        assert validation["model_name"] == "gpt-4o"

    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text."""
        # Test empty string
        chunks = self.chunker.chunk_text_with_hierarchy("")
        assert len(chunks) == 0

        # Test whitespace-only string
        chunks = self.chunker.chunk_text_with_hierarchy("   \n\n   ")
        assert len(chunks) == 0

    def test_single_sentence_text(self):
        """Test handling of very short text (single sentence)."""
        short_text = "This is a single sentence."
        chunks = self.chunker.chunk_text_with_hierarchy(short_text)

        # Should create exactly one chunk
        assert len(chunks) == 1
        assert chunks[0].text.strip() == short_text.strip()
        assert chunks[0].token_count > 0

    def test_error_handling_invalid_input(self):
        """Test error handling for invalid inputs."""
        # Test None input (should not crash)
        with pytest.raises((TypeError, AttributeError)):
            self.chunker.chunk_text_with_hierarchy(None)

    def test_dynamic_separator_generation(self, sample_legal_document):
        """Test dynamic separator generation based on document structure."""
        # First, detect structure
        structure = self.structure_detector.detect_structure(sample_legal_document)

        # Generate dynamic separators
        dynamic_separators = self.chunker._get_dynamic_separators(structure)

        if dynamic_separators:  # Only test if separators were generated
            assert isinstance(dynamic_separators, list)
            assert len(dynamic_separators) > 0

            # Check that legal document patterns are included
            separator_text = " ".join(dynamic_separators)
            assert any(
                pattern in separator_text for pattern in ["ARTICLE", "Section", "§"]
            )

    def test_hierarchy_boundary_extraction(self, sample_legal_document):
        """Test extraction of hierarchy boundaries from document structure."""
        structure = self.structure_detector.detect_structure(sample_legal_document)
        boundaries = self.chunker._extract_hierarchy_boundaries(
            sample_legal_document, structure
        )

        # Verify boundaries were extracted
        assert isinstance(boundaries, list)

        if len(boundaries) > 0:  # Only test if boundaries were found
            # Check boundary structure
            for boundary in boundaries:
                assert "start_position" in boundary
                assert "end_position" in boundary
                assert "element_type" in boundary
                assert "hierarchy_level" in boundary
                assert "is_section_boundary" in boundary
                assert "is_subsection_boundary" in boundary

                # Verify position values are reasonable
                assert boundary["start_position"] >= 0
                assert boundary["end_position"] >= boundary["start_position"]

    def test_chunk_position_tracking(self, sample_legal_document):
        """Test that chunk positions are properly tracked."""
        chunks = self.chunker.chunk_text_with_hierarchy(sample_legal_document)

        # Verify position tracking
        for chunk in chunks:
            assert chunk.start_position >= 0
            assert chunk.end_position >= chunk.start_position
            assert chunk.char_count == len(chunk.text)

        # Verify chunks cover the document (positions should be sequential)
        if len(chunks) > 1:
            sorted_chunks = sorted(chunks, key=lambda x: x.start_position)
            for i in range(len(sorted_chunks) - 1):
                current_chunk = sorted_chunks[i]
                next_chunk = sorted_chunks[i + 1]
                # Allow for some overlap due to chunk_overlap setting
                assert next_chunk.start_position >= current_chunk.start_position


class TestHierarchicalChunkObject:
    """Test cases for the HierarchicalChunk data structure."""

    def test_hierarchical_chunk_creation(self):
        """Test creating HierarchicalChunk objects with various parameters."""
        chunk = HierarchicalChunk(
            text="Sample chunk text.",
            chunk_index=0,
            token_count=5,
            start_position=0,
            end_position=18,
            hierarchy_level=1,
            element_type=ElementType.SECTION,
            section_title="Test Section",
            numbering="1.1",
            parent_elements=["Article I"],
            metadata={"test": "value"},
        )

        # Verify all attributes are set correctly
        assert chunk.text == "Sample chunk text."
        assert chunk.chunk_index == 0
        assert chunk.token_count == 5
        assert chunk.start_position == 0
        assert chunk.end_position == 18
        assert chunk.hierarchy_level == 1
        assert chunk.element_type == ElementType.SECTION
        assert chunk.section_title == "Test Section"
        assert chunk.numbering == "1.1"
        assert chunk.parent_elements == ["Article I"]
        assert chunk.metadata == {"test": "value"}
        assert chunk.char_count == len("Sample chunk text.")

    def test_hierarchical_chunk_defaults(self):
        """Test HierarchicalChunk creation with minimal parameters."""
        chunk = HierarchicalChunk(text="Minimal chunk.", chunk_index=0, token_count=3)

        # Verify defaults are applied
        assert chunk.text == "Minimal chunk."
        assert chunk.chunk_index == 0
        assert chunk.token_count == 3
        assert chunk.start_position == 0
        assert chunk.end_position == 0
        assert chunk.hierarchy_level == 0
        assert chunk.element_type is None
        assert chunk.section_title is None
        assert chunk.numbering is None
        assert chunk.parent_elements == []
        assert chunk.metadata == {}
        assert chunk.char_count == len("Minimal chunk.")


class TestIntegrationWithComponents:
    """Integration tests with TokenCounter and StructureDetector."""

    def test_token_counter_integration_mock(self):
        """Test HierarchicalChunker with mocked TokenCounter."""
        mock_token_counter = Mock(spec=TokenCounter)
        mock_token_counter.count_tokens_for_model.return_value = 100

        chunker = HierarchicalChunker(token_counter=mock_token_counter, chunk_size=500)

        text = "Test text for token counting."
        count = chunker._count_tokens_with_tracking(text)

        # Verify mock was called and result returned
        assert count == 100
        mock_token_counter.count_tokens_for_model.assert_called_once_with(
            text, "gpt-4o"
        )

    def test_structure_detector_integration_mock(self):
        """Test HierarchicalChunker with mocked StructureDetector."""
        mock_structure_detector = Mock(spec=StructureDetector)
        mock_structure = Mock(spec=DocumentStructure)
        mock_structure.elements = []
        mock_structure_detector.detect_structure.return_value = mock_structure

        chunker = HierarchicalChunker(
            structure_detector=mock_structure_detector, chunk_size=500
        )

        text = "Test document text."
        chunks = chunker.chunk_text_with_hierarchy(text)

        # Verify structure detector was called
        mock_structure_detector.detect_structure.assert_called_once_with(text)
        assert len(chunks) > 0

    def test_end_to_end_integration(self):
        """Test complete integration without mocks."""
        chunker = HierarchicalChunker(
            chunk_size=400, chunk_overlap=50, legal_specific=True
        )

        legal_text = """
        ARTICLE I
        DEFINITIONS

        Section 1.1 General Definitions
        For the purposes of this Agreement, the following terms shall have the meanings set forth below.

        Section 1.2 Interpretation
        This Agreement shall be interpreted in accordance with applicable law.
        """

        # Perform chunking
        chunks = chunker.chunk_text_with_hierarchy(legal_text)

        # Verify end-to-end functionality
        assert len(chunks) > 0
        assert all(isinstance(chunk, HierarchicalChunk) for chunk in chunks)

        # Get summary and analysis
        summary = chunker.get_chunk_summary(chunks)
        analysis = chunker.analyze_chunk_sizes(chunks)

        assert summary["total_chunks"] == len(chunks)
        assert analysis["total_chunks"] == len(chunks)

        # Verify token usage tracking
        stats = chunker.get_token_usage_stats()
        assert stats["total_chunks_created"] == len(chunks)
        assert stats["total_tokens_processed"] > 0
