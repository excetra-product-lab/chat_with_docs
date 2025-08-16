"""
Comprehensive tests for document structure detection module.

This module provides integration tests for the complete document structure detection
system, testing various legal document formats, numbering schemes, and edge cases.
"""

import pytest

from app.services.document_structure_detector import (
    DocumentStructure,
    ElementType,
    NumberingType,
    StructureDetector,
)


class TestDocumentStructureDetector:
    """Test suite for the StructureDetector class integration."""

    @pytest.fixture
    def detector(self):
        """Create a StructureDetector instance for testing."""
        return StructureDetector()

    @pytest.fixture
    def detector_with_config(self):
        """Create a StructureDetector with custom configuration."""
        config = {
            "min_heading_length": 2,
            "max_heading_length": 150,
            "case_sensitive": False,
            "min_confidence_threshold": 0.4,
        }
        return StructureDetector(config=config)

    @pytest.fixture
    def sample_legal_document(self):
        """Sample legal document with various structural elements."""
        return """
EMPLOYMENT AGREEMENT

ARTICLE I
GENERAL PROVISIONS

Section 1.1 Purpose
This agreement establishes the terms of employment.

Section 1.2 Definitions
For purposes of this agreement:
(a) "Company" means ABC Corporation
(b) "Employee" means John Doe

ARTICLE II
COMPENSATION AND BENEFITS

Section 2.1 Base Salary
§ 2.1.1 Amount
The base salary shall be $100,000 annually.

§ 2.1.2 Payment Terms
Salary shall be paid bi-weekly.

Section 2.2 Benefits
I. Health Insurance
II. Dental Coverage
III. Retirement Plan

CHAPTER 3
TERMINATION PROVISIONS

3.1 Voluntary Termination
Employee may terminate employment with two weeks notice.

3.2 Involuntary Termination
Company may terminate for cause.
"""

    @pytest.fixture
    def complex_numbering_document(self):
        """Document with complex and mixed numbering systems."""
        return """
CONTRACT FOR SERVICES

I. GENERAL TERMS

    A. Scope of Work
        1. Initial consultation
        2. Project planning
            a) Resource allocation
            b) Timeline development

    B. Deliverables
        1. Phase One
            (i) Requirements analysis
            (ii) Design specifications
        2. Phase Two
            (i) Implementation
            (ii) Testing

II. FINANCIAL TERMS

    A. Payment Schedule
        1. 50% upon signing
        2. 25% at midpoint
        3. 25% upon completion

§ 5 WARRANTIES

§ 5.1 Performance Warranties
Company warrants all work will meet specifications.

§ 5.2 Limitation of Liability
Liability shall not exceed contract value.
"""

    @pytest.fixture
    def formatted_headings_document(self):
        """Document with various heading formats."""
        return """
                            MASTER SERVICE AGREEMENT
                            ========================

DEFINITIONS AND INTERPRETATIONS
-------------------------------

This agreement contains the following defined terms:

        GENERAL PROVISIONS

1. SCOPE OF SERVICES

   The Provider shall deliver services as outlined.

2. TERM AND RENEWAL

   2.1 Initial Term
   This agreement shall remain in effect for one year.

   2.2 Renewal Options
   The agreement may be renewed by mutual consent.

PAYMENT TERMS
=============

All payments are due within thirty (30) days.

                               FINAL PROVISIONS
                               ================

This agreement constitutes the entire agreement.
"""

    def test_structure_detector_initialization(self, detector):
        """Test that StructureDetector initializes correctly."""
        assert detector is not None
        assert detector.pattern_handler is not None
        assert detector.numbering_handler is not None
        assert detector.heading_detector is not None

    def test_structure_detector_with_config(self, detector_with_config):
        """Test StructureDetector initialization with custom configuration."""
        assert detector_with_config.min_heading_length == 2
        assert detector_with_config.max_heading_length == 150
        assert not detector_with_config.case_sensitive

    def test_detect_structure_basic_document(self, detector, sample_legal_document):
        """Test structure detection on a basic legal document."""
        structure = detector.detect_structure(sample_legal_document)

        assert isinstance(structure, DocumentStructure)
        assert len(structure.headings) > 0
        assert structure.metadata["analysis_completed"] is True
        assert structure.metadata["input_length"] == len(sample_legal_document)

    def test_detect_headings_article_sections(self, detector, sample_legal_document):
        """Test detection of articles and sections."""
        headings = detector.detect_headings(sample_legal_document)

        # Should detect major structural elements
        heading_texts = [h.text for h in headings]

        # Check for major headings
        assert any("EMPLOYMENT AGREEMENT" in text for text in heading_texts)
        assert any("ARTICLE I" in text for text in heading_texts)
        assert any("ARTICLE II" in text for text in heading_texts)

        # Check for sections
        section_headings = [
            h for h in headings if h.element_type == ElementType.SECTION
        ]
        assert len(section_headings) > 0

    def test_complex_numbering_detection(self, detector, complex_numbering_document):
        """Test detection of complex and mixed numbering systems."""
        numbering_systems = detector.parse_all_numbering_systems(
            complex_numbering_document
        )

        assert len(numbering_systems) > 0

        # Check for different numbering types
        numbering_types = [ns.numbering_type for ns in numbering_systems]
        assert NumberingType.ROMAN_UPPER in numbering_types  # I, II
        assert NumberingType.LETTER_UPPER in numbering_types  # A, B
        assert NumberingType.DECIMAL in numbering_types  # 1, 2

    def test_formatted_headings_detection(self, detector, formatted_headings_document):
        """Test detection of various heading formats."""
        headings = detector.detect_headings(formatted_headings_document)

        heading_texts = [h.text.strip() for h in headings]

        # Should detect centered heading
        assert any("MASTER SERVICE AGREEMENT" in text for text in heading_texts)

        # Should detect underlined headings
        assert any("DEFINITIONS AND INTERPRETATIONS" in text for text in heading_texts)

        # Should detect all caps headings
        assert any("GENERAL PROVISIONS" in text for text in heading_texts)
        assert any("PAYMENT TERMS" in text for text in heading_texts)

    def test_numbering_analysis(self, detector, complex_numbering_document):
        """Test detailed numbering pattern analysis."""
        analysis = detector.analyze_numbering_patterns(complex_numbering_document)

        assert "total_numbering_systems" in analysis
        assert "numbering_types" in analysis
        assert "level_distribution" in analysis
        assert "hierarchy_depth" in analysis
        assert analysis["total_numbering_systems"] > 0
        assert analysis["hierarchy_depth"] > 1  # Should have multi-level hierarchy

    def test_empty_document(self, detector):
        """Test handling of empty document."""
        structure = detector.detect_structure("")

        assert isinstance(structure, DocumentStructure)
        assert len(structure.headings) == 0
        assert len(structure.elements) == 0
        assert structure.metadata["input_length"] == 0

    def test_single_heading_document(self, detector):
        """Test document with only one heading."""
        simple_doc = "TITLE\n\nSome content here."

        headings = detector.detect_headings(simple_doc)

        assert len(headings) >= 1
        title_heading = next((h for h in headings if "TITLE" in h.text), None)
        assert title_heading is not None

    def test_no_structure_document(self, detector):
        """Test document with no clear structure."""
        plain_text = """
        This is just plain text without any clear structure.
        It has multiple sentences and paragraphs but no headings
        or numbering systems that would indicate document structure.

        This paragraph continues the plain text format.
        There are no sections, articles, or numbered items.
        """

        structure = detector.detect_structure(plain_text)

        # Should handle gracefully even with no structure
        assert isinstance(structure, DocumentStructure)
        assert structure.metadata["analysis_completed"] is True

    def test_malformed_numbering(self, detector):
        """Test handling of malformed numbering systems."""
        malformed_doc = """
        Section 1.1.1.1.1.1.1.1 Too Deep
        Section 999999999999999 Too Large
        Section A.B.C.D.E.F Invalid Mixed
        § Invalid Section Symbol
        """

        # Should not crash on malformed input
        structure = detector.detect_structure(malformed_doc)
        assert isinstance(structure, DocumentStructure)

    def test_get_supported_types(self, detector):
        """Test retrieval of supported element and numbering types."""
        element_types = detector.get_supported_element_types()
        numbering_types = detector.get_supported_numbering_types()

        assert len(element_types) > 0
        assert len(numbering_types) > 0
        assert ElementType.HEADING in element_types
        assert NumberingType.DECIMAL in numbering_types

    def test_pattern_categories(self, detector):
        """Test retrieval of pattern categories."""
        categories = detector.get_pattern_categories()

        assert len(categories) > 0
        assert "sections" in categories
        assert "articles" in categories
        assert "headings" in categories

    def test_pattern_validation(self, detector):
        """Test regex pattern validation."""
        validation_results = detector.validate_patterns()

        assert isinstance(validation_results, dict)
        # All patterns should be valid
        assert all(validation_results.values()), (
            f"Invalid patterns: {validation_results}"
        )

    def test_hierarchical_structure(self, detector, sample_legal_document):
        """Test that hierarchical relationships are properly established."""
        structure = detector.detect_structure(sample_legal_document)

        # Check that elements have proper hierarchy
        elements = structure.elements
        if elements:
            # Should have different levels
            levels = {elem.level for elem in elements}
            assert len(levels) > 1, "Should have multiple hierarchy levels"

    def test_metadata_preservation(self, detector, sample_legal_document):
        """Test that metadata is properly preserved throughout processing."""
        structure = detector.detect_structure(sample_legal_document)

        # Check that headings have proper metadata
        for heading in structure.headings:
            assert heading.line_number >= 0
            assert heading.start_position >= 0
            assert heading.end_position > heading.start_position

    def test_performance_large_document(self, detector):
        """Test performance with a larger document."""
        # Create a larger document
        large_doc = """
LARGE DOCUMENT

CHAPTER 1
INTRODUCTION

Section 1.1 Overview
""" + "\n".join(
            [
                f"Paragraph {i}: This is content for paragraph {i}."
                for i in range(1, 1000)
            ]
        )

        large_doc += """

CHAPTER 2
DETAILED PROVISIONS

""" + "\n".join([f"Section 2.{i} Detail {i}" for i in range(1, 100)])

        # Should complete in reasonable time
        structure = detector.detect_structure(large_doc)
        assert isinstance(structure, DocumentStructure)
        assert structure.metadata["analysis_completed"] is True
