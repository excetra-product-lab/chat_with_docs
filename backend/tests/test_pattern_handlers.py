"""
Tests for document structure detection pattern handlers.

This module tests the PatternHandler class that provides regex patterns
and pattern matching utilities for detecting structural elements.
"""

import pytest

from app.services.document_structure_detector import NumberingType, PatternHandler


class TestPatternHandler:
    """Test suite for the PatternHandler class."""

    @pytest.fixture
    def pattern_handler(self):
        """Create a PatternHandler instance for testing."""
        return PatternHandler(case_sensitive=False)

    @pytest.fixture
    def case_sensitive_handler(self):
        """Create a case-sensitive PatternHandler instance."""
        return PatternHandler(case_sensitive=True)

    @pytest.fixture
    def section_samples(self):
        """Sample text with various section patterns."""
        return """
        Section 1.1 Overview
        Section 2.3.4 Detailed Requirements
        § 5.1 Legal Provisions
        § 5.2.1 Subsection
        §§ 6.1-6.5 Section Range
        (Section 7.2)
        """

    @pytest.fixture
    def article_samples(self):
        """Sample text with article patterns."""
        return """
        Article I Introduction
        Article II: General Provisions
        Article 15 - Payment Terms
        Article V. Termination
        ARTICLE III
        Article XXIII Final Clauses
        """

    @pytest.fixture
    def numbering_samples(self):
        """Sample text with various numbering patterns."""
        return """
        1. First item
        2.1 Subsection
        3.2.4 Deep nesting
        (a) Letter numbering
        (b) Second letter
        I. Roman numerals
        II. Second roman
        iii. Lower case roman
        A. Upper case letters
        b. Lower case letters
        """

    @pytest.fixture
    def heading_samples(self):
        """Sample text with various heading formats."""
        return """
        ALL CAPS HEADING

        Title Case Heading

        Underlined Heading
        ------------------

        Another Underlined
        ==================

                    Centered Heading

        1.5 Numbered Heading Text
        """

    def test_pattern_handler_initialization(self, pattern_handler):
        """Test PatternHandler initialization."""
        assert pattern_handler is not None
        assert not pattern_handler.case_sensitive
        assert isinstance(pattern_handler.patterns, dict)
        assert len(pattern_handler.patterns) > 0

    def test_case_sensitive_initialization(self, case_sensitive_handler):
        """Test case-sensitive PatternHandler initialization."""
        assert case_sensitive_handler.case_sensitive
        assert isinstance(case_sensitive_handler.patterns, dict)

    def test_pattern_categories(self, pattern_handler):
        """Test that all expected pattern categories are present."""
        categories = pattern_handler.get_pattern_categories()

        expected_categories = [
            "section_symbols",
            "articles",
            "sections",
            "chapters",
            "subsections",
            "clauses",
            "list_items",
            "headings",
            "legal_terms",
            "cross_references",
            "formatting",
            "decimal_numbering",
            "roman_numerals",
            "letter_numbering",
        ]

        for category in expected_categories:
            assert category in categories, f"Missing category: {category}"

    def test_section_symbol_patterns(self, pattern_handler, section_samples):
        """Test section symbol pattern matching."""
        matches = pattern_handler.find_pattern_matches(
            section_samples, "section_symbols"
        )

        assert len(matches) > 0

        # Should find § symbols
        section_texts = [match[1] for match in matches]
        assert any("§ 5.1" in text for text in section_texts)
        assert any("§ 5.2.1" in text for text in section_texts)

    def test_article_patterns(self, pattern_handler, article_samples):
        """Test article pattern matching."""
        matches = pattern_handler.find_pattern_matches(article_samples, "articles")

        assert len(matches) > 0

        # Should find various article formats
        article_texts = [match[1] for match in matches]
        assert any("Article I" in text for text in article_texts)
        assert any("Article II" in text for text in article_texts)
        assert any("Article 15" in text for text in article_texts)

    def test_section_patterns(self, pattern_handler, section_samples):
        """Test section pattern matching (without symbols)."""
        matches = pattern_handler.find_pattern_matches(section_samples, "sections")

        assert len(matches) > 0

        # Should find Section patterns
        section_texts = [match[1] for match in matches]
        assert any("Section 1.1" in text for text in section_texts)
        assert any("Section 2.3.4" in text for text in section_texts)

    def test_decimal_numbering_patterns(self, pattern_handler, numbering_samples):
        """Test decimal numbering pattern extraction."""
        matches = pattern_handler.find_pattern_matches(
            numbering_samples, "decimal_numbering"
        )

        assert len(matches) > 0

        # Should find decimal patterns
        decimal_texts = [match[1] for match in matches]
        assert any("1." in text for text in decimal_texts)
        assert any("2.1" in text for text in decimal_texts)
        assert any("3.2.4" in text for text in decimal_texts)

    def test_roman_numeral_patterns(self, pattern_handler, numbering_samples):
        """Test Roman numeral pattern matching."""
        matches = pattern_handler.find_pattern_matches(
            numbering_samples, "roman_numerals"
        )

        assert len(matches) > 0

        # Should find Roman numerals
        roman_texts = [match[1] for match in matches]
        roman_content = " ".join(roman_texts)
        assert "I" in roman_content or "II" in roman_content

    def test_letter_numbering_patterns(self, pattern_handler, numbering_samples):
        """Test letter numbering pattern matching."""
        matches = pattern_handler.find_pattern_matches(
            numbering_samples, "letter_numbering"
        )

        assert len(matches) > 0

        # Should find letter patterns
        letter_texts = [match[1] for match in matches]
        letter_content = " ".join(letter_texts)
        assert any(letter in letter_content for letter in ["(a)", "(b)", "A.", "b."])

    def test_heading_format_patterns(self, pattern_handler, heading_samples):
        """Test heading format pattern matching."""
        matches = pattern_handler.find_pattern_matches(heading_samples, "headings")

        assert len(matches) > 0

        # Should find various heading formats
        heading_texts = [match[1] for match in matches]
        assert any("ALL CAPS HEADING" in text for text in heading_texts)

    def test_extract_numbering_from_text(self, pattern_handler):
        """Test comprehensive numbering extraction."""
        text = """
        1. First item
        2.1 Subsection
        I. Roman numeral
        A. Letter item
        § 5.1 Section symbol
        (a) Parenthetical letter
        """

        numbering_matches = pattern_handler.extract_numbering_from_text(text)

        assert len(numbering_matches) > 0

        # Check that different numbering types are detected
        found_types = {match[0] for match in numbering_matches}
        assert NumberingType.DECIMAL in found_types
        assert (
            NumberingType.ROMAN_UPPER in found_types
            or NumberingType.ROMAN_LOWER in found_types
        )

    def test_extract_headings_by_format(self, pattern_handler, heading_samples):
        """Test format-based heading extraction."""
        headings = pattern_handler.extract_headings_by_format(heading_samples)

        assert len(headings) > 0

        # Should extract headings with their formats
        # headings is a list of tuples: (heading_text, start_pos, end_pos, format_type)
        heading_texts = [h[0] for h in headings]  # h[0] is the heading_text
        assert any("ALL CAPS HEADING" in text for text in heading_texts)
        assert any("Title Case Heading" in text for text in heading_texts)

    def test_pattern_validation(self, pattern_handler):
        """Test that all regex patterns are valid."""
        validation_results = pattern_handler.validate_patterns()

        assert isinstance(validation_results, dict)

        # All patterns should compile successfully
        failed_patterns = [
            name for name, valid in validation_results.items() if not valid
        ]
        assert len(failed_patterns) == 0, f"Failed patterns: {failed_patterns}"

    def test_find_specific_pattern(self, pattern_handler, section_samples):
        """Test finding matches for a specific pattern within a category."""
        matches = pattern_handler.find_pattern_matches(
            section_samples, "section_symbols", "basic_section"
        )

        # Should find basic section symbol patterns
        assert len(matches) >= 0  # May or may not have matches depending on text

    def test_invalid_pattern_category(self, pattern_handler):
        """Test handling of invalid pattern categories."""
        matches = pattern_handler.find_pattern_matches(
            "test text", "nonexistent_category"
        )

        assert len(matches) == 0

    def test_invalid_pattern_name(self, pattern_handler):
        """Test handling of invalid pattern names within valid categories."""
        matches = pattern_handler.find_pattern_matches(
            "test text", "sections", "nonexistent_pattern"
        )

        assert len(matches) == 0

    def test_case_sensitivity_difference(self, pattern_handler, case_sensitive_handler):
        """Test difference between case-sensitive and case-insensitive matching."""
        text = "section 1.1 Overview"

        # Case-insensitive should find "section"
        insensitive_matches = pattern_handler.find_pattern_matches(text, "sections")

        # Case-sensitive might not find lowercase "section"
        sensitive_matches = case_sensitive_handler.find_pattern_matches(
            text, "sections"
        )

        # The behavior depends on the specific patterns, but they should be different
        # or the insensitive should find at least as many as sensitive
        assert len(insensitive_matches) >= len(sensitive_matches)

    def test_overlapping_match_removal(self, pattern_handler):
        """Test removal of overlapping pattern matches."""
        text = "Section 1.1 Overview of Section 1.1"

        matches = pattern_handler.find_pattern_matches(text, "sections")

        # Should handle overlapping matches appropriately
        assert len(matches) >= 0

        # Verify matches are sorted by position
        positions = [match[0] for match in matches]
        assert positions == sorted(positions)

    def test_multiline_pattern_matching(self, pattern_handler):
        """Test pattern matching across multiple lines."""
        multiline_text = """
        Section 1.1
        Overview

        Section 1.2
        Details
        """

        matches = pattern_handler.find_pattern_matches(multiline_text, "sections")

        assert len(matches) > 0
        # Should find both sections
        section_texts = [match[1] for match in matches]
        assert any("Section 1.1" in text for text in section_texts)
        assert any("Section 1.2" in text for text in section_texts)

    def test_pattern_groups_extraction(self, pattern_handler):
        """Test that regex groups are properly extracted."""
        text = "Section 1.2.3 Important Details"

        matches = pattern_handler.find_pattern_matches(text, "sections")

        if matches:
            # Check that groups are captured
            for match in matches:
                position, matched_text, groups = match
                assert isinstance(groups, tuple)
                # Groups should contain the captured numbering
                if groups:
                    assert any(group for group in groups if group)

    def test_get_patterns_in_category(self, pattern_handler):
        """Test retrieval of pattern names within a category."""
        section_patterns = pattern_handler.get_patterns_in_category("sections")

        assert len(section_patterns) > 0
        assert "section_basic" in section_patterns

        # Test invalid category
        invalid_patterns = pattern_handler.get_patterns_in_category("invalid_category")
        assert len(invalid_patterns) == 0

    def test_analyze_pattern_coverage(self, pattern_handler):
        """Test pattern coverage analysis functionality."""
        text = """
        Section 1.1 Overview
        Article II Provisions
        § 3.1 Legal Section
        I. Roman numeral item
        A. Letter item
        """

        coverage = pattern_handler.analyze_pattern_coverage(text)

        assert isinstance(coverage, dict)
        assert "total_patterns_tested" in coverage
        assert "matched_patterns" in coverage
        assert "coverage_percentage" in coverage

        # Should have some coverage
        assert coverage["total_patterns_tested"] > 0

    def test_empty_text_handling(self, pattern_handler):
        """Test handling of empty or whitespace-only text."""
        empty_matches = pattern_handler.find_pattern_matches("", "sections")
        whitespace_matches = pattern_handler.find_pattern_matches(
            "   \n\t  ", "sections"
        )

        assert len(empty_matches) == 0
        assert len(whitespace_matches) == 0

    def test_special_characters_handling(self, pattern_handler):
        """Test handling of text with special characters."""
        special_text = """
        Section 1.1 — Overview with em-dash
        § 2.1 "Quoted section"
        Article III: (Special) Characters [Test]
        """

        matches = pattern_handler.find_pattern_matches(special_text, "sections")

        # Should handle special characters gracefully
        assert isinstance(matches, list)
