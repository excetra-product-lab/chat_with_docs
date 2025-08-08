"""Pattern handlers for document structure detection.

This module provides regex patterns and pattern matching utilities
for detecting structural elements in legal documents.
"""

import logging
import re

from .data_models import NumberingType

logger = logging.getLogger(__name__)


class PatternHandler:
    """Handler for regex patterns used in document structure detection."""

    def __init__(self, case_sensitive: bool = False):
        """Initialize the pattern handler.

        Args:
            case_sensitive: Whether patterns should be case sensitive
        """
        self.case_sensitive = case_sensitive
        self.logger = logging.getLogger(__name__)
        self.patterns = self._initialize_regex_patterns()

    def _initialize_regex_patterns(self) -> dict[str, dict[str, re.Pattern]]:
        """Initialize comprehensive regex patterns for legal document structure detection.

        Returns:
            Dict containing categorized compiled regex patterns
        """
        flags = 0 if self.case_sensitive else re.IGNORECASE

        patterns = {
            # Section symbol patterns (§)
            "section_symbols": {
                "basic_section": re.compile(r"§\s*(\d+(?:\.\d+)*)", flags),
                "section_with_text": re.compile(
                    r"§\s*(\d+(?:\.\d+)*)\s*([^\n\r]*)", flags
                ),
                "section_range": re.compile(
                    r"§§\s*(\d+(?:\.\d+)*)\s*-\s*(\d+(?:\.\d+)*)", flags
                ),
                "section_list": re.compile(
                    r"§\s*(\d+(?:\.\d+)*(?:\s*,\s*\d+(?:\.\d+)*)*)", flags
                ),
            },
            # Article patterns
            "articles": {
                "article_basic": re.compile(
                    r"\b[Aa]rticle\s+([IVXLCDM]+|\d+(?:\.\d+)*)\b", flags
                ),
                "article_with_title": re.compile(
                    r"\b[Aa]rticle\s+([IVXLCDM]+|\d+(?:\.\d+)*)\s*[:\-–—]\s*([^\n\r]*)",
                    flags,
                ),
                "article_standalone": re.compile(
                    r"^\s*[Aa]rticle\s+([IVXLCDM]+|\d+(?:\.\d+)*)\s*$",
                    flags | re.MULTILINE,
                ),
                "article_period": re.compile(
                    r"\b[Aa]rticle\s+([IVXLCDM]+|\d+(?:\.\d+)*)\.\s*([^\n\r]*)", flags
                ),
            },
            # Section patterns (without symbol)
            "sections": {
                "section_basic": re.compile(r"\b[Ss]ection\s+(\d+(?:\.\d+)*)\b", flags),
                "section_with_title": re.compile(
                    r"\b[Ss]ection\s+(\d+(?:\.\d+)*)\s*[:\-–—]\s*([^\n\r]*)", flags
                ),
                "section_space_title": re.compile(
                    r"\b[Ss]ection\s+(\d+(?:\.\d+)*)\s+([^\n\r]+)", flags
                ),
                "section_period": re.compile(
                    r"\b[Ss]ection\s+(\d+(?:\.\d+)*)\.\s*([^\n\r]*)", flags
                ),
                "section_parentheses": re.compile(
                    r"\([Ss]ection\s+(\d+(?:\.\d+)*)\)", flags
                ),
            },
            # Chapter patterns
            "chapters": {
                "chapter_basic": re.compile(
                    r"\b[Cc]hapter\s+([IVXLCDM]+|\d+(?:\.\d+)*)\b", flags
                ),
                "chapter_with_title": re.compile(
                    r"\b[Cc]hapter\s+([IVXLCDM]+|\d+(?:\.\d+)*)\s*[:\-–—]\s*([^\n\r]*)",
                    flags,
                ),
                "chapter_period": re.compile(
                    r"\b[Cc]hapter\s+([IVXLCDM]+|\d+(?:\.\d+)*)\.\s*([^\n\r]*)", flags
                ),
            },
            # Subsection patterns
            "subsections": {
                "subsection_basic": re.compile(
                    r"\b[Ss]ubsection\s+(\d+(?:\.\d+)*)\b", flags
                ),
                "subsection_with_title": re.compile(
                    r"\b[Ss]ubsection\s+(\d+(?:\.\d+)*)\s*[:\-–—]\s*([^\n\r]*)", flags
                ),
                "subsection_parentheses": re.compile(
                    r"\([Ss]ubsection\s+(\d+(?:\.\d+)*)\)", flags
                ),
            },
            # Decimal numbering patterns (1.2.3, 2.1, etc.)
            "decimal_numbering": {
                "standalone_decimal": re.compile(
                    r"^\s*(\d+(?:\.\d+)*)\s*$", flags | re.MULTILINE
                ),
                "decimal_with_text": re.compile(
                    r"^\s*(\d+(?:\.\d+)*)\s+([^\n\r]+)", flags | re.MULTILINE
                ),
                "decimal_period": re.compile(
                    r"^\s*(\d+(?:\.\d+)*)\.\s*([^\n\r]*)", flags | re.MULTILINE
                ),
                "decimal_parentheses": re.compile(
                    r"^\s*(\d+(?:\.\d+)*)\)\s*([^\n\r]*)", flags | re.MULTILINE
                ),
                "decimal_bracket": re.compile(
                    r"^\s*\((\d+(?:\.\d+)*)\)\s*([^\n\r]*)", flags | re.MULTILINE
                ),
            },
            # Roman numeral patterns
            "roman_numerals": {
                "roman_upper": re.compile(
                    r"^\s*([IVXLCDM]+)\s*\.?\s*([^\n\r]*)", flags | re.MULTILINE
                ),
                "roman_lower": re.compile(
                    r"^\s*([ivxlcdm]+)\s*\.?\s*([^\n\r]*)", flags | re.MULTILINE
                ),
                "roman_parentheses": re.compile(
                    r"^\s*\(([IVXLCDM]+|[ivxlcdm]+)\)\s*([^\n\r]*)",
                    flags | re.MULTILINE,
                ),
            },
            # Letter-based numbering
            "letter_numbering": {
                "letter_upper": re.compile(
                    r"^\s*([A-Z])\s*\.?\s*([^\n\r]*)", flags | re.MULTILINE
                ),
                "letter_lower": re.compile(
                    r"^\s*([a-z])\s*\.?\s*([^\n\r]*)", flags | re.MULTILINE
                ),
                "letter_parentheses": re.compile(
                    r"^\s*\(([A-Za-z])\)\s*([^\n\r]*)", flags | re.MULTILINE
                ),
            },
            # Clause and paragraph patterns
            "clauses": {
                "clause_basic": re.compile(r"\b[Cc]lause\s+(\d+(?:\.\d+)*)\b", flags),
                "clause_with_title": re.compile(
                    r"\b[Cc]lause\s+(\d+(?:\.\d+)*)\s*[:\-–—]\s*([^\n\r]*)", flags
                ),
                "paragraph_basic": re.compile(
                    r"\b[Pp]aragraph\s+(\d+(?:\.\d+)*)\b", flags
                ),
                "paragraph_with_title": re.compile(
                    r"\b[Pp]aragraph\s+(\d+(?:\.\d+)*)\s*[:\-–—]\s*([^\n\r]*)", flags
                ),
            },
            # List item patterns
            "list_items": {
                "bullet_dash": re.compile(
                    r"^\s*[-–—]\s*([^\n\r]+)", flags | re.MULTILINE
                ),
                "bullet_asterisk": re.compile(
                    r"^\s*\*\s*([^\n\r]+)", flags | re.MULTILINE
                ),
                "bullet_dot": re.compile(r"^\s*•\s*([^\n\r]+)", flags | re.MULTILINE),
                "numbered_list": re.compile(
                    r"^\s*(\d+)\.\s*([^\n\r]+)", flags | re.MULTILINE
                ),
                "lettered_list": re.compile(
                    r"^\s*([a-zA-Z])\.\s*([^\n\r]+)", flags | re.MULTILINE
                ),
            },
            # Heading patterns (based on formatting)
            "headings": {
                "all_caps": re.compile(
                    r"^\s*([A-Z][A-Z\s]{2,}[A-Z])\s*$", re.MULTILINE
                ),
                "title_case": re.compile(
                    r"^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$", flags | re.MULTILINE
                ),
                "underlined": re.compile(
                    r"^([^\n\r]+)\n[-=_]{3,}", flags | re.MULTILINE
                ),
                "numbered_heading": re.compile(
                    r"^\s*(\d+(?:\.\d+)*)\s+([A-Z][^\n\r]*)", flags | re.MULTILINE
                ),
                "centered": re.compile(
                    r"^\s{10,}([^\n\r]+)\s{10,}$", flags | re.MULTILINE
                ),
            },
            # Special legal terms and patterns
            "legal_terms": {
                "whereas": re.compile(r"\b[Ww][Hh][Ee][Rr][Ee][Aa][Ss]\b", flags),
                "therefore": re.compile(
                    r"\b[Tt][Hh][Ee][Rr][Ee][Ff][Oo][Rr][Ee]\b", flags
                ),
                "provided_that": re.compile(r"\b[Pp]rovided\s+that\b", flags),
                "subject_to": re.compile(r"\b[Ss]ubject\s+to\b", flags),
                "notwithstanding": re.compile(r"\b[Nn]otwithstanding\b", flags),
                "in_accordance_with": re.compile(
                    r"\b[Ii]n\s+accordance\s+with\b", flags
                ),
            },
            # Cross-reference patterns
            "cross_references": {
                "see_section": re.compile(
                    r"\b[Ss]ee\s+[Ss]ection\s+(\d+(?:\.\d+)*)", flags
                ),
                "see_article": re.compile(
                    r"\b[Ss]ee\s+[Aa]rticle\s+([IVXLCDM]+|\d+(?:\.\d+)*)", flags
                ),
                "pursuant_to": re.compile(
                    r"\b[Pp]ursuant\s+to\s+[Ss]ection\s+(\d+(?:\.\d+)*)", flags
                ),
                "in_section": re.compile(
                    r"\b[Ii]n\s+[Ss]ection\s+(\d+(?:\.\d+)*)", flags
                ),
                "under_section": re.compile(
                    r"\b[Uu]nder\s+[Ss]ection\s+(\d+(?:\.\d+)*)", flags
                ),
            },
            # Formatting and structure indicators
            "formatting": {
                "double_newline": re.compile(r"\n\s*\n", flags),
                "indentation": re.compile(r"^\s{4,}([^\n\r]+)", flags | re.MULTILINE),
                "tab_indented": re.compile(r"^\t+([^\n\r]+)", flags | re.MULTILINE),
                "line_breaks": re.compile(r"\n+", flags),
                "whitespace_only": re.compile(r"^\s*$", flags | re.MULTILINE),
            },
        }

        # Compile and validate all patterns
        compiled_count = 0
        for category, category_patterns in patterns.items():
            for pattern_name, pattern in category_patterns.items():
                if isinstance(pattern, re.Pattern):
                    compiled_count += 1

        self.logger.info(
            "Initialized %d regex patterns across %d categories",
            compiled_count,
            len(patterns),
        )

        return patterns

    def find_pattern_matches(
        self, text: str, category: str, pattern_name: str = None
    ) -> list[tuple[int, str, tuple]]:
        """Find all matches for a specific pattern category or individual pattern.

        Args:
            text: Input text to search
            category: Pattern category (e.g., 'sections', 'articles')
            pattern_name: Specific pattern name within category (optional)

        Returns:
            List of tuples: (start_position, matched_text, groups)
        """
        matches = []

        if category not in self.patterns:
            self.logger.warning("Pattern category '%s' not found", category)
            return matches

        patterns_to_search = {}
        if pattern_name:
            if pattern_name in self.patterns[category]:
                patterns_to_search[pattern_name] = self.patterns[category][pattern_name]
            else:
                self.logger.warning(
                    "Pattern '%s' not found in category '%s'", pattern_name, category
                )
                return matches
        else:
            patterns_to_search = self.patterns[category]

        for name, pattern in patterns_to_search.items():
            for match in pattern.finditer(text):
                matches.append((match.start(), match.group(0), match.groups()))

        # Sort by position in text
        matches.sort(key=lambda x: x[0])
        return matches

    def extract_numbering_from_text(
        self, text: str
    ) -> list[tuple[NumberingType, str, int, int]]:
        """Extract all numbering patterns from text.

        Args:
            text: Input text to analyze

        Returns:
            List of tuples: (numbering_type, value, start_pos, end_pos)
        """
        numbering_results = []

        # Check decimal numbering
        for match in self.patterns["decimal_numbering"]["decimal_period"].finditer(
            text
        ):
            numbering_results.append(
                (NumberingType.DECIMAL, match.group(1), match.start(), match.end())
            )

        # Check Roman numerals (uppercase)
        for match in self.patterns["roman_numerals"]["roman_upper"].finditer(text):
            if self._is_valid_roman_numeral(match.group(1)):
                numbering_results.append(
                    (
                        NumberingType.ROMAN_UPPER,
                        match.group(1),
                        match.start(),
                        match.end(),
                    )
                )

        # Check Roman numerals (lowercase)
        for match in self.patterns["roman_numerals"]["roman_lower"].finditer(text):
            if self._is_valid_roman_numeral(match.group(1).upper()):
                numbering_results.append(
                    (
                        NumberingType.ROMAN_LOWER,
                        match.group(1),
                        match.start(),
                        match.end(),
                    )
                )

        # Check letter numbering (uppercase)
        for match in self.patterns["letter_numbering"]["letter_upper"].finditer(text):
            # Only single letters to avoid false positives
            if len(match.group(1)) == 1:
                numbering_results.append(
                    (
                        NumberingType.LETTER_UPPER,
                        match.group(1),
                        match.start(),
                        match.end(),
                    )
                )

        # Check letter numbering (lowercase)
        for match in self.patterns["letter_numbering"]["letter_lower"].finditer(text):
            # Only single letters to avoid false positives
            if len(match.group(1)) == 1:
                numbering_results.append(
                    (
                        NumberingType.LETTER_LOWER,
                        match.group(1),
                        match.start(),
                        match.end(),
                    )
                )

        # Check section symbols
        for match in self.patterns["section_symbols"]["basic_section"].finditer(text):
            numbering_results.append(
                (
                    NumberingType.SECTION_SYMBOL,
                    match.group(1),
                    match.start(),
                    match.end(),
                )
            )

        # Sort by position and return
        numbering_results.sort(key=lambda x: x[2])
        return numbering_results

    def _is_valid_roman_numeral(self, text: str) -> bool:
        """Validate if a string is a valid Roman numeral.

        Args:
            text: String to validate

        Returns:
            bool: True if valid Roman numeral
        """
        if not text:
            return False

        # Simple validation - check if it only contains valid Roman numeral characters
        valid_chars = set("IVXLCDM")
        return all(char in valid_chars for char in text.upper())

    def extract_headings_by_format(self, text: str) -> list[tuple[str, int, int, str]]:
        """Extract headings based on formatting patterns.

        Args:
            text: Input text to analyze

        Returns:
            List of tuples: (heading_text, start_pos, end_pos, format_type)
        """
        headings = []

        # All caps headings
        for match in self.patterns["headings"]["all_caps"].finditer(text):
            headings.append(
                (match.group(1).strip(), match.start(), match.end(), "all_caps")
            )

        # Title case headings
        for match in self.patterns["headings"]["title_case"].finditer(text):
            heading_text = match.group(1).strip()
            # Filter out common false positives
            if len(heading_text.split()) >= 2 and not heading_text.endswith("."):
                headings.append(
                    (heading_text, match.start(), match.end(), "title_case")
                )

        # Underlined headings
        for match in self.patterns["headings"]["underlined"].finditer(text):
            headings.append(
                (match.group(1).strip(), match.start(), match.end(), "underlined")
            )

        # Numbered headings
        for match in self.patterns["headings"]["numbered_heading"].finditer(text):
            headings.append(
                (
                    f"{match.group(1)} {match.group(2)}".strip(),
                    match.start(),
                    match.end(),
                    "numbered",
                )
            )

        # Centered headings
        for match in self.patterns["headings"]["centered"].finditer(text):
            heading_text = match.group(1).strip()
            if len(heading_text) > 5:  # Avoid short false positives
                headings.append((heading_text, match.start(), match.end(), "centered"))

        # Sort by position and remove duplicates
        headings.sort(key=lambda x: x[1])
        return self._remove_overlapping_matches(headings)

    def _remove_overlapping_matches(self, matches: list[tuple]) -> list[tuple]:
        """Remove overlapping matches, keeping the longest ones.

        Args:
            matches: List of match tuples with (text, start, end, ...)

        Returns:
            List of non-overlapping matches
        """
        if not matches:
            return []

        # Sort by start position, then by length (descending)
        sorted_matches = sorted(matches, key=lambda x: (x[1], -(x[2] - x[1])))

        result = []
        for match in sorted_matches:
            # Check if this match overlaps with any already selected match
            overlaps = False
            for existing in result:
                if match[1] < existing[2] and match[2] > existing[1]:  # Overlap check
                    overlaps = True
                    break

            if not overlaps:
                result.append(match)

        return sorted(result, key=lambda x: x[1])  # Sort by position

    def get_pattern_categories(self) -> list[str]:
        """Get list of available pattern categories.

        Returns:
            List of pattern category names
        """
        return list(self.patterns.keys())

    def get_patterns_in_category(self, category: str) -> list[str]:
        """Get list of pattern names in a specific category.

        Args:
            category: Pattern category name

        Returns:
            List of pattern names in the category
        """
        if category in self.patterns:
            return list(self.patterns[category].keys())
        return []

    def validate_patterns(self) -> dict[str, bool]:
        """Validate all compiled regex patterns.

        Returns:
            Dict mapping pattern names to validation status
        """
        validation_results = {}

        for category, category_patterns in self.patterns.items():
            for pattern_name, pattern in category_patterns.items():
                try:
                    # Test pattern with empty string
                    pattern.search("")
                    validation_results[f"{category}.{pattern_name}"] = True
                except Exception as e:
                    self.logger.error(
                        "Pattern validation failed for %s.%s: %s",
                        category,
                        pattern_name,
                        e,
                    )
                    validation_results[f"{category}.{pattern_name}"] = False

        return validation_results

    def analyze_pattern_coverage(self, text: str) -> dict[str, int]:
        """Analyze which patterns matched and how often.

        Args:
            text: Input text to analyze

        Returns:
            Dict with coverage statistics including category counts and totals
        """
        coverage = {}
        total_patterns = 0
        matched_patterns = 0

        for category in self.patterns.keys():
            matches = self.find_pattern_matches(text, category)
            match_count = len(matches)
            coverage[category] = match_count

            # Count total patterns in this category
            category_pattern_count = len(self.patterns[category])
            total_patterns += category_pattern_count

            # Count if this category had matches
            if match_count > 0:
                matched_patterns += category_pattern_count

        # Add summary statistics
        coverage["total_patterns_tested"] = total_patterns
        coverage["matched_patterns"] = matched_patterns
        coverage["coverage_percentage"] = (
            (matched_patterns / total_patterns * 100) if total_patterns > 0 else 0
        )

        return coverage
