"""Heading detection logic for document structure detection.

This module provides the HeadingDetector class that integrates regex patterns
and numbering logic to reliably detect and classify headings and section
boundaries within legal documents.
"""

import logging
from dataclasses import dataclass

from .data_models import (
    ElementType,
    Heading,
    NumberingSystem,
    NumberingType,
)
from .numbering_systems import NumberingSystemHandler
from .pattern_handlers import PatternHandler

logger = logging.getLogger(__name__)


@dataclass
class HeadingCandidate:
    """Represents a potential heading found in the text."""

    text: str
    line_number: int
    start_position: int
    end_position: int
    confidence_score: float
    heading_type: ElementType
    numbering: NumberingSystem | None = None
    format_indicators: list[str] = None

    def __post_init__(self):
        if self.format_indicators is None:
            self.format_indicators = []


class HeadingDetector:
    """Handler for detecting and classifying headings in legal documents.

    This class integrates regex patterns and numbering logic to reliably
    detect headings, sections, and other structural boundaries with high
    accuracy and extensibility for various formatting styles.
    """

    def __init__(
        self,
        pattern_handler: PatternHandler,
        numbering_handler: NumberingSystemHandler,
        config: dict | None = None,
    ):
        """Initialize the heading detector.

        Args:
            pattern_handler: Instance of PatternHandler for regex patterns
            numbering_handler: Instance of NumberingSystemHandler for numbering logic
            config: Optional configuration for detection parameters
        """
        self.pattern_handler = pattern_handler
        self.numbering_handler = numbering_handler
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration parameters
        self.min_heading_length = self.config.get("min_heading_length", 1)
        self.max_heading_length = self.config.get("max_heading_length", 200)
        self.min_confidence_threshold = self.config.get("min_confidence_threshold", 0.3)
        self.multi_line_support = self.config.get("multi_line_support", True)

        # Heading type priority mapping
        self.element_type_priority = {
            ElementType.CHAPTER: 1,
            ElementType.ARTICLE: 2,
            ElementType.SECTION: 3,
            ElementType.SUBSECTION: 4,
            ElementType.HEADING: 5,
            ElementType.CLAUSE: 6,
            ElementType.PARAGRAPH: 7,
        }

        self.logger.info("HeadingDetector initialized with config: %s", self.config)

    def detect_headings(self, text: str) -> list[Heading]:
        """Detect and classify all headings in the given text.

        Args:
            text: Input text to analyze for headings

        Returns:
            List[Heading]: Detected and classified headings
        """
        self.logger.info("Starting heading detection for text of length %d", len(text))

        # Step 1: Find all heading candidates
        candidates = self._find_heading_candidates(text)
        self.logger.debug("Found %d heading candidates", len(candidates))

        # Step 2: Score and filter candidates
        scored_candidates = self._score_candidates(candidates, text)
        filtered_candidates = [
            c
            for c in scored_candidates
            if c.confidence_score >= self.min_confidence_threshold
        ]
        self.logger.debug(
            "Filtered to %d candidates above threshold", len(filtered_candidates)
        )

        # Step 3: Resolve conflicts and overlaps
        resolved_candidates = self._resolve_conflicts(filtered_candidates)
        self.logger.debug("Resolved to %d final candidates", len(resolved_candidates))

        # Step 4: Convert to Heading objects
        headings = self._candidates_to_headings(resolved_candidates)

        self.logger.info("Detected %d headings", len(headings))
        return headings

    def _find_heading_candidates(self, text: str) -> list[HeadingCandidate]:
        """Find potential heading candidates using various pattern matching strategies.

        Args:
            text: Input text to search

        Returns:
            List[HeadingCandidate]: All potential heading candidates found
        """
        candidates = []
        lines = text.split("\n")

        # Strategy 1: Pattern-based detection
        candidates.extend(self._detect_pattern_based_headings(text, lines))

        # Strategy 2: Format-based detection (all caps, centering, etc.)
        candidates.extend(self._detect_format_based_headings(text, lines))

        # Strategy 3: Numbering-based detection
        candidates.extend(self._detect_numbering_based_headings(text, lines))

        # Strategy 4: Multi-line heading detection
        if self.multi_line_support:
            candidates.extend(self._detect_multiline_headings(text, lines))

        return candidates

    def _detect_pattern_based_headings(
        self, text: str, lines: list[str]
    ) -> list[HeadingCandidate]:
        """Detect headings using explicit pattern matching (Section, Article, etc.).

        Args:
            text: Full input text
            lines: Text split into lines

        Returns:
            List[HeadingCandidate]: Pattern-based heading candidates
        """
        candidates = []

        # Use pattern handler to find structured elements
        pattern_categories = [
            ("section_symbols", ElementType.SECTION),
            ("articles", ElementType.ARTICLE),
            ("sections", ElementType.SECTION),
            ("chapters", ElementType.CHAPTER),
            ("subsections", ElementType.SUBSECTION),
        ]

        for category, element_type in pattern_categories:
            matches = self.pattern_handler.find_pattern_matches(text, category)

            for match_info in matches:
                # match_info is a tuple: (start_position, matched_text, groups)
                start_pos = match_info[0]
                matched_text = match_info[1].strip()
                end_pos = start_pos + len(matched_text)

                if self._is_valid_heading_length(matched_text):
                    line_num = self._get_line_number(text, start_pos)

                    candidate = HeadingCandidate(
                        text=matched_text,
                        line_number=line_num,
                        start_position=start_pos,
                        end_position=end_pos,
                        confidence_score=0.8,  # High confidence for explicit patterns
                        heading_type=element_type,
                        format_indicators=[f"pattern_{category}"],
                    )

                    # Try to extract numbering if present
                    numbering = self._extract_numbering_from_match(match_info)
                    if numbering:
                        candidate.numbering = numbering
                        candidate.confidence_score += 0.1  # Bonus for numbering

                    candidates.append(candidate)

        return candidates

    def _detect_format_based_headings(
        self, text: str, lines: list[str]
    ) -> list[HeadingCandidate]:
        """Detect headings based on formatting patterns (all caps, centering, etc.).

        Args:
            text: Full input text
            lines: Text split into lines

        Returns:
            List[HeadingCandidate]: Format-based heading candidates
        """
        candidates = []
        current_position = 0

        for line_num, line in enumerate(lines):
            stripped_line = line.strip()

            if not self._is_valid_heading_length(stripped_line):
                current_position += len(line) + 1  # +1 for newline
                continue

            format_indicators = []
            confidence_score = 0.2  # Base score for format-based detection

            # Check for all caps
            if stripped_line.isupper() and len(stripped_line) > 2:
                format_indicators.append("all_caps")
                confidence_score += 0.3

            # Check for title case
            if stripped_line.istitle():
                format_indicators.append("title_case")
                confidence_score += 0.2

            # Check for centered text (rough heuristic)
            if self._appears_centered(line, stripped_line):
                format_indicators.append("centered")
                confidence_score += 0.2

            # Check for underlined (next line is underline characters)
            if line_num + 1 < len(lines):
                next_line = lines[line_num + 1].strip()
                if self._is_underline(next_line, len(stripped_line)):
                    format_indicators.append("underlined")
                    confidence_score += 0.3

            # Check for standalone line (surrounded by empty lines)
            if self._is_standalone_line(lines, line_num):
                format_indicators.append("standalone")
                confidence_score += 0.1

            # Only create candidate if we found format indicators
            if format_indicators:
                start_pos = current_position + line.index(stripped_line)
                end_pos = start_pos + len(stripped_line)

                candidate = HeadingCandidate(
                    text=stripped_line,
                    line_number=line_num + 1,
                    start_position=start_pos,
                    end_position=end_pos,
                    confidence_score=min(confidence_score, 0.9),  # Cap at 0.9
                    heading_type=ElementType.HEADING,
                    format_indicators=format_indicators,
                )

                candidates.append(candidate)

            current_position += len(line) + 1

        return candidates

    def _detect_numbering_based_headings(
        self, text: str, lines: list[str]
    ) -> list[HeadingCandidate]:
        """Detect headings based on numbering patterns at line starts.

        Args:
            text: Full input text
            lines: Text split into lines

        Returns:
            List[HeadingCandidate]: Numbering-based heading candidates
        """
        candidates = []
        current_position = 0

        # Extract all numbering from text
        numbering_matches = self.pattern_handler.extract_numbering_from_text(text)

        for line_num, line in enumerate(lines):
            stripped_line = line.strip()

            if not self._is_valid_heading_length(stripped_line):
                current_position += len(line) + 1
                continue

            # Check if this line starts with numbering
            line_start_pos = current_position

            for numbering_info in numbering_matches:
                # numbering_info is a tuple: (numbering_type, value, start_pos, end_pos)
                match_start = numbering_info[2]
                numbering_info[3]

                # Check if numbering is at the start of this line
                if line_start_pos <= match_start < line_start_pos + len(line):
                    # Calculate confidence based on numbering type and context
                    confidence = self._calculate_numbering_confidence(
                        numbering_info, stripped_line
                    )

                    if (
                        confidence >= 0.3
                    ):  # Minimum threshold for numbering-based headings
                        candidate = HeadingCandidate(
                            text=stripped_line,
                            line_number=line_num + 1,
                            start_position=line_start_pos,
                            end_position=line_start_pos + len(stripped_line),
                            confidence_score=confidence,
                            heading_type=self._infer_type_from_numbering(
                                numbering_info
                            ),
                            format_indicators=["numbered"],
                        )

                        # Create numbering system object
                        candidate.numbering = self._create_numbering_system(
                            numbering_info
                        )
                        candidates.append(candidate)
                        break  # Only one numbering per line

            current_position += len(line) + 1

        return candidates

    def _detect_multiline_headings(
        self, text: str, lines: list[str]
    ) -> list[HeadingCandidate]:
        """Detect headings that span multiple lines.

        Args:
            text: Full input text
            lines: Text split into lines

        Returns:
            List[HeadingCandidate]: Multi-line heading candidates
        """
        candidates = []

        # Look for sequences of short lines that might form a multi-line heading
        i = 0
        while i < len(lines):
            if self._might_be_multiline_heading_start(lines, i):
                # Collect consecutive lines that might be part of the heading
                heading_lines = []
                j = i

                while j < len(lines) and self._might_be_heading_continuation(
                    lines, j, i
                ):
                    heading_lines.append(lines[j].strip())
                    j += 1

                if len(heading_lines) > 1:  # Multi-line heading found
                    combined_text = " ".join(heading_lines)

                    if self._is_valid_heading_length(combined_text):
                        # Calculate positions
                        start_line_pos = sum(len(lines[k]) + 1 for k in range(i))
                        end_line_pos = sum(len(lines[k]) + 1 for k in range(j))

                        confidence = self._calculate_multiline_confidence(
                            heading_lines, lines, i
                        )

                        candidate = HeadingCandidate(
                            text=combined_text,
                            line_number=i + 1,
                            start_position=start_line_pos,
                            end_position=end_line_pos - 1,  # -1 for last newline
                            confidence_score=confidence,
                            heading_type=ElementType.HEADING,
                            format_indicators=["multiline"],
                        )

                        candidates.append(candidate)

                i = j  # Skip processed lines
            else:
                i += 1

        return candidates

    def _score_candidates(
        self, candidates: list[HeadingCandidate], text: str
    ) -> list[HeadingCandidate]:
        """Score and rank heading candidates based on various factors.

        Args:
            candidates: List of heading candidates to score
            text: Full input text for context

        Returns:
            List[HeadingCandidate]: Candidates with updated confidence scores
        """
        for candidate in candidates:
            # Adjust scores based on additional factors

            # Length penalty/bonus
            if len(candidate.text) < 10:
                candidate.confidence_score += 0.1  # Short headings are often good
            elif len(candidate.text) > 100:
                candidate.confidence_score -= 0.2  # Very long headings are suspicious

            # Position bonus (headings often appear at document start)
            if candidate.start_position < len(text) * 0.1:  # First 10% of document
                candidate.confidence_score += 0.1

            # Numbering bonus
            if candidate.numbering:
                candidate.confidence_score += 0.15

            # Multiple format indicators bonus
            if len(candidate.format_indicators) > 1:
                candidate.confidence_score += 0.1

            # Element type adjustment
            type_priority = self.element_type_priority.get(candidate.heading_type, 5)
            if type_priority <= 3:  # High priority types (Chapter, Article, Section)
                candidate.confidence_score += 0.1

            # Ensure score doesn't exceed 1.0
            candidate.confidence_score = min(candidate.confidence_score, 1.0)

        # Sort by confidence score (highest first)
        candidates.sort(key=lambda c: c.confidence_score, reverse=True)

        return candidates

    def _resolve_conflicts(
        self, candidates: list[HeadingCandidate]
    ) -> list[HeadingCandidate]:
        """Resolve overlapping candidates and conflicts.

        Args:
            candidates: List of scored candidates

        Returns:
            List[HeadingCandidate]: Non-conflicting candidates
        """
        if not candidates:
            return []

        resolved = []

        # Sort by position to process in order
        position_sorted = sorted(candidates, key=lambda c: c.start_position)

        for candidate in position_sorted:
            # Check for overlap with already resolved candidates
            conflicts = [r for r in resolved if self._candidates_overlap(candidate, r)]

            if not conflicts:
                # No conflicts, add candidate
                resolved.append(candidate)
            else:
                # Handle conflict - keep highest confidence candidate
                max_confidence_candidate = max(
                    [candidate] + conflicts, key=lambda c: c.confidence_score
                )

                # Remove conflicting resolved candidates
                resolved = [r for r in resolved if r not in conflicts]

                # Add the best candidate
                if max_confidence_candidate not in resolved:
                    resolved.append(max_confidence_candidate)

        # Sort final result by position
        resolved.sort(key=lambda c: c.start_position)

        return resolved

    def _candidates_to_headings(
        self, candidates: list[HeadingCandidate]
    ) -> list[Heading]:
        """Convert heading candidates to Heading objects.

        Args:
            candidates: Final list of heading candidates

        Returns:
            List[Heading]: Converted Heading objects
        """
        headings = []

        for candidate in candidates:
            heading = Heading(
                element_type=candidate.heading_type,
                text=candidate.text,
                line_number=candidate.line_number,
                start_position=candidate.start_position,
                end_position=candidate.end_position,
                level=self._calculate_heading_level(candidate),
                numbering=candidate.numbering,
                metadata={
                    "confidence_score": candidate.confidence_score,
                    "format_indicators": candidate.format_indicators,
                    "detection_method": "heading_detector",
                },
            )

            headings.append(heading)

        return headings

    # Helper methods

    def _is_valid_heading_length(self, text: str) -> bool:
        """Check if text length is valid for a heading."""
        return self.min_heading_length <= len(text) <= self.max_heading_length

    def _get_line_number(self, text: str, position: int) -> int:
        """Get line number for a given character position."""
        return text[:position].count("\n") + 1

    def _appears_centered(self, full_line: str, stripped_line: str) -> bool:
        """Check if text appears to be centered in the line."""
        if not stripped_line:
            return False

        leading_spaces = len(full_line) - len(full_line.lstrip())
        trailing_spaces = len(full_line) - len(full_line.rstrip())

        # Consider centered if leading spaces are significant and roughly balanced
        return leading_spaces > 4 and abs(leading_spaces - trailing_spaces) <= 2

    def _is_underline(self, line: str, expected_length: int) -> bool:
        """Check if a line is an underline (repeated characters)."""
        if not line:
            return False

        underline_chars = {"-", "=", "_", "~", "*"}
        return (
            len(set(line)) == 1
            and line[0] in underline_chars
            and abs(len(line) - expected_length) <= 2
        )

    def _is_standalone_line(self, lines: list[str], line_index: int) -> bool:
        """Check if a line is standalone (surrounded by empty lines)."""
        prev_empty = line_index == 0 or not lines[line_index - 1].strip()
        next_empty = line_index == len(lines) - 1 or not lines[line_index + 1].strip()

        return prev_empty and next_empty

    def _extract_numbering_from_match(self, match_info: dict) -> NumberingSystem | None:
        """Extract numbering system from a pattern match."""
        # This would integrate with the numbering handler
        # Implementation depends on the match_info structure
        return None  # Placeholder

    def _calculate_numbering_confidence(
        self, numbering_info: tuple, text: str
    ) -> float:
        """Calculate confidence score for numbering-based heading."""
        base_confidence = 0.5

        # numbering_info is a tuple: (numbering_type, value, start_pos, end_pos)
        numbering_type = numbering_info[0]
        if numbering_type and hasattr(numbering_type, "value"):
            type_str = numbering_type.value.lower()
            if type_str in ["decimal", "section_symbol"]:
                base_confidence += 0.2
            elif type_str in ["roman", "letter"]:
                base_confidence += 0.1

        # Adjust based on text content
        if any(word in text.lower() for word in ["section", "article", "chapter"]):
            base_confidence += 0.2

        return min(base_confidence, 0.9)

    def _infer_type_from_numbering(self, numbering_info: tuple) -> ElementType:
        """Infer element type from numbering pattern."""
        # numbering_info is a tuple: (numbering_type, value, start_pos, end_pos)
        # Simple heuristic - can be enhanced
        numbering_type = numbering_info[0]
        value = numbering_info[1]

        if numbering_type and hasattr(numbering_type, "value"):
            type_str = numbering_type.value.lower()
            if "section" in type_str:
                return ElementType.SECTION
            elif "article" in type_str:
                return ElementType.ARTICLE
            elif "chapter" in type_str:
                return ElementType.CHAPTER

        # Check value for patterns
        if value and isinstance(value, str):
            value_lower = value.lower()
            if "section" in value_lower:
                return ElementType.SECTION
            elif "article" in value_lower:
                return ElementType.ARTICLE
            elif "chapter" in value_lower:
                return ElementType.CHAPTER

        return ElementType.HEADING

    def _create_numbering_system(self, numbering_info: tuple) -> NumberingSystem:
        """Create NumberingSystem object from numbering info."""
        # numbering_info is a tuple: (numbering_type, value, start_pos, end_pos)
        # This would integrate with the numbering handler
        # Placeholder implementation
        numbering_type = (
            numbering_info[0] if numbering_info[0] else NumberingType.DECIMAL
        )
        value = numbering_info[1] if numbering_info[1] else "1"

        return NumberingSystem(
            numbering_type=numbering_type,
            level=1,
            value=str(value),
            raw_text=str(value),
        )

    def _might_be_multiline_heading_start(self, lines: list[str], index: int) -> bool:
        """Check if line might be the start of a multi-line heading."""
        if index >= len(lines):
            return False

        line = lines[index].strip()

        # Short lines with title case or caps might be heading starts
        return len(line) < 50 and len(line) > 0 and (line.istitle() or line.isupper())

    def _might_be_heading_continuation(
        self, lines: list[str], index: int, start_index: int
    ) -> bool:
        """Check if line might be a continuation of a multi-line heading."""
        if index >= len(lines):
            return False

        line = lines[index].strip()

        # Empty lines break the heading
        if not line:
            return False

        # Lines that are too long are probably not headings
        if len(line) > 80:
            return False

        # Continue for a few lines maximum
        if index - start_index >= 3:
            return False

        return True

    def _calculate_multiline_confidence(
        self, heading_lines: list[str], all_lines: list[str], start_index: int
    ) -> float:
        """Calculate confidence for multi-line heading."""
        base_confidence = 0.4

        # Bonus for consistent formatting
        if all(line.istitle() for line in heading_lines):
            base_confidence += 0.2
        elif all(line.isupper() for line in heading_lines):
            base_confidence += 0.3

        # Bonus for being followed by empty line
        end_index = start_index + len(heading_lines)
        if end_index < len(all_lines) and not all_lines[end_index].strip():
            base_confidence += 0.1

        return min(base_confidence, 0.8)

    def _candidates_overlap(
        self, candidate1: HeadingCandidate, candidate2: HeadingCandidate
    ) -> bool:
        """Check if two candidates overlap in position."""
        return not (
            candidate1.end_position <= candidate2.start_position
            or candidate2.end_position <= candidate1.start_position
        )

    def _calculate_heading_level(self, candidate: HeadingCandidate) -> int:
        """Calculate hierarchical level for heading."""
        if candidate.numbering:
            return candidate.numbering.level

        # Use element type priority as fallback
        return self.element_type_priority.get(candidate.heading_type, 5)
