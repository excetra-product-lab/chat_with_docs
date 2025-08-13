"""Main document structure detector class.

This module provides the primary StructureDetector class that orchestrates
the document structure detection process using various specialized handlers.
"""

import logging
from typing import Any

from .data_models import (
    DocumentElement,
    DocumentStructure,
    ElementType,
    Heading,
    NumberingSystem,
    NumberingType,
)
from .heading_detector import HeadingDetector
from .numbering_systems import NumberingSystemHandler
from .pattern_handlers import PatternHandler

logger = logging.getLogger(__name__)


class StructureDetector:
    """Main class for detecting and parsing document structure in legal documents.

    This class provides methods to analyze text and identify structural elements
    such as sections, headings, numbering systems, and hierarchical relationships.
    """

    def __init__(self, config: dict | None = None):
        """Initialize the structure detector with optional configuration.

        Args:
            config: Optional configuration dictionary with detection parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration parameters
        self.min_heading_length = self.config.get("min_heading_length", 1)
        self.max_heading_length = self.config.get("max_heading_length", 200)
        self.case_sensitive = self.config.get("case_sensitive", False)

        # Initialize handlers
        self.numbering_handler = NumberingSystemHandler()
        self.pattern_handler = PatternHandler(case_sensitive=self.case_sensitive)
        self.heading_detector = HeadingDetector(
            pattern_handler=self.pattern_handler,
            numbering_handler=self.numbering_handler,
            config=self.config,
        )

        self.logger.info("StructureDetector initialized with config: %s", self.config)

    def detect_structure(self, text: str) -> DocumentStructure:
        """
        Main entry point for document structure detection.

        Args:
            text: The input text to analyze

        Returns:
            DocumentStructure: The parsed document structure
        """
        structure = DocumentStructure()

        # Validate input
        if text is None:
            self.logger.warning("Received None text for structure detection")
            text = ""
        elif not isinstance(text, str):
            self.logger.warning(
                f"Received non-string input for structure detection: {type(text)}"
            )
            text = str(text) if text is not None else ""

        structure.metadata["input_length"] = len(text)
        structure.metadata["analysis_completed"] = False

        if len(text) == 0:
            self.logger.warning("Empty text provided for structure detection")
            structure.metadata["analysis_completed"] = True
            structure.metadata["warning"] = "Empty or None input text"
            return structure

        self.logger.info(
            "Document structure detection initiated for text of length %d", len(text)
        )

        # Step 1: Detect headings and section boundaries
        headings = self.detect_headings(text)
        structure.headings = headings
        structure.metadata["headings_count"] = len(headings)

        # Step 2: Parse numbering systems
        numbering_systems = self.parse_all_numbering_systems(text)
        structure.metadata["numbering_systems_count"] = len(numbering_systems)

        # Step 3: Build hierarchical relationships
        # Convert headings to document elements and establish hierarchy
        elements = []
        for heading in headings:
            element = DocumentElement(
                element_type=heading.element_type,
                text=heading.text,
                line_number=heading.line_number,
                start_position=heading.start_position,
                end_position=heading.end_position,
                level=heading.level,
                numbering=heading.numbering,
                metadata=heading.metadata,
            )
            elements.append(element)

        structure.elements = elements
        structure.metadata["analysis_completed"] = True

        self.logger.info(
            "Document structure detection completed. Found %d headings, %d numbering systems",
            len(headings),
            len(numbering_systems),
        )

        return structure

    def parse_all_numbering_systems(self, text: str) -> list[NumberingSystem]:
        """Parse all numbering systems found in the text.

        Args:
            text: Input text to analyze

        Returns:
            List[NumberingSystem]: All detected numbering systems
        """
        numbering_systems = []

        # Extract numbering using regex patterns
        numbering_matches = self.pattern_handler.extract_numbering_from_text(text)

        for numbering_type, value, start_pos, end_pos in numbering_matches:
            # Extract the raw text for this match
            raw_text = text[start_pos:end_pos]

            # Create numbering system object
            num_sys = self.numbering_handler.create_numbering_system(
                numbering_type=numbering_type, value=value, raw_text=raw_text
            )

            numbering_systems.append(num_sys)

        # Generate hierarchical relationships
        organized_systems = self.numbering_handler.generate_numbering_hierarchy(
            numbering_systems
        )

        self.logger.info(
            "Parsed %d numbering systems from text", len(organized_systems)
        )
        return organized_systems

    def detect_headings(self, text: str) -> list[Heading]:
        """Detect and classify all headings in the given text.

        Args:
            text: Input text to analyze for headings

        Returns:
            List[Heading]: Detected and classified headings
        """
        return self.heading_detector.detect_headings(text)

    def analyze_numbering_patterns(self, text: str) -> dict[str, Any]:
        """Analyze numbering patterns and provide detailed statistics.

        Args:
            text: Input text to analyze

        Returns:
            Dict containing numbering analysis results
        """
        numbering_systems = self.parse_all_numbering_systems(text)

        analysis: dict[str, Any] = {
            "total_numbering_systems": len(numbering_systems),
            "numbering_types": {},
            "level_distribution": {},
            "hierarchy_depth": 0,
            "validation_results": {},
            "pattern_coverage": {},
            "recommendations": [],
        }

        if not numbering_systems:
            analysis["recommendations"].append("No numbering systems detected")
            return analysis

        # Analyze by type
        for num_sys in numbering_systems:
            type_name = num_sys.numbering_type.value
            if type_name not in analysis["numbering_types"]:
                analysis["numbering_types"][type_name] = 0
            analysis["numbering_types"][type_name] += 1

        # Analyze by level
        for num_sys in numbering_systems:
            level = num_sys.level
            if level not in analysis["level_distribution"]:
                analysis["level_distribution"][level] = 0
            analysis["level_distribution"][level] += 1

        # Calculate hierarchy depth
        analysis["hierarchy_depth"] = (
            max(sys.level for sys in numbering_systems) + 1 if numbering_systems else 0
        )

        # Validate numbering sequences
        analysis["validation_results"] = (
            self.numbering_handler.validate_numbering_sequence(numbering_systems)
        )

        # Pattern coverage analysis
        analysis["pattern_coverage"] = self.pattern_handler.analyze_pattern_coverage(
            text
        )

        # Generate recommendations
        analysis["recommendations"] = self._generate_numbering_recommendations(analysis)

        return analysis

    def _generate_numbering_recommendations(self, analysis: dict) -> list[str]:
        """Generate recommendations based on numbering analysis.

        Args:
            analysis: Analysis results dictionary

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check validation results
        validation = analysis.get("validation_results", {})
        if not validation.get("is_valid", True):
            recommendations.append("Numbering sequence validation failed")

            if validation.get("gaps"):
                recommendations.append(
                    f"Found {len(validation['gaps'])} gaps in numbering"
                )

            if validation.get("duplicates"):
                recommendations.append(
                    f"Found {len(validation['duplicates'])} duplicate numbers"
                )

            if validation.get("issues"):
                recommendations.append(
                    f"Found {len(validation['issues'])} structural issues"
                )

        # Check hierarchy depth
        depth = analysis.get("hierarchy_depth", 0)
        if depth > 5:
            recommendations.append(
                f"Deep hierarchy detected ({depth} levels) - consider restructuring"
            )
        elif depth == 0:
            recommendations.append("No hierarchical structure detected")

        # Check numbering type diversity
        types = analysis.get("numbering_types", {})
        if len(types) > 3:
            recommendations.append(
                "Multiple numbering types detected - ensure consistency"
            )

        # Check level distribution
        levels = analysis.get("level_distribution", {})
        if len(levels) > 1:
            level_0_count = levels.get(0, 0)
            total_count = sum(levels.values())
            if level_0_count / total_count < 0.3:
                recommendations.append("Consider adding more top-level sections")

        if not recommendations:
            recommendations.append("Numbering structure appears well-organized")

        return recommendations

    def extract_headings_by_format(self, text: str) -> list[tuple]:
        """Extract headings based on formatting patterns.

        Args:
            text: Input text to analyze

        Returns:
            List of tuples: (heading_text, start_pos, end_pos, format_type)
        """
        return self.pattern_handler.extract_headings_by_format(text)

    def find_pattern_matches(
        self, text: str, category: str, pattern_name: str | None = None
    ) -> list[tuple]:
        """Find all matches for a specific pattern category or individual pattern.

        Args:
            text: Input text to search
            category: Pattern category (e.g., 'sections', 'articles')
            pattern_name: Specific pattern name within category (optional)

        Returns:
            List of tuples: (start_position, matched_text, groups)
        """
        return self.pattern_handler.find_pattern_matches(text, category, pattern_name)

    def extract_numbering_from_text(self, text: str) -> list[tuple]:
        """Extract all numbering patterns from text.

        Args:
            text: Input text to analyze

        Returns:
            List of tuples: (numbering_type, value, start_pos, end_pos)
        """
        return self.pattern_handler.extract_numbering_from_text(text)

    def convert_numbering_format(
        self, value: str, from_type: NumberingType, to_type: NumberingType
    ) -> str:
        """Convert numbering from one format to another.

        Args:
            value: Original numbering value
            from_type: Source numbering type
            to_type: Target numbering type

        Returns:
            str: Converted numbering value
        """
        return self.numbering_handler.convert_numbering_format(
            value, from_type, to_type
        )

    def get_numbering_statistics(
        self, numbering_systems: list[NumberingSystem]
    ) -> dict[str, Any]:
        """Get detailed statistics about numbering systems.

        Args:
            numbering_systems: List of numbering systems to analyze

        Returns:
            Dict containing detailed statistics
        """
        return self.numbering_handler.get_numbering_statistics(numbering_systems)

    def get_supported_element_types(self) -> list[ElementType]:
        """Get list of supported element types for detection."""
        return list(ElementType)

    def get_supported_numbering_types(self) -> list[NumberingType]:
        """Get list of supported numbering types for detection."""
        return list(NumberingType)

    def get_pattern_categories(self) -> list[str]:
        """Get list of available pattern categories.

        Returns:
            List of pattern category names
        """
        return self.pattern_handler.get_pattern_categories()

    def get_patterns_in_category(self, category: str) -> list[str]:
        """Get list of pattern names in a specific category.

        Args:
            category: Pattern category name

        Returns:
            List of pattern names in the category
        """
        return self.pattern_handler.get_patterns_in_category(category)

    def validate_patterns(self) -> dict[str, bool]:
        """Validate all compiled regex patterns.

        Returns:
            Dict mapping pattern names to validation status
        """
        return self.pattern_handler.validate_patterns()

    def validate_config(self) -> bool:
        """Validate the current configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            if self.min_heading_length <= 0:
                return False
            if self.max_heading_length <= self.min_heading_length:
                return False
            return True
        except Exception as e:
            self.logger.error("Configuration validation failed: %s", e)
            return False
