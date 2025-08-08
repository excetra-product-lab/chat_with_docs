"""Numbering system handlers for document structure detection.

This module provides utilities for parsing, converting, and validating
different numbering systems used in legal documents.
"""

import logging
import re
from typing import Any

from .data_models import NumberingSystem, NumberingType

logger = logging.getLogger(__name__)


class NumberingSystemHandler:
    """Handler for various numbering systems in legal documents."""

    def __init__(self):
        """Initialize the numbering system handler."""
        self.logger = logging.getLogger(__name__)

    def roman_to_decimal(self, roman: str) -> int:
        """Convert Roman numeral to decimal number.

        Args:
            roman: Roman numeral string (e.g., 'IV', 'XII')

        Returns:
            int: Decimal equivalent

        Raises:
            ValueError: If invalid Roman numeral characters are found
        """
        if not roman:
            return 0

        roman = roman.upper().strip()

        # Roman numeral values
        values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

        total = 0
        prev_value = 0

        for char in reversed(roman):
            if char not in values:
                raise ValueError(f"Invalid Roman numeral character: {char}")

            value = values[char]
            if value < prev_value:
                total -= value
            else:
                total += value
            prev_value = value

        return total

    def decimal_to_roman(self, decimal: int) -> str:
        """Convert decimal number to Roman numeral.

        Args:
            decimal: Decimal number to convert

        Returns:
            str: Roman numeral representation

        Raises:
            ValueError: If number is <= 0 or > 3999
        """
        if decimal <= 0:
            raise ValueError("Roman numerals must be positive numbers")

        if decimal > 3999:
            raise ValueError("Traditional Roman numerals don't support numbers > 3999")

        values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        numerals = [
            "M",
            "CM",
            "D",
            "CD",
            "C",
            "XC",
            "L",
            "XL",
            "X",
            "IX",
            "V",
            "IV",
            "I",
        ]

        result = ""
        for i, value in enumerate(values):
            count = decimal // value
            if count:
                result += numerals[i] * count
                decimal -= value * count

        return result

    def letter_to_number(self, letter: str) -> int:
        """Convert letter to number (A=1, B=2, etc.).

        Args:
            letter: Single letter string

        Returns:
            int: Numeric equivalent

        Raises:
            ValueError: If invalid letter input
        """
        if not letter:
            raise ValueError("Letter cannot be empty")

        if len(letter) != 1:
            raise ValueError("Multi-letter sequences not supported")

        letter = letter.upper()
        if "A" <= letter <= "Z":
            return ord(letter) - ord("A") + 1
        else:
            raise ValueError(f"Invalid letter: {letter}")

    def number_to_letter(self, number: int, uppercase: bool = True) -> str:
        """Convert number to letter (1=A, 2=B, etc.).

        Args:
            number: Number to convert (1-26)
            uppercase: Whether to return uppercase letter

        Returns:
            str: Letter equivalent

        Raises:
            ValueError: If number is out of range
        """
        if number <= 0:
            raise ValueError("Number must be positive")

        if number > 26:
            raise ValueError("Number must be <= 26 for single letter conversion")

        if uppercase:
            return chr(ord("A") + number - 1)
        else:
            return chr(ord("a") + number - 1)

    def parse_decimal_numbering(self, decimal_str: str) -> NumberingSystem:
        """Parse decimal numbering into NumberingSystem object.

        Args:
            decimal_str: Decimal numbering string (e.g., '1.2.3')

        Returns:
            NumberingSystem: Parsed numbering system object
        """
        if not decimal_str:
            return NumberingSystem(
                numbering_type=NumberingType.DECIMAL,
                level=0,
                value="",
                raw_text=decimal_str,
            )

        try:
            # Remove trailing dots and split
            clean_str = decimal_str.rstrip(".")
            parts = clean_str.split(".")
            levels = [int(part) for part in parts if part.isdigit()]

            level = len(levels) - 1 if levels else 0

            return NumberingSystem(
                numbering_type=NumberingType.DECIMAL,
                level=level,
                value=clean_str,
                raw_text=decimal_str,
            )
        except ValueError as e:
            self.logger.warning(
                "Failed to parse decimal numbering '%s': %s", decimal_str, e
            )
            return NumberingSystem(
                numbering_type=NumberingType.DECIMAL,
                level=0,
                value=decimal_str,
                raw_text=decimal_str,
            )

    def parse_roman_numbering(self, roman_str: str) -> NumberingSystem:
        """Parse Roman numeral into NumberingSystem object.

        Args:
            roman_str: Roman numeral string

        Returns:
            NumberingSystem: Parsed numbering system object
        """
        if not roman_str:
            return NumberingSystem(
                numbering_type=NumberingType.ROMAN_UPPER,
                level=0,
                value="",
                raw_text=roman_str,
            )

        # Determine if upper or lower case
        is_upper = roman_str.isupper()
        numbering_type = (
            NumberingType.ROMAN_UPPER if is_upper else NumberingType.ROMAN_LOWER
        )

        return NumberingSystem(
            numbering_type=numbering_type,
            level=0,  # Roman numerals are typically top-level
            value=roman_str,
            raw_text=roman_str,
        )

    def parse_letter_numbering(self, letter_str: str) -> NumberingSystem:
        """Parse letter numbering into NumberingSystem object.

        Args:
            letter_str: Letter string

        Returns:
            NumberingSystem: Parsed numbering system object
        """
        if not letter_str:
            return NumberingSystem(
                numbering_type=NumberingType.LETTER_UPPER,
                level=0,
                value="",
                raw_text=letter_str,
            )

        # Determine if upper or lower case
        is_upper = letter_str.isupper()
        numbering_type = (
            NumberingType.LETTER_UPPER if is_upper else NumberingType.LETTER_LOWER
        )

        return NumberingSystem(
            numbering_type=numbering_type,
            level=0,  # Letters are typically top-level
            value=letter_str,
            raw_text=letter_str,
        )

    def extract_numbering_components(self, numbering: str) -> list[str]:
        """Extract numbering components from a numbering string.

        Args:
            numbering: Numbering string (e.g., '1.2.3')

        Returns:
            List[str]: List of numbering components
        """
        if not numbering:
            return []

        # Remove trailing dots and split by dots
        clean_str = numbering.rstrip(".")
        return clean_str.split(".")

    def normalize_numbering_format(
        self, raw_text: str, numbering_type: NumberingType
    ) -> str:
        """Normalize raw numbering text to clean format.

        Args:
            raw_text: Raw text containing numbering
            numbering_type: Type of numbering system

        Returns:
            str: Normalized numbering value
        """
        if not raw_text:
            return ""

        # Remove common formatting characters
        clean_text = raw_text.strip()

        if numbering_type == NumberingType.DECIMAL:
            # Remove trailing dots
            return clean_text.rstrip(".")

        elif numbering_type in (NumberingType.LETTER_UPPER, NumberingType.LETTER_LOWER):
            # Remove parentheses and dots
            clean_text = re.sub(r"[().]", "", clean_text)
            return clean_text.strip()

        elif numbering_type in (NumberingType.ROMAN_UPPER, NumberingType.ROMAN_LOWER):
            # Remove dots
            return clean_text.rstrip(".")

        elif numbering_type == NumberingType.SECTION_SYMBOL:
            # Remove section symbol and leading/trailing whitespace
            clean_text = re.sub(r"§\s*", "", clean_text)
            return clean_text.strip()

        return clean_text

    def determine_numbering_level(self, numbering: str | list[int]) -> int:
        """Determine the hierarchical level based on numbering.

        Args:
            numbering: Numbering string or list of hierarchical numbers

        Returns:
            int: Level in hierarchy (0-based)
        """
        if isinstance(numbering, str):
            # Parse decimal numbering to get level count
            components = self.extract_numbering_components(numbering)
            return len(components) - 1 if components else 0
        elif isinstance(numbering, list):
            return len(numbering) - 1 if numbering else 0
        else:
            return 0

    def analyze_numbering_systems(
        self, numbering_data: list[tuple[str, NumberingType, str]]
    ) -> dict[str, Any]:
        """Perform comprehensive analysis of numbering systems.

        Args:
            numbering_data: List of (value, type, raw_text) tuples

        Returns:
            Dict containing analysis results
        """
        if not numbering_data:
            return {"distribution": {}, "hierarchy_analysis": {}, "recommendations": []}

        # Create numbering systems from data
        numbering_systems = []
        for value, num_type, raw_text in numbering_data:
            num_sys = self.create_numbering_system(num_type, value, raw_text)
            numbering_systems.append(num_sys)

        # Analyze distribution
        type_counts = {}
        level_counts = {}

        for num_sys in numbering_systems:
            type_name = num_sys.numbering_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

            level = num_sys.level
            level_counts[level] = level_counts.get(level, 0) + 1

        # Generate recommendations
        recommendations = []
        if len(type_counts) > 3:
            recommendations.append(
                "Consider reducing the variety of numbering types for consistency"
            )

        if max(level_counts.keys()) > 4:
            recommendations.append(
                "Deep hierarchy detected - consider flattening structure"
            )

        return {
            "distribution": type_counts,
            "hierarchy_analysis": {
                "max_depth": max(level_counts.keys()) if level_counts else 0,
                "level_distribution": level_counts,
                "total_systems": len(numbering_systems),
            },
            "recommendations": recommendations,
        }

    def parse_section_symbol(self, section_str: str) -> tuple[str, list[int]]:
        """Parse section symbol numbering (e.g., '§ 1.2.3').

        Args:
            section_str: Section string with symbol

        Returns:
            Tuple of (symbol, level_numbers)
        """
        # Remove section symbol and whitespace
        clean_str = section_str.replace("§", "").strip()

        # Parse the remaining numbering
        levels = self._parse_decimal_levels(clean_str)

        return ("§", levels)

    def _parse_decimal_levels(self, decimal_str: str) -> list[int]:
        """Helper method to parse decimal numbering into levels.

        Args:
            decimal_str: Decimal numbering string

        Returns:
            List[int]: List of level numbers
        """
        if not decimal_str:
            return []

        try:
            clean_str = decimal_str.rstrip(".")
            parts = clean_str.split(".")
            return [int(part) for part in parts if part.isdigit()]
        except ValueError as e:
            self.logger.warning(
                "Failed to parse decimal levels '%s': %s", decimal_str, e
            )
            return []

    def normalize_numbering(
        self, numbering_type: NumberingType, value: str
    ) -> tuple[int, list[int]]:
        """Normalize different numbering types to standard format.

        Args:
            numbering_type: Type of numbering system
            value: Raw numbering value

        Returns:
            Tuple of (primary_number, hierarchy_levels)
        """
        try:
            if numbering_type == NumberingType.DECIMAL:
                levels = self._parse_decimal_levels(value)
                return (levels[0] if levels else 0, levels)

            elif numbering_type == NumberingType.ROMAN_UPPER:
                decimal_val = self.roman_to_decimal(value)
                return (decimal_val, [decimal_val])

            elif numbering_type == NumberingType.ROMAN_LOWER:
                decimal_val = self.roman_to_decimal(value.upper())
                return (decimal_val, [decimal_val])

            elif numbering_type == NumberingType.LETTER_UPPER:
                number_val = self.letter_to_number(value)
                return (number_val, [number_val])

            elif numbering_type == NumberingType.LETTER_LOWER:
                number_val = self.letter_to_number(value.upper())
                return (number_val, [number_val])

            elif numbering_type == NumberingType.SECTION_SYMBOL:
                symbol, levels = self.parse_section_symbol(value)
                return (levels[0] if levels else 0, levels)

            else:
                return (0, [])

        except Exception as e:
            self.logger.warning(
                "Failed to normalize numbering %s (%s): %s", value, numbering_type, e
            )
            return (0, [])

    def create_numbering_system(
        self,
        numbering_type: NumberingType,
        value: str,
        raw_text: str,
        pattern: str = None,
    ) -> NumberingSystem:
        """Create a NumberingSystem object from parsed components.

        Args:
            numbering_type: Type of numbering
            value: Parsed numbering value
            raw_text: Original text that contained the numbering
            pattern: Regex pattern used to extract (optional)

        Returns:
            NumberingSystem: Configured numbering system object
        """
        primary_num, hierarchy = self.normalize_numbering(numbering_type, value)
        level = self.determine_numbering_level(hierarchy)

        return NumberingSystem(
            numbering_type=numbering_type,
            level=level,
            value=value,
            raw_text=raw_text,
            pattern=pattern,
        )

    def compare_numbering_systems(
        self, num1: NumberingSystem, num2: NumberingSystem
    ) -> int:
        """Compare two numbering systems for sorting.

        Args:
            num1: First numbering system
            num2: Second numbering system

        Returns:
            int: -1 if num1 < num2, 0 if equal, 1 if num1 > num2
        """
        # First compare by level
        if num1.level != num2.level:
            return -1 if num1.level < num2.level else 1

        # Then compare by normalized values
        norm1 = self.normalize_numbering(num1.numbering_type, num1.value)
        norm2 = self.normalize_numbering(num2.numbering_type, num2.value)

        if norm1[0] != norm2[0]:
            return -1 if norm1[0] < norm2[0] else 1

        # Compare hierarchy levels
        for i, (v1, v2) in enumerate(zip(norm1[1], norm2[1], strict=False)):
            if v1 != v2:
                return -1 if v1 < v2 else 1

        # If one hierarchy is shorter but matches up to that point
        if len(norm1[1]) != len(norm2[1]):
            return -1 if len(norm1[1]) < len(norm2[1]) else 1

        return 0

    def generate_numbering_hierarchy(
        self, numbering_systems: list[NumberingSystem]
    ) -> list[NumberingSystem]:
        """Generate hierarchical relationships between numbering systems.

        Args:
            numbering_systems: List of numbering systems to organize

        Returns:
            List[NumberingSystem]: Organized with parent-child relationships
        """
        if not numbering_systems:
            return []

        # Sort numbering systems by level and position
        sorted_systems = sorted(numbering_systems, key=lambda x: (x.level, x.raw_text))

        # Establish parent-child relationships
        for i, current in enumerate(sorted_systems):
            # Look for parent in preceding systems
            for j in range(i - 1, -1, -1):
                candidate_parent = sorted_systems[j]

                if (
                    candidate_parent.level < current.level
                    and self.is_valid_parent_child(candidate_parent, current)
                ):
                    current.parent_numbering = candidate_parent
                    break

        return sorted_systems

    def is_valid_parent_child(
        self, parent: NumberingSystem, child: NumberingSystem
    ) -> bool:
        """Check if a parent-child relationship is valid.

        Args:
            parent: Potential parent numbering system
            child: Potential child numbering system

        Returns:
            bool: True if valid parent-child relationship
        """
        # Child must be at a deeper level
        if child.level <= parent.level:
            return False

        # For decimal numbering, check if child starts with parent numbering
        if (
            parent.numbering_type == NumberingType.DECIMAL
            and child.numbering_type == NumberingType.DECIMAL
        ):
            parent_hierarchy = self.normalize_numbering(
                parent.numbering_type, parent.value
            )[1]
            child_hierarchy = self.normalize_numbering(
                child.numbering_type, child.value
            )[1]

            # Child hierarchy should start with parent hierarchy
            if len(child_hierarchy) > len(parent_hierarchy):
                return child_hierarchy[: len(parent_hierarchy)] == parent_hierarchy

        return True

    def validate_numbering_sequence(
        self, numbering_systems: list[NumberingSystem]
    ) -> dict[str, Any]:
        """Validate that numbering sequences are logical and complete.

        Args:
            numbering_systems: List of numbering systems to validate

        Returns:
            Dict containing validation results and issues
        """
        validation_result = {
            "is_valid": True,
            "issues": [],
            "gaps": [],
            "duplicates": [],
            "format_inconsistencies": [],
            "level_statistics": {},
        }

        if not numbering_systems:
            return validation_result

        # Group by level
        levels = {}
        for num_sys in numbering_systems:
            level = num_sys.level
            if level not in levels:
                levels[level] = []
            levels[level].append(num_sys)

        validation_result["level_statistics"] = {
            level: len(systems) for level, systems in levels.items()
        }

        # Check for format inconsistencies
        type_counts = {}
        for num_sys in numbering_systems:
            type_name = num_sys.numbering_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        if len(type_counts) > 3:
            validation_result["format_inconsistencies"].append(
                f"Multiple numbering types detected: {list(type_counts.keys())}"
            )

        # Check each level for gaps and duplicates
        for level, systems in levels.items():
            if level == 0:  # Top level
                self._validate_top_level_sequence(systems, validation_result)
            else:
                self._validate_sub_level_sequence(systems, validation_result)

        validation_result["is_valid"] = (
            len(validation_result["issues"]) == 0
            and len(validation_result["gaps"]) == 0
            and len(validation_result["duplicates"]) == 0
            and len(validation_result["format_inconsistencies"]) == 0
        )

        return validation_result

    def _validate_top_level_sequence(
        self, systems: list[NumberingSystem], result: dict
    ):
        """Validate top-level numbering sequence."""
        normalized_values = []
        for sys in systems:
            norm = self.normalize_numbering(sys.numbering_type, sys.value)
            normalized_values.append((norm[0], sys))

        # Sort by normalized value
        normalized_values.sort(key=lambda x: x[0])

        # Check for duplicates
        seen_values = set()
        for value, sys in normalized_values:
            if value in seen_values:
                result["duplicates"].append(
                    f"Duplicate numbering at level 0: {sys.value}"
                )
            seen_values.add(value)

        # Check for gaps (simple consecutive check)
        if len(normalized_values) > 1:
            for i in range(1, len(normalized_values)):
                prev_val = normalized_values[i - 1][0]
                curr_val = normalized_values[i][0]
                if curr_val - prev_val > 1:
                    result["gaps"].append(
                        f"Gap in numbering between {prev_val} and {curr_val}"
                    )

    def _validate_sub_level_sequence(
        self, systems: list[NumberingSystem], result: dict
    ):
        """Validate sub-level numbering sequences."""
        # Group by parent
        parent_groups = {}
        for sys in systems:
            parent_key = (
                sys.parent_numbering.value if sys.parent_numbering else "orphan"
            )
            if parent_key not in parent_groups:
                parent_groups[parent_key] = []
            parent_groups[parent_key].append(sys)

        # Validate each parent group
        for parent_key, group_systems in parent_groups.items():
            if parent_key == "orphan":
                result["issues"].append(
                    f"Found {len(group_systems)} orphaned systems at level > 0"
                )
            else:
                self._validate_top_level_sequence(group_systems, result)

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
        try:
            # First normalize to decimal
            primary_num, hierarchy = self.normalize_numbering(from_type, value)

            if to_type == NumberingType.DECIMAL:
                return ".".join(str(num) for num in hierarchy)

            elif to_type == NumberingType.ROMAN_UPPER:
                return self.decimal_to_roman(primary_num).upper()

            elif to_type == NumberingType.ROMAN_LOWER:
                return self.decimal_to_roman(primary_num).lower()

            elif to_type == NumberingType.LETTER_UPPER:
                if primary_num <= 26:
                    return self.number_to_letter(primary_num, uppercase=True)
                return str(primary_num)  # Fallback for numbers > 26

            elif to_type == NumberingType.LETTER_LOWER:
                if primary_num <= 26:
                    return self.number_to_letter(primary_num, uppercase=False)
                return str(primary_num)  # Fallback for numbers > 26

            elif to_type == NumberingType.SECTION_SYMBOL:
                decimal_form = ".".join(str(num) for num in hierarchy)
                return f"§ {decimal_form}"

            else:
                return value  # Return original if conversion not supported

        except Exception as e:
            self.logger.warning(
                "Failed to convert numbering from %s to %s: %s", from_type, to_type, e
            )
            return value

    def get_numbering_statistics(
        self, numbering_systems: list[NumberingSystem]
    ) -> dict[str, Any]:
        """Get detailed statistics about numbering systems.

        Args:
            numbering_systems: List of numbering systems to analyze

        Returns:
            Dict containing detailed statistics
        """
        if not numbering_systems:
            return {"total": 0, "total_count": 0}

        stats = {
            "total": len(numbering_systems),
            "total_count": len(
                numbering_systems
            ),  # Add total_count for test compatibility
            "by_type": {},
            "by_level": {},
            "hierarchy_depth": 0,
            "max_depth": 0,  # Add max_depth for test compatibility
            "has_parent_child_relationships": False,
            "average_level": 0,
            "most_common_type": None,
            "level_coverage": [],
        }

        # Count by type
        for num_sys in numbering_systems:
            type_name = num_sys.numbering_type.value
            stats["by_type"][type_name] = stats["by_type"].get(type_name, 0) + 1

        # Count by level
        levels = []
        for num_sys in numbering_systems:
            level = num_sys.level
            levels.append(level)
            stats["by_level"][level] = stats["by_level"].get(level, 0) + 1

            if num_sys.parent_numbering:
                stats["has_parent_child_relationships"] = True

        # Calculate statistics
        max_level = max(levels) if levels else 0
        stats["hierarchy_depth"] = max_level + 1
        stats["max_depth"] = max_level + 1  # Add max_depth alias
        stats["average_level"] = sum(levels) / len(levels) if levels else 0
        stats["level_coverage"] = sorted(set(levels))

        # Find most common type
        if stats["by_type"]:
            stats["most_common_type"] = max(
                stats["by_type"].items(), key=lambda x: x[1]
            )[0]

        return stats

    def is_valid_roman_numeral(self, text: str) -> bool:
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
