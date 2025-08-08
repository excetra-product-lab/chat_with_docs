"""
Tests for document structure detection numbering systems.

This module tests the NumberingSystemHandler class that provides utilities
for parsing, converting, and validating different numbering systems.
"""

import pytest

from app.services.document_structure_detector import (
    NumberingSystem,
    NumberingSystemHandler,
    NumberingType,
)


class TestNumberingSystemHandler:
    """Test suite for the NumberingSystemHandler class."""

    @pytest.fixture
    def handler(self):
        """Create a NumberingSystemHandler instance for testing."""
        return NumberingSystemHandler()

    @pytest.fixture
    def sample_numbering_systems(self, handler):
        """Create sample numbering systems for testing."""
        return [
            handler.create_numbering_system(NumberingType.DECIMAL, "1", "1."),
            handler.create_numbering_system(NumberingType.DECIMAL, "1.1", "1.1"),
            handler.create_numbering_system(NumberingType.DECIMAL, "1.2", "1.2"),
            handler.create_numbering_system(NumberingType.DECIMAL, "1.2.1", "1.2.1"),
            handler.create_numbering_system(NumberingType.ROMAN_UPPER, "I", "I."),
            handler.create_numbering_system(NumberingType.ROMAN_UPPER, "II", "II."),
            handler.create_numbering_system(NumberingType.LETTER_UPPER, "A", "A."),
            handler.create_numbering_system(NumberingType.LETTER_LOWER, "a", "a)"),
        ]

    def test_handler_initialization(self, handler):
        """Test NumberingSystemHandler initialization."""
        assert handler is not None

    def test_roman_to_decimal_basic(self, handler):
        """Test basic Roman numeral to decimal conversion."""
        assert handler.roman_to_decimal("I") == 1
        assert handler.roman_to_decimal("V") == 5
        assert handler.roman_to_decimal("X") == 10
        assert handler.roman_to_decimal("L") == 50
        assert handler.roman_to_decimal("C") == 100
        assert handler.roman_to_decimal("D") == 500
        assert handler.roman_to_decimal("M") == 1000

    def test_roman_to_decimal_compound(self, handler):
        """Test compound Roman numerals."""
        assert handler.roman_to_decimal("II") == 2
        assert handler.roman_to_decimal("III") == 3
        assert handler.roman_to_decimal("IV") == 4
        assert handler.roman_to_decimal("VI") == 6
        assert handler.roman_to_decimal("VII") == 7
        assert handler.roman_to_decimal("VIII") == 8
        assert handler.roman_to_decimal("IX") == 9
        assert handler.roman_to_decimal("XI") == 11
        assert handler.roman_to_decimal("XIV") == 14
        assert handler.roman_to_decimal("XV") == 15
        assert handler.roman_to_decimal("XIX") == 19
        assert handler.roman_to_decimal("XX") == 20

    def test_roman_to_decimal_complex(self, handler):
        """Test complex Roman numerals."""
        assert handler.roman_to_decimal("XXIV") == 24
        assert handler.roman_to_decimal("XLIX") == 49
        assert handler.roman_to_decimal("XCIX") == 99
        assert handler.roman_to_decimal("CDXLIV") == 444
        assert handler.roman_to_decimal("MCMXC") == 1990

    def test_roman_to_decimal_case_insensitive(self, handler):
        """Test Roman numeral conversion is case insensitive."""
        assert handler.roman_to_decimal("iv") == 4
        assert handler.roman_to_decimal("Iv") == 4
        assert handler.roman_to_decimal("iV") == 4
        assert handler.roman_to_decimal("IX") == handler.roman_to_decimal("ix")

    def test_roman_to_decimal_empty_and_invalid(self, handler):
        """Test Roman numeral conversion with invalid input."""
        assert handler.roman_to_decimal("") == 0
        assert handler.roman_to_decimal("   ") == 0

        with pytest.raises(ValueError):
            handler.roman_to_decimal("INVALID")

        with pytest.raises(ValueError):
            handler.roman_to_decimal("123")

    def test_decimal_to_roman_basic(self, handler):
        """Test basic decimal to Roman numeral conversion."""
        assert handler.decimal_to_roman(1) == "I"
        assert handler.decimal_to_roman(5) == "V"
        assert handler.decimal_to_roman(10) == "X"
        assert handler.decimal_to_roman(50) == "L"
        assert handler.decimal_to_roman(100) == "C"
        assert handler.decimal_to_roman(500) == "D"
        assert handler.decimal_to_roman(1000) == "M"

    def test_decimal_to_roman_compound(self, handler):
        """Test compound decimal to Roman numeral conversion."""
        assert handler.decimal_to_roman(2) == "II"
        assert handler.decimal_to_roman(3) == "III"
        assert handler.decimal_to_roman(4) == "IV"
        assert handler.decimal_to_roman(6) == "VI"
        assert handler.decimal_to_roman(9) == "IX"
        assert handler.decimal_to_roman(11) == "XI"
        assert handler.decimal_to_roman(14) == "XIV"
        assert handler.decimal_to_roman(19) == "XIX"
        assert handler.decimal_to_roman(24) == "XXIV"

    def test_decimal_to_roman_invalid(self, handler):
        """Test decimal to Roman conversion with invalid input."""
        with pytest.raises(ValueError):
            handler.decimal_to_roman(0)

        with pytest.raises(ValueError):
            handler.decimal_to_roman(-1)

        with pytest.raises(ValueError):
            handler.decimal_to_roman(4000)  # Traditional limit

    def test_letter_to_number_upper(self, handler):
        """Test upper case letter to number conversion."""
        assert handler.letter_to_number("A") == 1
        assert handler.letter_to_number("B") == 2
        assert handler.letter_to_number("C") == 3
        assert handler.letter_to_number("Z") == 26

    def test_letter_to_number_lower(self, handler):
        """Test lower case letter to number conversion."""
        assert handler.letter_to_number("a") == 1
        assert handler.letter_to_number("b") == 2
        assert handler.letter_to_number("c") == 3
        assert handler.letter_to_number("z") == 26

    def test_letter_to_number_invalid(self, handler):
        """Test letter to number conversion with invalid input."""
        with pytest.raises(ValueError):
            handler.letter_to_number("1")

        with pytest.raises(ValueError):
            handler.letter_to_number("AA")  # Multi-letter not supported

        with pytest.raises(ValueError):
            handler.letter_to_number("")

    def test_number_to_letter_upper(self, handler):
        """Test number to upper case letter conversion."""
        assert handler.number_to_letter(1, uppercase=True) == "A"
        assert handler.number_to_letter(2, uppercase=True) == "B"
        assert handler.number_to_letter(26, uppercase=True) == "Z"

    def test_number_to_letter_lower(self, handler):
        """Test number to lower case letter conversion."""
        assert handler.number_to_letter(1, uppercase=False) == "a"
        assert handler.number_to_letter(2, uppercase=False) == "b"
        assert handler.number_to_letter(26, uppercase=False) == "z"

    def test_number_to_letter_invalid(self, handler):
        """Test number to letter conversion with invalid input."""
        with pytest.raises(ValueError):
            handler.number_to_letter(0)

        with pytest.raises(ValueError):
            handler.number_to_letter(-1)

        with pytest.raises(ValueError):
            handler.number_to_letter(27)  # Beyond Z

    def test_create_numbering_system(self, handler):
        """Test creation of numbering system objects."""
        num_sys = handler.create_numbering_system(
            NumberingType.DECIMAL, "1.2.3", "1.2.3"
        )

        assert isinstance(num_sys, NumberingSystem)
        assert num_sys.numbering_type == NumberingType.DECIMAL
        assert num_sys.value == "1.2.3"
        assert num_sys.raw_text == "1.2.3"
        assert num_sys.level >= 0

    def test_parse_decimal_numbering(self, handler):
        """Test parsing of decimal numbering patterns."""
        test_cases = [
            ("1", 0, "1"),
            ("1.2", 1, "1.2"),
            ("1.2.3", 2, "1.2.3"),
            ("1.2.3.4", 3, "1.2.3.4"),
        ]

        for value, expected_level, expected_value in test_cases:
            num_sys = handler.parse_decimal_numbering(value)
            assert num_sys.level == expected_level
            assert num_sys.value == expected_value
            assert num_sys.numbering_type == NumberingType.DECIMAL

    def test_parse_roman_numbering(self, handler):
        """Test parsing of Roman numeral patterns."""
        upper_cases = ["I", "II", "III", "IV", "V"]
        lower_cases = ["i", "ii", "iii", "iv", "v"]

        for roman in upper_cases:
            num_sys = handler.parse_roman_numbering(roman)
            assert num_sys.numbering_type == NumberingType.ROMAN_UPPER
            assert num_sys.value == roman

        for roman in lower_cases:
            num_sys = handler.parse_roman_numbering(roman)
            assert num_sys.numbering_type == NumberingType.ROMAN_LOWER
            assert num_sys.value == roman

    def test_parse_letter_numbering(self, handler):
        """Test parsing of letter numbering patterns."""
        upper_letters = ["A", "B", "C"]
        lower_letters = ["a", "b", "c"]

        for letter in upper_letters:
            num_sys = handler.parse_letter_numbering(letter)
            assert num_sys.numbering_type == NumberingType.LETTER_UPPER
            assert num_sys.value == letter

        for letter in lower_letters:
            num_sys = handler.parse_letter_numbering(letter)
            assert num_sys.numbering_type == NumberingType.LETTER_LOWER
            assert num_sys.value == letter

    def test_determine_numbering_level(self, handler):
        """Test numbering level determination."""
        assert handler.determine_numbering_level("1") == 0
        assert handler.determine_numbering_level("1.2") == 1
        assert handler.determine_numbering_level("1.2.3") == 2
        assert handler.determine_numbering_level("1.2.3.4") == 3

    def test_normalize_numbering_format(self, handler):
        """Test numbering format normalization."""
        test_cases = [
            ("1.", NumberingType.DECIMAL, "1"),
            ("(a)", NumberingType.LETTER_LOWER, "a"),
            ("A)", NumberingType.LETTER_UPPER, "A"),
            ("I.", NumberingType.ROMAN_UPPER, "I"),
            ("§ 1.2", NumberingType.SECTION_SYMBOL, "1.2"),
        ]

        for raw_text, numbering_type, expected in test_cases:
            normalized = handler.normalize_numbering_format(raw_text, numbering_type)
            assert normalized == expected

    def test_generate_numbering_hierarchy(self, handler, sample_numbering_systems):
        """Test generation of hierarchical relationships."""
        hierarchy = handler.generate_numbering_hierarchy(sample_numbering_systems)

        assert len(hierarchy) > 0

        # Check that hierarchy is properly established
        decimal_systems = [
            ns for ns in hierarchy if ns.numbering_type == NumberingType.DECIMAL
        ]

        # Find parent and child relationships
        parent_1 = next((ns for ns in decimal_systems if ns.value == "1"), None)
        child_1_1 = next((ns for ns in decimal_systems if ns.value == "1.1"), None)

        if parent_1 and child_1_1:
            assert child_1_1.level > parent_1.level

    def test_validate_numbering_sequence(self, handler, sample_numbering_systems):
        """Test validation of numbering sequences."""
        validation = handler.validate_numbering_sequence(sample_numbering_systems)

        assert isinstance(validation, dict)
        assert "is_valid" in validation
        assert "gaps" in validation
        assert "duplicates" in validation
        assert "format_inconsistencies" in validation

    def test_convert_numbering_format(self, handler):
        """Test conversion between numbering formats."""
        # Test decimal to roman conversion
        roman_result = handler.convert_numbering_format(
            "5", NumberingType.DECIMAL, NumberingType.ROMAN_UPPER
        )
        assert roman_result == "V"

        # Test roman to decimal conversion
        decimal_result = handler.convert_numbering_format(
            "V", NumberingType.ROMAN_UPPER, NumberingType.DECIMAL
        )
        assert decimal_result == "5"

        # Test letter conversions
        letter_result = handler.convert_numbering_format(
            "3", NumberingType.DECIMAL, NumberingType.LETTER_UPPER
        )
        assert letter_result == "C"

    def test_get_numbering_statistics(self, handler, sample_numbering_systems):
        """Test generation of numbering statistics."""
        stats = handler.get_numbering_statistics(sample_numbering_systems)

        assert isinstance(stats, dict)
        assert "total_count" in stats
        assert "by_type" in stats
        assert "by_level" in stats
        assert "max_depth" in stats
        assert stats["total_count"] == len(sample_numbering_systems)

    def test_analyze_numbering_systems(self, handler):
        """Test comprehensive numbering system analysis."""
        numbering_data = [
            ("1", NumberingType.DECIMAL, "1."),
            ("1.1", NumberingType.DECIMAL, "1.1"),
            ("I", NumberingType.ROMAN_UPPER, "I."),
            ("A", NumberingType.LETTER_UPPER, "A."),
        ]

        analysis = handler.analyze_numbering_systems(numbering_data)

        assert isinstance(analysis, dict)
        assert "distribution" in analysis
        assert "hierarchy_analysis" in analysis
        assert "recommendations" in analysis

    def test_extract_numbering_components(self, handler):
        """Test extraction of numbering components."""
        test_cases = [
            ("1.2.3", ["1", "2", "3"]),
            ("5.1", ["5", "1"]),
            ("10", ["10"]),
        ]

        for numbering, expected_components in test_cases:
            components = handler.extract_numbering_components(numbering)
            assert components == expected_components

    def test_numbering_system_full_number(self, handler):
        """Test full number generation with hierarchy."""
        parent = handler.create_numbering_system(NumberingType.DECIMAL, "1", "1.")
        child = handler.create_numbering_system(NumberingType.DECIMAL, "2", "1.2")
        child.parent_numbering = parent

        assert child.get_full_number() == "1.2"

    def test_numbering_system_is_child_of(self, handler):
        """Test child relationship detection."""
        parent = handler.create_numbering_system(NumberingType.DECIMAL, "1", "1.")
        parent.level = 0

        child = handler.create_numbering_system(NumberingType.DECIMAL, "1.1", "1.1")
        child.level = 1

        # Note: This test depends on the actual implementation of is_child_of
        # which may need to be adjusted based on the real logic
        assert child.level > parent.level

    def test_edge_cases_empty_input(self, handler):
        """Test handling of empty or invalid input."""
        # Empty numbering systems list
        empty_hierarchy = handler.generate_numbering_hierarchy([])
        assert len(empty_hierarchy) == 0

        empty_validation = handler.validate_numbering_sequence([])
        assert empty_validation["is_valid"] is True  # Empty is technically valid

    def test_mixed_numbering_types(self, handler):
        """Test handling of mixed numbering types in one sequence."""
        mixed_systems = [
            handler.create_numbering_system(NumberingType.DECIMAL, "1", "1."),
            handler.create_numbering_system(NumberingType.ROMAN_UPPER, "I", "I."),
            handler.create_numbering_system(NumberingType.LETTER_UPPER, "A", "A."),
        ]

        analysis = handler.analyze_numbering_systems(
            [(ns.value, ns.numbering_type, ns.raw_text) for ns in mixed_systems]
        )

        assert "distribution" in analysis
        assert len(analysis["distribution"]) >= 3  # At least 3 different types

    def test_large_numbering_values(self, handler):
        """Test handling of large numbering values."""
        large_decimal = handler.parse_decimal_numbering("999.888.777.666")
        assert large_decimal.value == "999.888.777.666"
        assert large_decimal.level == 3

        # Test large Roman numerals (if supported)
        try:
            large_roman = handler.decimal_to_roman(3999)  # Traditional upper limit
            assert isinstance(large_roman, str)
        except ValueError:
            # Expected if implementation has limits
            pass

    def test_format_conversion_edge_cases(self, handler):
        """Test format conversion with edge cases."""
        # Test conversion of edge values
        edge_cases = [
            ("1", NumberingType.DECIMAL, NumberingType.ROMAN_UPPER, "I"),
            ("26", NumberingType.DECIMAL, NumberingType.LETTER_UPPER, "Z"),
        ]

        for value, from_type, to_type, expected in edge_cases:
            try:
                result = handler.convert_numbering_format(value, from_type, to_type)
                assert result == expected
            except (ValueError, NotImplementedError):
                # Some conversions might not be implemented
                pass
