"""
Tests for document structure detection data models.

This module tests the data models, enums, and data classes used for representing
document structure elements, numbering systems, and hierarchical relationships.
"""

import pytest

from app.services.document_structure_detector import (
    DocumentElement,
    DocumentStructure,
    ElementType,
    Heading,
    NumberingSystem,
    NumberingType,
    Section,
)


class TestElementType:
    """Test suite for the ElementType enum."""

    def test_element_type_values(self):
        """Test that ElementType has expected values."""
        expected_types = [
            "heading",
            "section",
            "subsection",
            "clause",
            "paragraph",
            "list_item",
            "article",
            "chapter",
        ]

        for expected in expected_types:
            assert hasattr(ElementType, expected.upper())
            assert getattr(ElementType, expected.upper()).value == expected

    def test_element_type_membership(self):
        """Test ElementType membership operations."""
        assert ElementType.HEADING in ElementType
        assert ElementType.SECTION in ElementType
        assert "invalid_type" not in [e.value for e in ElementType]

    def test_element_type_string_representation(self):
        """Test string representation of ElementType."""
        assert str(ElementType.HEADING) == "ElementType.HEADING"
        assert repr(ElementType.SECTION) == "<ElementType.SECTION: 'section'>"


class TestNumberingType:
    """Test suite for the NumberingType enum."""

    def test_numbering_type_values(self):
        """Test that NumberingType has expected values."""
        expected_types = [
            "decimal",
            "roman_upper",
            "roman_lower",
            "letter_upper",
            "letter_lower",
            "section_symbol",
            "mixed",
        ]

        for expected in expected_types:
            assert hasattr(NumberingType, expected.upper())
            assert getattr(NumberingType, expected.upper()).value == expected

    def test_numbering_type_membership(self):
        """Test NumberingType membership operations."""
        assert NumberingType.DECIMAL in NumberingType
        assert NumberingType.ROMAN_UPPER in NumberingType
        assert "invalid_numbering" not in [n.value for n in NumberingType]

    def test_numbering_type_string_representation(self):
        """Test string representation of NumberingType."""
        assert str(NumberingType.DECIMAL) == "NumberingType.DECIMAL"
        assert (
            repr(NumberingType.ROMAN_UPPER)
            == "<NumberingType.ROMAN_UPPER: 'roman_upper'>"
        )


class TestNumberingSystem:
    """Test suite for the NumberingSystem data class."""

    def test_numbering_system_creation(self):
        """Test creation of NumberingSystem objects."""
        num_sys = NumberingSystem(
            numbering_type=NumberingType.DECIMAL, level=1, value="1.2", raw_text="1.2)"
        )

        assert num_sys.numbering_type == NumberingType.DECIMAL
        assert num_sys.level == 1
        assert num_sys.value == "1.2"
        assert num_sys.raw_text == "1.2)"
        assert num_sys.pattern is None
        assert num_sys.parent_numbering is None

    def test_numbering_system_with_optional_fields(self):
        """Test NumberingSystem with optional fields."""
        parent = NumberingSystem(
            numbering_type=NumberingType.DECIMAL, level=0, value="1", raw_text="1."
        )

        child = NumberingSystem(
            numbering_type=NumberingType.DECIMAL,
            level=1,
            value="1.2",
            raw_text="1.2)",
            pattern=r"\d+\.\d+",
            parent_numbering=parent,
        )

        assert child.pattern == r"\d+\.\d+"
        assert child.parent_numbering == parent

    def test_numbering_system_validation(self):
        """Test NumberingSystem validation in __post_init__."""
        # Valid numbering system
        valid_sys = NumberingSystem(
            numbering_type=NumberingType.DECIMAL, level=0, value="1", raw_text="1."
        )
        assert valid_sys.level == 0

        # Invalid level should raise ValueError
        with pytest.raises(ValueError, match="Numbering level cannot be negative"):
            NumberingSystem(
                numbering_type=NumberingType.DECIMAL, level=-1, value="1", raw_text="1."
            )

    def test_get_full_number_without_parent(self):
        """Test get_full_number method without parent."""
        num_sys = NumberingSystem(
            numbering_type=NumberingType.DECIMAL, level=0, value="1", raw_text="1."
        )

        assert num_sys.get_full_number() == "1"

    def test_get_full_number_with_parent(self):
        """Test get_full_number method with parent hierarchy."""
        parent = NumberingSystem(
            numbering_type=NumberingType.DECIMAL, level=0, value="1", raw_text="1."
        )

        child = NumberingSystem(
            numbering_type=NumberingType.DECIMAL,
            level=1,
            value="2",
            raw_text="1.2",
            parent_numbering=parent,
        )

        assert child.get_full_number() == "1.2"

    def test_is_child_of_method(self):
        """Test is_child_of method."""
        parent = NumberingSystem(
            numbering_type=NumberingType.DECIMAL, level=0, value="1", raw_text="1."
        )

        child = NumberingSystem(
            numbering_type=NumberingType.DECIMAL, level=1, value="1.2", raw_text="1.2"
        )

        # Child should be identified as child of parent
        assert child.is_child_of(parent)

        # Parent should not be child of child
        assert not parent.is_child_of(child)

    def test_numbering_system_equality(self):
        """Test NumberingSystem equality comparison."""
        sys1 = NumberingSystem(
            numbering_type=NumberingType.DECIMAL, level=1, value="1.2", raw_text="1.2"
        )

        sys2 = NumberingSystem(
            numbering_type=NumberingType.DECIMAL, level=1, value="1.2", raw_text="1.2"
        )

        sys3 = NumberingSystem(
            numbering_type=NumberingType.ROMAN_UPPER, level=1, value="I", raw_text="I."
        )

        assert sys1 == sys2
        assert sys1 != sys3


class TestDocumentElement:
    """Test suite for the DocumentElement data class."""

    def test_document_element_creation(self):
        """Test creation of DocumentElement objects."""
        element = DocumentElement(
            element_type=ElementType.HEADING,
            text="Sample Heading",
            line_number=5,
            start_position=100,
            end_position=120,
            level=1,
        )

        assert element.element_type == ElementType.HEADING
        assert element.text == "Sample Heading"
        assert element.line_number == 5
        assert element.start_position == 100
        assert element.end_position == 120
        assert element.level == 1
        assert element.numbering is None
        assert element.parent is None
        assert len(element.children) == 0
        assert len(element.metadata) == 0

    def test_document_element_with_optional_fields(self):
        """Test DocumentElement with optional fields."""
        numbering = NumberingSystem(
            numbering_type=NumberingType.DECIMAL, level=0, value="1", raw_text="1."
        )

        element = DocumentElement(
            element_type=ElementType.SECTION,
            text="Section 1",
            line_number=1,
            start_position=0,
            end_position=10,
            level=0,
            numbering=numbering,
            metadata={"custom_field": "value"},
        )

        assert element.numbering == numbering
        assert element.metadata["custom_field"] == "value"

    def test_document_element_validation(self):
        """Test DocumentElement validation in __post_init__."""
        # Valid element
        valid_element = DocumentElement(
            element_type=ElementType.HEADING,
            text="Valid",
            line_number=1,
            start_position=10,
            end_position=20,
        )
        assert valid_element.start_position == 10

        # Negative positions should raise ValueError
        with pytest.raises(ValueError, match="Positions cannot be negative"):
            DocumentElement(
                element_type=ElementType.HEADING,
                text="Invalid",
                line_number=1,
                start_position=-1,
                end_position=10,
            )

        # Start >= end should raise ValueError
        with pytest.raises(
            ValueError, match="Start position must be less than end position"
        ):
            DocumentElement(
                element_type=ElementType.HEADING,
                text="Invalid",
                line_number=1,
                start_position=20,
                end_position=10,
            )

        # Negative level should raise ValueError
        with pytest.raises(ValueError, match="Level cannot be negative"):
            DocumentElement(
                element_type=ElementType.HEADING,
                text="Invalid",
                line_number=1,
                start_position=10,
                end_position=20,
                level=-1,
            )

    def test_add_child_method(self):
        """Test add_child method for hierarchical relationships."""
        parent = DocumentElement(
            element_type=ElementType.SECTION,
            text="Parent Section",
            line_number=1,
            start_position=0,
            end_position=20,
            level=0,
        )

        child = DocumentElement(
            element_type=ElementType.SUBSECTION,
            text="Child Subsection",
            line_number=3,
            start_position=25,
            end_position=45,
            level=0,
        )

        parent.add_child(child)

        assert len(parent.children) == 1
        assert parent.children[0] == child
        assert child.parent == parent
        assert child.level == parent.level + 1

    def test_get_full_text_method(self):
        """Test get_full_text method with children."""
        parent = DocumentElement(
            element_type=ElementType.SECTION,
            text="Parent Text",
            line_number=1,
            start_position=0,
            end_position=10,
        )

        child1 = DocumentElement(
            element_type=ElementType.SUBSECTION,
            text="Child 1 Text",
            line_number=2,
            start_position=15,
            end_position=25,
        )

        child2 = DocumentElement(
            element_type=ElementType.SUBSECTION,
            text="Child 2 Text",
            line_number=3,
            start_position=30,
            end_position=40,
        )

        parent.add_child(child1)
        parent.add_child(child2)

        full_text = parent.get_full_text()
        expected = "Parent Text\nChild 1 Text\nChild 2 Text"
        assert full_text == expected

    def test_get_hierarchy_path_method(self):
        """Test get_hierarchy_path method."""
        # Create hierarchy with numbering
        parent_numbering = NumberingSystem(
            numbering_type=NumberingType.DECIMAL, level=0, value="1", raw_text="1."
        )

        child_numbering = NumberingSystem(
            numbering_type=NumberingType.DECIMAL, level=1, value="1.1", raw_text="1.1"
        )

        parent = DocumentElement(
            element_type=ElementType.SECTION,
            text="Parent",
            line_number=1,
            start_position=0,
            end_position=10,
            numbering=parent_numbering,
        )

        child = DocumentElement(
            element_type=ElementType.SUBSECTION,
            text="Child",
            line_number=2,
            start_position=15,
            end_position=25,
            numbering=child_numbering,
        )

        parent.add_child(child)

        path = child.get_hierarchy_path()
        assert "1" in path
        assert "1.1" in path

    def test_document_element_with_no_numbering(self):
        """Test hierarchy path for elements without numbering."""
        element = DocumentElement(
            element_type=ElementType.HEADING,
            text="Simple Heading",
            line_number=1,
            start_position=0,
            end_position=15,
        )

        path = element.get_hierarchy_path()
        assert "heading" in path


class TestSection:
    """Test suite for the Section data class."""

    def test_section_creation(self):
        """Test creation of Section objects."""
        section = Section(
            element_type=ElementType.SECTION,
            text="Section Text",
            line_number=1,
            start_position=0,
            end_position=12,
            section_number="1.1",
        )

        assert section.element_type == ElementType.SECTION
        assert section.text == "Section Text"
        assert section.section_number == "1.1"

    def test_section_inheritance(self):
        """Test that Section inherits from DocumentElement."""
        section = Section(
            element_type=ElementType.SECTION,
            text="Test Section",
            line_number=1,
            start_position=0,
            end_position=12,
            section_number="1",
        )

        assert isinstance(section, DocumentElement)
        assert hasattr(section, "add_child")
        assert hasattr(section, "get_full_text")


class TestHeading:
    """Test suite for the Heading data class."""

    def test_heading_creation(self):
        """Test creation of Heading objects."""
        heading = Heading(
            element_type=ElementType.HEADING,
            text="Heading Text",
            line_number=1,
            start_position=0,
            end_position=12,
            heading_level=2,
            is_numbered=True,
        )

        assert heading.element_type == ElementType.HEADING
        assert heading.text == "Heading Text"
        assert heading.heading_level == 2
        assert heading.is_numbered is True

    def test_heading_validation(self):
        """Test Heading validation in __post_init__."""
        # Valid heading
        valid_heading = Heading(
            element_type=ElementType.HEADING,
            text="Valid Heading",
            line_number=1,
            start_position=0,
            end_position=13,
            heading_level=1,
        )
        assert valid_heading.heading_level == 1

        # Invalid heading level should raise ValueError
        with pytest.raises(ValueError, match="Heading level must be at least 1"):
            Heading(
                element_type=ElementType.HEADING,
                text="Invalid Heading",
                line_number=1,
                start_position=0,
                end_position=15,
                heading_level=0,
            )

    def test_heading_inheritance(self):
        """Test that Heading inherits from DocumentElement."""
        heading = Heading(
            element_type=ElementType.HEADING,
            text="Test Heading",
            line_number=1,
            start_position=0,
            end_position=12,
        )

        assert isinstance(heading, DocumentElement)
        assert hasattr(heading, "add_child")
        assert hasattr(heading, "get_hierarchy_path")


class TestDocumentStructure:
    """Test suite for the DocumentStructure data class."""

    def test_document_structure_creation(self):
        """Test creation of DocumentStructure objects."""
        structure = DocumentStructure()

        assert len(structure.elements) == 0
        assert len(structure.numbering_systems) == 0
        assert len(structure.metadata) == 0

    def test_document_structure_with_data(self):
        """Test DocumentStructure with initial data."""
        element = DocumentElement(
            element_type=ElementType.HEADING,
            text="Test",
            line_number=1,
            start_position=0,
            end_position=4,
        )

        numbering = NumberingSystem(
            numbering_type=NumberingType.DECIMAL, level=0, value="1", raw_text="1."
        )

        structure = DocumentStructure(
            elements=[element],
            numbering_systems=[numbering],
            metadata={"source": "test"},
        )

        assert len(structure.elements) == 1
        assert len(structure.numbering_systems) == 1
        assert structure.metadata["source"] == "test"

    def test_add_element_method(self):
        """Test add_element method."""
        structure = DocumentStructure()

        numbering = NumberingSystem(
            numbering_type=NumberingType.DECIMAL, level=0, value="1", raw_text="1."
        )

        element = DocumentElement(
            element_type=ElementType.SECTION,
            text="Test Section",
            line_number=1,
            start_position=0,
            end_position=12,
            numbering=numbering,
        )

        structure.add_element(element)

        assert len(structure.elements) == 1
        assert len(structure.numbering_systems) == 1
        assert structure.elements[0] == element
        assert structure.numbering_systems[0] == numbering

    def test_get_elements_by_type_method(self):
        """Test get_elements_by_type method."""
        structure = DocumentStructure()

        heading = DocumentElement(
            element_type=ElementType.HEADING,
            text="Heading",
            line_number=1,
            start_position=0,
            end_position=7,
        )

        section = DocumentElement(
            element_type=ElementType.SECTION,
            text="Section",
            line_number=2,
            start_position=10,
            end_position=17,
        )

        structure.add_element(heading)
        structure.add_element(section)

        headings = structure.get_elements_by_type(ElementType.HEADING)
        sections = structure.get_elements_by_type(ElementType.SECTION)

        assert len(headings) == 1
        assert len(sections) == 1
        assert headings[0] == heading
        assert sections[0] == section

    def test_get_sections_method(self):
        """Test get_sections method."""
        structure = DocumentStructure()

        section = Section(
            element_type=ElementType.SECTION,
            text="Test Section",
            line_number=1,
            start_position=0,
            end_position=12,
            section_number="1",
        )

        structure.add_element(section)

        sections = structure.get_sections()
        assert len(sections) == 1
        assert sections[0] == section

    def test_get_headings_method(self):
        """Test get_headings method."""
        structure = DocumentStructure()

        heading = Heading(
            element_type=ElementType.HEADING,
            text="Test Heading",
            line_number=1,
            start_position=0,
            end_position=12,
        )

        structure.add_element(heading)

        headings = structure.get_headings()
        assert len(headings) == 1
        assert headings[0] == heading

    def test_get_element_by_numbering_method(self):
        """Test get_element_by_numbering method."""
        structure = DocumentStructure()

        numbering = NumberingSystem(
            numbering_type=NumberingType.DECIMAL, level=0, value="1.2", raw_text="1.2"
        )

        element = DocumentElement(
            element_type=ElementType.SECTION,
            text="Test Section",
            line_number=1,
            start_position=0,
            end_position=12,
            numbering=numbering,
        )

        structure.add_element(element)

        found_element = structure.get_element_by_numbering("1.2")
        assert found_element == element

        not_found = structure.get_element_by_numbering("2.1")
        assert not_found is None

    def test_get_max_level_method(self):
        """Test get_max_level method."""
        structure = DocumentStructure()

        # Empty structure should have max level 0
        assert structure.get_max_level() == 0

        # Add elements with different levels
        element1 = DocumentElement(
            element_type=ElementType.SECTION,
            text="Level 0",
            line_number=1,
            start_position=0,
            end_position=7,
            level=0,
        )

        element2 = DocumentElement(
            element_type=ElementType.SUBSECTION,
            text="Level 2",
            line_number=2,
            start_position=10,
            end_position=17,
            level=2,
        )

        structure.add_element(element1)
        structure.add_element(element2)

        assert structure.get_max_level() == 2

    def test_to_dict_method(self):
        """Test to_dict method."""
        structure = DocumentStructure()

        element = DocumentElement(
            element_type=ElementType.HEADING,
            text="Test",
            line_number=1,
            start_position=0,
            end_position=4,
        )

        numbering = NumberingSystem(
            numbering_type=NumberingType.DECIMAL, level=0, value="1", raw_text="1."
        )

        structure.add_element(element)
        structure.metadata["test_key"] = "test_value"

        result_dict = structure.to_dict()

        assert "total_elements" in result_dict
        assert "max_level" in result_dict
        assert "element_types" in result_dict
        assert "numbering_systems" in result_dict
        assert "metadata" in result_dict

        assert result_dict["total_elements"] == 1
        assert result_dict["metadata"]["test_key"] == "test_value"

    def test_document_structure_immutable_elements(self):
        """Test that elements list behaves correctly."""
        structure = DocumentStructure()

        element = DocumentElement(
            element_type=ElementType.HEADING,
            text="Test",
            line_number=1,
            start_position=0,
            end_position=4,
        )

        # Add element through method
        structure.add_element(element)

        # Direct modification should work (mutable list)
        structure.elements.append(element)
        assert len(structure.elements) == 2
