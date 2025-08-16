"""
Tests for enhanced hierarchy and relationship models.

This module tests the hierarchy models for proper validation, relationships,
and integration with existing document structure detection.
"""

from uuid import uuid4

import pytest

from app.models.hierarchy_models import (
    DocumentHierarchy,
    DocumentRelationship,
    EnhancedDocumentElement,
    EnhancedElementType,
    EnhancedNumberingSystem,
    find_common_ancestors,
    merge_hierarchies,
)
from app.services.document_structure_detector.data_models import (
    DocumentElement,
    ElementType,
    NumberingSystem,
    NumberingType,
)


class TestEnhancedElementType:
    """Test cases for EnhancedElementType model."""

    def test_create_enhanced_element_type(self):
        """Test creating enhanced element type with validation."""
        element_type = EnhancedElementType(
            element_type=ElementType.HEADING,
            confidence_score=0.95,
            detection_method="regex",
            metadata={"pattern": "# .*"},
        )

        assert element_type.element_type == ElementType.HEADING
        assert element_type.confidence_score == 0.95
        assert element_type.detection_method == "regex"
        assert element_type.metadata["pattern"] == "# .*"

    def test_detection_method_validation(self):
        """Test detection method validation and normalization."""
        element_type = EnhancedElementType(
            element_type=ElementType.SECTION, detection_method="RULE_BASED"
        )
        assert element_type.detection_method == "rule_based"


class TestEnhancedNumberingSystem:
    """Test cases for EnhancedNumberingSystem model."""

    def test_create_enhanced_numbering_system(self):
        """Test creating enhanced numbering system."""
        numbering = EnhancedNumberingSystem(
            numbering_type=NumberingType.DECIMAL,
            level=1,
            value="1.2",
            raw_text="1.2 Introduction",
            pattern=r"\d+\.\d+",
            confidence_score=0.9,
            is_validated=True,
        )

        assert numbering.numbering_type == NumberingType.DECIMAL
        assert numbering.level == 1
        assert numbering.value == "1.2"
        assert numbering.raw_text == "1.2 Introduction"
        assert numbering.pattern == r"\d+\.\d+"
        assert numbering.confidence_score == 0.9
        assert numbering.is_validated is True
        assert len(numbering.system_id) == 36  # UUID length

    def test_get_full_number(self):
        """Test full number generation with parent numbers."""
        numbering = EnhancedNumberingSystem(
            numbering_type=NumberingType.DECIMAL,
            level=2,
            value="3",
            raw_text="3 Subsection",
        )

        # Without parent numbers
        assert numbering.get_full_number() == "3"

        # With parent numbers
        assert numbering.get_full_number(["1", "2"]) == "1.2.3"

    def test_is_child_of(self):
        """Test child relationship detection."""
        parent = EnhancedNumberingSystem(
            numbering_type=NumberingType.DECIMAL,
            level=1,
            value="1",
            raw_text="1 Section",
        )

        child = EnhancedNumberingSystem(
            numbering_type=NumberingType.DECIMAL,
            level=2,
            value="1.1",
            raw_text="1.1 Subsection",
            parent_system_id=parent.system_id,
        )

        assert child.is_child_of(parent) is True
        assert parent.is_child_of(child) is False

    def test_from_base_numbering(self):
        """Test creation from base NumberingSystem."""
        base_numbering = NumberingSystem(
            numbering_type=NumberingType.ROMAN_UPPER,
            level=0,
            value="I",
            raw_text="I. Chapter One",
            pattern=r"[IVX]+\.",
        )

        enhanced = EnhancedNumberingSystem.from_base_numbering(
            base_numbering, confidence_score=0.8, is_validated=True
        )

        assert enhanced.numbering_type == base_numbering.numbering_type
        assert enhanced.level == base_numbering.level
        assert enhanced.value == base_numbering.value
        assert enhanced.raw_text == base_numbering.raw_text
        assert enhanced.pattern == base_numbering.pattern
        assert enhanced.confidence_score == 0.8
        assert enhanced.is_validated is True


class TestEnhancedDocumentElement:
    """Test cases for EnhancedDocumentElement model."""

    def test_create_enhanced_document_element(self):
        """Test creating enhanced document element."""
        element_type = EnhancedElementType(
            element_type=ElementType.HEADING, confidence_score=0.9
        )

        numbering = EnhancedNumberingSystem(
            numbering_type=NumberingType.DECIMAL,
            level=1,
            value="1",
            raw_text="1 Introduction",
        )

        element = EnhancedDocumentElement(
            element_type=element_type,
            text="Introduction",
            line_number=1,
            start_position=0,
            end_position=12,
            level=1,
            numbering=numbering,
            semantic_role="introduction",
            importance_score=0.8,
        )

        assert element.element_type == element_type
        assert element.text == "Introduction"
        assert element.line_number == 1
        assert element.start_position == 0
        assert element.end_position == 12
        assert element.level == 1
        assert element.numbering == numbering
        assert element.semantic_role == "introduction"
        assert element.importance_score == 0.8
        assert len(element.element_id) == 36  # UUID length

    def test_position_validation(self):
        """Test position validation."""
        element_type = EnhancedElementType(element_type=ElementType.PARAGRAPH)

        with pytest.raises(
            ValueError, match="end_position must be greater than start_position"
        ):
            EnhancedDocumentElement(
                element_type=element_type,
                text="Sample text",
                line_number=1,
                start_position=10,
                end_position=5,  # Invalid: end < start
                level=0,
            )

    def test_semantic_role_validation(self):
        """Test semantic role validation."""
        element_type = EnhancedElementType(element_type=ElementType.HEADING)

        element = EnhancedDocumentElement(
            element_type=element_type,
            text="Title",
            line_number=1,
            start_position=0,
            end_position=5,
            semantic_role="TITLE",
        )
        assert element.semantic_role == "title"

    def test_child_management(self):
        """Test adding and removing child elements."""
        element_type = EnhancedElementType(element_type=ElementType.SECTION)

        parent = EnhancedDocumentElement(
            element_type=element_type,
            text="Section",
            line_number=1,
            start_position=0,
            end_position=7,
        )

        child_id = str(uuid4())

        # Add child
        parent.add_child(child_id)
        assert child_id in parent.child_ids

        # Add same child again (should not duplicate)
        parent.add_child(child_id)
        assert parent.child_ids.count(child_id) == 1

        # Remove child
        parent.remove_child(child_id)
        assert child_id not in parent.child_ids

    def test_chunk_reference_management(self):
        """Test adding chunk references."""
        element_type = EnhancedElementType(element_type=ElementType.PARAGRAPH)

        element = EnhancedDocumentElement(
            element_type=element_type,
            text="Paragraph",
            line_number=1,
            start_position=0,
            end_position=9,
        )

        chunk_id = str(uuid4())

        # Add chunk reference
        element.add_chunk_reference(chunk_id)
        assert chunk_id in element.chunk_references

        # Add same reference again (should not duplicate)
        element.add_chunk_reference(chunk_id)
        assert element.chunk_references.count(chunk_id) == 1

    def test_from_base_element(self):
        """Test creation from base DocumentElement."""
        base_numbering = NumberingSystem(
            numbering_type=NumberingType.DECIMAL,
            level=1,
            value="1.1",
            raw_text="1.1 Subsection",
        )

        base_element = DocumentElement(
            element_type=ElementType.SUBSECTION,
            text="Subsection Title",
            line_number=5,
            start_position=100,
            end_position=116,
            level=1,
            numbering=base_numbering,
            metadata={"source": "document"},
        )

        enhanced = EnhancedDocumentElement.from_base_element(
            base_element, semantic_role="heading", importance_score=0.7
        )

        assert enhanced.element_type.element_type == ElementType.SUBSECTION
        assert enhanced.element_type.detection_method == "structure_detector"
        assert enhanced.text == "Subsection Title"
        assert enhanced.line_number == 5
        assert enhanced.start_position == 100
        assert enhanced.end_position == 116
        assert enhanced.level == 1
        assert enhanced.numbering.value == "1.1"
        assert enhanced.metadata == {"source": "document"}
        assert enhanced.semantic_role == "heading"
        assert enhanced.importance_score == 0.7


class TestDocumentRelationship:
    """Test cases for DocumentRelationship model."""

    def test_create_document_relationship(self):
        """Test creating document relationship."""
        relationship = DocumentRelationship(
            source_element_id="elem1",
            target_element_id="elem2",
            relationship_type="parent_child",
            confidence_score=0.95,
            bidirectional=True,
            distance=1,
            semantic_similarity=0.8,
        )

        assert relationship.source_element_id == "elem1"
        assert relationship.target_element_id == "elem2"
        assert relationship.relationship_type == "parent_child"
        assert relationship.confidence_score == 0.95
        assert relationship.bidirectional is True
        assert relationship.distance == 1
        assert relationship.semantic_similarity == 0.8
        assert len(relationship.relationship_id) == 36  # UUID length

    def test_relationship_type_validation(self):
        """Test relationship type validation."""
        relationship = DocumentRelationship(
            source_element_id="elem1",
            target_element_id="elem2",
            relationship_type="REFERENCE",
        )
        assert relationship.relationship_type == "reference"

    def test_get_reverse_relationship(self):
        """Test getting reverse relationship."""
        relationship = DocumentRelationship(
            source_element_id="elem1",
            target_element_id="elem2",
            relationship_type="parent_child",
            confidence_score=0.9,
            bidirectional=True,
            distance=1,
            metadata={"test": "data"},
        )

        reverse = relationship.get_reverse_relationship()

        assert reverse is not None
        assert reverse.source_element_id == "elem2"
        assert reverse.target_element_id == "elem1"
        assert reverse.relationship_type == "child_parent"
        assert reverse.confidence_score == 0.9
        assert reverse.bidirectional is True
        assert reverse.distance == 1
        assert reverse.metadata == {"test": "data"}

    def test_non_bidirectional_no_reverse(self):
        """Test that non-bidirectional relationships don't return reverse."""
        relationship = DocumentRelationship(
            source_element_id="elem1",
            target_element_id="elem2",
            relationship_type="reference",
            bidirectional=False,
        )

        reverse = relationship.get_reverse_relationship()
        assert reverse is None


class TestDocumentHierarchy:
    """Test cases for DocumentHierarchy model."""

    def test_create_document_hierarchy(self):
        """Test creating document hierarchy."""
        hierarchy = DocumentHierarchy(document_filename="test.pdf")

        assert hierarchy.document_filename == "test.pdf"
        assert len(hierarchy.hierarchy_id) == 36  # UUID length
        assert hierarchy.elements == {}
        assert hierarchy.relationships == {}
        assert hierarchy.max_depth == 0
        assert hierarchy.total_elements == 0

    def test_add_element(self):
        """Test adding elements to hierarchy."""
        hierarchy = DocumentHierarchy(document_filename="test.pdf")

        element_type = EnhancedElementType(element_type=ElementType.HEADING)
        element = EnhancedDocumentElement(
            element_type=element_type,
            text="Heading",
            line_number=1,
            start_position=0,
            end_position=7,
            level=1,
        )

        hierarchy.add_element(element)

        assert element.element_id in hierarchy.elements
        assert element.element_id in hierarchy.root_element_ids
        assert hierarchy.total_elements == 1
        assert hierarchy.max_depth == 1

    def test_add_relationship(self):
        """Test adding relationships to hierarchy."""
        hierarchy = DocumentHierarchy(document_filename="test.pdf")

        relationship = DocumentRelationship(
            source_element_id="elem1",
            target_element_id="elem2",
            relationship_type="parent_child",
            bidirectional=True,
        )

        hierarchy.add_relationship(relationship)

        assert relationship.relationship_id in hierarchy.relationships
        # Should also add reverse relationship
        assert len(hierarchy.relationships) == 2

    def test_get_children(self):
        """Test getting child elements."""
        hierarchy = DocumentHierarchy(document_filename="test.pdf")

        # Create parent element
        parent_type = EnhancedElementType(element_type=ElementType.SECTION)
        parent = EnhancedDocumentElement(
            element_type=parent_type,
            text="Section",
            line_number=1,
            start_position=0,
            end_position=7,
            level=0,
        )

        # Create child element
        child_type = EnhancedElementType(element_type=ElementType.SUBSECTION)
        child = EnhancedDocumentElement(
            element_type=child_type,
            text="Subsection",
            line_number=2,
            start_position=8,
            end_position=18,
            level=1,
            parent_id=parent.element_id,
        )

        parent.add_child(child.element_id)

        hierarchy.add_element(parent)
        hierarchy.add_element(child)

        children = hierarchy.get_children(parent.element_id)

        assert len(children) == 1
        assert children[0].element_id == child.element_id

    def test_get_descendants(self):
        """Test getting all descendant elements."""
        hierarchy = DocumentHierarchy(document_filename="test.pdf")

        # Create grandparent
        grandparent_type = EnhancedElementType(element_type=ElementType.CHAPTER)
        grandparent = EnhancedDocumentElement(
            element_type=grandparent_type,
            text="Chapter",
            line_number=1,
            start_position=0,
            end_position=7,
            level=0,
        )

        # Create parent
        parent_type = EnhancedElementType(element_type=ElementType.SECTION)
        parent = EnhancedDocumentElement(
            element_type=parent_type,
            text="Section",
            line_number=2,
            start_position=8,
            end_position=15,
            level=1,
            parent_id=grandparent.element_id,
        )

        # Create child
        child_type = EnhancedElementType(element_type=ElementType.SUBSECTION)
        child = EnhancedDocumentElement(
            element_type=child_type,
            text="Subsection",
            line_number=3,
            start_position=16,
            end_position=26,
            level=2,
            parent_id=parent.element_id,
        )

        grandparent.add_child(parent.element_id)
        parent.add_child(child.element_id)

        hierarchy.add_element(grandparent)
        hierarchy.add_element(parent)
        hierarchy.add_element(child)

        descendants = hierarchy.get_descendants(grandparent.element_id)

        assert len(descendants) == 2
        descendant_ids = [d.element_id for d in descendants]
        assert parent.element_id in descendant_ids
        assert child.element_id in descendant_ids

    def test_get_path_to_root(self):
        """Test getting path from element to root."""
        hierarchy = DocumentHierarchy(document_filename="test.pdf")

        # Create elements with parent-child relationships
        root_type = EnhancedElementType(element_type=ElementType.CHAPTER)
        root = EnhancedDocumentElement(
            element_type=root_type,
            text="Chapter",
            line_number=1,
            start_position=0,
            end_position=7,
            level=0,
        )

        child_type = EnhancedElementType(element_type=ElementType.SECTION)
        child = EnhancedDocumentElement(
            element_type=child_type,
            text="Section",
            line_number=2,
            start_position=8,
            end_position=15,
            level=1,
            parent_id=root.element_id,
        )

        grandchild_type = EnhancedElementType(element_type=ElementType.SUBSECTION)
        grandchild = EnhancedDocumentElement(
            element_type=grandchild_type,
            text="Subsection",
            line_number=3,
            start_position=16,
            end_position=26,
            level=2,
            parent_id=child.element_id,
        )

        hierarchy.add_element(root)
        hierarchy.add_element(child)
        hierarchy.add_element(grandchild)

        path = hierarchy.get_path_to_root(grandchild.element_id)

        assert len(path) == 3
        assert path[0].element_id == root.element_id
        assert path[1].element_id == child.element_id
        assert path[2].element_id == grandchild.element_id

    def test_get_elements_by_type(self):
        """Test getting elements by type."""
        hierarchy = DocumentHierarchy(document_filename="test.pdf")

        # Add different types of elements
        heading_type = EnhancedElementType(element_type=ElementType.HEADING)
        heading = EnhancedDocumentElement(
            element_type=heading_type,
            text="Heading",
            line_number=1,
            start_position=0,
            end_position=7,
        )

        paragraph_type = EnhancedElementType(element_type=ElementType.PARAGRAPH)
        paragraph = EnhancedDocumentElement(
            element_type=paragraph_type,
            text="Paragraph",
            line_number=2,
            start_position=8,
            end_position=17,
        )

        section_type = EnhancedElementType(element_type=ElementType.SECTION)
        section = EnhancedDocumentElement(
            element_type=section_type,
            text="Section",
            line_number=3,
            start_position=18,
            end_position=25,
        )

        hierarchy.add_element(heading)
        hierarchy.add_element(paragraph)
        hierarchy.add_element(section)

        headings = hierarchy.get_elements_by_type(ElementType.HEADING)
        paragraphs = hierarchy.get_elements_by_type(ElementType.PARAGRAPH)

        assert len(headings) == 1
        assert headings[0].element_id == heading.element_id
        assert len(paragraphs) == 1
        assert paragraphs[0].element_id == paragraph.element_id

    def test_get_elements_by_level(self):
        """Test getting elements by hierarchical level."""
        hierarchy = DocumentHierarchy(document_filename="test.pdf")

        # Add elements at different levels
        for level in [0, 1, 1, 2]:
            element_type = EnhancedElementType(element_type=ElementType.SECTION)
            element = EnhancedDocumentElement(
                element_type=element_type,
                text=f"Element at level {level}",
                line_number=level + 1,
                start_position=level * 10,
                end_position=(level + 1) * 10,
                level=level,
            )
            hierarchy.add_element(element)

        level_0 = hierarchy.get_elements_by_level(0)
        level_1 = hierarchy.get_elements_by_level(1)
        level_2 = hierarchy.get_elements_by_level(2)

        assert len(level_0) == 1
        assert len(level_1) == 2
        assert len(level_2) == 1

    def test_calculate_structure_confidence(self):
        """Test calculating structure confidence."""
        hierarchy = DocumentHierarchy(document_filename="test.pdf")

        # Add elements with different confidence scores
        element_type1 = EnhancedElementType(
            element_type=ElementType.HEADING, confidence_score=0.9
        )
        element1 = EnhancedDocumentElement(
            element_type=element_type1,
            text="Element 1",
            line_number=1,
            start_position=0,
            end_position=9,
        )

        element_type2 = EnhancedElementType(
            element_type=ElementType.PARAGRAPH, confidence_score=0.8
        )
        element2 = EnhancedDocumentElement(
            element_type=element_type2,
            text="Element 2",
            line_number=2,
            start_position=10,
            end_position=19,
        )

        hierarchy.add_element(element1)
        hierarchy.add_element(element2)

        confidence = hierarchy.calculate_structure_confidence()

        assert (
            abs(confidence - 0.85) < 0.0001
        )  # (0.9 + 0.8) / 2 with floating point tolerance

    def test_validate_hierarchy(self):
        """Test hierarchy validation."""
        hierarchy = DocumentHierarchy(document_filename="test.pdf")

        # Create valid elements
        element_type = EnhancedElementType(element_type=ElementType.SECTION)
        element = EnhancedDocumentElement(
            element_type=element_type,
            text="Element",
            line_number=1,
            start_position=0,
            end_position=7,
        )

        hierarchy.add_element(element)

        # Valid hierarchy should have no errors
        errors = hierarchy.validate_hierarchy()
        assert len(errors) == 0

        # Add element with missing parent
        orphan_type = EnhancedElementType(element_type=ElementType.SUBSECTION)
        orphan = EnhancedDocumentElement(
            element_type=orphan_type,
            text="Orphan",
            line_number=2,
            start_position=8,
            end_position=13,
            parent_id="nonexistent",
        )

        hierarchy.add_element(orphan)

        errors = hierarchy.validate_hierarchy()
        assert len(errors) == 1
        assert "missing parent" in errors[0]

    def test_to_dict_and_from_dict(self):
        """Test serialization to/from dictionary."""
        hierarchy = DocumentHierarchy(document_filename="test.pdf")

        element_type = EnhancedElementType(element_type=ElementType.HEADING)
        element = EnhancedDocumentElement(
            element_type=element_type,
            text="Heading",
            line_number=1,
            start_position=0,
            end_position=7,
        )

        hierarchy.add_element(element)

        # Convert to dict
        data = hierarchy.to_dict()

        assert data["document_filename"] == "test.pdf"
        assert data["total_elements"] == 1
        assert len(data["elements"]) == 1

        # Convert back from dict
        restored = DocumentHierarchy.from_dict(data)

        assert restored.document_filename == hierarchy.document_filename
        assert restored.total_elements == hierarchy.total_elements
        assert len(restored.elements) == len(hierarchy.elements)

    def test_from_base_elements(self):
        """Test creating hierarchy from base DocumentElement list."""
        # Create base elements
        base_parent = DocumentElement(
            element_type=ElementType.SECTION,
            text="Section",
            line_number=1,
            start_position=0,
            end_position=7,
            level=0,
        )

        base_child = DocumentElement(
            element_type=ElementType.SUBSECTION,
            text="Subsection",
            line_number=2,
            start_position=8,
            end_position=18,
            level=1,
            parent=base_parent,
        )

        base_parent.add_child(base_child)

        hierarchy = DocumentHierarchy.from_base_elements(
            [base_parent, base_child], "test.pdf"
        )

        assert hierarchy.document_filename == "test.pdf"
        assert hierarchy.total_elements == 2
        assert (
            len(hierarchy.relationships) >= 1
        )  # At least one parent-child relationship


class TestHierarchyUtilities:
    """Test cases for hierarchy utility functions."""

    def test_merge_hierarchies(self):
        """Test merging multiple hierarchies."""
        # Create first hierarchy
        hierarchy1 = DocumentHierarchy(document_filename="doc1.pdf")
        element_type1 = EnhancedElementType(element_type=ElementType.HEADING)
        element1 = EnhancedDocumentElement(
            element_type=element_type1,
            text="Element 1",
            line_number=1,
            start_position=0,
            end_position=9,
        )
        hierarchy1.add_element(element1)

        # Create second hierarchy
        hierarchy2 = DocumentHierarchy(document_filename="doc2.pdf")
        element_type2 = EnhancedElementType(element_type=ElementType.PARAGRAPH)
        element2 = EnhancedDocumentElement(
            element_type=element_type2,
            text="Element 2",
            line_number=1,
            start_position=0,
            end_position=9,
        )
        hierarchy2.add_element(element2)

        # Merge hierarchies
        merged = merge_hierarchies([hierarchy1, hierarchy2])

        assert merged.total_elements == 2
        assert "doc1.pdf_doc2.pdf" in merged.document_filename

    def test_merge_single_hierarchy(self):
        """Test merging single hierarchy returns original."""
        hierarchy = DocumentHierarchy(document_filename="test.pdf")
        merged = merge_hierarchies([hierarchy])
        assert merged is hierarchy

    def test_merge_empty_list_error(self):
        """Test merging empty list raises error."""
        with pytest.raises(ValueError, match="Cannot merge empty list"):
            merge_hierarchies([])

    def test_find_common_ancestors(self):
        """Test finding common ancestors."""
        hierarchy = DocumentHierarchy(document_filename="test.pdf")

        # Create hierarchy: root -> child1, child2 -> grandchild
        root_type = EnhancedElementType(element_type=ElementType.CHAPTER)
        root = EnhancedDocumentElement(
            element_type=root_type,
            text="Root",
            line_number=1,
            start_position=0,
            end_position=4,
            level=0,
        )

        child1_type = EnhancedElementType(element_type=ElementType.SECTION)
        child1 = EnhancedDocumentElement(
            element_type=child1_type,
            text="Child1",
            line_number=2,
            start_position=5,
            end_position=11,
            level=1,
            parent_id=root.element_id,
        )

        child2_type = EnhancedElementType(element_type=ElementType.SECTION)
        child2 = EnhancedDocumentElement(
            element_type=child2_type,
            text="Child2",
            line_number=3,
            start_position=12,
            end_position=18,
            level=1,
            parent_id=root.element_id,
        )

        grandchild_type = EnhancedElementType(element_type=ElementType.SUBSECTION)
        grandchild = EnhancedDocumentElement(
            element_type=grandchild_type,
            text="Grandchild",
            line_number=4,
            start_position=19,
            end_position=29,
            level=2,
            parent_id=child1.element_id,
        )

        hierarchy.add_element(root)
        hierarchy.add_element(child1)
        hierarchy.add_element(child2)
        hierarchy.add_element(grandchild)

        # Find common ancestors of child1 and child2
        ancestors = find_common_ancestors(
            hierarchy, [child1.element_id, child2.element_id]
        )
        assert root.element_id in ancestors

        # Find common ancestors of child1 and grandchild
        ancestors = find_common_ancestors(
            hierarchy, [child1.element_id, grandchild.element_id]
        )
        assert root.element_id in ancestors
        assert child1.element_id in ancestors

    def test_find_common_ancestors_single_element(self):
        """Test finding common ancestors with single element."""
        hierarchy = DocumentHierarchy(document_filename="test.pdf")
        ancestors = find_common_ancestors(hierarchy, ["elem1"])
        assert ancestors == ["elem1"]

    def test_find_common_ancestors_empty_list(self):
        """Test finding common ancestors with empty list."""
        hierarchy = DocumentHierarchy(document_filename="test.pdf")
        ancestors = find_common_ancestors(hierarchy, [])
        assert ancestors == []


if __name__ == "__main__":
    pytest.main([__file__])
