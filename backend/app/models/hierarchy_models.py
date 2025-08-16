"""
Enhanced hierarchy and relationship models for Langchain integration.

This module provides Langchain-compatible models for document hierarchy,
relationships, and structural elements that extend the existing document
structure detection capabilities with proper BaseModel integration.
"""

import logging
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.services.document_structure_detector.data_models import (
    DocumentElement,
    ElementType,
    NumberingSystem,
    NumberingType,
)

logger = logging.getLogger(__name__)


class EnhancedElementType(BaseModel):
    """Enhanced element type with Langchain BaseModel integration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    element_type: ElementType = Field(..., description="Type of document element")
    confidence_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Detection confidence"
    )
    detection_method: str = Field("manual", description="How the element was detected")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("detection_method")
    @classmethod
    def validate_detection_method(cls, v):
        """Validate detection method."""
        allowed_methods = {"manual", "regex", "ml", "rule_based", "hybrid"}
        if v.lower() not in allowed_methods:
            logger.warning(f"Unknown detection method: {v}")
        return v.lower()


class EnhancedNumberingSystem(BaseModel):
    """Enhanced numbering system with Langchain BaseModel integration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    numbering_type: NumberingType = Field(..., description="Type of numbering system")
    level: int = Field(..., ge=0, description="Hierarchical level")
    value: str = Field(..., description="Numbering value")
    raw_text: str = Field(..., description="Raw text of the numbering")
    pattern: str | None = Field(None, description="Regex pattern used for detection")

    # Enhanced fields
    system_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique system identifier"
    )
    parent_system_id: str | None = Field(None, description="Parent numbering system ID")
    confidence_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Detection confidence"
    )
    is_validated: bool = Field(False, description="Whether numbering is validated")
    validation_errors: list[str] = Field(
        default_factory=list, description="Validation errors"
    )

    def get_full_number(self, parent_numbers: list[str] | None = None) -> str:
        """Get the full hierarchical number including parent numbering."""
        if parent_numbers:
            return ".".join(parent_numbers + [self.value])
        return self.value

    def is_child_of(self, other: "EnhancedNumberingSystem") -> bool:
        """Check if this numbering is a child of another numbering system."""
        return self.level > other.level and self.parent_system_id == other.system_id

    @classmethod
    def from_base_numbering(
        cls, base_numbering: NumberingSystem, **kwargs
    ) -> "EnhancedNumberingSystem":
        """Create enhanced numbering from base NumberingSystem."""
        return cls(
            numbering_type=base_numbering.numbering_type,
            level=base_numbering.level,
            value=base_numbering.value,
            raw_text=base_numbering.raw_text,
            pattern=base_numbering.pattern,
            **kwargs,
        )


class EnhancedDocumentElement(BaseModel):
    """Enhanced document element with Langchain BaseModel integration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    element_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique element identifier"
    )
    element_type: EnhancedElementType = Field(..., description="Enhanced element type")
    text: str = Field(..., min_length=1, description="Element text content")
    line_number: int = Field(..., ge=1, description="Line number in document")
    start_position: int = Field(..., ge=0, description="Start character position")
    end_position: int = Field(..., gt=0, description="End character position")
    level: int = Field(0, ge=0, description="Hierarchical level")

    # Relationships
    parent_id: str | None = Field(None, description="Parent element ID")
    child_ids: list[str] = Field(default_factory=list, description="Child element IDs")
    numbering: EnhancedNumberingSystem | None = Field(
        None, description="Element numbering"
    )

    # Enhanced fields
    chunk_references: list[str] = Field(
        default_factory=list, description="Referenced chunk IDs"
    )
    semantic_role: str | None = Field(None, description="Semantic role in document")
    importance_score: float = Field(0.0, ge=0.0, le=1.0, description="Importance score")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("end_position")
    @classmethod
    def validate_positions(cls, v, info):
        """Validate that end_position > start_position."""
        if (
            hasattr(info, "data")
            and "start_position" in info.data
            and v <= info.data["start_position"]
        ):
            raise ValueError("end_position must be greater than start_position")
        return v

    @field_validator("semantic_role")
    @classmethod
    def validate_semantic_role(cls, v):
        """Validate semantic role."""
        if v is not None:
            allowed_roles = {
                "title",
                "heading",
                "introduction",
                "conclusion",
                "summary",
                "definition",
                "example",
                "note",
                "warning",
                "reference",
                "citation",
                "footnote",
                "table",
                "figure",
                "equation",
            }
            if v.lower() not in allowed_roles:
                logger.warning(f"Unknown semantic role: {v}")
            return v.lower()
        return v

    def add_child(self, child_id: str) -> None:
        """Add a child element ID."""
        if child_id not in self.child_ids:
            self.child_ids.append(child_id)

    def remove_child(self, child_id: str) -> None:
        """Remove a child element ID."""
        if child_id in self.child_ids:
            self.child_ids.remove(child_id)

    def add_chunk_reference(self, chunk_id: str) -> None:
        """Add a chunk reference."""
        if chunk_id not in self.chunk_references:
            self.chunk_references.append(chunk_id)

    def get_hierarchical_path(
        self, element_registry: dict[str, "EnhancedDocumentElement"]
    ) -> list[str]:
        """Get the hierarchical path from root to this element."""
        path: list[str] = []
        current_id: str | None = self.element_id

        while current_id:
            if current_id not in element_registry:
                break

            element = element_registry[current_id]
            if element.numbering:
                path.insert(0, element.numbering.get_full_number())
            else:
                path.insert(0, element.element_type.element_type.value)

            current_id = element.parent_id

        return path

    @classmethod
    def from_base_element(
        cls, base_element: DocumentElement, **kwargs
    ) -> "EnhancedDocumentElement":
        """Create enhanced element from base DocumentElement."""
        enhanced_type = EnhancedElementType(
            element_type=base_element.element_type,
            detection_method="structure_detector",
            confidence_score=0.8,  # Default confidence for structure detector
        )

        enhanced_numbering = None
        if base_element.numbering:
            enhanced_numbering = EnhancedNumberingSystem.from_base_numbering(
                base_element.numbering
            )

        return cls(
            element_type=enhanced_type,
            text=base_element.text,
            line_number=base_element.line_number,
            start_position=base_element.start_position,
            end_position=base_element.end_position,
            level=base_element.level,
            numbering=enhanced_numbering,
            metadata=base_element.metadata,
            **kwargs,
        )


class DocumentRelationship(BaseModel):
    """Model representing relationships between document elements."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    relationship_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique relationship identifier",
    )
    source_element_id: str = Field(..., description="Source element ID")
    target_element_id: str = Field(..., description="Target element ID")
    relationship_type: str = Field(..., description="Type of relationship")
    confidence_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Confidence in relationship"
    )
    bidirectional: bool = Field(
        False, description="Whether relationship is bidirectional"
    )

    # Additional relationship metadata
    distance: int | None = Field(
        None, ge=0, description="Distance between elements (e.g., sentence count)"
    )
    semantic_similarity: float | None = Field(
        None, ge=0.0, le=1.0, description="Semantic similarity score"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("relationship_type")
    @classmethod
    def validate_relationship_type(cls, v):
        """Validate relationship type."""
        allowed_types = {
            "parent_child",
            "sibling",
            "reference",
            "definition",
            "example",
            "continuation",
            "elaboration",
            "contrast",
            "sequence",
            "dependency",
            "cross_reference",
            "citation",
        }
        if v.lower() not in allowed_types:
            logger.warning(f"Unknown relationship type: {v}")
        return v.lower()

    def get_reverse_relationship(self) -> Optional["DocumentRelationship"]:
        """Get the reverse relationship if this is bidirectional."""
        if not self.bidirectional:
            return None

        return DocumentRelationship(
            source_element_id=self.target_element_id,
            target_element_id=self.source_element_id,
            relationship_type=self._get_reverse_type(),
            confidence_score=self.confidence_score,
            bidirectional=True,
            distance=self.distance,
            semantic_similarity=self.semantic_similarity,
            metadata=self.metadata.copy(),
        )

    def _get_reverse_type(self) -> str:
        """Get the reverse relationship type."""
        reverse_map = {
            "parent_child": "child_parent",
            "child_parent": "parent_child",
            "reference": "referenced_by",
            "referenced_by": "reference",
            "definition": "defined_by",
            "defined_by": "definition",
        }
        return reverse_map.get(self.relationship_type, self.relationship_type)


class DocumentHierarchy(BaseModel):
    """Model representing the complete document hierarchy."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hierarchy_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique hierarchy identifier"
    )
    document_filename: str = Field(..., description="Source document filename")
    root_element_ids: list[str] = Field(
        default_factory=list, description="Root element IDs"
    )
    elements: dict[str, EnhancedDocumentElement] = Field(
        default_factory=dict, description="All elements by ID"
    )
    relationships: dict[str, DocumentRelationship] = Field(
        default_factory=dict, description="All relationships by ID"
    )

    # Hierarchy metadata
    max_depth: int = Field(0, ge=0, description="Maximum hierarchy depth")
    total_elements: int = Field(0, ge=0, description="Total number of elements")
    structure_confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Overall structure confidence"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def add_element(self, element: EnhancedDocumentElement) -> None:
        """Add an element to the hierarchy."""
        self.elements[element.element_id] = element

        # Update root elements if no parent
        if not element.parent_id and element.element_id not in self.root_element_ids:
            self.root_element_ids.append(element.element_id)

        # Update statistics
        self.total_elements = len(self.elements)
        self.max_depth = max(self.max_depth, element.level)
        self.updated_at = datetime.utcnow()

    def add_relationship(self, relationship: DocumentRelationship) -> None:
        """Add a relationship to the hierarchy."""
        self.relationships[relationship.relationship_id] = relationship

        # Add reverse relationship if bidirectional
        if relationship.bidirectional:
            reverse_rel = relationship.get_reverse_relationship()
            if reverse_rel:
                self.relationships[reverse_rel.relationship_id] = reverse_rel

        self.updated_at = datetime.utcnow()

    def get_element(self, element_id: str) -> EnhancedDocumentElement | None:
        """Get an element by ID."""
        return self.elements.get(element_id)

    def get_children(self, element_id: str) -> list[EnhancedDocumentElement]:
        """Get all child elements of a given element."""
        element = self.get_element(element_id)
        if not element:
            return []

        return [
            self.elements[child_id]
            for child_id in element.child_ids
            if child_id in self.elements
        ]

    def get_descendants(self, element_id: str) -> list[EnhancedDocumentElement]:
        """Get all descendant elements of a given element."""
        descendants = []
        children = self.get_children(element_id)

        for child in children:
            descendants.append(child)
            descendants.extend(self.get_descendants(child.element_id))

        return descendants

    def get_path_to_root(self, element_id: str) -> list[EnhancedDocumentElement]:
        """Get the path from an element to the root."""
        path: list[EnhancedDocumentElement] = []
        current_id: str | None = element_id
        visited = set()  # Track visited elements to detect circular references

        while current_id and current_id in self.elements:
            if current_id in visited:
                # Circular reference detected - break to avoid infinite loop
                logger.warning(f"Circular reference detected at element {current_id}")
                break

            visited.add(current_id)
            element = self.elements[current_id]
            path.insert(0, element)
            current_id = element.parent_id

        return path

    def _has_circular_reference(self, element_id: str) -> bool:
        """Check if an element has a circular reference in its parent chain."""
        visited = set()
        current_id: str | None = element_id

        while current_id and current_id in self.elements:
            if current_id in visited:
                return True  # Circular reference found
            visited.add(current_id)
            current_id = self.elements[current_id].parent_id

        return False

    def get_elements_by_type(
        self, element_type: ElementType
    ) -> list[EnhancedDocumentElement]:
        """Get all elements of a specific type."""
        return [
            element
            for element in self.elements.values()
            if element.element_type.element_type == element_type
        ]

    def get_elements_by_level(self, level: int) -> list[EnhancedDocumentElement]:
        """Get all elements at a specific hierarchical level."""
        return [element for element in self.elements.values() if element.level == level]

    def calculate_structure_confidence(self) -> float:
        """Calculate overall structure confidence score."""
        if not self.elements:
            return 0.0

        # Average confidence across all elements and numbering systems
        element_confidences = []

        for element in self.elements.values():
            element_confidences.append(element.element_type.confidence_score)
            if element.numbering:
                element_confidences.append(element.numbering.confidence_score)

        if element_confidences:
            self.structure_confidence = sum(element_confidences) / len(
                element_confidences
            )
        else:
            self.structure_confidence = 0.0

        return self.structure_confidence

    def validate_hierarchy(self) -> list[str]:
        """Validate the hierarchy for consistency and return error messages."""
        errors = []

        # Check for orphaned elements
        for element in self.elements.values():
            if element.parent_id and element.parent_id not in self.elements:
                errors.append(
                    f"Element {element.element_id} has missing parent {element.parent_id}"
                )

        # Check for circular references
        for element in self.elements.values():
            if self._has_circular_reference(element.element_id):
                errors.append(
                    f"Circular reference detected in path to {element.element_id}"
                )

        # Check relationship consistency
        for relationship in self.relationships.values():
            if (
                relationship.source_element_id not in self.elements
                or relationship.target_element_id not in self.elements
            ):
                errors.append(
                    f"Relationship {relationship.relationship_id} references missing elements"
                )

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert hierarchy to dictionary for serialization."""
        return {
            "hierarchy_id": self.hierarchy_id,
            "document_filename": self.document_filename,
            "root_element_ids": self.root_element_ids,
            "elements": {k: v.model_dump() for k, v in self.elements.items()},
            "relationships": {k: v.model_dump() for k, v in self.relationships.items()},
            "max_depth": self.max_depth,
            "total_elements": self.total_elements,
            "structure_confidence": self.structure_confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentHierarchy":
        """Create hierarchy from dictionary."""
        hierarchy = cls(
            hierarchy_id=data["hierarchy_id"],
            document_filename=data["document_filename"],
            root_element_ids=data["root_element_ids"],
            max_depth=data["max_depth"],
            total_elements=data["total_elements"],
            structure_confidence=data["structure_confidence"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )

        # Reconstruct elements
        for element_id, element_data in data["elements"].items():
            hierarchy.elements[element_id] = EnhancedDocumentElement(**element_data)

        # Reconstruct relationships
        for rel_id, rel_data in data["relationships"].items():
            hierarchy.relationships[rel_id] = DocumentRelationship(**rel_data)

        return hierarchy

    @classmethod
    def from_base_elements(
        cls, base_elements: list[DocumentElement], document_filename: str
    ) -> "DocumentHierarchy":
        """Create hierarchy from base DocumentElement list."""
        hierarchy = cls(
            document_filename=document_filename,
            max_depth=0,
            total_elements=0,
            structure_confidence=0.0,
        )

        # Convert base elements to enhanced elements
        for base_element in base_elements:
            enhanced_element = EnhancedDocumentElement.from_base_element(base_element)
            hierarchy.add_element(enhanced_element)

            # Create parent-child relationships
            if base_element.parent:
                # Find parent in already converted elements
                parent_enhanced = None
                for elem in hierarchy.elements.values():
                    if (
                        elem.text == base_element.parent.text
                        and elem.line_number == base_element.parent.line_number
                    ):
                        parent_enhanced = elem
                        break

                if parent_enhanced:
                    enhanced_element.parent_id = parent_enhanced.element_id
                    parent_enhanced.add_child(enhanced_element.element_id)

                    # Create relationship
                    relationship = DocumentRelationship(
                        source_element_id=parent_enhanced.element_id,
                        target_element_id=enhanced_element.element_id,
                        relationship_type="parent_child",
                        confidence_score=0.95,
                        bidirectional=True,
                        distance=0,  # Parent-child are directly connected
                        semantic_similarity=0.9,  # High similarity for parent-child
                    )
                    hierarchy.add_relationship(relationship)

        hierarchy.calculate_structure_confidence()
        return hierarchy


# Utility functions for hierarchy management


def merge_hierarchies(hierarchies: list[DocumentHierarchy]) -> DocumentHierarchy:
    """Merge multiple document hierarchies into one."""
    if not hierarchies:
        raise ValueError("Cannot merge empty list of hierarchies")

    if len(hierarchies) == 1:
        return hierarchies[0]

    # Create new merged hierarchy
    merged = DocumentHierarchy(
        document_filename="merged_"
        + "_".join(h.document_filename for h in hierarchies[:3]),
        max_depth=0,
        total_elements=0,
        structure_confidence=0.0,
    )

    # Merge all elements and relationships
    for hierarchy in hierarchies:
        for element in hierarchy.elements.values():
            merged.add_element(element)

        for relationship in hierarchy.relationships.values():
            merged.add_relationship(relationship)

    merged.calculate_structure_confidence()
    return merged


def find_common_ancestors(
    hierarchy: DocumentHierarchy, element_ids: list[str]
) -> list[str]:
    """Find common ancestors of multiple elements."""
    if not element_ids:
        return []

    if len(element_ids) == 1:
        return [element_ids[0]]

    # Get paths to root for all elements
    paths = []
    for element_id in element_ids:
        path = hierarchy.get_path_to_root(element_id)
        paths.append([e.element_id for e in path])

    # Find common prefixes
    common_ancestors = []
    if paths:
        min_length = min(len(path) for path in paths)
        for i in range(min_length):
            if all(path[i] == paths[0][i] for path in paths):
                common_ancestors.append(paths[0][i])
            else:
                break

    return common_ancestors
