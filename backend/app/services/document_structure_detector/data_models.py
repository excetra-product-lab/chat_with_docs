"""Data models for document structure detection.

This module contains enums and data classes used for representing
document structure elements, numbering systems, and hierarchical relationships.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ElementType(Enum):
    """Types of document elements that can be detected."""

    HEADING = "heading"
    SECTION = "section"
    SUBSECTION = "subsection"
    CLAUSE = "clause"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    ARTICLE = "article"
    CHAPTER = "chapter"


class NumberingType(Enum):
    """Types of numbering systems used in legal documents."""

    DECIMAL = "decimal"  # 1.2.3
    ROMAN_UPPER = "roman_upper"  # I, II, III
    ROMAN_LOWER = "roman_lower"  # i, ii, iii
    LETTER_UPPER = "letter_upper"  # A, B, C
    LETTER_LOWER = "letter_lower"  # a, b, c
    SECTION_SYMBOL = "section_symbol"  # §
    MIXED = "mixed"  # Mixed numbering systems


@dataclass
class NumberingSystem:
    """Represents a numbering system with its format and level."""

    numbering_type: NumberingType
    level: int
    value: str
    raw_text: str
    pattern: str | None = None
    parent_numbering: Optional["NumberingSystem"] = None

    def __post_init__(self):
        """Validate numbering system after initialization."""
        if self.level < 0:
            raise ValueError("Numbering level cannot be negative")

    def get_full_number(self) -> str:
        """Get the full hierarchical number including parent numbering."""
        if self.parent_numbering:
            return f"{self.parent_numbering.get_full_number()}.{self.value}"
        return self.value

    def is_child_of(self, other: "NumberingSystem") -> bool:
        """Check if this numbering is a child of another numbering system."""
        return self.level > other.level and self.raw_text.startswith(other.raw_text)


@dataclass
class DocumentElement:
    """Base class for all document structural elements."""

    element_type: ElementType
    text: str
    line_number: int
    start_position: int
    end_position: int
    level: int = 0
    numbering: NumberingSystem | None = None
    parent: Optional["DocumentElement"] = None
    children: list["DocumentElement"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate element after initialization."""
        if self.start_position < 0 or self.end_position < 0:
            raise ValueError("Positions cannot be negative")
        if self.start_position >= self.end_position:
            raise ValueError("Start position must be less than end position")
        if self.level < 0:
            raise ValueError("Level cannot be negative")

    def add_child(self, child: "DocumentElement") -> None:
        """Add a child element and set parent relationship."""
        child.parent = self
        child.level = self.level + 1
        self.children.append(child)

    def get_full_text(self) -> str:
        """Get the full text including all children."""
        texts = [self.text]
        for child in self.children:
            texts.append(child.get_full_text())
        return "\n".join(texts)

    def get_hierarchy_path(self) -> list[str]:
        """Get the path from root to this element."""
        path: list[str] = []
        current: DocumentElement | None = self
        while current:
            if current.numbering:
                path.insert(0, current.numbering.get_full_number())
            else:
                path.insert(0, current.element_type.value)
            current = current.parent
        return path

    def is_ancestor_of(self, other: "DocumentElement") -> bool:
        """Check if this element is an ancestor of another element."""
        current = other.parent
        while current:
            if current == self:
                return True
            current = current.parent
        return False


@dataclass
class Section(DocumentElement):
    """Represents a section in a legal document."""

    section_number: str | None = None
    title: str | None = None

    def __post_init__(self):
        """Initialize section with proper element type."""
        super().__post_init__()
        if self.element_type not in [ElementType.SECTION, ElementType.SUBSECTION]:
            self.element_type = ElementType.SECTION


@dataclass
class Heading(DocumentElement):
    """Represents a heading or subheading in a legal document."""

    heading_level: int = 1
    is_numbered: bool = False

    def __post_init__(self):
        """Initialize heading with proper element type."""
        super().__post_init__()
        # Only set default element_type if none was provided or if it's None
        if self.element_type is None:
            self.element_type = ElementType.HEADING
        if self.heading_level < 1:
            raise ValueError("Heading level must be at least 1")


@dataclass
class DocumentStructure:
    """Container for the complete hierarchical structure of a document."""

    elements: list[DocumentElement] = field(default_factory=list)
    headings: list["Heading"] = field(
        default_factory=list
    )  # Add explicit headings field
    numbering_systems: list[NumberingSystem] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_element(self, element: DocumentElement) -> None:
        """Add an element to the document structure."""
        self.elements.append(element)
        if element.numbering and element.numbering not in self.numbering_systems:
            self.numbering_systems.append(element.numbering)

    def get_elements_by_type(self, element_type: ElementType) -> list[DocumentElement]:
        """Get all elements of a specific type."""
        return [elem for elem in self.elements if elem.element_type == element_type]

    def get_sections(self) -> list[Section]:
        """Get all sections in the document."""
        return [elem for elem in self.elements if isinstance(elem, Section)]

    def get_headings(self) -> list[Heading]:
        """Get all headings in the document."""
        return [elem for elem in self.elements if isinstance(elem, Heading)]

    def get_element_by_numbering(self, numbering: str) -> DocumentElement | None:
        """Find an element by its numbering."""
        for element in self.elements:
            if element.numbering and element.numbering.get_full_number() == numbering:
                return element
        return None

    def get_max_level(self) -> int:
        """Get the maximum nesting level in the document."""
        if not self.elements:
            return 0
        return max(elem.level for elem in self.elements)

    def to_dict(self) -> dict:
        """Convert the document structure to a dictionary representation."""
        return {
            "total_elements": len(self.elements),
            "max_level": self.get_max_level(),
            "element_types": {
                element_type.value: len(self.get_elements_by_type(element_type))
                for element_type in ElementType
            },
            "numbering_systems": [
                {"type": ns.numbering_type.value, "level": ns.level, "value": ns.value}
                for ns in self.numbering_systems
            ],
            "metadata": self.metadata,
        }
