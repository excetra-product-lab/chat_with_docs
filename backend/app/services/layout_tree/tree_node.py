"""Tree node class for layout tree generation.

This module provides the TreeNode class that serves as the building block
for the hierarchical tree structure representing document layout.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ..document_structure_detector.data_models import DocumentElement, ElementType

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """A tree node that wraps document elements and provides tree operations.

    This class serves as the building block for the layout tree structure,
    providing efficient tree operations while maintaining integration with
    the existing document structure detection system.
    """

    value: DocumentElement | None = None
    children: list["TreeNode"] = field(default_factory=list)
    parent: Optional["TreeNode"] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize the tree node after creation."""
        # Set parent relationships for existing children
        for child in self.children:
            child.parent = self

    @property
    def element_type(self) -> ElementType | None:
        """Get the element type from the wrapped DocumentElement."""
        return self.value.element_type if self.value else None

    @property
    def text(self) -> str:
        """Get the text content from the wrapped DocumentElement."""
        return self.value.text if self.value else ""

    @property
    def level(self) -> int:
        """Get the hierarchical level from the wrapped DocumentElement or calculate from tree."""
        if self.value and hasattr(self.value, "level"):
            return self.value.level
        return self.depth

    @property
    def depth(self) -> int:
        """Calculate the depth of this node in the tree (0-based)."""
        depth = 0
        current = self.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth

    @property
    def height(self) -> int:
        """Calculate the height of the subtree rooted at this node."""
        if not self.children:
            return 0
        return max(child.height for child in self.children) + 1

    @property
    def is_root(self) -> bool:
        """Check if this node is the root of the tree."""
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (has no children)."""
        return len(self.children) == 0

    @property
    def size(self) -> int:
        """Get the total number of nodes in the subtree rooted at this node."""
        size = 1  # Count this node
        for child in self.children:
            size += child.size
        return size

    def add_child(self, child: "TreeNode") -> None:
        """Add a child node to this node.

        Args:
            child: The child node to add
        """
        if child.parent is not None:
            child.parent.remove_child(child)

        child.parent = self
        self.children.append(child)

        # Update DocumentElement relationship if both nodes have values
        if self.value and child.value:
            child.value.parent = self.value
            if child.value not in self.value.children:
                self.value.children.append(child.value)

    def remove_child(self, child: "TreeNode") -> bool:
        """Remove a child node from this node.

        Args:
            child: The child node to remove

        Returns:
            bool: True if the child was removed, False if not found
        """
        if child in self.children:
            child.parent = None
            self.children.remove(child)

            # Update DocumentElement relationship if both nodes have values
            if self.value and child.value:
                child.value.parent = None
                if child.value in self.value.children:
                    self.value.children.remove(child.value)

            return True
        return False

    def get_child_by_index(self, index: int) -> Optional["TreeNode"]:
        """Get a child node by its index.

        Args:
            index: Index of the child (0-based)

        Returns:
            TreeNode: The child node at the given index, or None if invalid
        """
        if 0 <= index < len(self.children):
            return self.children[index]
        return None

    def get_children_by_type(self, element_type: ElementType) -> list["TreeNode"]:
        """Get all direct children of a specific element type.

        Args:
            element_type: The element type to filter by

        Returns:
            List[TreeNode]: Children matching the element type
        """
        return [child for child in self.children if child.element_type == element_type]

    def find_descendants_by_type(self, element_type: ElementType) -> list["TreeNode"]:
        """Find all descendants (not just direct children) of a specific type.

        Args:
            element_type: The element type to search for

        Returns:
            List[TreeNode]: All descendants matching the element type
        """
        descendants = []
        for child in self.children:
            if child.element_type == element_type:
                descendants.append(child)
            descendants.extend(child.find_descendants_by_type(element_type))
        return descendants

    def get_path_to_root(self) -> list["TreeNode"]:
        """Get the path from this node to the root.

        Returns:
            List[TreeNode]: Path from this node to root (inclusive)
        """
        path = []
        current: TreeNode | None = self
        while current is not None:
            path.append(current)
            current = current.parent
        return path

    def get_common_ancestor(self, other: "TreeNode") -> Optional["TreeNode"]:
        """Find the lowest common ancestor with another node.

        Args:
            other: The other node to find common ancestor with

        Returns:
            TreeNode: The lowest common ancestor, or None if no common ancestor
        """
        self_path = set(self.get_path_to_root())
        other_path = other.get_path_to_root()

        for node in other_path:
            if node in self_path:
                return node
        return None

    def is_ancestor_of(self, other: "TreeNode") -> bool:
        """Check if this node is an ancestor of another node.

        Args:
            other: The node to check

        Returns:
            bool: True if this node is an ancestor of the other node
        """
        current = other.parent
        while current is not None:
            if current == self:
                return True
            current = current.parent
        return False

    def is_descendant_of(self, other: "TreeNode") -> bool:
        """Check if this node is a descendant of another node.

        Args:
            other: The node to check

        Returns:
            bool: True if this node is a descendant of the other node
        """
        return other.is_ancestor_of(self)

    def clone(self, deep: bool = True) -> "TreeNode":
        """Create a copy of this node.

        Args:
            deep: If True, recursively clone all children

        Returns:
            TreeNode: A copy of this node
        """
        cloned_node = TreeNode(
            value=self.value,  # Shallow copy of the DocumentElement
            metadata=self.metadata.copy(),
        )

        if deep:
            for child in self.children:
                cloned_child = child.clone(deep=True)
                cloned_node.add_child(cloned_child)

        return cloned_node

    def to_dict(self, include_children: bool = True) -> dict[str, Any]:
        """Convert the node to a dictionary representation.

        Args:
            include_children: Whether to include children in the output

        Returns:
            Dict: Dictionary representation of the node
        """
        result: dict[str, Any] = {
            "element_type": self.element_type.value if self.element_type else None,
            "text": self.text,
            "level": self.level,
            "depth": self.depth,
            "height": self.height,
            "is_leaf": self.is_leaf,
            "child_count": len(self.children),
            "metadata": self.metadata,
        }

        if self.value:
            result.update(
                {
                    "start_position": self.value.start_position,
                    "end_position": self.value.end_position,
                    "line_number": self.value.line_number,
                    "numbering": str(self.value.numbering)
                    if self.value.numbering
                    else None,
                }
            )

        if include_children and self.children:
            result["children"] = [
                child.to_dict(include_children=True) for child in self.children
            ]

        return result

    def __str__(self) -> str:
        """String representation of the node."""
        element_type = self.element_type.value if self.element_type else "None"
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"TreeNode(type={element_type}, text='{text_preview}', children={len(self.children)})"

    def __repr__(self) -> str:
        """Detailed string representation of the node."""
        return self.__str__()
