"""Layout tree class for managing document hierarchical structure.

This module provides the LayoutTree class that manages collections of TreeNode
objects and provides high-level tree operations for document structure representation.
"""

import logging
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from ..document_structure_detector.data_models import (
    DocumentStructure,
    ElementType,
)
from .tree_node import TreeNode

logger = logging.getLogger(__name__)


@dataclass
class TreeStatistics:
    """Statistics about tree structure and composition."""

    total_nodes: int = 0
    max_depth: int = 0
    max_height: int = 0
    leaf_count: int = 0
    element_type_counts: dict[str, int] = field(default_factory=dict)
    level_distribution: dict[int, int] = field(default_factory=dict)
    average_branching_factor: float = 0.0
    balance_factor: float = 0.0


class LayoutTree:
    """Main tree class for managing document layout structure.

    This class provides a comprehensive interface for building, manipulating,
    and analyzing hierarchical tree structures representing document layouts.
    It integrates with the existing document structure detection system.
    """

    def __init__(
        self, root: TreeNode | None = None, metadata: dict[str, Any] | None = None
    ):
        """Initialize the layout tree.

        Args:
            root: Optional root node for the tree
            metadata: Optional metadata for the tree
        """
        self._root = root
        self.metadata = metadata or {}
        self._node_registry: dict[str, TreeNode] = {}
        self._change_listeners: list[Callable[[str, TreeNode], None]] = []
        self.logger = logging.getLogger(__name__)

        if root:
            self._register_subtree(root)

    @property
    def root(self) -> TreeNode | None:
        """Get the root node of the tree."""
        return self._root

    @root.setter
    def root(self, node: TreeNode | None) -> None:
        """Set the root node of the tree.

        Args:
            node: The new root node
        """
        if self._root:
            self._unregister_subtree(self._root)

        self._root = node
        if node:
            node.parent = None  # Ensure root has no parent
            self._register_subtree(node)

        self._notify_listeners("root_changed", node)

    @property
    def is_empty(self) -> bool:
        """Check if the tree is empty."""
        return self._root is None

    @property
    def size(self) -> int:
        """Get the total number of nodes in the tree."""
        return self._root.size if self._root else 0

    @property
    def height(self) -> int:
        """Get the height of the tree."""
        return self._root.height if self._root else 0

    @property
    def max_depth(self) -> int:
        """Get the maximum depth of any node in the tree."""
        if not self._root:
            return 0

        max_depth = 0
        for node in self.traverse_preorder():
            max_depth = max(max_depth, node.depth)
        return max_depth

    def add_change_listener(self, listener: Callable[[str, TreeNode], None]) -> None:
        """Add a listener for tree changes.

        Args:
            listener: Function to call when tree changes occur
        """
        self._change_listeners.append(listener)

    def remove_change_listener(self, listener: Callable[[str, TreeNode], None]) -> None:
        """Remove a change listener.

        Args:
            listener: The listener function to remove
        """
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)

    def _notify_listeners(self, event_type: str, node: TreeNode | None) -> None:
        """Notify all change listeners of an event.

        Args:
            event_type: Type of event that occurred
            node: Node involved in the event
        """
        for listener in self._change_listeners:
            try:
                if node is not None:
                    listener(event_type, node)
            except Exception as e:
                self.logger.error("Error in change listener: %s", e)

    def _register_node(self, node: TreeNode) -> None:
        """Register a node in the internal registry.

        Args:
            node: The node to register
        """
        if node.value and hasattr(node.value, "line_number"):
            key = f"line_{node.value.line_number}"
            self._node_registry[key] = node

        # Register by element type and position if available
        if node.value and node.element_type:
            type_key = f"{node.element_type.value}_{id(node)}"
            self._node_registry[type_key] = node

    def _unregister_node(self, node: TreeNode) -> None:
        """Unregister a node from the internal registry.

        Args:
            node: The node to unregister
        """
        keys_to_remove = []
        for key, registered_node in self._node_registry.items():
            if registered_node == node:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._node_registry[key]

    def _register_subtree(self, node: TreeNode) -> None:
        """Register all nodes in a subtree.

        Args:
            node: Root of the subtree to register
        """
        self._register_node(node)
        for child in node.children:
            self._register_subtree(child)

    def _unregister_subtree(self, node: TreeNode) -> None:
        """Unregister all nodes in a subtree.

        Args:
            node: Root of the subtree to unregister
        """
        self._unregister_node(node)
        for child in node.children:
            self._unregister_subtree(child)

    def find_node_by_line(self, line_number: int) -> TreeNode | None:
        """Find a node by its line number.

        Args:
            line_number: Line number to search for

        Returns:
            TreeNode: The node at the given line, or None if not found
        """
        key = f"line_{line_number}"
        return self._node_registry.get(key)

    def find_nodes_by_type(self, element_type: ElementType) -> list[TreeNode]:
        """Find all nodes of a specific element type.

        Args:
            element_type: The element type to search for

        Returns:
            List[TreeNode]: All nodes matching the element type
        """
        if not self._root:
            return []

        matching_nodes = []
        for node in self.traverse_preorder():
            if node.element_type == element_type:
                matching_nodes.append(node)
        return matching_nodes

    def find_nodes_by_text(
        self, text: str, exact_match: bool = False
    ) -> list[TreeNode]:
        """Find nodes containing specific text.

        Args:
            text: Text to search for
            exact_match: Whether to require exact text match

        Returns:
            List[TreeNode]: Nodes containing the text
        """
        if not self._root:
            return []

        matching_nodes = []
        search_text = text.lower() if not exact_match else text

        for node in self.traverse_preorder():
            node_text = node.text if exact_match else node.text.lower()
            if (exact_match and node_text == search_text) or (
                not exact_match and search_text in node_text
            ):
                matching_nodes.append(node)

        return matching_nodes

    def traverse_preorder(self) -> Iterator[TreeNode]:
        """Traverse the tree in pre-order (root, left, right).

        Yields:
            TreeNode: Nodes in pre-order sequence
        """
        if not self._root:
            return

        stack = [self._root]
        while stack:
            node = stack.pop()
            yield node
            # Add children in reverse order so they're processed left-to-right
            for child in reversed(node.children):
                stack.append(child)

    def traverse_postorder(self) -> Iterator[TreeNode]:
        """Traverse the tree in post-order (left, right, root).

        Yields:
            TreeNode: Nodes in post-order sequence
        """
        if not self._root:
            return

        def _postorder_recursive(node: TreeNode) -> Iterator[TreeNode]:
            for child in node.children:
                yield from _postorder_recursive(child)
            yield node

        yield from _postorder_recursive(self._root)

    def traverse_levelorder(self) -> Iterator[TreeNode]:
        """Traverse the tree level by level (breadth-first).

        Yields:
            TreeNode: Nodes in level-order sequence
        """
        if not self._root:
            return

        queue = deque([self._root])
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node.children)

    def get_nodes_at_level(self, level: int) -> list[TreeNode]:
        """Get all nodes at a specific depth level.

        Args:
            level: The depth level (0-based)

        Returns:
            List[TreeNode]: Nodes at the specified level
        """
        if not self._root or level < 0:
            return []

        nodes_at_level = []
        for node in self.traverse_levelorder():
            if node.depth == level:
                nodes_at_level.append(node)
            elif node.depth > level:
                break  # We've gone past the target level

        return nodes_at_level

    def get_leaf_nodes(self) -> list[TreeNode]:
        """Get all leaf nodes in the tree.

        Returns:
            List[TreeNode]: All leaf nodes
        """
        if not self._root:
            return []

        return [node for node in self.traverse_preorder() if node.is_leaf]

    def prune_subtree(self, node: TreeNode) -> bool:
        """Remove a subtree rooted at the given node.

        Args:
            node: Root of the subtree to remove

        Returns:
            bool: True if the subtree was removed, False otherwise
        """
        if node == self._root:
            self.root = None
            return True

        if node.parent:
            self._unregister_subtree(node)
            result = node.parent.remove_child(node)
            if result:
                self._notify_listeners("subtree_pruned", node)
            return result

        return False

    def graft_subtree(self, parent: TreeNode, subtree: TreeNode) -> bool:
        """Attach a subtree to a parent node.

        Args:
            parent: The parent node to attach to
            subtree: The subtree to attach

        Returns:
            bool: True if the graft was successful, False otherwise
        """
        if not parent or not subtree:
            return False

        # Check if parent is actually in this tree
        if not any(node == parent for node in self.traverse_preorder()):
            return False

        parent.add_child(subtree)
        self._register_subtree(subtree)
        self._notify_listeners("subtree_grafted", subtree)
        return True

    def calculate_statistics(self) -> TreeStatistics:
        """Calculate comprehensive statistics about the tree.

        Returns:
            TreeStatistics: Statistics about the tree structure
        """
        stats = TreeStatistics()

        if not self._root:
            return stats

        # Count nodes and analyze structure
        level_counts: dict[int, int] = {}
        element_type_counts: dict[str, int] = {}
        non_leaf_children_counts = []

        for node in self.traverse_preorder():
            stats.total_nodes += 1

            # Track depth
            stats.max_depth = max(stats.max_depth, node.depth)
            level_counts[node.depth] = level_counts.get(node.depth, 0) + 1

            # Track element types
            if node.element_type:
                type_name = node.element_type.value
                element_type_counts[type_name] = (
                    element_type_counts.get(type_name, 0) + 1
                )

            # Track leaf nodes
            if node.is_leaf:
                stats.leaf_count += 1
            else:
                non_leaf_children_counts.append(len(node.children))

        stats.max_height = self.height
        stats.level_distribution = level_counts
        stats.element_type_counts = element_type_counts

        # Calculate average branching factor
        if non_leaf_children_counts:
            stats.average_branching_factor = sum(non_leaf_children_counts) / len(
                non_leaf_children_counts
            )

        # Calculate balance factor (simplified metric)
        if stats.total_nodes > 1:
            # Ratio of actual depth to optimal depth for balanced tree
            optimal_depth = stats.total_nodes.bit_length() - 1
            stats.balance_factor = (
                optimal_depth / stats.max_depth if stats.max_depth > 0 else 1.0
            )

        return stats

    def to_dict(self, include_statistics: bool = False) -> dict[str, Any]:
        """Convert the tree to a dictionary representation.

        Args:
            include_statistics: Whether to include tree statistics

        Returns:
            Dict: Dictionary representation of the tree
        """
        result = {
            "is_empty": self.is_empty,
            "size": self.size,
            "height": self.height,
            "max_depth": self.max_depth,
            "metadata": self.metadata,
        }

        if self._root:
            result["root"] = self._root.to_dict(include_children=True)

        if include_statistics:
            stats = self.calculate_statistics()
            result["statistics"] = {
                "total_nodes": stats.total_nodes,
                "max_depth": stats.max_depth,
                "max_height": stats.max_height,
                "leaf_count": stats.leaf_count,
                "element_type_counts": stats.element_type_counts,
                "level_distribution": stats.level_distribution,
                "average_branching_factor": stats.average_branching_factor,
                "balance_factor": stats.balance_factor,
            }

        return result

    @classmethod
    def from_document_structure(cls, doc_structure: DocumentStructure) -> "LayoutTree":
        """Create a LayoutTree from a DocumentStructure.

        Args:
            doc_structure: DocumentStructure to convert

        Returns:
            LayoutTree: New tree built from the document structure
        """
        tree = cls()

        if not doc_structure.elements:
            return tree

        # Build a mapping of element IDs to nodes
        element_to_node = {}

        # Create nodes for all elements
        for element in doc_structure.elements:
            node = TreeNode(value=element)
            element_to_node[id(element)] = node

        # Build parent-child relationships
        root_candidates = []

        for element in doc_structure.elements:
            node = element_to_node[id(element)]

            if element.parent and id(element.parent) in element_to_node:
                # Has a parent - add as child
                parent_node = element_to_node[id(element.parent)]
                parent_node.add_child(node)
            else:
                # No parent - potential root
                root_candidates.append(node)

        # Handle root selection
        if len(root_candidates) == 1:
            tree.root = root_candidates[0]
        elif len(root_candidates) > 1:
            # Multiple root candidates - create a synthetic root
            synthetic_root = TreeNode()
            synthetic_root.metadata["synthetic"] = True

            for candidate in root_candidates:
                synthetic_root.add_child(candidate)

            tree.root = synthetic_root

        # Copy metadata from document structure
        tree.metadata.update(doc_structure.metadata)

        return tree

    def __str__(self) -> str:
        """String representation of the tree."""
        if self.is_empty:
            return "LayoutTree(empty)"

        return f"LayoutTree(size={self.size}, height={self.height}, root_type={self._root.element_type if self._root else None})"

    def __repr__(self) -> str:
        """Detailed string representation of the tree."""
        return self.__str__()
