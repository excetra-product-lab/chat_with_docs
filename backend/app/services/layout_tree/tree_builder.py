"""Tree building utilities for layout tree construction.

This module provides algorithms and utilities for building tree structures
from various data sources and formats, including optimized bulk operations
and specialized tree construction patterns.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..document_structure_detector.data_models import (
    DocumentElement,
    DocumentStructure,
    ElementType,
)
from .layout_tree import LayoutTree
from .metadata_manager import MetadataManager
from .tree_node import TreeNode

logger = logging.getLogger(__name__)


class TreeBuildStrategy(Enum):
    """Strategies for building trees from flat data."""

    DEPTH_FIRST = "depth_first"
    BREADTH_FIRST = "breadth_first"
    BALANCED = "balanced"
    HIERARCHICAL = "hierarchical"
    LEVEL_ORDER = "level_order"


@dataclass
class TreeBuildConfig:
    """Configuration for tree building operations."""

    strategy: TreeBuildStrategy = TreeBuildStrategy.HIERARCHICAL
    balance_threshold: float = 0.7
    max_children_per_node: int = 100
    enable_metadata_caching: bool = True
    validate_during_build: bool = True
    auto_rebalance: bool = False
    preserve_order: bool = True


class TreeBuilder:
    """Advanced tree building system for layout trees.

    This class provides comprehensive tree building capabilities including
    construction from various data sources, optimization algorithms, and
    specialized building patterns for different use cases.
    """

    def __init__(self, config: TreeBuildConfig | None = None):
        """Initialize the tree builder.

        Args:
            config: Optional configuration for tree building
        """
        self.config = config or TreeBuildConfig()
        self.metadata_manager = MetadataManager()
        self.logger = logging.getLogger(__name__)

    def build_from_document_structure(
        self, doc_structure: DocumentStructure
    ) -> LayoutTree:
        """Build a layout tree from a DocumentStructure.

        Args:
            doc_structure: DocumentStructure to convert

        Returns:
            LayoutTree: Constructed tree
        """
        if not doc_structure.elements:
            return LayoutTree()

        # Use the optimized method from LayoutTree
        tree = LayoutTree.from_document_structure(doc_structure)

        # Apply post-processing based on configuration
        if self.config.auto_rebalance:
            tree = self.rebalance_tree(tree)

        if self.config.validate_during_build:
            self._validate_tree_structure(tree)

        return tree

    def build_from_flat_list(
        self, elements: list[DocumentElement], strategy: TreeBuildStrategy | None = None
    ) -> LayoutTree:
        """Build a tree from a flat list of document elements.

        Args:
            elements: List of document elements
            strategy: Optional strategy override

        Returns:
            LayoutTree: Constructed tree
        """
        if not elements:
            return LayoutTree()

        build_strategy = strategy or self.config.strategy

        if build_strategy == TreeBuildStrategy.HIERARCHICAL:
            return self._build_hierarchical(elements)
        elif build_strategy == TreeBuildStrategy.LEVEL_ORDER:
            return self._build_level_order(elements)
        elif build_strategy == TreeBuildStrategy.BALANCED:
            return self._build_balanced(elements)
        elif build_strategy == TreeBuildStrategy.DEPTH_FIRST:
            return self._build_depth_first(elements)
        elif build_strategy == TreeBuildStrategy.BREADTH_FIRST:
            return self._build_breadth_first(elements)
        else:
            raise ValueError(f"Unsupported build strategy: {build_strategy}")

    def build_from_node_data(self, node_data: list[dict[str, Any]]) -> LayoutTree:
        """Build a tree from raw node data.

        Args:
            node_data: List of dictionaries containing node information

        Returns:
            LayoutTree: Constructed tree
        """
        # Convert dictionaries to DocumentElement objects
        elements = []
        for data in node_data:
            element = self._create_element_from_data(data)
            elements.append(element)

        return self.build_from_flat_list(elements)

    def build_subtree(
        self,
        parent: TreeNode,
        child_elements: list[DocumentElement],
        strategy: TreeBuildStrategy | None = None,
    ) -> None:
        """Build a subtree under a specific parent node.

        Args:
            parent: Parent node to attach subtree to
            child_elements: Elements to build subtree from
            strategy: Optional strategy override
        """
        if not child_elements:
            return

        # Create child tree
        child_tree = self.build_from_flat_list(child_elements, strategy)

        if child_tree.root:
            # Attach the root of the child tree to the parent
            parent.add_child(child_tree.root)

    def bulk_insert(
        self,
        tree: LayoutTree,
        elements: list[DocumentElement],
        parent_finder: Callable[[DocumentElement], TreeNode | None] = None,
    ) -> int:
        """Perform bulk insertion of elements into an existing tree.

        Args:
            tree: Target tree for insertion
            elements: Elements to insert
            parent_finder: Function to find parent for each element

        Returns:
            int: Number of elements successfully inserted
        """
        inserted_count = 0

        for element in elements:
            try:
                # Find parent node
                parent = None
                if parent_finder:
                    parent = parent_finder(element)
                elif element.parent:
                    # Find parent by line number or other criteria
                    parent = tree.find_node_by_line(element.parent.line_number)

                # Create new node
                new_node = TreeNode(value=element)

                if parent:
                    parent.add_child(new_node)
                elif not tree.root:
                    tree.root = new_node
                else:
                    # Attach to root if no better parent found
                    tree.root.add_child(new_node)

                inserted_count += 1

            except Exception as e:
                self.logger.warning("Failed to insert element: %s", e)

        return inserted_count

    def merge_trees(
        self, tree1: LayoutTree, tree2: LayoutTree, merge_strategy: str = "append"
    ) -> LayoutTree:
        """Merge two trees into a single tree.

        Args:
            tree1: First tree
            tree2: Second tree
            merge_strategy: Strategy for merging ("append", "interleave", "balanced")

        Returns:
            LayoutTree: Merged tree
        """
        if tree1.is_empty:
            return tree2
        if tree2.is_empty:
            return tree1

        if merge_strategy == "append":
            # Add tree2's root as a child of tree1's root
            if tree2.root:
                tree1.root.add_child(tree2.root)
            return tree1

        elif merge_strategy == "balanced":
            # Create new root and balance the trees
            new_root = TreeNode()
            new_root.metadata["synthetic"] = True
            new_root.add_child(tree1.root)
            new_root.add_child(tree2.root)

            result_tree = LayoutTree(root=new_root)
            return self.rebalance_tree(result_tree)

        elif merge_strategy == "interleave":
            # Interleave nodes based on some criteria (e.g., line numbers)
            return self._interleave_trees(tree1, tree2)

        else:
            raise ValueError(f"Unsupported merge strategy: {merge_strategy}")

    def split_tree(
        self, tree: LayoutTree, split_condition: Callable[[TreeNode], bool]
    ) -> tuple[LayoutTree, LayoutTree]:
        """Split a tree into two trees based on a condition.

        Args:
            tree: Tree to split
            split_condition: Function that returns True for nodes to move to second tree

        Returns:
            Tuple[LayoutTree, LayoutTree]: Two resulting trees
        """
        if tree.is_empty:
            return LayoutTree(), LayoutTree()

        tree1_nodes = []
        tree2_nodes = []

        # Collect nodes based on condition
        for node in tree.traverse_preorder():
            if split_condition(node):
                tree2_nodes.append(node)
            else:
                tree1_nodes.append(node)

        # Build new trees
        tree1 = self._build_tree_from_nodes(tree1_nodes)
        tree2 = self._build_tree_from_nodes(tree2_nodes)

        return tree1, tree2

    def rebalance_tree(self, tree: LayoutTree) -> LayoutTree:
        """Rebalance a tree to improve performance.

        Args:
            tree: Tree to rebalance

        Returns:
            LayoutTree: Rebalanced tree
        """
        if tree.is_empty:
            return tree

        # Collect all nodes in level order
        nodes = list(tree.traverse_levelorder())

        # Extract DocumentElements
        elements = [node.value for node in nodes if node.value]

        # Rebuild using balanced strategy
        return self._build_balanced(elements)

    def _build_hierarchical(self, elements: list[DocumentElement]) -> LayoutTree:
        """Build tree using hierarchical relationships."""
        tree = LayoutTree()

        if not elements:
            return tree

        # Create mapping of element IDs to nodes
        element_to_node = {id(element): TreeNode(value=element) for element in elements}

        # Build hierarchy based on parent-child relationships
        root_candidates = []

        for element in elements:
            node = element_to_node[id(element)]

            if element.parent and id(element.parent) in element_to_node:
                parent_node = element_to_node[id(element.parent)]
                parent_node.add_child(node)
            else:
                root_candidates.append(node)

        # Handle root selection
        if len(root_candidates) == 1:
            tree.root = root_candidates[0]
        elif len(root_candidates) > 1:
            # Create synthetic root
            synthetic_root = TreeNode()
            synthetic_root.metadata["synthetic"] = True
            for candidate in root_candidates:
                synthetic_root.add_child(candidate)
            tree.root = synthetic_root

        return tree

    def _build_level_order(self, elements: list[DocumentElement]) -> LayoutTree:
        """Build tree in level order based on element levels."""
        if not elements:
            return LayoutTree()

        # Sort elements by level
        sorted_elements = sorted(elements, key=lambda e: (e.level, e.line_number))

        # Group by level
        levels = {}
        for element in sorted_elements:
            level = element.level
            if level not in levels:
                levels[level] = []
            levels[level].append(element)

        # Build tree level by level
        tree = LayoutTree()
        level_nodes = {}

        for level in sorted(levels.keys()):
            level_nodes[level] = []

            for element in levels[level]:
                node = TreeNode(value=element)
                level_nodes[level].append(node)

                if level == 0:
                    # Root level
                    if not tree.root:
                        tree.root = node
                    else:
                        # Multiple roots - create synthetic root
                        if not tree.root.metadata.get("synthetic"):
                            old_root = tree.root
                            tree.root = TreeNode()
                            tree.root.metadata["synthetic"] = True
                            tree.root.add_child(old_root)
                        tree.root.add_child(node)
                else:
                    # Find appropriate parent from previous level
                    parent_level = level - 1
                    if parent_level in level_nodes and level_nodes[parent_level]:
                        # Use last node from parent level as default parent
                        parent = level_nodes[parent_level][-1]
                        parent.add_child(node)

        return tree

    def _build_balanced(self, elements: list[DocumentElement]) -> LayoutTree:
        """Build a balanced tree from elements."""
        if not elements:
            return LayoutTree()

        # Sort elements by line number for consistent ordering
        sorted_elements = sorted(elements, key=lambda e: e.line_number)

        def build_balanced_recursive(
            elem_list: list[DocumentElement],
        ) -> TreeNode | None:
            if not elem_list:
                return None

            mid = len(elem_list) // 2
            root = TreeNode(value=elem_list[mid])

            # Build left and right subtrees
            left_subtree = build_balanced_recursive(elem_list[:mid])
            right_subtree = build_balanced_recursive(elem_list[mid + 1 :])

            if left_subtree:
                root.add_child(left_subtree)
            if right_subtree:
                root.add_child(right_subtree)

            return root

        tree = LayoutTree()
        tree.root = build_balanced_recursive(sorted_elements)
        return tree

    def _build_depth_first(self, elements: list[DocumentElement]) -> LayoutTree:
        """Build tree using depth-first construction."""
        if not elements:
            return LayoutTree()

        # Sort by level and line number
        sorted_elements = sorted(elements, key=lambda e: (e.level, e.line_number))

        tree = LayoutTree()
        stack = []

        for element in sorted_elements:
            node = TreeNode(value=element)

            # Find appropriate parent based on level
            while stack and stack[-1].level >= element.level:
                stack.pop()

            if stack:
                stack[-1].add_child(node)
            else:
                # This is a root
                if not tree.root:
                    tree.root = node
                else:
                    # Multiple roots - create synthetic root
                    if not tree.root.metadata.get("synthetic"):
                        old_root = tree.root
                        tree.root = TreeNode()
                        tree.root.metadata["synthetic"] = True
                        tree.root.add_child(old_root)
                    tree.root.add_child(node)

            stack.append(node)

        return tree

    def _build_breadth_first(self, elements: list[DocumentElement]) -> LayoutTree:
        """Build tree using breadth-first construction."""
        if not elements:
            return LayoutTree()

        # Sort by level then line number
        sorted_elements = sorted(elements, key=lambda e: (e.level, e.line_number))

        tree = LayoutTree()
        nodes_by_level = {}

        # Group nodes by level
        for element in sorted_elements:
            level = element.level
            if level not in nodes_by_level:
                nodes_by_level[level] = []

            node = TreeNode(value=element)
            nodes_by_level[level].append(node)

        # Build tree level by level
        current_level_parents = []

        for level in sorted(nodes_by_level.keys()):
            if level == 0:
                # Root level
                for node in nodes_by_level[level]:
                    if not tree.root:
                        tree.root = node
                        current_level_parents = [node]
                    else:
                        # Multiple roots - attach to existing root
                        tree.root.add_child(node)
                        current_level_parents.append(node)
            else:
                # Distribute children among parents from previous level
                parents = (
                    current_level_parents
                    if current_level_parents
                    else [tree.root]
                    if tree.root
                    else []
                )
                children = nodes_by_level[level]
                next_level_parents = []

                if parents and children:
                    children_per_parent = len(children) // len(parents)
                    remainder = len(children) % len(parents)

                    child_idx = 0
                    for i, parent in enumerate(parents):
                        # Calculate how many children this parent gets
                        num_children = children_per_parent + (1 if i < remainder else 0)

                        for _ in range(num_children):
                            if child_idx < len(children):
                                parent.add_child(children[child_idx])
                                next_level_parents.append(children[child_idx])
                                child_idx += 1

                # Update parents for next level
                current_level_parents = next_level_parents

        return tree

    def _build_tree_from_nodes(self, nodes: list[TreeNode]) -> LayoutTree:
        """Build a tree from existing TreeNode objects."""
        if not nodes:
            return LayoutTree()

        # Extract elements and rebuild
        elements = [node.value for node in nodes if node.value]
        return self.build_from_flat_list(elements)

    def _interleave_trees(self, tree1: LayoutTree, tree2: LayoutTree) -> LayoutTree:
        """Interleave two trees based on element properties."""
        # Collect all nodes from both trees
        nodes1 = list(tree1.traverse_preorder()) if tree1.root else []
        nodes2 = list(tree2.traverse_preorder()) if tree2.root else []

        # Sort all nodes by line number
        all_nodes = nodes1 + nodes2
        all_nodes.sort(key=lambda n: n.value.line_number if n.value else 0)

        # Rebuild tree from sorted nodes
        elements = [node.value for node in all_nodes if node.value]
        return self.build_from_flat_list(elements)

    def _create_element_from_data(self, data: dict[str, Any]) -> DocumentElement:
        """Create a DocumentElement from dictionary data."""
        element = DocumentElement(
            element_type=ElementType(data.get("element_type", "heading")),
            text=data.get("text", ""),
            line_number=data.get("line_number", 0),
            start_position=data.get("start_position", 0),
            end_position=data.get("end_position", 0),
            level=data.get("level", 0),
        )

        # Add any additional metadata
        if "metadata" in data:
            element.metadata.update(data["metadata"])

        return element

    def _validate_tree_structure(self, tree: LayoutTree) -> None:
        """Validate the constructed tree structure."""
        if tree.is_empty:
            return

        validation_result = self.metadata_manager.validate_metadata(
            tree.root, recursive=True
        )

        if not validation_result.is_valid:
            self.logger.warning(
                "Tree validation found %d errors", len(validation_result.errors)
            )
            for error in validation_result.errors[:5]:  # Log first 5 errors
                self.logger.warning("Validation error: %s", error)
