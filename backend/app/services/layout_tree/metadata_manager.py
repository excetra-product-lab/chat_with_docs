"""Metadata management utilities for layout tree nodes.

This module provides enhanced metadata handling capabilities including
dynamic calculation, propagation, validation, and performance optimization.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .tree_node import TreeNode

logger = logging.getLogger(__name__)


@dataclass
class MetadataValidationResult:
    """Result of metadata validation operation."""

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validated_nodes: int = 0

    def add_error(self, node: TreeNode, message: str) -> None:
        """Add a validation error."""
        error_msg = f"Node {id(node)}: {message}"
        self.errors.append(error_msg)
        self.is_valid = False

    def add_warning(self, node: TreeNode, message: str) -> None:
        """Add a validation warning."""
        warning_msg = f"Node {id(node)}: {message}"
        self.warnings.append(warning_msg)


class MetadataValidator(ABC):
    """Abstract base class for metadata validators."""

    @abstractmethod
    def validate(self, node: TreeNode) -> list[str]:
        """Validate metadata for a single node.

        Args:
            node: Node to validate

        Returns:
            List[str]: List of validation error messages
        """
        pass


class DepthConsistencyValidator(MetadataValidator):
    """Validator for depth consistency in tree nodes."""

    def validate(self, node: TreeNode) -> list[str]:
        """Validate that node depth is consistent with tree structure."""
        errors = []

        # Check parent-child depth relationship
        if node.parent:
            expected_depth = node.parent.depth + 1
            if node.depth != expected_depth:
                errors.append(
                    f"Depth inconsistency: expected {expected_depth}, got {node.depth}"
                )
        else:
            # Root node should have depth 0
            if node.depth != 0:
                errors.append(f"Root node depth should be 0, got {node.depth}")

        # Check children depth consistency
        for child in node.children:
            expected_child_depth = node.depth + 1
            if child.depth != expected_child_depth:
                errors.append(
                    f"Child depth inconsistency: expected {expected_child_depth}, got {child.depth}"
                )

        return errors


class HeightConsistencyValidator(MetadataValidator):
    """Validator for height consistency in tree nodes."""

    def validate(self, node: TreeNode) -> list[str]:
        """Validate that node height is consistent with subtree structure."""
        errors = []

        if node.is_leaf:
            if node.height != 0:
                errors.append(f"Leaf node height should be 0, got {node.height}")
        else:
            expected_height = max(child.height for child in node.children) + 1
            if node.height != expected_height:
                errors.append(
                    f"Height inconsistency: expected {expected_height}, got {node.height}"
                )

        return errors


class DocumentElementConsistencyValidator(MetadataValidator):
    """Validator for DocumentElement consistency."""

    def validate(self, node: TreeNode) -> list[str]:
        """Validate DocumentElement relationships."""
        errors = []

        if node.value:
            # Check that DocumentElement parent-child relationships match TreeNode relationships
            if node.parent and node.parent.value:
                if node.value.parent != node.parent.value:
                    errors.append(
                        "DocumentElement parent relationship doesn't match TreeNode parent"
                    )

            # Check children relationships
            if len(node.value.children) != len(node.children):
                errors.append(
                    f"DocumentElement children count ({len(node.value.children)}) doesn't match TreeNode children count ({len(node.children)})"
                )

        return errors


class MetadataManager:
    """Advanced metadata management system for layout tree nodes.

    This class provides comprehensive metadata handling including validation,
    propagation, caching, and performance optimization for tree operations.
    """

    def __init__(self):
        """Initialize the metadata manager."""
        self.logger = logging.getLogger(__name__)
        self._validators: list[MetadataValidator] = [
            DepthConsistencyValidator(),
            HeightConsistencyValidator(),
            DocumentElementConsistencyValidator(),
        ]
        self._cached_metadata: dict[int, dict[str, Any]] = {}
        self._invalidated_nodes: set[int] = set()
        self._metadata_calculators: dict[str, Callable[[TreeNode], Any]] = {}

        # Register default metadata calculators
        self._register_default_calculators()

    def _register_default_calculators(self) -> None:
        """Register default metadata calculation functions."""
        self._metadata_calculators.update(
            {
                "subtree_size": lambda node: node.size,
                "subtree_height": lambda node: node.height,
                "leaf_count": self._calculate_leaf_count,
                "element_type_distribution": self._calculate_element_type_distribution,
                "max_child_depth": self._calculate_max_child_depth,
                "path_to_root_length": lambda node: len(node.get_path_to_root()),
                "sibling_count": lambda node: len(node.parent.children) - 1
                if node.parent
                else 0,
                "is_balanced": self._calculate_balance_factor,
            }
        )

    def add_validator(self, validator: MetadataValidator) -> None:
        """Add a custom metadata validator.

        Args:
            validator: Validator to add
        """
        self._validators.append(validator)

    def remove_validator(self, validator: MetadataValidator) -> None:
        """Remove a metadata validator.

        Args:
            validator: Validator to remove
        """
        if validator in self._validators:
            self._validators.remove(validator)

    def register_metadata_calculator(
        self, name: str, calculator: Callable[[TreeNode], Any]
    ) -> None:
        """Register a custom metadata calculation function.

        Args:
            name: Name of the metadata field
            calculator: Function that calculates the metadata value
        """
        self._metadata_calculators[name] = calculator

    def calculate_metadata(
        self, node: TreeNode, metadata_name: str, force_recalculate: bool = False
    ) -> Any:
        """Calculate a specific metadata value for a node.

        Args:
            node: Node to calculate metadata for
            metadata_name: Name of the metadata to calculate
            force_recalculate: Whether to force recalculation even if cached

        Returns:
            Any: Calculated metadata value
        """
        node_id = id(node)

        # Check cache first
        if not force_recalculate and node_id not in self._invalidated_nodes:
            cached_values = self._cached_metadata.get(node_id, {})
            if metadata_name in cached_values:
                return cached_values[metadata_name]

        # Calculate the metadata
        if metadata_name not in self._metadata_calculators:
            raise ValueError(f"No calculator registered for metadata '{metadata_name}'")

        calculator = self._metadata_calculators[metadata_name]
        value = calculator(node)

        # Cache the result
        if node_id not in self._cached_metadata:
            self._cached_metadata[node_id] = {}
        self._cached_metadata[node_id][metadata_name] = value

        # Remove from invalidated nodes
        self._invalidated_nodes.discard(node_id)

        return value

    def invalidate_node_cache(self, node: TreeNode) -> None:
        """Invalidate cached metadata for a node.

        Args:
            node: Node to invalidate cache for
        """
        node_id = id(node)
        self._invalidated_nodes.add(node_id)
        if node_id in self._cached_metadata:
            del self._cached_metadata[node_id]

    def invalidate_subtree_cache(self, node: TreeNode) -> None:
        """Invalidate cached metadata for an entire subtree.

        Args:
            node: Root of subtree to invalidate
        """
        self.invalidate_node_cache(node)
        for child in node.children:
            self.invalidate_subtree_cache(child)

    def propagate_metadata_changes(
        self, node: TreeNode, changed_fields: list[str] | None = None
    ) -> None:
        """Propagate metadata changes up the tree hierarchy.

        Args:
            node: Node where changes occurred
            changed_fields: Specific fields that changed (None for all)
        """
        current = node
        while current:
            self.invalidate_node_cache(current)

            # Update dynamic metadata that depends on children
            if changed_fields is None or any(
                field in ["subtree_size", "subtree_height", "leaf_count"]
                for field in changed_fields
            ):
                # Recalculate size, height, and other dependent metadata
                current.metadata.update(
                    {
                        "last_updated": self._get_current_timestamp(),
                        "cache_invalidated": True,
                    }
                )

            current = current.parent

    def validate_metadata(
        self, node: TreeNode, recursive: bool = True
    ) -> MetadataValidationResult:
        """Validate metadata consistency for a node and optionally its subtree.

        Args:
            node: Node to validate
            recursive: Whether to validate the entire subtree

        Returns:
            MetadataValidationResult: Validation results
        """
        result = MetadataValidationResult()

        nodes_to_validate = [node]
        if recursive:
            nodes_to_validate.extend(self._get_all_descendants(node))

        for validate_node in nodes_to_validate:
            result.validated_nodes += 1

            for validator in self._validators:
                errors = validator.validate(validate_node)
                for error in errors:
                    result.add_error(validate_node, error)

        return result

    def update_node_metadata(
        self, node: TreeNode, updates: dict[str, Any], propagate: bool = True
    ) -> None:
        """Update metadata for a node with optional propagation.

        Args:
            node: Node to update
            updates: Dictionary of metadata updates
            propagate: Whether to propagate changes up the tree
        """
        # Update the node's metadata
        node.metadata.update(updates)
        node.metadata["last_modified"] = self._get_current_timestamp()

        # Invalidate cache
        self.invalidate_node_cache(node)

        # Propagate changes if requested
        if propagate:
            self.propagate_metadata_changes(node, list(updates.keys()))

    def get_metadata_summary(self, node: TreeNode) -> dict[str, Any]:
        """Get a comprehensive metadata summary for a node.

        Args:
            node: Node to get summary for

        Returns:
            Dict[str, Any]: Comprehensive metadata summary
        """
        summary = {
            "basic_properties": {
                "depth": node.depth,
                "height": node.height,
                "is_leaf": node.is_leaf,
                "is_root": node.is_root,
                "child_count": len(node.children),
            },
            "calculated_metadata": {},
            "custom_metadata": dict(node.metadata),
        }

        # Calculate all registered metadata
        for metadata_name in self._metadata_calculators:
            try:
                summary["calculated_metadata"][metadata_name] = self.calculate_metadata(
                    node, metadata_name
                )
            except Exception as e:
                self.logger.warning(
                    "Failed to calculate metadata '%s' for node: %s", metadata_name, e
                )
                summary["calculated_metadata"][metadata_name] = None

        return summary

    def _calculate_leaf_count(self, node: TreeNode) -> int:
        """Calculate the number of leaf nodes in the subtree."""
        if node.is_leaf:
            return 1
        return sum(self._calculate_leaf_count(child) for child in node.children)

    def _calculate_element_type_distribution(self, node: TreeNode) -> dict[str, int]:
        """Calculate element type distribution in the subtree."""
        distribution = {}
        stack = [node]

        while stack:
            current = stack.pop()
            if current.element_type:
                type_name = current.element_type.value
                distribution[type_name] = distribution.get(type_name, 0) + 1
            stack.extend(current.children)

        return distribution

    def _calculate_max_child_depth(self, node: TreeNode) -> int:
        """Calculate the maximum depth of any child in the subtree."""
        if node.is_leaf:
            return node.depth

        max_depth = node.depth
        for child in node.children:
            child_max = self._calculate_max_child_depth(child)
            max_depth = max(max_depth, child_max)

        return max_depth

    def _calculate_balance_factor(self, node: TreeNode) -> float:
        """Calculate a balance factor for the subtree (0.0 = perfectly balanced, 1.0 = completely unbalanced)."""
        if node.is_leaf:
            return 0.0

        if len(node.children) == 0:
            return 0.0

        # Calculate heights of children
        child_heights = [child.height for child in node.children]
        min_height = min(child_heights)
        max_height = max(child_heights)

        if max_height == 0:
            return 0.0

        # Balance factor based on height difference
        return (max_height - min_height) / max_height

    def _get_all_descendants(self, node: TreeNode) -> list[TreeNode]:
        """Get all descendants of a node."""
        descendants = []
        for child in node.children:
            descendants.append(child)
            descendants.extend(self._get_all_descendants(child))
        return descendants

    def _get_current_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime

        return datetime.now().isoformat()

    def clear_cache(self) -> None:
        """Clear all cached metadata."""
        self._cached_metadata.clear()
        self._invalidated_nodes.clear()

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get statistics about the metadata cache.

        Returns:
            Dict[str, Any]: Cache statistics
        """
        return {
            "cached_nodes": len(self._cached_metadata),
            "invalidated_nodes": len(self._invalidated_nodes),
            "total_cached_entries": sum(
                len(metadata) for metadata in self._cached_metadata.values()
            ),
            "registered_calculators": list(self._metadata_calculators.keys()),
            "active_validators": len(self._validators),
        }
