"""Layout tree generation system for document structure.

This package provides tools for generating and manipulating layout trees
that represent the hierarchical structure of legal documents.
"""

from .layout_tree import LayoutTree, TreeStatistics
from .metadata_manager import (
    DepthConsistencyValidator,
    DocumentElementConsistencyValidator,
    HeightConsistencyValidator,
    MetadataManager,
    MetadataValidationResult,
    MetadataValidator,
)
from .tree_builder import TreeBuildConfig, TreeBuilder, TreeBuildStrategy
from .tree_node import TreeNode

__all__ = [
    "TreeNode",
    "LayoutTree",
    "TreeStatistics",
    "MetadataManager",
    "MetadataValidator",
    "MetadataValidationResult",
    "DepthConsistencyValidator",
    "HeightConsistencyValidator",
    "DocumentElementConsistencyValidator",
    "TreeBuilder",
    "TreeBuildStrategy",
    "TreeBuildConfig",
]
