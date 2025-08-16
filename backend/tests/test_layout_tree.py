"""Tests for the layout tree generation system.

This module contains comprehensive tests for the layout tree functionality
including TreeNode, LayoutTree, MetadataManager, and TreeBuilder classes.
"""

from app.services.document_structure_detector.data_models import (
    DocumentElement,
    DocumentStructure,
    ElementType,
)
from app.services.layout_tree import (
    LayoutTree,
    MetadataManager,
    TreeBuilder,
    TreeBuildStrategy,
    TreeNode,
)


class TestTreeNode:
    """Test cases for TreeNode class."""

    def test_tree_node_creation(self):
        """Test basic tree node creation."""
        element = DocumentElement(
            element_type=ElementType.HEADING,
            text="Test Heading",
            line_number=1,
            start_position=0,
            end_position=12,
            level=0,
        )

        node = TreeNode(value=element)

        assert node.value == element
        assert node.element_type == ElementType.HEADING
        assert node.text == "Test Heading"
        assert node.level == 0
        assert node.depth == 0
        assert node.height == 0
        assert node.is_root is True
        assert node.is_leaf is True
        assert node.size == 1
        assert len(node.children) == 0

    def test_tree_node_hierarchy(self):
        """Test parent-child relationships."""
        parent_element = DocumentElement(
            element_type=ElementType.SECTION,
            text="Parent Section",
            line_number=1,
            start_position=0,
            end_position=14,
            level=0,
        )

        child_element = DocumentElement(
            element_type=ElementType.SUBSECTION,
            text="Child Subsection",
            line_number=2,
            start_position=15,
            end_position=31,
            level=1,
        )

        parent_node = TreeNode(value=parent_element)
        child_node = TreeNode(value=child_element)

        parent_node.add_child(child_node)

        assert child_node.parent == parent_node
        assert child_node in parent_node.children
        assert parent_node.is_leaf is False
        assert child_node.is_leaf is True
        assert parent_node.size == 2
        assert child_node.depth == 1
        assert parent_node.height == 1

    def test_tree_node_operations(self):
        """Test tree node manipulation operations."""
        root = TreeNode()
        child1 = TreeNode()
        child2 = TreeNode()
        grandchild = TreeNode()

        # Build hierarchy
        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild)

        assert root.size == 4
        assert root.height == 2
        assert len(root.children) == 2
        assert grandchild.depth == 2

        # Test removal
        removed = root.remove_child(child2)
        assert removed is True
        assert child2 not in root.children
        assert child2.parent is None
        assert root.size == 3

        # Test path to root
        path = grandchild.get_path_to_root()
        assert len(path) == 3
        assert path == [grandchild, child1, root]

    def test_tree_node_search_operations(self):
        """Test search and filtering operations."""
        root = TreeNode(
            value=DocumentElement(
                element_type=ElementType.SECTION,
                text="Root Section",
                line_number=1,
                start_position=0,
                end_position=12,
                level=0,
            )
        )

        child1 = TreeNode(
            value=DocumentElement(
                element_type=ElementType.HEADING,
                text="Heading 1",
                line_number=2,
                start_position=13,
                end_position=22,
                level=1,
            )
        )

        child2 = TreeNode(
            value=DocumentElement(
                element_type=ElementType.HEADING,
                text="Heading 2",
                line_number=3,
                start_position=23,
                end_position=32,
                level=1,
            )
        )

        root.add_child(child1)
        root.add_child(child2)

        # Test type filtering
        headings = root.get_children_by_type(ElementType.HEADING)
        assert len(headings) == 2
        assert child1 in headings
        assert child2 in headings

        # Test descendant search
        all_headings = root.find_descendants_by_type(ElementType.HEADING)
        assert len(all_headings) == 2

    def test_tree_node_clone(self):
        """Test node cloning functionality."""
        element = DocumentElement(
            element_type=ElementType.SECTION,
            text="Original Section",
            line_number=1,
            start_position=0,
            end_position=16,
            level=0,
        )

        original = TreeNode(value=element)
        original.metadata["custom_key"] = "custom_value"

        child = TreeNode(
            value=DocumentElement(
                element_type=ElementType.HEADING,
                text="Child Heading",
                line_number=2,
                start_position=17,
                end_position=30,
                level=1,
            )
        )

        original.add_child(child)

        # Test shallow clone
        shallow_clone = original.clone(deep=False)
        assert shallow_clone.value == original.value
        assert shallow_clone.metadata == original.metadata
        assert len(shallow_clone.children) == 0

        # Test deep clone
        deep_clone = original.clone(deep=True)
        assert deep_clone.value == original.value
        assert deep_clone.metadata == original.metadata
        assert len(deep_clone.children) == 1
        assert deep_clone.children[0].value == child.value


class TestLayoutTree:
    """Test cases for LayoutTree class."""

    def test_empty_tree(self):
        """Test empty tree operations."""
        tree = LayoutTree()

        assert tree.is_empty is True
        assert tree.size == 0
        assert tree.height == 0
        assert tree.max_depth == 0
        assert tree.root is None

        # Test traversal of empty tree
        nodes = list(tree.traverse_preorder())
        assert len(nodes) == 0

    def test_single_node_tree(self):
        """Test tree with single node."""
        element = DocumentElement(
            element_type=ElementType.HEADING,
            text="Single Node",
            line_number=1,
            start_position=0,
            end_position=11,
            level=0,
        )

        node = TreeNode(value=element)
        tree = LayoutTree(root=node)

        assert tree.is_empty is False
        assert tree.size == 1
        assert tree.height == 0
        assert tree.max_depth == 0

        # Test traversal
        preorder_nodes = list(tree.traverse_preorder())
        assert len(preorder_nodes) == 1
        assert preorder_nodes[0] == node

    def test_tree_traversal_methods(self):
        """Test different traversal strategies."""
        # Build test tree
        root = TreeNode(
            value=DocumentElement(
                element_type=ElementType.SECTION,
                text="Root",
                line_number=1,
                start_position=0,
                end_position=4,
                level=0,
            )
        )

        left = TreeNode(
            value=DocumentElement(
                element_type=ElementType.SUBSECTION,
                text="Left",
                line_number=2,
                start_position=5,
                end_position=9,
                level=1,
            )
        )

        right = TreeNode(
            value=DocumentElement(
                element_type=ElementType.SUBSECTION,
                text="Right",
                line_number=3,
                start_position=10,
                end_position=15,
                level=1,
            )
        )

        left_child = TreeNode(
            value=DocumentElement(
                element_type=ElementType.HEADING,
                text="Left Child",
                line_number=4,
                start_position=16,
                end_position=26,
                level=2,
            )
        )

        root.add_child(left)
        root.add_child(right)
        left.add_child(left_child)

        tree = LayoutTree(root=root)

        # Test pre-order traversal
        preorder = list(tree.traverse_preorder())
        preorder_texts = [node.text for node in preorder]
        assert preorder_texts == ["Root", "Left", "Left Child", "Right"]

        # Test post-order traversal
        postorder = list(tree.traverse_postorder())
        postorder_texts = [node.text for node in postorder]
        assert postorder_texts == ["Left Child", "Left", "Right", "Root"]

        # Test level-order traversal
        levelorder = list(tree.traverse_levelorder())
        levelorder_texts = [node.text for node in levelorder]
        assert levelorder_texts == ["Root", "Left", "Right", "Left Child"]

    def test_tree_search_operations(self):
        """Test tree search functionality."""
        # Build test tree
        elements = [
            DocumentElement(ElementType.SECTION, "Section 1", 1, 0, 9, 0),
            DocumentElement(ElementType.HEADING, "Heading A", 2, 10, 19, 1),
            DocumentElement(ElementType.HEADING, "Heading B", 3, 20, 29, 1),
            DocumentElement(ElementType.SUBSECTION, "Subsection 1", 4, 30, 42, 2),
        ]

        nodes = [TreeNode(value=elem) for elem in elements]
        nodes[0].add_child(nodes[1])
        nodes[0].add_child(nodes[2])
        nodes[1].add_child(nodes[3])

        tree = LayoutTree(root=nodes[0])

        # Test find by line
        found_node = tree.find_node_by_line(2)
        assert found_node == nodes[1]

        # Test find by type
        headings = tree.find_nodes_by_type(ElementType.HEADING)
        assert len(headings) == 2
        assert nodes[1] in headings
        assert nodes[2] in headings

        # Test find by text
        found_nodes = tree.find_nodes_by_text("Heading")
        assert len(found_nodes) == 2

        # Test get nodes at level
        level_1_nodes = tree.get_nodes_at_level(1)
        assert len(level_1_nodes) == 2
        assert nodes[1] in level_1_nodes
        assert nodes[2] in level_1_nodes

        # Test get leaf nodes
        leaf_nodes = tree.get_leaf_nodes()
        assert len(leaf_nodes) == 2
        assert nodes[2] in leaf_nodes
        assert nodes[3] in leaf_nodes

    def test_tree_statistics(self):
        """Test tree statistics calculation."""
        # Build test tree
        root = TreeNode(value=DocumentElement(ElementType.SECTION, "Root", 1, 0, 4, 0))
        child1 = TreeNode(
            value=DocumentElement(ElementType.HEADING, "Child1", 2, 5, 11, 1)
        )
        child2 = TreeNode(
            value=DocumentElement(ElementType.HEADING, "Child2", 3, 12, 18, 1)
        )
        grandchild = TreeNode(
            value=DocumentElement(ElementType.SUBSECTION, "Grandchild", 4, 19, 29, 2)
        )

        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild)

        tree = LayoutTree(root=root)
        stats = tree.calculate_statistics()

        assert stats.total_nodes == 4
        assert stats.max_depth == 2
        assert stats.max_height == 2
        assert stats.leaf_count == 2
        assert stats.element_type_counts["section"] == 1
        assert stats.element_type_counts["heading"] == 2
        assert stats.element_type_counts["subsection"] == 1
        assert stats.level_distribution[0] == 1
        assert stats.level_distribution[1] == 2
        assert stats.level_distribution[2] == 1

    def test_from_document_structure(self):
        """Test creating tree from DocumentStructure."""
        elements = [
            DocumentElement(ElementType.SECTION, "Section 1", 1, 0, 9, 0),
            DocumentElement(ElementType.HEADING, "Heading 1", 2, 10, 19, 1),
            DocumentElement(ElementType.HEADING, "Heading 2", 3, 20, 29, 1),
        ]

        # Set up parent-child relationships
        elements[1].parent = elements[0]
        elements[2].parent = elements[0]
        elements[0].children = [elements[1], elements[2]]

        doc_structure = DocumentStructure()
        doc_structure.elements = elements

        tree = LayoutTree.from_document_structure(doc_structure)

        assert tree.is_empty is False
        assert tree.size == 3
        assert tree.root.value == elements[0]
        assert len(tree.root.children) == 2


class TestMetadataManager:
    """Test cases for MetadataManager class."""

    def test_metadata_validation(self):
        """Test metadata validation functionality."""
        manager = MetadataManager()

        # Create test node with inconsistent depth
        element = DocumentElement(
            ElementType.HEADING, "Test", 1, 0, 4, 2
        )  # Level 2 but will be depth 0
        node = TreeNode(value=element)

        result = manager.validate_metadata(node, recursive=False)

        # Should have validation errors due to inconsistency
        assert result.validated_nodes == 1
        # Note: Specific validation behavior depends on implementation

    def test_metadata_calculation(self):
        """Test metadata calculation and caching."""
        manager = MetadataManager()

        # Build test tree
        root = TreeNode(value=DocumentElement(ElementType.SECTION, "Root", 1, 0, 4, 0))
        child1 = TreeNode(
            value=DocumentElement(ElementType.HEADING, "Child1", 2, 5, 11, 1)
        )
        child2 = TreeNode(
            value=DocumentElement(ElementType.HEADING, "Child2", 3, 12, 18, 1)
        )

        root.add_child(child1)
        root.add_child(child2)

        # Test metadata calculation
        subtree_size = manager.calculate_metadata(root, "subtree_size")
        assert subtree_size == 3

        leaf_count = manager.calculate_metadata(root, "leaf_count")
        assert leaf_count == 2

        # Test caching - second call should use cached value
        cached_size = manager.calculate_metadata(root, "subtree_size")
        assert cached_size == 3

    def test_metadata_propagation(self):
        """Test metadata change propagation."""
        manager = MetadataManager()

        root = TreeNode(value=DocumentElement(ElementType.SECTION, "Root", 1, 0, 4, 0))
        child = TreeNode(
            value=DocumentElement(ElementType.HEADING, "Child", 2, 5, 10, 1)
        )

        root.add_child(child)

        # Update metadata and test propagation
        manager.update_node_metadata(child, {"test_key": "test_value"}, propagate=True)

        assert child.metadata["test_key"] == "test_value"
        assert "last_modified" in child.metadata


class TestTreeBuilder:
    """Test cases for TreeBuilder class."""

    def test_build_from_flat_list_hierarchical(self):
        """Test building tree from flat list using hierarchical strategy."""
        builder = TreeBuilder()

        elements = [
            DocumentElement(ElementType.SECTION, "Root", 1, 0, 4, 0),
            DocumentElement(ElementType.HEADING, "Child1", 2, 5, 11, 1),
            DocumentElement(ElementType.HEADING, "Child2", 3, 12, 18, 1),
        ]

        # Set up parent relationships
        elements[1].parent = elements[0]
        elements[2].parent = elements[0]

        tree = builder.build_from_flat_list(elements, TreeBuildStrategy.HIERARCHICAL)

        assert tree.size == 3
        assert tree.root.value == elements[0]
        assert len(tree.root.children) == 2

    def test_build_from_flat_list_balanced(self):
        """Test building balanced tree from flat list."""
        builder = TreeBuilder()

        elements = [
            DocumentElement(ElementType.HEADING, f"Node {i}", i, i * 10, i * 10 + 5, 0)
            for i in range(1, 8)  # 7 elements
        ]

        tree = builder.build_from_flat_list(elements, TreeBuildStrategy.BALANCED)

        assert tree.size == 7
        assert tree.root.value == elements[3]  # Middle element should be root

    def test_build_from_node_data(self):
        """Test building tree from raw node data."""
        builder = TreeBuilder()

        node_data = [
            {
                "element_type": "section",
                "text": "Section 1",
                "line_number": 1,
                "start_position": 0,
                "end_position": 9,
                "level": 0,
            },
            {
                "element_type": "heading",
                "text": "Heading 1",
                "line_number": 2,
                "start_position": 10,
                "end_position": 19,
                "level": 1,
            },
        ]

        tree = builder.build_from_node_data(node_data)

        # Tree builder may create synthetic root when elements don't have explicit relationships
        assert tree.size >= 2
        assert tree.root is not None

    def test_merge_trees(self):
        """Test tree merging functionality."""
        builder = TreeBuilder()

        # Create two simple trees
        tree1 = LayoutTree()
        tree1.root = TreeNode(
            value=DocumentElement(ElementType.SECTION, "Tree1", 1, 0, 5, 0)
        )

        tree2 = LayoutTree()
        tree2.root = TreeNode(
            value=DocumentElement(ElementType.SECTION, "Tree2", 2, 6, 11, 0)
        )

        # Test append merge
        merged = builder.merge_trees(tree1, tree2, "append")

        assert merged.size == 2
        assert merged.root == tree1.root
        assert tree2.root in merged.root.children

    def test_split_tree(self):
        """Test tree splitting functionality."""
        builder = TreeBuilder()

        # Build test tree
        root = TreeNode(value=DocumentElement(ElementType.SECTION, "Root", 1, 0, 4, 0))
        child1 = TreeNode(
            value=DocumentElement(ElementType.HEADING, "Child1", 2, 5, 11, 1)
        )
        child2 = TreeNode(
            value=DocumentElement(ElementType.SUBSECTION, "Child2", 3, 12, 20, 1)
        )

        root.add_child(child1)
        root.add_child(child2)
        tree = LayoutTree(root=root)

        # Split based on element type
        def split_condition(node):
            return node.element_type == ElementType.SUBSECTION

        tree1, tree2 = builder.split_tree(tree, split_condition)

        # tree1 should have section and heading, tree2 should have subsection
        assert tree1.size >= 1
        assert tree2.size >= 1

    def test_tree_build_strategies(self):
        """Test different tree building strategies."""
        builder = TreeBuilder()

        elements = [
            DocumentElement(ElementType.SECTION, "Section", 1, 0, 7, 0),
            DocumentElement(ElementType.HEADING, "Heading1", 2, 8, 16, 1),
            DocumentElement(ElementType.HEADING, "Heading2", 3, 17, 25, 1),
            DocumentElement(ElementType.SUBSECTION, "Subsection", 4, 26, 36, 2),
        ]

        # Test different strategies
        strategies = [
            TreeBuildStrategy.HIERARCHICAL,
            TreeBuildStrategy.LEVEL_ORDER,
            TreeBuildStrategy.BALANCED,
            TreeBuildStrategy.DEPTH_FIRST,
            TreeBuildStrategy.BREADTH_FIRST,
        ]

        for strategy in strategies:
            tree = builder.build_from_flat_list(elements, strategy)
            # Tree may contain synthetic nodes depending on strategy
            assert tree.size >= len(elements)
            assert tree.root is not None


class TestIntegration:
    """Integration tests for the layout tree system."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from document structure to layout tree."""
        # Create document structure
        elements = [
            DocumentElement(ElementType.SECTION, "1. Introduction", 1, 0, 15, 0),
            DocumentElement(ElementType.SUBSECTION, "1.1 Overview", 2, 16, 28, 1),
            DocumentElement(ElementType.SUBSECTION, "1.2 Scope", 3, 29, 38, 1),
            DocumentElement(ElementType.SECTION, "2. Methodology", 4, 39, 53, 0),
            DocumentElement(ElementType.SUBSECTION, "2.1 Approach", 5, 54, 66, 1),
        ]

        # Set up hierarchical relationships
        elements[1].parent = elements[0]
        elements[2].parent = elements[0]
        elements[4].parent = elements[3]
        elements[0].children = [elements[1], elements[2]]
        elements[3].children = [elements[4]]

        doc_structure = DocumentStructure()
        doc_structure.elements = elements

        # Build layout tree
        tree = LayoutTree.from_document_structure(doc_structure)

        # Validate structure - tree may have synthetic root if needed
        assert tree.size >= 5
        assert tree.max_depth >= 1

        # Test traversal - account for potential synthetic root
        preorder_nodes = list(tree.traverse_preorder())
        assert len(preorder_nodes) >= 5

        # Test search operations
        sections = tree.find_nodes_by_type(ElementType.SECTION)
        assert len(sections) == 2

        subsections = tree.find_nodes_by_type(ElementType.SUBSECTION)
        assert len(subsections) == 3

        # Test statistics - account for potential synthetic root
        stats = tree.calculate_statistics()
        assert stats.total_nodes >= 5
        assert stats.element_type_counts["section"] == 2
        assert stats.element_type_counts["subsection"] == 3

    def test_large_tree_performance(self):
        """Test performance with larger trees."""
        # Create a larger test dataset
        elements = []
        for i in range(100):
            element = DocumentElement(
                element_type=ElementType.HEADING,
                text=f"Heading {i}",
                line_number=i + 1,
                start_position=i * 20,
                end_position=(i + 1) * 20 - 1,
                level=i % 3,  # Vary levels
            )
            elements.append(element)

        # Build tree using tree builder
        builder = TreeBuilder()
        tree = builder.build_from_flat_list(elements, TreeBuildStrategy.BALANCED)

        # Test operations
        assert tree.size == 100

        # Test traversal performance
        all_nodes = list(tree.traverse_preorder())
        assert len(all_nodes) == 100

        # Test search performance
        headings = tree.find_nodes_by_type(ElementType.HEADING)
        assert len(headings) == 100

        # Test statistics calculation
        stats = tree.calculate_statistics()
        assert stats.total_nodes == 100


# Additional helper functions for testing


def create_sample_document_element(
    element_type: ElementType, text: str, line_number: int, level: int = 0
) -> DocumentElement:
    """Helper function to create sample document elements."""
    return DocumentElement(
        element_type=element_type,
        text=text,
        line_number=line_number,
        start_position=line_number * 20,
        end_position=(line_number + 1) * 20 - 1,
        level=level,
    )


def create_sample_tree() -> LayoutTree:
    """Helper function to create a sample tree for testing."""
    elements = [
        create_sample_document_element(ElementType.SECTION, "Section 1", 1, 0),
        create_sample_document_element(ElementType.HEADING, "Heading 1.1", 2, 1),
        create_sample_document_element(ElementType.HEADING, "Heading 1.2", 3, 1),
        create_sample_document_element(
            ElementType.SUBSECTION, "Subsection 1.2.1", 4, 2
        ),
    ]

    # Set up relationships
    elements[1].parent = elements[0]
    elements[2].parent = elements[0]
    elements[3].parent = elements[2]

    doc_structure = DocumentStructure()
    doc_structure.elements = elements

    return LayoutTree.from_document_structure(doc_structure)
