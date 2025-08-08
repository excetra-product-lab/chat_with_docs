"""
Beautiful visualization utility for displaying document chunks and their hierarchical structure.

This module provides functions to visualize how documents are chunked, showing:
- Chunk boundaries and sizes
- Hierarchical structure (if available)
- Token counts and overlap
- Metadata and source information
"""

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from app.models.langchain_models import EnhancedDocument

    ENHANCED_DOCUMENT_AVAILABLE = True
except ImportError:
    ENHANCED_DOCUMENT_AVAILABLE = False

try:
    from langchain.schema import Document

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


@dataclass
class ChunkVisualization:
    """Configuration for chunk visualization display."""

    max_content_length: int = 300
    show_metadata: bool = True
    show_hierarchy: bool = True
    show_tokens: bool = True
    show_overlap: bool = True
    indent_size: int = 2
    border_char: str = "═"
    section_char: str = "─"


class ChunkVisualizer:
    """Beautiful visualization of document chunks and their hierarchical structure."""

    def __init__(self, config: ChunkVisualization | None = None):
        """Initialize the visualizer with configuration."""
        self.config = config or ChunkVisualization()

    def visualize_chunks(
        self,
        chunks: list[Any],
        title: str = "Document Chunks",
        show_before_openai: bool = False,
    ) -> str:
        """
        Create a beautiful visualization of document chunks.

        Args:
            chunks: List of Document objects (LangChain or EnhancedDocument)
            title: Title for the visualization
            show_before_openai: If True, adds a warning that these chunks will be sent to OpenAI

        Returns:
            Formatted string representation of the chunks
        """
        if not chunks:
            return "📄 No chunks to display"

        output = []

        # Header
        header = f"🌳 {title}"
        if show_before_openai:
            header += " (→ OpenAI)"

        output.append(self._create_header(header))

        if show_before_openai:
            output.append("⚠️  These chunks will be sent to OpenAI for processing\n")

        # Summary
        output.append(self._create_summary(chunks))
        output.append("")

        # Individual chunks
        for i, chunk in enumerate(chunks, 1):
            chunk_viz = self._visualize_single_chunk(chunk, i)
            output.append(chunk_viz)

            # Add separator between chunks (except for last)
            if i < len(chunks):
                output.append(self._create_separator())

        # Footer
        output.append(self._create_footer(len(chunks)))

        return "\n".join(output)

    def print_chunks(
        self,
        chunks: list[Any],
        title: str = "Document Chunks",
        show_before_openai: bool = False,
    ) -> None:
        """Print chunks with beautiful formatting."""
        print(self.visualize_chunks(chunks, title, show_before_openai))

    def _create_header(self, title: str) -> str:
        """Create a beautiful header."""
        border = self.config.border_char * 80
        centered_title = title.center(80)
        return f"{border}\n{centered_title}\n{border}"

    def _create_footer(self, chunk_count: int) -> str:
        """Create a footer with summary."""
        border = self.config.border_char * 80
        summary = f"Total chunks processed: {chunk_count}".center(80)
        return f"{border}\n{summary}\n{border}"

    def _create_separator(self) -> str:
        """Create a separator between chunks."""
        return self.config.section_char * 80

    def _create_summary(self, chunks: list[Any]) -> str:
        """Create a summary of the chunks."""
        total_chunks = len(chunks)

        # Calculate statistics
        content_lengths = []
        has_hierarchy = False
        has_tokens = False
        sources = set()

        for chunk in chunks:
            content = self._get_content(chunk)
            content_lengths.append(len(content))

            metadata = self._get_metadata(chunk)
            if metadata:
                # Check for hierarchy information
                if any(
                    key in metadata
                    for key in ["hierarchy_level", "section_path", "parent_section"]
                ):
                    has_hierarchy = True

                # Check for token information
                if any(key in metadata for key in ["token_count", "tokens"]):
                    has_tokens = True

                # Collect sources
                if "source" in metadata:
                    sources.add(Path(metadata["source"]).name)

        avg_length = (
            sum(content_lengths) / len(content_lengths) if content_lengths else 0
        )
        min_length = min(content_lengths) if content_lengths else 0
        max_length = max(content_lengths) if content_lengths else 0

        summary_lines = [
            f"📊 Summary: {total_chunks} chunks",
            f"📏 Content length: avg={avg_length:.0f}, min={min_length}, max={max_length} chars",
        ]

        if sources:
            source_list = ", ".join(list(sources)[:3])
            if len(sources) > 3:
                source_list += f" (+{len(sources) - 3} more)"
            summary_lines.append(f"📁 Sources: {source_list}")

        if has_hierarchy:
            summary_lines.append("🌳 Contains hierarchical structure")

        if has_tokens:
            summary_lines.append("🔢 Contains token counts")

        return "\n".join(summary_lines)

    def _visualize_single_chunk(self, chunk: Any, index: int) -> str:
        """Visualize a single chunk with all its details."""
        content = self._get_content(chunk)
        metadata = self._get_metadata(chunk)

        lines = []

        # Chunk header
        chunk_header = f"📄 Chunk {index}"

        # Add hierarchy info to header if available
        if metadata and self.config.show_hierarchy:
            hierarchy_info = self._extract_hierarchy_info(metadata)
            if hierarchy_info:
                chunk_header += f" {hierarchy_info}"

        lines.append(f"┌─ {chunk_header}")

        # Content preview
        content_preview = self._create_content_preview(content)
        for line in content_preview.split("\n"):
            lines.append(f"│ {line}")

        # Metadata
        if metadata and self.config.show_metadata:
            metadata_lines = self._format_metadata(metadata)
            if metadata_lines:
                lines.append("│")
                for line in metadata_lines:
                    lines.append(f"│ {line}")

        lines.append("└" + "─" * 78)

        return "\n".join(lines)

    def _get_content(self, chunk: Any) -> str:
        """Extract content from various chunk types."""
        if hasattr(chunk, "page_content"):
            return chunk.page_content
        elif hasattr(chunk, "content"):
            return chunk.content
        elif isinstance(chunk, str):
            return chunk
        elif isinstance(chunk, dict) and "content" in chunk:
            return chunk["content"]
        else:
            return str(chunk)

    def _get_metadata(self, chunk: Any) -> dict[str, Any] | None:
        """Extract metadata from various chunk types."""
        if hasattr(chunk, "metadata"):
            return chunk.metadata
        elif isinstance(chunk, dict) and "metadata" in chunk:
            return chunk["metadata"]
        return None

    def _extract_hierarchy_info(self, metadata: dict[str, Any]) -> str:
        """Extract and format hierarchy information from metadata."""
        hierarchy_parts = []

        # Try different hierarchy field names
        if "section_path" in metadata:
            hierarchy_parts.append(f"📍 {metadata['section_path']}")
        elif "hierarchy_level" in metadata:
            level = metadata["hierarchy_level"]
            hierarchy_parts.append(f"📊 Level {level}")

        if "parent_section" in metadata:
            hierarchy_parts.append(f"⬆️ {metadata['parent_section']}")

        if "section_title" in metadata and metadata["section_title"]:
            title = str(metadata["section_title"])
            if len(title) > 40:
                title = title[:37] + "..."
            hierarchy_parts.append(f"📑 {title}")

        return " ".join(hierarchy_parts) if hierarchy_parts else ""

    def _create_content_preview(self, content: str) -> str:
        """Create a formatted preview of the content."""
        if len(content) <= self.config.max_content_length:
            preview = content
        else:
            preview = content[: self.config.max_content_length] + "..."

        # Clean up the preview
        preview = preview.replace("\n", " ").replace("\r", " ")
        preview = " ".join(preview.split())  # Normalize whitespace

        # Wrap the text nicely
        wrapped = textwrap.fill(preview, width=76, subsequent_indent="  ")

        return wrapped

    def _format_metadata(self, metadata: dict[str, Any]) -> list[str]:
        """Format metadata for display."""
        lines = []

        # Key metadata fields to highlight
        important_fields = {
            "source": "📁 Source",
            "page": "📄 Page",
            "chunk_index": "🔢 Index",
            "token_count": "🎯 Tokens",
            "tokens": "🎯 Tokens",
            "chunk_size": "📏 Size",
            "overlap_with_previous": "🔗 Overlap",
            "hierarchy_level": "📊 Level",
            "section_number": "🔢 Section",
            "section_title": "📑 Title",
        }

        # Display important fields first
        for key, label in important_fields.items():
            if key in metadata:
                value = metadata[key]
                if key == "source":
                    value = Path(str(value)).name  # Show only filename
                elif key in ["section_title"] and len(str(value)) > 50:
                    value = str(value)[:47] + "..."

                lines.append(f"{label}: {value}")

        # Add any other interesting metadata
        other_fields = set(metadata.keys()) - set(important_fields.keys())
        interesting_others = [
            key
            for key in other_fields
            if key.lower() not in ["content", "text"] and not key.startswith("_")
        ]

        if interesting_others and len(lines) < 5:  # Don't overcrowd
            for key in sorted(interesting_others)[:3]:  # Limit to 3 extra fields
                value = metadata[key]
                if isinstance(value, (str, int, float, bool)):
                    if len(str(value)) > 50:
                        value = str(value)[:47] + "..."
                    lines.append(f"📋 {key}: {value}")

        return lines


# Convenience functions for easy import and use


def print_chunks_before_openai(
    chunks: list[Any], title: str = "Chunks → OpenAI"
) -> None:
    """Convenience function to print chunks with OpenAI warning."""
    visualizer = ChunkVisualizer()
    visualizer.print_chunks(chunks, title, show_before_openai=True)


def print_chunks(chunks: list[Any], title: str = "Document Chunks") -> None:
    """Convenience function to print chunks with default formatting."""
    visualizer = ChunkVisualizer()
    visualizer.print_chunks(chunks, title)


def visualize_chunks(chunks: list[Any], title: str = "Document Chunks") -> str:
    """Convenience function to get formatted chunk visualization."""
    visualizer = ChunkVisualizer()
    return visualizer.visualize_chunks(chunks, title)


# Example usage
if __name__ == "__main__":
    # Demo with mock chunks
    mock_chunks = [
        type(
            "MockChunk",
            (),
            {
                "page_content": "This is the first chunk containing important information about legal procedures and contracts that must be followed according to regulation XYZ.",
                "metadata": {
                    "source": "/path/to/document.pdf",
                    "page": 1,
                    "chunk_index": 0,
                    "token_count": 25,
                    "hierarchy_level": 1,
                    "section_title": "Introduction to Legal Framework",
                    "section_path": "1. Introduction > 1.1 Legal Framework",
                },
            },
        )(),
        type(
            "MockChunk",
            (),
            {
                "page_content": "The second chunk discusses implementation details and specific requirements for compliance.",
                "metadata": {
                    "source": "/path/to/document.pdf",
                    "page": 1,
                    "chunk_index": 1,
                    "token_count": 18,
                    "hierarchy_level": 2,
                    "section_title": "Implementation Requirements",
                    "section_path": "1. Introduction > 1.2 Implementation",
                },
            },
        )(),
    ]

    print_chunks_before_openai(mock_chunks, "Demo Legal Document Chunks")
