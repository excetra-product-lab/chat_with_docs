"""Hierarchical text chunking service that preserves document structure boundaries.

This module provides the HierarchicalChunker class that extends Langchain's
RecursiveCharacterTextSplitter to support hierarchy-aware chunking for legal documents.
It integrates with the document structure detection system and token counter for
optimal chunking that respects document boundaries and maintains semantic coherence.
"""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.document_structure_detector import StructureDetector
from app.services.document_structure_detector.data_models import (
    DocumentElement,
    DocumentStructure,
    ElementType,
    NumberingType,
)
from app.utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class HierarchicalChunk:
    """Container for a hierarchical chunk with structure metadata."""

    def __init__(
        self,
        text: str,
        chunk_index: int,
        token_count: int,
        start_position: int = 0,
        end_position: int = 0,
        hierarchy_level: int = 0,
        element_type: ElementType | None = None,
        section_title: str | None = None,
        numbering: str | None = None,
        parent_elements: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        page_number: int | None = None,
    ):
        """Initialize hierarchical chunk with structure information.

        Args:
            text: The chunk text content
            chunk_index: Index of chunk in the document
            token_count: Number of tokens in the chunk
            start_position: Start character position in original document
            end_position: End character position in original document
            hierarchy_level: Hierarchical level (0=root, 1=section, 2=subsection, etc.)
            element_type: Type of document element this chunk represents
            section_title: Title of the section this chunk belongs to
            numbering: Numbering identifier (e.g., "1.2.3", "§5", "Article II")
            parent_elements: List of parent element identifiers
            metadata: Additional metadata dictionary
            page_number: Page number for PDF documents (None for text-based documents)
        """
        self.text = text
        self.chunk_index = chunk_index
        self.token_count = token_count
        self.start_position = start_position
        self.end_position = end_position
        self.hierarchy_level = hierarchy_level
        self.element_type = element_type
        self.section_title = section_title
        self.numbering = numbering
        self.parent_elements = parent_elements or []
        self.metadata = metadata or {}
        self.page_number = page_number
        self.char_count = len(text)


class HierarchicalChunker(RecursiveCharacterTextSplitter):
    """Hierarchy-aware text splitter for legal documents.

    This class extends Langchain's RecursiveCharacterTextSplitter to provide
    intelligent chunking that respects document structure boundaries while
    maintaining optimal chunk sizes for downstream processing.

    Key features:
    - Hierarchy-aware separators based on document structure
    - Token-based size management with boundary preservation
    - Integration with legal document structure detection
    - Metadata enrichment with hierarchy information
    """

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        structure_detector: StructureDetector | None = None,
        chunk_size: int = 600,  # Optimized for MVP: 400-800 token range
        chunk_overlap: int = 100,  # Reduced for better efficiency
        model_name: str = "gpt-4o",
        legal_specific: bool = True,
        separators: list[str] | None = None,
        min_chunk_size: int = 100,  # Minimum viable chunk size
        max_chunk_size: int = 1024,  # Maximum chunk size before forced split
        target_range: tuple[int, int] = (400, 800),  # Optimal token range
        **kwargs,
    ):
        """Initialize the hierarchical chunker with optimized chunk size logic.

        Args:
            token_counter: TokenCounter instance for accurate token counting
            structure_detector: StructureDetector for hierarchy analysis
            chunk_size: Target chunk size in tokens (default: 600, optimized for MVP)
            chunk_overlap: Number of overlapping tokens between chunks (default: 100)
            model_name: Model name for token counting (default: "gpt-4")
            legal_specific: Enable legal-specific tokenization patterns
            separators: Custom hierarchy-aware separators (auto-detected if None)
            min_chunk_size: Minimum acceptable chunk size in tokens (default: 100)
            max_chunk_size: Maximum chunk size before forced splitting (default: 1024)
            target_range: Optimal token range tuple (min, max) (default: 400-800)
            **kwargs: Additional arguments passed to RecursiveCharacterTextSplitter
        """
        # Initialize dependencies
        self.token_counter = token_counter or TokenCounter.create_for_model(
            model_name=model_name, legal_specific=legal_specific
        )
        self.structure_detector = structure_detector or StructureDetector()
        self.model_name = model_name
        self.legal_specific = legal_specific

        # Token usage analytics
        self.token_usage_stats = {
            "total_tokens_processed": 0,
            "total_chunks_created": 0,
            "token_counting_calls": 0,
            "large_chunks_split": 0,
            "small_chunks_merged": 0,
            "boundary_violations_fixed": 0,
        }

        # Chunk size configuration
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.target_range = target_range
        self.target_min, self.target_max = target_range

        # Initialize logger first
        self.logger = logging.getLogger(__name__)

        # Validate and adjust chunk size parameters
        validated_chunk_size, validated_overlap = self._validate_chunk_parameters(
            chunk_size, chunk_overlap, min_chunk_size, max_chunk_size, target_range
        )

        # Use legal-specific separators by default
        if separators is None:
            separators = self._get_default_legal_separators()

        # Create custom length function for token-based splitting
        length_function = self._create_token_length_function()

        # Initialize parent RecursiveCharacterTextSplitter
        super().__init__(
            chunk_size=validated_chunk_size,
            chunk_overlap=validated_overlap,
            length_function=length_function,
            separators=separators,
            **kwargs,
        )

        self.logger.info(
            f"HierarchicalChunker initialized: chunk_size={validated_chunk_size}, "
            f"overlap={validated_overlap}, target_range={target_range}, "
            f"model={model_name}, legal_specific={legal_specific}"
        )

    def _get_default_legal_separators(self) -> list[str]:
        """Get default hierarchy-aware separators for legal documents.

        Returns:
            List of separators ordered by hierarchy level (highest to lowest)
        """
        return [
            # Chapter/Title level separators
            "\n\nCHAPTER ",
            "\n\nTITLE ",
            "\n\nPART ",
            # Article/Section level separators
            "\n\nARTICLE ",
            "\n\nSECTION ",
            "\n\nSec. ",
            "\n\n§ ",
            # Subsection level separators
            "\n\n(",  # (a), (1), etc.
            "\n\n(i)",  # Roman numerals
            "\n\n(A)",  # Letters
            # Paragraph separators
            "\n\n",  # Double newline (standard paragraph)
            "\n",  # Single newline
            # Clause/sentence separators
            ". ",  # Sentence boundary
            "; ",  # Clause boundary
            ", ",  # Sub-clause boundary
            " ",  # Word boundary
            "",  # Character boundary (last resort)
        ]

    def _create_token_length_function(self):
        """Create a token-based length function for accurate chunking.

        Returns:
            Callable that counts tokens instead of characters
        """

        def token_length_function(text: str) -> int:
            """Count tokens in text using the configured TokenCounter with analytics."""
            return self._count_tokens_with_tracking(text)

        return token_length_function

    def _count_tokens_with_tracking(self, text: str) -> int:
        """Count tokens with error handling and usage tracking.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens in the text
        """
        try:
            self.token_usage_stats["token_counting_calls"] += 1
            token_count = self.token_counter.count_tokens_for_model(
                text, self.model_name
            )
            self.token_usage_stats["total_tokens_processed"] += token_count
            return token_count
        except Exception as e:
            self.logger.warning(
                f"Token counting failed for text of length {len(text)}: {e}"
            )
            # Fallback to character-based estimation (rough approximation: ~4 chars per token)
            fallback_count = max(1, len(text) // 4)
            self.logger.info(f"Using fallback token count: {fallback_count}")
            return fallback_count

    async def _count_tokens_async_with_tracking(self, text: str) -> int:
        """Async token counting with error handling and usage tracking.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens in the text
        """
        try:
            self.token_usage_stats["token_counting_calls"] += 1
            token_count = await self.token_counter.count_tokens_for_model_async(
                text, self.model_name
            )
            self.token_usage_stats["total_tokens_processed"] += token_count
            return token_count
        except Exception as e:
            self.logger.warning(
                f"Async token counting failed for text of length {len(text)}: {e}"
            )
            # Fallback to sync counting
            return self._count_tokens_with_tracking(text)

    def _batch_count_tokens(self, texts: list[str]) -> list[int]:
        """Count tokens for multiple texts efficiently.

        Args:
            texts: List of texts to count tokens for

        Returns:
            List of token counts corresponding to input texts
        """
        token_counts = []

        for text in texts:
            count = self._count_tokens_with_tracking(text)
            token_counts.append(count)

        self.logger.debug(f"Batch counted tokens for {len(texts)} texts")
        return token_counts

    def _calculate_optimal_overlap(
        self, chunk_size: int, model_name: str | None = None
    ) -> int:
        """Calculate optimal overlap based on TokenCounter insights.

        Args:
            chunk_size: Current chunk size in tokens
            model_name: Model name for token-specific optimization

        Returns:
            Optimal overlap size in tokens
        """
        effective_model = model_name or self.model_name

        # Model-specific overlap optimization
        if any(
            gpt4_variant in effective_model.lower()
            for gpt4_variant in ["gpt-4", "gpt-4o", "gpt-4.1"]
        ):
            # GPT-4 family benefits from slightly higher overlap for context preservation
            base_percentage = 0.15  # 15%
        else:
            # Default for other models (fallback only)
            base_percentage = 0.125  # 12.5%

        optimal_overlap = int(chunk_size * base_percentage)

        # Apply bounds: minimum 50 tokens, maximum 200 tokens
        optimal_overlap = max(50, min(200, optimal_overlap))

        self.logger.debug(
            f"Calculated optimal overlap for {effective_model}: {optimal_overlap} tokens "
            f"({base_percentage * 100:.1f}% of {chunk_size})"
        )

        return optimal_overlap

    def _validate_chunk_parameters(
        self,
        chunk_size: int,
        chunk_overlap: int,
        min_chunk_size: int,
        max_chunk_size: int,
        target_range: tuple[int, int],
    ) -> tuple[int, int]:
        """Validate and adjust chunk size parameters for optimal performance.

        Args:
            chunk_size: Requested chunk size
            chunk_overlap: Requested chunk overlap
            min_chunk_size: Minimum acceptable chunk size
            max_chunk_size: Maximum acceptable chunk size
            target_range: Optimal token range (min, max)

        Returns:
            Tuple of (validated_chunk_size, validated_overlap)
        """
        target_min, target_max = target_range

        # Validate chunk_size is within acceptable bounds
        if chunk_size < min_chunk_size:
            self.logger.warning(
                f"Chunk size {chunk_size} below minimum {min_chunk_size}, adjusting to {min_chunk_size}"
            )
            chunk_size = min_chunk_size
        elif chunk_size > max_chunk_size:
            self.logger.warning(
                f"Chunk size {chunk_size} above maximum {max_chunk_size}, adjusting to {max_chunk_size}"
            )
            chunk_size = max_chunk_size

        # Recommend target range if outside optimal bounds
        if chunk_size < target_min:
            recommended = target_min
            self.logger.info(
                f"Chunk size {chunk_size} below optimal range {target_range}. "
                f"Consider using {recommended} for better performance."
            )
        elif chunk_size > target_max:
            recommended = target_max
            self.logger.info(
                f"Chunk size {chunk_size} above optimal range {target_range}. "
                f"Consider using {recommended} for better performance."
            )

        # Validate chunk_overlap
        if chunk_overlap >= chunk_size:
            # Overlap should be at most 25% of chunk size for efficiency
            max_overlap = max(1, chunk_size // 4)
            self.logger.warning(
                f"Chunk overlap {chunk_overlap} >= chunk size {chunk_size}, "
                f"adjusting to {max_overlap}"
            )
            chunk_overlap = max_overlap
        elif chunk_overlap < 0:
            self.logger.warning(
                f"Negative chunk overlap {chunk_overlap}, adjusting to 0"
            )
            chunk_overlap = 0

        # Recommend optimal overlap (10-15% of chunk size)
        optimal_overlap = max(50, min(150, chunk_size // 8))  # 12.5% with bounds
        if abs(chunk_overlap - optimal_overlap) > optimal_overlap // 2:
            self.logger.info(
                f"Current overlap {chunk_overlap} may not be optimal. "
                f"Consider using {optimal_overlap} (≈12.5% of chunk size)"
            )

        return chunk_size, chunk_overlap

    def _optimize_chunk_sizes(self, chunks: list[str]) -> list[str]:
        """Optimize chunk sizes to fit within the target range.

        This method handles chunks that are too large (splits them) or too small
        (merges them) to ensure optimal token distribution.

        Args:
            chunks: List of text chunks to optimize

        Returns:
            List of optimized chunks
        """
        if not chunks:
            return chunks

        optimized_chunks = []
        small_chunks_buffer: list[str] = []

        for chunk in chunks:
            token_count = self._count_tokens_with_tracking(chunk)

            if token_count > self.max_chunk_size:
                # Handle oversized chunks
                self.token_usage_stats["large_chunks_split"] += 1
                # First, flush any small chunks in buffer
                if small_chunks_buffer:
                    merged_chunk = self._merge_small_chunks(small_chunks_buffer)
                    if merged_chunk:
                        optimized_chunks.append(merged_chunk)
                    small_chunks_buffer = []

                # Split the large chunk
                split_chunks = self._split_large_chunk(chunk, token_count)
                optimized_chunks.extend(split_chunks)

            elif token_count < self.min_chunk_size:
                # Buffer small chunks for potential merging
                small_chunks_buffer.append(chunk)

                # Check if buffer is getting too large
                buffer_tokens = sum(
                    self._count_tokens_with_tracking(c) for c in small_chunks_buffer
                )

                if buffer_tokens >= self.target_min:
                    # Merge and flush buffer
                    self.token_usage_stats["small_chunks_merged"] += len(
                        small_chunks_buffer
                    )
                    merged_chunk = self._merge_small_chunks(small_chunks_buffer)
                    if merged_chunk:
                        optimized_chunks.append(merged_chunk)
                    small_chunks_buffer = []

            else:
                # Chunk is within acceptable size range
                # First, flush any small chunks in buffer
                if small_chunks_buffer:
                    merged_chunk = self._merge_small_chunks(small_chunks_buffer)
                    if merged_chunk:
                        optimized_chunks.append(merged_chunk)
                    small_chunks_buffer = []

                optimized_chunks.append(chunk)

        # Handle any remaining small chunks in buffer
        if small_chunks_buffer:
            merged_chunk = self._merge_small_chunks(small_chunks_buffer)
            if merged_chunk:
                optimized_chunks.append(merged_chunk)

        self.logger.info(
            f"Chunk optimization: {len(chunks)} → {len(optimized_chunks)} chunks"
        )
        return optimized_chunks

    def _merge_small_chunks(self, small_chunks: list[str]) -> str | None:
        """Merge small chunks together to create a chunk within the target range.

        Args:
            small_chunks: List of small chunks to merge

        Returns:
            Merged chunk text or None if empty
        """
        if not small_chunks:
            return None

        # Join chunks with appropriate separator
        merged_text = "\n\n".join(
            chunk.strip() for chunk in small_chunks if chunk.strip()
        )

        if not merged_text:
            return None

        # Verify merged chunk is within acceptable size
        merged_tokens = self._count_tokens_with_tracking(merged_text)

        if merged_tokens > self.max_chunk_size:
            # If merged chunk is too large, return the largest small chunk
            self.logger.warning(
                f"Merged chunk ({merged_tokens} tokens) exceeds maximum size. "
                f"Using largest individual chunk instead."
            )
            return max(small_chunks, key=lambda x: len(x))

        self.logger.debug(
            f"Merged {len(small_chunks)} small chunks into {merged_tokens} tokens"
        )
        return merged_text

    def _split_large_chunk(self, chunk: str, token_count: int) -> list[str]:
        """Split a large chunk that exceeds the maximum size limit.

        Args:
            chunk: The chunk text to split
            token_count: Current token count of the chunk

        Returns:
            List of smaller chunks
        """
        # Calculate how many pieces we need
        target_pieces = max(2, (token_count // self.target_max) + 1)
        target_size = token_count // target_pieces

        self.logger.info(
            f"Splitting large chunk ({token_count} tokens) into ~{target_pieces} pieces "
            f"of ~{target_size} tokens each"
        )

        # Create a temporary splitter with smaller chunk size
        temp_splitter = RecursiveCharacterTextSplitter(
            chunk_size=target_size,
            chunk_overlap=self._chunk_overlap,
            length_function=self._create_token_length_function(),
            separators=self._separators,
        )

        split_chunks = temp_splitter.split_text(chunk)

        # Verify split chunks are reasonable
        for i, split_chunk in enumerate(split_chunks):
            split_tokens = self._count_tokens_with_tracking(split_chunk)
            if split_tokens > self.max_chunk_size:
                self.logger.warning(
                    f"Split chunk {i} still too large ({split_tokens} tokens). "
                    f"May need more aggressive splitting."
                )

        return split_chunks

    def chunk_text_with_hierarchy(self, text: str) -> list[HierarchicalChunk]:
        """Split text into hierarchical chunks with structure metadata.

        This method implements the core hierarchy-aware chunking algorithm:
        1. Detect document structure and hierarchy
        2. Generate dynamic separators based on structure
        3. Apply hierarchy-aware boundary enforcement
        4. Split text with respect to hierarchical boundaries
        5. Enrich chunks with structure metadata

        Args:
            text: The input text to chunk

        Returns:
            List of HierarchicalChunk objects with structure information
        """
        self.logger.info(
            f"Starting hierarchical chunking for text of length {len(text)}"
        )

        # Step 1: Detect document structure
        document_structure = self.structure_detector.detect_structure(text)
        self.logger.info(
            f"Detected {len(document_structure.elements)} structural elements"
        )

        # Step 2: Update separators based on detected structure
        dynamic_separators = self._get_dynamic_separators(document_structure)
        if dynamic_separators:
            self._separators = dynamic_separators
            self.logger.info("Updated separators based on document structure")

        # Step 3: Apply hierarchy-aware boundary enforcement
        boundary_markers = self._extract_hierarchy_boundaries(text, document_structure)
        self.logger.info(f"Identified {len(boundary_markers)} hierarchy boundaries")

        # Step 4: Split text using parent RecursiveCharacterTextSplitter with boundary awareness
        text_chunks = self._split_with_boundary_preservation(text, boundary_markers)
        self.logger.info(f"Created {len(text_chunks)} hierarchy-aware text chunks")

        # Step 4.5: Optimize chunk sizes to fit within target range
        optimized_chunks = self._optimize_chunk_sizes(text_chunks)
        self.logger.info(
            f"Optimized chunks: {len(text_chunks)} → {len(optimized_chunks)} chunks"
        )

        # Step 5: Enrich chunks with hierarchy information
        hierarchical_chunks = self._create_hierarchical_chunks(
            optimized_chunks, text, document_structure
        )

        self.logger.info("Enriched chunks with hierarchy metadata")
        return hierarchical_chunks

    def _extract_hierarchy_boundaries(
        self, text: str, structure: DocumentStructure
    ) -> list[dict[str, Any]]:
        """Extract hierarchical boundary markers from the document.

        Args:
            text: The full document text
            structure: Detected document structure

        Returns:
            List of boundary markers with position and hierarchy information
        """
        boundaries = []

        for element in structure.elements:
            # Create boundary marker for each structural element
            boundary = {
                "start_position": element.start_position,
                "end_position": element.end_position,
                "element_type": element.element_type,
                "hierarchy_level": element.level,
                "numbering": element.numbering.get_full_number()
                if element.numbering
                else None,
                "text_preview": element.text[:50] + "..."
                if len(element.text) > 50
                else element.text,
                "is_section_boundary": element.element_type
                in [ElementType.CHAPTER, ElementType.ARTICLE, ElementType.SECTION],
                "is_subsection_boundary": element.element_type
                in [ElementType.SUBSECTION, ElementType.CLAUSE, ElementType.PARAGRAPH],
            }
            boundaries.append(boundary)

        # Sort boundaries by position
        boundaries.sort(
            key=lambda x: int(x["start_position"])
            if isinstance(x["start_position"], int | str)
            else 0
        )

        return boundaries

    def _split_with_boundary_preservation(
        self, text: str, boundaries: list[dict[str, Any]]
    ) -> list[str]:
        """Split text while preserving hierarchical boundaries.

        This method ensures that chunks don't split across important hierarchical
        boundaries, maintaining semantic coherence within each chunk.

        Args:
            text: The text to split
            boundaries: List of hierarchy boundary markers

        Returns:
            List of text chunks that respect hierarchical boundaries
        """
        # First, do the normal splitting
        initial_chunks = self.split_text(text)

        if not boundaries:
            return initial_chunks

        # Check each chunk for boundary violations and adjust if necessary
        refined_chunks = []

        for chunk_text in initial_chunks:
            # Find the chunk's position in the original text
            chunk_start = text.find(chunk_text)
            if chunk_start == -1:
                # Fallback if exact match not found
                refined_chunks.append(chunk_text)
                continue

            chunk_end = chunk_start + len(chunk_text)

            # Check if this chunk violates any important boundaries
            violating_boundaries = self._find_boundary_violations(
                chunk_start, chunk_end, boundaries
            )

            if not violating_boundaries:
                # No violations, keep the chunk as is
                refined_chunks.append(chunk_text)
            else:
                # Split the chunk at the violating boundary
                sub_chunks = self._split_at_boundaries(
                    chunk_text, chunk_start, violating_boundaries, text
                )
                refined_chunks.extend(sub_chunks)

        return refined_chunks

    def _find_boundary_violations(
        self, chunk_start: int, chunk_end: int, boundaries: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Find boundaries that are violated by a chunk's span.

        A boundary is violated if it falls within a chunk but isn't at the chunk's start.

        Args:
            chunk_start: Start position of the chunk
            chunk_end: End position of the chunk
            boundaries: List of boundary markers

        Returns:
            List of boundary markers that are violated by this chunk
        """
        violations = []

        for boundary in boundaries:
            boundary_pos = boundary["start_position"]

            # Check if boundary falls within chunk (but not at the start)
            if chunk_start < boundary_pos < chunk_end:
                # Only consider violations for important boundaries
                if boundary["is_section_boundary"] or (
                    boundary["is_subsection_boundary"]
                    and boundary["hierarchy_level"] <= 2
                ):
                    violations.append(boundary)

        return violations

    def _split_at_boundaries(
        self,
        chunk_text: str,
        chunk_start: int,
        boundaries: list[dict[str, Any]],
        full_text: str,
    ) -> list[str]:
        """Split a chunk at specific boundary points.

        Args:
            chunk_text: The chunk text to split
            chunk_start: Start position of chunk in full text
            boundaries: Boundary markers to split at
            full_text: The full original text

        Returns:
            List of sub-chunks that respect boundaries
        """
        if not boundaries:
            return [chunk_text]

        # Sort boundaries by position
        boundaries.sort(key=lambda x: x["start_position"])

        sub_chunks = []
        current_pos = 0

        for boundary in boundaries:
            boundary_pos = boundary["start_position"] - chunk_start

            # Extract text up to the boundary
            if boundary_pos > current_pos:
                sub_chunk = chunk_text[current_pos:boundary_pos].strip()
                if sub_chunk:
                    sub_chunks.append(sub_chunk)

            current_pos = boundary_pos

        # Add remaining text after the last boundary
        if current_pos < len(chunk_text):
            remaining = chunk_text[current_pos:].strip()
            if remaining:
                sub_chunks.append(remaining)

        return sub_chunks

    def _get_dynamic_separators(self, structure: DocumentStructure) -> list[str] | None:
        """Generate dynamic separators based on detected document structure.

        Uses the PatternHandler regex patterns to create hierarchy-aware separators
        that match the actual document structure patterns found.

        Args:
            structure: The detected document structure

        Returns:
            List of dynamic separators ordered by hierarchy level, or None if using defaults
        """
        if not structure.elements:
            return None

        # Analyze detected patterns by hierarchy level
        hierarchy_patterns: dict[int, set[str]] = {}  # level -> set of patterns
        numbering_types = set()

        for element in structure.elements:
            level = element.level
            element_type = element.element_type

            if level not in hierarchy_patterns:
                hierarchy_patterns[level] = set()

            # Extract numbering system patterns
            if element.numbering:
                raw_text = element.numbering.raw_text.strip()
                numbering_type = element.numbering.numbering_type
                numbering_types.add(numbering_type)

                # Generate separator patterns based on element type and numbering
                patterns = self._generate_separator_patterns(
                    element_type, raw_text, numbering_type
                )
                hierarchy_patterns[level].update(patterns)

            # Add element type-based patterns
            type_patterns = self._get_element_type_patterns(element_type)
            hierarchy_patterns[level].update(type_patterns)

        # Build hierarchical separator list
        if hierarchy_patterns:
            separators = []

            # Sort levels (0 = highest hierarchy)
            for level in sorted(hierarchy_patterns.keys()):
                separators.extend(list(hierarchy_patterns[level]))

            # Add numbering-type specific separators
            separators.extend(self._get_numbering_type_separators(numbering_types))

            # Add default legal separators for fallback
            separators.extend(self._get_default_legal_separators())

            # Remove duplicates while preserving hierarchy order
            unique_separators = list(dict.fromkeys(separators))

            self.logger.info(
                f"Generated {len(unique_separators)} dynamic separators for {len(hierarchy_patterns)} hierarchy levels"
            )
            return unique_separators

        return None

    def _generate_separator_patterns(
        self, element_type: ElementType, raw_text: str, numbering_type: NumberingType
    ) -> list[str]:
        """Generate separator patterns for a specific element.

        Args:
            element_type: Type of document element
            raw_text: Raw numbering text
            numbering_type: Type of numbering system

        Returns:
            List of separator patterns for this element
        """
        patterns = []

        # Clean up raw text for pattern generation
        clean_text = raw_text.strip()

        if element_type == ElementType.CHAPTER:
            patterns.extend(
                [
                    f"\n\nCHAPTER {clean_text}",
                    f"\n\nChapter {clean_text}",
                    f"\n\n{clean_text}",  # Just the number/identifier
                ]
            )

        elif element_type == ElementType.ARTICLE:
            patterns.extend(
                [
                    f"\n\nARTICLE {clean_text}",
                    f"\n\nArticle {clean_text}",
                    f"\n\n{clean_text}",
                ]
            )

        elif element_type == ElementType.SECTION:
            patterns.extend(
                [
                    f"\n\nSECTION {clean_text}",
                    f"\n\nSection {clean_text}",
                    f"\n\nSec. {clean_text}",
                    f"\n\n{clean_text}",
                ]
            )

            # Add section symbol patterns if applicable
            if clean_text.startswith("§"):
                patterns.append(f"\n\n{clean_text}")
            elif numbering_type == NumberingType.SECTION_SYMBOL:
                patterns.append(f"\n\n§ {clean_text}")
                patterns.append(f"\n\n§{clean_text}")

        elif element_type == ElementType.SUBSECTION:
            patterns.extend(
                [
                    f"\n\nSubsection {clean_text}",
                    f"\n\n{clean_text}",
                ]
            )

            # Handle parenthetical subsections
            if clean_text.startswith("(") and clean_text.endswith(")"):
                patterns.append(f"\n\n{clean_text}")
            elif not clean_text.startswith("("):
                patterns.append(f"\n\n({clean_text})")

        elif element_type == ElementType.CLAUSE:
            patterns.extend(
                [
                    f"\n\nClause {clean_text}",
                    f"\n\n{clean_text}",
                ]
            )

        elif element_type == ElementType.PARAGRAPH:
            patterns.extend(
                [
                    f"\n\nParagraph {clean_text}",
                    f"\n\n{clean_text}",
                ]
            )

        return patterns

    def _get_element_type_patterns(self, element_type: ElementType) -> list[str]:
        """Get generic separator patterns for an element type.

        Args:
            element_type: Type of document element

        Returns:
            List of generic separator patterns
        """
        patterns = []

        if element_type == ElementType.CHAPTER:
            patterns.extend(["\n\nCHAPTER ", "\n\nChapter "])
        elif element_type == ElementType.ARTICLE:
            patterns.extend(["\n\nARTICLE ", "\n\nArticle "])
        elif element_type == ElementType.SECTION:
            patterns.extend(["\n\nSECTION ", "\n\nSection ", "\n\nSec. ", "\n\n§ "])
        elif element_type == ElementType.SUBSECTION:
            patterns.extend(["\n\nSubsection ", "\n\n("])
        elif element_type == ElementType.CLAUSE:
            patterns.extend(["\n\nClause "])
        elif element_type == ElementType.PARAGRAPH:
            patterns.extend(["\n\nParagraph "])
        elif element_type == ElementType.HEADING:
            patterns.extend(["\n\n"])  # Generic heading separator

        return patterns

    def _get_numbering_type_separators(self, numbering_types: set) -> list[str]:
        """Get separators based on detected numbering types.

        Args:
            numbering_types: Set of detected NumberingType values

        Returns:
            List of numbering-specific separators
        """
        separators = []

        for numbering_type in numbering_types:
            if numbering_type == NumberingType.DECIMAL:
                separators.extend(
                    [
                        "\n\n1.",
                        "\n\n2.",
                        "\n\n3.",  # Common decimal patterns
                        "\n\n1.1",
                        "\n\n1.2",
                        "\n\n2.1",  # Multi-level decimal
                    ]
                )
            elif numbering_type == NumberingType.ROMAN_UPPER:
                separators.extend(
                    [
                        "\n\nI.",
                        "\n\nII.",
                        "\n\nIII.",
                        "\n\nIV.",
                        "\n\nV.",
                        "\n\nI ",
                        "\n\nII ",
                        "\n\nIII ",  # Without periods
                    ]
                )
            elif numbering_type == NumberingType.ROMAN_LOWER:
                separators.extend(
                    [
                        "\n\ni.",
                        "\n\nii.",
                        "\n\niii.",
                        "\n\niv.",
                        "\n\nv.",
                        "\n\ni ",
                        "\n\nii ",
                        "\n\niii ",  # Without periods
                    ]
                )
            elif numbering_type == NumberingType.LETTER_UPPER:
                separators.extend(
                    [
                        "\n\nA.",
                        "\n\nB.",
                        "\n\nC.",
                        "\n\nD.",
                        "\n\nA ",
                        "\n\nB ",
                        "\n\nC ",  # Without periods
                        "\n\n(A)",
                        "\n\n(B)",
                        "\n\n(C)",  # Parenthetical
                    ]
                )
            elif numbering_type == NumberingType.LETTER_LOWER:
                separators.extend(
                    [
                        "\n\na.",
                        "\n\nb.",
                        "\n\nc.",
                        "\n\nd.",
                        "\n\na ",
                        "\n\nb ",
                        "\n\nc ",  # Without periods
                        "\n\n(a)",
                        "\n\n(b)",
                        "\n\n(c)",  # Parenthetical
                    ]
                )
            elif numbering_type == NumberingType.SECTION_SYMBOL:
                separators.extend(
                    [
                        "\n\n§ ",
                        "\n\n§",
                        "§ ",  # Various section symbol formats
                    ]
                )

        return separators

    def _create_hierarchical_chunks(
        self, text_chunks: list[str], original_text: str, structure: DocumentStructure
    ) -> list[HierarchicalChunk]:
        """Convert text chunks to HierarchicalChunk objects with metadata.

        Args:
            text_chunks: List of text chunks from splitting
            original_text: The original full text
            structure: The detected document structure

        Returns:
            List of HierarchicalChunk objects with hierarchy metadata
        """
        hierarchical_chunks = []
        current_position = 0

        for i, chunk_text in enumerate(text_chunks):
            # Find chunk position in original text
            start_pos = original_text.find(chunk_text, current_position)
            if start_pos == -1:
                # If exact match not found, use current position as fallback
                start_pos = current_position
            end_pos = start_pos + len(chunk_text)

            # Count tokens in chunk
            token_count = self._count_tokens_with_tracking(chunk_text)
            self.token_usage_stats["total_chunks_created"] += 1

            # Find relevant structural element for this chunk
            chunk_element = self._find_chunk_element(
                start_pos, end_pos, structure.elements
            )

            # Extract hierarchy information
            hierarchy_info = self._extract_hierarchy_info(
                chunk_element, structure.elements
            )

            # Determine page number for this chunk
            page_number = self._get_chunk_page_number(
                start_pos, end_pos, original_text, structure
            )

            # Create hierarchical chunk
            hierarchical_chunk = HierarchicalChunk(
                text=chunk_text,
                chunk_index=i,
                token_count=token_count,
                start_position=start_pos,
                end_position=end_pos,
                hierarchy_level=hierarchy_info["level"],
                element_type=hierarchy_info["element_type"],
                section_title=hierarchy_info["section_title"],
                numbering=hierarchy_info["numbering"],
                parent_elements=hierarchy_info["parent_elements"],
                page_number=page_number,
                metadata={
                    "model_name": self.model_name,
                    "legal_specific": self.legal_specific,
                    "chunk_strategy": "hierarchical",
                    "has_structure_info": chunk_element is not None,
                    "page_determined": page_number is not None,
                },
            )

            hierarchical_chunks.append(hierarchical_chunk)
            current_position = end_pos

        return hierarchical_chunks

    def _find_chunk_element(
        self, start_pos: int, end_pos: int, elements: list[DocumentElement]
    ) -> DocumentElement | None:
        """Find the most relevant structural element for a chunk.

        Args:
            start_pos: Chunk start position
            end_pos: Chunk end position
            elements: List of document elements

        Returns:
            Most relevant DocumentElement or None
        """
        # Find elements that overlap with the chunk
        overlapping_elements = []

        for element in elements:
            # Check if element overlaps with chunk
            if element.start_position <= end_pos and element.end_position >= start_pos:
                overlap_size = min(end_pos, element.end_position) - max(
                    start_pos, element.start_position
                )
                if overlap_size > 0:
                    overlapping_elements.append((element, overlap_size))

        if not overlapping_elements:
            return None

        # Return element with largest overlap
        return max(overlapping_elements, key=lambda x: x[1])[0]

    def _extract_hierarchy_info(
        self, element: DocumentElement | None, all_elements: list[DocumentElement]
    ) -> dict[str, Any]:
        """Extract hierarchy information from a document element.

        Args:
            element: The document element (can be None)
            all_elements: All document elements for parent lookup

        Returns:
            Dictionary with hierarchy information
        """
        if not element:
            return {
                "level": 0,
                "element_type": None,
                "section_title": None,
                "numbering": None,
                "parent_elements": [],
            }

        # Get hierarchy path
        element.get_hierarchy_path()

        # Find parent elements
        parent_elements: list[str] = []
        current = element.parent
        while current:
            if current.numbering:
                parent_elements.insert(0, current.numbering.get_full_number())
            else:
                parent_elements.insert(0, current.element_type.value)
            current = current.parent

        return {
            "level": element.level,
            "element_type": element.element_type,
            "section_title": element.text[:100] + "..."
            if len(element.text) > 100
            else element.text,
            "numbering": element.numbering.get_full_number()
            if element.numbering
            else None,
            "parent_elements": parent_elements,
        }

    def _get_chunk_page_number(
        self, start_pos: int, end_pos: int, text: str, structure: DocumentStructure
    ) -> int | None:
        """Determine the page number for a chunk based on its position in the document.

        This is particularly useful for PDF documents where page boundaries are
        marked by structural elements or text markers.

        Args:
            start_pos: Chunk start position in the full document text
            end_pos: Chunk end position in the full document text
            text: The full document text
            structure: The detected document structure

        Returns:
            The page number (1-based) or None if not found
        """
        # Method 1: Look for PDF page markers in the text (common format: "--- Page X ---")
        import re

        page_pattern = r"--- Page (\d+) ---"
        matches = list(re.finditer(page_pattern, text))

        if matches:
            current_page = 1
            for match in matches:
                if start_pos < match.start():
                    return current_page
                try:
                    current_page = int(match.group(1))
                except (ValueError, IndexError):
                    current_page += 1
            return current_page

        # Method 2: Look for other common page markers
        page_markers = [
            r"\f",  # Form feed character (page break)
            r"\n\s*Page\s+(\d+)\s*\n",  # "Page X" markers
            r"\n\s*\[Page\s+(\d+)\]\s*\n",  # "[Page X]" markers
        ]

        for pattern in page_markers:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                page_count = 1
                for match in matches:
                    if start_pos >= match.end():
                        page_count += 1
                return page_count

        # Method 3: Estimate page based on character position (very rough)
        # Assuming average 2000-3000 characters per page for text documents
        if len(text) > 3000:  # Only estimate if document is substantial
            chars_per_page = 2500  # Conservative estimate
            estimated_page = (start_pos // chars_per_page) + 1
            return min(estimated_page, max(1, len(text) // chars_per_page + 1))

        # Default: return None if no page information can be determined
        return None

    def chunk_document(self, document: Document) -> list[Document]:
        """Chunk a Langchain Document with hierarchy preservation.

        Args:
            document: The Langchain Document to chunk

        Returns:
            List of Document chunks with hierarchy metadata
        """
        # Get hierarchical chunks
        hierarchical_chunks = self.chunk_text_with_hierarchy(document.page_content)

        # Convert to Langchain Document objects
        document_chunks = []
        for chunk in hierarchical_chunks:
            # Create metadata combining original and hierarchy info
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_index": chunk.chunk_index,
                    "chunk_tokens": chunk.token_count,
                    "chunk_chars": chunk.char_count,
                    "hierarchy_level": chunk.hierarchy_level,
                    "element_type": chunk.element_type.value
                    if chunk.element_type
                    else None,
                    "section_title": chunk.section_title,
                    "numbering": chunk.numbering,
                    "parent_elements": chunk.parent_elements,
                    "start_position": chunk.start_position,
                    "end_position": chunk.end_position,
                    **chunk.metadata,
                }
            )

            doc_chunk = Document(page_content=chunk.text, metadata=chunk_metadata)
            document_chunks.append(doc_chunk)

        return document_chunks

    def get_chunk_summary(self, chunks: list[HierarchicalChunk]) -> dict[str, Any]:
        """Get summary statistics about hierarchical chunks.

        Args:
            chunks: List of hierarchical chunks

        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {"total_chunks": 0}

        total_tokens = sum(chunk.token_count for chunk in chunks)
        total_chars = sum(chunk.char_count for chunk in chunks)

        # Analyze hierarchy levels
        hierarchy_levels: dict[int, int] = {}
        element_types: dict[str, int] = {}

        for chunk in chunks:
            # Count by hierarchy level
            level = chunk.hierarchy_level
            hierarchy_levels[level] = hierarchy_levels.get(level, 0) + 1

            # Count by element type
            element_type = chunk.element_type.value if chunk.element_type else "unknown"
            element_types[element_type] = element_types.get(element_type, 0) + 1

        return {
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "total_characters": total_chars,
            "average_tokens_per_chunk": round(total_tokens / len(chunks), 2),
            "average_chars_per_chunk": round(total_chars / len(chunks), 2),
            "min_tokens": min(chunk.token_count for chunk in chunks),
            "max_tokens": max(chunk.token_count for chunk in chunks),
            "hierarchy_distribution": hierarchy_levels,
            "element_type_distribution": element_types,
            "chunks_with_numbering": sum(1 for chunk in chunks if chunk.numbering),
            "max_hierarchy_level": max(chunk.hierarchy_level for chunk in chunks),
        }

    def analyze_chunk_sizes(self, chunks: list[HierarchicalChunk]) -> dict[str, Any]:
        """Analyze chunk size distribution and provide optimization insights.

        Args:
            chunks: List of hierarchical chunks to analyze

        Returns:
            Dictionary with detailed chunk size analysis
        """
        if not chunks:
            return {"total_chunks": 0, "analysis": "No chunks to analyze"}

        token_counts = [chunk.token_count for chunk in chunks]

        # Basic statistics
        total_tokens = sum(token_counts)
        avg_tokens = total_tokens / len(chunks)
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)

        # Size distribution analysis
        within_target = sum(
            1 for count in token_counts if self.target_min <= count <= self.target_max
        )
        too_small = sum(1 for count in token_counts if count < self.min_chunk_size)
        small_but_acceptable = sum(
            1
            for count in token_counts
            if self.min_chunk_size <= count < self.target_min
        )
        too_large = sum(1 for count in token_counts if count > self.max_chunk_size)
        large_but_acceptable = sum(
            1
            for count in token_counts
            if self.target_max < count <= self.max_chunk_size
        )

        # Performance metrics
        target_efficiency = (within_target / len(chunks)) * 100
        size_violations = too_small + too_large

        # Recommendations
        recommendations = []
        if too_small > 0:
            recommendations.append(
                f"Consider merging {too_small} small chunks (< {self.min_chunk_size} tokens)"
            )
        if too_large > 0:
            recommendations.append(
                f"Consider splitting {too_large} large chunks (> {self.max_chunk_size} tokens)"
            )
        if target_efficiency < 80:
            recommendations.append(
                f"Only {target_efficiency:.1f}% of chunks are in optimal range "
                f"({self.target_min}-{self.target_max} tokens). "
                f"Consider adjusting chunk_size parameter."
            )

        # Token efficiency
        avg_chunk_utilization = (
            (avg_tokens / self._chunk_size) * 100 if self._chunk_size > 0 else 0
        )

        return {
            "total_chunks": len(chunks),
            "total_tokens": total_tokens,
            "token_statistics": {
                "average": round(avg_tokens, 2),
                "minimum": min_tokens,
                "maximum": max_tokens,
                "median": sorted(token_counts)[len(token_counts) // 2],
            },
            "size_distribution": {
                "within_target_range": within_target,
                "too_small": too_small,
                "small_but_acceptable": small_but_acceptable,
                "too_large": too_large,
                "large_but_acceptable": large_but_acceptable,
            },
            "configuration": {
                "target_range": self.target_range,
                "min_chunk_size": self.min_chunk_size,
                "max_chunk_size": self.max_chunk_size,
                "configured_chunk_size": self._chunk_size,
                "configured_overlap": self._chunk_overlap,
            },
            "performance_metrics": {
                "target_efficiency_percent": round(target_efficiency, 2),
                "size_violations": size_violations,
                "avg_chunk_utilization_percent": round(avg_chunk_utilization, 2),
            },
            "recommendations": recommendations,
        }

    def get_token_usage_stats(self) -> dict[str, Any]:
        """Get comprehensive token usage statistics and performance metrics.

        Returns:
            Dictionary with token usage analytics
        """
        stats: dict[str, Any] = self.token_usage_stats.copy()

        # Calculate derived metrics
        if stats["total_chunks_created"] > 0:
            stats["avg_tokens_per_chunk"] = round(
                stats["total_tokens_processed"] / stats["total_chunks_created"], 2
            )

        if stats["token_counting_calls"] > 0:
            stats["efficiency_metrics"] = {
                "chunks_per_counting_call": round(
                    stats["total_chunks_created"] / stats["token_counting_calls"], 2
                ),
                "avg_tokens_per_call": round(
                    stats["total_tokens_processed"] / stats["token_counting_calls"], 2
                ),
            }

        # Add configuration context
        stats["configuration"] = {
            "model_name": self.model_name,
            "legal_specific": self.legal_specific,
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
            "target_range": self.target_range,
        }

        return stats

    def reset_token_usage_stats(self) -> None:
        """Reset token usage statistics for a new analysis session."""
        self.token_usage_stats = {
            "total_tokens_processed": 0,
            "total_chunks_created": 0,
            "token_counting_calls": 0,
            "large_chunks_split": 0,
            "small_chunks_merged": 0,
            "boundary_violations_fixed": 0,
        }
        self.logger.info("Token usage statistics reset")

    def estimate_processing_cost(
        self, text: str, cost_per_1k_tokens: float = 0.002
    ) -> dict[str, Any]:
        """Estimate the cost of processing text with the current model.

        Args:
            text: Text to analyze
            cost_per_1k_tokens: Cost per 1000 tokens in USD (default: $0.002 for GPT-4)

        Returns:
            Dictionary with cost analysis
        """
        # Count total tokens
        total_tokens = self._count_tokens_with_tracking(text)

        # Estimate chunks that will be created
        estimated_chunks = max(1, total_tokens // self._chunk_size)

        # Calculate overlap tokens (approximate)
        overlap_tokens = estimated_chunks * self._chunk_overlap

        # Total processing tokens (including overlap)
        processing_tokens = total_tokens + overlap_tokens

        # Calculate costs
        base_cost = (total_tokens / 1000) * cost_per_1k_tokens
        processing_cost = (processing_tokens / 1000) * cost_per_1k_tokens

        return {
            "input_tokens": total_tokens,
            "estimated_chunks": estimated_chunks,
            "processing_tokens": processing_tokens,
            "overlap_tokens": overlap_tokens,
            "base_cost_usd": round(base_cost, 6),
            "processing_cost_usd": round(processing_cost, 6),
            "cost_per_chunk_usd": round(processing_cost / estimated_chunks, 6),
            "model_name": self.model_name,
            "cost_per_1k_tokens": cost_per_1k_tokens,
        }

    def optimize_for_model(self, model_name: str) -> dict[str, Any]:
        """Get optimization recommendations for a specific model.

        Args:
            model_name: Target model name

        Returns:
            Dictionary with optimization recommendations
        """
        # Model-specific recommendations
        recommendations: dict[str, Any] = {
            "model_name": model_name,
            "current_config": {
                "chunk_size": self._chunk_size,
                "chunk_overlap": self._chunk_overlap,
                "target_range": self.target_range,
            },
        }

        if any(
            gpt4_variant in model_name.lower()
            for gpt4_variant in ["gpt-4", "gpt-4o", "gpt-4.1"]
        ):
            recommendations.update(
                {
                    "recommended_chunk_size": min(800, max(400, self._chunk_size)),
                    "recommended_overlap": self._calculate_optimal_overlap(
                        self._chunk_size, model_name
                    ),
                    "notes": [
                        "GPT-4 family handles larger contexts well, can use higher chunk sizes",
                        "Benefits from slightly higher overlap for context preservation",
                        "Legal-specific tokenization is beneficial for legal documents",
                    ],
                }
            )
        else:
            # Only GPT-4 family models are supported
            recommendations.update(
                {
                    "recommended_chunk_size": self._chunk_size,
                    "recommended_overlap": self._calculate_optimal_overlap(
                        self._chunk_size, model_name
                    ),
                    "notes": [
                        "Warning: Only GPT-4, GPT-4o, and GPT-4.1 models are officially supported",
                        "Using default recommendations for unsupported model",
                        "Consider switching to a supported GPT-4 family model",
                    ],
                }
            )

        return recommendations

    def validate_token_counter_integration(self) -> dict[str, Any]:
        """Validate TokenCounter integration and provide diagnostic information.

        Returns:
            Dictionary with validation results and diagnostics
        """
        validation_results = {
            "token_counter_initialized": self.token_counter is not None,
            "model_name": self.model_name,
            "legal_specific": self.legal_specific,
        }

        if self.token_counter:
            try:
                # Test basic token counting
                test_text = "This is a test sentence for token counting validation."
                token_count = self._count_tokens_with_tracking(test_text)

                validation_results.update(
                    {
                        "basic_counting_works": True,
                        "test_text_tokens": token_count,
                        "test_text_length": len(test_text),
                        "tokens_per_char_ratio": round(token_count / len(test_text), 3),
                    }
                )

                # Test model support
                model_info = TokenCounter.get_model_encoding_info(self.model_name)
                validation_results["model_support"] = model_info

                # Test legal-specific features if enabled
                if self.legal_specific:
                    legal_test = (
                        "Section 1.2.3 pursuant to Article IV, § 15 of the Agreement."
                    )
                    legal_tokens = self._count_tokens_with_tracking(legal_test)
                    validation_results.update(
                        {
                            "legal_specific_test": True,
                            "legal_text_tokens": legal_tokens,
                            "legal_patterns_detected": legal_tokens > 0,
                        }
                    )

            except Exception as e:
                validation_results.update(
                    {
                        "basic_counting_works": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )
        else:
            validation_results["error"] = "TokenCounter not initialized"

        # Add usage statistics
        validation_results["usage_stats"] = self.token_usage_stats

        return validation_results
