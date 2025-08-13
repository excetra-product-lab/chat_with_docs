"""Models package initialization."""

# Export database models
from .database import (
    Base,
    Chunk,
    ChunkElementReference,
    Document,
    DocumentElement,
    DocumentHierarchy,
    ElementRelationship,
    User,
)

# Export hierarchy and relationship models
from .hierarchy_models import (
    DocumentRelationship,
    EnhancedDocumentElement,
    EnhancedElementType,
    EnhancedNumberingSystem,
    find_common_ancestors,
    merge_hierarchies,
)

# Export enhanced Langchain models
from .langchain_models import (
    EnhancedDocument,
    EnhancedDocumentChunk,
    EnhancedDocumentMetadata,
    convert_from_enhanced_document,
    convert_to_enhanced_document,
    integrate_with_langchain_pipeline,
)

# Export schema models
from .schemas import (
    Answer,
    DocumentChunk,
    DocumentCreate,
    DocumentMetadata,
    DocumentProcessingResult,
    ProcessingConfig,
    ProcessingStats,
    Query,
)
from .schemas import (
    Document as DocumentSchema,
)

__all__ = [
    # Database models
    "Base",
    "User",
    "Document",
    "Chunk",
    "DocumentHierarchy",
    "DocumentElement",
    "ChunkElementReference",
    "ElementRelationship",
    # Schema models
    "DocumentCreate",
    "DocumentSchema",
    "DocumentMetadata",
    "DocumentChunk",
    "ProcessingStats",
    "DocumentProcessingResult",
    "ProcessingConfig",
    "Query",
    "Answer",
    # Enhanced Langchain models
    "EnhancedDocument",
    "EnhancedDocumentMetadata",
    "EnhancedDocumentChunk",
    "convert_to_enhanced_document",
    "convert_from_enhanced_document",
    "integrate_with_langchain_pipeline",
    # Hierarchy and relationship models
    "EnhancedElementType",
    "EnhancedNumberingSystem",
    "EnhancedDocumentElement",
    "DocumentRelationship",
    "DocumentHierarchy",
    "merge_hierarchies",
    "find_common_ancestors",
]
