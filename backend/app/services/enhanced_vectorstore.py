"""
Enhanced vector store integration for Langchain-compatible models.

This module provides enhanced vector storage and retrieval capabilities that
integrate with the enhanced document models and support advanced features
like hierarchical metadata, relationship tracking, and confidence scoring.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from fastapi import HTTPException
from langchain_core.documents import Document as LangchainDocument
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Import pgvector to register types with SQLAlchemy
try:
    import pgvector.sqlalchemy  # noqa: F401
except ImportError:
    # pgvector not available - will use fallback types
    pass

from app.core.settings import settings
from app.models.database import (
    Chunk,
    ChunkElementReference,
    Document,
    ElementRelationship,
)
from app.models.database import DocumentElement as DBDocumentElement
from app.models.database import DocumentHierarchy as DBDocumentHierarchy
from app.models.hierarchy_models import (
    DocumentHierarchy,
)
from app.models.langchain_models import (
    EnhancedDocument,
    EnhancedDocumentChunk,
    # EnhancedCitation removed
)
from app.services.embedding_service import get_embedding_service
from app.utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)

# Database connection setup
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class EnhancedVectorStore:
    """
    Enhanced vector store with support for hierarchical documents,
    relationship tracking, and advanced metadata.
    """

    def __init__(self, embedding_service=None):
        """Initialize the enhanced vector store."""
        self.embedding_service = embedding_service or get_embedding_service()
        self.logger = logging.getLogger(__name__)

    async def store_enhanced_document(
        self, enhanced_doc: EnhancedDocument, hierarchy: DocumentHierarchy | None = None
    ) -> int:
        """
        Store an enhanced document with all its chunks and metadata.

        Args:
            enhanced_doc: Enhanced document to store
            hierarchy: Optional document hierarchy for additional metadata

        Returns:
            Number of chunks successfully stored
        """
        if not enhanced_doc.chunks:
            return 0

        try:
            # Generate embeddings for all chunks
            chunk_texts = [chunk.text for chunk in enhanced_doc.chunks]
            embeddings = await self.embedding_service.generate_embeddings_batch(
                chunk_texts
            )

            # Update chunk token counts if not already set
            for chunk in enhanced_doc.chunks:
                if chunk.token_count == 0:
                    chunk.token_count = await self._count_tokens(chunk.text)

            # Store in database
            db = SessionLocal()
            try:
                stored_count = 0

                # Create or update document record
                document_id = await self._store_document_record(db, enhanced_doc)

                for chunk, embedding in zip(
                    enhanced_doc.chunks, embeddings, strict=False
                ):
                    # Create enhanced metadata for the chunk
                    enhanced_metadata = self._create_enhanced_chunk_metadata(
                        chunk, enhanced_doc, hierarchy
                    )

                    # Store chunk with enhanced metadata
                    db_chunk = Chunk(
                        document_id=document_id,
                        content=chunk.text,
                        embedding=embedding,
                        page=chunk.page_number,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        chunk_hash=chunk.chunk_hash,
                        chunk_type=chunk.chunk_type,
                        hierarchical_level=chunk.hierarchical_level,
                        token_count=chunk.token_count,
                        quality_score=chunk.quality_score,
                        enhanced_metadata=enhanced_metadata,
                    )
                    db.add(db_chunk)
                    db.flush()  # Get the chunk ID

                    # Store chunk-element references if hierarchy is provided
                    if hierarchy and chunk.chunk_references:
                        for element_ref in chunk.chunk_references:
                            if element_ref in hierarchy.elements:
                                # Find the database element ID
                                hierarchy_record = (
                                    db.query(DBDocumentHierarchy)
                                    .filter(
                                        DBDocumentHierarchy.hierarchy_id
                                        == hierarchy.hierarchy_id
                                    )
                                    .first()
                                )

                                if hierarchy_record:
                                    element = (
                                        db.query(DBDocumentElement)
                                        .filter(
                                            DBDocumentElement.hierarchy_id
                                            == hierarchy_record.id,
                                            DBDocumentElement.element_id == element_ref,
                                        )
                                        .first()
                                    )

                                    if element:
                                        chunk_element_ref = ChunkElementReference(
                                            chunk_id=db_chunk.id,
                                            element_id=element.id,
                                            reference_type="contains",
                                            confidence_score=1.0,
                                        )
                                        db.add(chunk_element_ref)

                    stored_count += 1

                db.commit()
                self.logger.info(
                    f"Successfully stored enhanced document {enhanced_doc.filename}: "
                    f"{stored_count} chunks"
                )
                return stored_count

            finally:
                db.close()

        except Exception as e:
            self.logger.error(f"Failed to store enhanced document: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to store enhanced document: {str(e)}"
            ) from e

    # enhanced_similarity_search method removed (was citation-related)

    async def store_hierarchy(self, hierarchy: DocumentHierarchy) -> str:
        """
        Store document hierarchy information for retrieval enhancement.

        Args:
            hierarchy: Document hierarchy to store

        Returns:
            Hierarchy ID
        """
        try:
            # Serialize hierarchy for storage
            hierarchy_data = hierarchy.to_dict()

            db = SessionLocal()
            try:
                # Check if hierarchy already exists
                existing_hierarchy = (
                    db.query(DBDocumentHierarchy)
                    .filter(DBDocumentHierarchy.hierarchy_id == hierarchy.hierarchy_id)
                    .first()
                )

                if existing_hierarchy:
                    # Update existing hierarchy
                    existing_hierarchy.hierarchy_data = hierarchy_data
                    existing_hierarchy.total_elements = hierarchy.total_elements
                    existing_hierarchy.updated_at = datetime.utcnow()
                    db.commit()
                    hierarchy_id = existing_hierarchy.hierarchy_id
                else:
                    # Create new hierarchy record
                    db_hierarchy = DBDocumentHierarchy(
                        hierarchy_id=hierarchy.hierarchy_id,
                        document_filename=hierarchy.document_filename,
                        total_elements=hierarchy.total_elements,
                        hierarchy_data=hierarchy_data,
                    )
                    db.add(db_hierarchy)
                    db.flush()

                    # Store individual elements
                    for element_id, element in hierarchy.elements.items():
                        db_element = DBDocumentElement(
                            hierarchy_id=db_hierarchy.id,
                            element_id=element_id,
                            element_type=element.element_type.element_type.value,
                            semantic_role=element.semantic_role,
                            importance_score=element.importance_score,
                            level=element.level,
                            content=element.content,
                            start_char=element.start_char,
                            end_char=element.end_char,
                            page_number=element.page_number,
                            element_metadata=element.additional_metadata,
                        )
                        db.add(db_element)

                    # Store relationships
                    for relationship in hierarchy.relationships:
                        # Find the database IDs for the elements
                        source_element = (
                            db.query(DBDocumentElement)
                            .filter(
                                DBDocumentElement.hierarchy_id == db_hierarchy.id,
                                DBDocumentElement.element_id
                                == relationship.source_element_id,
                            )
                            .first()
                        )
                        target_element = (
                            db.query(DBDocumentElement)
                            .filter(
                                DBDocumentElement.hierarchy_id == db_hierarchy.id,
                                DBDocumentElement.element_id
                                == relationship.target_element_id,
                            )
                            .first()
                        )

                        if source_element and target_element:
                            db_relationship = ElementRelationship(
                                source_element_id=source_element.id,
                                target_element_id=target_element.id,
                                relationship_type=relationship.relationship_type.value,
                                confidence_score=relationship.confidence_score,
                                relationship_metadata=relationship.additional_metadata,
                            )
                            db.add(db_relationship)

                    db.commit()
                    hierarchy_id = hierarchy.hierarchy_id

                self.logger.info(
                    f"Successfully stored hierarchy {hierarchy_id}: "
                    f"{hierarchy.total_elements} elements, {len(hierarchy.relationships)} relationships"
                )
                return hierarchy_id

            finally:
                db.close()

        except Exception as e:
            self.logger.error(f"Failed to store hierarchy: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to store hierarchy: {str(e)}"
            ) from e

    async def get_related_chunks(
        self,
        chunk_id: str,
        relation_types: list[str] | None = None,
        max_distance: int = 2,
    ) -> list[EnhancedDocumentChunk]:
        """
        Get chunks related to a given chunk through hierarchy or relationships.

        Args:
            chunk_id: ID of the source chunk
            relation_types: Types of relationships to include
            max_distance: Maximum relationship distance to traverse

        Returns:
            List of related enhanced chunks
        """
        try:
            db = SessionLocal()
            try:
                # Find elements referenced by this chunk
                chunk_refs = (
                    db.query(ChunkElementReference)
                    .filter(ChunkElementReference.chunk_id == int(chunk_id))
                    .all()
                )

                if not chunk_refs:
                    return []

                related_element_ids = set()

                # For each element referenced by the chunk, find related elements
                for chunk_ref in chunk_refs:
                    element_id = chunk_ref.element_id
                    related_ids = await self._find_related_elements(
                        db, element_id, relation_types, max_distance, set()
                    )
                    related_element_ids.update(related_ids)

                # Find chunks that reference these related elements
                related_chunk_refs = (
                    db.query(ChunkElementReference)
                    .filter(
                        ChunkElementReference.element_id.in_(related_element_ids),
                        ChunkElementReference.chunk_id
                        != int(chunk_id),  # Exclude original chunk
                    )
                    .all()
                )

                related_chunks = []
                for chunk_ref in related_chunk_refs:
                    chunk = (
                        db.query(Chunk).filter(Chunk.id == chunk_ref.chunk_id).first()
                    )
                    if chunk:
                        # Convert to EnhancedDocumentChunk
                        enhanced_chunk = EnhancedDocumentChunk(
                            text=chunk.content,
                            chunk_index=chunk.id,
                            page_number=chunk.page or 0,
                            start_char=chunk.start_char or 0,
                            end_char=chunk.end_char or len(chunk.content),
                            chunk_hash=chunk.chunk_hash or "",
                            chunk_type=chunk.chunk_type or "text",
                            hierarchical_level=chunk.hierarchical_level or 0,
                            token_count=chunk.token_count or 0,
                            quality_score=chunk.quality_score or 1.0,
                            chunk_references=[str(chunk_ref.element_id)],
                        )
                        related_chunks.append(enhanced_chunk)

                self.logger.info(
                    f"Found {len(related_chunks)} related chunks for {chunk_id}"
                )
                return related_chunks

            finally:
                db.close()

        except Exception as e:
            self.logger.error(f"Failed to get related chunks: {str(e)}")
            return []

    async def _find_related_elements(
        self,
        db,
        element_id: int,
        relation_types: list[str] | None,
        max_distance: int,
        visited: set,
    ) -> set:
        """Recursively find related elements up to max_distance."""
        if max_distance <= 0 or element_id in visited:
            return set()

        visited.add(element_id)
        related_ids = set()

        # Find direct relationships
        relationships = (
            db.query(ElementRelationship)
            .filter(
                (ElementRelationship.source_element_id == element_id)
                | (ElementRelationship.target_element_id == element_id)
            )
            .all()
        )

        for rel in relationships:
            if relation_types and rel.relationship_type not in relation_types:
                continue

            # Add the related element
            if rel.source_element_id == element_id:
                related_ids.add(rel.target_element_id)
                # Recursively find elements related to this one
                deeper_related = await self._find_related_elements(
                    db, rel.target_element_id, relation_types, max_distance - 1, visited
                )
                related_ids.update(deeper_related)
            else:
                related_ids.add(rel.source_element_id)
                # Recursively find elements related to this one
                deeper_related = await self._find_related_elements(
                    db, rel.source_element_id, relation_types, max_distance - 1, visited
                )
                related_ids.update(deeper_related)

        return related_ids

    async def _enhanced_vector_search(
        self, embedding: list[float], user_id: int, k: int
    ) -> list[dict[str, Any]]:
        """Enhanced vector search with additional metadata."""
        try:
            db = SessionLocal()
            try:
                # Enhanced query with more metadata
                query_sql = text("""
                    SELECT
                        c.id,
                        c.document_id,
                        c.content,
                        c.page,
                        c.created_at,
                        d.filename,
                        d.user_id,
                        d.status,
                        d.storage_key,
                        (c.embedding <=> CAST(:embedding AS vector)) as distance,
                        (1.0 - (c.embedding <=> CAST(:embedding AS vector))) as similarity
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE d.user_id = :user_id
                    ORDER BY c.embedding <=> CAST(:embedding AS vector)
                    LIMIT :k
                """)

                result = db.execute(
                    query_sql,
                    {
                        "embedding": embedding,
                        "user_id": user_id,
                        "k": k,
                    },
                )

                chunks = []
                for row in result:
                    chunks.append(
                        {
                            "id": row.id,
                            "document_id": row.document_id,
                            "content": row.content,
                            "page": row.page,
                            "filename": row.filename,
                            "distance": float(row.distance),
                            "similarity": float(row.similarity),
                            "status": row.status,
                            "storage_key": row.storage_key,
                            "created_at": row.created_at.isoformat()
                            if row.created_at
                            else None,
                        }
                    )

                return chunks

            finally:
                db.close()

        except Exception as e:
            self.logger.error(f"Enhanced vector search failed: {str(e)}")
            raise

    def _create_enhanced_chunk_metadata(
        self,
        chunk: EnhancedDocumentChunk,
        document: EnhancedDocument,
        hierarchy: DocumentHierarchy | None = None,
    ) -> dict[str, Any]:
        """Create enhanced metadata for chunk storage."""
        metadata = {
            "chunk_hash": chunk.chunk_hash,
            "langchain_source": chunk.langchain_source,
            "chunk_type": chunk.chunk_type,
            "hierarchical_level": chunk.hierarchical_level,
            "token_count": chunk.token_count,
            "quality_score": chunk.quality_score,
            "document_metadata": {
                "total_chars": document.metadata.total_chars,
                "total_tokens": document.metadata.total_tokens,
                "structure_detected": document.metadata.structure_detected,
                "langchain_source": document.metadata.langchain_source,
            },
        }

        # Add hierarchy information if available
        if hierarchy and chunk.chunk_references:
            for ref in chunk.chunk_references:
                if ref in hierarchy.elements:
                    element = hierarchy.elements[ref]
                    metadata["hierarchy_element"] = {
                        "element_type": element.element_type.element_type.value,
                        "semantic_role": element.semantic_role,
                        "importance_score": element.importance_score,
                        "level": element.level,
                    }
                    break

        return metadata

    def _calculate_confidence_score(self, result: dict[str, Any]) -> float:
        """Calculate confidence score for search result."""
        # Base confidence from similarity
        confidence = result["similarity"]

        # Adjust based on document status
        if result.get("status") == "completed":
            confidence *= 1.1
        elif result.get("status") == "processing":
            confidence *= 0.9

        # Adjust based on content length (avoid very short snippets)
        content_length = len(result["content"])
        if content_length < 50:
            confidence *= 0.8
        elif content_length > 500:
            confidence *= 1.05

        return min(confidence, 1.0)

    def _create_snippet(self, content: str, max_length: int = 200) -> str:
        """Create snippet from content."""
        if len(content) <= max_length:
            return content

        # Try to break at word boundary
        snippet = content[:max_length]
        last_space = snippet.rfind(" ")
        if last_space > max_length * 0.8:  # If space is reasonably close to end
            snippet = snippet[:last_space]

        return snippet + "..."

    # Citation helper functions removed (_add_hierarchical_context and _add_relationship_context)

    # _store_document_record function content was corrupted during citation removal,
    # the correct function is defined later in the file

    async def _get_element_path(self, db, element_id: int) -> list[dict[str, Any]]:
        """Get the path from element to root."""
        path = []
        current_id = element_id
        visited = set()

        while current_id and current_id not in visited:
            visited.add(current_id)

            element = (
                db.query(DBDocumentElement)
                .filter(DBDocumentElement.id == current_id)
                .first()
            )

            if not element:
                break

            path.append(
                {
                    "element_id": element.element_id,
                    "element_type": element.element_type,
                    "semantic_role": element.semantic_role,
                    "level": element.level,
                }
            )

            # Find parent
            parent_rel = (
                db.query(ElementRelationship)
                .filter(
                    ElementRelationship.target_element_id == current_id,
                    ElementRelationship.relationship_type == "parent-child",
                )
                .first()
            )

            current_id = parent_rel.source_element_id if parent_rel else None

        return path

    # _add_relationship_context function removed (was citation-related)

    async def _store_document_record(self, db, enhanced_doc: EnhancedDocument) -> int:
        """Store or update document record and return ID."""
        # Check if document already exists
        existing_doc = (
            db.query(Document)
            .filter(Document.filename == enhanced_doc.filename)
            .first()
        )

        if existing_doc:
            # Update existing document
            existing_doc.status = enhanced_doc.status
            existing_doc.updated_at = datetime.utcnow()
            db.commit()
            return existing_doc.id
        else:
            # Create new document record
            new_doc = Document(
                filename=enhanced_doc.filename,
                user_id=enhanced_doc.user_id or 1,  # Default user ID
                status=enhanced_doc.status,
                storage_key=enhanced_doc.storage_key,
            )
            db.add(new_doc)
            db.flush()  # Get the ID
            return new_doc.id

    async def _count_tokens(self, text: str) -> int:
        """Count tokens in text using proper tokenizer."""
        try:
            token_counter = TokenCounter()
            return token_counter.count_tokens(text)
        except Exception as e:
            self.logger.error(f"Failed to count tokens: {str(e)}")
            # Fallback to word count approximation
            return len(text.split())

    async def enhanced_similarity_search(
        self,
        query: str,
        user_id: int,
        k: int = 5,
        include_hierarchy: bool = False,
        include_relationships: bool = False,
        confidence_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Enhanced similarity search that returns search results with metadata.

        Args:
            query: Search query text
            user_id: User ID for filtering
            k: Number of results to return
            include_hierarchy: Whether to include hierarchical context (placeholder)
            include_relationships: Whether to include relationship context (placeholder)
            confidence_threshold: Minimum confidence/similarity score to include results

        Returns:
            List of search results with metadata
        """
        # Generate embedding for the query
        embedding = await self.embedding_service.generate_embedding(query)

        # Perform vector search
        search_results = await self._enhanced_vector_search(
            embedding=embedding, user_id=user_id, k=k
        )

        # Apply confidence threshold filtering if specified
        if confidence_threshold > 0.0:
            search_results = [
                result
                for result in search_results
                if result.get("similarity", 0.0) >= confidence_threshold
            ]

        # Apply hierarchical context if requested
        if include_hierarchy:
            search_results = await self._add_hierarchical_context(search_results)

        # Apply relationship context if requested
        if include_relationships:
            search_results = await self._add_relationship_context(search_results)

        return search_results

    async def _add_hierarchical_context(
        self, search_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Add hierarchical context to search results.

        TODO: Implement hierarchical context enhancement
        For now, returns results unchanged.
        """
        return search_results

    async def _add_relationship_context(
        self, search_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Add relationship context to search results.

        TODO: Implement relationship context enhancement
        For now, returns results unchanged.
        """
        return search_results


# Utility functions for vector store integration


async def store_enhanced_document_with_embeddings(
    enhanced_doc: EnhancedDocument, hierarchy: DocumentHierarchy | None = None
) -> int:
    """
    Convenience function to store enhanced document with embeddings.

    Args:
        enhanced_doc: Enhanced document to store
        hierarchy: Optional document hierarchy

    Returns:
        Number of chunks stored
    """
    vector_store = EnhancedVectorStore()
    return await vector_store.store_enhanced_document(enhanced_doc, hierarchy)


# enhanced_search_with_context function removed (was citation-related)


def create_langchain_retriever(
    vector_store: EnhancedVectorStore, user_id: int, k: int = 5
):
    """
    Create a Langchain-compatible retriever from the enhanced vector store.

    Args:
        vector_store: Enhanced vector store instance
        user_id: User ID for filtering
        k: Number of results to return

    Returns:
        Langchain-compatible retriever
    """

    class EnhancedRetriever:
        """Langchain-compatible retriever wrapper."""

        def __init__(self, store: EnhancedVectorStore, user_id: int, k: int):
            self.store = store
            self.user_id = user_id
            self.k = k

        async def aget_relevant_documents(self, query: str) -> list[LangchainDocument]:
            """Get relevant documents as Langchain Documents."""
            # Generate embedding for the query
            embedding = await self.store.embedding_service.generate_embedding(query)

            # Perform vector search
            search_results = await self.store._enhanced_vector_search(
                embedding=embedding, user_id=self.user_id, k=self.k
            )

            # Convert to Langchain documents
            documents = []
            for result in search_results:
                doc = LangchainDocument(
                    page_content=result.get("content", ""),
                    metadata={
                        "id": result.get("id"),
                        "document_id": result.get("document_id"),
                        "filename": result.get("filename"),
                        "page": result.get("page"),
                        "similarity": result.get("similarity"),
                        "distance": result.get("distance"),
                        "status": result.get("status"),
                    },
                )
                documents.append(doc)

            return documents

        def get_relevant_documents(self, query: str) -> list[LangchainDocument]:
            """Sync version - runs the async method using asyncio."""
            try:
                # Run the async method in a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.aget_relevant_documents(query))
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"Error in sync get_relevant_documents: {e}")
                # If asyncio fails, try to get existing event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If we're already in an async context, we can't use run_until_complete
                        raise RuntimeError(
                            "Cannot use sync get_relevant_documents from within async context. "
                            "Use aget_relevant_documents instead."
                        )
                    else:
                        return loop.run_until_complete(
                            self.aget_relevant_documents(query)
                        )
                except Exception:
                    raise NotImplementedError(
                        "Sync method unavailable in this context. Use aget_relevant_documents for async operations"
                    ) from e

    return EnhancedRetriever(vector_store, user_id, k)
