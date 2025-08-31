"""Vector storage and similarity search using PostgreSQL with pgvector."""

import logging

from fastapi import HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Import pgvector to register types with SQLAlchemy
try:
    import pgvector.sqlalchemy  # noqa: F401
except ImportError:
    # pgvector not available - will use fallback types
    pass

from app.core.settings import settings
from app.models.database import Chunk
from app.models.schemas import DocumentChunk
from app.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)

# Database connection setup
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


async def store_chunks_with_embeddings(
    chunks: list[DocumentChunk], document_id: str
) -> int:
    """
    Store document chunks with their embeddings in the database.

    Args:
        chunks: List of document chunks to store
        document_id: ID of the document these chunks belong to (as string)

    Returns:
        int: Number of chunks successfully stored

    Raises:
        HTTPException: If storage fails
    """
    if not chunks:
        return 0

    try:
        # Generate embeddings for all chunks
        embedding_service = get_embedding_service()
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = await embedding_service.generate_embeddings_batch(chunk_texts)

        # Store chunks with embeddings in database
        db = SessionLocal()
        try:
            stored_count = 0
            for chunk, embedding in zip(chunks, embeddings, strict=False):
                # Pass embedding as list directly - SQLAlchemy will handle conversion to vector type
                db_chunk = Chunk(
                    document_id=int(document_id),  # Convert string to int for database
                    content=chunk.text,
                    embedding=embedding,  # Pass as list - SQLAlchemy handles vector conversion
                    page=chunk.page_number,
                )
                db.add(db_chunk)
                stored_count += 1

            db.commit()
            logger.info(
                f"Successfully stored {stored_count} chunks with embeddings for document {document_id}"
            )
            return stored_count

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Failed to store chunks with embeddings: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to store chunks with embeddings: {str(e)}"
        ) from e


async def similarity_search(query: str, user_id: str, k: int = 5) -> list[dict]:
    """
    Perform similarity search in pgvector to find relevant chunks.

    Args:
        query: The search query text
        user_id: ID of the user to search within their documents
        k: Number of top results to return

    Returns:
        List[dict]: List of similar chunks with metadata

    Raises:
        HTTPException: If search fails
    """
    try:
        # Generate embedding for the query
        embedding_service = get_embedding_service()
        query_embedding = await embedding_service.generate_embedding(query)

        # Search for similar vectors in database
        results = await search_vectors(query_embedding, user_id, k)
        return results

    except Exception as e:
        logger.error(f"Failed to perform similarity search: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to perform similarity search: {str(e)}"
        ) from e


async def search_vectors(embedding: list[float], user_id: str, k: int) -> list[dict]:
    """
    Search for similar vectors in database using pgvector cosine similarity.

    Args:
        embedding: Query embedding vector
        user_id: ID of the user to search within their documents
        k: Number of top results to return

    Returns:
        List[dict]: List of similar chunks with similarity scores and metadata
    """
    try:
        db = SessionLocal()
        try:
            # Use pgvector's cosine distance operator (<=>)
            # Join with documents table to filter by user_id
            # Cast the embedding parameter to vector type to ensure compatibility
            query_sql = text("""
                SELECT
                    c.id,
                    c.document_id,
                    c.content,
                    c.page,
                    c.created_at,
                    d.filename,
                    d.user_id,
                    (c.embedding <=> CAST(:embedding AS vector)) as distance
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.user_id = :user_id
                ORDER BY c.embedding <=> CAST(:embedding AS vector)
                LIMIT :k
            """)

            result = db.execute(
                query_sql,
                {
                    "embedding": embedding,  # Pass as list - pgvector handles conversion
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
                        "similarity": 1.0
                        - float(row.distance),  # Convert distance to similarity
                        "created_at": row.created_at.isoformat()
                        if row.created_at
                        else None,
                    }
                )

            logger.info(f"Found {len(chunks)} similar chunks for user {user_id}")
            return chunks

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Failed to search vectors: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to search vectors: {str(e)}"
        ) from e
