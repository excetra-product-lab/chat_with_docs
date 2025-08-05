import asyncio
import random
import re

import tiktoken
from langchain_core.messages import HumanMessage, SystemMessage
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

from app.core.exceptions import (
    AzureAPIError,
    CitationParsingError,
    DocumentNotFoundError,
    EmptyContextError,
    LLMGenerationError,
    TokenLimitExceededError,
    VectorSearchError,
)
from app.core.langchain_config import langchain_config
from app.core.logging_config import get_rag_logger
from app.core.prompts import SystemPrompts
from app.core.vectorstore import similarity_search
from app.models.schemas import Answer, Citation, RetrievedChunk


def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Estimate the number of tokens in a text string for a given model.
    
    Args:
        text: The input text to count tokens for
        model: The model name to use for tokenization
        
    Returns:
        Estimated number of tokens
    """
    try:
        # Use tiktoken for accurate token counting
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to rough estimation (4 characters per token)
        return len(text) // 4


async def answer_question(
    question: str,
    user_id: int,
    document_ids: list[str] | None = None,
    limit: int | None = None
) -> Answer:
    """
    Process a question through the RAG pipeline with comprehensive error handling.
    
    Args:
        question: The user's question
        user_id: The authenticated user's ID
        document_ids: Optional list of document IDs to filter search results
        limit: Optional limit on number of chunks to retrieve (default: 5)
        
    Returns:
        Answer object with response, citations, and confidence
        
    Raises:
        DocumentNotFoundError: When no relevant documents are found
        VectorSearchError: When vector search fails
        EmptyContextError: When context cannot be built from chunks
        LLMGenerationError: When answer generation fails
        CitationParsingError: When citation extraction fails
    """
    logger = get_rag_logger(__name__, {
        "user_id": user_id,
        "question_preview": question[:50] + "..." if len(question) > 50 else question,
        "document_ids": document_ids,
        "limit": limit
    })

    logger.info("Starting RAG pipeline processing")

    try:
        # Step 1: Search for relevant chunks
        search_limit = limit if limit is not None else 5
        logger.info(f"Performing vector similarity search with limit={search_limit}")
        try:
            relevant_chunks = await similarity_search(question, user_id, k=search_limit)
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise VectorSearchError(
                query=question,
                reason=f"Database or search index error: {str(e)}",
                user_id=user_id,
                original_error=e
            )

        # Step 2: Filter by document IDs if specified
        if document_ids is not None:
            logger.info(f"Filtering results by document IDs: {document_ids}")
            # Convert document_ids to integers for comparison (assuming they're stored as integers)
            try:
                doc_id_ints = [int(doc_id) for doc_id in document_ids]
                filtered_chunks = [
                    chunk for chunk in relevant_chunks
                    if chunk.get('document_id') in doc_id_ints
                ]
                logger.info(f"Filtered from {len(relevant_chunks)} to {len(filtered_chunks)} chunks")
                relevant_chunks = filtered_chunks
            except ValueError as e:
                logger.warning(f"Invalid document ID format: {e}")
                # Continue with unfiltered results rather than failing

        # Step 3: Validate search results
        if not relevant_chunks:
            logger.warning("No relevant chunks found for query")
            raise DocumentNotFoundError(query=question, user_id=user_id)

        logger.info(f"Found {len(relevant_chunks)} relevant chunks")

        # Step 4: Build context from chunks
        logger.info("Building context from retrieved chunks")
        try:
            context, chunk_mapping = build_context(relevant_chunks)
        except Exception as e:
            logger.error(f"Context building failed: {str(e)}")
            raise EmptyContextError(
                chunks_retrieved=len(relevant_chunks),
                reason=f"Context formatting error: {str(e)}"
            )

        # Validate context is not empty
        if not context or context.strip() == "No relevant documents found.":
            raise EmptyContextError(
                chunks_retrieved=len(relevant_chunks),
                reason="All chunks were empty or invalid"
            )

        # Step 5: Generate answer using LLM
        logger.info("Generating answer with LLM")
        try:
            answer_text = await generate_answer(question, context)
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            # Re-raise specific exceptions, wrap others
            if isinstance(e, (AzureAPIError, TokenLimitExceededError, LLMGenerationError)):
                raise
            else:
                raise LLMGenerationError(
                    reason=f"Unexpected error during generation: {str(e)}",
                    question=question,
                    original_error=e
                )

        # Validate generated answer
        if not answer_text or not answer_text.strip():
            raise LLMGenerationError(
                reason="LLM returned empty response",
                question=question
            )

        # Step 6: Extract citations
        logger.info("Extracting citations from answer")
        try:
            citations = extract_citations(answer_text, relevant_chunks)
        except Exception as e:
            logger.warning(f"Citation extraction failed: {str(e)}")
            # Citation parsing is not critical - continue with empty citations
            citations = []

        # Step 7: Format retrieved chunks for response
        logger.info("Formatting retrieved chunks for response")
        retrieved_chunks = []
        for chunk in relevant_chunks:
            try:
                # Extract metadata
                metadata = chunk.get('metadata', {})
                document_name = metadata.get('filename') or metadata.get('source', 'Unknown Document')
                page_num = metadata.get('page') or metadata.get('page_number')

                # Create document ID (use actual document_id if available, otherwise generate one)
                document_id = str(chunk.get('document_id', document_name))

                # Get similarity score (default to 0.0 if not available)
                similarity_score = chunk.get('similarity_score', chunk.get('score', 0.0))

                retrieved_chunk = RetrievedChunk(
                    document_id=document_id,
                    document_name=document_name,
                    content=chunk.get('content', ''),
                    page=int(page_num) if page_num is not None else None,
                    similarity_score=float(similarity_score),
                    metadata=metadata
                )
                retrieved_chunks.append(retrieved_chunk)
            except Exception as e:
                logger.warning(f"Failed to format chunk for response: {e}")
                continue

        logger.info(f"Successfully completed RAG pipeline with {len(citations)} citations and {len(retrieved_chunks)} chunks")

        return Answer(
            answer=answer_text,
            citations=citations,
            confidence=0.95,
            query=question,
            chunks=retrieved_chunks,
            metadata={
                "chunks_retrieved": len(relevant_chunks),
                "chunks_formatted": len(retrieved_chunks),
                "citations_found": len(citations),
                "search_limit": search_limit,
                "document_ids_filter": document_ids
            }
        )

    except (DocumentNotFoundError, VectorSearchError, EmptyContextError,
            LLMGenerationError, AzureAPIError, TokenLimitExceededError, CitationParsingError):
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Unexpected error in RAG pipeline: {str(e)}", exc_info=True)
        raise LLMGenerationError(
            reason=f"Unexpected pipeline error: {str(e)}",
            question=question,
            original_error=e
        )


def build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Build context from retrieved chunks with error handling
    
    Args:
        chunks: List of chunk dictionaries containing content, metadata, and source info
        
    Returns:
        Tuple of (formatted context string for LLM processing, chunk mapping for citation generation)
        
    Raises:
        EmptyContextError: When no valid context can be built from chunks
    """
    logger = get_rag_logger(__name__)

    if not chunks:
        logger.warning("No chunks provided for context building")
        raise EmptyContextError(chunks_retrieved=0, reason="No chunks provided")

    context_parts = []
    chunk_mapping = []

    for i, chunk in enumerate(chunks, 1):
        # Extract metadata
        content = chunk.get('content', '') or ''
        content = content.strip() if content else ''
        if not content:
            continue

        # Get document metadata
        metadata = chunk.get('metadata', {})
        filename = metadata.get('filename', metadata.get('source', f'Document {i}'))
        page_num = metadata.get('page', metadata.get('page_number'))

        # Format the chunk with source information
        chunk_header = f"=== Source {i}: {filename}"
        if page_num:
            chunk_header += f" (Page {page_num})"
        chunk_header += " ==="

        # Add the formatted chunk to context
        context_parts.append(f"{chunk_header}\n{content}\n")

        # Create mapping entry for citation generation
        chunk_mapping.append({
            'source_id': i,
            'content': content,
            'filename': filename,
            'page': page_num,
            'original_chunk': chunk
        })

    if not context_parts:
        logger.warning(f"No valid content found in {len(chunks)} chunks")
        raise EmptyContextError(
            chunks_retrieved=len(chunks),
            reason="All chunks were empty or invalid"        )

    # Join all context parts with separators
    full_context = "\n".join(context_parts)

    # Add a header explaining the context structure
    context_header = f"The following {len(context_parts)} document excerpt(s) contain relevant information:\n\n"

    return context_header + full_context, chunk_mapping


async def generate_answer(question: str, context: str) -> str:
    """
    Generate answer using Azure OpenAI with comprehensive error handling and retry logic
    
    Args:
        question: The user's question
        context: The formatted context from retrieved documents
        
    Returns:
        Generated answer string
        
    Raises:
        TokenLimitExceededError: When input exceeds token limits
        AzureAPIError: When Azure OpenAI API fails
        LLMGenerationError: When generation fails for other reasons
    """

    logger = get_rag_logger(__name__)

    # Input validation
    if not question or not question.strip():
        logger.warning("Empty question provided to generate_answer")
        raise LLMGenerationError(reason="Empty question provided")

    if not context or not context.strip():
        logger.warning("Empty context provided to generate_answer")
        raise EmptyContextError(reason="No context provided for answer generation")

    # Token limit validation
    question_tokens = estimate_tokens(question)
    context_tokens = estimate_tokens(context)
    system_prompt_tokens = estimate_tokens(SystemPrompts.get_rag_system_prompt())

    # Rough estimation of total input tokens
    total_input_tokens = question_tokens + context_tokens + system_prompt_tokens

    # GPT-4o-mini has a 128k context window, leave room for response
    max_input_tokens = 120000  # Reserve 8k tokens for response

    if total_input_tokens > max_input_tokens:
        logger.warning(f"Input too long: {total_input_tokens} tokens (max: {max_input_tokens})")
        raise TokenLimitExceededError(
            token_count=total_input_tokens,
            token_limit=max_input_tokens,
            component="total_input"
        )

    if question_tokens > 4000:  # Reasonable limit for questions
        logger.warning(f"Question too long: {question_tokens} tokens")
        raise TokenLimitExceededError(
            token_count=question_tokens,
            token_limit=4000,
            component="question"
        )

    # Retry configuration
    max_retries = 3
    base_delay = 1.0  # Initial delay in seconds
    max_delay = 60.0  # Maximum delay in seconds

    for attempt in range(max_retries + 1):
        try:
            # Get the configured LLM instance
            llm = langchain_config.llm

            # Format the system prompt with context using the structured prompt system
            formatted_system_prompt = SystemPrompts.format_rag_prompt(context)

            # Create messages for the chat completion
            messages = [
                SystemMessage(content=formatted_system_prompt),
                HumanMessage(content=question)
            ]

            # Log attempt details
            logger.info(f"Generating answer for question (attempt {attempt + 1}/{max_retries + 1}): {question[:50]}...")

            # Generate response using LangChain's async invoke
            response = await llm.ainvoke(messages)

            # Extract the content from the response
            answer = response.content if hasattr(response, 'content') else str(response)

            # Validate response
            if not answer or not answer.strip():
                logger.warning("Empty response received from LLM")
                if attempt < max_retries:
                    continue
                raise LLMGenerationError(
                    reason="LLM returned empty response after all retries",
                    question=question,
                    attempts=max_retries + 1
                )

            logger.info(f"Successfully generated answer for question: {question[:50]}...")
            return answer.strip()

        except RateLimitError as e:
            logger.warning(f"Rate limit exceeded (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries:
                # Exponential backoff with jitter for rate limits
                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                logger.info(f"Waiting {delay:.2f} seconds before retry due to rate limit...")
                await asyncio.sleep(delay)
                continue
            raise AzureAPIError(
                message="Rate limit exceeded after all retries",
                status_code=429,
                error_type="RateLimitError",
                original_error=e
            )

        except APITimeoutError as e:
            logger.warning(f"API timeout (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.info(f"Waiting {delay:.2f} seconds before retry due to timeout...")
                await asyncio.sleep(delay)
                continue
            raise AzureAPIError(
                message="Request timed out after all retries",
                status_code=408,
                error_type="APITimeoutError",
                original_error=e
            )

        except APIConnectionError as e:
            logger.warning(f"API connection error (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.info(f"Waiting {delay:.2f} seconds before retry due to connection error...")
                await asyncio.sleep(delay)
                continue
            raise AzureAPIError(
                message="Connection to AI service failed after all retries",
                status_code=503,
                error_type="APIConnectionError",
                original_error=e
            )

        except APIError as e:
            logger.error(f"API error (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries and hasattr(e, 'status_code') and e.status_code >= 500:
                # Retry on server errors (5xx)
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.info(f"Waiting {delay:.2f} seconds before retry due to server error...")
                await asyncio.sleep(delay)
                continue
            raise AzureAPIError(
                message=f"Azure API error: {str(e)}",
                status_code=getattr(e, 'status_code', 500),
                error_type="APIError",
                original_error=e
            )

        except Exception as e:
            logger.error(f"Unexpected error generating answer (attempt {attempt + 1}): {str(e)}", exc_info=True)
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.info(f"Waiting {delay:.2f} seconds before retry due to unexpected error...")
                await asyncio.sleep(delay)
                continue
            raise LLMGenerationError(
                reason=f"Unexpected error during generation: {str(e)}",
                question=question,
                attempts=max_retries + 1,
                original_error=e
            )

    # This should never be reached, but included for completeness
    raise LLMGenerationError(
        reason="Failed to generate answer after all retries",
        question=question,
        attempts=max_retries + 1
    )


def extract_citations(answer: str, chunks: list[dict]) -> list[Citation]:
    """
    Extract citations from LLM answer and link them to source documents.
    
    Args:
        answer: The LLM's response text containing potential citations
        chunks: List of source document chunks used to generate the answer
        
    Returns:
        List of Citation objects for valid citations found
        
    Raises:
        CitationParsingError: When citation extraction fails completely
    """
    logger = get_rag_logger(__name__)

    if not answer or not answer.strip():
        return []

    if not chunks:
        return []

    # Task 4.1: Regex pattern for parsing citations [filename p. X] or [filename]
    # This pattern captures:
    # - Optional opening bracket whitespace
    # - Filename (non-greedy, stopping at p. or closing bracket)
    # - Optional page number after "p." or "page"
    # - Optional closing bracket whitespace
    citation_pattern = r'\[\s*([^[\]]+?)(?:\s+(?:p\.?\s*|page\s*)(\d+))?\s*\]'

    citations = []
    matches = re.finditer(citation_pattern, answer, re.IGNORECASE)

    for match in matches:
        filename_raw = match.group(1).strip()
        page_raw = match.group(2)

        # Parse page number
        page_num = None
        if page_raw:
            try:
                page_num = int(page_raw.strip())
            except (ValueError, AttributeError):
                logger.warning(f"Invalid page number in citation: {page_raw}")
                continue

        # Task 4.2: Map citation to source document
        matched_chunk = _find_matching_chunk(filename_raw, page_num, chunks, logger)

        if matched_chunk:
            # Task 4.3: Construct Citation object
            try:
                citation = _create_citation_object(
                    matched_chunk, filename_raw, page_num, logger
                )
                if citation:
                    citations.append(citation)
            except Exception as e:
                # Task 4.4: Handle construction errors gracefully
                logger.warning(f"Failed to create citation object: {e}")
                continue
        else:
            # Task 4.4: Handle malformed/unmappable citations
            logger.warning(
                f"Citation [{filename_raw}" +
                (f" p. {page_num}" if page_num else "") +
                "] could not be mapped to source documents"
            )

    # Remove duplicate citations based on document_id and page
    unique_citations = []
    seen_citations = set()

    for citation in citations:
        citation_key = (citation.document_id, citation.page)
        if citation_key not in seen_citations:
            unique_citations.append(citation)
            seen_citations.add(citation_key)

    return unique_citations


def _find_matching_chunk(
    filename: str, page_num: int | None, chunks: list[dict], logger
) -> dict | None:
    """
    Find the chunk that matches the citation filename and page number.
    
    Args:
        filename: The filename from the citation
        page_num: Optional page number from the citation
        chunks: List of source chunks to search
        logger: Logger instance
        
    Returns:
        Matching chunk dict or None
    """
    # Normalize filename for comparison (remove common extensions, lowercase)
    normalized_filename = filename.lower().strip()

    # Try exact filename matches first
    for chunk in chunks:
        metadata = chunk.get('metadata', {})
        chunk_filename = metadata.get('filename') or metadata.get('source', '')

        if not chunk_filename:
            continue

        # Normalize chunk filename for comparison
        normalized_chunk_filename = chunk_filename.lower().strip()

        # Check for exact match first
        if normalized_filename == normalized_chunk_filename:

            # If page number specified, verify it matches
            if page_num is not None:
                chunk_page = metadata.get('page') or metadata.get('page_number')
                if chunk_page is not None:
                    try:
                        if int(chunk_page) == page_num:
                            return chunk
                    except (ValueError, TypeError):
                        continue
                else:
                    # No page info in chunk, but citation has page - still consider it a match
                    return chunk
            else:
                # No page number in citation, filename match is sufficient
                return chunk

    # Try partial filename matches (without extensions)
    filename_base = normalized_filename.split('.')[0] if '.' in normalized_filename else normalized_filename

    for chunk in chunks:
        metadata = chunk.get('metadata', {})
        chunk_filename = metadata.get('filename') or metadata.get('source', '')

        if not chunk_filename:
            continue

        chunk_filename_base = chunk_filename.lower().split('.')[0] if '.' in chunk_filename else chunk_filename.lower()

        if filename_base == chunk_filename_base:
            # If page number specified, verify it matches
            if page_num is not None:
                chunk_page = metadata.get('page') or metadata.get('page_number')
                if chunk_page is not None:
                    try:
                        if int(chunk_page) == page_num:
                            return chunk
                    except (ValueError, TypeError):
                        continue
            else:
                return chunk

    logger.warning(f"No matching chunk found for citation: {filename}" +
                  (f" p. {page_num}" if page_num else ""))
    return None


def _create_citation_object(
    chunk: dict, filename: str, page_num: int | None, logger
) -> Citation | None:
    """
    Create a Citation object from a matched chunk.
    
    Args:
        chunk: The matched source chunk
        filename: Original filename from citation
        page_num: Optional page number from citation
        logger: Logger instance
        
    Returns:
        Citation object or None if creation fails
    """
    try:
        metadata = chunk.get('metadata', {})
        content = chunk.get('content', '').strip()

        # Extract document information
        document_name = metadata.get('filename') or metadata.get('source') or filename
        chunk_page = metadata.get('page') or metadata.get('page_number')

        # Use page from citation if available, otherwise use chunk page
        final_page = page_num if page_num is not None else chunk_page
        if final_page is not None:
            try:
                final_page = int(final_page)
            except (ValueError, TypeError):
                final_page = None

        # Generate document_id (could be enhanced to use actual document IDs from database)
        document_id = f"{document_name}:{final_page}" if final_page else document_name

        # Create snippet from chunk content (limit length for display)
        max_snippet_length = 200
        snippet = content[:max_snippet_length]
        if len(content) > max_snippet_length:
            snippet += "..."

        # Validate required fields
        if not document_name or not snippet:
            logger.warning(f"Missing required fields for citation: document_name='{document_name}', snippet length={len(snippet)}")
            return None

        return Citation(
            document_id=document_id,
            document_name=document_name,
            page=final_page,
            snippet=snippet
        )

    except Exception as e:
        logger.error(f"Error creating citation object: {e}")
        return None
