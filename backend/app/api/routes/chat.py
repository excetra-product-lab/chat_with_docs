from fastapi import APIRouter, Depends

from app.api.dependencies import get_current_user
from app.core.langchain_config import langchain_config
from app.services.enhanced_vectorstore import EnhancedVectorStore, create_langchain_retriever
from app.models.schemas import Answer, Query

router = APIRouter()


@router.post("/query", response_model=Answer)
async def chat_query(query: Query, current_user: dict = Depends(get_current_user)):
    """Process query through existing RAG pipeline using enhanced vectorstore and LangChain."""

    # Use existing services built by fauzi
    vector_store = EnhancedVectorStore()
    llm = langchain_config.llm

    # Search documents using enhanced similarity search (existing method)
    search_results = await vector_store.enhanced_similarity_search(
        query=query.question,
        user_id=current_user["id"],  # Use string user_id as stored in database
        k=5,
        confidence_threshold=0.1
    )

    if not search_results:
        return Answer(answer="No relevant documents found for your question.", confidence=0.0)

    # Build context using existing pattern from test_local_rag_run.py
    context_parts = []
    for i, result in enumerate(search_results[:3]):  # Top 3 results
        context_part = f"[Source {i + 1}: {result.get('filename', 'Unknown')}, Page {result.get('page', 'N/A')}]"
        context_part += f"\n{result.get('content', '')}"
        context_parts.append(context_part)

    context = "\n\n".join(context_parts)

    # Generate answer using existing prompt pattern from test_local_rag_run.py
    prompt = f"""Based on the following document excerpts, provide a comprehensive answer to the question.

Question: {query.question}

Document Excerpts:
{context}

Please provide a detailed answer based on the information in the documents:"""

    # Use existing LLM
    if hasattr(llm, 'ainvoke'):
        response = await llm.ainvoke(prompt)
    else:
        response = llm.invoke(prompt)

    # Extract answer text based on response type
    if hasattr(response, 'content'):
        answer_text = str(response.content)
    else:
        answer_text = str(response)

    # Calculate confidence using existing pattern from test_local_rag_run.py
    avg_similarity = sum(r.get('similarity', 0) for r in search_results) / len(search_results)
    confidence = min(avg_similarity * 1.2, 0.95)  # Boost and cap at 95%

    return Answer(answer=answer_text, confidence=confidence)
