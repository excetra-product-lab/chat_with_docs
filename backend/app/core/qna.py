from typing import List, Dict
from app.core.vectorstore import similarity_search
from app.models.schemas import Answer, Citation

async def answer_question(question: str, user_id: int) -> Answer:
    """
    Process a question through the RAG pipeline
    """
    # TODO: Search for relevant chunks
    relevant_chunks = await similarity_search(question, user_id)
    
    # TODO: Build context from chunks
    context = build_context(relevant_chunks)
    
    # TODO: Generate answer using LLM
    answer_text = await generate_answer(question, context)
    
    # TODO: Extract citations
    citations = extract_citations(answer_text, relevant_chunks)
    
    return Answer(
        answer=answer_text,
        citations=citations,
        confidence=0.95
    )

def build_context(chunks: List[Dict]) -> str:
    """Build context from retrieved chunks"""
    # TODO: Format chunks into context
    return "Context from documents"

async def generate_answer(question: str, context: str) -> str:
    """Generate answer using Azure OpenAI"""
    # TODO: Call Azure OpenAI chat API
    return "This is a generated answer based on the documents."

def extract_citations(answer: str, chunks: List[Dict]) -> List[Citation]:
    """Extract citations from answer"""
    # TODO: Parse citations from answer
    return []
