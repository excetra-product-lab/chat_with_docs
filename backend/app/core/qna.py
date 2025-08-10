from app.core.vectorstore import similarity_search
from app.models.schemas import Answer


async def answer_question(question: str, user_id: str) -> Answer:
    """
    Process a question through the RAG pipeline
    """
    # TODO: Search for relevant chunks
    relevant_chunks = await similarity_search(question, user_id)

    # TODO: Build context from chunks
    context = build_context(relevant_chunks)

    # TODO: Generate answer using LLM
    answer_text = await generate_answer(question, context)

    # Citations removed

    return Answer(answer=answer_text, confidence=0.95)


def build_context(chunks: list[dict]) -> str:
    """Combine chunks into context"""
    # TODO: Format chunks into context
    return "Context from documents"


async def generate_answer(question: str, context: str) -> str:
    """Generate answer using Azure OpenAI"""
    # TODO: Call Azure OpenAI chat API
    return "This is a generated answer based on the documents."


# Citation extraction function removed
