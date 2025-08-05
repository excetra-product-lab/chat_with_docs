from fastapi import APIRouter, Depends

from app.api.dependencies import get_current_user
from app.core.rag_pipeline import answer_question
from app.models.schemas import Answer, Query

router = APIRouter()


@router.post("/query", response_model=Answer)
async def chat_query(query: Query, current_user: dict = Depends(get_current_user)):
    # Process query through RAG pipeline with optional parameters
    answer = await answer_question(
        question=query.question,
        user_id=current_user["id"],
        document_ids=query.document_ids,
        limit=query.limit,
    )
    return answer
