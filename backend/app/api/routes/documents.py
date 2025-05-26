from typing import List

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.dependencies import get_current_user
from app.models.schemas import Document

router = APIRouter()


@router.post("/upload", response_model=Document)
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    # TODO: Save file to storage
    # TODO: Process document (parse, chunk, embed)
    # TODO: Store in database
    return {
        "id": 1,
        "filename": file.filename,
        "user_id": current_user["id"],
        "status": "processing",
        "created_at": "2024-01-01T00:00:00",
    }


@router.get("/", response_model=List[Document])
async def list_documents(current_user: dict = Depends(get_current_user)):
    # TODO: Fetch user's documents from database
    return []


@router.delete("/{document_id}")
async def delete_document(document_id: int, current_user: dict = Depends(get_current_user)):
    # TODO: Verify ownership and delete document
    return {"message": "Document deleted successfully"}
