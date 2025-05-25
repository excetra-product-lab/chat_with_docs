from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime

# Authentication schemas
class UserCreate(BaseModel):
    email: EmailStr
    password: str

class User(BaseModel):
    id: int
    email: EmailStr
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str

# Document schemas
class DocumentCreate(BaseModel):
    filename: str

class Document(BaseModel):
    id: int
    filename: str
    user_id: int
    status: str
    created_at: datetime

# Chat schemas
class Query(BaseModel):
    question: str

class Citation(BaseModel):
    document_id: int
    document_name: str
    page: Optional[int]
    snippet: str

class Answer(BaseModel):
    answer: str
    citations: List[Citation]
    confidence: float
