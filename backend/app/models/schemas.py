from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, EmailStr


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
    id: str
    filename: str
    user_id: int
    status: str
    storage_key: Optional[str] = None
    created_at: datetime


class DocumentMetadata(BaseModel):
    filename: str
    file_type: str
    total_pages: Optional[int] = None
    total_chars: int
    total_tokens: int = 0
    sections: List[str] = []


class DocumentChunk(BaseModel):
    text: str
    chunk_index: int
    document_filename: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    start_char: int
    end_char: int
    char_count: int
    metadata: Dict = {}


class ProcessingStats(BaseModel):
    document: Dict
    parsing: Dict
    chunking: Dict
    processing: Dict


class DocumentProcessingResult(BaseModel):
    success: bool
    message: str
    document_metadata: DocumentMetadata
    chunks: List[DocumentChunk]
    processing_stats: ProcessingStats


class ProcessingConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int
    max_file_size_mb: float
    supported_formats: List[str]


# Chat schemas
class Query(BaseModel):
    question: str


class Citation(BaseModel):
    document_id: str
    document_name: str
    page: Optional[int]
    snippet: str


class Answer(BaseModel):
    answer: str
    citations: List[Citation]
    confidence: float
