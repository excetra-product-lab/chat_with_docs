from datetime import datetime

from pydantic import BaseModel


# Authentication schemas
class UserCreate(BaseModel):
    email: str
    password: str


class User(BaseModel):
    id: str
    email: str
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str


# Document schemas
class DocumentCreate(BaseModel):
    filename: str


class Document(BaseModel):
    id: int  # Changed from str to int to match database model
    filename: str
    user_id: str
    status: str
    storage_key: str | None = None
    created_at: datetime
    chunk_count: int | None = None


class DocumentMetadata(BaseModel):
    filename: str
    file_type: str
    total_pages: int | None = None
    total_chars: int
    total_tokens: int = 0
    sections: list[str] = []


class DocumentChunk(BaseModel):
    text: str
    chunk_index: int
    document_filename: str
    page_number: int | None = None
    section_title: str | None = None
    start_char: int
    end_char: int
    char_count: int
    metadata: dict = {}


class ProcessingStats(BaseModel):
    document: dict
    parsing: dict
    chunking: dict
    processing: dict


class DocumentProcessingResult(BaseModel):
    success: bool
    message: str
    document_metadata: DocumentMetadata
    chunks: list[DocumentChunk]
    processing_stats: ProcessingStats


class ProcessingConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int
    max_file_size_mb: float
    supported_formats: list[str]


# Chat schemas
class Query(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str
    confidence: float
