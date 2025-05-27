#!/bin/bash
# Backend initialization script

# Create backend directory structure
mkdir -p backend/app/api/routes
mkdir -p backend/app/core
mkdir -p backend/app/models
mkdir -p backend/app/utils
mkdir -p backend/tests

# Create main.py
cat > backend/app/main.py << 'EOF'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import auth, documents, chat
from app.core.settings import settings

app = FastAPI(
    title="Chat With Docs API",
    description="RAG-powered document Q&A system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])

@app.get("/")
async def read_root():
    return {"message": "Chat With Docs API", "version": "1.0.0"}
EOF

# Create settings.py
cat > backend/app/core/settings.py << 'EOF'
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost/chatwithdocs"

    # Azure OpenAI
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_DEPLOYMENT_NAME: str = "gpt-4o"
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = "text-embedding-ada-002"

    # Authentication
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]

    # Vector store
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    class Config:
        env_file = ".env"

settings = Settings()
EOF

# Create auth.py route
cat > backend/app/api/routes/auth.py << 'EOF'
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from app.models.schemas import Token, UserCreate, User
from app.utils.security import verify_password, get_password_hash, create_access_token
from datetime import timedelta
from app.core.settings import settings

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

@router.post("/signup", response_model=User)
async def signup(user: UserCreate):
    # TODO: Check if user exists in database
    # TODO: Create user with hashed password
    return {"id": 1, "email": user.email, "created_at": "2024-01-01T00:00:00"}

@router.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # TODO: Authenticate user against database
    # TODO: Create and return JWT token
    access_token = create_access_token(
        data={"sub": form_data.username},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}
EOF

# Create documents.py route
cat > backend/app/api/routes/documents.py << 'EOF'
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import List
from app.models.schemas import Document, DocumentCreate
from app.api.dependencies import get_current_user
from app.core.ingestion import process_document

router = APIRouter()

@router.post("/upload", response_model=Document)
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    # TODO: Save file to storage
    # TODO: Process document (parse, chunk, embed)
    # TODO: Store in database
    return {
        "id": 1,
        "filename": file.filename,
        "user_id": current_user["id"],
        "status": "processing",
        "created_at": "2024-01-01T00:00:00"
    }

@router.get("/", response_model=List[Document])
async def list_documents(current_user: dict = Depends(get_current_user)):
    # TODO: Fetch user's documents from database
    return []

@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    current_user: dict = Depends(get_current_user)
):
    # TODO: Verify ownership and delete document
    return {"message": "Document deleted successfully"}
EOF

# Create chat.py route
cat > backend/app/api/routes/chat.py << 'EOF'
from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import Query, Answer
from app.api.dependencies import get_current_user
from app.core.qna import answer_question

router = APIRouter()

@router.post("/query", response_model=Answer)
async def chat_query(
    query: Query,
    current_user: dict = Depends(get_current_user)
):
    # TODO: Process query through RAG pipeline
    answer = await answer_question(
        question=query.question,
        user_id=current_user["id"]
    )
    return answer
EOF

# Create dependencies.py
cat > backend/app/api/dependencies.py << 'EOF'
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from app.utils.security import decode_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    # TODO: Fetch user from database
    return {"id": 1, "email": payload.get("sub")}
EOF

# Create ingestion.py
cat > backend/app/core/ingestion.py << 'EOF'
from typing import List, Dict
import asyncio
from app.core.settings import settings

async def process_document(file_path: str, user_id: int) -> Dict:
    """
    Process a document: parse, chunk, and generate embeddings
    """
    # TODO: Parse document (PDF, DOCX, etc.)
    text = await parse_document(file_path)

    # TODO: Split into chunks
    chunks = chunk_text(text)

    # TODO: Generate embeddings
    embeddings = await generate_embeddings(chunks)

    # TODO: Store in vector database
    await store_embeddings(chunks, embeddings, user_id)

    return {"status": "success", "chunks": len(chunks)}

async def parse_document(file_path: str) -> str:
    """Extract text from document"""
    # TODO: Implement document parsing
    return "Document content"

def chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks"""
    # TODO: Implement text chunking
    return [text]

async def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    """Generate embeddings using Azure OpenAI"""
    # TODO: Call Azure OpenAI embedding API
    return [[0.0] * 1536 for _ in chunks]

async def store_embeddings(chunks: List[str], embeddings: List[List[float]], user_id: int):
    """Store chunks and embeddings in pgvector"""
    # TODO: Store in database
    pass
EOF

# Create qna.py
cat > backend/app/core/qna.py << 'EOF'
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
EOF

# Create vectorstore.py
cat > backend/app/core/vectorstore.py << 'EOF'
from typing import List, Dict
from app.core.settings import settings

async def similarity_search(query: str, user_id: int, k: int = 5) -> List[Dict]:
    """
    Perform similarity search in pgvector
    """
    # TODO: Generate embedding for query
    query_embedding = await generate_query_embedding(query)

    # TODO: Search in pgvector
    results = await search_vectors(query_embedding, user_id, k)

    return results

async def generate_query_embedding(query: str) -> List[float]:
    """Generate embedding for search query"""
    # TODO: Call Azure OpenAI embedding API
    return [0.0] * 1536

async def search_vectors(embedding: List[float], user_id: int, k: int) -> List[Dict]:
    """Search for similar vectors in database"""
    # TODO: Execute pgvector similarity search
    return []
EOF

# Create schemas.py
cat > backend/app/models/schemas.py << 'EOF'
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
EOF

# Create database.py
cat > backend/app/models/database.py << 'EOF'
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, VECTOR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    documents = relationship("Document", back_populates="user")

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String, default="processing")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document")

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    content = Column(Text)
    embedding = Column(VECTOR(1536))
    page = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")
EOF

# Create security.py
cat > backend/app/utils/security.py << 'EOF'
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.core.settings import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        return None
EOF

# Create requirements.txt
cat > backend/requirements.txt << 'EOF'
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
pydantic-settings==2.1.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
sqlalchemy==2.0.25
psycopg2-binary==2.9.9
pgvector==0.2.4
openai==1.9.0
langchain==0.1.0
langchain-openai==0.0.3
pypdf==3.17.4
python-docx==1.1.0
pytest==7.4.4
httpx==0.26.0
EOF

# Create Dockerfile
cat > backend/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app ./app

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Create .env.example
cat > backend/.env.example << 'EOF'
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/chatwithdocs

# Azure OpenAI
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Authentication
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
ALLOWED_ORIGINS=["http://localhost:3000"]
EOF

# Create test files
cat > backend/tests/test_ingestion.py << 'EOF'
import pytest
from app.core.ingestion import chunk_text

def test_chunk_text():
    text = "This is a test document with some content."
    chunks = chunk_text(text)
    assert len(chunks) > 0
    assert chunks[0] == text  # Simple test for now
EOF

cat > backend/tests/test_qna.py << 'EOF'
import pytest
from app.core.qna import build_context

def test_build_context():
    chunks = [{"content": "Test chunk 1"}, {"content": "Test chunk 2"}]
    context = build_context(chunks)
    assert isinstance(context, str)
    assert len(context) > 0
EOF

cat > backend/tests/test_api.py << 'EOF'
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Chat With Docs API"

def test_signup():
    response = client.post(
        "/api/auth/signup",
        json={"email": "test@example.com", "password": "testpassword"}
    )
    assert response.status_code == 200
    assert response.json()["email"] == "test@example.com"
EOF

# Create backend README
cat > backend/README.md << 'EOF'
# Chat With Docs Backend

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the application:
```bash
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000
API documentation at http://localhost:8000/docs

## Testing

Run tests with:
```bash
pytest
```

## Deployment

Build and run with Docker:
```bash
docker build -t chat-with-docs-backend .
docker run -p 8000:8000 --env-file .env chat-with-docs-backend
```
EOF

echo "Backend structure created successfully!"
