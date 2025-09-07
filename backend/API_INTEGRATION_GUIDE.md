# Chat With Docs API Integration Guide

## Overview
This API provides a RAG (Retrieval-Augmented Generation) powered document Q&A system with document processing, storage, and intelligent querying capabilities.

**Base URL**: Configurable (see deployment section)
- Development: `http://127.0.0.1:8000`
- Production: `https://your-domain.com` or `https://your-server-ip:port`

**Authentication**: Clerk JWT Bearer tokens  
**Content Type**: `application/json` (except file uploads)

## Authentication

The API uses Clerk authentication with JWT tokens. All protected endpoints require a Bearer token in the Authorization header.

### Authentication Header
```
Authorization: Bearer <clerk_jwt_token>
```

### Endpoints

#### Basic Auth Endpoints (Legacy - Not Fully Implemented)
- **POST** `/api/auth/signup` - User registration (TODO: Implementation needed)
- **POST** `/api/auth/token` - Login and token generation (TODO: Implementation needed)

### User Context
Authenticated requests provide user context through `get_current_user` dependency, returning:
```json
{
  "id": "user_id_from_clerk",
  "email": "user@example.com",
  // Additional Clerk user data
}
```

## Document Management

### Upload Document
**POST** `/api/documents/upload`

Uploads, processes, and stores a document in the system.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Form data with file field

**Response:**
```json
{
  "id": 123,
  "filename": "example.pdf",
  "user_id": "clerk_user_id",
  "status": "processed",
  "storage_key": "documents/123/example.pdf",
  "created_at": "2024-01-01T00:00:00",
  "chunk_count": 42
}
```

**Process:**
1. Creates document record in database
2. Uploads file to Supabase storage
3. Processes document (extracts text, creates chunks)
4. Stores chunks in vector database
5. Updates document status to "processed"

### Process Document (Without Storage)
**POST** `/api/documents/process`

Processes a document and returns chunks without storing them.

**Request:**
- Content-Type: `multipart/form-data`
- Body: Form data with file field

**Response:**
```json
{
  "success": true,
  "message": "Successfully processed document: example.pdf",
  "document_metadata": {
    "filename": "example.pdf",
    "file_type": "pdf",
    "total_pages": 10,
    "total_chars": 5000,
    "total_tokens": 1200,
    "sections": ["Introduction", "Chapter 1"]
  },
  "chunks": [
    {
      "text": "Document content...",
      "chunk_index": 0,
      "document_filename": "example.pdf",
      "page_number": 1,
      "section_title": "Introduction",
      "start_char": 0,
      "end_char": 500,
      "char_count": 500,
      "metadata": {}
    }
  ],
  "processing_stats": {
    "document": {"processing_time": 0.5},
    "parsing": {"processing_time": 1.2},
    "chunking": {"chunk_count": 42},
    "processing": {"total_time": 2.1}
  }
}
```

### List Documents
**GET** `/api/documents/`

Returns all documents for the authenticated user.

**Response:**
```json
[
  {
    "id": 123,
    "filename": "example.pdf",
    "user_id": "clerk_user_id",
    "status": "processed",
    "storage_key": "documents/123/example.pdf",
    "created_at": "2024-01-01T00:00:00",
    "chunk_count": 42
  }
]
```

### Delete Document
**DELETE** `/api/documents/{document_id}`

Deletes a document and all its chunks.

**Response:**
```json
{
  "message": "Document 'example.pdf' deleted successfully"
}
```

### Get Processing Configuration
**GET** `/api/documents/processing-config`

Returns current document processing settings.

**Response:**
```json
{
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "min_chunk_size": 100,
  "max_file_size_mb": 10.0,
  "supported_formats": [".pdf", ".docx", ".txt", ".md"]
}
```

### Advanced Document Operations

#### Get Related Chunks
**GET** `/api/documents/chunks/{chunk_id}/related`

Gets chunks related through hierarchical relationships.

**Query Parameters:**
- `relation_types` (optional): Comma-separated list of relationship types
- `max_distance` (optional): Maximum relationship distance (default: 2)

**Response:**
```json
[
  {
    "text": "Related chunk content...",
    "chunk_index": 5,
    "page_number": 2,
    "start_char": 1000,
    "end_char": 1500,
    "chunk_type": "paragraph",
    "hierarchical_level": 2,
    "quality_score": 0.85,
    "chunk_references": ["chunk_1", "chunk_3"]
  }
]
```

#### Store Document Hierarchy
**POST** `/api/documents/hierarchy/store`

Stores document hierarchy information for enhanced retrieval.

**Request:**
```json
{
  "document_id": "123",
  "hierarchy": {
    "sections": [],
    "relationships": []
  }
}
```

**Response:**
```json
{
  "hierarchy_id": "hier_456",
  "total_elements": 25,
  "status": "stored"
}
```

## Chat/RAG System

### Query Documents
**POST** `/api/chat/query`

Performs a RAG query against uploaded documents.

**Request:**
```json
{
  "question": "What is the main topic of the document?"
}
```

**Response:**
```json
{
  "answer": "The main topic of the document is...",
  "confidence": 0.89
}
```

**Process:**
1. Embeds the question
2. Searches relevant chunks in vector database
3. Uses LLM to generate answer based on retrieved context
4. Returns answer with confidence score

## Error Handling

### Standard Error Responses
```json
{
  "detail": "Error description"
}
```

### Common HTTP Status Codes
- `200` - Success
- `400` - Bad Request (invalid file format, missing data)
- `401` - Unauthorized (invalid/missing token)
- `404` - Not Found (document/resource not found)
- `500` - Internal Server Error

### Authentication Errors
```json
{
  "detail": "Bearer token required",
  "headers": {"WWW-Authenticate": "Bearer"}
}
```

## Data Models

### Key Schemas

#### Document
```typescript
{
  id: number
  filename: string
  user_id: string
  status: "processing" | "processed" | "failed"
  storage_key?: string
  created_at: string
  chunk_count?: number
}
```

#### DocumentChunk
```typescript
{
  text: string
  chunk_index: number
  document_filename: string
  page_number?: number
  section_title?: string
  start_char: number
  end_char: number
  char_count: number
  metadata: Record<string, any>
}
```

#### Query/Answer
```typescript
// Request
{
  question: string
}

// Response
{
  answer: string
  confidence: number
}
```

## Integration Examples

### Frontend Integration (Next.js + Clerk)

The frontend is built with:
- **Next.js 14** with TypeScript
- **Clerk** for authentication (`@clerk/nextjs`)
- **Tailwind CSS** for styling
- **Lucide React** for icons

#### Environment Configuration
```env
# Frontend .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_your_key
CLERK_SECRET_KEY=sk_test_your_key
```

#### API Client Implementation
The frontend already includes a configured API client (`src/lib/api.ts`):

```typescript
// src/lib/api.ts
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export const api = {
  uploadDocument: async (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    
    const response = await fetch(`${API_URL}/api/documents/upload`, {
      method: 'POST',
      body: formData, // No Content-Type header for FormData
    })
    
    return response.json()
  },
  
  getDocuments: async () => {
    return apiRequest('/api/documents')
  },
  
  sendQuery: async (question: string) => {
    return apiRequest('/api/chat/query', {
      method: 'POST',
      body: JSON.stringify({ question }),
    })
  }
}
```

#### Clerk Authentication Integration
The frontend uses a dedicated upload service with Clerk authentication (`src/services/uploadApi.ts`):

```typescript
// src/services/uploadApi.ts
import { useAuth } from '@clerk/nextjs'

export function useUploadApi() {
  const { getToken, isSignedIn } = useAuth()

  const uploadDocument = async (file: File, onProgress?: (progress: number) => void) => {
    if (!isSignedIn) {
      throw new Error('You must be signed in to upload documents')
    }

    const token = await getToken()
    const xhr = new XMLHttpRequest()

    // Add auth header
    if (token) {
      xhr.setRequestHeader('Authorization', `Bearer ${token}`)
    }

    // Handle progress updates
    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable && onProgress) {
        const progress = Math.round((event.loaded / event.total) * 100)
        onProgress(progress)
      }
    })

    // Send request
    const formData = new FormData()
    formData.append('file', file)
    xhr.send(formData)
  }
}
```

#### Document Management Hook
The frontend includes a comprehensive document management system (`src/hooks/useDocuments.ts`):

```typescript
export const useDocuments = () => {
  const [documents, setDocuments] = useState<Document[]>([])
  const [isLoading, setIsLoading] = useState(false)

  // Fetches documents from /api/documents/
  const fetchDocuments = useCallback(async () => {
    // Implementation with proper error handling
  }, [])

  return { documents, isLoading, refetch: fetchDocuments }
}

export const useDocumentUpload = () => {
  const { uploadDocument, isSignedIn } = useUploadApi()
  // Upload with progress tracking
}
```

#### Component Structure
Key components implemented:
- `DocumentUpload` - Drag & drop file upload with validation
- `ChatInterface` - Message-based chat with suggested questions
- `DocumentLibrary` - List view of uploaded documents
- `MessageBubble` - Chat message rendering

#### Type Definitions
Frontend types are defined in `src/types/index.ts`:

```typescript
export interface Document {
  id: number;  // Matches backend integer ID
  filename: string;
  user_id: string;
  status: 'processing' | 'ready' | 'failed';
  created_at: string;
  file_size: number;
  file_type: string;
  pages?: number;
  upload_progress?: number;
}

export interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}
```

#### File Upload Validation
The frontend validates files before upload:
- **Supported types**: PDF, DOCX, TXT
- **Size limit**: 10MB
- **MIME type checking**: `application/pdf`, `application/vnd.openxmlformats-officedocument.wordprocessingml.document`, `text/plain`

#### Ready-to-Use Integration Examples

##### Complete Upload Flow
```typescript
import { useDocumentUpload } from '@/hooks/useDocuments'

function UploadComponent() {
  const { uploadDocument, isUploading, uploadProgress } = useDocumentUpload()

  const handleUpload = async (file: File) => {
    try {
      const document = await uploadDocument(file)
      console.log('Upload successful:', document)
    } catch (error) {
      console.error('Upload failed:', error)
    }
  }

  return (
    <DocumentUpload
      onUpload={handleUpload}
      isUploading={isUploading}
      uploadProgress={uploadProgress}
    />
  )
}
```

##### Chat Integration
```typescript
import { ChatInterface } from '@/components/ChatInterface'

function ChatComponent() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)

  const handleSendMessage = async (question: string) => {
    setIsLoading(true)
    try {
      const response = await api.sendQuery(question)
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        type: 'assistant',
        content: response.answer,
        timestamp: new Date()
      }])
    } catch (error) {
      console.error('Query failed:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <ChatInterface
      messages={messages}
      isLoading={isLoading}
      onSendMessage={handleSendMessage}
      hasDocuments={documents.length > 0}
    />
  )
}
```

## Environment Configuration

### Environment-Based Configuration

The API supports flexible deployment through environment-based configuration. Create appropriate `.env` files for each environment:

#### Development Environment (`.env.development`)
```env
# Database
DATABASE_URL=postgresql://localhost/chatwithdocs

# Server Configuration
HOST=127.0.0.1
PORT=8000
DEBUG=true

# CORS - Allow local development
ALLOWED_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]

# Azure OpenAI (Required)
AZURE_OPENAI_API_KEY=your_dev_key
AZURE_OPENAI_ENDPOINT=https://your-dev-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_API_VERSION=2024-02-01

# Clerk Authentication
CLERK_SECRET_KEY=your_dev_clerk_secret
CLERK_PUBLISHABLE_KEY=your_dev_clerk_publishable_key

# Supabase Storage
SUPABASE_URL=your_dev_supabase_url
SUPABASE_KEY=your_dev_supabase_key
SUPABASE_BUCKET_NAME=documents-dev
```

#### Production Environment (`.env.production`)
```env
# Database - Use production database
DATABASE_URL=postgresql://prod-user:password@prod-host:5432/chatwithdocs_prod

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# CORS - Configure for your frontend domain
ALLOWED_ORIGINS=["https://your-frontend-domain.com", "https://www.your-frontend-domain.com"]

# Security
SECRET_KEY=your-strong-production-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Azure OpenAI (Production)
AZURE_OPENAI_API_KEY=your_prod_key
AZURE_OPENAI_ENDPOINT=https://your-prod-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-prod
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small-prod
AZURE_OPENAI_API_VERSION=2024-02-01

# Clerk Authentication (Production)
CLERK_SECRET_KEY=your_prod_clerk_secret
CLERK_PUBLISHABLE_KEY=your_prod_clerk_publishable_key

# Supabase Storage (Production)
SUPABASE_URL=your_prod_supabase_url
SUPABASE_KEY=your_prod_supabase_key
SUPABASE_BUCKET_NAME=documents-prod

# File Processing
MAX_FILE_SIZE_MB=50
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Performance
USE_OPENAI_EMBEDDINGS=true
```

#### Staging Environment (`.env.staging`)
```env
# Similar to production but with staging resources
DATABASE_URL=postgresql://staging-user:password@staging-host:5432/chatwithdocs_staging
ALLOWED_ORIGINS=["https://staging.your-frontend-domain.com"]
# ... other staging-specific configs
```

## Frontend-Backend Integration Status

### ⚠️ Integration Issues Found

#### 1. API Authentication Mismatch
- **Frontend**: Uses Clerk authentication with `useAuth` hook
- **Backend**: Requires Bearer token in all protected endpoints
- **Status**: ❌ `src/lib/api.ts` missing authentication headers

#### 2. Document Status Field Mismatch  
- **Frontend**: Expects `status: 'processing' | 'ready' | 'failed'`
- **Backend**: Returns `status: 'processing' | 'processed' | 'failed'`
- **Issue**: Frontend expects 'ready' but backend returns 'processed'

#### 3. Missing API Integration
- **Frontend**: `useDocuments` hook uses mock data
- **Backend**: Real endpoints available at `/api/documents/`
- **Status**: ❌ Frontend not connected to real API

#### 4. Environment Configuration
- **Frontend**: Uses `NEXT_PUBLIC_API_URL` (currently missing auth)
- **Backend**: Configurable via `ALLOWED_ORIGINS`
- **Required**: Update frontend `.env.local` with proper API URL

### 🔧 Required Fixes

#### Update Frontend API Client (`src/lib/api.ts`)
```typescript
import { auth } from '@clerk/nextjs'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const getAuthHeaders = async () => {
  const { getToken } = auth()
  const token = await getToken()
  return {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  }
}

export const api = {
  getDocuments: async () => {
    const response = await fetch(`${API_URL}/api/documents/`, {
      headers: await getAuthHeaders()
    })
    return response.json()
  },
  
  deleteDocument: async (id: number) => {
    const response = await fetch(`${API_URL}/api/documents/${id}`, {
      method: 'DELETE',
      headers: await getAuthHeaders()
    })
    return response.json()
  }
}
```

#### Fix Document Status Types (`src/types/index.ts`)
```typescript
export interface Document {
  id: number;
  filename: string;
  user_id: string;
  status: 'processing' | 'processed' | 'failed'; // Changed from 'ready' to 'processed'
  storage_key?: string;
  created_at: string;
  chunk_count?: number;
}
```

#### Update useDocuments Hook
Replace mock data with real API calls:
```typescript
const fetchDocuments = useCallback(async () => {
  setIsLoading(true)
  try {
    const documents = await api.getDocuments()
    setDocuments(documents)
  } catch (error) {
    setError('Failed to fetch documents')
  } finally {
    setIsLoading(false)
  }
}, [])
```

## Rate Limits & Constraints

- Maximum file size: 10MB (configurable via `MAX_FILE_SIZE_MB`)
- Supported formats: PDF, DOCX, TXT, MD
- Chunk size: 1000 characters with 200 character overlap
- CORS: Configurable via `ALLOWED_ORIGINS` (currently `["http://localhost:3000"]`)

## Development Notes

### Starting the Server
```bash
# Using uv (recommended)
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Access points
# API: http://127.0.0.1:8000
# Swagger Docs: http://127.0.0.1:8000/docs
# ReDoc: http://127.0.0.1:8000/redoc
```

### Testing Endpoints
```bash
# Health check
curl http://127.0.0.1:8000

# Expected response:
# {"message":"Chat With Docs API","version":"1.0.0"}
```

This documentation provides comprehensive information for integrating with the Chat With Docs API, including authentication, all endpoints, data models, error handling, and practical examples.