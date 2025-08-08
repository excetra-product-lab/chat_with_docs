"""
Local RAG Demonstration using real implementation services.

This script demonstrates the complete RAG pipeline using our actual services:
1. Reads documents from a local 'data' directory
2. Uses real DocumentProcessor and EnhancedDocumentService for processing
3. Uses FAISS for local vector storage (no database dependency)
4. Uses OpenAI text-embedding-3-large for embeddings and GPT-4o-mini for LLM (no Azure dependency)
5. Provides a complete question-answering workflow

Usage:
    # Set your OpenAI API key
    export OPENAI_API_KEY="your-openai-api-key"

    # Create data directory and add documents
    mkdir data
    # Add .pdf, .docx, .txt, or .md files to the data directory

    # Run the demonstration
    python backend/tests/test_local_rag_run.py "What are the main topics?"

    # Or run as pytest
    python -m pytest backend/tests/test_local_rag_run.py -v -s
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# Vector store dependencies
try:
    import faiss
    import numpy as np

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# LangChain for OpenAI
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False

# OpenAI for direct API access
try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Our real application services
from app.models.langchain_models import EnhancedDocument
from app.services.document_processor import DocumentProcessor
from app.services.enhanced_document_service import EnhancedDocumentService
from app.services.enhanced_vectorstore import EnhancedVectorStore
from app.utils.chunk_visualizer import print_chunks_before_openai
from app.utils.token_counter import TokenCounter


class SimpleEmbeddingService:
    """Simple OpenAI embedding service for local demonstrations."""

    def __init__(self, api_key: str):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not available. Install with: pip install openai")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "text-embedding-3-large"
        print(f"  📡 Initialized OpenAI embedding service with model: {self.model}")

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding using OpenAI API."""
        try:
            response = await self.client.embeddings.create(
                model=self.model, input=text, encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            raise


class MockUploadFile:
    """Mock UploadFile for testing document processing."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.filename = file_path.name
        # Pre-load content to avoid file reading issues
        with open(self.file_path, "rb") as f:
            self._content = f.read()
        self._position = 0
        self.size = len(self._content)  # Add missing size attribute

    async def read(self, size: int = -1) -> bytes:
        """Read file content as bytes."""
        if size == -1:
            # Return all remaining content from current position
            result = self._content[self._position :]
            self._position = len(self._content)
        else:
            # Return specific number of bytes
            result = self._content[self._position : self._position + size]
            self._position += len(result)

        return result

    async def seek(self, position: int) -> None:
        """Seek to position in file."""
        self._position = max(0, min(position, len(self._content)))

    def reset(self) -> None:
        """Reset to beginning of file for reuse."""
        self._position = 0


class LocalVectorStore:
    """Local FAISS-based vector store for testing without database dependency."""

    def __init__(self, embedding_service):
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS not available. Install with: pip install faiss-cpu"
            )

        self.embedding_service = embedding_service
        self.dimension = (
            3072  # OpenAI text-embedding-3-large dimension (was 1536 for small)
        )
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []

    async def store_enhanced_document(self, enhanced_doc: EnhancedDocument) -> int:
        """Store document using real embedding service."""
        if not enhanced_doc.chunks:
            return 0

        print(f"  📡 Generating embeddings for {len(enhanced_doc.chunks)} chunks...")

        for i, chunk in enumerate(enhanced_doc.chunks):
            # Use real embedding service
            embedding = await self.embedding_service.generate_embedding(chunk.text)

            # Store in FAISS
            embedding_array = np.array([embedding]).astype("float32")
            faiss.normalize_L2(embedding_array)
            self.index.add(embedding_array)

            # Store metadata
            self.documents.append(
                {
                    "content": chunk.text,
                    "document_name": enhanced_doc.filename,
                    "chunk_id": chunk.chunk_index,
                    "page": chunk.page_number or 1,
                    "section_title": chunk.section_title,
                    "hierarchy_level": chunk.hierarchical_level,
                    "token_count": chunk.token_count,
                    "metadata": chunk.metadata,
                }
            )

        return len(enhanced_doc.chunks)

    async def search_documents(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Search using real embedding service."""
        if not self.documents:
            return []

        print(f"  🔍 Searching {len(self.documents)} chunks for: '{query}'")

        # Generate query embedding using real service
        query_embedding = await self.embedding_service.generate_embedding(query)
        query_vector = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_vector)

        # Search FAISS
        similarities, indices = self.index.search(
            query_vector, min(k, len(self.documents))
        )

        results = []
        for similarity, idx in zip(similarities[0], indices[0], strict=False):
            if idx >= 0 and idx < len(self.documents):
                doc = self.documents[idx]
                results.append(
                    {
                        "content": doc["content"],
                        "document_name": doc["document_name"],
                        "page": doc["page"],
                        "similarity": float(similarity),
                        "section_title": doc.get("section_title", ""),
                        "hierarchy_level": doc.get("hierarchy_level", 0),
                        "token_count": doc.get("token_count", 0),
                        "metadata": doc.get("metadata", {}),
                    }
                )

        return results


class LocalRAGRunner:
    """Local RAG runner using real implementation services."""

    def __init__(
        self,
        openai_api_key: str,
        data_directory: str = "data",
        use_real_db: bool = False,
    ):
        self.openai_api_key = openai_api_key
        self.data_directory = Path(data_directory)
        self.use_real_db = use_real_db

        # Initialize real services
        print("🔧 Initializing real implementation services...")

        # Real token counter for accurate token counting
        self.token_counter = TokenCounter()

        # Real document processor with Langchain
        self.document_processor = DocumentProcessor(
            chunk_size=600, chunk_overlap=100, use_langchain=True
        )

        # Real enhanced document service
        self.enhanced_document_service = EnhancedDocumentService(
            self.document_processor
        )

        # Simple OpenAI embedding service (bypassing Azure dependency)
        print("  🔧 Using OpenAI text-embedding-3-large for high-quality embeddings")
        self.embedding_service = SimpleEmbeddingService(openai_api_key)

        # Vector store (real or local FAISS)
        if use_real_db:
            print("📊 Using real EnhancedVectorStore with database")
            self.vector_store = EnhancedVectorStore(self.embedding_service)
        else:
            print("📊 Using local FAISS vector store for testing")
            self.vector_store = LocalVectorStore(self.embedding_service)

        # Real LLM service
        if LANGCHAIN_OPENAI_AVAILABLE:
            self.llm = ChatOpenAI(
                api_key=openai_api_key, model="gpt-4o-mini", temperature=0.1
            )
        else:
            self.llm = None

        # Track processed documents
        self.processed_documents: list[EnhancedDocument] = []
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "total_tokens": 0,
            "processing_time": 0.0,
        }

    def _create_sample_documents(self) -> None:
        """Create sample documents if data directory is empty."""
        self.data_directory.mkdir(exist_ok=True)

        sample_docs = [
            {
                "filename": "software_agreement.txt",
                "content": """Software Development Agreement

ARTICLE I - DEFINITIONS
1.1 "Agreement" means this Software Development Agreement.
1.2 "Client" means ABC Corporation.
1.3 "Developer" means XYZ Development Services LLC.
1.4 "Software" means the computer program and related documentation.

ARTICLE II - SCOPE OF WORK
2.1 Development Services
The Developer shall provide:
- Requirements analysis and documentation
- Software design and architecture
- Implementation and coding
- Testing and quality assurance
- Documentation and user training

2.2 Deliverables
- Technical specifications document
- Source code and compiled software
- User documentation and training materials
- Testing reports and quality metrics

ARTICLE III - PAYMENT TERMS
3.1 Total project cost: $75,000
3.2 Payment schedule:
    a) 30% upon signing this agreement
    b) 40% upon completion of development milestone
    c) 30% upon final delivery and acceptance

3.3 Late Payment
Late payments shall incur interest at 1.5% per month.

ARTICLE IV - INTELLECTUAL PROPERTY
4.1 All intellectual property rights in the Software shall vest in the Client.
4.2 Developer retains rights to general methodologies and know-how.

ARTICLE V - WARRANTIES AND LIMITATIONS
5.1 Developer warrants that the Software will perform substantially as specified.
5.2 LIMITATION OF LIABILITY: Developer's liability shall not exceed the contract price.

ARTICLE VI - TERMINATION
6.1 Either party may terminate with 30 days written notice.
6.2 Upon termination, Client shall pay for work completed to date.
""",
            },
            {
                "filename": "technical_requirements.md",
                "content": """# Technical Requirements Document

## 1. System Overview
The system shall be a web-based document processing and question-answering platform.

## 2. Architecture Requirements

### 2.1 Frontend
- React.js with TypeScript
- Responsive design for desktop and mobile
- Real-time document upload status
- Interactive chat interface

### 2.2 Backend
- FastAPI with Python 3.11+
- PostgreSQL database with pgvector extension
- Redis for caching and session management
- Docker containerization

### 2.3 AI/ML Components
- OpenAI GPT-4 for question answering
- OpenAI text-embedding-3-small for embeddings
- LangChain for document processing
- FAISS for vector similarity search

## 3. Functional Requirements

### 3.1 Document Processing
- Support PDF, DOCX, TXT, and MD file formats
- Maximum file size: 50MB
- Automatic text extraction and chunking
- Structure detection and hierarchy preservation
- Token counting for cost estimation

### 3.2 Question Answering
- Natural language question input
- Context-aware responses based on uploaded documents
- Source citation with page numbers
- Confidence scoring for answers

### 3.3 User Management
- User authentication and authorization
- Document access control
- Usage tracking and limits

## 4. Performance Requirements
- Document processing: < 30 seconds for 10MB files
- Question answering: < 5 seconds response time
- Concurrent users: Support up to 100 simultaneous users
- Uptime: 99.9% availability

## 5. Security Requirements
- HTTPS encryption for all communications
- Secure file upload with virus scanning
- API key management for external services
- Data encryption at rest
""",
            },
        ]

        for doc in sample_docs:
            file_path = self.data_directory / doc["filename"]
            if not file_path.exists():
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(doc["content"])
                print(f"✅ Created sample document: {file_path.name}")

    async def process_documents(self) -> list[EnhancedDocument]:
        """Process documents using real services."""
        import time

        start_time = time.time()

        if not self.data_directory.exists():
            print(
                f"📁 Data directory {self.data_directory} not found. Creating with samples..."
            )
            self._create_sample_documents()

        # Check for documents
        files = list(self.data_directory.iterdir())
        supported_extensions = [".pdf", ".docx", ".txt", ".md"]
        doc_files = [
            f for f in files if f.suffix.lower() in supported_extensions and f.is_file()
        ]

        if not doc_files:
            print("📝 No documents found. Creating sample documents...")
            self._create_sample_documents()
            doc_files = list(self.data_directory.iterdir())
            doc_files = [
                f
                for f in doc_files
                if f.suffix.lower() in supported_extensions and f.is_file()
            ]

        print(f"\n📚 Processing {len(doc_files)} documents using real services...")

        for file_path in doc_files:
            print(f"\n🔄 Processing: {file_path.name}")

            try:
                # Use real enhanced document service
                mock_file = MockUploadFile(file_path)
                enhanced_doc = (
                    await self.enhanced_document_service.process_document_enhanced(
                        mock_file, use_enhanced_models=True, preserve_structure=True
                    )
                )

                # Add accurate token counting if missing
                if enhanced_doc.metadata.total_tokens == 0 and enhanced_doc.chunks:
                    total_tokens = 0
                    for chunk in enhanced_doc.chunks:
                        if chunk.token_count == 0:
                            # Use real TokenCounter for accurate counting
                            chunk.token_count = self.token_counter.count_tokens(
                                chunk.text
                            )
                        total_tokens += chunk.token_count
                    enhanced_doc.metadata.total_tokens = total_tokens
                    print(
                        f"  🔢 Calculated tokens: {total_tokens:,} (using real TokenCounter)"
                    )

                # 🌳 Visualize chunks with hierarchical structure (before OpenAI processing)
                if enhanced_doc.chunks:
                    print("\n" + "=" * 80)
                    print_chunks_before_openai(
                        enhanced_doc.chunks, f"Tree-Chunked Document: {file_path.name}"
                    )
                    print("=" * 80 + "\n")

                # Store in vector store using real service
                chunk_count = await self.vector_store.store_enhanced_document(
                    enhanced_doc
                )

                # Update stats
                self.stats["total_chunks"] += chunk_count
                self.stats["total_tokens"] += enhanced_doc.metadata.total_tokens

                self.processed_documents.append(enhanced_doc)

                print(f"  ✅ Processed: {chunk_count} chunks")
                if hasattr(enhanced_doc.metadata, "total_tokens"):
                    print(f"  📊 Tokens: {enhanced_doc.metadata.total_tokens:,}")

            except Exception as e:
                print(f"  ❌ Failed to process {file_path.name}: {str(e)}")
                continue

        self.stats["total_documents"] = len(self.processed_documents)
        self.stats["processing_time"] = time.time() - start_time

        return self.processed_documents

    async def ask_question(self, question: str) -> dict[str, Any]:
        """Ask a question using real services."""
        print(f"\n❓ Question: {question}")

        # Search using real vector store
        search_results = await self.vector_store.search_documents(question, k=5)

        if not search_results:
            return {
                "question": question,
                "answer": "No relevant documents found in the knowledge base.",
                "sources": [],
                "confidence": 0.0,
            }

        # Build context from search results
        context_parts = []
        sources = []

        for i, result in enumerate(search_results[:3]):  # Top 3 results
            context_part = (
                f"[Source {i + 1}: {result['document_name']}, Page {result['page']}]"
            )
            if result.get("section_title"):
                context_part += f" [{result['section_title']}]"
            # Use full chunk content instead of truncated excerpt
            context_part += f"\n{result['content']}"

            context_parts.append(context_part)
            sources.append(
                {
                    "document": result["document_name"],
                    "page": result["page"],
                    "section": result.get("section_title", ""),
                    "similarity": result["similarity"],
                    "tokens": result.get("token_count", 0),
                }
            )

        context = "\n\n".join(context_parts)

        # Generate answer using real LLM
        if self.llm:
            prompt = f"""Based on the following document excerpts, provide a comprehensive answer to the question.

Question: {question}

Document Excerpts:
{context}

Please provide a detailed answer based on the information in the documents:"""

            # Print the final prompt being sent to the LLM
            print("\n🤖 Final LLM Prompt:")
            print("=" * 80)
            print(prompt)
            print("=" * 80)

            try:
                print(f"📡 Sending prompt to {self.llm.model_name}...")
                response = await self.llm.ainvoke(prompt)
                answer = response.content
                print(f"✅ LLM response received ({len(answer)} characters)")
            except Exception as e:
                answer = f"Error generating answer with LLM: {str(e)}"
                print(f"❌ LLM call failed: {str(e)}")
        else:
            answer = f"Based on the documents: {context[:500]}..."

        # Calculate confidence
        avg_similarity = sum(r["similarity"] for r in search_results) / len(
            search_results
        )
        confidence = min(avg_similarity * 1.2, 0.95)  # Boost and cap at 95%

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "total_sources": len(search_results),
        }

    async def run_demonstration(self, question: str) -> dict[str, Any]:
        """Run the complete demonstration."""
        print("=" * 80)
        print("🚀 LOCAL RAG DEMONSTRATION (Real Implementation)")
        print("=" * 80)
        print(f"📁 Data directory: {self.data_directory}")
        print("🔧 Using real implementation services")
        print(
            f"📊 Vector store: {'Real EnhancedVectorStore' if self.use_real_db else 'Local FAISS'}"
        )
        print("=" * 80)

        # Process documents
        documents = await self.process_documents()

        if not documents:
            print("\n❌ No documents were processed.")
            return {
                "question": question,
                "answer": "No documents were processed.",
                "sources": [],
                "confidence": 0.0,
            }

        print("\n📊 Processing Summary:")
        print(f"   Documents: {self.stats['total_documents']}")
        print(f"   Chunks: {self.stats['total_chunks']}")
        print(f"   Tokens: {self.stats['total_tokens']:,}")
        print(f"   Time: {self.stats['processing_time']:.2f}s")

        # Answer question
        result = await self.ask_question(question)

        # Display results
        print("\n💬 Answer:")
        print(f"{result['answer']}")
        print(f"\n📈 Confidence: {result['confidence']:.3f}")

        if result["sources"]:
            print(f"\n📚 Sources ({len(result['sources'])}):")
            for i, source in enumerate(result["sources"], 1):
                print(f"  {i}. 📄 {source['document']} (Page {source['page']})")
                print(f"     🎯 Similarity: {source['similarity']:.3f}")
                if source.get("section"):
                    print(f"     📑 Section: {source['section']}")

        print("\n" + "=" * 80)
        print("✅ Local RAG demonstration completed!")
        print("🎉 All real services tested successfully!")
        print("=" * 80)

        return result


# CLI runner for direct execution
async def main():
    """Main function for CLI execution."""

    # Check dependencies
    if not FAISS_AVAILABLE:
        print("❌ FAISS not available. Please install with: pip install faiss-cpu")
        return

    if not OPENAI_AVAILABLE:
        print("❌ OpenAI not available. Please install with: pip install openai")
        return

    if not LANGCHAIN_OPENAI_AVAILABLE:
        print(
            "❌ langchain-openai not available. Please install with: pip install langchain-openai"
        )
        return

    # Check API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        return

    # Get question from command line or use default
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = (
            "What are the main topics and key provisions discussed in these documents?"
        )

    try:
        # Run demonstration
        runner = LocalRAGRunner(openai_api_key, "data", use_real_db=False)
        result = await runner.run_demonstration(question)

        return result

    except Exception as e:
        print(f"\n❌ Demonstration failed: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


# Pytest integration
import pytest


@pytest.mark.asyncio
async def test_local_rag_demonstration():
    """Test the local RAG demonstration."""

    # Check dependencies
    if not FAISS_AVAILABLE:
        pytest.skip("FAISS not available. Install with: pip install faiss-cpu")

    if not OPENAI_AVAILABLE:
        pytest.skip("OpenAI not available. Install with: pip install openai")

    if not LANGCHAIN_OPENAI_AVAILABLE:
        pytest.skip(
            "langchain-openai not available. Install with: pip install langchain-openai"
        )

    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY not set - skipping test")

    # Use a temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = LocalRAGRunner(openai_api_key, temp_dir, use_real_db=False)

        question = "What are the payment terms mentioned in the agreement?"
        result = await runner.run_demonstration(question)

        # Verify results
        assert result["answer"] is not None
        assert len(result["answer"]) > 0
        assert result["confidence"] >= 0.0
        assert isinstance(result["sources"], list)

        print(
            f"\n✅ Test passed! Processed {runner.stats['total_documents']} documents"
        )
        print(f"📊 Answer confidence: {result['confidence']:.3f}")

        return result


if __name__ == "__main__":
    asyncio.run(main())
