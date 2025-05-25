# Chat With Docs (RAG)
## Tech Stack
| Layer             | Choice                          | Rationale                           |
| ----------------- | ------------------------------- | ----------------------------------- |
| Vector store      | **pgvector (managed Supabase)** | 1-click, UK region, SQL familiarity |
| Embeddings & chat | Azure OpenAI - gpt-4o           | Enterprise SLA, GDPR-aligned        |
| API               | FastAPI                         | Async, quick setup                  |
| Front-end         | Next.js + shadcn/ui             | Rapid UI, SSR                       |
| Auth              | Clerk.dev free tier             | Offload security; JWT passthrough   |
| Hosting           | Fly.io UK or Railway            | Minutes to deploy, EU data          |

## Architecture Overview

The application will use Retrieval-Augmented Generation (RAG) for legal document Q&A, structured as two main pipelines: an offline indexing pipeline for document ingestion, and an online query pipeline for retrieval and generation. In the indexing stage, uploaded documents are parsed and split into chunks (e.g. by paragraphs or clauses) to fit the LLM context window. Each chunk is vectorized into embeddings and stored in a vector database for later semantic search. We will use Supabase (Postgres) with the pgvector extension so that embeddings reside alongside other application data, benefiting from fast vector search with minimal infrastructure. This design ensures all data (documents, embeddings, user info) lives in a single Postgres instance hosted in a UK/EU region (for GDPR compliance) while still enabling low-latency semantic search.

For the query pipeline, when a user asks a question, the system will retrieve the top relevant document chunks from the vector store (using cosine similarity) and assemble a prompt for the LLM. A FastAPI backend will orchestrate this: accepting the query, performing vector similarity search via Supabase, and constructing the prompt with retrieved text. The prompt will include instructions to ground the LLM in the provided legal text and cite sources. For example, we’ll instruct the model not to fabricate information and to say “Not found in documents” if the answer isn’t in the context. This approach, combined with in-line citation formatting (e.g. “[DocName, p. 3]”), helps minimize hallucinations and increases answer trustworthiness. In practice, tools like LlamaIndex’s CitationQueryEngine can automate in-line source citations by attaching document metadata to each chunk. We will implement a similar strategy: each stored chunk knows its source document and page/section, and the LLM’s answer will reference those identifiers. The frontend can then display citations (e.g. as footnotes or links) pointing to the original document text.

To further minimize hallucinations, we evaluate advanced RAG variants. GraphRAG is an emerging technique that incorporates knowledge graphs into RAG. Instead of relying solely on semantic similarity, GraphRAG builds a graph of entities and relationships from the documents and uses graph traversal for multi-hop queries. This can improve accuracy by up to 35% compared to vector-only retrieval in complex domains. Legal documents often have structured hierarchies and cross-references (clauses referencing definitions, precedents, etc.), making them a strong candidate for GraphRAG’s relational approach. For example, a contract might define a term in Section 1 that’s referenced in Section 10 – a graph-based retriever could automatically pull in the definition when Section 10 is queried. Trade-off: GraphRAG provides more explainable, contextually rich answers for such scenarios, but it adds complexity. Constructing a knowledge graph from unstructured legal text requires robust NLP (entity extraction, resolution of references) and extra processing time. Given our 4-week MVP timeline, we recommend starting with standard vector-based RAG using well-proven libraries (LangChain or LlamaIndex) and focusing on prompt techniques (e.g. strict grounding, citations) to reduce hallucinations. This approach is simpler to implement and already addresses our core use case of Q&A with provided documents. We will, however, design the system to be extensible: in the future, a GraphRAG module can be added to pre-process documents into a graph of clauses/definitions, enhancing multi-hop query handling without a complete rewrite.

Additionally, the architecture includes a “claims verification” step as a safeguard. In a legal context, every statement the AI makes should be backed by the documents. We plan an optional post-processing step where the LLM’s answer is broken into factual claims and each claim is checked against the retrieved sources. For example, we can programmatically search the retrieved text snippets for key terms or numbers from the answer to ensure alignment. This is inspired by evidence verification approaches in RAG pipelines – e.g. using an auxiliary function to confirm that any quoted figures/dates in the answer appear in the sources. If a claim is not supported, the system could flag it or append a disclaimer in the answer (or in a future iteration, automatically re-query for the missing piece). This three-step RAG loop (retrieve → answer → verify) will further reduce hallucinations and build user trust in the answers.

System Extensibility: The overall design uses a modular FastAPI backend, making it easy to extend features. We isolate components for authentication, document ingestion, QA chain, and analytics. This separation means future capabilities like multi-tenancy, fine-tuned models, or full-document summaries can plug in with minimal refactoring. For multi-tenancy, we plan to introduce an organization_id (tenant identifier) in relevant tables (Users, Documents, Embeddings) so that queries and data access are always scoped per tenant. In the short term, we’ll run a single-tenant MVP (one law firm = one tenant) to reduce complexity, but the code will be written with tenant-awareness (for example, filtering document queries by org_id). Scaling to multi-tenant SaaS then becomes mostly a configuration change – either using row-level security in Postgres or separate schema/DB per tenant if required by clients. Similarly, the LLM service is abstracted behind an interface so we can swap it out: today we might call OpenAI’s API (gpt-3.5-turbo for low latency), but later we could integrate an Azure OpenAI (UK datacenter) or a fine-tuned local model for data privacy. The system will initially use OpenAI with no user data retention (and possibly through Azure to ensure UK/EU data residency), aligning with GDPR. We’ll also include a document deletion feature – users can remove uploaded files, which triggers deletion of the file and its vectors, to comply with right-to-be-forgotten requirements.

Performance and scalability: On the backend, FastAPI will serve a REST (or GraphQL) API, which the Next.js frontend will call for actions like uploading files, asking questions, and user auth. We’ll containerize the FastAPI app for deployment on Fly.io, choosing a European region (London) to host near our users for low latency. The vector search is handled in-database via pgvector, which is efficient and horizontally scalable (Postgres can be scaled read-heavy workloads, and we can cache frequent embeddings in memory). We also plan basic caching at the application layer – e.g., recently asked questions and their results per document set – to speed up repeat queries. The MVP will focus on correctness and latency at small scale (a few thousand pages); as we approach larger scale or more concurrent users, we can introduce optimizations like async background processing for file ingestion, streaming responses for the chat, and eventually consider distributed vector stores or question indexing if needed.

## Suggested File Structures

A well-organized codebase will accelerate development. Below are the recommended file structures for the FastAPI backend and Next.js frontend, organized for clarity and future growth:

Backend (FastAPI) – A Python package app with clear submodules for routers, services, and models:

backend/
├── app/
│   ├── main.py               # FastAPI app initialization, route inclusion, CORS, etc.
│   ├── api/
│   │   ├── routes/
│   │   │   ├── auth.py       # Authentication endpoints (login, signup, token)
│   │   │   ├── documents.py  # File upload, list documents, delete document
│   │   │   └── chat.py       # Chat query endpoint (accepts question, returns answer)
│   │   └── dependencies.py   # Common dependencies (e.g. get_db session, auth checker)
│   ├── core/                 # Core logic and services
│   │   ├── ingestion.py      # Functions for parsing files, chunking, embedding
│   │   ├── qna.py            # Functions for retrieval and LLM query (RAG chain)
│   │   ├── vectorstore.py    # Utility to query or update the pgvector index
│   │   └── settings.py       # Config (API keys, DB connection, etc.)
│   ├── models/
│   │   ├── schemas.py        # Pydantic models for request/response (Document, Query, Answer)
│   │   └── database.py       # SQLAlchemy models or queries for Users, Documents, Chunks
│   ├── utils/
│   │   └── security.py       # Auth helpers (password hashing, JWT decoding)
│   └── tests/
│       ├── test_ingestion.py # Unit tests for document parsing & chunking
│       ├── test_qna.py       # Tests for retrieval+LLM pipeline (with mocked LLM)
│       └── test_api.py       # API endpoint tests (upload, query flows)
├── Dockerfile                # Container configuration for Fly.io deployment
├── requirements.txt          # Python dependencies (FastAPI, langchain, openai, etc.)
└── README.md                 # Developer setup and run instructions


Frontend (Next.js) – A TypeScript project bootstrapped with create-next-app, structured into pages and reusable components:

frontend/
├── pages/ or app/            # Next.js routes (using `pages` directory or `app` directory)
│   ├── index.tsx             # Landing page / login (if not authenticated, or intro)
│   ├── chat.tsx              # Protected page hosting the chat interface
│   └── upload.tsx            # Protected page for uploading and listing documents
├── components/
│   ├── Layout.tsx            # Common layout (header with app name, perhaps logout)
│   ├── DocumentList.tsx      # Shows uploaded documents and statuses
│   ├── UploadForm.tsx        # Drag-and-drop or file input component
│   ├── ChatWindow.tsx        # Main chat UI component (messages display + input box)
│   ├── MessageBubble.tsx     # Sub-component for a single message (question or answer with citations)
│   └── CitationViewer.tsx    # Component to show source snippet when citation is clicked
├── context/ or hooks/
│   └── AuthProvider.tsx      # Context provider for authentication state (JWT, user info)
├── lib/
│   ├── api.ts                # Wrapper functions for calling backend API (using fetch/Axios)
│   └── auth.ts               # Helpers for JWT management, e.g., storing token, redirect if unauth
├── styles/
│   └── globals.css           # Global styles or Tailwind CSS config if used
├── public/
│   └── ...                   # Static assets (e.g., logo)
├── next.config.js            # Next.js configuration (set API URLs, etc.)
├── package.json
└── README.md


Notes: We will use Next.js’s built-in routing for simplicity; the chat and upload pages will be behind authentication. After login, the user lands on upload (to add/view documents) and can navigate to chat to query their documents. We separate UI concerns: the chat interface is isolated in ChatWindow and can manage the conversational state (messages array). The CitationViewer might be a popover or modal that, given a citation reference (like a doc ID and page), fetches and displays the corresponding snippet or highlights it – this can be implemented in Week 3 or 4 once basic chat is working. Styling can use a component library or custom CSS; given the timeline, we might opt for a simple design or Tailwind CSS for speed.
