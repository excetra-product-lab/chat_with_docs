#!/bin/bash

# Local RAG Demonstration Runner
# This script sets up and runs the RAG demonstration using real implementation services
# Uses FAISS for local vector storage (no database dependency)

set -e  # Exit on any error

echo "🚀 Local RAG Demonstration Setup (Real Implementation Services)"
echo "================================================================"

# Check if we're in the right directory
if [ ! -f "backend/tests/test_local_rag_run.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected to find: backend/tests/test_local_rag_run.py"
    exit 1
fi

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY environment variable is not set"
    echo ""
    echo "Please set your OpenAI API key:"
    echo "  export OPENAI_API_KEY='your-openai-api-key-here'"
    echo ""
    echo "Or create a .env file in the project root with:"
    echo "  OPENAI_API_KEY=your-openai-api-key-here"
    exit 1
fi

echo "✅ OpenAI API key found"

# Check virtual environment exists
if [ ! -f "backend/.venv/bin/python" ]; then
    echo "❌ Error: Virtual environment not found at backend/.venv"
    echo ""
    echo "Please create it first with:"
    echo "  cd backend && uv venv && uv pip install -r requirements.txt"
    exit 1
fi

# Use virtual environment's Python and uv pip
VENV_PYTHON="backend/.venv/bin/python"

echo "🔧 Using uv-managed virtual environment at backend/.venv"
echo ""
echo "📦 Checking dependencies..."

if ! $VENV_PYTHON -c "import faiss" 2>/dev/null; then
    echo "📥 Installing FAISS..."
    cd backend && uv pip install faiss-cpu && cd ..
else
    echo "✅ FAISS already installed"
fi

if ! $VENV_PYTHON -c "import langchain_openai" 2>/dev/null; then
    echo "📥 Installing LangChain OpenAI..."
    cd backend && uv pip install langchain-openai && cd ..
else
    echo "✅ LangChain OpenAI already installed"
fi

if ! $VENV_PYTHON -c "import numpy" 2>/dev/null; then
    echo "📥 Installing NumPy..."
    cd backend && uv pip install numpy && cd ..
else
    echo "✅ NumPy already installed"
fi

# Create data directory if it doesn't exist
echo ""
echo "📁 Setting up data directory..."
if [ ! -d "data" ]; then
    mkdir -p data
    echo "✅ Created data directory"
else
    echo "✅ Data directory exists"
fi

# Check if data directory has files
file_count=$(find data -name "*.pdf" -o -name "*.docx" -o -name "*.txt" -o -name "*.md" | wc -l)
if [ "$file_count" -eq 0 ]; then
    echo "📝 No documents found in data directory"
    echo "   Sample documents will be created automatically"
else
    echo "✅ Found $file_count document(s) in data directory"
fi

# Determine the question to ask
QUESTION="${1:-What are the main topics discussed in these documents?}"

echo ""
echo "❓ Question: $QUESTION"
echo ""

# Run the demonstration using real implementation services
echo "🏃 Running Local RAG Demonstration (Real Implementation Services)..."
echo "==================================================================="

# Set PYTHONPATH and run using virtual environment's Python
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)/backend"
echo "🔧 Using virtual environment Python with proper PYTHONPATH"
$VENV_PYTHON backend/tests/test_local_rag_run.py "$QUESTION"

echo ""
echo "🎉 Local RAG demonstration completed!"
echo "✅ Tested real implementation services with local FAISS storage"
echo ""
echo "💡 To run with a different question:"
echo "   ./run_e2e_rag_demo.sh \"Your question here\""
echo ""
echo "💡 To run as a pytest:"
echo "   export PYTHONPATH=\$(pwd)/backend && backend/.venv/bin/python -m pytest backend/tests/test_local_rag_run.py -v -s"
echo ""
echo "💡 To run directly with Python:"
echo "   export PYTHONPATH=\$(pwd)/backend && backend/.venv/bin/python backend/tests/test_local_rag_run.py \"Your question\""
echo ""
echo "💡 Or using uv from backend directory:"
echo "   cd backend && uv run python tests/test_local_rag_run.py \"Your question\""
