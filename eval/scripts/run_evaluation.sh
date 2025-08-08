#!/bin/bash

# RAG Evaluation Demo Script
# This script demonstrates how to run evaluations on your RAG system

set -e  # Exit on any error

echo "🧪 RAG System Evaluation Demo"
echo "=============================="

# Always calculate paths relative to script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"

# Verify we found the right directory
if [ ! -f "$PROJECT_ROOT/backend/app/main.py" ]; then
    echo "❌ Error: Cannot find backend/app/main.py"
    echo "   Script directory: $SCRIPT_DIR"
    echo "   Calculated project root: $PROJECT_ROOT"
    echo ""
    echo "Please ensure the script is in the correct location."
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
VENV_PYTHON="$PROJECT_ROOT/backend/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Error: Virtual environment not found at $VENV_PYTHON"
    echo ""
    echo "Please create it first with:"
    echo "  cd $PROJECT_ROOT/backend && uv venv && uv pip install -r requirements.txt"
    exit 1
fi

echo "🔧 Using virtual environment at backend/.venv"
echo ""

# Check and install evaluation dependencies
echo "📦 Checking evaluation dependencies..."

if ! $VENV_PYTHON -c "import faiss" 2>/dev/null; then
    echo "📥 Installing FAISS for vector search..."
    (cd "$PROJECT_ROOT/backend" && uv pip install faiss-cpu)
else
    echo "✅ FAISS already installed"
fi

if ! $VENV_PYTHON -c "import requests" 2>/dev/null; then
    echo "📥 Installing requests for data download..."
    (cd "$PROJECT_ROOT/backend" && uv pip install requests)
else
    echo "✅ requests already installed"
fi

echo ""

# Set PYTHONPATH to include backend and eval directories
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PROJECT_ROOT/backend:$PROJECT_ROOT/eval"

# Function to run evaluation and show results
run_evaluation() {
    local mode=$1
    local description=$2
    local extra_args=$3

    echo "🏃 Running $description..."
    echo "=================================================="

    # Run evaluation (ensure we're in the right directory)
    (cd "$(dirname "${BASH_SOURCE[0]}")" && $VENV_PYTHON evaluate_rag.py --mode $mode $extra_args)

    echo ""
    echo "=================================================="
    echo "✅ $description completed!"
    echo ""
}

# Parse command line arguments
MODE="quick"
SHOW_HELP=false
FULL_DEMO=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --full-demo)
            FULL_DEMO=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ "$SHOW_HELP" = true ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --mode MODE         Evaluation mode: 'quick' or 'full' (default: quick)"
    echo "  --full-demo         Run complete demo with dataset setup"
    echo ""
    echo "Examples:"
    echo "  $0                  # Run quick evaluation"
    echo "  $0 --mode quick     # Run quick evaluation"
    echo "  $0 --mode full      # Run full LegalBench-RAG evaluation"
    echo "  $0 --full-demo      # Run complete demo with setup"
    echo ""
    echo "Environment Variables:"
    echo "  OPENAI_API_KEY      Required: Your OpenAI API key"
    echo ""
    exit 0
fi

# Main evaluation flow
echo "🎯 Starting evaluation in '$MODE' mode"
echo ""

if [ "$FULL_DEMO" = true ]; then
    echo "📚 Setting up LegalBench-RAG dataset..."
    echo "======================================="

    # Check dataset status
    echo "🔍 Checking dataset status..."
    (cd "$(dirname "${BASH_SOURCE[0]}")" && $VENV_PYTHON setup_legalbench_data.py --info)

    echo ""
    echo "🛠️  Creating sample dataset for demonstration..."
    (cd "$(dirname "${BASH_SOURCE[0]}")" && $VENV_PYTHON setup_legalbench_data.py --create-sample)

    echo ""
    echo "✅ Dataset setup complete!"
    echo ""
fi

# Create results directory
mkdir -p "$PROJECT_ROOT/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ "$MODE" = "quick" ]; then
    # Quick evaluation
    OUTPUT_FILE="$PROJECT_ROOT/results/quick_evaluation_${TIMESTAMP}.json"
    run_evaluation "quick" "Quick Evaluation with Sample Documents" "--output $OUTPUT_FILE"

    echo "💾 Results saved to: $OUTPUT_FILE"

elif [ "$MODE" = "full" ]; then
    # Full evaluation
    if [ ! -d "$PROJECT_ROOT/data/legalbench_corpus" ] || [ ! -d "$PROJECT_ROOT/data/legalbench_benchmarks" ]; then
        echo "📚 LegalBench-RAG dataset not found. Creating sample dataset..."
        (cd "$(dirname "${BASH_SOURCE[0]}")" && $VENV_PYTHON setup_legalbench_data.py --create-sample --data-directory "$PROJECT_ROOT/data")
        echo ""
    fi

    OUTPUT_FILE="$PROJECT_ROOT/results/full_evaluation_${TIMESTAMP}.json"
    run_evaluation "full" "Full LegalBench-RAG Evaluation" "--output $OUTPUT_FILE --max-cases 5"

    echo "💾 Results saved to: $OUTPUT_FILE"

else
    echo "❌ Error: Unknown evaluation mode '$MODE'"
    echo "   Supported modes: 'quick', 'full'"
    exit 1
fi

echo ""
echo "🎉 Evaluation Demo Completed!"
echo "============================="
echo ""
echo "📊 What was evaluated:"
if [ "$MODE" = "quick" ]; then
    echo "   • Sample legal documents (contracts, NDAs, licenses)"
    echo "   • Keyword coverage and document relevance"
    echo "   • Retrieval and generation performance"
else
    echo "   • LegalBench-RAG benchmark dataset"
    echo "   • Precision, recall, and F1 scores"
    echo "   • Character-level exact matching"
fi

echo ""
echo "📈 Key metrics measured:"
echo "   • Document retrieval accuracy"
echo "   • Answer generation quality"
echo "   • System response time"
echo "   • Overall success rate"

echo ""
echo "💡 Next steps:"
echo "   1. Review the results in the generated JSON file"
echo "   2. Identify areas for improvement"
echo "   3. Experiment with different configurations"
echo "   4. Run evaluations regularly during development"

echo ""
echo "📚 For detailed information, see:"
echo "   • eval/docs/evaluation_guide.md"
echo "   • https://arxiv.org/abs/2408.10343 (LegalBench-RAG paper)"
echo "   • https://github.com/zeroentropy-ai/legalbenchrag"

echo ""
echo "🔧 To run other evaluation modes:"
echo "   • Quick evaluation: eval/scripts/run_evaluation.sh --mode quick"
echo "   • Full evaluation: eval/scripts/run_evaluation.sh --mode full"
echo "   • Custom evaluation: cd eval/scripts && python evaluate_rag.py --help"

echo ""
echo "✨ Evaluation demo complete!"
