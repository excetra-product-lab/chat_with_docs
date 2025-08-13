# RAG System Evaluation

This directory contains a comprehensive evaluation system for your RAG (Retrieval-Augmented Generation) pipeline using the [LegalBench-RAG benchmark](https://github.com/zeroentropy-ai/legalbenchrag).

## Quick Start

```bash
# From project root
./eval/run_evaluation.sh

# Or from eval directory
cd eval && ./run_evaluation.sh

# Or from eval/scripts directory
cd eval/scripts && ./run_evaluation.sh
```

## Directory Structure

```text
eval/
├── run_evaluation.sh              # Main entry point (from project root)
├── README.md                      # This file
├── services/
│   ├── __init__.py               # Python module initialization
│   └── evaluation_service.py     # Core evaluation logic
├── scripts/
│   ├── run_evaluation.sh         # Main evaluation script
│   ├── evaluate_rag.py          # Python evaluation CLI
│   └── setup_legalbench_data.py # Dataset management
└── docs/
    ├── README.md                 # Comprehensive usage guide
    └── evaluation_guide.md       # Detailed evaluation documentation
```

## What This Evaluation System Does

### 🎯 Purpose

- Evaluate your RAG system's performance on complex legal document understanding
- Measure both retrieval quality and generation accuracy
- Provide industry-standard benchmark comparisons

### 📊 Evaluation Modes

#### Quick Mode (Recommended for beginners)

```bash
./run_evaluation.sh --mode quick
```

- Uses sample legal documents (contracts, NDAs, licenses)
- Fast execution (~2-3 minutes)
- Measures keyword coverage and document relevance
- Great for development and testing

#### Full Mode (Complete benchmark)

```bash
./run_evaluation.sh --mode full
```

- Uses complete LegalBench-RAG benchmark
- Character-level precision measurement
- Industry-standard metrics (Precision, Recall, F1)
- Comprehensive evaluation across multiple legal domains

## Key Features

### 🏆 Industry Standard

- Based on LegalBench-RAG research (https://arxiv.org/abs/2408.10343)
- Character-level ground truth matching
- Deterministic precision and recall calculation

### ⚡ Easy to Use

- One-command evaluation
- Automatic dependency installation
- Clear, interpretable results
- Multiple evaluation modes

### 🔧 Flexible & Integrated

- Works with your existing RAG pipeline
- Supports custom test cases
- JSON result output for analysis
- Modular architecture

## Usage Examples

### Basic Evaluation

```bash
# Quick evaluation with default settings
./run_evaluation.sh

# Full demo with dataset setup
./run_evaluation.sh --full-demo

# Specific evaluation mode
./run_evaluation.sh --mode full
```

### Advanced Usage

```bash
# Custom evaluation with output file
cd eval/scripts
python evaluate_rag.py --mode quick --output ../../results/my_eval.json

# Limited test cases for faster testing
python evaluate_rag.py --mode full --max-cases 10

# Dataset management
python setup_legalbench_data.py --create-sample
python setup_legalbench_data.py --info
```

## Understanding Results

### Quick Mode Metrics

| Metric | Good Score | What It Means |
|--------|------------|---------------|
| Keyword Coverage | > 80% | System finds expected terms |
| Relevance Score | > 85% | Retrieved documents are relevant |
| Retrieval Time | < 0.5s | Fast document search |
| Generation Time | < 3s | Fast answer generation |

### Full Mode Metrics

| Metric | Good Score | What It Means |
|--------|------------|---------------|
| Precision | > 0.7 | Retrieved documents are relevant |
| Recall | > 0.6 | Finding most relevant documents |
| F1 Score | > 0.65 | Good balance of precision/recall |
| Exact Match | > 0.4 | High accuracy in content matching |

### Performance Levels

- **🟢 Excellent (0.9+)**: Production-ready
- **🟡 Good (0.7-0.9)**: Suitable for most use cases
- **🟠 Fair (0.6-0.7)**: Needs optimization
- **🔴 Poor (<0.6)**: Requires significant improvements

## Prerequisites

1. **OpenAI API Key**:

   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

2. **Python Virtual Environment**:

   ```bash
   cd backend
   uv venv && uv pip install -r requirements.txt
   ```

3. **Dependencies** (auto-installed):
   - `faiss-cpu` for vector search
   - `langchain-openai` for embeddings/generation
   - `requests` for data download

## Configuration

The evaluation system uses your existing RAG pipeline configuration:

- Document processing settings from `backend/app/services/`
- Embedding and LLM models from your current setup
- Vector store configuration (defaults to local FAISS)

## Troubleshooting

### Common Issues

**Missing API Key**:

```bash
export OPENAI_API_KEY="your-key-here"
```

**Virtual Environment Not Found**:

```bash
cd backend && uv venv && uv pip install -r requirements.txt
```

**Import Errors**:

- The evaluation system automatically sets up Python paths
- Run from the correct directory (project root or eval/)

**Low Scores**:

- Try different chunk sizes (400-800 tokens)
- Experiment with different embedding models
- Increase retrieval count (k parameter)
- Improve document preprocessing

### Getting Help

1. **Check logs**: Use `--verbose` flag for detailed output
2. **Review documentation**: See `docs/evaluation_guide.md`
3. **Examine results**: Look at detailed JSON results
4. **Performance tuning**: Try different configurations

## Integration with Development Workflow

### Regular Evaluation

```bash
# Add to your development routine
./eval/run_evaluation.sh --mode quick
```

### CI/CD Integration

```yaml
# Example GitHub Actions
- name: Run RAG Evaluation
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: ./eval/run_evaluation.sh --mode quick
```

### Custom Test Cases

```python
# Create domain-specific evaluations
custom_cases = [
    {
        "query": "What is the termination clause?",
        "expected_keywords": ["termination", "notice", "30 days"],
        "document_name": "employment_contract.txt"
    }
]
```

## API Reference

### Core Classes

```python
from eval.services import LegalBenchRAGEvaluator, QuickEvaluator

# Quick evaluation
evaluator = QuickEvaluator(document_service, vector_store, llm)
results = await evaluator.run_quick_evaluation()

# Full benchmark evaluation
evaluator = LegalBenchRAGEvaluator(document_service, vector_store, llm)
results = await evaluator.run_benchmark("benchmark.json")
```

### Command Line Interface

```bash
# Evaluation modes
python eval/scripts/evaluate_rag.py --mode [quick|full]

# Dataset management
python eval/scripts/setup_legalbench_data.py [--info|--create-sample|--download]

# Demo script
./eval/run_evaluation.sh [--mode MODE] [--full-demo] [--help]
```

## Contributing

To extend the evaluation system:

1. **Add new metrics**: Extend `EvaluationMetrics` class
2. **Custom evaluators**: Inherit from base evaluator classes
3. **New benchmarks**: Add benchmark files to data directory
4. **Integration**: Update import paths and documentation

## References

- **LegalBench-RAG Paper**: https://arxiv.org/abs/2408.10343
- **LegalBench-RAG GitHub**: https://github.com/zeroentropy-ai/legalbenchrag
- **Detailed Documentation**: `docs/evaluation_guide.md`

---

**Ready to evaluate your RAG system?**

```bash
./eval/run_evaluation.sh
```
