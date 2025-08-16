# RAG System Evaluation Guide

This guide explains how to evaluate your RAG (Retrieval-Augmented Generation) system using the LegalBench-RAG benchmark and quick evaluation methods.

## Table of Contents

1. [What are Evaluations?](#what-are-evaluations)
2. [LegalBench-RAG Overview](#legalbench-rag-overview)
3. [Quick Start](#quick-start)
4. [Evaluation Modes](#evaluation-modes)
5. [Understanding Results](#understanding-results)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

## What are Evaluations?

Evaluations (evals) are systematic ways to measure how well your AI system performs on specific tasks. For RAG systems, evaluations typically measure:

### Retrieval Quality

- **Precision**: Of the documents retrieved, how many are actually relevant?
- **Recall**: Of all relevant documents, how many were retrieved?
- **F1 Score**: Harmonic mean of precision and recall

### Generation Quality

- **Accuracy**: Are the answers factually correct?
- **Relevance**: Do answers address the question asked?
- **Completeness**: Are answers comprehensive?

### End-to-End Performance

- **Response Time**: How fast does the system respond?
- **Success Rate**: What percentage of queries are handled successfully?

## LegalBench-RAG Overview

[LegalBench-RAG](https://github.com/zeroentropy-ai/legalbenchrag) is a specialized benchmark for evaluating RAG systems on legal documents. It provides:

- **Legal Document Corpus**: Real legal contracts and documents
- **Complex Legal Questions**: Questions requiring deep understanding
- **Ground Truth Answers**: Character-level precise expected results
- **Deterministic Metrics**: Exact precision/recall calculation

### Key Features

- Character-level precision for exact matching
- Complex legal reasoning tasks
- Multiple legal document types (contracts, agreements, policies)
- Standardized evaluation metrics

## Quick Start

### Prerequisites

1. **OpenAI API Key**: Set your OpenAI API key

   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

2. **Dependencies**: Install required packages

   ```bash
   cd backend
   uv pip install faiss-cpu langchain-openai requests
   ```

### Quick Evaluation (Recommended for First Time)

Run a quick evaluation using sample documents:

```bash
cd backend
python evaluate_rag.py --mode quick
```

This will:

1. Create sample legal documents in the `data/` directory
2. Process them through your RAG pipeline
3. Run evaluation queries
4. Display results

### Setup LegalBench-RAG Dataset

To use the full LegalBench-RAG benchmark:

```bash
# Check dataset status
python setup_legalbench_data.py --info

# Create sample dataset for testing
python setup_legalbench_data.py --create-sample

# Download full dataset (when available)
python setup_legalbench_data.py --download
```

### Full Evaluation

Run the complete LegalBench-RAG evaluation:

```bash
python evaluate_rag.py --mode full --corpus-path data/legalbench_corpus --benchmarks-path data/legalbench_benchmarks
```

## Evaluation Modes

### 1. Quick Mode (`--mode quick`)

Uses sample legal documents and pre-defined test cases:

```bash
# Basic quick evaluation
python evaluate_rag.py --mode quick

# Save results to file
python evaluate_rag.py --mode quick --output results/quick_eval.json

# Use custom data directory
python evaluate_rag.py --mode quick --data-directory my_data/
```

**Features:**

- Fast setup and execution
- No external dataset required
- Good for initial testing and development
- Covers common legal document types

**Metrics:**

- Keyword coverage
- Document relevance
- Retrieval accuracy
- Response time

### 2. Full Mode (`--mode full`)

Uses the complete LegalBench-RAG benchmark:

```bash
# Full evaluation with all benchmarks
python evaluate_rag.py --mode full

# Evaluate specific benchmark file
python evaluate_rag.py --mode full --benchmark-file cuad_benchmark.json

# Limit test cases for faster testing
python evaluate_rag.py --mode full --max-cases 10

# Custom paths
python evaluate_rag.py --mode full --corpus-path /path/to/corpus --benchmarks-path /path/to/benchmarks
```

**Features:**

- Comprehensive evaluation
- Industry-standard benchmark
- Character-level precision
- Multiple legal domains

**Metrics:**

- Precision, Recall, F1 Score
- Exact match accuracy
- Response time
- Success rate

## Understanding Results

### Quick Mode Results

```json
{
  "total_test_cases": 5,
  "avg_keyword_coverage": 0.85,
  "avg_relevance_score": 0.92,
  "avg_retrieval_time": 0.123,
  "avg_generation_time": 1.456,
  "detailed_results": [
    {
      "query": "What are the payment terms?",
      "keyword_coverage": 0.90,
      "relevance_score": 0.95,
      "retrieved_count": 3,
      "found_keywords": ["payment", "75000", "30%"],
      "missing_keywords": ["milestone"],
      "generated_answer": "The payment terms include..."
    }
  ]
}
```

**Key Metrics:**

- **Keyword Coverage**: Percentage of expected keywords found (0.0-1.0)
- **Relevance Score**: How relevant retrieved documents are (0.0-1.0)
- **Retrieval Time**: Time to find relevant documents (seconds)
- **Generation Time**: Time to generate answer (seconds)

### Full Mode Results

```json
{
  "benchmark_name": "cuad_benchmark.json",
  "total_test_cases": 100,
  "avg_precision": 0.78,
  "avg_recall": 0.65,
  "avg_f1_score": 0.71,
  "avg_exact_match": 0.45,
  "success_rate": 0.95,
  "detailed_results": [...]
}
```

**Key Metrics:**

- **Precision**: Relevance of retrieved documents (0.0-1.0)
- **Recall**: Coverage of relevant documents (0.0-1.0)
- **F1 Score**: Balance of precision and recall (0.0-1.0)
- **Exact Match**: Percentage of perfect retrievals (0.0-1.0)
- **Success Rate**: Percentage of queries processed successfully (0.0-1.0)

### Interpreting Scores

| Score Range | Quality Level | Interpretation |
|-------------|---------------|----------------|
| 0.9 - 1.0   | Excellent     | Production-ready performance |
| 0.8 - 0.9   | Good          | Suitable for most use cases |
| 0.7 - 0.8   | Fair          | Needs improvement for critical applications |
| 0.6 - 0.7   | Poor          | Significant issues, requires optimization |
| < 0.6       | Unacceptable  | Major problems, system redesign needed |

## Advanced Usage

### Custom Test Cases

Create your own test cases for domain-specific evaluation:

```python
from app.services.evaluation_service import QuickEvaluator

# Create custom test cases
custom_cases = [
    {
        "query": "What is the termination clause?",
        "expected_keywords": ["termination", "notice", "30 days"],
        "document_name": "employment_contract.txt"
    }
]

# Run custom evaluation
evaluator = QuickEvaluator(document_service, vector_store, llm)
results = await evaluator.run_custom_evaluation(custom_cases)
```

### Batch Evaluation

Evaluate multiple configurations:

```bash
# Test different chunk sizes
for chunk_size in 400 600 800; do
    echo "Testing chunk size: $chunk_size"
    # Update configuration
    python evaluate_rag.py --mode quick --output "results/chunk_${chunk_size}.json"
done
```

### Performance Benchmarking

Compare different models or configurations:

```bash
# Baseline evaluation
python evaluate_rag.py --mode quick --output results/baseline.json

# Test with different embedding model
# (update configuration)
python evaluate_rag.py --mode quick --output results/new_embeddings.json

# Compare results
python -c "
import json
baseline = json.load(open('results/baseline.json'))
new_model = json.load(open('results/new_embeddings.json'))
print(f'Baseline F1: {baseline[\"avg_relevance_score\"]:.3f}')
print(f'New Model F1: {new_model[\"avg_relevance_score\"]:.3f}')
"
```

### Integration with CI/CD

Add evaluation to your continuous integration:

```yaml
# .github/workflows/eval.yml
name: RAG Evaluation
on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt
          pip install faiss-cpu langchain-openai

      - name: Run evaluation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          cd backend
          python evaluate_rag.py --mode quick --output results/ci_eval.json

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: evaluation-results
          path: backend/results/
```

## Troubleshooting

### Common Issues

#### 1. Missing Dependencies

```bash
❌ FAISS not available. Please install with: pip install faiss-cpu
```

**Solution**: Install required packages

```bash
cd backend
uv pip install faiss-cpu langchain-openai
```

#### 2. Missing API Key

```bash
❌ OPENAI_API_KEY environment variable not set.
```

**Solution**: Set your OpenAI API key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

#### 3. No Documents Found

```bash
📝 No documents found in data directory
```

**Solution**: The system will create sample documents automatically, or you can add your own documents to the `data/` directory.

#### 4. Low Evaluation Scores

**Possible Causes:**

- Inappropriate chunk size
- Poor document quality
- Inadequate embedding model
- Insufficient context

**Solutions:**

- Experiment with different chunk sizes (400-800 tokens)
- Improve document preprocessing
- Use higher-quality embedding models
- Increase retrieval count (k parameter)

#### 5. Slow Evaluation

**Optimizations:**

- Use local FAISS instead of database vector store
- Limit test cases with `--max-cases`
- Use faster embedding models
- Optimize document processing

### Performance Tuning

#### Retrieval Optimization

```python
# Experiment with different parameters
search_results = await vector_store.search_documents(
    query,
    k=10,  # Try different values: 5, 10, 15
    similarity_threshold=0.7  # Filter low-relevance results
)
```

#### Generation Optimization

```python
# Adjust LLM parameters
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Try different models
    temperature=0.1,      # Lower for more consistent results
    max_tokens=500        # Limit response length
)
```

### Getting Help

1. **Check logs**: Enable verbose logging with `--verbose`
2. **Review documentation**: Read the LegalBench-RAG paper
3. **Examine results**: Look at detailed results for patterns
4. **Community support**: Check the GitHub repository for issues and discussions

## Best Practices

1. **Start with Quick Mode**: Get familiar with evaluation concepts
2. **Establish Baseline**: Run initial evaluation to set expectations
3. **Iterate Systematically**: Change one parameter at a time
4. **Document Changes**: Keep track of configuration changes and results
5. **Regular Evaluation**: Run evaluations as part of development workflow
6. **Domain-Specific Testing**: Create test cases relevant to your use case
7. **Performance Monitoring**: Track evaluation metrics over time

## Next Steps

1. Run your first quick evaluation
2. Analyze the results and identify improvement areas
3. Experiment with different configurations
4. Set up the full LegalBench-RAG dataset
5. Integrate evaluation into your development workflow
6. Create custom test cases for your specific domain

For more information, refer to:

- [LegalBench-RAG Paper](https://arxiv.org/abs/2408.10343)
- [LegalBench-RAG GitHub](https://github.com/zeroentropy-ai/legalbenchrag)
- Your project's existing RAG documentation
