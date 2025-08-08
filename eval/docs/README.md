# RAG System Evaluation with LegalBench-RAG

This implementation provides comprehensive evaluation capabilities for your RAG (Retrieval-Augmented Generation) system using the [LegalBench-RAG benchmark](https://github.com/zeroentropy-ai/legalbenchrag).

## What This Evaluation System Does

### 🎯 Purpose

Evaluate your RAG system's performance on complex legal document understanding tasks using industry-standard benchmarks.

### 📊 What Gets Measured

- **Retrieval Quality**: How well your system finds relevant document chunks
- **Generation Quality**: How accurately your system answers questions
- **Performance Metrics**: Response times and success rates
- **Character-Level Precision**: Exact matching against ground truth

### 🏆 Why It Matters

- **Objective Assessment**: Get quantitative metrics on your RAG system's performance
- **Benchmarking**: Compare against industry standards using LegalBench-RAG
- **Optimization**: Identify areas for improvement in your pipeline
- **Confidence**: Know how well your system performs before deployment

## Quick Start (5 minutes)

### 1. Set up your environment

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-openai-api-key-here"

# Install evaluation dependencies
cd backend
uv pip install faiss-cpu langchain-openai requests
```

### 2. Run a quick evaluation

```bash
# From project root
./run_rag_evaluation.sh
```

This will:

- Create sample legal documents
- Process them through your RAG pipeline
- Run evaluation queries
- Show detailed results

### 3. View results

The evaluation will display results like:

```
📊 EVALUATION RESULTS SUMMARY
=====================================
📈 Average Keyword Coverage: 85.00%
🎯 Average Relevance Score: 92.00%
⏱️  Average Retrieval Time: 0.123s
🤖 Average Generation Time: 1.456s
📝 Total Test Cases: 5
```

## Evaluation Modes

### Quick Mode (Recommended for first-time users)

```bash
# Basic quick evaluation
./run_rag_evaluation.sh --mode quick

# Save results to file
python backend/evaluate_rag.py --mode quick --output results/my_eval.json
```

**What it does:**

- Uses sample legal documents (contracts, NDAs, licenses)
- Tests common legal queries
- Measures keyword coverage and document relevance
- Fast execution (~2-3 minutes)

### Full Mode (Complete LegalBench-RAG benchmark)

```bash
# Setup the dataset first
python backend/setup_legalbench_data.py --create-sample

# Run full evaluation
./run_rag_evaluation.sh --mode full
```

**What it does:**

- Uses the complete LegalBench-RAG benchmark
- Character-level precision measurement
- Industry-standard metrics (Precision, Recall, F1)
- Comprehensive legal document types

## Understanding Your Results

### Quick Mode Metrics

| Metric | Good Score | What It Means |
|--------|------------|---------------|
| Keyword Coverage | > 80% | Your system finds most expected terms |
| Relevance Score | > 85% | Retrieved documents are highly relevant |
| Retrieval Time | < 0.5s | Fast document search |
| Generation Time | < 3s | Fast answer generation |

### Full Mode Metrics

| Metric | Good Score | What It Means |
|--------|------------|---------------|
| Precision | > 0.7 | Retrieved documents are relevant |
| Recall | > 0.6 | You're finding most relevant documents |
| F1 Score | > 0.65 | Good balance of precision and recall |
| Exact Match | > 0.4 | High accuracy in finding exact content |

### Performance Interpretation

- **🟢 Excellent (0.9+)**: Production-ready performance
- **🟡 Good (0.7-0.9)**: Suitable for most use cases
- **🟠 Fair (0.6-0.7)**: Needs optimization
- **🔴 Poor (<0.6)**: Requires significant improvements

## File Structure

```
backend/
├── app/services/
│   └── evaluation_service.py          # Core evaluation logic
├── docs/
│   ├── evaluation_guide.md           # Detailed usage guide
│   └── RAG_EVALUATION_README.md      # This file
├── evaluate_rag.py                   # Main evaluation CLI
├── setup_legalbench_data.py          # Dataset management
└── tests/
    └── test_local_rag_run.py         # Your existing RAG pipeline

run_rag_evaluation.sh                 # Easy-to-use demo script
```

## Advanced Usage

### Custom Test Cases

```python
# Create domain-specific test cases
custom_cases = [
    {
        "query": "What is the termination notice period?",
        "expected_keywords": ["termination", "notice", "30 days"],
        "document_name": "employment_contract.txt"
    }
]
```

### Batch Evaluation

```bash
# Test different configurations
for chunk_size in 400 600 800; do
    echo "Testing chunk size: $chunk_size"
    # Update your configuration
    python backend/evaluate_rag.py --mode quick --output "results/chunk_${chunk_size}.json"
done
```

### Integration with CI/CD

Add evaluation to your development workflow:

```yaml
# .github/workflows/eval.yml
- name: Run RAG Evaluation
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    cd backend
    python evaluate_rag.py --mode quick --output results/ci_eval.json
```

## Available Commands

### Evaluation Commands

```bash
# Quick evaluation
python backend/evaluate_rag.py --mode quick

# Full evaluation with specific benchmark
python backend/evaluate_rag.py --mode full --benchmark-file cuad.json

# Limit test cases for faster testing
python backend/evaluate_rag.py --mode full --max-cases 10

# Custom output location
python backend/evaluate_rag.py --mode quick --output my_results.json
```

### Dataset Management Commands

```bash
# Check dataset status
python backend/setup_legalbench_data.py --info

# Create sample dataset
python backend/setup_legalbench_data.py --create-sample

# Validate existing dataset
python backend/setup_legalbench_data.py --validate

# Download full dataset (when available)
python backend/setup_legalbench_data.py --download
```

### Demo Script Commands

```bash
# Basic demo
./run_rag_evaluation.sh

# Full demo with dataset setup
./run_rag_evaluation.sh --full-demo

# Specific mode
./run_rag_evaluation.sh --mode full

# Help
./run_rag_evaluation.sh --help
```

## Troubleshooting

### Common Issues

**❌ Missing API Key**

```bash
export OPENAI_API_KEY="your-key-here"
```

**❌ Missing Dependencies**

```bash
cd backend
uv pip install faiss-cpu langchain-openai requests
```

**❌ Virtual Environment Issues**

```bash
cd backend
uv venv
uv pip install -r requirements.txt
```

**❌ Low Scores**

- Try different chunk sizes (400-800 tokens)
- Experiment with different embedding models
- Increase retrieval count (k parameter)
- Improve document preprocessing

### Getting Help

1. **Check logs**: Use `--verbose` flag for detailed output
2. **Review documentation**: See `backend/docs/evaluation_guide.md`
3. **Examine results**: Look at detailed results JSON for patterns
4. **Performance tuning**: Experiment with different configurations

## What Makes This Evaluation Special

### 🎯 Domain-Specific

- Designed specifically for legal document RAG systems
- Tests complex legal reasoning and understanding
- Uses real legal contracts and agreements

### 📏 Precise Measurement

- Character-level ground truth matching
- Deterministic precision and recall calculation
- Industry-standard benchmark for comparison

### ⚡ Easy to Use

- One-command evaluation with `./run_rag_evaluation.sh`
- Clear, interpretable results
- Both quick testing and comprehensive benchmarking

### 🔧 Flexible

- Works with your existing RAG pipeline
- Supports custom test cases
- Multiple evaluation modes for different needs

## Next Steps

1. **Run your first evaluation**: `./run_rag_evaluation.sh`
2. **Analyze results**: Look for areas of improvement
3. **Optimize your system**: Try different configurations
4. **Set up regular evaluation**: Add to your development workflow
5. **Create custom test cases**: For your specific legal domain

## References

- **LegalBench-RAG Paper**: https://arxiv.org/abs/2408.10343
- **LegalBench-RAG GitHub**: https://github.com/zeroentropy-ai/legalbenchrag
- **Detailed Guide**: `backend/docs/evaluation_guide.md`

---

**Ready to evaluate your RAG system? Start with:**

```bash
./run_rag_evaluation.sh
```
