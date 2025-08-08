"""
LegalBench-RAG Evaluation Service

This service provides comprehensive evaluation capabilities for RAG systems using the LegalBench-RAG benchmark.
It measures both retrieval quality (precision/recall) and generation quality.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import statistics

# Our real application services (import from backend)
import sys
import os
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Set minimal environment variables to avoid settings issues
os.environ.setdefault("ALLOWED_ORIGINS", "[]")
os.environ.setdefault("DATABASE_URL", "sqlite:///temp.db")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "dummy")
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "https://dummy.openai.azure.com/")

from app.services.enhanced_document_service import EnhancedDocumentService
from app.services.enhanced_vectorstore import EnhancedVectorStore
from app.models.langchain_models import EnhancedDocument
from app.utils.chunk_visualizer import print_chunks_before_openai, ChunkVisualizer

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics for a single test case evaluation."""
    precision: float
    recall: float
    f1_score: float
    exact_match: float
    response_time: float
    retrieved_docs_count: int
    ground_truth_count: int
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass
class BenchmarkResult:
    """Results for an entire benchmark evaluation."""
    benchmark_name: str
    total_test_cases: int
    avg_precision: float
    avg_recall: float
    avg_f1_score: float
    avg_exact_match: float
    avg_response_time: float
    detailed_results: List[Dict[str, Any]]
    error_count: int
    success_rate: float


@dataclass
class GroundTruthSnippet:
    """Represents a ground truth snippet from LegalBench-RAG."""
    file_path: str
    start_char: int
    end_char: int
    text: str


@dataclass
class TestCase:
    """Represents a test case from LegalBench-RAG."""
    query: str
    ground_truth_snippets: List[GroundTruthSnippet]
    expected_answer: Optional[str] = None


class LegalBenchRAGEvaluator:
    """
    Evaluator for RAG systems using LegalBench-RAG benchmark.

    This evaluator can measure:
    1. Retrieval Quality: How well the system retrieves relevant documents
    2. Generation Quality: How well the system generates answers
    3. End-to-End Performance: Overall system effectiveness
    """

    def __init__(
        self,
        document_service: EnhancedDocumentService,
        vector_store: EnhancedVectorStore,
        llm_service: Any = None,
        corpus_path: str = "data/corpus",
        benchmarks_path: str = "data/benchmarks"
    ):
        self.document_service = document_service
        self.vector_store = vector_store
        self.llm_service = llm_service

        # Handle relative paths by calculating from project root
        if not Path(corpus_path).is_absolute():
            project_root = Path(__file__).parent.parent.parent
            self.corpus_path = project_root / corpus_path
        else:
            self.corpus_path = Path(corpus_path)

        if not Path(benchmarks_path).is_absolute():
            project_root = Path(__file__).parent.parent.parent
            self.benchmarks_path = project_root / benchmarks_path
        else:
            self.benchmarks_path = Path(benchmarks_path)

        # Track loaded documents for character-level matching
        self.corpus_documents: Dict[str, str] = {}
        self.processed_documents: List[EnhancedDocument] = []

    async def setup_corpus(self) -> bool:
        """
        Load and process the LegalBench-RAG corpus.

        Returns:
            bool: True if corpus was successfully loaded and processed
        """
        logger.info("Setting up LegalBench-RAG corpus...")

        if not self.corpus_path.exists():
            logger.error(f"Corpus path not found: {self.corpus_path}")
            return False

        # Load all text files from corpus
        text_files = list(self.corpus_path.rglob("*.txt"))
        logger.info(f"Found {len(text_files)} text files in corpus")

        if not text_files:
            logger.error("No text files found in corpus")
            return False

        # Load file contents for character-level matching
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Store relative path as key for matching with benchmark references
                relative_path = str(file_path.relative_to(self.corpus_path))
                self.corpus_documents[relative_path] = content

                logger.debug(f"Loaded {relative_path}: {len(content)} characters")

            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {len(self.corpus_documents)} corpus documents")

        # Process documents through RAG pipeline
        logger.info("Processing documents through RAG pipeline...")
        processing_start = time.time()

        for file_path in text_files:
            try:
                # Create mock upload file
                from tests.test_local_rag_run import MockUploadFile
                mock_file = MockUploadFile(file_path)

                # Process through enhanced document service
                enhanced_doc = await self.document_service.process_document_enhanced(
                    mock_file,
                    use_enhanced_models=True,
                    preserve_structure=True
                )

                # Store in vector store
                await self.vector_store.store_enhanced_document(enhanced_doc)
                self.processed_documents.append(enhanced_doc)

                logger.debug(f"Processed {file_path.name}: {len(enhanced_doc.chunks)} chunks")

            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")

        processing_time = time.time() - processing_start
        logger.info(f"Processed {len(self.processed_documents)} documents in {processing_time:.2f}s")

        return len(self.processed_documents) > 0

    def load_benchmark(self, benchmark_file: str) -> List[TestCase]:
        """
        Load a benchmark file and parse test cases.

        Args:
            benchmark_file: Name of the benchmark JSON file

        Returns:
            List of test cases
        """
        benchmark_path = self.benchmarks_path / benchmark_file

        if not benchmark_path.exists():
            logger.error(f"Benchmark file not found: {benchmark_path}")
            return []

        try:
            with open(benchmark_path, 'r', encoding='utf-8') as f:
                benchmark_data = json.load(f)

            test_cases = []

            # Parse test cases based on LegalBench-RAG format
            for case_data in benchmark_data.get('test_cases', []):
                query = case_data.get('query', '')
                ground_truth_data = case_data.get('ground_truth', [])

                # Parse ground truth snippets
                ground_truth_snippets = []
                for gt_data in ground_truth_data:
                    file_path = gt_data.get('file_path', '')
                    start_char = gt_data.get('start_char', 0)
                    end_char = gt_data.get('end_char', 0)

                    # Extract text from corpus
                    text = ""
                    if file_path in self.corpus_documents:
                        text = self.corpus_documents[file_path][start_char:end_char]

                    snippet = GroundTruthSnippet(
                        file_path=file_path,
                        start_char=start_char,
                        end_char=end_char,
                        text=text
                    )
                    ground_truth_snippets.append(snippet)

                test_case = TestCase(
                    query=query,
                    ground_truth_snippets=ground_truth_snippets,
                    expected_answer=case_data.get('expected_answer')
                )
                test_cases.append(test_case)

            logger.info(f"Loaded {len(test_cases)} test cases from {benchmark_file}")
            return test_cases

        except Exception as e:
            logger.error(f"Failed to load benchmark {benchmark_file}: {e}")
            return []

    async def evaluate_retrieval(self, query: str, ground_truth_snippets: List[GroundTruthSnippet], k: int = 10) -> EvaluationMetrics:
        """
        Evaluate retrieval quality for a single query.

        Args:
            query: The search query
            ground_truth_snippets: Expected relevant snippets
            k: Number of documents to retrieve

        Returns:
            Evaluation metrics
        """
        start_time = time.time()

        # Retrieve documents
        search_results = await self.vector_store.search_documents(query, k=k)
        response_time = time.time() - start_time

        # Convert ground truth to set of unique texts for matching
        ground_truth_texts = set()
        for snippet in ground_truth_snippets:
            if snippet.text.strip():
                ground_truth_texts.add(snippet.text.strip())

        # Extract retrieved texts
        retrieved_texts = set()
        for result in search_results:
            if result.get('content', '').strip():
                retrieved_texts.add(result['content'].strip())

        # Calculate metrics
        true_positives = len(ground_truth_texts.intersection(retrieved_texts))
        false_positives = len(retrieved_texts) - true_positives
        false_negatives = len(ground_truth_texts) - true_positives

        precision = true_positives / len(retrieved_texts) if retrieved_texts else 0.0
        recall = true_positives / len(ground_truth_texts) if ground_truth_texts else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Exact match: check if all ground truth snippets were retrieved
        exact_match = 1.0 if false_negatives == 0 and len(ground_truth_texts) > 0 else 0.0

        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            exact_match=exact_match,
            response_time=response_time,
            retrieved_docs_count=len(retrieved_texts),
            ground_truth_count=len(ground_truth_texts),
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives
        )

    async def evaluate_generation(self, query: str, context: str, expected_answer: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate generation quality for a single query.

        Args:
            query: The question
            context: Retrieved context
            expected_answer: Expected answer (if available)

        Returns:
            Generation evaluation metrics
        """
        if not self.llm_service:
            return {"error": "No LLM service available for generation evaluation"}

        start_time = time.time()

        # Generate answer
        prompt = f"""Based on the following context, provide a comprehensive answer to the question.

Question: {query}

Context:
{context}

Please provide a detailed answer based on the information in the context:"""

        try:
            response = await self.llm_service.ainvoke(prompt)
            generated_answer = response.content
            response_time = time.time() - start_time

            # Basic quality metrics
            metrics = {
                "generated_answer": generated_answer,
                "response_time": response_time,
                "answer_length": len(generated_answer),
                "has_answer": len(generated_answer.strip()) > 0
            }

            # If expected answer is provided, calculate similarity
            if expected_answer:
                # Simple similarity based on common words (can be enhanced)
                expected_words = set(expected_answer.lower().split())
                generated_words = set(generated_answer.lower().split())

                if expected_words:
                    word_overlap = len(expected_words.intersection(generated_words))
                    similarity = word_overlap / len(expected_words)
                    metrics["answer_similarity"] = similarity

            return metrics

        except Exception as e:
            return {
                "error": str(e),
                "response_time": time.time() - start_time,
                "generated_answer": "",
                "has_answer": False
            }

    async def evaluate_test_case(self, test_case: TestCase, k: int = 10) -> Dict[str, Any]:
        """
        Evaluate a single test case end-to-end.

        Args:
            test_case: The test case to evaluate
            k: Number of documents to retrieve

        Returns:
            Comprehensive evaluation results
        """
        logger.debug(f"Evaluating query: {test_case.query[:100]}...")

        # Evaluate retrieval
        retrieval_metrics = await self.evaluate_retrieval(
            test_case.query,
            test_case.ground_truth_snippets,
            k=k
        )

        # Get retrieved context for generation evaluation
        search_results = await self.vector_store.search_documents(test_case.query, k=3)

        # 🌳 Visualize retrieved chunks before sending to OpenAI
        if search_results:
            print(f"\n" + "="*80)
            visualizer = ChunkVisualizer()
            # Convert search results to chunks for visualization
            chunk_objects = []
            for i, result in enumerate(search_results[:3]):
                # Create a mock chunk object with content and metadata
                chunk = type('RetrievedChunk', (), {
                    'page_content': result.get('content', ''),
                    'metadata': {
                        'source': result.get('source', 'unknown'),
                        'score': result.get('score', 0.0),
                        'chunk_index': i,
                        'retrieval_rank': i + 1,
                        **result.get('metadata', {})
                    }
                })()
                chunk_objects.append(chunk)

            visualizer.print_chunks(
                chunk_objects,
                f"Retrieved Chunks for Query: '{test_case.query[:50]}...' → OpenAI LLM",
                show_before_openai=True
            )
            print("="*80 + "\n")

        context = "\n\n".join([result.get('content', '') for result in search_results[:3]])

        # Evaluate generation if LLM is available
        generation_metrics = {}
        if self.llm_service:
            generation_metrics = await self.evaluate_generation(
                test_case.query,
                context,
                test_case.expected_answer
            )

        return {
            "query": test_case.query,
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": generation_metrics,
            "ground_truth_count": len(test_case.ground_truth_snippets),
            "retrieved_context_length": len(context)
        }

    async def run_benchmark(self, benchmark_file: str, k: int = 10, max_test_cases: Optional[int] = None) -> BenchmarkResult:
        """
        Run a complete benchmark evaluation.

        Args:
            benchmark_file: Name of the benchmark JSON file
            k: Number of documents to retrieve per query
            max_test_cases: Maximum number of test cases to run (for testing)

        Returns:
            Comprehensive benchmark results
        """
        logger.info(f"Running benchmark: {benchmark_file}")

        # Load test cases
        test_cases = self.load_benchmark(benchmark_file)
        if not test_cases:
            logger.error("No test cases loaded")
            return BenchmarkResult(
                benchmark_name=benchmark_file,
                total_test_cases=0,
                avg_precision=0.0,
                avg_recall=0.0,
                avg_f1_score=0.0,
                avg_exact_match=0.0,
                avg_response_time=0.0,
                detailed_results=[],
                error_count=1,
                success_rate=0.0
            )

        # Limit test cases if specified
        if max_test_cases:
            test_cases = test_cases[:max_test_cases]
            logger.info(f"Limited to {len(test_cases)} test cases for evaluation")

        # Run evaluations
        detailed_results = []
        error_count = 0
        start_time = time.time()

        for i, test_case in enumerate(test_cases, 1):
            try:
                logger.info(f"Evaluating test case {i}/{len(test_cases)}")
                result = await self.evaluate_test_case(test_case, k=k)
                detailed_results.append(result)

            except Exception as e:
                logger.error(f"Failed to evaluate test case {i}: {e}")
                error_count += 1
                detailed_results.append({
                    "query": test_case.query,
                    "error": str(e),
                    "retrieval_metrics": None,
                    "generation_metrics": None
                })

        total_time = time.time() - start_time

        # Calculate aggregate metrics
        successful_results = [r for r in detailed_results if r.get('retrieval_metrics')]
        success_count = len(successful_results)

        if successful_results:
            avg_precision = statistics.mean([r['retrieval_metrics'].precision for r in successful_results])
            avg_recall = statistics.mean([r['retrieval_metrics'].recall for r in successful_results])
            avg_f1_score = statistics.mean([r['retrieval_metrics'].f1_score for r in successful_results])
            avg_exact_match = statistics.mean([r['retrieval_metrics'].exact_match for r in successful_results])
            avg_response_time = statistics.mean([r['retrieval_metrics'].response_time for r in successful_results])
        else:
            avg_precision = avg_recall = avg_f1_score = avg_exact_match = avg_response_time = 0.0

        success_rate = success_count / len(test_cases) if test_cases else 0.0

        result = BenchmarkResult(
            benchmark_name=benchmark_file,
            total_test_cases=len(test_cases),
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f1_score=avg_f1_score,
            avg_exact_match=avg_exact_match,
            avg_response_time=avg_response_time,
            detailed_results=detailed_results,
            error_count=error_count,
            success_rate=success_rate
        )

        logger.info(f"Benchmark completed in {total_time:.2f}s")
        logger.info(f"Success rate: {success_rate:.2%}")
        logger.info(f"Average F1: {avg_f1_score:.3f}")
        logger.info(f"Average Precision: {avg_precision:.3f}")
        logger.info(f"Average Recall: {avg_recall:.3f}")

        return result

    def save_results(self, result: BenchmarkResult, output_file: str) -> None:
        """Save benchmark results to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclass to dict for JSON serialization
        results_dict = {
            "benchmark_name": result.benchmark_name,
            "total_test_cases": result.total_test_cases,
            "avg_precision": result.avg_precision,
            "avg_recall": result.avg_recall,
            "avg_f1_score": result.avg_f1_score,
            "avg_exact_match": result.avg_exact_match,
            "avg_response_time": result.avg_response_time,
            "error_count": result.error_count,
            "success_rate": result.success_rate,
            "detailed_results": []
        }

        # Convert detailed results
        for detail in result.detailed_results:
            detail_dict = {
                "query": detail.get("query", ""),
                "ground_truth_count": detail.get("ground_truth_count", 0),
                "retrieved_context_length": detail.get("retrieved_context_length", 0)
            }

            # Add retrieval metrics if available
            if detail.get("retrieval_metrics"):
                rm = detail["retrieval_metrics"]
                detail_dict["retrieval_metrics"] = {
                    "precision": rm.precision,
                    "recall": rm.recall,
                    "f1_score": rm.f1_score,
                    "exact_match": rm.exact_match,
                    "response_time": rm.response_time,
                    "retrieved_docs_count": rm.retrieved_docs_count,
                    "ground_truth_count": rm.ground_truth_count,
                    "true_positives": rm.true_positives,
                    "false_positives": rm.false_positives,
                    "false_negatives": rm.false_negives
                }

            # Add generation metrics if available
            if detail.get("generation_metrics"):
                detail_dict["generation_metrics"] = detail["generation_metrics"]

            results_dict["detailed_results"].append(detail_dict)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_path}")


class QuickEvaluator:
    """
    Simplified evaluator for quick testing without downloading the full LegalBench-RAG dataset.
    Creates synthetic legal test cases to demonstrate evaluation capabilities.
    """

    def __init__(self, document_service: EnhancedDocumentService, vector_store: EnhancedVectorStore, llm_service: Any = None, data_directory: str = "data"):
        self.document_service = document_service
        self.vector_store = vector_store
        self.llm_service = llm_service

        # Set up paths for benchmark data (same as LegalBenchRAGEvaluator)
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        self.corpus_path = project_root / "data" / "corpus"
        self.benchmarks_path = project_root / "data" / "benchmarks"

        # Set up data directory path for analyzing actual documents
        if not Path(data_directory).is_absolute():
            self.data_directory = project_root / data_directory
        else:
            self.data_directory = Path(data_directory)

    async def create_sample_test_cases(self) -> List[Dict[str, Any]]:
        """Create test cases from real LegalBench-RAG benchmark data or analyze actual documents."""
        try:
            # First, try to analyze actual documents in the data directory
            document_based_questions = await self._create_test_cases_from_actual_documents()

            if document_based_questions:
                logger.info(f"Created {len(document_based_questions)} test cases based on actual documents in {self.data_directory}")
                return document_based_questions

            # Try to load real benchmark questions from the downloaded LegalBench-RAG dataset
            benchmark_questions = self._load_benchmark_questions_for_documents()

            if benchmark_questions:
                logger.info(f"Loaded {len(benchmark_questions)} real benchmark questions from LegalBench-RAG dataset")
                return benchmark_questions

        except Exception as e:
            logger.warning(f"Could not load real benchmark questions: {e}, falling back to generic")

        # Fallback to generic legal document questions that work across many document types
        logger.info("Using fallback generic legal questions")
        return [
            {
                "query": "What are the main parties involved in this agreement?",
                "expected_keywords": ["party", "parties", "company", "corporation", "agreement"],
                "document_name": "legal_document"
            },
            {
                "query": "What are the key terms and definitions?",
                "expected_keywords": ["definition", "terms", "means", "shall", "defined"],
                "document_name": "legal_document"
            },
            {
                "query": "What are the main obligations and responsibilities?",
                "expected_keywords": ["obligation", "responsibility", "shall", "must", "required"],
                "document_name": "legal_document"
            },
            {
                "query": "What are the termination or expiration conditions?",
                "expected_keywords": ["termination", "terminate", "expiration", "expire", "end"],
                "document_name": "legal_document"
            },
            {
                "query": "What governing law and jurisdiction apply?",
                "expected_keywords": ["governing", "law", "jurisdiction", "court", "state"],
                "document_name": "legal_document"
            }
        ]

    async def _create_test_cases_from_actual_documents(self) -> List[Dict[str, Any]]:
        """Analyze actual documents in the data directory and create appropriate test cases."""
        if not self.data_directory.exists():
            logger.warning(f"Data directory {self.data_directory} does not exist")
            return []

        # Find all document files in the data directory
        document_files = []
        for ext in ['.txt', '.md', '.pdf', '.docx']:
            document_files.extend(list(self.data_directory.glob(f'**/*{ext}')))

        if not document_files:
            logger.warning(f"No document files found in {self.data_directory}")
            return []

        logger.info(f"Found {len(document_files)} documents in {self.data_directory}")

        # Read and analyze document content
        all_content = ""
        document_names = []

        for doc_file in document_files[:3]:  # Limit to first 3 files for performance
            try:
                if doc_file.suffix.lower() == '.txt':
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        content = f.read()[:10000]  # First 10k chars to avoid memory issues
                        all_content += content + " "
                        document_names.append(doc_file.name)
                        logger.info(f"Analyzed document: {doc_file.name} ({len(content)} chars)")
            except Exception as e:
                logger.warning(f"Could not read {doc_file}: {e}")
                continue

        if not all_content:
            logger.warning("No document content could be read")
            return []

        # Try to match with benchmark questions based on document content
        matched_questions = self._match_benchmark_questions_to_content(all_content, document_names)

        if matched_questions:
            return matched_questions

        # Fallback to dynamic test case creation based on content
        return self._create_dynamic_test_cases([all_content])

    def _match_benchmark_questions_to_content(self, content: str, document_names: List[str]) -> List[Dict[str, Any]]:
        """Match benchmark questions to document content based on entity recognition."""
        import json

        content_lower = content.lower()
        test_cases = []

        # Check for specific entities mentioned in benchmark questions
        # Michaels Companies / Magic AcquireCo merger (MAUD dataset)
        if any(entity in content_lower for entity in ['michaels', 'magic acquireco', 'magic mergeco', 'apollo']):
            logger.info("Detected Michaels/Magic merger document - loading MAUD questions")

            # Load MAUD benchmark questions
            maud_file = self.benchmarks_path / 'maud.json'
            if maud_file.exists():
                try:
                    with open(maud_file, 'r', encoding='utf-8') as f:
                        maud_data = json.load(f)

                    # Extract relevant questions for Michaels/Magic merger
                    if 'tests' in maud_data:
                        for test in maud_data['tests'][:20]:  # First 20 questions
                            if 'query' in test and any(entity in test['query'].lower() for entity in ['michaels', 'magic']):
                                # Extract keywords from the query and context
                                query_words = [word.lower().strip('.,;:') for word in test['query'].split() if len(word) > 3]
                                keywords = [word for word in query_words if word not in ['consider', 'what', 'are', 'the', 'this', 'that', 'with', 'from', 'agreement']]

                                # Add specific legal terms based on query content
                                if 'consideration' in test['query'].lower():
                                    keywords.extend(['merger', 'consideration', 'cash', 'stock', 'shares'])
                                elif 'covenant' in test['query'].lower():
                                    keywords.extend(['covenant', 'ordinary', 'course', 'business'])
                                elif 'termination' in test['query'].lower():
                                    keywords.extend(['termination', 'fiduciary', 'board'])
                                elif 'definition' in test['query'].lower():
                                    keywords.extend(['definition', 'means', 'affiliate', 'subsidiary'])

                                test_cases.append({
                                    "query": test['query'],
                                    "expected_keywords": list(set(keywords))[:10],
                                    "document_name": f"merger_agreement ({', '.join(document_names)})",
                                    "benchmark_source": "maud"
                                })

                                if len(test_cases) >= 20:  # Limit to 20 questions
                                    break

                    logger.info(f"Loaded {len(test_cases)} MAUD questions for Michaels/Magic merger")

                except Exception as e:
                    logger.warning(f"Failed to load MAUD benchmark: {e}")

        return test_cases

    def _create_dynamic_test_cases(self, sample_docs: List[str]) -> List[Dict[str, Any]]:
        """Create test cases based on actual document content."""
        # Analyze the content to determine document type and create relevant questions
        combined_content = " ".join(sample_docs).lower()

        # Detect document type based on content
        if any(keyword in combined_content for keyword in ["merger", "acquisition", "apollo", "michaels"]):
            # This appears to be a merger agreement
            return [
                {
                    "query": "Who are the parties to this merger agreement?",
                    "expected_keywords": ["Michaels", "Apollo", "Company", "Buyer", "parties"],
                    "document_name": "merger_agreement"
                },
                {
                    "query": "What is the structure of the merger transaction?",
                    "expected_keywords": ["merger", "transaction", "structure", "effective time", "surviving"],
                    "document_name": "merger_agreement"
                },
                {
                    "query": "What are the key definitions in this agreement?",
                    "expected_keywords": ["affiliate", "person", "subsidiary", "control", "agreement"],
                    "document_name": "merger_agreement"
                },
                {
                    "query": "What conditions must be satisfied for the merger to close?",
                    "expected_keywords": ["conditions", "closing", "satisfaction", "approval", "consent"],
                    "document_name": "merger_agreement"
                },
                {
                    "query": "What are the representations and warranties of the parties?",
                    "expected_keywords": ["representations", "warranties", "true", "correct", "material"],
                    "document_name": "merger_agreement"
                }
            ]

        elif any(keyword in combined_content for keyword in ["contract", "services", "payment"]):
            # This appears to be a service/commercial contract
            return [
                {
                    "query": "What services are being provided under this contract?",
                    "expected_keywords": ["services", "provide", "perform", "deliverables"],
                    "document_name": "service_contract"
                },
                {
                    "query": "What are the payment terms and amounts?",
                    "expected_keywords": ["payment", "fee", "amount", "invoice", "due"],
                    "document_name": "service_contract"
                },
                {
                    "query": "What are the contract duration and termination provisions?",
                    "expected_keywords": ["term", "duration", "termination", "notice", "expire"],
                    "document_name": "service_contract"
                }
            ]

        # Default to generic legal questions if we can't identify the type
        return self.create_sample_test_cases()

    def _load_benchmark_questions_for_documents(self) -> List[Dict[str, Any]]:
        """Load real benchmark questions from LegalBench-RAG dataset for available documents."""
        import json
        from pathlib import Path

        # Check for benchmark files in the data directory
        benchmark_files = {
            'maud': self.benchmarks_path / 'maud.json',
            'cuad': self.benchmarks_path / 'cuad.json',
            'contractnli': self.benchmarks_path / 'contractnli.json',
            'privacy_qa': self.benchmarks_path / 'privacy_qa.json'
        }

        test_cases = []

        for benchmark_name, benchmark_file in benchmark_files.items():
            if not benchmark_file.exists():
                continue

            try:
                with open(benchmark_file, 'r', encoding='utf-8') as f:
                    benchmark_data = json.load(f)

                # Extract questions for documents we have
                if 'tests' in benchmark_data:
                    for test in benchmark_data['tests'][:10]:  # Limit to first 10 questions per benchmark
                        if 'query' in test and 'snippets' in test:
                            # Extract file names mentioned in snippets
                            file_paths = set()
                            ground_truth_snippets = []

                            for snippet in test['snippets']:
                                if 'file_path' in snippet:
                                    file_paths.add(snippet['file_path'])

                                    # Create ground truth snippet for evaluation
                                    snippet_text = snippet.get('answer', '')
                                    if snippet_text:
                                        ground_truth_snippets.append(snippet_text)

                            # Check if we have any of these documents in our corpus
                            has_document = False
                            for file_path in file_paths:
                                doc_name = Path(file_path).name
                                if (self.corpus_path / benchmark_name / doc_name).exists():
                                    has_document = True
                                    break

                            if has_document and ground_truth_snippets:
                                # Extract keywords from the ground truth answer
                                all_text = " ".join(ground_truth_snippets).lower()
                                keywords = []

                                # Extract important terms (simple keyword extraction)
                                import re
                                # Extract capitalized terms and important legal terms
                                capitalized_terms = re.findall(r'\b[A-Z][a-zA-Z]+\b', " ".join(ground_truth_snippets))
                                keywords.extend([term.lower() for term in capitalized_terms[:5]])

                                # Add some query terms as keywords too
                                query_words = [word.lower() for word in test['query'].split() if len(word) > 3]
                                keywords.extend(query_words[:3])

                                test_cases.append({
                                    "query": test['query'],
                                    "expected_keywords": list(set(keywords))[:10],  # Limit keywords
                                    "ground_truth_snippets": ground_truth_snippets,
                                    "document_name": f"{benchmark_name}_benchmark",
                                    "benchmark_source": benchmark_name
                                })

                logger.info(f"Loaded {len([tc for tc in test_cases if tc.get('benchmark_source') == benchmark_name])} questions from {benchmark_name} benchmark")

            except Exception as e:
                logger.warning(f"Failed to load benchmark {benchmark_name}: {e}")

        return test_cases[:20]  # Limit total questions for quick evaluation

    async def run_quick_evaluation(self, max_results: int = 5) -> Dict[str, Any]:
        """
        Run a quick evaluation using sample test cases.

        Args:
            max_results: Maximum number of results to retrieve per query

        Returns:
            Evaluation results
        """
        logger.info("Running quick evaluation with sample test cases...")

        test_cases = await self.create_sample_test_cases()
        results = []

        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"Evaluating test case {i}/{len(test_cases)}: {test_case['query']}")

            start_time = time.time()

            # Search for relevant documents
            search_results = await self.vector_store.search_documents(
                test_case["query"],
                k=max_results
            )

            retrieval_time = time.time() - start_time

            # Check if expected keywords are found in results
            found_keywords = []
            all_content = " ".join([result.get('content', '') for result in search_results])

            for keyword in test_case["expected_keywords"]:
                if keyword.lower() in all_content.lower():
                    found_keywords.append(keyword)

            keyword_coverage = len(found_keywords) / len(test_case["expected_keywords"]) if test_case["expected_keywords"] else 0.0

            # Generate answer if LLM is available
            generated_answer = None
            generation_time = 0.0

            if self.llm_service and search_results:
                gen_start = time.time()

                # 🌳 Visualize retrieved chunks before sending to OpenAI
                print(f"\n" + "="*80)
                visualizer = ChunkVisualizer()
                # Convert search results to chunks for visualization
                chunk_objects = []
                for idx, result in enumerate(search_results[:3]):
                    # Create a mock chunk object with content and metadata
                    chunk = type('RetrievedChunk', (), {
                        'page_content': result.get('content', ''),
                        'metadata': {
                            'source': result.get('source', 'unknown'),
                            'score': result.get('score', 0.0),
                            'chunk_index': idx,
                            'retrieval_rank': idx + 1,
                            **result.get('metadata', {})
                        }
                    })()
                    chunk_objects.append(chunk)

                visualizer.print_chunks(
                    chunk_objects,
                    f"Retrieved Chunks for Query: '{test_case['query'][:50]}...' → OpenAI LLM",
                    show_before_openai=True
                )
                print("="*80 + "\n")

                context = "\n\n".join([result.get('content', '') for result in search_results[:3]])

                prompt = f"""Based on the following context, provide a comprehensive answer to the question.

Question: {test_case['query']}

Context:
{context}

Please provide a detailed answer:"""

                try:
                    response = await self.llm_service.ainvoke(prompt)
                    generated_answer = response.content
                    generation_time = time.time() - gen_start
                except Exception as e:
                    generated_answer = f"Error: {str(e)}"
                    generation_time = time.time() - gen_start

            # Calculate relevance score based on content relevance and retrieval quality
            relevance_score = 0.0
            if search_results:
                # Method 1: Use ground truth snippets if available (for real benchmark questions)
                if "ground_truth_snippets" in test_case and test_case["ground_truth_snippets"]:
                    # Calculate how well retrieved content matches ground truth
                    ground_truth_text = " ".join(test_case["ground_truth_snippets"]).lower()
                    retrieved_text = " ".join([r.get('content', '') for r in search_results]).lower()

                    # Simple word overlap relevance score
                    ground_truth_words = set(ground_truth_text.split())
                    retrieved_words = set(retrieved_text.split())

                    if ground_truth_words:
                        word_overlap = len(ground_truth_words.intersection(retrieved_words))
                        relevance_score = min(1.0, word_overlap / len(ground_truth_words))
                else:
                    # Method 2: Use similarity scores from FAISS (fallback for synthetic questions)
                    similarities = [r.get('similarity', 0.0) for r in search_results]
                    if similarities:
                        # Average similarity score as relevance (FAISS returns cosine similarity)
                        relevance_score = sum(similarities) / len(similarities)

            result = {
                "query": test_case["query"],
                "expected_document": test_case["document_name"],
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "retrieved_count": len(search_results),
                "keyword_coverage": keyword_coverage,
                "found_keywords": found_keywords,
                "missing_keywords": [k for k in test_case["expected_keywords"] if k not in found_keywords],
                "relevance_score": relevance_score,
                "generated_answer": generated_answer,
                "top_sources": [
                    {
                        "document": r.get('document_name', ''),
                        "similarity": r.get('similarity', 0.0),
                        "content_preview": r.get('content', '')[:200] + "..." if len(r.get('content', '')) > 200 else r.get('content', '')
                    }
                    for r in search_results[:3]
                ]
            }

            results.append(result)

        # Calculate aggregate metrics
        avg_retrieval_time = statistics.mean([r['retrieval_time'] for r in results])
        avg_generation_time = statistics.mean([r['generation_time'] for r in results])
        avg_keyword_coverage = statistics.mean([r['keyword_coverage'] for r in results])
        avg_relevance_score = statistics.mean([r['relevance_score'] for r in results])

        summary = {
            "total_test_cases": len(test_cases),
            "avg_retrieval_time": avg_retrieval_time,
            "avg_generation_time": avg_generation_time,
            "avg_keyword_coverage": avg_keyword_coverage,
            "avg_relevance_score": avg_relevance_score,
            "detailed_results": results
        }

        logger.info(f"Quick evaluation completed!")
        logger.info(f"Average keyword coverage: {avg_keyword_coverage:.2%}")
        logger.info(f"Average relevance score: {avg_relevance_score:.2%}")
        logger.info(f"Average retrieval time: {avg_retrieval_time:.3f}s")

        return summary
