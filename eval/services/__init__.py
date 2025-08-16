"""
RAG Evaluation Services

This module provides evaluation services for RAG systems using the LegalBench-RAG benchmark.
"""

from .evaluation_service import (
    LegalBenchRAGEvaluator,
    QuickEvaluator,
    EvaluationMetrics,
    BenchmarkResult,
    GroundTruthSnippet,
    TestCase
)

__all__ = [
    "LegalBenchRAGEvaluator",
    "QuickEvaluator",
    "EvaluationMetrics",
    "BenchmarkResult",
    "GroundTruthSnippet",
    "TestCase"
]
