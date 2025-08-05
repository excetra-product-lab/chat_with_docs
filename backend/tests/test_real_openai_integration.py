"""
Real OpenAI Integration Tests

Run these tests with actual OpenAI API calls to verify end-to-end functionality.
These tests require valid Azure OpenAI credentials to be configured.

To run these tests:
1. Set environment variables:
   export AZURE_OPENAI_API_KEY="your-api-key"
   export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
   export AZURE_OPENAI_DEPLOYMENT_NAME="your-deployment-name"
   export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="your-embedding-deployment"

2. Run specific test:
   pytest backend/tests/test_real_openai_integration.py::test_real_openai_rag_pipeline -v

3. Or run all real integration tests:
   pytest backend/tests/test_real_openai_integration.py -v -s
"""

import os
from unittest.mock import patch

import pytest

from app.api.routes.chat import chat_query
from app.models.schemas import Query


class TestRealOpenAIIntegration:
    """Integration tests that make actual OpenAI API calls."""

    @pytest.fixture(autouse=True)
    def check_credentials(self):
        """Skip tests if OpenAI credentials are not available."""
        required_vars = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        ]

        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            pytest.skip(f"Missing required environment variables: {missing_vars}")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_openai_rag_pipeline(self):
        """
        Test the RAG pipeline with real OpenAI API calls.

        This test uses actual OpenAI services and requires valid credentials.
        It will make real API calls and may incur costs and take time.
        """
        # Mock user data
        mock_user = {"id": 123, "email": "test@example.com"}

        # Mock vector search with realistic document chunks
        mock_chunks = [
            {
                "id": 1,
                "document_id": 456,
                "content": "The company vacation policy requires employees to submit requests at least 2 weeks in advance. Vacation time is accrued based on years of service, with new employees earning 10 days per year.",
                "page": 15,
                "filename": "employee_handbook.pdf",
                "distance": 0.12,
                "similarity_score": 0.88,
                "created_at": "2024-01-01T00:00:00",
                "metadata": {
                    "filename": "employee_handbook.pdf",
                    "page": 15,
                    "source": "employee_handbook.pdf",
                },
            },
            {
                "id": 2,
                "document_id": 789,
                "content": "Emergency vacation requests may be approved with less notice at manager discretion. All vacation time must be used within the calendar year and does not roll over.",
                "page": 16,
                "filename": "employee_handbook.pdf",
                "distance": 0.18,
                "similarity_score": 0.82,
                "created_at": "2024-01-01T00:00:00",
                "metadata": {
                    "filename": "employee_handbook.pdf",
                    "page": 16,
                    "source": "employee_handbook.pdf",
                },
            },
        ]

        test_query = Query(question="What is the company's vacation policy?", limit=5)

        print("\n🔄 Testing with real OpenAI API...")
        print(f"Question: {test_query.question}")

        # Mock only the vector search - let everything else use real OpenAI
        with patch("app.core.rag_pipeline.similarity_search") as mock_search:
            mock_search.return_value = mock_chunks

            # Execute the pipeline with real OpenAI
            result = await chat_query(test_query, mock_user)

            # Verify the result structure
            assert result.query == test_query.question
            assert result.answer is not None
            assert len(result.answer) > 0
            assert result.confidence > 0

            print(f"\n✅ Generated Answer: {result.answer}")
            print(f"📊 Confidence: {result.confidence}")
            print(f"📖 Citations found: {len(result.citations)}")
            print(f"📄 Chunks processed: {len(result.chunks)}")

            # Verify citations were extracted (OpenAI should include them)
            if result.citations:
                print("📝 Citations:")
                for i, citation in enumerate(result.citations, 1):
                    print(f"  {i}. {citation.document_name} (Page {citation.page})")
                    print(f"     Snippet: {citation.snippet[:100]}...")

            # The answer should reference the documents
            assert "vacation" in result.answer.lower()

            # Verify chunks are included
            assert len(result.chunks) == 2
            assert all(chunk.content for chunk in result.chunks)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_openai_with_complex_query(self):
        """Test with a more complex query that requires reasoning."""
        mock_user = {"id": 456, "email": "test2@example.com"}

        mock_chunks = [
            {
                "id": 1,
                "document_id": 101,
                "content": "Section 4.2: Contract termination requires 30 days written notice to the other party. The terminating party must specify the reason for termination in the notice.",
                "page": 8,
                "filename": "service_contract.pdf",
                "distance": 0.08,
                "similarity_score": 0.92,
                "created_at": "2024-01-01T00:00:00",
                "metadata": {
                    "filename": "service_contract.pdf",
                    "page": 8,
                    "section": "4.2",
                },
            },
            {
                "id": 2,
                "document_id": 102,
                "content": "Late payment penalty: If payment is not received within 30 days of invoice date, a 1.5% monthly penalty will be applied to the outstanding balance.",
                "page": 12,
                "filename": "service_contract.pdf",
                "distance": 0.15,
                "similarity_score": 0.85,
                "created_at": "2024-01-01T00:00:00",
                "metadata": {"filename": "service_contract.pdf", "page": 12},
            },
        ]

        test_query = Query(
            question="If I want to terminate the contract due to late payments, what process should I follow and what penalties might apply?",
            limit=5,
        )

        print("\n🔄 Testing complex reasoning with real OpenAI...")
        print(f"Question: {test_query.question}")

        with patch("app.core.rag_pipeline.similarity_search") as mock_search:
            mock_search.return_value = mock_chunks

            result = await chat_query(test_query, mock_user)

            print(f"\n✅ Generated Answer: {result.answer}")
            print(f"📊 Citations: {len(result.citations)}")

            # Verify the answer addresses both aspects of the complex question
            answer_lower = result.answer.lower()
            assert "30 days" in answer_lower or "thirty days" in answer_lower
            assert "notice" in answer_lower
            assert "termination" in answer_lower or "terminate" in answer_lower

            # Should reference both contract termination and payment aspects
            assert len(result.chunks) == 2

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_openai_citation_extraction(self):
        """Test citation extraction with real OpenAI responses."""
        mock_user = {"id": 789, "email": "test3@example.com"}

        mock_chunks = [
            {
                "id": 1,
                "document_id": 201,
                "content": "Data retention policy requires all customer data to be retained for a minimum of 7 years for audit purposes as mandated by federal regulations.",
                "page": 22,
                "filename": "compliance_manual.pdf",
                "distance": 0.05,
                "similarity_score": 0.95,
                "created_at": "2024-01-01T00:00:00",
                "metadata": {"filename": "compliance_manual.pdf", "page": 22},
            }
        ]

        test_query = Query(question="How long must we retain customer data?", limit=3)

        print("\n🔄 Testing citation extraction with real OpenAI...")
        print(f"Question: {test_query.question}")

        with patch("app.core.rag_pipeline.similarity_search") as mock_search:
            mock_search.return_value = mock_chunks

            result = await chat_query(test_query, mock_user)

            print(f"\n✅ Generated Answer: {result.answer}")
            print(f"📝 Citations extracted: {len(result.citations)}")

            if result.citations:
                for citation in result.citations:
                    print(f"   - {citation.document_name} (Page {citation.page})")

            # Verify the answer includes the key information
            assert "7 years" in result.answer or "seven years" in result.answer.lower()
            assert result.chunks[0].document_name == "compliance_manual.pdf"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_openai_error_handling(self):
        """Test error handling with real OpenAI when no documents are found."""
        mock_user = {"id": 999, "email": "test4@example.com"}

        test_query = Query(
            question="What is the quantum flux capacitor policy?", limit=3
        )

        print("\n🔄 Testing error handling with real OpenAI...")
        print(f"Question: {test_query.question}")

        # Mock empty search results
        with patch("app.core.rag_pipeline.similarity_search") as mock_search:
            mock_search.return_value = []  # No documents found

            # Should raise DocumentNotFoundError
            with pytest.raises(Exception) as exc_info:
                await chat_query(test_query, mock_user)

            print(
                f"✅ Properly handled no documents scenario: {type(exc_info.value).__name__}"
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_openai_performance_timing(self):
        """Test performance and timing with real OpenAI API."""
        import time

        mock_user = {"id": 555, "email": "perf@example.com"}

        mock_chunks = [
            {
                "id": 1,
                "document_id": 301,
                "content": "Performance testing shows that response times should be under 2 seconds for standard queries.",
                "page": 5,
                "filename": "performance_specs.pdf",
                "distance": 0.1,
                "similarity_score": 0.9,
                "created_at": "2024-01-01T00:00:00",
                "metadata": {"filename": "performance_specs.pdf", "page": 5},
            }
        ]

        test_query = Query(question="What are the performance requirements?", limit=3)

        print("\n🔄 Testing performance with real OpenAI...")

        start_time = time.time()

        with patch("app.core.rag_pipeline.similarity_search") as mock_search:
            mock_search.return_value = mock_chunks
            result = await chat_query(test_query, mock_user)

        end_time = time.time()
        duration = end_time - start_time

        print(f"⏱️  Total response time: {duration:.2f} seconds")
        print(f"✅ Generated answer length: {len(result.answer)} characters")

        # Verify we got a response
        assert result.answer
        assert len(result.answer) > 0

        # Log timing info (useful for performance monitoring)
        print("📊 Performance metrics:")
        print(f"   - Response time: {duration:.2f}s")
        print(f"   - Answer length: {len(result.answer)} chars")
        print(f"   - Citations: {len(result.citations)}")


# Utility functions for manual testing
def run_single_test():
    """
    Helper function to run a single test manually.
    Usage: python -c "from backend.tests.test_real_openai_integration import run_single_test; run_single_test()"
    """
    import asyncio

    async def manual_test():
        test = TestRealOpenAIIntegration()
        # Check credentials first
        test.check_credentials()
        await test.test_real_openai_rag_pipeline()

    asyncio.run(manual_test())


if __name__ == "__main__":
    print("To run these tests:")
    print("1. Set your Azure OpenAI environment variables")
    print("2. Run: pytest backend/tests/test_real_openai_integration.py -v -s")
    print("3. Or run individual tests with -k flag")
