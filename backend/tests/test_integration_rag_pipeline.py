"""
Integration tests for the complete RAG pipeline.

These tests verify the end-to-end functionality from user query to generated answer,
including authentication, vector search, context building, LLM generation, and citation extraction.
"""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.api.routes.chat import chat_query
from app.core.exceptions import DocumentNotFoundError, LLMGenerationError
from app.models.schemas import Answer, Citation, Query, RetrievedChunk


class TestRagPipelineIntegration:
    """Integration tests for the complete RAG pipeline."""

    @pytest.mark.asyncio
    async def test_complete_rag_pipeline_integration(self):
        """
        Test the complete RAG pipeline from API endpoint to final response.

        This test verifies:
        1. Authentication and user extraction
        2. Vector similarity search with user isolation
        3. Context building from retrieved chunks
        4. LLM answer generation with proper prompts
        5. Citation extraction and linking
        6. Response formatting with all required fields
        """
        # Mock user authentication
        mock_user = {"id": 123, "email": "test@example.com"}

        # Mock vector search results
        mock_chunks = [
            {
                "id": 1,
                "document_id": 456,
                "content": "The payment terms are Net 30 days from the invoice date as specified in section 3.2 of the contract.",
                "page": 3,
                "filename": "contract.pdf",
                "distance": 0.15,
                "similarity_score": 0.85,
                "created_at": "2024-01-01T00:00:00",
                "metadata": {
                    "filename": "contract.pdf",
                    "page": 3,
                    "source": "contract.pdf",
                },
            },
            {
                "id": 2,
                "document_id": 789,
                "content": "Additional payment terms include a 2% discount for payments made within 10 days.",
                "page": 1,
                "filename": "payment_policy.pdf",
                "distance": 0.25,
                "similarity_score": 0.75,
                "created_at": "2024-01-01T00:00:00",
                "metadata": {
                    "filename": "payment_policy.pdf",
                    "page": 1,
                    "source": "payment_policy.pdf",
                },
            },
        ]

        # Mock LLM response with citations
        mock_llm_response = AIMessage(
            content="Based on the provided documents, the payment terms are Net 30 days from the invoice date [contract.pdf p. 3]. Additionally, there is a 2% discount available for early payment within 10 days [payment_policy.pdf p. 1]."
        )

        # Test query
        test_query = Query(
            question="What are the payment terms?", document_ids=None, limit=5
        )

        with (
            patch("app.core.rag_pipeline.similarity_search") as mock_search,
            patch("app.core.rag_pipeline.langchain_config") as mock_config,
            patch(
                "app.core.rag_pipeline.SystemPrompts.format_rag_prompt"
            ) as mock_format_prompt,
        ):
            # Setup mocks
            mock_search.return_value = mock_chunks
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_llm_response
            mock_config.llm = mock_llm
            mock_format_prompt.return_value = "System prompt with context"

            # Execute the complete pipeline
            result = await chat_query(test_query, mock_user)

            # Verify the result is an Answer object
            assert isinstance(result, Answer)

            # Verify core response fields
            assert result.query == "What are the payment terms?"
            assert "Net 30 days" in result.answer
            assert "2% discount" in result.answer
            assert result.confidence == 0.95

            # Verify citations were extracted
            assert len(result.citations) >= 1
            citation_filenames = [c.document_name for c in result.citations]
            assert (
                "contract.pdf" in citation_filenames
                or "payment_policy.pdf" in citation_filenames
            )

            # Verify retrieved chunks are included
            assert len(result.chunks) == 2
            assert all(isinstance(chunk, RetrievedChunk) for chunk in result.chunks)

            # Verify chunk content
            chunk_contents = [chunk.content for chunk in result.chunks]
            assert any("Net 30 days" in content for content in chunk_contents)
            assert any("2% discount" in content for content in chunk_contents)

            # Verify similarity scores
            assert all(chunk.similarity_score > 0 for chunk in result.chunks)

            # Verify metadata
            assert result.metadata["chunks_retrieved"] == 2
            assert result.metadata["search_limit"] == 5

            # Verify mock calls
            mock_search.assert_called_once_with("What are the payment terms?", 123, k=5)
            mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_rag_pipeline_with_document_filtering(self):
        """
        Test RAG pipeline with document ID filtering.

        Verifies that document_ids parameter properly filters search results.
        """
        mock_user = {"id": 123, "email": "test@example.com"}

        # Mock chunks from specific document
        mock_chunks = [
            {
                "id": 1,
                "document_id": 456,  # This should match our filter
                "content": "Specific contract terms for payment processing.",
                "page": 2,
                "filename": "specific_contract.pdf",
                "distance": 0.1,
                "similarity_score": 0.9,
                "created_at": "2024-01-01T00:00:00",
                "metadata": {"filename": "specific_contract.pdf", "page": 2},
            }
        ]

        mock_llm_response = AIMessage(
            content="According to the specific contract, payment processing terms are outlined [specific_contract.pdf p. 2]."
        )

        test_query = Query(
            question="What are the contract terms?",
            document_ids=["456"],  # Filter to specific document
            limit=3,
        )

        with (
            patch("app.core.rag_pipeline.similarity_search") as mock_search,
            patch("app.core.rag_pipeline.langchain_config") as mock_config,
            patch("app.core.rag_pipeline.SystemPrompts.format_rag_prompt"),
        ):
            mock_search.return_value = mock_chunks
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_llm_response
            mock_config.llm = mock_llm

            result = await chat_query(test_query, mock_user)

            # Verify document filtering was applied
            assert result.metadata["document_ids_filter"] == ["456"]
            assert result.metadata["search_limit"] == 3

            # Verify only filtered document content appears
            assert len(result.chunks) == 1
            assert result.chunks[0].document_id == "456"

    @pytest.mark.asyncio
    async def test_rag_pipeline_error_handling(self):
        """
        Test RAG pipeline error handling scenarios.

        Verifies that various error conditions are properly handled and propagated.
        """
        mock_user = {"id": 123, "email": "test@example.com"}

        test_query = Query(question="What are the terms?")

        # Test case 1: No documents found
        with patch("app.core.rag_pipeline.similarity_search") as mock_search:
            mock_search.return_value = []  # No results

            with pytest.raises(DocumentNotFoundError):
                await chat_query(test_query, mock_user)

        # Test case 2: LLM generation failure
        mock_chunks = [
            {
                "id": 1,
                "document_id": 456,
                "content": "Some content",
                "page": 1,
                "filename": "test.pdf",
                "distance": 0.1,
                "similarity_score": 0.9,
                "created_at": "2024-01-01T00:00:00",
                "metadata": {"filename": "test.pdf", "page": 1},
            }
        ]

        with (
            patch("app.core.rag_pipeline.similarity_search") as mock_search,
            patch("app.core.rag_pipeline.langchain_config") as mock_config,
        ):
            mock_search.return_value = mock_chunks
            mock_llm = AsyncMock()
            mock_llm.ainvoke.side_effect = Exception("LLM API failure")
            mock_config.llm = mock_llm

            with pytest.raises(LLMGenerationError):
                await chat_query(test_query, mock_user)

    @pytest.mark.asyncio
    async def test_rag_pipeline_user_isolation(self):
        """
        Test that user isolation is properly enforced throughout the pipeline.

        Verifies that user_id is passed correctly through all components.
        """
        mock_user = {"id": 999, "email": "isolated@example.com"}

        mock_chunks = [
            {
                "id": 1,
                "document_id": 123,
                "content": "User-specific content",
                "page": 1,
                "filename": "user_doc.pdf",
                "distance": 0.1,
                "similarity_score": 0.9,
                "created_at": "2024-01-01T00:00:00",
                "metadata": {"filename": "user_doc.pdf", "page": 1},
            }
        ]

        mock_llm_response = AIMessage(
            content="User-specific answer [user_doc.pdf p. 1]."
        )

        test_query = Query(question="What's in my documents?")

        with (
            patch("app.core.rag_pipeline.similarity_search") as mock_search,
            patch("app.core.rag_pipeline.langchain_config") as mock_config,
            patch("app.core.rag_pipeline.SystemPrompts.format_rag_prompt"),
        ):
            mock_search.return_value = mock_chunks
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_llm_response
            mock_config.llm = mock_llm

            result = await chat_query(test_query, mock_user)

            # Verify user_id was passed to similarity search
            mock_search.assert_called_once_with("What's in my documents?", 999, k=5)

            # Verify result contains user's content
            assert len(result.chunks) == 1
            assert "User-specific content" in result.chunks[0].content

    @pytest.mark.asyncio
    async def test_rag_pipeline_citation_extraction_integration(self):
        """
        Test citation extraction integration within the complete pipeline.

        Verifies that citations are properly extracted and linked to source chunks.
        """
        mock_user = {"id": 123, "email": "test@example.com"}

        mock_chunks = [
            {
                "id": 1,
                "document_id": 456,
                "content": "The company policy states that vacation requests must be submitted 2 weeks in advance.",
                "page": 5,
                "filename": "employee_handbook.pdf",
                "distance": 0.1,
                "similarity_score": 0.95,
                "created_at": "2024-01-01T00:00:00",
                "metadata": {"filename": "employee_handbook.pdf", "page": 5},
            }
        ]

        # LLM response with multiple citation formats
        mock_llm_response = AIMessage(
            content="According to company policy, vacation requests must be submitted 2 weeks in advance [employee_handbook.pdf p. 5]. This ensures proper staffing coverage [employee_handbook.pdf]."
        )

        test_query = Query(question="What is the vacation request policy?")

        with (
            patch("app.core.rag_pipeline.similarity_search") as mock_search,
            patch("app.core.rag_pipeline.langchain_config") as mock_config,
            patch("app.core.rag_pipeline.SystemPrompts.format_rag_prompt"),
        ):
            mock_search.return_value = mock_chunks
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_llm_response
            mock_config.llm = mock_llm

            result = await chat_query(test_query, mock_user)

            # Verify citations were extracted
            assert len(result.citations) >= 1

            # Verify citation details
            citation = result.citations[0]
            assert isinstance(citation, Citation)
            assert citation.document_name == "employee_handbook.pdf"
            assert citation.page == 5
            assert "vacation requests" in citation.snippet.lower()

            # Verify citations are properly linked to chunks
            chunk_filenames = {chunk.document_name for chunk in result.chunks}
            citation_filenames = {cite.document_name for cite in result.citations}
            assert citation_filenames.issubset(
                chunk_filenames
            )  # All citations should link to retrieved chunks


class TestRagPipelineEndToEnd:
    """End-to-end integration tests simulating real usage scenarios."""

    @pytest.mark.asyncio
    async def test_realistic_legal_document_query(self):
        """
        Test a realistic scenario: querying legal documents for contract terms.

        This test simulates a real-world usage pattern with legal document analysis.
        """
        mock_user = {"id": 42, "email": "lawyer@lawfirm.com"}

        # Realistic legal document chunks
        mock_chunks = [
            {
                "id": 1,
                "document_id": 1001,
                "content": "Section 4.2 Termination: Either party may terminate this Agreement upon thirty (30) days written notice to the other party. Upon termination, all obligations shall cease except those which by their nature should survive termination.",
                "page": 8,
                "filename": "service_agreement_2024.pdf",
                "distance": 0.05,
                "similarity_score": 0.95,
                "created_at": "2024-01-15T00:00:00",
                "metadata": {
                    "filename": "service_agreement_2024.pdf",
                    "page": 8,
                    "section": "4.2",
                },
            },
            {
                "id": 2,
                "document_id": 1002,
                "content": "Termination without cause requires ninety (90) days advance written notice as outlined in the master services agreement dated January 1, 2024.",
                "page": 12,
                "filename": "master_services_agreement.pdf",
                "distance": 0.12,
                "similarity_score": 0.88,
                "created_at": "2024-01-01T00:00:00",
                "metadata": {"filename": "master_services_agreement.pdf", "page": 12},
            },
        ]

        # Realistic LLM response with proper legal citations
        mock_llm_response = AIMessage(
            content="Based on the provided agreements, termination notice requirements vary by contract. The Service Agreement requires thirty (30) days written notice [service_agreement_2024.pdf p. 8], while the Master Services Agreement requires ninety (90) days advance written notice [master_services_agreement.pdf p. 12]. Both agreements specify that certain obligations survive termination."
        )

        test_query = Query(
            question="What are the termination notice requirements in our contracts?",
            limit=10,  # Legal queries often need more comprehensive results
        )

        with (
            patch("app.core.rag_pipeline.similarity_search") as mock_search,
            patch("app.core.rag_pipeline.langchain_config") as mock_config,
            patch("app.core.rag_pipeline.SystemPrompts.format_rag_prompt"),
        ):
            mock_search.return_value = mock_chunks
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_llm_response
            mock_config.llm = mock_llm

            result = await chat_query(test_query, mock_user)

            # Verify comprehensive legal analysis
            assert "thirty (30) days" in result.answer
            assert "ninety (90) days" in result.answer
            assert len(result.citations) == 2  # Both contracts cited

            # Verify legal citation format
            citation_pages = [c.page for c in result.citations]
            assert 8 in citation_pages and 12 in citation_pages

            # Verify document names in citations
            doc_names = [c.document_name for c in result.citations]
            assert "service_agreement_2024.pdf" in doc_names
            assert "master_services_agreement.pdf" in doc_names

            # Verify search parameters for legal use case
            mock_search.assert_called_once_with(
                "What are the termination notice requirements in our contracts?",
                42,
                k=10,
            )
