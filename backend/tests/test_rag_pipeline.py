import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import AIMessage
from openai import RateLimitError, APITimeoutError, APIConnectionError, APIError
import httpx

from app.core.rag_pipeline import build_context, estimate_tokens, generate_answer, answer_question
from app.models.schemas import Citation
from app.core.exceptions import (
    EmptyContextError, 
    LLMGenerationError, 
    TokenLimitExceededError,
    AzureAPIError,
    DocumentNotFoundError
)


# Helper functions for creating proper OpenAI exceptions
def create_mock_response():
    """Create a mock httpx.Response for OpenAI exceptions"""
    mock_response = MagicMock(spec=httpx.Response)
    mock_request = MagicMock(spec=httpx.Request)
    mock_response.request = mock_request
    mock_response.status_code = 429  # Rate limit status code
    mock_response.headers = {"x-request-id": "test-request-id"}
    return mock_response


def create_mock_request():
    """Create a mock httpx.Request for OpenAI exceptions"""
    return MagicMock(spec=httpx.Request)


class TestBuildContext:
    """Tests for Task 1: Context Building Function Implementation"""
    
    def test_build_context_with_valid_chunks(self):
        """Test 1.1 & 1.2: Format chunks and generate context string"""
        chunks = [
            {
                "content": "This is the first document content.",
                "metadata": {
                    "filename": "document1.pdf",
                    "page": 1
                }
            },
            {
                "content": "This is the second document content.",
                "metadata": {
                    "filename": "document2.pdf", 
                    "page": 3
                }
            }
        ]
        
        context, chunk_mapping = build_context(chunks)
        
        # Verify context string format
        assert isinstance(context, str)
        assert "document1.pdf" in context
        assert "document2.pdf" in context
        assert "Page 1" in context
        assert "Page 3" in context
        assert "Source 1:" in context
        assert "Source 2:" in context
        assert "The following 2 document excerpt(s)" in context
        
        # Verify chunk mapping for citation generation (Task 1.4)
        assert isinstance(chunk_mapping, list)
        assert len(chunk_mapping) == 2
        
        # Check first mapping entry
        assert chunk_mapping[0]["source_id"] == 1
        assert chunk_mapping[0]["filename"] == "document1.pdf"
        assert chunk_mapping[0]["page"] == 1
        assert chunk_mapping[0]["content"] == "This is the first document content."
        
        # Check second mapping entry
        assert chunk_mapping[1]["source_id"] == 2
        assert chunk_mapping[1]["filename"] == "document2.pdf"
        assert chunk_mapping[1]["page"] == 3
        assert chunk_mapping[1]["content"] == "This is the second document content."
    
    def test_build_context_empty_chunks(self):
        """Test 1.3: Handle empty search results"""
        with pytest.raises(EmptyContextError):
            build_context([])
    
    def test_build_context_chunks_with_empty_content(self):
        """Test 1.3: Handle chunks with no valid content"""
        chunks = [
            {"content": "", "metadata": {"filename": "empty.pdf"}},
            {"content": "   ", "metadata": {"filename": "whitespace.pdf"}},
            {"content": None, "metadata": {"filename": "none.pdf"}}
        ]
        
        with pytest.raises(EmptyContextError):
            build_context(chunks)
    
    def test_build_context_mixed_valid_invalid_chunks(self):
        """Test handling of mixed valid and invalid chunks"""
        chunks = [
            {"content": "", "metadata": {"filename": "empty.pdf"}},
            {"content": "Valid content", "metadata": {"filename": "valid.pdf", "page": 2}},
            {"content": "   ", "metadata": {"filename": "whitespace.pdf"}}
        ]
        
        context, chunk_mapping = build_context(chunks)
        
        assert "valid.pdf" in context
        assert "Valid content" in context
        assert "Page 2" in context
        assert len(chunk_mapping) == 1
        assert chunk_mapping[0]["filename"] == "valid.pdf"
    
    def test_build_context_missing_metadata(self):
        """Test handling of chunks with missing metadata"""
        chunks = [
            {"content": "Content without filename", "metadata": {}},
            {"content": "Content with partial metadata", "metadata": {"filename": "partial.pdf"}},
            {"content": "Content without metadata"}
        ]
        
        context, chunk_mapping = build_context(chunks)
        
        assert "Document 1" in context  # Fallback for missing filename
        assert "partial.pdf" in context
        assert "Document 3" in context  # Fallback for missing metadata
        assert len(chunk_mapping) == 3
    
    def test_build_context_alternative_metadata_keys(self):
        """Test handling of alternative metadata key formats"""
        chunks = [
            {
                "content": "Content with source key",
                "metadata": {"source": "source_doc.pdf", "page_number": 5}
            }
        ]
        
        context, chunk_mapping = build_context(chunks)
        
        assert "source_doc.pdf" in context
        assert "Page 5" in context
        assert chunk_mapping[0]["filename"] == "source_doc.pdf"
        assert chunk_mapping[0]["page"] == 5


class TestEstimateTokens:
    """Tests for token estimation utility function"""
    
    def test_estimate_tokens_basic(self):
        """Test basic token estimation"""
        text = "This is a test sentence."
        tokens = estimate_tokens(text)
        assert isinstance(tokens, int)
        assert tokens > 0
    
    def test_estimate_tokens_empty_string(self):
        """Test token estimation with empty string"""
        tokens = estimate_tokens("")
        assert tokens == 0
    
    def test_estimate_tokens_different_models(self):
        """Test token estimation with different model names"""
        text = "Test text for token counting."
        
        # Should work with valid model
        tokens_gpt4 = estimate_tokens(text, "gpt-4o-mini")
        assert tokens_gpt4 > 0
        
        # Should fallback gracefully for invalid model
        tokens_invalid = estimate_tokens(text, "invalid-model")
        assert tokens_invalid > 0  # Should use fallback calculation


class TestGenerateAnswer:
    """Tests for Task 3: Azure OpenAI Integration Implementation"""
    
    @patch('app.core.rag_pipeline.langchain_config')
    @pytest.mark.anyio
    async def test_generate_answer_success(self, mock_config):
        """Test 3.1, 3.2, 3.5: Successful answer generation"""
        # Mock LLM response
        mock_llm = AsyncMock()
        mock_response = AIMessage(content="This is a test answer with citation [doc.pdf p. 1].")
        mock_llm.ainvoke.return_value = mock_response
        mock_config.llm = mock_llm
        
        question = "What is the main topic?"
        context = "This document discusses machine learning basics."
        
        with patch('app.core.rag_pipeline.SystemPrompts.format_rag_prompt') as mock_format:
            mock_format.return_value = "Formatted system prompt"
            
            result = await generate_answer(question, context)
            
            # Verify result
            assert isinstance(result, str)
            assert result == "This is a test answer with citation [doc.pdf p. 1]."
            
            # Verify LLM was called with correct parameters
            mock_llm.ainvoke.assert_called_once()
            call_args = mock_llm.ainvoke.call_args[0][0]
            assert len(call_args) == 2  # SystemMessage and HumanMessage
            assert call_args[1].content == question
    
    @patch('app.core.rag_pipeline.langchain_config')
    @pytest.mark.anyio
    async def test_generate_answer_empty_question(self, mock_config):
        """Test input validation for empty question"""
        with pytest.raises(LLMGenerationError):
            await generate_answer("", "context")
        
        with pytest.raises(LLMGenerationError):
            await generate_answer("   ", "context")
    
    @patch('app.core.rag_pipeline.langchain_config')
    @pytest.mark.anyio
    async def test_generate_answer_empty_context(self, mock_config):
        """Test input validation for empty context"""
        with pytest.raises(EmptyContextError):
            await generate_answer("question", "")
        
        with pytest.raises(EmptyContextError):
            await generate_answer("question", "   ")
    
    @patch('app.core.rag_pipeline.langchain_config')
    @patch('app.core.rag_pipeline.estimate_tokens')
    @pytest.mark.anyio
    async def test_generate_answer_token_limit_exceeded(self, mock_tokens, mock_config):
        """Test token limit validation"""
        # Mock token estimation to exceed limits
        mock_tokens.side_effect = [1000, 125000, 1000]  # question, context, system_prompt
        
        with pytest.raises(TokenLimitExceededError):
            await generate_answer("question", "very long context")
        
        # Test question too long
        mock_tokens.side_effect = [5000, 1000, 1000]  # question, context, system_prompt
        with pytest.raises(TokenLimitExceededError):
            await generate_answer("very long question", "context")
    
    @patch('app.core.rag_pipeline.langchain_config')
    @pytest.mark.anyio
    async def test_generate_answer_empty_response(self, mock_config):
        """Test 3.5: Handle empty LLM response"""
        mock_llm = AsyncMock()
        mock_response = AIMessage(content="")
        mock_llm.ainvoke.return_value = mock_response
        mock_config.llm = mock_llm
        
        with patch('app.core.rag_pipeline.SystemPrompts.format_rag_prompt'):
            with pytest.raises(LLMGenerationError):
                await generate_answer("question", "context")
    
    @patch('app.core.rag_pipeline.langchain_config')
    @pytest.mark.anyio
    async def test_generate_answer_rate_limit_error(self, mock_config):
        """Test 3.3, 3.4: Rate limit error handling with exponential backoff"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            RateLimitError("Rate limit exceeded", response=create_mock_response(), body=None),
            RateLimitError("Rate limit exceeded", response=create_mock_response(), body=None),
            AIMessage(content="Success after retries")
        ]
        mock_config.llm = mock_llm
        
        with patch('app.core.rag_pipeline.SystemPrompts.format_rag_prompt'):
            with patch('asyncio.sleep') as mock_sleep:
                result = await generate_answer("question", "context")
                
                assert result == "Success after retries"
                assert mock_llm.ainvoke.call_count == 3
                assert mock_sleep.call_count == 2  # Two retries with delays
    
    @patch('app.core.rag_pipeline.langchain_config')
    @pytest.mark.anyio
    async def test_generate_answer_rate_limit_exhausted(self, mock_config):
        """Test 3.4: Rate limit retries exhausted"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = RateLimitError("Rate limit exceeded", response=create_mock_response(), body=None)
        mock_config.llm = mock_llm
        
        with patch('app.core.rag_pipeline.SystemPrompts.format_rag_prompt'):
            with patch('asyncio.sleep'):
                with pytest.raises(AzureAPIError):
                    await generate_answer("question", "context")
    
    @patch('app.core.rag_pipeline.langchain_config')
    @pytest.mark.anyio
    async def test_generate_answer_timeout_error(self, mock_config):
        """Test 3.3, 3.4: API timeout error handling"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            APITimeoutError("Request timed out"),
            AIMessage(content="Success after timeout retry")
        ]
        mock_config.llm = mock_llm
        
        with patch('app.core.rag_pipeline.SystemPrompts.format_rag_prompt'):
            with patch('asyncio.sleep') as mock_sleep:
                result = await generate_answer("question", "context")
                
                assert result == "Success after timeout retry"
                assert mock_llm.ainvoke.call_count == 2
                assert mock_sleep.call_count == 1
    
    @patch('app.core.rag_pipeline.langchain_config')
    @pytest.mark.anyio
    async def test_generate_answer_connection_error(self, mock_config):
        """Test 3.3: API connection error handling"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = APIConnectionError(message="Connection failed", request=create_mock_request())
        mock_config.llm = mock_llm
        
        with patch('app.core.rag_pipeline.SystemPrompts.format_rag_prompt'):
            with patch('asyncio.sleep'):
                with pytest.raises(AzureAPIError):
                    await generate_answer("question", "context")
    
    @patch('app.core.rag_pipeline.langchain_config')
    @pytest.mark.anyio
    async def test_generate_answer_api_error_server(self, mock_config):
        """Test 3.3: Server API error (5xx) with retry"""
        mock_llm = AsyncMock()
        server_error = APIError("Server error", request=create_mock_request(), body=None)
        server_error.status_code = 500
        mock_llm.ainvoke.side_effect = [
            server_error,
            AIMessage(content="Success after server error")
        ]
        mock_config.llm = mock_llm
        
        with patch('app.core.rag_pipeline.SystemPrompts.format_rag_prompt'):
            with patch('asyncio.sleep'):
                result = await generate_answer("question", "context")
                
                assert result == "Success after server error"
                assert mock_llm.ainvoke.call_count == 2
    
    @patch('app.core.rag_pipeline.langchain_config')
    @pytest.mark.anyio
    async def test_generate_answer_api_error_client(self, mock_config):
        """Test 3.3: Client API error (4xx) without retry"""
        mock_llm = AsyncMock()
        client_error = APIError("Bad request", request=create_mock_request(), body=None)
        client_error.status_code = 400
        mock_llm.ainvoke.side_effect = client_error
        mock_config.llm = mock_llm
        
        with patch('app.core.rag_pipeline.SystemPrompts.format_rag_prompt'):
            with pytest.raises(AzureAPIError):
                await generate_answer("question", "context")
    
    @patch('app.core.rag_pipeline.langchain_config')
    @pytest.mark.anyio
    async def test_generate_answer_unexpected_error(self, mock_config):
        """Test 3.3: Unexpected error handling"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = Exception("Unexpected error")
        mock_config.llm = mock_llm
        
        with patch('app.core.rag_pipeline.SystemPrompts.format_rag_prompt'):
            with patch('asyncio.sleep'):
                with pytest.raises(LLMGenerationError):
                    await generate_answer("question", "context")
    
    @patch('app.core.rag_pipeline.langchain_config')
    @pytest.mark.anyio
    async def test_generate_answer_exponential_backoff(self, mock_config):
        """Test 3.4: Exponential backoff timing"""
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = [
            APITimeoutError("Timeout 1"),
            APITimeoutError("Timeout 2"),
            APITimeoutError("Timeout 3"),
            AIMessage(content="Final success")
        ]
        mock_config.llm = mock_llm
        
        with patch('app.core.rag_pipeline.SystemPrompts.format_rag_prompt'):
            with patch('asyncio.sleep') as mock_sleep:
                result = await generate_answer("question", "context")
                
                assert result == "Final success"
                assert mock_sleep.call_count == 3
                
                # Verify exponential backoff delays (base_delay * 2^attempt)
                delays = [call.args[0] for call in mock_sleep.call_args_list]
                assert delays[0] >= 1.0  # First retry: base_delay * 2^0 = 1.0
                assert delays[1] >= 2.0  # Second retry: base_delay * 2^1 = 2.0  
                assert delays[2] >= 4.0  # Third retry: base_delay * 2^2 = 4.0


class TestAnswerQuestion:
    """Integration tests for the complete RAG pipeline"""
    
    @patch('app.core.rag_pipeline.similarity_search')
    @patch('app.core.rag_pipeline.generate_answer')
    @patch('app.core.rag_pipeline.extract_citations')
    @pytest.mark.anyio
    async def test_answer_question_integration(self, mock_extract, mock_generate, mock_search):
        """Test complete RAG pipeline integration"""
        # Mock the pipeline components
        mock_chunks = [
            {"content": "Test content", "metadata": {"filename": "test.pdf", "page": 1}}
        ]
        mock_search.return_value = mock_chunks
        mock_generate.return_value = "Test answer with [test.pdf p. 1]"
        mock_extract.return_value = [
            Citation(document_id="1", page=1, snippet="Test content", document_name="test.pdf")
        ]
        
        result = await answer_question("What is the test?", 123)
        
        # Verify all components were called
        mock_search.assert_called_once_with("What is the test?", 123, k=5)  # Default limit of 5
        mock_generate.assert_called_once()
        mock_extract.assert_called_once_with("Test answer with [test.pdf p. 1]", mock_chunks)
        
        # Verify result structure - should be an Answer model
        assert hasattr(result, 'answer')
        assert hasattr(result, 'citations')
        assert hasattr(result, 'confidence')
    
    @patch('app.core.rag_pipeline.similarity_search')
    @patch('app.core.rag_pipeline.generate_answer')
    @patch('app.core.rag_pipeline.extract_citations')
    @pytest.mark.anyio
    async def test_answer_question_with_custom_limit(self, mock_extract, mock_generate, mock_search):
        """Test RAG pipeline with custom limit parameter (Task 5)"""
        mock_chunks = [
            {"content": "Test content", "metadata": {"filename": "test.pdf", "page": 1}}
        ]
        mock_search.return_value = mock_chunks
        mock_generate.return_value = "Test answer"
        mock_extract.return_value = []
        
        # Test with custom limit
        await answer_question("What is the test?", 123, limit=10)
        
        # Verify similarity_search was called with custom limit
        mock_search.assert_called_once_with("What is the test?", 123, k=10)
    
    @patch('app.core.rag_pipeline.similarity_search')
    @patch('app.core.rag_pipeline.generate_answer')
    @patch('app.core.rag_pipeline.extract_citations')
    @pytest.mark.anyio
    async def test_answer_question_with_document_ids_filter(self, mock_extract, mock_generate, mock_search):
        """Test RAG pipeline with document_ids filtering (Task 5)"""
        # Mock chunks with different document IDs
        mock_chunks = [
            {"content": "Content 1", "document_id": 1, "metadata": {"filename": "doc1.pdf", "page": 1}},
            {"content": "Content 2", "document_id": 2, "metadata": {"filename": "doc2.pdf", "page": 1}},
            {"content": "Content 3", "document_id": 3, "metadata": {"filename": "doc3.pdf", "page": 1}}
        ]
        mock_search.return_value = mock_chunks
        mock_generate.return_value = "Test answer"
        mock_extract.return_value = []
        
        # Test with document_ids filter
        await answer_question("What is the test?", 123, document_ids=["1", "3"])
        
        # Verify generate_answer was called with filtered chunks (only doc IDs 1 and 3)
        mock_generate.assert_called_once()
        context_arg = mock_generate.call_args[0][1]  # Second argument is context
        
        # The context should only contain content from documents 1 and 3
        assert "Content 1" in context_arg
        assert "Content 3" in context_arg
        assert "Content 2" not in context_arg
    
    @patch('app.core.rag_pipeline.similarity_search')
    @patch('app.core.rag_pipeline.generate_answer')
    @patch('app.core.rag_pipeline.extract_citations')
    @pytest.mark.anyio
    async def test_answer_question_with_both_limit_and_document_ids(self, mock_extract, mock_generate, mock_search):
        """Test RAG pipeline with both limit and document_ids parameters (Task 5)"""
        mock_chunks = [
            {"content": "Content 1", "document_id": 1, "metadata": {"filename": "doc1.pdf", "page": 1}},
            {"content": "Content 2", "document_id": 2, "metadata": {"filename": "doc2.pdf", "page": 1}}
        ]
        mock_search.return_value = mock_chunks
        mock_generate.return_value = "Test answer"
        mock_extract.return_value = []
        
        # Test with both parameters
        await answer_question("What is the test?", 123, document_ids=["1"], limit=15)
        
        # Verify similarity_search was called with custom limit
        mock_search.assert_called_once_with("What is the test?", 123, k=15)
        
        # Verify filtering was applied
        mock_generate.assert_called_once()
        context_arg = mock_generate.call_args[0][1]
        assert "Content 1" in context_arg
        assert "Content 2" not in context_arg
    
    @patch('app.core.rag_pipeline.similarity_search')
    @pytest.mark.anyio
    async def test_answer_question_backward_compatibility(self, mock_search):
        """Test backward compatibility - old signature still works (Task 5)"""
        mock_search.return_value = []
        
        # This should raise DocumentNotFoundError but not fail due to signature mismatch
        with pytest.raises(DocumentNotFoundError):
            await answer_question("What is the test?", 123)
        
        # Verify it was called with default parameters
        mock_search.assert_called_once_with("What is the test?", 123, k=5)
    
    @patch('app.core.rag_pipeline.similarity_search')
    @patch('app.core.rag_pipeline.generate_answer')
    @patch('app.core.rag_pipeline.extract_citations')
    @pytest.mark.anyio
    async def test_answer_question_invalid_document_ids(self, mock_extract, mock_generate, mock_search):
        """Test RAG pipeline handles invalid document_ids gracefully (Task 5)"""
        mock_chunks = [
            {"content": "Content 1", "document_id": 1, "metadata": {"filename": "doc1.pdf", "page": 1}}
        ]
        mock_search.return_value = mock_chunks
        mock_generate.return_value = "Test answer"
        mock_extract.return_value = []
        
        # Test with invalid document_ids (non-numeric)
        await answer_question("What is the test?", 123, document_ids=["invalid", "also_invalid"])
        
        # Should continue with unfiltered results when document_ids are invalid
        mock_generate.assert_called_once()
        context_arg = mock_generate.call_args[0][1]
        assert "Content 1" in context_arg  # Original content should still be there
    
    @patch('app.core.rag_pipeline.similarity_search')
    @pytest.mark.anyio
    async def test_answer_question_document_filter_no_matches(self, mock_search):
        """Test RAG pipeline when document_ids filter results in no matches (Task 5)"""
        mock_chunks = [
            {"content": "Content 1", "document_id": 1, "metadata": {"filename": "doc1.pdf", "page": 1}}
        ]
        mock_search.return_value = mock_chunks
        
        # Test with document_ids that don't match any chunks
        with pytest.raises(DocumentNotFoundError):
            await answer_question("What is the test?", 123, document_ids=["999", "888"])
        
        # Verify search was called but filtering resulted in no chunks


class TestExtractCitations:
    """Tests for Task 4: Citation Extraction and Linking"""
    
    def test_extract_citations_basic_format(self):
        """Test 4.1: Basic regex pattern matching for [filename p. X] format"""
        from app.core.rag_pipeline import extract_citations
        
        answer = "According to the study [research.pdf p. 5], machine learning is growing. See also [data.pdf p. 10]."
        chunks = [
            {
                "content": "Machine learning is a rapidly growing field with applications in various domains.",
                "metadata": {"filename": "research.pdf", "page": 5}
            },
            {
                "content": "Data analysis shows significant growth in AI adoption across industries.",
                "metadata": {"filename": "data.pdf", "page": 10}
            }
        ]
        
        citations = extract_citations(answer, chunks)
        
        assert len(citations) == 2
        assert citations[0].document_name == "research.pdf"
        assert citations[0].page == 5
        assert citations[1].document_name == "data.pdf"
        assert citations[1].page == 10
    
    def test_extract_citations_filename_only_format(self):
        """Test 4.1: Citations without page numbers [filename]"""
        from app.core.rag_pipeline import extract_citations
        
        answer = "The report [annual_review.pdf] shows positive trends."
        chunks = [
            {
                "content": "Annual review content here.",
                "metadata": {"filename": "annual_review.pdf", "page": 1}
            }
        ]
        
        citations = extract_citations(answer, chunks)
        
        assert len(citations) == 1
        assert citations[0].document_name == "annual_review.pdf"
        assert citations[0].page == 1  # Should use chunk page when citation has no page
    
    def test_extract_citations_whitespace_variations(self):
        """Test 4.1: Regex handles whitespace variations"""
        from app.core.rag_pipeline import extract_citations
        
        answer = "See [ doc.pdf p. 3 ] and [  file.pdf   p.   7  ]."
        chunks = [
            {
                "content": "Document content",
                "metadata": {"filename": "doc.pdf", "page": 3}
            },
            {
                "content": "File content",
                "metadata": {"filename": "file.pdf", "page": 7}
            }
        ]
        
        citations = extract_citations(answer, chunks)
        
        assert len(citations) == 2
        assert citations[0].document_name == "doc.pdf"
        assert citations[0].page == 3
        assert citations[1].document_name == "file.pdf"
        assert citations[1].page == 7
    
    def test_extract_citations_case_insensitive_matching(self):
        """Test 4.2: Citation-to-source mapping is case insensitive"""
        from app.core.rag_pipeline import extract_citations
        
        answer = "The data [REPORT.PDF p. 2] shows results."
        chunks = [
            {
                "content": "Report content",
                "metadata": {"filename": "report.pdf", "page": 2}
            }
        ]
        
        citations = extract_citations(answer, chunks)
        
        assert len(citations) == 1
        assert citations[0].document_name == "report.pdf"
        assert citations[0].page == 2
    
    def test_extract_citations_partial_filename_matching(self):
        """Test 4.2: Partial filename matching (without extensions)"""
        from app.core.rag_pipeline import extract_citations
        
        answer = "According to [report p. 1], the findings are significant."
        chunks = [
            {
                "content": "Report findings here",
                "metadata": {"filename": "report.pdf", "page": 1}
            }
        ]
        
        citations = extract_citations(answer, chunks)
        
        assert len(citations) == 1
        assert citations[0].document_name == "report.pdf"
        assert citations[0].page == 1
    
    def test_extract_citations_alternative_page_formats(self):
        """Test 4.1: Support for different page number formats"""
        from app.core.rag_pipeline import extract_citations
        
        answer = "See [doc.pdf page 5] and [file.pdf p.10] and [book.pdf p 15]."
        chunks = [
            {
                "content": "Doc content",
                "metadata": {"filename": "doc.pdf", "page": 5}
            },
            {
                "content": "File content", 
                "metadata": {"filename": "file.pdf", "page": 10}
            },
            {
                "content": "Book content",
                "metadata": {"filename": "book.pdf", "page": 15}
            }
        ]
        
        citations = extract_citations(answer, chunks)
        
        assert len(citations) == 3
        assert citations[0].page == 5
        assert citations[1].page == 10
        assert citations[2].page == 15
    
    def test_extract_citations_snippet_creation(self):
        """Test 4.3: Proper snippet creation and truncation"""
        from app.core.rag_pipeline import extract_citations
        
        long_content = "This is a very long piece of content that should be truncated when creating snippets for citations. " * 10
        answer = "According to [doc.pdf p. 1], this is true."
        chunks = [
            {
                "content": long_content,
                "metadata": {"filename": "doc.pdf", "page": 1}
            }
        ]
        
        citations = extract_citations(answer, chunks)
        
        assert len(citations) == 1
        assert len(citations[0].snippet) <= 203  # 200 chars + "..."
        assert citations[0].snippet.endswith("...")
        assert "This is a very long piece of content" in citations[0].snippet
    
    def test_extract_citations_document_id_generation(self):
        """Test 4.3: Proper document_id generation"""
        from app.core.rag_pipeline import extract_citations
        
        answer = "See [doc.pdf p. 5] and [nodoc.pdf]."
        chunks = [
            {
                "content": "Content with page",
                "metadata": {"filename": "doc.pdf", "page": 5}
            },
            {
                "content": "Content without page",
                "metadata": {"filename": "nodoc.pdf"}
            }
        ]
        
        citations = extract_citations(answer, chunks)
        
        assert len(citations) == 2
        assert citations[0].document_id == "doc.pdf:5"
        assert citations[1].document_id == "nodoc.pdf"
    
    def test_extract_citations_malformed_citations_ignored(self):
        """Test 4.4: Malformed citations are gracefully ignored"""
        from app.core.rag_pipeline import extract_citations
        
        answer = "See [nonexistent.pdf p. 1] and [valid.pdf p. 2] and [p. 3] and []."
        chunks = [
            {
                "content": "Valid content",
                "metadata": {"filename": "valid.pdf", "page": 2}
            }
        ]
        
        citations = extract_citations(answer, chunks)
        
        # Only the valid citation should be returned
        assert len(citations) == 1
        assert citations[0].document_name == "valid.pdf"
        assert citations[0].page == 2
    
    def test_extract_citations_duplicate_removal(self):
        """Test 4.4: Duplicate citations are removed"""
        from app.core.rag_pipeline import extract_citations
        
        answer = "See [doc.pdf p. 1] and later [doc.pdf p. 1] again."
        chunks = [
            {
                "content": "Document content",
                "metadata": {"filename": "doc.pdf", "page": 1}
            }
        ]
        
        citations = extract_citations(answer, chunks)
        
        # Should only return one citation despite two matches
        assert len(citations) == 1
        assert citations[0].document_name == "doc.pdf"
        assert citations[0].page == 1
    
    def test_extract_citations_empty_input_handling(self):
        """Test 4.4: Handle empty inputs gracefully"""
        from app.core.rag_pipeline import extract_citations
        
        # Empty answer
        assert extract_citations("", [{"content": "test"}]) == []
        
        # Empty chunks
        assert extract_citations("See [doc.pdf p. 1]", []) == []
        
        # None answer
        assert extract_citations(None, [{"content": "test"}]) == []
    
    def test_extract_citations_missing_metadata_handling(self):
        """Test 4.4: Handle chunks with missing metadata"""
        from app.core.rag_pipeline import extract_citations
        
        answer = "See [doc.pdf p. 1] and [missing.pdf p. 2]."
        chunks = [
            {
                "content": "Content with full metadata",
                "metadata": {"filename": "doc.pdf", "page": 1}
            },
            {
                "content": "Content with missing metadata"
                # No metadata field
            },
            {
                "content": "Content with empty metadata",
                "metadata": {}
            }
        ]
        
        citations = extract_citations(answer, chunks)
        
        # Only citation with valid metadata should be returned
        assert len(citations) == 1
        assert citations[0].document_name == "doc.pdf"
        assert citations[0].page == 1
    
    def test_extract_citations_alternative_metadata_keys(self):
        """Test 4.2: Support alternative metadata key formats"""
        from app.core.rag_pipeline import extract_citations
        
        answer = "See [source_doc.pdf p. 3]."
        chunks = [
            {
                "content": "Content with source key",
                "metadata": {"source": "source_doc.pdf", "page_number": 3}
            }
        ]
        
        citations = extract_citations(answer, chunks)
        
        assert len(citations) == 1
        assert citations[0].document_name == "source_doc.pdf"
        assert citations[0].page == 3
    
    def test_extract_citations_invalid_page_numbers(self):
        """Test 4.4: Handle invalid page numbers gracefully"""
        from app.core.rag_pipeline import extract_citations
        
        answer = "See [doc.pdf p. invalid] and [doc2.pdf p. 5]."
        chunks = [
            {
                "content": "Valid content",
                "metadata": {"filename": "doc2.pdf", "page": 5}
            }
        ]
        
        citations = extract_citations(answer, chunks)
        
        # Only citation with valid page number should be returned
        assert len(citations) == 1
        assert citations[0].document_name == "doc2.pdf"
        assert citations[0].page == 5
    
    def test_extract_citations_page_mismatch_handling(self):
        """Test 4.2: Handle page number mismatches"""
        from app.core.rag_pipeline import extract_citations
        
        answer = "See [doc.pdf p. 5] and [doc.pdf p. 10]."
        chunks = [
            {
                "content": "Content from page 5",
                "metadata": {"filename": "doc.pdf", "page": 5}
            },
            {
                "content": "Content from page 8", 
                "metadata": {"filename": "doc.pdf", "page": 8}
            }
        ]
        
        citations = extract_citations(answer, chunks)
        
        # Should only match the citation with exact page match
        assert len(citations) == 1
        assert citations[0].document_name == "doc.pdf"
        assert citations[0].page == 5
    
    def test_extract_citations_no_page_in_chunk_but_page_in_citation(self):
        """Test 4.2: Handle chunks without page info when citation has page"""
        from app.core.rag_pipeline import extract_citations
        
        answer = "See [doc.pdf p. 5]."
        chunks = [
            {
                "content": "Content without page info",
                "metadata": {"filename": "doc.pdf"}  # No page field
            }
        ]
        
        citations = extract_citations(answer, chunks)
        
        # Should still match based on filename, use citation page
        assert len(citations) == 1
        assert citations[0].document_name == "doc.pdf"
        assert citations[0].page == 5
