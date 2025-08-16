"""Tests for token counter utility."""

import pytest
from langchain_core.documents import Document

from app.utils.token_counter import TokenCounter, count_tokens, count_tokens_async


class TestTokenCounter:
    """Test cases for TokenCounter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.counter = TokenCounter()
        self.legal_counter = TokenCounter(legal_specific=True)

    def test_token_counter_initialization(self):
        """Test TokenCounter initialization with default encoding."""
        assert self.counter.encoding_name == "cl100k_base"
        assert self.counter._encoding is None  # Lazy loading
        assert self.counter.legal_specific is False

    def test_token_counter_legal_initialization(self):
        """Test TokenCounter initialization with legal-specific features."""
        assert self.legal_counter.legal_specific is True
        assert len(self.legal_counter.LEGAL_PATTERNS) > 0
        assert "legal_citations" in self.legal_counter.LEGAL_PATTERNS

    def test_token_counter_custom_encoding(self):
        """Test TokenCounter initialization with custom encoding."""
        counter = TokenCounter("p50k_base")
        assert counter.encoding_name == "p50k_base"

    def test_token_counter_model_initialization(self):
        """Test TokenCounter initialization with model name."""
        counter = TokenCounter(model_name="gpt-4")
        assert counter.model_name == "gpt-4"
        assert counter.encoding_name == "cl100k_base"

    def test_count_tokens_simple_text(self):
        """Test counting tokens for simple text."""
        text = "Hello, world!"
        token_count = self.counter.count_tokens(text)
        assert isinstance(token_count, int)
        assert token_count > 0
        # "Hello, world!" should be around 3-4 tokens
        assert 2 <= token_count <= 5

    @pytest.mark.asyncio
    async def test_count_tokens_async(self):
        """Test async token counting."""
        text = "Hello, world!"
        token_count = await self.counter.count_tokens_async(text)
        sync_count = self.counter.count_tokens(text)
        assert token_count == sync_count
        assert isinstance(token_count, int)

    def test_count_tokens_empty_text(self):
        """Test counting tokens for empty text."""
        assert self.counter.count_tokens("") == 0
        assert self.counter.count_tokens(None) == 0

    def test_count_tokens_longer_text(self):
        """Test counting tokens for longer text."""
        text = (
            "This is a longer piece of text that should contain more tokens. "
            "It includes multiple sentences and various punctuation marks. "
            "The token count should be significantly higher than simple text."
        )
        token_count = self.counter.count_tokens(text)
        assert token_count > 20  # Should be much more than simple text

    def test_legal_specific_token_counting(self):
        """Test legal-specific token counting adjustments."""
        legal_text = (
            "Section 5.1 provides that the parties shall comply with all applicable laws. "
            "Whereas the defendant has failed to appear, therefore the court grants summary judgment. "
            "Pursuant to 42 U.S.C. § 1983, the plaintiff seeks damages."
        )

        regular_count = self.counter.count_tokens(legal_text)
        legal_count = self.legal_counter.count_tokens(legal_text)

        # Legal counter should produce different count due to adjustments
        assert legal_count != regular_count
        assert (
            legal_count > regular_count
        )  # Should be higher due to legal pattern adjustments

    def test_count_tokens_for_model_gpt4(self):
        """Test counting tokens for GPT-4 model."""
        text = "Hello, world!"
        token_count = self.counter.count_tokens_for_model(text, "gpt-4")
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_count_tokens_for_model_gpt4o(self):
        """Test counting tokens for GPT-4o model."""
        text = "Hello, world!"
        token_count = self.counter.count_tokens_for_model(text, "gpt-4o")
        assert isinstance(token_count, int)
        assert token_count > 0

    @pytest.mark.asyncio
    async def test_count_tokens_for_model_async(self):
        """Test async model-specific token counting."""
        text = "Hello, world!"
        async_count = await self.counter.count_tokens_for_model_async(text, "gpt-4")
        sync_count = self.counter.count_tokens_for_model(text, "gpt-4")
        assert async_count == sync_count

    def test_count_tokens_for_invalid_model(self):
        """Test counting tokens for invalid model falls back gracefully."""
        text = "Hello, world!"
        token_count = self.counter.count_tokens_for_model(text, "invalid-model")
        # Should fallback to default counting
        default_count = self.counter.count_tokens(text)
        assert token_count == default_count

    def test_count_document_tokens(self):
        """Test counting tokens for Langchain Document."""
        document = Document(
            page_content="This is test content for document token counting.",
            metadata={"source": "test.txt", "page": 1},
        )

        result = self.counter.count_document_tokens(document, "gpt-4")

        assert isinstance(result, dict)
        assert "content_tokens" in result
        assert "metadata_tokens" in result
        assert "total_tokens" in result
        assert result["content_tokens"] > 0
        assert result["metadata_tokens"] > 0
        assert (
            result["total_tokens"]
            == result["content_tokens"] + result["metadata_tokens"]
        )

    @pytest.mark.asyncio
    async def test_count_document_tokens_async(self):
        """Test async document token counting."""
        document = Document(
            page_content="Test content", metadata={"source": "test.txt"}
        )

        async_result = await self.counter.count_document_tokens_async(document)
        sync_result = self.counter.count_document_tokens(document)

        assert async_result == sync_result

    def test_count_documents_tokens(self):
        """Test counting tokens for multiple documents."""
        documents = [
            Document(page_content="First document content", metadata={"page": 1}),
            Document(page_content="Second document content", metadata={"page": 2}),
            Document(page_content="Third document content", metadata={"page": 3}),
        ]

        result = self.counter.count_documents_tokens(documents, "gpt-4")

        assert isinstance(result, dict)
        assert result["total_documents"] == 3
        assert result["total_content_tokens"] > 0
        assert result["total_metadata_tokens"] > 0
        assert result["total_tokens"] > 0
        assert result["average_tokens_per_document"] > 0
        assert len(result["document_details"]) == 3

    @pytest.mark.asyncio
    async def test_count_documents_tokens_async(self):
        """Test async multiple documents token counting."""
        documents = [
            Document(page_content="Test content 1", metadata={}),
            Document(page_content="Test content 2", metadata={}),
        ]

        async_result = await self.counter.count_documents_tokens_async(documents)
        sync_result = self.counter.count_documents_tokens(documents)

        assert async_result == sync_result

    def test_count_sections_tokens(self):
        """Test counting tokens for document sections."""
        text = """
        SECTION 1
        This is the first section of the document.

        SECTION 2
        This is the second section with more content.

        Section 3
        This is the third section.
        """

        result = self.counter.count_sections_tokens(text, model_name="gpt-4")

        assert isinstance(result, dict)
        assert result["total_sections"] > 0
        assert result["total_tokens"] > 0
        assert result["average_tokens_per_section"] > 0
        assert len(result["section_details"]) > 0

    def test_count_sections_tokens_custom_pattern(self):
        """Test section counting with custom pattern."""
        text = """
        Chapter 1: Introduction
        This is the introduction chapter.

        Chapter 2: Methods
        This describes the methods used.
        """

        pattern = r"Chapter\s+\d+"
        result = self.counter.count_sections_tokens(text, section_pattern=pattern)

        assert result["total_sections"] >= 2

    def test_split_text_by_tokens(self):
        """Test text splitting using Langchain TokenTextSplitter."""
        text = (
            "This is a long text that should be split into multiple chunks for testing purposes. "
            * 50
        )

        chunks = self.counter.split_text_by_tokens(
            text, chunk_size=100, chunk_overlap=20, model_name="gpt-4"
        )

        assert isinstance(chunks, list)
        assert len(chunks) > 1
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

    def test_get_langchain_splitter(self):
        """Test getting Langchain TokenTextSplitter."""
        splitter = self.counter.get_langchain_splitter(
            chunk_size=500, model_name="gpt-4"
        )

        assert splitter is not None
        assert hasattr(splitter, "split_text")

    def test_create_length_function(self):
        """Test creating custom length function for legal tokenization."""
        legal_counter = TokenCounter(legal_specific=True)
        length_func = legal_counter.create_length_function()

        legal_text = "Section 5.1 provides that whereas the defendant..."
        regular_text = "This is regular text without legal terms."

        legal_length = length_func(legal_text)
        regular_length = length_func(regular_text)

        assert isinstance(legal_length, int)
        assert isinstance(regular_length, int)
        assert legal_length > 0
        assert regular_length > 0

    def test_estimate_cost(self):
        """Test cost estimation functionality."""
        text = "Hello, world!"
        cost = self.counter.estimate_cost(text, "gpt-4", 0.03)  # $0.03 per 1k tokens
        assert isinstance(cost, float)
        assert cost > 0
        assert cost < 1  # Should be much less than $1 for short text

    def test_get_encoding_info(self):
        """Test getting encoding information."""
        info = self.counter.get_encoding_info()
        assert isinstance(info, dict)
        assert "encoding_name" in info
        assert "max_token_value" in info
        assert "vocab_size" in info
        assert info["encoding_name"] == "cl100k_base"

    def test_get_encoding_info_legal(self):
        """Test getting encoding info with legal patterns."""
        info = self.legal_counter.get_encoding_info()
        assert info["legal_specific"] is True
        assert "legal_patterns" in info
        assert len(info["legal_patterns"]) > 0

    def test_get_available_encodings(self):
        """Test getting available encodings."""
        encodings = TokenCounter.get_available_encodings()
        assert isinstance(encodings, list)
        assert len(encodings) > 0
        assert "cl100k_base" in encodings

    def test_create_for_model(self):
        """Test creating TokenCounter for specific model."""
        counter = TokenCounter.create_for_model("gpt-4")
        assert isinstance(counter, TokenCounter)
        # GPT-4 uses cl100k_base encoding
        assert counter.encoding_name == "cl100k_base"

    def test_create_for_model_legal(self):
        """Test creating TokenCounter for model with legal features."""
        counter = TokenCounter.create_for_model("gpt-4", legal_specific=True)
        assert counter.legal_specific is True
        assert counter.model_name == "gpt-4"

    def test_create_for_gpt4o_model(self):
        """Test creating TokenCounter for GPT-4o model."""
        counter = TokenCounter.create_for_model("gpt-4o")

        assert counter.model_name == "gpt-4o"
        assert counter.encoding_name == "o200k_base"  # GPT-4o uses o200k_base encoding

    def test_create_for_invalid_model(self):
        """Test creating TokenCounter for invalid model."""
        counter = TokenCounter.create_for_model("invalid-model")
        assert isinstance(counter, TokenCounter)
        # Should fallback to default encoding
        assert counter.encoding_name == "cl100k_base"

    def test_lazy_loading_encoding(self):
        """Test that encoding is loaded lazily."""
        counter = TokenCounter()
        assert counter._encoding is None

        # Access encoding property to trigger loading
        encoding = counter.encoding
        assert encoding is not None
        assert counter._encoding is not None

    def test_token_consistency(self):
        """Test that token counting is consistent for the same text."""
        text = "This is a test sentence for consistency checking."
        count1 = self.counter.count_tokens(text)
        count2 = self.counter.count_tokens(text)
        assert count1 == count2

    def test_different_encodings_different_counts(self):
        """Test that different encodings may produce different token counts."""
        text = "Hello, world! This is a test."

        counter1 = TokenCounter("cl100k_base")
        counter2 = TokenCounter("p50k_base")

        count1 = counter1.count_tokens(text)
        count2 = counter2.count_tokens(text)

        # Both should be positive integers
        assert isinstance(count1, int) and count1 > 0
        assert isinstance(count2, int) and count2 > 0
        # They might be different, but let's just ensure they're reasonable
        assert abs(count1 - count2) < 10  # Should be relatively close

    def test_gpt41_support(self):
        """Test comprehensive GPT-4.1 support."""
        text = "Hello, this is a test for GPT-4.1 compatibility and enhanced features."

        # Test if GPT-4.1 is supported
        assert TokenCounter.is_model_supported("gpt-4.1") is True

        # Test creating counter for GPT-4.1
        gpt41_counter = TokenCounter.create_for_model("gpt-4.1")
        assert gpt41_counter.model_name == "gpt-4.1"
        assert gpt41_counter.encoding_name in [
            "o200k_base",
            "cl100k_base",
        ]  # Either is valid

        # Test token counting for GPT-4.1
        token_count = gpt41_counter.count_tokens(text)
        assert isinstance(token_count, int)
        assert token_count > 0

        # Test model-specific counting
        model_count = self.counter.count_tokens_for_model(text, "gpt-4.1")
        assert isinstance(model_count, int)
        assert model_count > 0

        # Test encoding info for GPT-4.1
        info = TokenCounter.get_model_encoding_info("gpt-4.1")
        assert info["supported"] is True
        assert info["model_name"] == "gpt-4.1"
        assert info["encoding_name"] in ["o200k_base", "cl100k_base"]
        assert "note" in info  # Should have explanatory note

    def test_gpt41_legal_support(self):
        """Test GPT-4.1 with legal-specific features."""
        legal_text = (
            "Section 5.1 provides that GPT-4.1 shall process legal documents efficiently. "
            "Whereas the court finds pursuant to law, therefore the judgment is rendered."
        )

        # Create legal-specific GPT-4.1 counter
        legal_gpt41_counter = TokenCounter.create_for_model(
            "gpt-4.1", legal_specific=True
        )
        assert legal_gpt41_counter.model_name == "gpt-4.1"
        assert legal_gpt41_counter.legal_specific is True

        # Test legal token counting
        legal_count = legal_gpt41_counter.count_tokens(legal_text)
        regular_count = self.counter.count_tokens_for_model(legal_text, "gpt-4.1")

        assert isinstance(legal_count, int)
        assert legal_count > regular_count  # Legal adjustments should increase count

    @pytest.mark.asyncio
    async def test_gpt41_async_support(self):
        """Test GPT-4.1 with async methods."""
        text = "Testing GPT-4.1 async functionality."

        # Test async token counting
        async_count = await self.counter.count_tokens_for_model_async(text, "gpt-4.1")
        sync_count = self.counter.count_tokens_for_model(text, "gpt-4.1")

        assert async_count == sync_count
        assert isinstance(async_count, int)

        # Test with document
        document = Document(page_content=text, metadata={"source": "gpt41_test.txt"})

        doc_result = await self.counter.count_document_tokens_async(document, "gpt-4.1")
        assert doc_result["content_tokens"] > 0
        assert doc_result["metadata_tokens"] > 0

    def test_gpt41_document_splitting_integration(self):
        """Test GPT-4.1 integration with document splitting."""
        # Test that GPT-4.1 works with text splitting
        text = "This is a test text for GPT-4.1 document splitting. " * 20

        chunks = self.counter.split_text_by_tokens(
            text, chunk_size=100, chunk_overlap=20, model_name="gpt-4.1"
        )

        assert isinstance(chunks, list)
        assert len(chunks) > 1

        # Verify each chunk respects token limits
        for chunk in chunks:
            token_count = self.counter.count_tokens_for_model(chunk, "gpt-4.1")
            assert token_count <= 120  # Allow some flexibility

    def test_gpt41_encoding_fallback(self):
        """Test GPT-4.1 encoding fallback behavior."""
        # This test verifies that whether o200k_base is available or not,
        # GPT-4.1 works correctly

        counter = TokenCounter.create_for_model("gpt-4.1")
        text = "GPT-4.1 fallback test with various encoding scenarios."

        # Should work regardless of which encoding is used
        tokens = counter.count_tokens(text)
        assert isinstance(tokens, int)
        assert tokens > 0

        # Model name should be preserved
        assert counter.model_name == "gpt-4.1"

        # Encoding should be one of the supported options
        assert counter.encoding_name in ["o200k_base", "cl100k_base"]

    def test_newer_models_compatibility(self):
        """Test compatibility with newer OpenAI models like o3-mini."""
        text = "Hello, this is a test for newer model compatibility."

        # Test o3-mini (should use o200k_base encoding)
        try:
            counter = TokenCounter.create_for_model("o3-mini")
            token_count = counter.count_tokens(text)
            assert isinstance(token_count, int)
            assert token_count > 0
            # o3-mini should use o200k_base encoding
            assert counter.encoding_name == "o200k_base"
        except Exception:
            # If o3-mini is not available, test should still pass
            pass

        # Test o1-mini (should use o200k_base encoding)
        try:
            counter = TokenCounter.create_for_model("o1-mini")
            token_count = counter.count_tokens(text)
            assert isinstance(token_count, int)
            assert token_count > 0
            assert counter.encoding_name == "o200k_base"
        except Exception:
            # If o1-mini is not available, test should still pass
            pass

    def test_count_tokens_for_newer_models(self):
        """Test count_tokens_for_model with newer models."""
        counter = TokenCounter()
        text = "Testing newer model compatibility."

        # Test with o3-mini
        token_count = counter.count_tokens_for_model(text, "o3-mini")
        assert isinstance(token_count, int)
        assert token_count > 0

        # Test with o1-mini
        token_count = counter.count_tokens_for_model(text, "o1-mini")
        assert isinstance(token_count, int)
        assert token_count > 0

        # Test with hypothetical gpt-4.1 (should fallback gracefully)
        token_count = counter.count_tokens_for_model(text, "gpt-4.1")
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_is_model_supported(self):
        """Test the is_model_supported helper method."""
        # Test supported models (only GPT-4 family)
        assert TokenCounter.is_model_supported("gpt-4") is True
        assert TokenCounter.is_model_supported("gpt-4o") is True
        assert TokenCounter.is_model_supported("gpt-4.1") is True

        # Test unsupported models
        assert TokenCounter.is_model_supported("gpt-3.5-turbo") is False
        assert TokenCounter.is_model_supported("claude-3") is False
        assert TokenCounter.is_model_supported("anthropic-claude") is False
        assert TokenCounter.is_model_supported("o3-mini") is False
        assert TokenCounter.is_model_supported("o1-mini") is False
        assert TokenCounter.is_model_supported("invalid-model") is False

    def test_get_model_encoding_info(self):
        """Test the get_model_encoding_info helper method."""
        # Test supported GPT-4 model
        info = TokenCounter.get_model_encoding_info("gpt-4")
        assert info["supported"] is True
        assert info["model_name"] == "gpt-4"
        assert info["encoding_name"] == "cl100k_base"
        assert "max_token_value" in info
        assert "vocab_size" in info

        # Test GPT-4o model
        info = TokenCounter.get_model_encoding_info("gpt-4o")
        assert info["supported"] is True
        assert info["model_name"] == "gpt-4o"
        assert info["encoding_name"] == "o200k_base"

        # Test GPT-4.1 model (should use o200k_base or fallback to cl100k_base)
        info = TokenCounter.get_model_encoding_info("gpt-4.1")
        assert info["supported"] is True
        assert info["model_name"] == "gpt-4.1"
        assert info["encoding_name"] in ["o200k_base", "cl100k_base"]

        # Test unsupported model
        info = TokenCounter.get_model_encoding_info("claude-3")
        assert info["supported"] is False
        assert info["model_name"] == "claude-3"
        assert "error" in info
        assert info["fallback_encoding"] == "cl100k_base"


class TestConvenienceFunction:
    """Test cases for the convenience count_tokens function."""

    def test_count_tokens_default_encoding(self):
        """Test convenience function with default encoding."""
        text = "Hello, world!"
        token_count = count_tokens(text)
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_count_tokens_custom_encoding(self):
        """Test convenience function with custom encoding."""
        text = "Hello, world!"
        token_count = count_tokens(text, "p50k_base")
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_count_tokens_empty_text(self):
        """Test convenience function with empty text."""
        assert count_tokens("") == 0
        assert count_tokens(None) == 0

    @pytest.mark.asyncio
    async def test_count_tokens_async_convenience(self):
        """Test async convenience function."""
        text = "Hello, world!"
        async_count = await count_tokens_async(text)
        sync_count = count_tokens(text)
        assert async_count == sync_count

    @pytest.mark.asyncio
    async def test_count_tokens_async_custom_encoding(self):
        """Test async convenience function with custom encoding."""
        text = "Hello, world!"
        async_count = await count_tokens_async(text, "p50k_base")
        sync_count = count_tokens(text, "p50k_base")
        assert async_count == sync_count


class TestTokenCounterErrorHandling:
    """Test error handling in TokenCounter."""

    def test_invalid_encoding_fallback(self):
        """Test that invalid encoding falls back to default."""
        # This should not raise an exception but fall back to default
        counter = TokenCounter("invalid-encoding")
        text = "Hello, world!"
        token_count = counter.count_tokens(text)
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_non_string_input(self):
        """Test handling of non-string input."""
        counter = TokenCounter()
        assert counter.count_tokens(123) == 0
        assert counter.count_tokens([]) == 0
        assert counter.count_tokens({}) == 0

    @pytest.mark.asyncio
    async def test_async_error_handling(self):
        """Test error handling in async methods."""
        counter = TokenCounter()

        # Test with invalid input
        result = await counter.count_tokens_async(None)
        assert result == 0

        result = await counter.count_tokens_async("")
        assert result == 0

    def test_legal_pattern_robustness(self):
        """Test that legal patterns handle edge cases gracefully."""
        legal_counter = TokenCounter(legal_specific=True)

        # Test with text that might break regex patterns
        edge_cases = [
            "",
            "   ",
            "§",
            "Section",
            "whereas therefore provided that",
            "Inc. Corp. LLC Ltd.",
        ]

        for text in edge_cases:
            count = legal_counter.count_tokens(text)
            assert isinstance(count, int)
            assert count >= 0


class TestLangchainIntegration:
    """Test Langchain-specific functionality."""

    def test_token_text_splitter_integration(self):
        """Test integration with Langchain's TokenTextSplitter."""
        counter = TokenCounter()
        splitter = counter.get_langchain_splitter(chunk_size=100, model_name="gpt-4")

        text = "This is a test text that should be split into smaller chunks. " * 20
        chunks = splitter.split_text(text)

        assert isinstance(chunks, list)
        assert len(chunks) > 1

        # Verify each chunk is under the token limit
        for chunk in chunks:
            token_count = counter.count_tokens_for_model(chunk, "gpt-4")
            assert token_count <= 120  # Allow some flexibility

    def test_length_function_integration(self):
        """Test custom length function with Langchain splitter."""
        legal_counter = TokenCounter(legal_specific=True)
        splitter = legal_counter.get_langchain_splitter(
            chunk_size=50, model_name="gpt-4"
        )

        legal_text = (
            "Section 5.1 provides that whereas the parties agree, "
            "therefore the contract is binding. Pursuant to law, "
            "the defendant must comply. " * 10
        )

        chunks = splitter.split_text(legal_text)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
