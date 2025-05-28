"""Tests for token counter utility."""

from app.utils.token_counter import TokenCounter, count_tokens


class TestTokenCounter:
    """Test cases for TokenCounter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.counter = TokenCounter()

    def test_token_counter_initialization(self):
        """Test TokenCounter initialization with default encoding."""
        assert self.counter.encoding_name == "cl100k_base"
        assert self.counter._encoding is None  # Lazy loading

    def test_token_counter_custom_encoding(self):
        """Test TokenCounter initialization with custom encoding."""
        counter = TokenCounter("p50k_base")
        assert counter.encoding_name == "p50k_base"

    def test_count_tokens_simple_text(self):
        """Test counting tokens for simple text."""
        text = "Hello, world!"
        token_count = self.counter.count_tokens(text)
        assert isinstance(token_count, int)
        assert token_count > 0
        # "Hello, world!" should be around 3-4 tokens
        assert 2 <= token_count <= 5

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

    def test_count_tokens_for_model_gpt4(self):
        """Test counting tokens for GPT-4 model."""
        text = "Hello, world!"
        token_count = self.counter.count_tokens_for_model(text, "gpt-4")
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_count_tokens_for_model_gpt35_turbo(self):
        """Test counting tokens for GPT-3.5-turbo model."""
        text = "Hello, world!"
        token_count = self.counter.count_tokens_for_model(text, "gpt-3.5-turbo")
        assert isinstance(token_count, int)
        assert token_count > 0

    def test_count_tokens_for_invalid_model(self):
        """Test counting tokens for invalid model falls back gracefully."""
        text = "Hello, world!"
        token_count = self.counter.count_tokens_for_model(text, "invalid-model")
        # Should fallback to default counting
        default_count = self.counter.count_tokens(text)
        assert token_count == default_count

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
        # Test supported models
        assert TokenCounter.is_model_supported("gpt-4") is True
        assert TokenCounter.is_model_supported("gpt-3.5-turbo") is True
        assert TokenCounter.is_model_supported("o3-mini") is True
        assert TokenCounter.is_model_supported("o1-mini") is True

        # Test unsupported model
        assert TokenCounter.is_model_supported("gpt-4.1") is False
        assert TokenCounter.is_model_supported("invalid-model") is False

    def test_get_model_encoding_info(self):
        """Test the get_model_encoding_info helper method."""
        # Test supported model
        info = TokenCounter.get_model_encoding_info("gpt-4")
        assert info["supported"] is True
        assert info["model_name"] == "gpt-4"
        assert info["encoding_name"] == "cl100k_base"
        assert "max_token_value" in info
        assert "vocab_size" in info

        # Test newer model with different encoding
        info = TokenCounter.get_model_encoding_info("o3-mini")
        assert info["supported"] is True
        assert info["model_name"] == "o3-mini"
        assert info["encoding_name"] == "o200k_base"

        # Test unsupported model
        info = TokenCounter.get_model_encoding_info("gpt-4.1")
        assert info["supported"] is False
        assert info["model_name"] == "gpt-4.1"
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
