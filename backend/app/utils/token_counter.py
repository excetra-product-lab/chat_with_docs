"""Token counting utility using tiktoken for various text encoding models."""

import logging
from typing import Optional

import tiktoken

logger = logging.getLogger(__name__)


class TokenCounter:
    """Utility class for counting tokens in text using tiktoken."""

    # Common encoding models
    DEFAULT_ENCODING = "cl100k_base"  # Used by GPT-4, GPT-3.5-turbo
    GPT4_ENCODING = "cl100k_base"
    GPT3_ENCODING = "p50k_base"  # Used by text-davinci-003, text-davinci-002
    CODEX_ENCODING = "p50k_base"  # Used by code-davinci-002

    def __init__(self, encoding_name: str = DEFAULT_ENCODING):
        """
        Initialize the token counter with a specific encoding.

        Args:
            encoding_name: The name of the tiktoken encoding to use
        """
        self.encoding_name = encoding_name
        self._encoding = None
        self.logger = logging.getLogger(__name__)

    @property
    def encoding(self):
        """Lazy load the encoding to avoid initialization overhead."""
        if self._encoding is None:
            try:
                self._encoding = tiktoken.get_encoding(self.encoding_name)
            except Exception as e:
                self.logger.error(f"Failed to load encoding {self.encoding_name}: {e}")
                # Fallback to default encoding
                self._encoding = tiktoken.get_encoding(self.DEFAULT_ENCODING)
        return self._encoding

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Args:
            text: The text to count tokens for

        Returns:
            int: The number of tokens in the text
        """
        if not text or not isinstance(text, str):
            return 0

        try:
            tokens = self.encoding.encode(text)
            return len(tokens)
        except Exception as e:
            self.logger.error(f"Error counting tokens: {e}")
            # Fallback to character-based estimation (rough approximation)
            return len(text) // 4

    def count_tokens_for_model(self, text: str, model_name: str) -> int:
        """
        Count tokens for a specific model by using the appropriate encoding.

        Args:
            text: The text to count tokens for
            model_name: The model name (e.g., 'gpt-4', 'gpt-3.5-turbo')

        Returns:
            int: The number of tokens in the text
        """
        try:
            encoding = tiktoken.encoding_for_model(model_name)
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            self.logger.warning(f"Could not get encoding for model {model_name}: {e}")
            # Fallback to default counting
            return self.count_tokens(text)

    def estimate_cost(self, text: str, model_name: str, cost_per_1k_tokens: float) -> float:
        """
        Estimate the cost of processing text with a given model.

        Args:
            text: The text to estimate cost for
            model_name: The model name
            cost_per_1k_tokens: Cost per 1000 tokens in USD

        Returns:
            float: Estimated cost in USD
        """
        token_count = self.count_tokens_for_model(text, model_name)
        return (token_count / 1000) * cost_per_1k_tokens

    def get_encoding_info(self) -> dict:
        """
        Get information about the current encoding.

        Returns:
            dict: Information about the encoding
        """
        return {
            "encoding_name": self.encoding_name,
            "max_token_value": self.encoding.max_token_value,
            "vocab_size": self.encoding.n_vocab,
        }

    @staticmethod
    def get_available_encodings() -> list:
        """
        Get a list of available tiktoken encodings.

        Returns:
            list: List of available encoding names
        """
        try:
            return tiktoken.list_encoding_names()
        except Exception as e:
            logger.error(f"Error getting available encodings: {e}")
            return ["cl100k_base", "p50k_base", "r50k_base"]

    @staticmethod
    def create_for_model(model_name: str) -> "TokenCounter":
        """
        Create a TokenCounter instance configured for a specific model.

        Args:
            model_name: The model name (e.g., 'gpt-4', 'gpt-3.5-turbo', 'o3-mini')

        Returns:
            TokenCounter: Configured token counter instance
        """
        try:
            encoding = tiktoken.encoding_for_model(model_name)
            return TokenCounter(encoding.name)
        except Exception as e:
            logger.warning(f"Could not create counter for model {model_name}: {e}")
            return TokenCounter()  # Use default encoding

    @staticmethod
    def is_model_supported(model_name: str) -> bool:
        """
        Check if a model is supported by tiktoken.

        Args:
            model_name: The model name to check

        Returns:
            bool: True if the model is supported, False otherwise
        """
        try:
            tiktoken.encoding_for_model(model_name)
            return True
        except Exception:
            return False

    @staticmethod
    def get_model_encoding_info(model_name: str) -> dict:
        """
        Get encoding information for a specific model.

        Args:
            model_name: The model name

        Returns:
            dict: Information about the model's encoding, or error info if not supported
        """
        try:
            encoding = tiktoken.encoding_for_model(model_name)
            return {
                "model_name": model_name,
                "encoding_name": encoding.name,
                "supported": True,
                "max_token_value": encoding.max_token_value,
                "vocab_size": encoding.n_vocab,
            }
        except Exception as e:
            return {
                "model_name": model_name,
                "supported": False,
                "error": str(e),
                "fallback_encoding": TokenCounter.DEFAULT_ENCODING,
            }


# Global instance for convenience
default_token_counter = TokenCounter()


def count_tokens(text: str, encoding_name: Optional[str] = None) -> int:
    """
    Convenience function to count tokens in text.

    Args:
        text: The text to count tokens for
        encoding_name: Optional encoding name, uses default if not provided

    Returns:
        int: The number of tokens in the text
    """
    if encoding_name:
        counter = TokenCounter(encoding_name)
        return counter.count_tokens(text)
    return default_token_counter.count_tokens(text)
