"""Token counting utility using tiktoken and Langchain for various text encoding models."""

import asyncio
import logging
import re
from collections.abc import Callable
from typing import Any

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter

logger = logging.getLogger(__name__)


class TokenCounter:
    """Utility class for counting tokens in text using tiktoken and Langchain."""

    # Common encoding models
    DEFAULT_ENCODING = "cl100k_base"  # Used by GPT-4, GPT-3.5-turbo
    GPT4_ENCODING = "cl100k_base"
    GPT3_ENCODING = "p50k_base"  # Used by text-davinci-003, text-davinci-002
    CODEX_ENCODING = "p50k_base"  # Used by code-davinci-002

    # Legal-specific token patterns
    LEGAL_PATTERNS = {
        "legal_citations": re.compile(
            r"\b(?:\d+\s+[A-Z][a-z]+\.?\s+\d+|[A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+)\b"
        ),
        "statute_references": re.compile(r"\b(?:§|Section|Sec\.?)\s*\d+(?:\.\d+)*\b"),
        "legal_terms": re.compile(
            r"\b(?:whereas|therefore|provided\s+that|subject\s+to|notwithstanding|pursuant\s+to)\b",
            re.IGNORECASE,
        ),
        "case_law": re.compile(r"\b\d+\s+[A-Z][a-z]+\.?\s+\d+d?\s+\d+\b"),
        "legal_abbreviations": re.compile(
            r"\b(?:Inc\.|Corp\.|LLC|Ltd\.|L\.P\.|P\.C\.|LLP)\b"
        ),
    }

    def __init__(
        self,
        encoding_name: str = DEFAULT_ENCODING,
        model_name: str | None = None,
        legal_specific: bool = False,
    ):
        """
        Initialize the token counter with a specific encoding.

        Args:
            encoding_name: The name of the tiktoken encoding to use
            model_name: Optional model name for automatic encoding selection
            legal_specific: Whether to use legal-specific tokenization patterns
        """
        self.encoding_name = encoding_name
        self.model_name = model_name
        self.legal_specific = legal_specific
        self._encoding = None
        self._langchain_splitter = None
        self.logger = logging.getLogger(__name__)

        # Set encoding based on model if provided
        if model_name:
            self.encoding_name = self._get_encoding_for_model(model_name)

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

    def _get_encoding_for_model(self, model_name: str) -> str:
        """Get the appropriate encoding for a given model."""
        try:
            # Special handling for GPT-4.1
            if model_name.lower() == "gpt-4.1":
                try:
                    # Try o200k_base encoding first (likely encoding for GPT-4.1)
                    tiktoken.get_encoding("o200k_base")
                    return "o200k_base"
                except Exception:
                    # Fallback to cl100k_base for GPT-4.1
                    self.logger.info(
                        f"Using cl100k_base encoding for {model_name} (o200k_base not available)"
                    )
                    return self.DEFAULT_ENCODING

            encoding = tiktoken.encoding_for_model(model_name)
            return encoding.name
        except Exception:
            # For unsupported models, use default encoding
            return self.DEFAULT_ENCODING

    def get_langchain_splitter(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str | None = None,
    ) -> TokenTextSplitter:
        """Get or create a Langchain TokenTextSplitter instance."""
        effective_model = model_name or self.model_name or "gpt-4"

        # Ensure chunk_overlap is smaller than chunk_size
        if chunk_overlap >= chunk_size:
            chunk_overlap = max(0, chunk_size // 5)  # Use 20% of chunk_size as overlap
            self.logger.debug(
                f"Adjusted chunk_overlap to {chunk_overlap} (chunk_size: {chunk_size})"
            )

        # Handle GPT-4.1 by using a supported model for Langchain while preserving accuracy
        langchain_model = effective_model
        if effective_model.lower() == "gpt-4.1":
            # Use gpt-4 for Langchain TokenTextSplitter (similar tokenization)
            # but keep GPT-4.1 for our custom token counting
            langchain_model = "gpt-4"
            self.logger.debug(
                f"Using gpt-4 for Langchain splitter instead of {effective_model}"
            )

        splitter_key = (
            f"{langchain_model}_{chunk_size}_{chunk_overlap}_{self.legal_specific}"
        )

        if (
            self._langchain_splitter is None
            or getattr(self._langchain_splitter, "_splitter_key", None) != splitter_key
        ):
            try:
                # Create length function for legal-specific tokenization
                length_function = (
                    self.create_length_function() if self.legal_specific else None
                )

                self._langchain_splitter = TokenTextSplitter(
                    model_name=langchain_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=length_function,
                )

                # Store metadata for future reference
                self._langchain_splitter._effective_model = effective_model
                self._langchain_splitter._splitter_key = splitter_key

            except Exception as e:
                self.logger.warning(
                    f"Failed to create TokenTextSplitter for {langchain_model}: {e}"
                )
                # Fallback to basic splitter with corrected parameters
                safe_overlap = min(
                    chunk_overlap, chunk_size // 4
                )  # Ensure safe overlap
                self._langchain_splitter = TokenTextSplitter(
                    model_name="gpt-4",
                    chunk_size=chunk_size,
                    chunk_overlap=safe_overlap,
                )
                self._langchain_splitter._effective_model = effective_model
                self._langchain_splitter._splitter_key = splitter_key

        return self._langchain_splitter

    def create_length_function(self) -> Callable[[str], int]:
        """Create a custom length function for legal-specific token counting."""

        def legal_length_function(text: str) -> int:
            """Custom length function that accounts for legal-specific terms."""
            base_tokens = self.count_tokens(text)

            if self.legal_specific:
                # Adjust token count for legal patterns
                legal_adjustments = 0

                for pattern_name, pattern in self.LEGAL_PATTERNS.items():
                    matches = pattern.findall(text)
                    # Legal terms tend to be tokenized differently, add small adjustment
                    legal_adjustments += len(matches) * 0.5

                return int(base_tokens + legal_adjustments)

            return base_tokens

        return legal_length_function

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
            token_count = len(tokens)

            # Apply legal-specific adjustments if enabled
            if self.legal_specific:
                token_count = self._apply_legal_adjustments(text, token_count)

            return token_count

        except Exception as e:
            self.logger.error(f"Error counting tokens: {e}")
            # Fallback to character-based estimation (rough approximation)
            return len(text) // 4

    async def count_tokens_async(self, text: str) -> int:
        """
        Async version of count_tokens for compatibility with DocumentSplitter.

        Args:
            text: The text to count tokens for

        Returns:
            int: The number of tokens in the text
        """
        # For now, run the sync version in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.count_tokens, text)

    def _apply_legal_adjustments(self, text: str, base_tokens: int) -> int:
        """Apply legal-specific token count adjustments."""
        adjustments = 0

        for pattern_name, pattern in self.LEGAL_PATTERNS.items():
            matches = pattern.findall(text)

            # Different patterns have different token impacts
            if pattern_name == "legal_citations":
                adjustments += len(matches) * 2  # Citations tend to use more tokens
            elif pattern_name == "statute_references":
                adjustments += len(matches) * 1.5  # Section references are compact
            elif pattern_name == "legal_terms":
                adjustments += (
                    len(matches) * 0.5
                )  # Legal terms might be tokenized specially
            elif pattern_name == "case_law":
                adjustments += len(matches) * 2.5  # Case citations are token-heavy
            elif pattern_name == "legal_abbreviations":
                adjustments -= len(matches) * 0.5  # Abbreviations save tokens

        return int(base_tokens + adjustments)

    def count_tokens_for_model(self, text: str, model_name: str) -> int:
        """
        Count tokens for a specific model by using the appropriate encoding.

        Args:
            text: The text to count tokens for
            model_name: The model name (e.g., 'gpt-4', 'gpt-4o', 'gpt-4.1')

        Returns:
            int: The number of tokens in the text
        """
        try:
            # Special handling for GPT-4.1
            if model_name.lower() == "gpt-4.1":
                try:
                    # Try to use o200k_base encoding for GPT-4.1
                    encoding = tiktoken.get_encoding("o200k_base")
                    tokens = encoding.encode(text)
                    token_count = len(tokens)

                    # Apply legal adjustments if enabled
                    if self.legal_specific:
                        token_count = self._apply_legal_adjustments(text, token_count)

                    return token_count
                except Exception:
                    # Fallback to cl100k_base for GPT-4.1
                    self.logger.debug(f"Using cl100k_base fallback for {model_name}")
                    encoding = tiktoken.get_encoding(self.DEFAULT_ENCODING)
                    tokens = encoding.encode(text)
                    token_count = len(tokens)

                    # Apply legal adjustments if enabled
                    if self.legal_specific:
                        token_count = self._apply_legal_adjustments(text, token_count)

                    return token_count

            encoding = tiktoken.encoding_for_model(model_name)
            tokens = encoding.encode(text)
            token_count = len(tokens)

            # Apply legal adjustments if enabled
            if self.legal_specific:
                token_count = self._apply_legal_adjustments(text, token_count)

            return token_count

        except Exception as e:
            self.logger.warning(f"Could not get encoding for model {model_name}: {e}")
            # Fallback to default counting
            return self.count_tokens(text)

    async def count_tokens_for_model_async(self, text: str, model_name: str) -> int:
        """Async version of count_tokens_for_model."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.count_tokens_for_model, text, model_name
        )

    def count_document_tokens(
        self, document: Document, model_name: str | None = None
    ) -> dict[str, int]:
        """
        Count tokens for a Langchain Document, including metadata.

        Args:
            document: The Langchain Document to analyze
            model_name: Optional model name for specific tokenizer

        Returns:
            Dict with token counts for content, metadata, and total
        """
        effective_model = model_name or self.model_name

        content_tokens = 0
        metadata_tokens = 0

        # Count content tokens
        if document.page_content:
            if effective_model:
                content_tokens = self.count_tokens_for_model(
                    document.page_content, effective_model
                )
            else:
                content_tokens = self.count_tokens(document.page_content)

        # Count metadata tokens (convert metadata to string)
        if document.metadata:
            metadata_str = str(document.metadata)
            if effective_model:
                metadata_tokens = self.count_tokens_for_model(
                    metadata_str, effective_model
                )
            else:
                metadata_tokens = self.count_tokens(metadata_str)

        return {
            "content_tokens": content_tokens,
            "metadata_tokens": metadata_tokens,
            "total_tokens": content_tokens + metadata_tokens,
        }

    async def count_document_tokens_async(
        self, document: Document, model_name: str | None = None
    ) -> dict[str, int]:
        """Async version of count_document_tokens."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.count_document_tokens, document, model_name
        )

    def count_documents_tokens(
        self, documents: list[Document], model_name: str | None = None
    ) -> dict[str, Any]:
        """
        Count tokens for multiple documents with detailed breakdown.

        Args:
            documents: List of Langchain Documents
            model_name: Optional model name for specific tokenizer

        Returns:
            Dict with detailed token analysis
        """
        total_content = 0
        total_metadata = 0
        document_details = []

        for i, doc in enumerate(documents):
            doc_tokens = self.count_document_tokens(doc, model_name)
            total_content += doc_tokens["content_tokens"]
            total_metadata += doc_tokens["metadata_tokens"]

            document_details.append(
                {
                    "document_index": i,
                    "content_tokens": doc_tokens["content_tokens"],
                    "metadata_tokens": doc_tokens["metadata_tokens"],
                    "total_tokens": doc_tokens["total_tokens"],
                }
            )

        return {
            "total_documents": len(documents),
            "total_content_tokens": total_content,
            "total_metadata_tokens": total_metadata,
            "total_tokens": total_content + total_metadata,
            "average_tokens_per_document": (total_content + total_metadata)
            / len(documents)
            if documents
            else 0,
            "document_details": document_details,
        }

    async def count_documents_tokens_async(
        self, documents: list[Document], model_name: str | None = None
    ) -> dict[str, Any]:
        """Async version of count_documents_tokens."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.count_documents_tokens, documents, model_name
        )

    def count_sections_tokens(
        self,
        text: str,
        section_pattern: str = r"\n\s*(?:SECTION|Section|§)\s*\d+",
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Count tokens for document sections based on a pattern.

        Args:
            text: The full document text
            section_pattern: Regex pattern to identify section boundaries
            model_name: Optional model name for specific tokenizer

        Returns:
            Dict with section-wise token analysis
        """
        try:
            # Split text into sections
            pattern = re.compile(section_pattern, re.MULTILINE | re.IGNORECASE)
            sections = pattern.split(text)

            section_details = []
            total_tokens = 0

            for i, section in enumerate(sections):
                if section.strip():  # Skip empty sections
                    if model_name:
                        tokens = self.count_tokens_for_model(section, model_name)
                    else:
                        tokens = self.count_tokens(section)

                    total_tokens += tokens
                    section_details.append(
                        {
                            "section_index": i,
                            "section_preview": section[:100] + "..."
                            if len(section) > 100
                            else section,
                            "character_count": len(section),
                            "token_count": tokens,
                        }
                    )

            return {
                "total_sections": len(section_details),
                "total_tokens": total_tokens,
                "average_tokens_per_section": total_tokens / len(section_details)
                if section_details
                else 0,
                "section_details": section_details,
            }

        except Exception as e:
            self.logger.error(f"Error counting section tokens: {e}")
            # Fallback to single section
            tokens = (
                self.count_tokens_for_model(text, model_name)
                if model_name
                else self.count_tokens(text)
            )
            return {
                "total_sections": 1,
                "total_tokens": tokens,
                "average_tokens_per_section": tokens,
                "section_details": [
                    {
                        "section_index": 0,
                        "section_preview": text[:100],
                        "character_count": len(text),
                        "token_count": tokens,
                    }
                ],
            }

    def split_text_by_tokens(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: str | None = None,
    ) -> list[str]:
        """
        Split text using Langchain's TokenTextSplitter.

        Args:
            text: Text to split
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Token overlap between chunks
            model_name: Model name for tokenizer

        Returns:
            List of text chunks
        """
        splitter = self.get_langchain_splitter(chunk_size, chunk_overlap, model_name)
        return splitter.split_text(text)

    def estimate_cost(
        self, text: str, model_name: str, cost_per_1k_tokens: float
    ) -> float:
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
        info = {
            "encoding_name": self.encoding_name,
            "model_name": self.model_name,
            "legal_specific": self.legal_specific,
            "max_token_value": self.encoding.max_token_value,
            "vocab_size": self.encoding.n_vocab,
        }

        if self.legal_specific:
            info["legal_patterns"] = list(self.LEGAL_PATTERNS.keys())

        return info

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
    def create_for_model(
        model_name: str, legal_specific: bool = False
    ) -> "TokenCounter":
        """
        Create a TokenCounter instance configured for a specific model.

        Args:
            model_name: The model name (e.g., 'gpt-4', 'gpt-4o', 'gpt-4.1')
            legal_specific: Whether to enable legal-specific tokenization

        Returns:
            TokenCounter: Configured token counter instance
        """
        try:
            if model_name.lower() == "gpt-4.1":
                # Special handling for GPT-4.1 - use o200k_base if available, fallback to cl100k_base
                try:
                    # Try o200k_base encoding first (likely encoding for GPT-4.1)
                    tiktoken.get_encoding("o200k_base")
                    return TokenCounter(
                        encoding_name="o200k_base",
                        model_name=model_name,
                        legal_specific=legal_specific,
                    )
                except Exception:
                    # Fallback to cl100k_base for GPT-4.1
                    logger.info(
                        f"Using cl100k_base encoding for {model_name} (o200k_base not available)"
                    )
                    return TokenCounter(
                        encoding_name=TokenCounter.DEFAULT_ENCODING,
                        model_name=model_name,
                        legal_specific=legal_specific,
                    )
            else:
                encoding = tiktoken.encoding_for_model(model_name)
                return TokenCounter(
                    encoding_name=encoding.name,
                    model_name=model_name,
                    legal_specific=legal_specific,
                )
        except Exception as e:
            logger.warning(f"Could not create counter for model {model_name}: {e}")
            # Always preserve the model name even if encoding detection fails
            return TokenCounter(
                encoding_name=TokenCounter.DEFAULT_ENCODING,
                model_name=model_name,
                legal_specific=legal_specific,
            )

    @staticmethod
    def is_model_supported(model_name: str) -> bool:
        """
        Check if a model is supported by tiktoken or our custom handlers.

        Args:
            model_name: The model name to check

        Returns:
            bool: True if the model is supported, False otherwise
        """
        try:
            # Special case for GPT-4.1 - we support it with fallback
            if model_name.lower() == "gpt-4.1":
                return True

            # Only support GPT-4 family models
            if any(
                gpt4_variant in model_name.lower()
                for gpt4_variant in ["gpt-4", "gpt-4o"]
            ):
                tiktoken.encoding_for_model(model_name)
                return True

            return False
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
            # Special handling for GPT-4.1
            if model_name.lower() == "gpt-4.1":
                try:
                    # Try to use o200k_base encoding
                    encoding = tiktoken.get_encoding("o200k_base")
                    return {
                        "model_name": model_name,
                        "encoding_name": "o200k_base",
                        "supported": True,
                        "max_token_value": encoding.max_token_value,
                        "vocab_size": encoding.n_vocab,
                        "note": "GPT-4.1 using o200k_base encoding (estimated)",
                    }
                except Exception:
                    # Fallback for GPT-4.1
                    encoding = tiktoken.get_encoding(TokenCounter.DEFAULT_ENCODING)
                    return {
                        "model_name": model_name,
                        "encoding_name": TokenCounter.DEFAULT_ENCODING,
                        "supported": True,
                        "max_token_value": encoding.max_token_value,
                        "vocab_size": encoding.n_vocab,
                        "note": "GPT-4.1 using cl100k_base fallback encoding",
                        "fallback_reason": "o200k_base not available",
                    }

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


def count_tokens(text: str, encoding_name: str | None = None) -> int:
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


async def count_tokens_async(text: str, encoding_name: str | None = None) -> int:
    """
    Async convenience function to count tokens in text.

    Args:
        text: The text to count tokens for
        encoding_name: Optional encoding name, uses default if not provided

    Returns:
        int: The number of tokens in the text
    """
    if encoding_name:
        counter = TokenCounter(encoding_name)
        return await counter.count_tokens_async(text)
    return await default_token_counter.count_tokens_async(text)
