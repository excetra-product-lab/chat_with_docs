"""Text document loader service."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from app.utils.encoding_utils import (
    detect_file_encoding,
    get_encoding_fallback_list,
    try_decode_with_fallback,
    validate_text_encoding,
)

from .base_loader import BaseDocumentLoader

logger = logging.getLogger(__name__)


class TextDocumentLoader(BaseDocumentLoader):
    """Document loader for text files."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialize the text document loader.

        Args:
            encoding_name: The tokenizer encoding to use for token counting
        """
        super().__init__(encoding_name)
        self.logger = logging.getLogger(__name__)

    async def load_document(
        self,
        file_path: str,
        password: Optional[str] = None,
        preserve_layout: bool = True,
        encoding_info: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Load a text document.

        Args:
            file_path: Path to the text file
            password: Not used for text files
            preserve_layout: Not used for text files (always preserved)
            encoding_info: Optional pre-detected encoding information

        Returns:
            List of Document objects (typically one document)
        """
        self.logger.info(f"Loading text document: {file_path}")

        # If encoding info is provided, use it; otherwise detect encoding
        if not encoding_info:
            with open(file_path, "rb") as f:
                file_content = f.read()
            encoding_info = await detect_file_encoding(file_content)

        return await self._load_text_with_encoding_fallback(file_path, encoding_info)

    async def _load_text_with_encoding_fallback(
        self, file_path: str, encoding_info: Dict[str, Any]
    ) -> List[Document]:
        """Load text file with encoding fallback mechanism."""

        # Try the detected encoding first
        detected_encoding = encoding_info.get("encoding", "utf-8")

        try:
            loader = TextLoader(file_path, encoding=detected_encoding)
            documents = loader.load()

            # Validate the loaded text
            if documents and documents[0].page_content:
                validation = await validate_text_encoding(
                    documents[0].page_content, detected_encoding
                )

                if validation["is_valid"]:
                    # Add metadata
                    for doc in documents:
                        doc.metadata.update(
                            {
                                "source": file_path,
                                "file_type": "text",
                                "encoding": detected_encoding,
                                "encoding_confidence": encoding_info.get("confidence", 0.0),
                                "encoding_validation": validation,
                                "file_size": Path(file_path).stat().st_size,
                            }
                        )

                    self.logger.info(
                        f"Successfully loaded text file with encoding {detected_encoding}"
                    )
                    return documents
                else:
                    self.logger.warning(
                        f"Text validation failed for {detected_encoding}, trying fallback"
                    )

        except Exception as e:
            self.logger.warning(
                f"Failed to load with detected encoding {detected_encoding}: {str(e)}"
            )

        # Fallback to trying multiple encodings
        return await self._load_with_encoding_fallback_list(file_path, detected_encoding)

    async def _load_with_encoding_fallback_list(
        self, file_path: str, detected_encoding: Optional[str] = None
    ) -> List[Document]:
        """Try loading with multiple encoding fallbacks."""

        # Get fallback encoding list
        encoding_list = get_encoding_fallback_list(detected_encoding)

        # Read file content for fallback decoding
        with open(file_path, "rb") as f:
            file_content = f.read()

        # Try decoding with fallback mechanism
        decode_result = await try_decode_with_fallback(file_content, encoding_list)

        if decode_result["success"]:
            # Create document from successfully decoded text
            text_content = decode_result["text"]
            encoding_used = decode_result["encoding"]

            # Validate the decoded text
            validation = await validate_text_encoding(text_content, encoding_used)

            metadata = {
                "source": file_path,
                "file_type": "text",
                "encoding": encoding_used,
                "encoding_confidence": decode_result.get("confidence", 0.0),
                "encoding_validation": validation,
                "encoding_attempts": decode_result.get("attempts", []),
                "file_size": len(file_content),
            }

            document = Document(page_content=text_content, metadata=metadata)

            self.logger.info(
                f"Successfully loaded text file with fallback encoding {encoding_used}"
            )
            return [document]

        else:
            # All encodings failed
            error_msg = f"Could not decode text file {file_path} with any encoding"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    async def load_text_with_custom_encoding(self, file_path: str, encoding: str) -> List[Document]:
        """Load text file with a specific encoding.

        Args:
            file_path: Path to the text file
            encoding: Specific encoding to use

        Returns:
            List of Document objects

        Raises:
            ValueError: If the file cannot be loaded with the specified encoding
        """
        try:
            loader = TextLoader(file_path, encoding=encoding)
            documents = loader.load()

            # Add metadata
            for doc in documents:
                doc.metadata.update(
                    {
                        "source": file_path,
                        "file_type": "text",
                        "encoding": encoding,
                        "encoding_forced": True,
                        "file_size": Path(file_path).stat().st_size,
                    }
                )

            self.logger.info(f"Successfully loaded text file with custom encoding {encoding}")
            return documents

        except Exception as e:
            error_msg = f"Failed to load text file {file_path} with encoding {encoding}: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    async def detect_text_encoding(self, file_path: str) -> Dict[str, Any]:
        """Detect encoding of a text file without loading it.

        Args:
            file_path: Path to the text file

        Returns:
            Dictionary containing encoding detection results
        """
        try:
            with open(file_path, "rb") as f:
                file_content = f.read()

            encoding_info = await detect_file_encoding(file_content)
            encoding_info["file_path"] = file_path
            encoding_info["file_size"] = len(file_content)

            return encoding_info

        except Exception as e:
            self.logger.error(f"Error detecting encoding for {file_path}: {str(e)}")
            return {"error": str(e), "file_path": file_path}

    async def validate_text_file_encoding(self, file_path: str, encoding: str) -> Dict[str, Any]:
        """Validate that a text file can be properly decoded with a specific encoding.

        Args:
            file_path: Path to the text file
            encoding: Encoding to validate

        Returns:
            Dictionary containing validation results
        """
        try:
            with open(file_path, "r", encoding=encoding) as f:
                text_content = f.read()

            validation = await validate_text_encoding(text_content, encoding)
            validation["file_path"] = file_path
            validation["file_size"] = Path(file_path).stat().st_size

            return validation

        except Exception as e:
            self.logger.error(f"Error validating encoding {encoding} for {file_path}: {str(e)}")
            return {
                "is_valid": False,
                "error": str(e),
                "file_path": file_path,
                "encoding": encoding,
            }

    async def get_text_file_stats(self, file_path: str) -> Dict[str, Any]:
        """Get statistics about a text file.

        Args:
            file_path: Path to the text file

        Returns:
            Dictionary containing file statistics
        """
        try:
            # Detect encoding first
            encoding_info = await self.detect_text_encoding(file_path)

            if "error" in encoding_info:
                return encoding_info

            # Load the file to get content statistics
            documents = await self.load_document(file_path, encoding_info=encoding_info)

            if not documents:
                return {"error": "No content loaded", "file_path": file_path}

            content = documents[0].page_content
            lines = content.split("\n")

            stats = {
                "file_path": file_path,
                "file_size": Path(file_path).stat().st_size,
                "encoding": encoding_info.get("encoding", "unknown"),
                "encoding_confidence": encoding_info.get("confidence", 0.0),
                "char_count": len(content),
                "word_count": len(content.split()),
                "line_count": len(lines),
                "empty_lines": sum(1 for line in lines if not line.strip()),
                "max_line_length": max(len(line) for line in lines) if lines else 0,
                "avg_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0,
            }

            return stats

        except Exception as e:
            self.logger.error(f"Error getting stats for {file_path}: {str(e)}")
            return {"error": str(e), "file_path": file_path}

    def get_supported_mime_types(self) -> List[str]:
        """Get the MIME types supported by this loader.

        Returns:
            List of supported MIME types
        """
        return [
            "text/plain",
            "text/csv",
            "text/tab-separated-values",
            "application/csv",
        ]

    def get_supported_extensions(self) -> List[str]:
        """Get the file extensions supported by this loader.

        Returns:
            List of supported file extensions
        """
        return [
            ".txt",
            ".text",
            ".csv",
            ".tsv",
            ".log",
            ".md",
            ".markdown",
            ".rst",
            ".rtf",
        ]
