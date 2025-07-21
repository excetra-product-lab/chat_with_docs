"""Base document loader interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialize the document loader.

        Args:
            encoding_name: The tokenizer encoding to use for token counting
        """
        self.encoding_name = encoding_name

    @abstractmethod
    async def load_document(
        self,
        file_path: str,
        password: Optional[str] = None,
        preserve_layout: bool = True,
        encoding_info: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Load a document from the given file path.

        Args:
            file_path: Path to the document file
            password: Optional password for protected documents
            preserve_layout: Whether to preserve document layout information
            encoding_info: Optional encoding information for text files

        Returns:
            List of Document objects
        """

    @abstractmethod
    def get_supported_mime_types(self) -> List[str]:
        """Get the MIME types supported by this loader.

        Returns:
            List of supported MIME types
        """

    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Get the file extensions supported by this loader.

        Returns:
            List of supported file extensions
        """
