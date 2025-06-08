"""
Langchain-based document processing service.

This module provides document processing capabilities using Langchain's document loaders
and text splitters, while maintaining compatibility with the existing document processing pipeline.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

from fastapi import HTTPException, UploadFile
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.core.langchain_config import langchain_config
from app.services.document_parser import DocumentMetadata, ParsedContent
from app.utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class LangchainDocumentProcessor:
    """Document processor using Langchain loaders and splitters."""

    SUPPORTED_FORMATS = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/msword": "doc",
        "text/plain": "txt",
    }

    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize the Langchain document processor.

        Args:
            encoding_name: The tokenizer encoding to use for token counting
        """
        self.logger = logging.getLogger(__name__)
        self.token_counter = TokenCounter(encoding_name)
        self.chunk_config = langchain_config.get_chunk_config()

    async def process_document_with_langchain(self, file: UploadFile) -> ParsedContent:
        """
        Process a document using Langchain loaders.

        Args:
            file: The uploaded file to process

        Returns:
            ParsedContent: Processed document content using Langchain loaders

        Raises:
            HTTPException: If file format is unsupported or processing fails
        """
        try:
            # Validate file
            self._validate_file(file)

            # Get file type
            file_type = self._get_file_type(file.content_type, file.filename or "")

            # Create a temporary file for Langchain loaders
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(file.filename or "").suffix
            ) as temp_file:
                # Write uploaded content to temp file
                content = await file.read()
                temp_file.write(content)
                temp_file.flush()

                # Process using appropriate Langchain loader
                if file_type == "pdf":
                    documents = await self._load_pdf_with_langchain(temp_file.name)
                elif file_type in ["docx", "doc"]:
                    documents = await self._load_word_with_langchain(temp_file.name)
                elif file_type == "txt":
                    documents = await self._load_text_with_langchain(temp_file.name)
                else:
                    raise HTTPException(
                        status_code=400, detail=f"Unsupported file format: {file.content_type}"
                    )

                # Clean up temp file
                Path(temp_file.name).unlink(missing_ok=True)

            # Convert Langchain documents to ParsedContent format
            return self._convert_to_parsed_content(documents, file.filename or "", file_type)

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error processing document {file.filename} with Langchain: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to process document with Langchain: {str(e)}"
            )

    async def _load_pdf_with_langchain(self, file_path: str) -> List[Document]:
        """Load PDF using Langchain PyPDFLoader."""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            self.logger.info(f"Loaded {len(documents)} pages from PDF using PyPDFLoader")
            return documents
        except Exception as e:
            self.logger.error(f"Error loading PDF with PyPDFLoader: {str(e)}")
            raise

    async def _load_word_with_langchain(self, file_path: str) -> List[Document]:
        """Load Word document using Langchain UnstructuredWordDocumentLoader."""
        try:
            loader = UnstructuredWordDocumentLoader(file_path)
            documents = loader.load()
            self.logger.info(
                f"Loaded {len(documents)} sections from Word document "
                "using UnstructuredWordDocumentLoader"
            )
            return documents
        except Exception as e:
            self.logger.error(
                "Error loading Word document with UnstructuredWordDocumentLoader: " f"{str(e)}"
            )
            raise

    async def _load_text_with_langchain(self, file_path: str) -> List[Document]:
        """Load text file using Langchain TextLoader."""
        try:
            # Try different encodings for text files
            encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]
            documents = None

            for encoding in encodings:
                try:
                    loader = TextLoader(file_path, encoding=encoding)
                    documents = loader.load()
                    self.logger.info(f"Loaded text file using TextLoader with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue

            if documents is None:
                raise ValueError("Could not decode text file with any supported encoding")

            return documents
        except Exception as e:
            self.logger.error(f"Error loading text file with TextLoader: {str(e)}")
            raise

    def _convert_to_parsed_content(
        self, documents: List[Document], filename: str, file_type: str
    ) -> ParsedContent:
        """
        Convert Langchain documents to ParsedContent format.

        Args:
            documents: List of Langchain Document objects
            filename: Original filename
            file_type: Type of the file

        Returns:
            ParsedContent: Converted content in existing format
        """
        # Combine all document content
        full_text = ""
        page_texts = []
        structured_content = []

        for i, doc in enumerate(documents):
            page_content = doc.page_content.strip()
            if page_content:
                page_texts.append(page_content)

                # Add page separator for combined text
                if file_type == "pdf":
                    full_text += f"\n--- Page {i + 1} ---\n{page_content}\n"
                else:
                    full_text += page_content + "\n\n"

                # Create structured content entry
                structured_content.append(
                    {
                        "type": "page" if file_type == "pdf" else "section",
                        "index": i,
                        "text": page_content,
                        "char_count": len(page_content),
                        "metadata": doc.metadata,
                        "langchain_source": True,  # Flag to indicate Langchain processing
                    }
                )

        # Extract sections from metadata if available
        sections = []
        for doc in documents:
            if "section" in doc.metadata:
                sections.append(doc.metadata["section"])

        # Create metadata with token counting
        metadata = self._create_metadata_with_tokens(
            filename=filename,
            file_type=file_type,
            full_text=full_text,
            total_pages=len(documents) if file_type == "pdf" else None,
            sections=sections,
        )

        return ParsedContent(
            text=full_text.strip(),
            metadata=metadata,
            page_texts=page_texts,
            structured_content=structured_content,
        )

    def create_langchain_text_splitter(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        use_recursive: bool = True,
    ) -> Union[RecursiveCharacterTextSplitter, CharacterTextSplitter]:
        """
        Create a Langchain text splitter with configuration.

        Args:
            chunk_size: Size of each chunk (defaults to config value)
            chunk_overlap: Overlap between chunks (defaults to config value)
            use_recursive: Whether to use RecursiveCharacterTextSplitter

        Returns:
            Configured Langchain text splitter
        """
        chunk_size = chunk_size or self.chunk_config["chunk_size"]
        chunk_overlap = chunk_overlap or self.chunk_config["chunk_overlap"]

        splitter: Union[RecursiveCharacterTextSplitter, CharacterTextSplitter]
        if use_recursive:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
            )
        else:
            splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n\n",
            )

        self.logger.info(
            f"Created {'Recursive' if use_recursive else 'Character'} text splitter "
            f"(chunk_size={chunk_size}, chunk_overlap={chunk_overlap})"
        )
        return splitter

    def split_documents_with_langchain(
        self,
        documents: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        use_recursive: bool = True,
    ) -> List[Document]:
        """
        Split Langchain documents using Langchain text splitters.

        Args:
            documents: List of Langchain documents to split
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            use_recursive: Whether to use RecursiveCharacterTextSplitter

        Returns:
            List of split document chunks
        """
        splitter = self.create_langchain_text_splitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, use_recursive=use_recursive
        )

        split_docs = splitter.split_documents(documents)
        self.logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")

        return split_docs

    def _validate_file(self, file: UploadFile) -> None:
        """Validate uploaded file."""
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        if file.size and file.size > self.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {self.MAX_FILE_SIZE / 1024 / 1024}MB",
            )

    def _get_file_type(self, content_type: Optional[str], filename: str) -> str:
        """Determine file type from content type or filename extension."""
        if content_type and content_type in self.SUPPORTED_FORMATS:
            return self.SUPPORTED_FORMATS[content_type]

        # Fallback to file extension
        extension = Path(filename).suffix.lower()
        extension_map = {
            ".pdf": "pdf",
            ".docx": "docx",
            ".doc": "doc",
            ".txt": "txt",
        }

        if extension in extension_map:
            return extension_map[extension]

        raise HTTPException(
            status_code=400, detail=f"Unsupported file format: {content_type or extension}"
        )

    def _create_metadata_with_tokens(
        self,
        filename: str,
        file_type: str,
        full_text: str,
        total_pages: Optional[int] = None,
        sections: Optional[List[str]] = None,
    ) -> DocumentMetadata:
        """Create metadata with token count included."""
        total_chars = len(full_text)
        total_tokens = self.token_counter.count_tokens(full_text)

        return DocumentMetadata(
            filename=filename,
            file_type=file_type,
            total_pages=total_pages,
            total_chars=total_chars,
            total_tokens=total_tokens,
            sections=sections or [],
        )

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return list(self.SUPPORTED_FORMATS.keys())

    def get_processing_config(self) -> Dict:
        """Get current processing configuration."""
        return {
            "chunk_size": self.chunk_config["chunk_size"],
            "chunk_overlap": self.chunk_config["chunk_overlap"],
            "max_tokens_per_chunk": self.chunk_config["max_tokens_per_chunk"],
            "max_file_size_mb": self.MAX_FILE_SIZE / 1024 / 1024,
            "supported_formats": self.get_supported_formats(),
            "langchain_enabled": True,
        }
