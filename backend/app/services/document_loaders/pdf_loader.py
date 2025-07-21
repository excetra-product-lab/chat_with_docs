"""PDF document loader service."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from app.utils.pdf_utils import (
    extract_pdf_formatting,
    extract_pdf_structure_elements,
)

from .base_loader import BaseDocumentLoader

logger = logging.getLogger(__name__)


class PDFDocumentLoader(BaseDocumentLoader):
    """Document loader for PDF files."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialize the PDF document loader.

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
        """Load a PDF document.

        Args:
            file_path: Path to the PDF file
            password: Optional password for protected PDFs
            preserve_layout: Whether to preserve PDF layout information
            encoding_info: Not used for PDF files

        Returns:
            List of Document objects
        """
        self.logger.info(f"Loading PDF document: {file_path}")

        if preserve_layout:
            return await self._load_pdf_with_layout_preservation(file_path, password)
        else:
            return await self._load_pdf_standard(file_path, password)

    async def _load_pdf_with_layout_preservation(
        self, file_path: str, password: Optional[str] = None
    ) -> List[Document]:
        """Load PDF with layout preservation using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            self.logger.warning("PyMuPDF not available, falling back to standard PDF loading")
            return await self._load_pdf_standard(file_path, password)

        documents = []

        try:
            # Open PDF with PyMuPDF
            pdf_doc = fitz.open(file_path)

            # Handle password protection
            if pdf_doc.needs_pass:
                if not password:
                    raise ValueError("PDF is password protected but no password provided")
                if not pdf_doc.authenticate(password):
                    raise ValueError("Invalid password for PDF")

            for page_num in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_num)

                # Extract text with character-level information
                text_dict = page.get_text("dict")
                page_text = page.get_text()

                # Extract character information for formatting analysis
                chars = []
                for block in text_dict.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line.get("spans", []):
                                for char in span.get("chars", []):
                                    chars.append(
                                        {
                                            "char": char.get("c", ""),
                                            "fontname": span.get("font", ""),
                                            "size": span.get("size", 12),
                                            "flags": span.get("flags", 0),
                                            "color": span.get("color", 0),
                                        }
                                    )

                # Extract formatting and structure information
                formatting = extract_pdf_formatting(chars)
                structure = extract_pdf_structure_elements(page_text, chars)

                # Create document with rich metadata
                metadata = {
                    "source": file_path,
                    "page": page_num,
                    "total_pages": len(pdf_doc),
                    "layout_preserved": True,
                    "formatting": formatting,
                    "structure": structure,
                    "file_type": "pdf",
                }

                documents.append(Document(page_content=page_text, metadata=metadata))

            pdf_doc.close()
            self.logger.info(
                f"Successfully loaded PDF with layout preservation: {len(documents)} pages"
            )
            return documents

        except Exception as e:
            self.logger.error(f"Error loading PDF with layout preservation: {str(e)}")
            # Fallback to standard loading
            return await self._load_pdf_standard(file_path, password)

    async def _load_pdf_standard(
        self, file_path: str, password: Optional[str] = None
    ) -> List[Document]:
        """Load PDF using standard PyPDFLoader."""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Add basic metadata
            for i, doc in enumerate(documents):
                doc.metadata.update(
                    {
                        "source": file_path,
                        "page": i,
                        "total_pages": len(documents),
                        "layout_preserved": False,
                        "file_type": "pdf",
                    }
                )

            self.logger.info(
                f"Successfully loaded PDF with standard loader: {len(documents)} pages"
            )
            return documents

        except Exception as e:
            self.logger.error(f"Error loading PDF: {str(e)}")
            raise ValueError(f"Failed to load PDF document: {str(e)}")

    async def extract_pdf_with_multiple_passwords(
        self, file_path: str, password_candidates: List[str]
    ) -> List[Document]:
        """Try to extract PDF with multiple password candidates.

        Args:
            file_path: Path to the PDF file
            password_candidates: List of passwords to try

        Returns:
            List of Document objects

        Raises:
            ValueError: If none of the passwords work
        """
        for password in password_candidates:
            try:
                return await self.load_document(file_path, password=password)
            except ValueError as e:
                if "password" not in str(e).lower():
                    raise  # Re-raise non-password errors
                continue

        raise ValueError("None of the provided passwords worked for the PDF")

    async def extract_pdf_metadata_only(
        self, file_path: str, password: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract only metadata from PDF without loading full content.

        Args:
            file_path: Path to the PDF file
            password: Optional password for protected PDFs

        Returns:
            Dictionary containing PDF metadata
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            self.logger.warning("PyMuPDF not available for metadata extraction")
            return {"error": "PyMuPDF not available"}

        try:
            pdf_doc = fitz.open(file_path)

            if pdf_doc.needs_pass:
                if not password:
                    raise ValueError("PDF is password protected but no password provided")
                if not pdf_doc.authenticate(password):
                    raise ValueError("Invalid password for PDF")

            metadata = {
                "title": pdf_doc.metadata.get("title", ""),
                "author": pdf_doc.metadata.get("author", ""),
                "subject": pdf_doc.metadata.get("subject", ""),
                "creator": pdf_doc.metadata.get("creator", ""),
                "producer": pdf_doc.metadata.get("producer", ""),
                "creation_date": pdf_doc.metadata.get("creationDate", ""),
                "modification_date": pdf_doc.metadata.get("modDate", ""),
                "total_pages": len(pdf_doc),
                "is_encrypted": pdf_doc.needs_pass,
                "file_size": Path(file_path).stat().st_size,
            }

            pdf_doc.close()
            return metadata

        except Exception as e:
            self.logger.error(f"Error extracting PDF metadata: {str(e)}")
            return {"error": str(e)}

    def is_pdf_password_protected(self, file_path: str) -> bool:
        """Check if PDF is password protected.

        Args:
            file_path: Path to the PDF file

        Returns:
            True if password protected, False otherwise
        """
        try:
            import fitz  # PyMuPDF

            pdf_doc = fitz.open(file_path)
            is_protected = pdf_doc.needs_pass
            pdf_doc.close()
            return is_protected
        except Exception:
            return False

    def get_supported_mime_types(self) -> List[str]:
        """Get the MIME types supported by this loader.

        Returns:
            List of supported MIME types
        """
        return ["application/pdf"]

    def get_supported_extensions(self) -> List[str]:
        """Get the file extensions supported by this loader.

        Returns:
            List of supported file extensions
        """
        return [".pdf"]
