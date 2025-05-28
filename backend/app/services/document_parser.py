"""Document parsing service for extracting text from various file formats."""

import io
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pypdf
from docx import Document as DocxDocument
from fastapi import HTTPException, UploadFile

from app.utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class DocumentMetadata:
    """Container for document metadata."""

    def __init__(
        self,
        filename: str,
        file_type: str,
        total_pages: Optional[int] = None,
        total_chars: int = 0,
        total_tokens: int = 0,
        sections: Optional[List[str]] = None,
    ):
        self.filename = filename
        self.file_type = file_type
        self.total_pages = total_pages
        self.total_chars = total_chars
        self.total_tokens = total_tokens
        self.sections = sections or []


class ParsedContent:
    """Container for parsed document content with metadata."""

    def __init__(
        self,
        text: str,
        metadata: DocumentMetadata,
        page_texts: Optional[List[str]] = None,
        structured_content: Optional[List[Dict]] = None,
    ):
        self.text = text
        self.metadata = metadata
        self.page_texts = page_texts or []
        self.structured_content = structured_content or []


class DocumentParser:
    """Service for parsing documents and extracting text content."""

    SUPPORTED_FORMATS = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "application/msword": "doc",
        "text/plain": "txt",
    }

    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.logger = logging.getLogger(__name__)
        self.token_counter = TokenCounter(encoding_name)

    async def parse_document(self, file: UploadFile) -> ParsedContent:
        """
        Parse a document and extract text content with metadata.

        Args:
            file: The uploaded file to parse

        Returns:
            ParsedContent: Parsed content with metadata

        Raises:
            HTTPException: If file format is unsupported or parsing fails
        """
        # Validate file
        self._validate_file(file)

        # Read file content
        content = await file.read()
        await file.seek(0)  # Reset file pointer for potential reuse

        # Determine file type and parse accordingly
        file_type = self._get_file_type(file.content_type, file.filename or "")

        try:
            if file_type == "pdf":
                return await self._parse_pdf(content, file.filename or "")
            elif file_type == "docx":
                return await self._parse_docx(content, file.filename or "")
            elif file_type == "txt":
                return await self._parse_text(content, file.filename or "")
            else:
                raise HTTPException(
                    status_code=400, detail=f"Unsupported file format: {file.content_type}"
                )
        except Exception as e:
            self.logger.error(f"Error parsing document {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to parse document: {str(e)}")

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

    async def _parse_pdf(self, content: bytes, filename: str) -> ParsedContent:
        """Parse PDF document."""
        try:
            pdf_file = io.BytesIO(content)
            reader = pypdf.PdfReader(pdf_file)

            if reader.is_encrypted:
                raise HTTPException(status_code=400, detail="Encrypted PDFs are not supported")

            page_texts = []
            full_text = ""

            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        page_texts.append(page_text)
                        full_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    self.logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                    continue

            if not full_text.strip():
                raise HTTPException(status_code=400, detail="No text content found in PDF")

            # Create metadata
            metadata = self._create_metadata_with_tokens(
                filename=filename,
                file_type="pdf",
                full_text=full_text,
                total_pages=len(reader.pages),
            )

            # Create structured content for each page
            structured_content = [
                {
                    "type": "page",
                    "page_number": i + 1,
                    "text": text,
                    "char_count": len(text),
                }
                for i, text in enumerate(page_texts)
            ]

            return ParsedContent(
                text=full_text.strip(),
                metadata=metadata,
                page_texts=page_texts,
                structured_content=structured_content,
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")

    async def _parse_docx(self, content: bytes, filename: str) -> ParsedContent:
        """Parse DOCX document."""
        try:
            docx_file = io.BytesIO(content)
            doc = DocxDocument(docx_file)

            paragraphs = []
            full_text = ""
            sections = []

            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    paragraphs.append(text)
                    full_text += text + "\n\n"

                    # Detect potential section headers (simple heuristic)
                    if len(text) < 100 and (
                        text.isupper()
                        or any(
                            text.startswith(prefix) for prefix in ["Chapter", "Section", "Article"]
                        )
                    ):
                        sections.append(text)

            if not full_text.strip():
                raise HTTPException(status_code=400, detail="No text content found in document")

            # Create metadata
            metadata = self._create_metadata_with_tokens(
                filename=filename,
                file_type="docx",
                full_text=full_text,
                sections=sections,
            )

            # Create structured content for each paragraph
            structured_content = [
                {
                    "type": "paragraph",
                    "index": i,
                    "text": text,
                    "char_count": len(text),
                    "is_potential_header": len(text) < 100
                    and (
                        text.isupper()
                        or any(
                            text.startswith(prefix) for prefix in ["Chapter", "Section", "Article"]
                        )
                    ),
                }
                for i, text in enumerate(paragraphs)
            ]

            return ParsedContent(
                text=full_text.strip(),
                metadata=metadata,
                structured_content=structured_content,
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse DOCX: {str(e)}")

    async def _parse_text(self, content: bytes, filename: str) -> ParsedContent:
        """Parse plain text document."""
        try:
            # Try different encodings
            encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]
            text = None

            for encoding in encodings:
                try:
                    text = content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue

            if text is None:
                raise HTTPException(
                    status_code=400, detail="Unable to decode text file with supported encodings"
                )

            if not text.strip():
                raise HTTPException(status_code=400, detail="No text content found in file")

            # Split into paragraphs
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            # Create metadata
            metadata = self._create_metadata_with_tokens(
                filename=filename,
                file_type="txt",
                full_text=text,
            )

            # Create structured content
            structured_content = [
                {
                    "type": "paragraph",
                    "index": i,
                    "text": para,
                    "char_count": len(para),
                }
                for i, para in enumerate(paragraphs)
            ]

            return ParsedContent(
                text=text.strip(),
                metadata=metadata,
                structured_content=structured_content,
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse text file: {str(e)}")

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
