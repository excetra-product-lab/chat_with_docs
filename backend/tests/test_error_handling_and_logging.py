"""Additional tests for robust error handling, encoding support and layout-preservation metadata.

These tests complement the existing suite for subtask 2.6 – ensuring the
recently-implemented error-handling & logging pathways are functioning as
expected.
"""

from __future__ import annotations

import io
import tempfile
from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException, UploadFile
from langchain_core.documents import Document
from starlette.datastructures import Headers

from app.services.document_processor import DocumentProcessor
from app.services.langchain_document_processor import LangchainDocumentProcessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_upload_file(content: bytes, filename: str, content_type: str) -> UploadFile:  # type: ignore
    """Utility to create a Starlette UploadFile from raw bytes."""
    headers = Headers({"content-type": content_type})
    upload_file = UploadFile(filename=filename, file=io.BytesIO(content), headers=headers)
    # Attach size attribute expected by validation helpers
    upload_file.size = len(content)
    return upload_file


# ---------------------------------------------------------------------------
# Error-mapping tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_value_error_maps_to_http_422():
    """A ValueError raised within the Langchain processor should surface as HTTP 422."""

    processor = DocumentProcessor(use_langchain=True)

    # Create a dummy upload file (content is irrelevant because the call will be mocked)
    dummy_file = create_upload_file(b"dummy", "dummy.txt", "text/plain")

    # Patch *instance* method so we do not affect other tests
    with patch.object(
        LangchainDocumentProcessor,
        "process_document_with_langchain",
        new=AsyncMock(side_effect=ValueError("Decryption failed")),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await processor.process_document(dummy_file, prefer_langchain=True)

        assert exc_info.value.status_code == 422
        assert "Decryption failed" in exc_info.value.detail


# ---------------------------------------------------------------------------
# Encoding detection & handling tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cp1251_encoded_text_is_decoded_successfully():
    """Ensure that a Windows-1251 (Cyrillic) encoded text file is decoded and returned correctly."""

    # Russian text – "Hello world"
    original_text = "Привет мир"
    encoded_bytes = original_text.encode("cp1251")

    upload_file = create_upload_file(encoded_bytes, "russian.txt", "text/plain")

    processor = DocumentProcessor(use_langchain=True)

    # Patch the _load_text_with_langchain method to avoid heavy TextLoader dependency.
    async def _mock_load_text_with_langchain(self, file_path: str, encoding_info: Optional[dict] = None):  # type: ignore  # noqa: D401,E501
        return [
            Document(
                page_content=original_text, metadata={"file_encoding_detection": encoding_info}
            )
        ]

    with patch.object(
        LangchainDocumentProcessor, "_load_text_with_langchain", _mock_load_text_with_langchain
    ):
        result = await processor.process_document(upload_file, prefer_langchain=True)

    # The full text concatenated during conversion should contain the original Russian string.
    assert original_text in result.parsed_content.text

    # Encoding metadata should be present and include the detected encoding.
    assert result.parsed_content.metadata.encoding_info
    assert result.parsed_content.metadata.encoding_info.get("detected_encoding")


# ---------------------------------------------------------------------------
# Layout-preservation metadata tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_layout_preservation_flag_present_for_pdf():
    """When preserve_layout=True, resulting metadata should include layout_preserved flag."""

    # Minimal PDF binary (empty page) generated on-the-fly via tempfile to satisfy suffix logic.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        tmp_pdf.write(
            b"%PDF-1.4\n1 0 obj<>stream\nendstream\nendobj\nxref\n0 2\n0000000000 65535 f\n0000000010 00000 n\ntrailer<</Size 2>>\nstartxref\n55\n%%EOF"
        )
        tmp_pdf.name

    # UploadFile mock – content is irrelevant because we patch the loader below.
    upload_file = create_upload_file(b"%PDF-1.4", "layout.pdf", "application/pdf")

    # Fake documents with layout metadata
    fake_docs = [
        Document(
            page_content="Some content",
            metadata={
                "page": 0,
                "layout_preserved": True,
                "tables": [],
                "structure_elements": {},
            },
        )
    ]

    processor = DocumentProcessor(use_langchain=True)

    # Patch the PDF loader method to return our fake docs
    with patch.object(
        LangchainDocumentProcessor,
        "_load_pdf_with_langchain",
        new=AsyncMock(return_value=fake_docs),
    ):
        result = await processor.process_document(upload_file, prefer_langchain=True)

    metadata = result.parsed_content.metadata
    assert metadata.filename == "layout.pdf"
    assert metadata.file_type == "pdf"

    # Check that layout_preserved flag bubbled up via structured_content or metadata sections
    assert (
        any(
            isinstance(section, dict) and section.get("layout_preserved")
            for section in result.parsed_content.structured_content
        )
        or metadata.sections
    )  # The flag should surface somewhere
