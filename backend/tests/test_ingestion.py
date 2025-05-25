import pytest
from app.core.ingestion import chunk_text

def test_chunk_text():
    text = "This is a test document with some content."
    chunks = chunk_text(text)
    assert len(chunks) > 0
    assert chunks[0] == text  # Simple test for now
