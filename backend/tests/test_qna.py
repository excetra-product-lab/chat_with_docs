import pytest
from app.core.qna import build_context

def test_build_context():
    chunks = [{"content": "Test chunk 1"}, {"content": "Test chunk 2"}]
    context = build_context(chunks)
    assert isinstance(context, str)
    assert len(context) > 0
