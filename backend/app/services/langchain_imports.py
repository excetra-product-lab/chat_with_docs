from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)

# BeautifulSoupTransformer requires the optional BeautifulSoup4 dependency. Import
# it lazily and degrade gracefully when bs4 is not installed so that importing
# this helper module never crashes the application or the test-suite.

try:
    from langchain_community.document_transformers import BeautifulSoupTransformer  # type: ignore
except ModuleNotFoundError:
    # Fallback dummy stub so that downstream `is not None` checks can be used
    # to determine availability.
    BeautifulSoupTransformer = None  # type: ignore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
)

__all__ = [
    "DocumentCompressorPipeline",
    "EmbeddingsFilter",
    "BeautifulSoupTransformer",
    "OpenAIEmbeddings",
    "CharacterTextSplitter",
]
