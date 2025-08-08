#!/usr/bin/env python3
"""
Example script demonstrating Langchain document processing integration.

This script shows how to use the new LangchainDocumentProcessor
alongside the existing DocumentProcessor to process documents with enhanced capabilities.
"""

import asyncio
import logging
import tempfile
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock UploadFile for demonstration


class MockUploadFile:
    """Mock UploadFile for demonstration purposes."""

    def __init__(self, file_path: str | Path, content_type: str):
        self.file_path = Path(file_path)
        self.filename = self.file_path.name
        self.content_type = content_type
        self.size = self.file_path.stat().st_size if self.file_path.exists() else 0

    async def read(self) -> bytes:
        """Read file content."""
        with open(self.file_path, "rb") as f:
            return f.read()

    async def seek(self, position: int = 0) -> None:
        """Seek to position (no-op for this mock)."""


async def demonstrate_langchain_processing():
    """Demonstrate Langchain document processing capabilities."""

    # Import the processors (would normally be available in the app)
    try:
        from app.services.document_processor import DocumentProcessor
    except ImportError:
        print(
            "Error: Could not import processors. Make sure you're running from the backend directory."
        )
        return

    # Create processors
    enhanced_processor = DocumentProcessor(use_langchain=True)
    standard_processor = DocumentProcessor(use_langchain=False)

    # Create sample documents for testing
    sample_documents = {
        "sample.txt": "This is a sample text document.\n\nIt contains multiple paragraphs to demonstrate text processing.\n\nEach paragraph has different content to show how Langchain handles text extraction and structuring.",
        "sample.pdf": "PDF content would go here",  # Note: This is just text for demo
        "sample.docx": "DOCX content would go here",  # Note: This is just text for demo
    }

    print("=== Langchain Document Processing Demo ===\n")

    # Create temporary files for demonstration
    temp_files = {}
    try:
        for filename, content in sample_documents.items():
            if filename.endswith(".txt"):
                # Create actual text file for demonstration
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, encoding="utf-8"
                ) as f:
                    f.write(content)
                    temp_files[filename] = f.name

        # Process the text file with different processors
        for filename, temp_path in temp_files.items():
            print(f"\n--- Processing {filename} ---")

            # Create mock upload file
            mock_file = MockUploadFile(temp_path, "text/plain")

            # 1. Process with enhanced processor (with Langchain)
            print("\n1. Enhanced Processor (with Langchain):")
            try:
                result_enhanced = await enhanced_processor.process_document(mock_file)
                print("   ✓ Processed with enhanced processor")
                print(
                    f"   - Text length: {len(result_enhanced.parsed_content.text)} characters"
                )
                print(f"   - Chunks created: {len(result_enhanced.chunks)}")
                print(
                    f"   - Langchain used: {result_enhanced.processing_stats['processing']['langchain_used']}"
                )
                print(
                    f"   - Processing method: {'Langchain' if result_enhanced.processing_stats['processing']['langchain_used'] else 'Standard'}"
                )
            except Exception as e:
                print(f"   ✗ Enhanced processing failed: {e}")

            # 2. Process with standard processor (no Langchain)
            await mock_file.seek(0)  # Reset file pointer
            print("\n2. Standard Processor (no Langchain):")
            try:
                result_standard = await standard_processor.process_document(mock_file)
                print("   ✓ Processed with standard processor")
                print(
                    f"   - Text length: {len(result_standard.parsed_content.text)} characters"
                )
                print(f"   - Chunks created: {len(result_standard.chunks)}")
                print(
                    f"   - Langchain used: {result_standard.processing_stats['processing']['langchain_used']}"
                )
                print(
                    f"   - Processing method: {'Langchain' if result_standard.processing_stats['processing']['langchain_used'] else 'Standard'}"
                )
            except Exception as e:
                print(f"   ✗ Standard processing failed: {e}")

    finally:
        # Clean up temporary files
        for temp_path in temp_files.values():
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception as e:
                print(f"Warning: Could not delete temp file {temp_path}: {e}")

    # Demonstrate Langchain text splitting capabilities
    print("\n\n=== Langchain Text Splitting Demo ===")
    await demonstrate_text_splitting()

    # Show configuration comparison
    print("\n\n=== Configuration Comparison ===")
    demonstrate_configuration_comparison()


async def demonstrate_text_splitting():
    """Demonstrate Langchain text splitting capabilities."""

    try:
        from langchain_core.documents import Document

        from app.services.document_processor import DocumentProcessor
    except ImportError:
        print("Could not import required modules for text splitting demo")
        return

    processor = DocumentProcessor(use_langchain=True)

    # Create sample documents
    long_text = """This is a very long document that needs to be split into smaller chunks for processing.
    It contains multiple sentences and paragraphs that will be divided based on the chunk size and overlap settings.

    The purpose of this demonstration is to show how Langchain's text splitters work with different configurations.
    We can use either RecursiveCharacterTextSplitter or CharacterTextSplitter depending on our needs.

    RecursiveCharacterTextSplitter is generally better as it tries to keep related text together by using
    hierarchical separators like double newlines, single newlines, spaces, and finally individual characters.

    This allows for more semantic chunking compared to simple character-based splitting."""

    documents = [
        Document(page_content=long_text, metadata={"source": "demo.txt"}),
        Document(page_content="Short document.", metadata={"source": "short.txt"}),
    ]

    print("Original documents:")
    for i, doc in enumerate(documents):
        print(f"  Document {i + 1}: {len(doc.page_content)} characters")

    # Test different splitting configurations
    configs = [
        {"chunk_size": 200, "chunk_overlap": 50, "use_recursive": True},
        {"chunk_size": 200, "chunk_overlap": 50, "use_recursive": False},
        {"chunk_size": 100, "chunk_overlap": 20, "use_recursive": True},
    ]

    for config in configs:
        print(f"\nSplitting with config: {config}")
        # Note: This is a simplified demo - the current DocumentProcessor
        # handles chunking internally during document processing
        print("  Text splitting with current processor (simplified demo)")
        print(f"  Would create chunks based on config: {config}")
        print("    Configuration applied to internal chunking pipeline")


def demonstrate_configuration_comparison():
    """Show configuration differences between processors."""

    try:
        from app.services.document_processor import DocumentProcessor
    except ImportError:
        print("Could not import processors for configuration demo")
        return

    # Create processors
    enhanced_processor = DocumentProcessor(use_langchain=True)
    standard_processor = DocumentProcessor(use_langchain=False)

    processors = {
        "Enhanced Processor (with Langchain)": enhanced_processor,
        "Standard Processor": standard_processor,
    }

    print("Processor configurations:")
    for name, processor in processors.items():
        print(f"\n{name}:")
        config = processor.get_processing_config()
        for key, value in config.items():
            if key != "langchain_config":  # Skip nested config for brevity
                print(f"  {key}: {value}")
            elif key == "langchain_config":
                print(f"  {key}: Available")


if __name__ == "__main__":
    print("Starting Langchain Document Processing Demo...")
    print(
        "Note: This demo requires the app to be properly set up with all dependencies."
    )
    print("Make sure you're running this from the backend directory.\n")

    asyncio.run(demonstrate_langchain_processing())
