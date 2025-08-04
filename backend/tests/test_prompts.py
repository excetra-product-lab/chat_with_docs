import pytest
from app.core.prompts import SystemPrompts


class TestSystemPrompts:
    """Tests for Task 2: System Prompt Template Implementation"""
    
    def test_get_rag_system_prompt_structure(self):
        """Test 2.1: Citation format instructions in prompt"""
        prompt = SystemPrompts.get_rag_system_prompt()
        
        # Verify basic structure
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        
        # Test citation format instructions (Task 2.1)
        assert "[filename p. X]" in prompt
        assert "square brackets" in prompt
        assert "Cite your sources" in prompt
        
        # Test grounding enforcement (Task 2.2)
        assert "ONLY the provided document excerpts" in prompt
        assert "using only information from the provided context" in prompt
        assert "Do not make assumptions or add information not present in the context" in prompt
        
        # Test insufficient context handling (Task 2.3)
        assert "cannot be found in the provided context" in prompt
        assert "I cannot find this information in the provided documents" in prompt
    
    def test_format_rag_prompt_with_context(self):
        """Test prompt formatting with context"""
        test_context = "This is test context from documents."
        formatted_prompt = SystemPrompts.format_rag_prompt(test_context)
        
        assert isinstance(formatted_prompt, str)
        assert test_context in formatted_prompt
        assert "Context:" in formatted_prompt
        assert len(formatted_prompt) > len(SystemPrompts.get_rag_system_prompt())
    
    def test_citation_validation_prompt_structure(self):
        """Test citation validation prompt template"""
        prompt = SystemPrompts.get_citation_validation_prompt()
        
        assert isinstance(prompt, str)
        assert "citations are properly formatted" in prompt
        assert "square brackets: [filename p. X]" in prompt
        assert "All claims must be supported by citations" in prompt
        assert "Remove any unsupported claims" in prompt
    
    def test_format_citation_validation_prompt(self):
        """Test citation validation prompt formatting"""
        test_answer = "This is a test answer with citations [doc.pdf p. 1]."
        test_sources = "doc.pdf: Test document content"
        
        formatted_prompt = SystemPrompts.format_citation_validation_prompt(
            test_answer, test_sources
        )
        
        assert test_answer in formatted_prompt
        assert test_sources in formatted_prompt
        assert "Answer to review:" in formatted_prompt
        assert "Available sources:" in formatted_prompt
    
    def test_insufficient_context_prompt_structure(self):
        """Test insufficient context response template"""
        prompt = SystemPrompts.get_context_insufficient_prompt()
        
        assert isinstance(prompt, str)
        assert "cannot find sufficient information" in prompt
        assert "available context covers" in prompt
        assert "Provide additional relevant documents" in prompt
        assert "Rephrase your question" in prompt
    
    def test_format_insufficient_context_response(self):
        """Test insufficient context response formatting"""
        test_topics = "document management, file processing"
        
        formatted_response = SystemPrompts.format_insufficient_context_response(
            test_topics
        )
        
        assert test_topics in formatted_response
        assert "available context covers:" in formatted_response
        assert "Provide additional relevant documents" in formatted_response
    
    def test_prompt_consistency_across_methods(self):
        """Test that all prompt methods return consistent, non-empty strings"""
        prompts = [
            SystemPrompts.get_rag_system_prompt(),
            SystemPrompts.get_citation_validation_prompt(),
            SystemPrompts.get_context_insufficient_prompt()
        ]
        
        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt.strip()) > 0
            assert prompt == prompt.strip()  # No leading/trailing whitespace
    
    def test_citation_format_enforcement(self):
        """Test that citation format is consistently enforced across prompts"""
        rag_prompt = SystemPrompts.get_rag_system_prompt()
        validation_prompt = SystemPrompts.get_citation_validation_prompt()
        
        # Both should reference the same citation format
        citation_formats = ["[filename p. X]", "[filename]"]
        
        for fmt in citation_formats:
            assert fmt in rag_prompt or fmt.replace("filename", "filename") in rag_prompt
            assert fmt in validation_prompt or fmt.replace("filename", "filename") in validation_prompt
    
    def test_grounding_enforcement_strength(self):
        """Test that grounding enforcement is strong enough"""
        prompt = SystemPrompts.get_rag_system_prompt()
        
        # Should have multiple layers of grounding enforcement
        grounding_keywords = [
            "ONLY",
            "only information from the provided context",
            "Do not make assumptions",
            "not present in the context"
        ]
        
        for keyword in grounding_keywords:
            assert keyword in prompt, f"Missing grounding keyword: {keyword}"
    
    def test_prompt_templates_handle_edge_cases(self):
        """Test prompt templates with edge case inputs"""
        # Empty context
        empty_context_prompt = SystemPrompts.format_rag_prompt("")
        assert isinstance(empty_context_prompt, str)
        assert "Context:" in empty_context_prompt
        
        # Very long context
        long_context = "Test content. " * 1000
        long_context_prompt = SystemPrompts.format_rag_prompt(long_context)
        assert isinstance(long_context_prompt, str)
        assert long_context in long_context_prompt
        
        # Special characters in context
        special_context = "Context with special chars: @#$%^&*()[]{}|\\:;\"'<>,.?/~`"
        special_prompt = SystemPrompts.format_rag_prompt(special_context)
        assert isinstance(special_prompt, str)
        assert special_context in special_prompt