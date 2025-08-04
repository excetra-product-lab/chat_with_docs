"""
System prompt templates for RAG-based question answering.
"""

from typing import Dict, Any


class SystemPrompts:
    """Collection of system prompt templates for different use cases."""
    
    @staticmethod
    def get_rag_system_prompt() -> str:
        """
        Get the main RAG system prompt template that enforces citation requirements
        and grounded responses.
        
        Returns:
            System prompt template string with context placeholder
        """
        return """You are a helpful assistant that answers questions using ONLY the provided document excerpts.

Instructions:
1. Answer the question using only information from the provided context
2. Cite your sources using square brackets with the format: [filename p. X] or [filename] if no page number
3. If the answer cannot be found in the provided context, clearly state "I cannot find this information in the provided documents"
4. Be concise and factual
5. Include relevant citations for each claim you make
6. Do not make assumptions or add information not present in the context
7. If multiple sources support the same information, cite all relevant sources

Context:
{context}"""

    @staticmethod
    def get_citation_validation_prompt() -> str:
        """
        Get prompt for validating and standardizing citations.
        
        Returns:
            Citation validation prompt template
        """
        return """Review the following answer and ensure all citations are properly formatted:

Requirements:
- Citations must use square brackets: [filename p. X] or [filename]
- All claims must be supported by citations
- Citations must reference actual content from the provided sources
- Remove any unsupported claims

Answer to review:
{answer}

Available sources:
{sources}"""

    @staticmethod
    def get_context_insufficient_prompt() -> str:
        """
        Get prompt for handling cases with insufficient context.
        
        Returns:
            Insufficient context response template
        """
        return """I cannot find sufficient information in the provided documents to answer your question comprehensively. 

The available context covers: {available_topics}

To get a complete answer, you may need to:
1. Provide additional relevant documents
2. Rephrase your question to focus on the available information
3. Ask a more specific question about the topics covered in the documents"""

    @staticmethod
    def format_rag_prompt(context: str) -> str:
        """
        Format the RAG system prompt with the provided context.
        
        Args:
            context: The document context to include in the prompt
            
        Returns:
            Formatted system prompt ready for use
        """
        return SystemPrompts.get_rag_system_prompt().format(context=context)

    @staticmethod
    def format_citation_validation_prompt(answer: str, sources: str) -> str:
        """
        Format the citation validation prompt with answer and sources.
        
        Args:
            answer: The answer text to validate
            sources: Available source information
            
        Returns:
            Formatted citation validation prompt
        """
        return SystemPrompts.get_citation_validation_prompt().format(
            answer=answer, 
            sources=sources
        )

    @staticmethod
    def format_insufficient_context_response(available_topics: str) -> str:
        """
        Format response for insufficient context scenarios.
        
        Args:
            available_topics: Summary of topics covered in available context
            
        Returns:
            Formatted insufficient context response
        """
        return SystemPrompts.get_context_insufficient_prompt().format(
            available_topics=available_topics
        )