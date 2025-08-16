"""
Markdown-aware HTML cleaner that preserves legitimate HTML while removing dangerous content.
"""

import logging
import re

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class MarkdownHtmlCleaner:
    """
    HTML cleaner specifically designed for markdown content.

    This cleaner:
    1. Preserves legitimate HTML elements commonly used in markdown
    2. Removes dangerous HTML (scripts, iframes, etc.)
    3. Cleans up messy HTML while preserving content
    4. Maintains markdown formatting
    """

    # HTML tags that are safe and commonly used in markdown
    SAFE_HTML_TAGS = {
        # Text formatting
        "b",
        "strong",
        "i",
        "em",
        "u",
        "mark",
        "small",
        "del",
        "ins",
        "sub",
        "sup",
        # Structure
        "p",
        "br",
        "hr",
        "div",
        "span",
        "blockquote",
        # Lists
        "ul",
        "ol",
        "li",
        "dl",
        "dt",
        "dd",
        # Tables
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "th",
        "td",
        "caption",
        "colgroup",
        "col",
        # Headers (though markdown has these)
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        # Links and images
        "a",
        "img",
        # Code
        "code",
        "pre",
        "kbd",
        "samp",
        "var",
        # Other semantic elements
        "abbr",
        "cite",
        "dfn",
        "time",
        "details",
        "summary",
    }

    # HTML tags that should be completely removed (dangerous or unwanted)
    DANGEROUS_HTML_TAGS = {
        "script",
        "style",
        "iframe",
        "frame",
        "frameset",
        "noframes",
        "object",
        "embed",
        "applet",
        "form",
        "input",
        "button",
        "select",
        "textarea",
        "fieldset",
        "legend",
        "label",
    }

    # Attributes that should be preserved
    SAFE_ATTRIBUTES = {
        "href",
        "src",
        "alt",
        "title",
        "class",
        "id",
        "colspan",
        "rowspan",
        "width",
        "height",
        "align",
        "datetime",
    }

    def __init__(self):
        """Initialize the markdown HTML cleaner."""
        self.logger = logging.getLogger(__name__)

    def clean_html_in_markdown(self, text: str) -> str:
        """
        Clean HTML content in markdown while preserving legitimate elements.

        Args:
            text: The markdown text that may contain HTML

        Returns:
            Cleaned text with safe HTML preserved and dangerous HTML removed
        """
        if not text or "<" not in text:
            # No HTML to process
            return text

        # Check if this appears to be real HTML content vs markdown with < characters
        # Look for actual HTML tag patterns vs false positives like email addresses
        actual_html_tags = re.findall(r"<([a-zA-Z][a-zA-Z0-9]*)[^>]*>", text)

        if not actual_html_tags:
            # No actual HTML tags found, likely just markdown with < characters
            # Return the original text unchanged
            self.logger.debug("No actual HTML tags detected, preserving content as-is")
            return text

        self.logger.debug(f"Found actual HTML tags: {set(actual_html_tags)}")

        # Remove dangerous tags completely (including content)
        for tag in self.DANGEROUS_HTML_TAGS:
            pattern = rf"<{tag}[^>]*>.*?</{tag}>"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
            # Also remove self-closing versions
            pattern = rf"<{tag}[^>]*/?>"
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Clean attributes from safe tags (remove potentially dangerous attributes)
        def clean_attributes(match):
            tag_content = match.group(1)
            tag_name = tag_content.split()[0].lower()

            if tag_name in self.SAFE_HTML_TAGS:
                # Keep the tag but clean attributes
                return f"<{self._clean_tag_attributes(tag_content)}>"
            else:
                # Remove unknown tags but keep content
                return ""

        # Only process actual HTML tags (not markdown link formats or other < content)
        def is_html_tag(match):
            tag_content = match.group(1)
            # Check if this looks like an actual HTML tag
            return re.match(r"^[a-zA-Z][a-zA-Z0-9]*(\s|$|/)", tag_content)

        # Process only actual HTML tags, preserve everything else
        def process_tag(match):
            if is_html_tag(match):
                return clean_attributes(match)
            else:
                # Not a real HTML tag, preserve it
                return match.group(0)

        text = re.sub(r"<([^>]+)>", process_tag, text)

        # Only clean up malformed HTML if we actually processed HTML
        if actual_html_tags:
            text = self._clean_malformed_html(text)

        # Normalize whitespace
        text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)  # Remove excessive newlines
        text = re.sub(r"[ \t]+", " ", text)  # Normalize spaces within lines

        return text.strip()

    def _clean_tag_attributes(self, tag_content: str) -> str:
        """Clean attributes from an HTML tag, keeping only safe ones."""
        parts = tag_content.split()
        if not parts:
            return tag_content

        tag_name = parts[0]
        cleaned_parts = [tag_name]

        # Process attributes
        attr_text = " ".join(parts[1:])
        safe_attrs = []

        # Simple attribute parsing (not perfect but good enough for cleaning)
        for attr_match in re.finditer(
            r'(\w+)(?:=("[^"]*"|\'[^\']*\'|[^\s>]+))?', attr_text
        ):
            attr_name = attr_match.group(1).lower()
            attr_value = attr_match.group(2) if attr_match.group(2) else ""

            if attr_name in self.SAFE_ATTRIBUTES:
                if attr_value:
                    safe_attrs.append(f"{attr_name}={attr_value}")
                else:
                    safe_attrs.append(attr_name)

        if safe_attrs:
            cleaned_parts.extend(safe_attrs)

        return " ".join(cleaned_parts)

    def _clean_malformed_html(self, text: str) -> str:
        """Clean up malformed HTML that might remain."""
        # Only remove orphaned closing tags that look like actual HTML tags
        text = re.sub(r"</([a-zA-Z][a-zA-Z0-9]*)[^>]*>", "", text)

        # Remove broken or incomplete HTML tags at boundaries
        text = re.sub(
            r"<([a-zA-Z][a-zA-Z0-9]*)[^>]*$", "", text
        )  # Incomplete HTML tag at end
        text = re.sub(r"^[^<]*>(?=[a-zA-Z])", "", text)  # Incomplete HTML tag at start

        return text

    async def transform_documents(self, documents: list[Document]) -> list[Document]:
        """
        Transform documents by cleaning HTML in markdown content.

        Args:
            documents: List of documents to clean

        Returns:
            List of documents with cleaned HTML
        """
        if not documents:
            return documents

        cleaned_documents = []

        for doc in documents:
            # Determine if this might be markdown content
            source = str(doc.metadata.get("source", ""))
            is_likely_markdown = source.endswith((".md", ".markdown")) or any(
                marker in doc.page_content for marker in ["#", "**", "__", "```", "---"]
            )

            if is_likely_markdown and "<" in doc.page_content:
                # Clean HTML in markdown context
                cleaned_content = self.clean_html_in_markdown(doc.page_content)
                cleaned_doc = Document(
                    page_content=cleaned_content,
                    metadata={
                        **doc.metadata,
                        "html_cleaned_markdown": True,
                        "original_length": len(doc.page_content),
                        "cleaned_length": len(cleaned_content),
                    },
                )
                cleaned_documents.append(cleaned_doc)

                self.logger.info(
                    f"Cleaned HTML in markdown document: "
                    f"{len(doc.page_content)} -> {len(cleaned_content)} chars"
                )
            else:
                # Not markdown or no HTML, keep as-is
                cleaned_documents.append(doc)

        return cleaned_documents


def create_markdown_html_cleaner() -> MarkdownHtmlCleaner:
    """Factory function to create a markdown HTML cleaner."""
    return MarkdownHtmlCleaner()
