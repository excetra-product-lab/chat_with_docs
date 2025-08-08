"""Document structure detection package.

This package provides tools for detecting and parsing hierarchical structure
in legal documents, including sections, headings, numbering systems, and
other structural elements.
"""

from .data_models import (
    DocumentElement,
    DocumentStructure,
    ElementType,
    Heading,
    NumberingSystem,
    NumberingType,
    Section,
)
from .heading_detector import HeadingDetector
from .numbering_systems import NumberingSystemHandler
from .pattern_handlers import PatternHandler
from .structure_detector import StructureDetector

__all__ = [
    "ElementType",
    "NumberingType",
    "NumberingSystem",
    "DocumentElement",
    "Section",
    "Heading",
    "DocumentStructure",
    "NumberingSystemHandler",
    "PatternHandler",
    "HeadingDetector",
    "StructureDetector",
]
