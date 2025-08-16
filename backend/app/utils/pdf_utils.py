from __future__ import annotations

from typing import Any

"""PDF-specific helper utilities extracted from the bulky
`LangchainDocumentProcessor` class so the parent module can stay focused on API
orchestration.

These helpers are **pure functions** – they do not depend on FastAPI, project
settings, or class instances – which makes them easier to unit-test in
isolation.
"""

__all__ = [
    "extract_pdf_formatting",
    "extract_pdf_structure_elements",
    "detect_table_header",
    "is_heading",
    "is_list_item",
    "determine_heading_level",
]


# ---------------------------------------------------------------------------
# Low-level heuristics
# ---------------------------------------------------------------------------


def extract_pdf_formatting(chars: list[dict[str, Any]]) -> dict[str, Any]:
    """Return high-level font / colour statistics for a PDF page."""

    # Collect mutable sets first – this avoids the *set vs list* type clash
    # that confused mypy when we mutably updated the dict in-place.
    fonts: set[str] = set()
    font_sizes: set[float] = set()
    colors: set[str] = set()

    text_styles: dict[str, int] = {
        "bold_chars": 0,
        "italic_chars": 0,
        "total_chars": len(chars),
    }

    for char in chars:
        if "fontname" in char:
            fonts.add(char["fontname"])
        if "size" in char:
            try:
                font_sizes.add(float(char["size"]))
            except (TypeError, ValueError):
                # Ignore non-numeric font sizes
                pass
        if "ncs" in char:  # Non-stroking colour spec
            colors.add(str(char["ncs"]))

        font_name = str(char.get("fontname", "")).lower()
        if "bold" in font_name:
            text_styles["bold_chars"] += 1
        if "italic" in font_name or "oblique" in font_name:
            text_styles["italic_chars"] += 1

    # Return JSON-serialisable payload
    return {
        "fonts": sorted(fonts),
        "font_sizes": sorted(font_sizes),
        "colors": sorted(colors),
        "text_styles": text_styles,
    }


# ---------------------------------------------------------------------------
# Structural element detection
# ---------------------------------------------------------------------------


def is_heading(
    line: str, avg_font_size: float, chars: list[dict], line_idx: int
) -> bool:  # noqa: D401
    """Heuristic to decide if *line* looks like a heading."""
    if len(line) > 100:
        return False
    if line.endswith(".") and len(line) > 50:
        return False
    if not line[0].isupper():
        return False

    common = ["chapter", "section", "introduction", "conclusion", "abstract"]
    if any(pat in line.lower() for pat in common):
        return True

    if line.isupper() and len(line) > 3:
        return True
    return False


def determine_heading_level(
    line: str, avg_font_size: float, chars: list[dict], line_idx: int
) -> int:
    """Crude mapping of line to heading level 1-3."""
    if line.isupper():
        return 1
    lower = line.lower()
    if any(w in lower for w in ["chapter", "part"]):
        return 1
    if any(w in lower for w in ["section", "introduction", "conclusion"]):
        return 2
    return 3


def is_list_item(line: str) -> bool:
    """Return True if line appears to be part of a bullet/numbered list."""
    line = line.strip()
    if not line:
        return False

    if line[0].isdigit() and ("." in line[:5] or ")" in line[:5]):
        return True

    bullet_chars = ["•", "◦", "▪", "▫", "‣", "⁃", "-", "*"]
    return line[0] in bullet_chars


def detect_table_header(table: list[list[str | None]]) -> bool:
    """Heuristic: first row is header if it contains fewer numbers than second row."""
    if not table or len(table) < 2:
        return False

    first_row, second_row = table[0], table[1]
    nums_first = sum(
        1 for cell in first_row if cell and any(c.isdigit() for c in str(cell))
    )
    nums_second = sum(
        1 for cell in second_row if cell and any(c.isdigit() for c in str(cell))
    )
    return nums_first < nums_second


# ---------------------------------------------------------------------------
# Higher-level aggregator
# ---------------------------------------------------------------------------


def extract_pdf_structure_elements(text: str, chars: list[dict]) -> dict[str, Any]:
    """Return detected headings, lists, paragraph counts and text blocks."""
    structure: dict[str, Any] = {
        "headings": [],
        "lists": [],
        "paragraphs": 0,
        "text_blocks": [],
    }

    lines = text.split("\n")
    font_sizes = [char.get("size", 12) for char in chars if "size" in char]
    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        if is_heading(stripped, avg_font_size, chars, idx):
            level = determine_heading_level(stripped, avg_font_size, chars, idx)
            structure["headings"].append(
                {"text": stripped, "level": level, "line_number": idx}
            )
        elif is_list_item(stripped):
            list_type = "numbered" if stripped[0].isdigit() else "bulleted"
            structure["lists"].append(
                {"text": stripped, "type": list_type, "line_number": idx}
            )
        elif len(stripped) > 20:
            structure["paragraphs"] += 1

    return structure
