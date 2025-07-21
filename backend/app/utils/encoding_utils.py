"""Utility functions for robust character-encoding detection, validation and decoding.

These helpers are extracted from `LangchainDocumentProcessor` so that the
main service class can focus on orchestration while the heavy-weight
encoding logic lives in this dedicated module.
"""

from __future__ import annotations

import codecs
import logging
from typing import Any, Dict, List, Optional

import chardet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _detect_bom(file_content: bytes) -> Dict[str, Any]:
    """Detect a Byte Order Mark (BOM) in *file_content*.

    Returns a mapping with *bom_type*, *encoding* and *bom_bytes* keys.
    """
    bom_signatures = [
        (codecs.BOM_UTF8, "utf-8-sig", "utf-8"),
        (codecs.BOM_UTF16_BE, "utf-16-be", "utf-16-be"),
        (codecs.BOM_UTF16_LE, "utf-16-le", "utf-16-le"),
        (codecs.BOM_UTF32_BE, "utf-32-be", "utf-32-be"),
        (codecs.BOM_UTF32_LE, "utf-32-le", "utf-32-le"),
    ]

    for bom_bytes, bom_type, encoding in bom_signatures:
        if file_content.startswith(bom_bytes):
            return {
                "bom_type": bom_type,
                "encoding": encoding,
                "bom_bytes": bom_bytes,
            }

    return {"bom_type": "none", "encoding": None, "bom_bytes": b""}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def detect_file_encoding(
    file_content: bytes,
    confidence_threshold: float = 0.7,
) -> Dict[str, Any]:
    """Detect the character encoding of *file_content* using **chardet**.

    The return value mirrors the structure previously produced by the
    `LangchainDocumentProcessor.detect_file_encoding` method.
    """
    try:
        detection_result: Dict[str, Any] = chardet.detect(file_content)  # type: ignore[assignment]
        bom_info = _detect_bom(file_content)

        encoding_info = {
            "encoding": detection_result.get("encoding", "utf-8").lower(),
            "confidence": detection_result.get("confidence", 0.0),
            "detected_bom": bom_info["bom_type"],
            "bom_bytes": bom_info["bom_bytes"],
            "is_reliable": detection_result.get("confidence", 0.0) >= confidence_threshold,
            "original_detection": detection_result,
        }

        # Prefer BOM-indicated encoding when available.
        if bom_info["bom_type"] != "none":
            encoding_info["encoding"] = bom_info["encoding"]
            encoding_info["is_reliable"] = True

        logger.info(
            "Detected encoding %s (confidence %.2f, BOM: %s)",
            encoding_info["encoding"],
            encoding_info["confidence"],
            encoding_info["detected_bom"],
        )

        return encoding_info

    except Exception as exc:  # pragma: no cover – defensive
        logger.error("Error detecting file encoding: %s", exc)
        return {
            "encoding": "utf-8",
            "confidence": 0.0,
            "detected_bom": "none",
            "bom_bytes": b"",
            "is_reliable": False,
            "original_detection": None,
            "error": str(exc),
        }


async def validate_text_encoding(text: str, encoding: str) -> Dict[str, Any]:
    """Light-weight heuristics to validate that *text* matches *encoding*."""

    try:
        validation_info: Dict[str, Any] = {
            "is_valid": True,
            "encoding": encoding,
            "char_count": len(text),
            "issues": [],
            "confidence_score": 1.0,
        }

        issues: List[str] = []

        # Replacement characters (\uFFFD)
        repl_chars = text.count("\ufffd")
        if repl_chars:
            issues.append(f"Found {repl_chars} replacement characters")
            validation_info["confidence_score"] -= repl_chars / max(len(text), 1) * 0.5

        # Suspicious sequences
        suspicious = ["\x00", "\uffff", "\ufeff"]
        for seq in suspicious:
            count = text.count(seq)
            if count:
                issues.append(f"Found {count} instances of {repr(seq)}")
                validation_info["confidence_score"] -= count / max(len(text), 1) * 0.3

        # Round-trip encode check for UTF encodings.
        if encoding.lower().startswith("utf") and any(ord(c) > 127 for c in text[:1000]):
            try:
                text.encode(encoding)
            except UnicodeEncodeError as exc:
                issues.append(f"Unicode encode error: {exc}")
                validation_info["confidence_score"] -= 0.4

        validation_info["issues"] = issues
        validation_info["is_valid"] = validation_info["confidence_score"] > 0.6

        if issues:
            logger.warning("Text validation issues for %s: %s", encoding, issues)

        return validation_info

    except Exception as exc:
        logger.error("Error validating text encoding: %s", exc)
        return {
            "is_valid": False,
            "encoding": encoding,
            "char_count": len(text) if text else 0,
            "issues": [f"Validation error: {exc}"],
            "confidence_score": 0.0,
            "error": str(exc),
        }


def get_encoding_fallback_list(detected_encoding: Optional[str] = None) -> List[str]:
    """Return a prioritized list of encodings to try when decoding bytes."""

    base_encodings = [
        # UTF family
        "utf-8",
        "utf-8-sig",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "utf-32",
        "utf-32-le",
        "utf-32-be",
        # Western / Latin-1 variants
        "latin1",
        "iso-8859-1",
        "cp1252",
        "windows-1252",
        # Misc common single-byte encodings
        "cp1251",
        "windows-1251",
        "cp1256",
        "windows-1256",
        # CJK
        "cp936",
        "gb2312",
        "gbk",
        "big5",
        "shift_jis",
        "cp932",
        "euc-kr",
        "cp949",
        # Central/Eastern European
        "iso-8859-2",
        "cp1250",
        # Various script-specific encodings
        "iso-8859-5",
        "iso-8859-6",
        "iso-8859-7",
        "iso-8859-8",
        "iso-8859-9",
        "iso-8859-15",
        "ascii",
    ]

    priority: List[str] = []

    if detected_encoding:
        norm = detected_encoding.lower().replace("_", "-")
        priority.append(norm)

    common = ["utf-8", "utf-8-sig", "latin1", "cp1252", "utf-16"]
    priority.extend([enc for enc in common if enc not in priority])

    priority.extend([enc for enc in base_encodings if enc not in priority])
    return priority


async def try_decode_with_fallback(
    file_content: bytes,
    encoding_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Attempt to decode *file_content* using an ordered *encoding_list*.

    Falls back through the list until a valid decoding is found (validated by
    `validate_text_encoding`). Returns details of all attempts together with
    the chosen best attempt.
    """

    # Prepare list of encodings to try.
    if encoding_list is None:
        detection_info = await detect_file_encoding(file_content)
        encoding_list = get_encoding_fallback_list(detection_info["encoding"])

    attempts: List[Dict[str, Any]] = []

    for enc in encoding_list:
        try:
            content_to_decode = file_content
            bom_removed = False

            if enc in ("utf-8-sig", "utf-16", "utf-32"):
                # These encodings strip BOM automatically.
                pass
            elif enc.startswith("utf-8") and file_content.startswith(codecs.BOM_UTF8):
                content_to_decode = file_content[len(codecs.BOM_UTF8) :]
                bom_removed = True
            elif enc.startswith("utf-16") and (
                file_content.startswith(codecs.BOM_UTF16_LE)
                or file_content.startswith(codecs.BOM_UTF16_BE)
            ):
                bom_removed = True
                content_to_decode = file_content[len(codecs.BOM_UTF16_LE) :]

            decoded_text = content_to_decode.decode(enc, errors="strict")
            validation = await validate_text_encoding(decoded_text, enc)

            attempts.append(
                {
                    "encoding": enc,
                    "success": True,
                    "text": decoded_text,
                    "bom_removed": bom_removed,
                    "validation": validation,
                    "text_length": len(decoded_text),
                    "error": None,
                }
            )

            if validation["is_valid"]:
                logger.info("Decoded content using '%s'", enc)
                return {
                    "success": True,
                    "final_encoding": enc,
                    "text": decoded_text,
                    "bom_removed": bom_removed,
                    "validation": validation,
                    "attempts": attempts,
                }
        except (UnicodeDecodeError, UnicodeError, LookupError) as exc:
            attempts.append(
                {
                    "encoding": enc,
                    "success": False,
                    "text": None,
                    "bom_removed": False,
                    "validation": None,
                    "text_length": 0,
                    "error": str(exc),
                }
            )
            continue

    # No valid decoding. Pick best unsuccessful attempt if any.
    successful = [a for a in attempts if a["success"]]
    if successful:
        best = max(successful, key=lambda a: a["validation"]["confidence_score"])
        logger.warning(
            "Using best-effort decoding '%s' with confidence %.2f",
            best["encoding"],
            best["validation"]["confidence_score"],
        )
        return {
            "success": True,
            "final_encoding": best["encoding"],
            "text": best["text"],
            "bom_removed": best["bom_removed"],
            "validation": best["validation"],
            "attempts": attempts,
        }

    logger.error("Failed to decode content with any supported encoding")
    return {
        "success": False,
        "final_encoding": None,
        "text": None,
        "bom_removed": False,
        "validation": None,
        "attempts": attempts,
        "error": "Unable to decode content with any supported encoding",
    }
