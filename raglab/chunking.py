"""Sentence/line-aware chunking with no third-party dependencies."""

import re
from typing import List


def _units(text: str) -> List[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    paragraphs = re.split(r"\n\s*\n", normalized)
    units: List[str] = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        # Keep bullets as separate units; split prose at sentence boundaries.
        pieces = re.split(r"(?<=[。！？!?])\s*|\n+", paragraph)
        for piece in pieces:
            piece = piece.strip()
            if piece:
                units.append(piece)
    return units


def chunk_text(text: str, max_chars: int = 600) -> List[str]:
    """Split text into chunks of at most roughly ``max_chars`` characters."""

    if max_chars < 80:
        raise ValueError("max_chars must be at least 80")

    chunks: List[str] = []
    current: List[str] = []

    def flush() -> None:
        if current:
            value = " ".join(current).strip()
            if value:
                chunks.append(value)
            current.clear()

    for unit in _units(text):
        if len(unit) > max_chars:
            flush()
            start = 0
            while start < len(unit):
                end = min(start + max_chars, len(unit))
                chunks.append(unit[start:end].strip())
                if end >= len(unit):
                    break
                start = max(end - 80, start + 1)
            continue

        candidate = " ".join(current + [unit])
        if current and len(candidate) > max_chars:
            flush()
        current.append(unit)

    flush()
    return [chunk for chunk in chunks if chunk]
