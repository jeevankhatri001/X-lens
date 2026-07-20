from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from uuid import uuid4


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_url: str
    content_hash: str


def chunk_text(
    text: str,
    source_url: str,
    size: int = 180,
    overlap: int = 30,
) -> list[Chunk]:
    """
    Split cleaned website text into small, searchable chunks.

    Smaller chunks improve retrieval of precise facts such as:
    - exact addresses
    - course names
    - scholarships
    - admission requirements

    Paragraph and sentence boundaries are preserved where possible.
    """

    clean_text = _normalize_text(text)

    if not clean_text:
        return []

    sections = _split_into_sections(clean_text)

    chunks: list[Chunk] = []
    seen_hashes: set[str] = set()

    for section in sections:
        section_chunks = _split_section(
            section=section,
            size=size,
            overlap=overlap,
        )

        for part in section_chunks:
            normalized = " ".join(part.lower().split())

            if not normalized:
                continue

            digest = sha256(
                normalized.encode("utf-8")
            ).hexdigest()

            if digest in seen_hashes:
                continue

            seen_hashes.add(digest)

            chunks.append(
                Chunk(
                    chunk_id=str(uuid4()),
                    text=part,
                    source_url=source_url,
                    content_hash=digest,
                )
            )

    return chunks


def _normalize_text(text: str) -> str:
    """
    Normalize spaces while preserving useful line boundaries.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []

    for line in text.split("\n"):
        clean_line = re.sub(r"[ \t]+", " ", line).strip()

        if clean_line:
            lines.append(clean_line)

    return "\n".join(lines)


def _split_into_sections(text: str) -> list[str]:
    """
    Create sections using line and sentence boundaries.

    A question-like line is kept with the text that follows it,
    which is useful for FAQ pages.
    """

    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    if len(lines) <= 1:
        return _split_by_sentences(text)

    sections: list[str] = []
    current: list[str] = []

    for line in lines:
        is_question = line.endswith("?")

        if is_question and current:
            sections.append(" ".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append(" ".join(current).strip())

    return [
        section
        for section in sections
        if section
    ]


def _split_by_sentences(text: str) -> list[str]:
    """
    Fallback when the cleaner has removed paragraph boundaries.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def _split_section(
    section: str,
    size: int,
    overlap: int,
) -> list[str]:
    """
    Split one section into word-limited chunks.
    """

    words = section.split()

    if len(words) <= size:
        return [section.strip()]

    chunks: list[str] = []
    step = max(1, size - overlap)

    for start in range(0, len(words), step):
        part_words = words[start:start + size]

        if not part_words:
            break

        part = " ".join(part_words).strip()

        if part:
            chunks.append(part)

        if start + size >= len(words):
            break

    return chunks