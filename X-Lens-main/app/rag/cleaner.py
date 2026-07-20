from __future__ import annotations

import re

from bs4 import BeautifulSoup


def clean_html(html: str) -> str:
    """
    Convert website HTML into clean text while preserving useful
    paragraph, heading, list and FAQ boundaries.

    Keeping line boundaries allows the chunker to place each FAQ
    question together with its answer.
    """

    soup = BeautifulSoup(html, "html.parser")

    # Remove content that is normally unrelated to the page knowledge.
    for tag in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "noscript",
            "svg",
            "form",
            "button",
        ]
    ):
        tag.decompose()

    # Add line breaks after content blocks before extracting text.
    block_tags = [
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "li",
        "dt",
        "dd",
        "article",
        "section",
        "div",
        "br",
    ]

    for tag in soup.find_all(block_tags):
        tag.append("\n")

    raw_text = soup.get_text(separator=" ")

    lines: list[str] = []

    for raw_line in raw_text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()

        if not line:
            continue

        # Avoid repeated consecutive lines caused by nested HTML blocks.
        if lines and line == lines[-1]:
            continue

        lines.append(line)

    return "\n".join(lines)