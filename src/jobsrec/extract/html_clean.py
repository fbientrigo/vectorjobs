"""HTML cleaning and structured extraction for job description fields."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser


class _OrderedExtractor(HTMLParser):
    """Extract (source, text) pairs from HTML in DOM order.

    source is 'paragraph' for <p> content (outside <li>) or 'li' for <li> content.
    Nested <li> text is merged into the outermost <li> item.
    """

    def __init__(self) -> None:
        super().__init__()
        self._in_p = 0
        self._in_li = 0
        self._current: list[str] = []
        self._results: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag == "li":
            self._in_li += 1
            if self._in_li == 1:
                self._current = []
        elif tag == "p":
            self._in_p += 1
            if self._in_li == 0:
                self._current = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "li" and self._in_li > 0:
            self._in_li -= 1
            if self._in_li == 0:
                text = clean_text(" ".join(self._current))
                if text:
                    self._results.append(("li", text))
                self._current = []
        elif tag == "p" and self._in_p > 0:
            self._in_p -= 1
            if self._in_li == 0:
                text = clean_text(" ".join(self._current))
                if text:
                    self._results.append(("paragraph", text))
                self._current = []

    def handle_data(self, data: str) -> None:
        if self._in_li > 0 or self._in_p > 0:
            self._current.append(data)

    def results(self) -> list[tuple[str, str]]:
        return self._results


def clean_text(text: str) -> str:
    """Apply html.unescape and collapse whitespace."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def extract_ordered(html_text: str) -> list[tuple[str, str]]:
    """Return (source, cleaned_text) pairs in DOM order.

    source is 'paragraph' or 'li'.
    """
    if not html_text:
        return []
    parser = _OrderedExtractor()
    parser.feed(html_text)
    parser.close()
    return parser.results()


def extract_li_items(html_text: str) -> list[str]:
    """Return cleaned text of each <li> element in DOM order."""
    return [text for source, text in extract_ordered(html_text) if source == "li"]


def extract_paragraphs(html_text: str) -> list[str]:
    """Return cleaned text of each <p> element (outside <li>) in DOM order."""
    return [text for source, text in extract_ordered(html_text) if source == "paragraph"]
