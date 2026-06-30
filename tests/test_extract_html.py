"""Tests for html_clean extraction utilities."""

from __future__ import annotations

from jobsrec.extract.html_clean import (
    clean_text,
    extract_li_items,
    extract_ordered,
    extract_paragraphs,
)

SIMPLE_HTML = """
<p>Responsabilidades principales:</p>
<ul>
  <li><strong>Gestionar el equipo de ventas.</strong></li>
  <li>Reportar resultados mensuales.</li>
</ul>
<p>Requisitos:</p>
<ul>
  <li>Experiencia en Excel y SAP.</li>
</ul>
"""


def test_extract_li_items_basic() -> None:
    items = extract_li_items(SIMPLE_HTML)
    assert len(items) == 3
    assert items[0] == "Gestionar el equipo de ventas."
    assert items[1] == "Reportar resultados mensuales."
    assert items[2] == "Experiencia en Excel y SAP."


def test_extract_paragraphs_basic() -> None:
    paras = extract_paragraphs(SIMPLE_HTML)
    assert len(paras) == 2
    assert paras[0] == "Responsabilidades principales:"
    assert paras[1] == "Requisitos:"


def test_extract_ordered_dom_order() -> None:
    items = extract_ordered(SIMPLE_HTML)
    sources = [s for s, _ in items]
    # paragraph comes before its li items
    assert sources[0] == "paragraph"
    assert sources[1] == "li"
    assert sources[2] == "li"
    assert sources[3] == "paragraph"
    assert sources[4] == "li"


def test_extract_li_empty_html() -> None:
    assert extract_li_items("") == []
    assert extract_li_items("   ") == []


def test_extract_paragraphs_empty_html() -> None:
    assert extract_paragraphs("") == []


def test_extract_li_no_lists() -> None:
    assert extract_li_items("<p>Solo texto.</p>") == []


def test_extract_paragraphs_no_paragraphs() -> None:
    assert extract_paragraphs("<ul><li>Item</li></ul>") == []


def test_clean_text_unescapes_html_entities() -> None:
    assert clean_text("&amp;") == "&"
    assert clean_text("&lt;b&gt;") == "<b>"
    assert clean_text("café &amp; vino") == "café & vino"


def test_clean_text_collapses_whitespace() -> None:
    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text("\n\ttab\n") == "tab"


def test_clean_text_empty() -> None:
    assert clean_text("") == ""


def test_li_inside_p_is_not_paragraph() -> None:
    """Text inside <li> should not appear as a paragraph even with nested tags."""
    html = "<ul><li><p>nested para</p></li></ul>"
    assert extract_paragraphs(html) == []
    assert extract_li_items(html) == ["nested para"]


def test_nested_li_content_merged() -> None:
    """Nested <li> text is merged into the outer <li>."""
    html = "<ul><li>outer <ul><li>inner</li></ul> text</li></ul>"
    items = extract_li_items(html)
    assert len(items) == 1
    assert "outer" in items[0]
    assert "inner" in items[0]
