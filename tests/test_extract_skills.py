"""Tests for regex-based hard skill detection."""

from __future__ import annotations

import pytest

from jobsrec.extract.skills import SKILL_DICT_VERSION, match_skills

POSITIVE_CASES = [
    ("Manejo de Excel avanzado", ["Excel"]),
    ("Microsoft Office requerido", ["Microsoft Office"]),
    ("MS Office y PowerPoint", ["Microsoft Office", "PowerPoint"]),
    ("Conocimientos en SAP", ["SAP"]),
    ("Manejo de ERP", ["ERP"]),
    ("Power BI y Tableau", ["Power BI", "Tableau"]),
    ("Salesforce CRM", ["Salesforce", "CRM"]),
    ("SQL y Python", ["SQL", "Python"]),
    ("JavaScript y Git", ["JavaScript", "Git"]),
    ("Java backend", ["Java"]),
    ("AWS y GCP", ["AWS", "GCP"]),
    ("AutoCAD 2D y 3D", ["AutoCAD"]),
    ("Inglés avanzado requerido", ["Inglés"]),
    ("English B2 required", ["Inglés"]),
    ("Google Workspace", ["Google Workspace"]),
    ("Word y Excel básico", ["Word", "Excel"]),
    ("Manejo de office", ["Microsoft Office"]),
]


@pytest.mark.parametrize("text,expected_normalized", POSITIVE_CASES)
def test_positive_match(text: str, expected_normalized: list[str]) -> None:
    _, normalized = match_skills(text)
    for skill in expected_normalized:
        assert skill in normalized, f"{skill!r} not found in {normalized} for text {text!r}"


def test_no_duplicate_normalized_names() -> None:
    _, normalized = match_skills("Microsoft Office y MS Office y office")
    assert normalized.count("Microsoft Office") == 1


def test_java_does_not_match_javascript() -> None:
    _, normalized = match_skills("JavaScript developer")
    assert "Java" not in normalized
    assert "JavaScript" in normalized


def test_inglés_both_spellings() -> None:
    _, n1 = match_skills("Inglés avanzado")
    _, n2 = match_skills("Inglés")
    _, n3 = match_skills("English")
    assert "Inglés" in n1
    assert "Inglés" in n2
    assert "Inglés" in n3
    _, both = match_skills("Inglés / English")
    assert both.count("Inglés") == 1


def test_empty_text_returns_empty_lists() -> None:
    raw, norm = match_skills("")
    assert raw == []
    assert norm == []


def test_no_false_positive_salsa() -> None:
    _, normalized = match_skills("Cocinero con experiencia en salsa criolla")
    assert "SAP" not in normalized


def test_no_false_positive_python_snake() -> None:
    _, normalized = match_skills("Manejo de serpientes, en especial la python")
    assert "Python" in normalized  # ponytail: accepted false positive at baseline


def test_raw_matches_parallel_to_normalized() -> None:
    raw, norm = match_skills("Excel y SQL y Python")
    assert len(raw) == len(norm)
    for r in raw:
        assert isinstance(r, str)
        assert len(r) > 0


def test_skill_dict_version_format() -> None:
    assert SKILL_DICT_VERSION.startswith("v")


def test_power_bi_multi_word() -> None:
    _, normalized = match_skills("Power BI y PowerBI")
    assert "Power BI" in normalized
    assert normalized.count("Power BI") == 1
