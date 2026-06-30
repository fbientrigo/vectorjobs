"""Tests for Spanish section anchor detection."""

from __future__ import annotations

import pytest

from jobsrec.extract.sections import SECTION_NAMES, detect_section

ALL_ANCHORS = [
    ("Requisitos del cargo:", "requisitos"),
    ("Habilidades requeridas", "habilidades"),
    ("Conocimientos técnicos", "conocimientos"),
    ("Funciones del cargo", "funciones"),
    ("Responsabilidades principales:", "responsabilidades"),
    ("Beneficios de la empresa", "beneficios"),
    ("Horario de trabajo", "horario"),
    ("Modalidad de trabajo", "modalidad"),
    ("Tipo de contrato", "contrato"),
]


@pytest.mark.parametrize("text,expected", ALL_ANCHORS)
def test_detect_each_anchor(text: str, expected: str) -> None:
    assert detect_section(text) == expected


def test_detect_case_insensitive() -> None:
    assert detect_section("REQUISITOS") == "requisitos"
    assert detect_section("Habilidades") == "habilidades"
    assert detect_section("RESPONSABILIDADES") == "responsabilidades"


def test_detect_no_match_returns_empty() -> None:
    assert detect_section("Seleccionaremos al mejor candidato.") == ""
    assert detect_section("") == ""
    assert detect_section("Empresa líder del sector") == ""


def test_detect_first_match_wins() -> None:
    # "requisitos" and "habilidades" both present; requisitos comes first in dict
    assert detect_section("Requisitos y habilidades requeridas") == "requisitos"


def test_section_names_constant() -> None:
    assert "requisitos" in SECTION_NAMES
    assert "responsabilidades" in SECTION_NAMES
    assert len(SECTION_NAMES) == 9


def test_carry_forward_pattern() -> None:
    """Simulate carry-forward: section detected from one paragraph applies to next items."""
    texts = [
        "Responsabilidades principales:",
        "Alcanzar objetivos de ventas.",
        "Gestionar el equipo.",
        "Requisitos del cargo:",
        "Experiencia mínima 2 años.",
    ]
    current = ""
    assignments = []
    for t in texts:
        detected = detect_section(t)
        if detected:
            current = detected
        assignments.append(current)

    assert assignments[0] == "responsabilidades"
    assert assignments[1] == "responsabilidades"  # carries forward
    assert assignments[2] == "responsabilidades"  # carries forward
    assert assignments[3] == "requisitos"          # new section
    assert assignments[4] == "requisitos"          # carries forward
