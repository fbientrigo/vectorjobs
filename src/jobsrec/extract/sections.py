"""Spanish section anchor detection for job description text."""

from __future__ import annotations

import re

# Maps section name → pattern. Order matters: first match wins.
_ANCHORS: dict[str, str] = {
    "requisitos": r"requisitos",
    "habilidades": r"habilidades",
    "conocimientos": r"conocimientos",
    "funciones": r"funciones",
    "responsabilidades": r"responsabilidades",
    "beneficios": r"beneficios",
    "horario": r"horario",
    "modalidad": r"modalidad",
    "contrato": r"contrato",
}

_PATTERNS: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in _ANCHORS.items()
}

SECTION_NAMES: tuple[str, ...] = tuple(_ANCHORS.keys())


def detect_section(text: str) -> str:
    """Return the first matched Spanish section anchor name, or '' if none."""
    for name, pattern in _PATTERNS.items():
        if pattern.search(text):
            return name
    return ""
