"""Spanish section anchor detection for job description text."""

from __future__ import annotations

# Ordered sequence of section names/anchors. Order matters: first match wins.
_ANCHORS: tuple[str, ...] = (
    "requisitos",
    "habilidades",
    "conocimientos",
    "funciones",
    "responsabilidades",
    "beneficios",
    "horario",
    "modalidad",
    "contrato",
)

SECTION_NAMES: tuple[str, ...] = _ANCHORS


def detect_section(text: str) -> str:
    """Return the first matched Spanish section anchor name, or '' if none."""
    text_lc = text.lower()
    for name in _ANCHORS:
        if name in text_lc:
            return name
    return ""

