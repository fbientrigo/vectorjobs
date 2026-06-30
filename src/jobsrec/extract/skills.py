"""Regex-based hard-skill detector for Chilean job postings.

Only covers obvious, unambiguous named hard skills. Domain skills
(e.g. "gestión de inventario") require a trained model — not in scope here.
"""

from __future__ import annotations

import re

SKILL_DICT_VERSION = "v0.1"

# (pattern, normalized_name) — order matters for overlapping patterns.
# More specific patterns (e.g. "power bi") come before generic substrings.
_RAW_SKILLS: list[tuple[str, str]] = [
    (r"\bpower\s*bi\b", "Power BI"),
    (r"\bgoogle\s*workspace\b", "Google Workspace"),
    (r"\bmicrosoft\s*office\b", "Microsoft Office"),
    (r"\bms\s*office\b", "Microsoft Office"),
    (r"\bpowerpoint\b", "PowerPoint"),
    (r"\bword\b", "Word"),
    (r"\bexcel\b", "Excel"),
    (r"\boffice\b", "Microsoft Office"),
    (r"\bsalesforce\b", "Salesforce"),
    (r"\btableau\b", "Tableau"),
    (r"\bsap\b", "SAP"),
    (r"\berp\b", "ERP"),
    (r"\bcrm\b", "CRM"),
    (r"\bsql\b", "SQL"),
    (r"\bpython\b", "Python"),
    (r"\bjavascript\b", "JavaScript"),
    (r"\bjava\b", "Java"),
    (r"\bgit\b", "Git"),
    (r"\baws\b", "AWS"),
    (r"\bgcp\b", "GCP"),
    (r"\bautocad\b", "AutoCAD"),
    (r"\bingl[eé]s\b", "Inglés"),
    (r"\benglish\b", "Inglés"),
]

_COMPILED: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern, re.IGNORECASE), name)
    for pattern, name in _RAW_SKILLS
]


def match_skills(text: str) -> tuple[list[str], list[str]]:
    """Return (raw_matches, normalized_names) found in text.

    Each normalized skill appears at most once per call.
    """
    if not text:
        return [], []
    raw_matches: list[str] = []
    normalized: list[str] = []
    seen: set[str] = set()
    for pattern, name in _COMPILED:
        if name in seen:
            continue
        m = pattern.search(text)
        if m:
            raw_matches.append(m.group())
            normalized.append(name)
            seen.add(name)
    return raw_matches, normalized
