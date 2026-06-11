"""
job_card_text builder — deterministic, section-ordered concatenation.

The output format is defined in docs/specs/01_data_contract.md §3.
Only non-blank sections are included so that missing optional fields
do not leave "Skills: " ghost lines in the text.
"""

from __future__ import annotations


# Section labels in the canonical order defined by the data contract.
_SECTION_ORDER: tuple[str, ...] = (
    "title",
    "experience",
    "work_type",
    "location",
    "skills",
    "description",
)

_LABELS: dict[str, str] = {
    "title": "Title",
    "experience": "Experience",
    "work_type": "Work type",
    "location": "Location",
    "skills": "Skills",
    "description": "Description",
}


def build_job_card_text(
    *,
    title: str,
    description: str,
    experience: str = "",
    work_type: str = "",
    location: str = "",
    skills_text: str = "",
) -> str:
    """
    Build a deterministic plain-text job card.

    Sections are always emitted in the canonical order defined in the data
    contract.  Optional sections whose value is blank (empty string or
    whitespace-only) are silently omitted.

    Parameters
    ----------
    title:
        Job title — always required.
    description:
        Full job description — always required.
    experience:
        Formatted experience level (e.g. ``"Mid-Senior level"``).
    work_type:
        Formatted work type (e.g. ``"Full-time"``).
    location:
        Job location string.
    skills_text:
        Comma-separated skill names built by joining ``job_skills`` with
        ``skills``.

    Returns
    -------
    str
        Multi-line text with ``\n`` line endings, ready for TF-IDF ingestion.

    Examples
    --------
    >>> text = build_job_card_text(
    ...     title="Data Engineer",
    ...     description="Build pipelines.",
    ...     skills_text="Python, SQL",
    ... )
    >>> text.startswith("Title: Data Engineer")
    True
    >>> "Skills: Python, SQL" in text
    True
    >>> "Experience:" not in text  # blank field omitted
    True
    """
    values: dict[str, str] = {
        "title": title.strip(),
        "experience": experience.strip(),
        "work_type": work_type.strip(),
        "location": location.strip(),
        "skills": skills_text.strip(),
        "description": description.strip(),
    }

    lines: list[str] = []
    for key in _SECTION_ORDER:
        val = values[key]
        if val:
            lines.append(f"{_LABELS[key]}: {val}")

    return "\n".join(lines)
