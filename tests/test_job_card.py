"""
Tests for the job_card_text builder.

Covers:
* Determinism — calling build_job_card_text twice with the same args yields
  identical output.
* Section ordering matches the data contract exactly.
* Blank optional fields are omitted (no ghost "Skills: " lines).
* Required fields always appear even if optional fields are absent.
* Whitespace in field values is stripped.
"""

from __future__ import annotations

import pytest

from jobsrec.text.job_card import build_job_card_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sections(text: str) -> list[str]:
    """Return the section label tokens from each line of *text*."""
    return [line.split(":")[0].strip() for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_args_same_output(self) -> None:
        kwargs = dict(
            title="Data Engineer",
            description="Build scalable data pipelines.",
            experience="Mid-Senior level",
            work_type="Full-time",
            location="New York, NY",
            skills_text="Python, SQL, Spark",
        )
        first = build_job_card_text(**kwargs)
        second = build_job_card_text(**kwargs)
        assert first == second

    def test_different_args_different_output(self) -> None:
        a = build_job_card_text(title="A", description="Desc A")
        b = build_job_card_text(title="B", description="Desc B")
        assert a != b

    def test_repeated_calls_are_identical(self) -> None:
        """Multiple repeated calls must always match the first."""
        kwargs = dict(
            title="ML Engineer",
            description="Train LLMs.",
            skills_text="PyTorch, Transformers",
        )
        results = [build_job_card_text(**kwargs) for _ in range(10)]
        assert len(set(results)) == 1


# ---------------------------------------------------------------------------
# Section ordering
# ---------------------------------------------------------------------------

class TestSectionOrdering:
    _EXPECTED_ORDER = ["Title", "Experience", "Work type", "Location", "Skills", "Description"]

    def test_all_sections_in_canonical_order(self) -> None:
        text = build_job_card_text(
            title="SWE",
            description="Code stuff.",
            experience="Senior",
            work_type="Full-time",
            location="Remote",
            skills_text="Python, Go",
        )
        sections = _sections(text)
        assert sections == self._EXPECTED_ORDER

    def test_partial_sections_preserve_order(self) -> None:
        """With only title + skills + description, they must still be in order."""
        text = build_job_card_text(
            title="SWE",
            description="Code stuff.",
            skills_text="Python",
        )
        sections = _sections(text)
        assert sections == ["Title", "Skills", "Description"]

    def test_title_is_always_first(self) -> None:
        text = build_job_card_text(title="Engineer", description="Desc.")
        assert text.startswith("Title:")

    def test_description_is_always_last(self) -> None:
        text = build_job_card_text(
            title="Engineer",
            description="Desc.",
            skills_text="Python",
        )
        assert text.splitlines()[-1].startswith("Description:")


# ---------------------------------------------------------------------------
# Blank / missing optional fields
# ---------------------------------------------------------------------------

class TestOptionalFieldOmission:
    def test_blank_experience_omitted(self) -> None:
        text = build_job_card_text(title="T", description="D", experience="")
        assert "Experience:" not in text

    def test_blank_work_type_omitted(self) -> None:
        text = build_job_card_text(title="T", description="D", work_type="")
        assert "Work type:" not in text

    def test_blank_location_omitted(self) -> None:
        text = build_job_card_text(title="T", description="D", location="")
        assert "Location:" not in text

    def test_blank_skills_omitted(self) -> None:
        text = build_job_card_text(title="T", description="D", skills_text="")
        assert "Skills:" not in text

    def test_whitespace_only_skills_omitted(self) -> None:
        text = build_job_card_text(title="T", description="D", skills_text="   ")
        assert "Skills:" not in text

    def test_none_defaults_work(self) -> None:
        """Calling with only required args should not raise."""
        text = build_job_card_text(title="Eng", description="Desc.")
        assert "Title: Eng" in text
        assert "Description: Desc." in text


# ---------------------------------------------------------------------------
# Content fidelity
# ---------------------------------------------------------------------------

class TestContentFidelity:
    def test_title_value_is_preserved(self) -> None:
        text = build_job_card_text(title="Senior Data Engineer", description="D")
        assert "Title: Senior Data Engineer" in text

    def test_skills_text_is_preserved(self) -> None:
        text = build_job_card_text(
            title="T", description="D", skills_text="Python, SQL, dbt"
        )
        assert "Skills: Python, SQL, dbt" in text

    def test_description_value_is_preserved(self) -> None:
        desc = "We build real-time pipelines using Apache Kafka."
        text = build_job_card_text(title="T", description=desc)
        assert desc in text

    def test_leading_trailing_whitespace_stripped(self) -> None:
        text = build_job_card_text(
            title="  Engineer  ",
            description="  Build things.  ",
        )
        assert "Title: Engineer" in text
        assert "Description: Build things." in text
