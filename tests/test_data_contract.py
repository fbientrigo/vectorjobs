"""
Tests for the data contract schema module.

Covers:
* validate_columns returns correct ValidationResult for valid DataFrames.
* assert_columns raises ValueError with a clear message on missing columns.
* All three CSV contracts are exercised (postings, job_skills, skills).
"""

from __future__ import annotations

import pandas as pd
import pytest

from jobsrec.data.schema import (
    JOB_SKILLS_REQUIRED,
    POSTINGS_REQUIRED,
    SKILLS_REQUIRED,
    assert_columns,
    validate_columns,
)


# ---------------------------------------------------------------------------
# Fixtures — tiny synthetic DataFrames (no Kaggle data needed)
# ---------------------------------------------------------------------------

@pytest.fixture()
def valid_postings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "job_id": [1, 2, 3],
            "title": ["Data Engineer", "ML Engineer", "DevOps Engineer"],
            "description": ["Build pipelines.", "Train models.", "Ship infra."],
            "formatted_experience_level": ["Mid-Senior", "Entry", "Senior"],
            "formatted_work_type": ["Full-time", "Contract", "Full-time"],
            "location": ["NYC", "Remote", "SF"],
        }
    )


@pytest.fixture()
def valid_job_skills() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "job_id": [1, 1, 2],
            "skill_abr": ["PY", "SQL", "PY"],
        }
    )


@pytest.fixture()
def valid_skills() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "skill_abr": ["PY", "SQL", "TF"],
            "skill_name": ["Python", "SQL", "TensorFlow"],
        }
    )


# ---------------------------------------------------------------------------
# validate_columns — happy paths
# ---------------------------------------------------------------------------

class TestValidateColumnsHappy:
    def test_postings_valid(self, valid_postings: pd.DataFrame) -> None:
        result = validate_columns(valid_postings, POSTINGS_REQUIRED, source="postings.csv")
        assert result.valid is True
        assert result.missing_columns == []
        assert result.source == "postings.csv"

    def test_job_skills_valid(self, valid_job_skills: pd.DataFrame) -> None:
        result = validate_columns(valid_job_skills, JOB_SKILLS_REQUIRED, source="job_skills.csv")
        assert result.valid is True

    def test_skills_valid(self, valid_skills: pd.DataFrame) -> None:
        result = validate_columns(valid_skills, SKILLS_REQUIRED, source="skills.csv")
        assert result.valid is True

    def test_extra_columns_are_not_flagged_as_missing(
        self, valid_postings: pd.DataFrame
    ) -> None:
        """Extra columns do not cause failure."""
        result = validate_columns(valid_postings, POSTINGS_REQUIRED)
        assert result.valid is True


# ---------------------------------------------------------------------------
# validate_columns — missing columns
# ---------------------------------------------------------------------------

class TestValidateColumnsMissing:
    def test_single_missing_column(self) -> None:
        df = pd.DataFrame({"job_id": [1], "title": ["Engineer"]})  # no description
        result = validate_columns(df, POSTINGS_REQUIRED, source="postings.csv")
        assert result.valid is False
        assert "description" in result.missing_columns

    def test_multiple_missing_columns(self) -> None:
        df = pd.DataFrame({"job_id": [1]})  # missing title + description
        result = validate_columns(df, POSTINGS_REQUIRED)
        assert sorted(result.missing_columns) == ["description", "title"]

    def test_all_missing(self) -> None:
        df = pd.DataFrame({"unrelated": [1, 2, 3]})
        result = validate_columns(df, POSTINGS_REQUIRED)
        assert set(result.missing_columns) == set(POSTINGS_REQUIRED)

    def test_job_skills_missing_skill_abr(self) -> None:
        df = pd.DataFrame({"job_id": [1, 2]})
        result = validate_columns(df, JOB_SKILLS_REQUIRED, source="job_skills.csv")
        assert result.valid is False
        assert "skill_abr" in result.missing_columns

    def test_skills_missing_skill_name(self) -> None:
        df = pd.DataFrame({"skill_abr": ["PY"]})
        result = validate_columns(df, SKILLS_REQUIRED, source="skills.csv")
        assert result.valid is False
        assert "skill_name" in result.missing_columns


# ---------------------------------------------------------------------------
# assert_columns — exception path
# ---------------------------------------------------------------------------

class TestAssertColumns:
    def test_raises_value_error_on_missing(self) -> None:
        df = pd.DataFrame({"job_id": [1]})
        with pytest.raises(ValueError, match="Missing required columns"):
            assert_columns(df, POSTINGS_REQUIRED, source="postings.csv")

    def test_error_message_names_the_source(self) -> None:
        df = pd.DataFrame({"job_id": [1]})
        with pytest.raises(ValueError, match="postings.csv"):
            assert_columns(df, POSTINGS_REQUIRED, source="postings.csv")

    def test_error_message_lists_missing_columns(self) -> None:
        df = pd.DataFrame({"job_id": [1]})
        with pytest.raises(ValueError, match="description") as exc_info:
            assert_columns(df, POSTINGS_REQUIRED, source="postings.csv")
        assert "title" in str(exc_info.value) or "description" in str(exc_info.value)

    def test_no_error_on_valid_dataframe(self, valid_postings: pd.DataFrame) -> None:
        """Should not raise for a correctly-structured DataFrame."""
        assert_columns(valid_postings, POSTINGS_REQUIRED, source="postings.csv")

    def test_empty_dataframe_with_correct_columns(self) -> None:
        """Zero rows is fine as long as the columns are present."""
        df = pd.DataFrame(columns=list(POSTINGS_REQUIRED))
        assert_columns(df, POSTINGS_REQUIRED)  # should not raise

    def test_extra_columns_do_not_raise(self) -> None:
        df = pd.DataFrame(
            {
                "job_id": [1],
                "title": ["Eng"],
                "description": ["Desc"],
                "extra_col": [99],
            }
        )
        assert_columns(df, POSTINGS_REQUIRED)  # should not raise
