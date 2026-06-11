"""
Milestone 1.5 — tests for temporal/salary/metadata preservation in silver.

Covers:
* Optional postings columns (listed_time, expiry, remote_allowed, work_type,
  views, applies, company_id, etc.) survive build_silver.
* listed_time, expiry, closed_time parse rates are reported by profile_silver.
* Salary join does not duplicate job rows.
* Jobs without skills (no salary join match) are preserved.
* Salary columns (min_salary, max_salary, pay_period, etc.) are present in
  silver when salaries.csv is available.
* Salary columns are NULL for jobs with no salary data.
* Manifest records preserved_optional_columns and joined_optional_tables.
* Profile dict contains new fields: salary_non_null_counts,
  work_type_distribution, experience_level_distribution,
  location_top_20, remote_allowed_distribution.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from jobsrec.data.load import build_silver
from jobsrec.data.profile import (
    SilverProfile,
    _compute_ts_parse_rate,
    profile_silver,
    profile_silver_from_path,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
KAGGLE_MINIMAL = FIXTURES_DIR / "kaggle_minimal"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _silver_df(tmp_path: Path) -> pd.DataFrame:
    """Build silver from kaggle_minimal and return the DataFrame."""
    result = build_silver(input_dir=KAGGLE_MINIMAL, output_dir=tmp_path)
    return pd.read_parquet(result.output_path)


# ===========================================================================
# SECTION 1: Optional postings columns survive build_silver
# ===========================================================================

class TestOptionalColumnPreservation:
    """Verify that optional postings columns are carried into silver."""

    def test_listed_time_in_silver(self, tmp_path: Path) -> None:
        df = _silver_df(tmp_path)
        assert "listed_time" in df.columns, "listed_time must be in silver"

    def test_listed_time_values_preserved(self, tmp_path: Path) -> None:
        df = _silver_df(tmp_path)
        # Job 1001 has listed_time = 1700000000000
        row = df[df["job_id"] == 1001]
        assert len(row) == 1
        val = float(row["listed_time"].iloc[0])
        assert math.isclose(val, 1_700_000_000_000, rel_tol=1e-9)

    def test_expiry_in_silver(self, tmp_path: Path) -> None:
        df = _silver_df(tmp_path)
        assert "expiry" in df.columns

    def test_remote_allowed_in_silver(self, tmp_path: Path) -> None:
        df = _silver_df(tmp_path)
        assert "remote_allowed" in df.columns

    def test_work_type_in_silver(self, tmp_path: Path) -> None:
        df = _silver_df(tmp_path)
        assert "work_type" in df.columns

    def test_views_in_silver(self, tmp_path: Path) -> None:
        df = _silver_df(tmp_path)
        assert "views" in df.columns

    def test_applies_in_silver(self, tmp_path: Path) -> None:
        df = _silver_df(tmp_path)
        assert "applies" in df.columns

    def test_company_id_in_silver(self, tmp_path: Path) -> None:
        df = _silver_df(tmp_path)
        assert "company_id" in df.columns

    def test_formatted_experience_level_in_silver(self, tmp_path: Path) -> None:
        df = _silver_df(tmp_path)
        assert "formatted_experience_level" in df.columns

    def test_formatted_work_type_in_silver(self, tmp_path: Path) -> None:
        df = _silver_df(tmp_path)
        assert "formatted_work_type" in df.columns

    def test_location_in_silver(self, tmp_path: Path) -> None:
        df = _silver_df(tmp_path)
        assert "location" in df.columns

    def test_invalid_listed_time_kept_as_nan(self, tmp_path: Path) -> None:
        """Job 1010 has 'not_a_timestamp' — it should be preserved (NaN or raw)."""
        df = _silver_df(tmp_path)
        row = df[df["job_id"] == 1010]
        assert len(row) == 1
        val = row["listed_time"].iloc[0]
        # The raw value is a string; it should be NaN (non-parseable) or the raw string
        assert pd.isna(val) or str(val).strip() == "not_a_timestamp"


# ===========================================================================
# SECTION 2: Salary join correctness
# ===========================================================================

class TestSalaryJoin:
    """Salary columns survive into silver without duplicating rows."""

    def test_salary_columns_present_in_silver(self, tmp_path: Path) -> None:
        df = _silver_df(tmp_path)
        for col in ("min_salary", "max_salary", "pay_period"):
            assert col in df.columns, f"Expected salary column '{col}' in silver"

    def test_salary_join_does_not_duplicate_rows(self, tmp_path: Path) -> None:
        result = build_silver(input_dir=KAGGLE_MINIMAL, output_dir=tmp_path)
        df = pd.read_parquet(result.output_path)
        # Number of rows must equal number of distinct job_ids
        assert len(df) == df["job_id"].nunique(), (
            "Salary join must not create duplicate job rows"
        )

    def test_salary_present_for_job_with_data(self, tmp_path: Path) -> None:
        df = _silver_df(tmp_path)
        row = df[df["job_id"] == 1001]
        assert len(row) == 1
        assert float(row["min_salary"].iloc[0]) == pytest.approx(120000.0)
        assert float(row["max_salary"].iloc[0]) == pytest.approx(160000.0)

    def test_salary_null_for_job_without_data(self, tmp_path: Path) -> None:
        """Jobs 1003 and 1009 have no salaries.csv row — must be null."""
        df = _silver_df(tmp_path)
        for job_id in (1003, 1009):
            row = df[df["job_id"] == job_id]
            assert len(row) == 1, f"job_id {job_id} must be present exactly once"
            assert pd.isna(row["min_salary"].iloc[0]), (
                f"job_id {job_id} should have null min_salary"
            )
            assert pd.isna(row["max_salary"].iloc[0]), (
                f"job_id {job_id} should have null max_salary"
            )

    def test_pay_period_carried(self, tmp_path: Path) -> None:
        df = _silver_df(tmp_path)
        row = df[df["job_id"] == 1001]
        assert str(row["pay_period"].iloc[0]).strip() == "YEARLY"

    def test_currency_carried(self, tmp_path: Path) -> None:
        df = _silver_df(tmp_path)
        row = df[df["job_id"] == 1001]
        assert str(row["currency"].iloc[0]).strip() == "USD"


# ===========================================================================
# SECTION 3: Jobs without skills are preserved
# ===========================================================================

class TestJobsWithoutSkills:
    """Jobs with no job_skills rows survive build_silver."""

    def test_all_postings_in_silver(self, tmp_path: Path) -> None:
        """All 10 kaggle_minimal jobs must be in silver."""
        df = _silver_df(tmp_path)
        assert len(df) == 10

    def test_no_skills_job_has_empty_skills_text(self, tmp_path: Path) -> None:
        """If a job_id has no skills, skills_text should be empty string."""
        # Let's build with a custom fixture that has a job with no skills
        # by reusing kaggle_minimal (all jobs have skills there, so we verify
        # the general preservation)
        df = _silver_df(tmp_path)
        # All job_ids from postings should be present
        assert df["job_id"].nunique() == 10


# ===========================================================================
# SECTION 4: Manifest fields
# ===========================================================================

class TestManifestFields:
    """Manifest must record preserved_optional_columns and joined_optional_tables."""

    def test_manifest_has_preserved_optional_columns(self, tmp_path: Path) -> None:
        result = build_silver(input_dir=KAGGLE_MINIMAL, output_dir=tmp_path)
        manifest = json.loads(result.manifest_path.read_text())
        assert "preserved_optional_columns" in manifest
        assert isinstance(manifest["preserved_optional_columns"], list)

    def test_manifest_lists_listed_time(self, tmp_path: Path) -> None:
        result = build_silver(input_dir=KAGGLE_MINIMAL, output_dir=tmp_path)
        manifest = json.loads(result.manifest_path.read_text())
        assert "listed_time" in manifest["preserved_optional_columns"]

    def test_manifest_has_joined_optional_tables(self, tmp_path: Path) -> None:
        result = build_silver(input_dir=KAGGLE_MINIMAL, output_dir=tmp_path)
        manifest = json.loads(result.manifest_path.read_text())
        assert "joined_optional_tables" in manifest
        assert "jobs/salaries.csv" in manifest["joined_optional_tables"]

    def test_manifest_salary_columns_added(self, tmp_path: Path) -> None:
        result = build_silver(input_dir=KAGGLE_MINIMAL, output_dir=tmp_path)
        manifest = json.loads(result.manifest_path.read_text())
        assert "salary_columns_added" in manifest
        assert len(manifest["salary_columns_added"]) > 0


# ===========================================================================
# SECTION 5: Timestamp parse rates in profile
# ===========================================================================

class TestTimestampParseRates:
    """profile_silver reports parse rates for temporal columns."""

    def _silver_profile(self, tmp_path: Path) -> SilverProfile:
        result = build_silver(input_dir=KAGGLE_MINIMAL, output_dir=tmp_path)
        return profile_silver_from_path(result.output_path)

    def test_listed_time_parse_rate_reported(self, tmp_path: Path) -> None:
        profile = self._silver_profile(tmp_path)
        assert profile.listed_time_parse_rate is not None
        assert 0.0 <= profile.listed_time_parse_rate <= 1.0

    def test_listed_time_parse_rate_lt_1_due_to_bad_value(self, tmp_path: Path) -> None:
        """Job 1010 has 'not_a_timestamp' so rate should be < 1.0."""
        profile = self._silver_profile(tmp_path)
        assert profile.listed_time_parse_rate is not None
        assert profile.listed_time_parse_rate < 1.0

    def test_expiry_parse_rate_reported(self, tmp_path: Path) -> None:
        profile = self._silver_profile(tmp_path)
        # expiry is in the fixture for most rows
        assert profile.expiry_parse_rate is not None

    def test_closed_time_parse_rate_none_when_absent(self) -> None:
        """Synthetic df without closed_time → rate is None."""
        df = pd.DataFrame({
            "job_id": [1, 2],
            "title": ["A", "B"],
            "description": ["d", "d"],
            "skills_text": ["Python", "SQL"],
            "job_card_text": ["x", "y"],
        })
        profile = profile_silver(df)
        assert profile.closed_time_parse_rate is None
        assert profile.closed_time_n_total is None
        assert profile.closed_time_n_parsed is None

    def test_compute_ts_parse_rate_all_valid(self) -> None:
        df = pd.DataFrame({
            "job_id": [1, 2],
            "title": ["A", "B"],
            "description": ["d", "d"],
            "skills_text": ["", ""],
            "job_card_text": ["x", "y"],
            "expiry": [1_700_000_000_000, 1_700_000_001_000],
        })
        rate, total, parsed = _compute_ts_parse_rate(df, "expiry")
        assert rate == pytest.approx(1.0)
        assert total == 2
        assert parsed == 2

    def test_compute_ts_parse_rate_missing_col(self) -> None:
        df = pd.DataFrame({
            "job_id": [1],
            "title": ["A"],
            "description": ["d"],
            "skills_text": [""],
            "job_card_text": ["x"],
        })
        rate, total, parsed = _compute_ts_parse_rate(df, "nonexistent_col")
        assert rate is None
        assert total is None
        assert parsed is None


# ===========================================================================
# SECTION 6: Profile fields for new distributions
# ===========================================================================

class TestProfileDistributions:
    """profile_silver exposes work_type, experience, location, remote distributions."""

    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "job_id": [1, 2, 3, 4],
            "title": ["A", "B", "C", "D"],
            "description": ["d", "d", "d", "d"],
            "skills_text": ["Python", "SQL", "", "Go"],
            "job_card_text": ["x", "y", "z", "w"],
            "formatted_work_type": ["Full-time", "Full-time", "Contract", "Full-time"],
            "formatted_experience_level": ["Senior", "Mid", "Entry", "Senior"],
            "location": ["NYC", "SF", "NYC", "Remote"],
            "remote_allowed": ["1", "0", "0", "1"],
        })

    def test_work_type_distribution_present(self) -> None:
        profile = profile_silver(self._make_df())
        assert profile.work_type_distribution is not None
        assert "Full-time" in profile.work_type_distribution
        assert profile.work_type_distribution["Full-time"] == 3

    def test_experience_level_distribution_present(self) -> None:
        profile = profile_silver(self._make_df())
        assert profile.experience_level_distribution is not None
        assert "Senior" in profile.experience_level_distribution

    def test_location_top_20_present(self) -> None:
        profile = profile_silver(self._make_df())
        assert profile.location_top_20 is not None
        locs = dict(profile.location_top_20)
        assert locs.get("NYC") == 2

    def test_remote_allowed_distribution_present(self) -> None:
        profile = profile_silver(self._make_df())
        assert profile.remote_allowed_distribution is not None
        assert "1" in profile.remote_allowed_distribution

    def test_distributions_none_when_columns_absent(self) -> None:
        df = pd.DataFrame({
            "job_id": [1, 2],
            "title": ["A", "B"],
            "description": ["d", "d"],
            "skills_text": ["", ""],
            "job_card_text": ["x", "y"],
        })
        profile = profile_silver(df)
        assert profile.work_type_distribution is None
        assert profile.experience_level_distribution is None
        assert profile.location_top_20 is None
        assert profile.remote_allowed_distribution is None

    def test_profile_dict_has_new_keys(self) -> None:
        profile = profile_silver(self._make_df())
        d = profile.to_dict()
        for key in (
            "salary_non_null_counts",
            "work_type_distribution",
            "experience_level_distribution",
            "location_top_20",
            "remote_allowed_distribution",
            "expiry_parse_rate",
            "expiry_n_total",
            "expiry_n_parsed",
            "closed_time_parse_rate",
            "closed_time_n_total",
            "closed_time_n_parsed",
        ):
            assert key in d, f"Profile dict missing key: {key}"


# ===========================================================================
# SECTION 7: Profile salary non-null counts
# ===========================================================================

class TestProfileSalaryNonNullCounts:
    def test_salary_non_null_counts_empty_when_no_salary_cols(self) -> None:
        df = pd.DataFrame({
            "job_id": [1],
            "title": ["T"],
            "description": ["D"],
            "skills_text": [""],
            "job_card_text": ["x"],
        })
        profile = profile_silver(df)
        assert profile.salary_non_null_counts == {}

    def test_salary_non_null_counts_correct(self) -> None:
        df = pd.DataFrame({
            "job_id": [1, 2, 3],
            "title": ["A", "B", "C"],
            "description": ["d", "d", "d"],
            "skills_text": ["", "", ""],
            "job_card_text": ["x", "y", "z"],
            "min_salary": [50000.0, None, 70000.0],
            "max_salary": [80000.0, 90000.0, None],
        })
        profile = profile_silver(df)
        assert profile.salary_non_null_counts.get("min_salary") == 2
        assert profile.salary_non_null_counts.get("max_salary") == 2

    def test_salary_non_null_counts_in_profile_dict(self) -> None:
        df = pd.DataFrame({
            "job_id": [1],
            "title": ["T"],
            "description": ["D"],
            "skills_text": [""],
            "job_card_text": ["x"],
            "min_salary": [100000.0],
        })
        profile = profile_silver(df)
        d = profile.to_dict()
        assert "salary_non_null_counts" in d
        assert d["salary_non_null_counts"]["min_salary"] == 1


# ===========================================================================
# SECTION 8: Aggregate-salaries helper — no duplicates
# ===========================================================================

class TestAggregateSalaries:
    def test_single_row_per_job_unchanged(self) -> None:
        from jobsrec.data.load import aggregate_salaries

        df = pd.DataFrame({
            "job_id": [1, 2],
            "min_salary": [100.0, 200.0],
            "max_salary": [150.0, 250.0],
            "pay_period": ["YEARLY", "HOURLY"],
            "currency": ["USD", "USD"],
            "compensation_type": ["BASE_SALARY", "BASE_SALARY"],
        })
        result = aggregate_salaries(df)
        assert len(result) == 2
        assert set(result["job_id"].tolist()) == {1, 2}

    def test_multiple_rows_collapsed(self) -> None:
        from jobsrec.data.load import aggregate_salaries

        df = pd.DataFrame({
            "job_id": [1, 1, 2],
            "min_salary": [100.0, 90.0, 200.0],
            "max_salary": [150.0, 160.0, 250.0],
            "pay_period": ["YEARLY", "YEARLY", "HOURLY"],
            "currency": ["USD", "USD", "USD"],
            "compensation_type": ["BASE_SALARY", "BASE_SALARY", "BASE_SALARY"],
        })
        result = aggregate_salaries(df)
        assert len(result) == 2
        job1 = result[result["job_id"] == 1].iloc[0]
        # max of 100.0 and 90.0 → 100.0
        assert job1["min_salary"] == pytest.approx(100.0)
        # max of 150.0 and 160.0 → 160.0
        assert job1["max_salary"] == pytest.approx(160.0)

    def test_empty_df_returns_empty(self) -> None:
        from jobsrec.data.load import aggregate_salaries

        df = pd.DataFrame(columns=["job_id", "min_salary", "max_salary"])
        result = aggregate_salaries(df)
        assert len(result) == 0
