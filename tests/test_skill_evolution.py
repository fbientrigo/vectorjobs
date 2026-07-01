import json
from pathlib import Path
import numpy as np
import pandas as pd
from click.testing import CliRunner

from jobsrec.cli import main
from jobsrec.trends.skill_evolution import (
    prepare_skill_composition_plot_data,
    run_skill_evolution,
    normalize_skill,
    split_skills,
)


def _mock_jobs() -> pd.DataFrame:
    rows = []
    # Generate 5 tech-like jobs and 5 health-like jobs
    for i in range(10):
        is_tech = i < 5
        month = 4
        day = (i % 5) + 1
        rows.append(
            {
                "job_id": i + 1,
                "title": "Software Engineer" if is_tech else "Registered Nurse",
                "description": (
                    "python sql software developer cloud engineer"
                    if is_tech
                    else "patient clinic nursing care hospital"
                ),
                "skills_text": "python; sql" if is_tech else "nursing; patient care",
                "listed_time": f"2024-0{month}-{day:02d}T12:00:00",
            }
        )
    return pd.DataFrame(rows)


def test_normalize_skill() -> None:
    assert normalize_skill("sql") == "SQL"
    assert normalize_skill("python") == "Python"
    assert normalize_skill("  nursing  ") == "Nursing"
    assert normalize_skill(None) is None


def test_split_skills() -> None:
    assert split_skills("python; sql; python") == ["Python", "SQL"]
    assert split_skills("nursing; patient care") == ["Nursing", "Patient Care"]
    assert split_skills(None) == []


def test_run_skill_evolution_succeeds(tmp_path: Path) -> None:
    input_path = tmp_path / "jobs.parquet"
    outdir = tmp_path / "skill_evolution"
    _mock_jobs().to_parquet(input_path, index=False)

    result = run_skill_evolution(
        input_path=input_path,
        outdir=outdir,
        bin_size="W",
        max_rows=10,
    )

    assert result.output_dir == outdir
    assert (outdir / "manifest.json").exists()
    assert (outdir / "report.md").exists()
    assert (outdir / "domain_assignments.parquet").exists()
    assert (outdir / "domain_skill_monthly.parquet").exists()
    assert (outdir / "skill_long.parquet").exists()

    manifest = result.manifest
    assert manifest["input_row_count"] == 10
    assert manifest["selected_row_count"] == 10
    assert (outdir / "skill_evolution_health.png").exists()
    assert (outdir / "skill_evolution_health.png").stat().st_size > 0


def test_skill_evolution_cli(tmp_path: Path) -> None:
    silver_path = tmp_path / "jobs.parquet"
    outdir = tmp_path / "skill_evolution"
    _mock_jobs().to_parquet(silver_path, index=False)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "skill-evolution",
            "--input",
            str(silver_path),
            "--outdir",
            str(outdir),
            "--bin",
            "W",
            "--max-rows",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads((outdir / "manifest.json").read_text())
    assert manifest["selected_row_count"] == 10
    assert (outdir / "report.md").exists()


def test_skill_composition_sums_to_100_after_top_n_plus_other() -> None:
    # Synthetic dataset for checking threshold-based grouping
    # Month 1 (2024-03): Total mentions = 100
    # Nursing: 50, Patient Care: 40, SQL: 5, Java: 4.9, Python: 0.1
    # Month 2 (2024-04): Total mentions = 100
    # Nursing: 60, Patient Care: 30, SQL: 2, Java: 8
    domain_df = pd.DataFrame(
        {
            "domain": ["health"] * 9,
            "time_bin": ["2024-03"] * 5 + ["2024-04"] * 4,
            "skill": [
                "Nursing", "Patient Care", "SQL", "Java", "Python",
                "Nursing", "Patient Care", "SQL", "Java"
            ],
            "skill_job_count": [
                50.0, 40.0, 5.0, 4.9, 0.1,
                60.0, 30.0, 2.0, 8.0
            ],
            "job_count": [100] * 9,
        }
    )

    composition, ordered = prepare_skill_composition_plot_data(
        domain_df,
        low_support_threshold=5,
        min_skill_share_pct=5.0,
        label="Others"
    )

    # 1. Share totals per group are preserved (sum to 100% / 1.0)
    sums = composition.groupby("time_bin")["composition_share"].sum()
    np.testing.assert_allclose(sums.to_numpy(), np.ones(len(sums)))

    # 2. Ordered list contains explicit skills above threshold + Others, ordered by global counts
    # Global counts: Nursing (110.0), Patient Care (70.0), Java (12.9), SQL (7.0)
    assert ordered == ["Nursing", "Patient Care", "Java", "SQL", "Others"]

    # 3. Threshold boundary:
    # SQL in 2024-03 has share exactly 5.0% -> remains explicit
    m1_sql = composition[(composition["time_bin"] == "2024-03") & (composition["skill"] == "SQL")]
    assert len(m1_sql) == 1
    assert m1_sql.iloc[0]["composition_share"] == 0.05

    # Java in 2024-03 has share 4.9% < 5.0% -> goes to Others (Java share is 0 in composition)
    m1_java = composition[(composition["time_bin"] == "2024-03") & (composition["skill"] == "Java")]
    assert len(m1_java) == 1
    assert m1_java.iloc[0]["composition_share"] == 0.0

    # 4. Grouping is done per month, not globally:
    # Java is explicit in 2024-04 (8.0% >= 5.0%) -> has explicit share in 2024-04
    m2_java = composition[(composition["time_bin"] == "2024-04") & (composition["skill"] == "Java")]
    assert len(m2_java) == 1
    assert m2_java.iloc[0]["composition_share"] == 0.08

    # SQL is grouped in 2024-04 (2.0% < 5.0%) -> goes to Others
    m2_sql = composition[(composition["time_bin"] == "2024-04") & (composition["skill"] == "SQL")]
    assert len(m2_sql) == 1
    assert m2_sql.iloc[0]["composition_share"] == 0.0

    # 5. Others share per group is correct:
    # 2024-03: Java (4.9%) + Python (0.1%) = 5.0%
    m1_others = composition[(composition["time_bin"] == "2024-03") & (composition["skill"] == "Others")]
    assert len(m1_others) == 1
    assert m1_others.iloc[0]["composition_share"] == 0.05

    # 2024-04: SQL (2.0%) = 2.0%
    m2_others = composition[(composition["time_bin"] == "2024-04") & (composition["skill"] == "Others")]
    assert len(m2_others) == 1
    assert m2_others.iloc[0]["composition_share"] == 0.02

    # Support-related assertion (low support logic remains intact)
    # job_count in data is 100, low_support_threshold is 5, so support (100) >= threshold (5) -> not low support
    assert not composition["low_support"].any()


def test_spanish_stopwords_and_wiring() -> None:
    from jobsrec.text.spanish_stopwords import SPANISH_STOPWORDS, APOLO_STOPWORDS
    # 1. Stopword list includes common Spanish connectors.
    for word in ["de", "del", "la", "el", "los", "las", "un", "una", "y", "o", "para", "por", "con", "en", "al"]:
        assert word in SPANISH_STOPWORDS
        assert word in APOLO_STOPWORDS

    # 2. Stopword list does not include meaningful domain words.
    for word in ["salud", "educación", "ventas", "minería", "construcción", "software", "logística", "retail"]:
        assert word not in SPANISH_STOPWORDS
        assert word not in APOLO_STOPWORDS

    # 3. TF-IDF vectorizer wiring receives the custom stopword list where relevant.
    from jobsrec.trends.temporal_clusters import _build_tfidf_svd_embeddings
    _, vec, _, _, _ = _build_tfidf_svd_embeddings(pd.Series([
        "desarrollador de software en Santiago",
        "enfermero de hospital con experiencia",
        "ingeniero civil de obras",
        "vendedor de retail para tienda",
        "profesor de escuela primaria",
    ]), random_state=42)
    assert vec.stop_words is not None
    assert "de" in vec.stop_words
    assert "software" not in vec.stop_words


def test_skill_evolution_with_synthetic_candidates(tmp_path: Path) -> None:
    # 4. skill-evolution can read skills_normalized from a synthetic candidates DataFrame.
    # 5. JSON list parsing works with json.loads.
    # 6. Empty skills_normalized rows are ignored.
    # 7. Candidate skills join correctly to silver jobs by string job_id.
    # 8. Existing behavior still works when --candidates-path is not provided.
    # 9. Manifest/report records the skill source.

    jobs_df = pd.DataFrame([
        {"job_id": 101, "title": "Software Dev", "description": "python sql developer", "skills_text": "ignored", "listed_time": "2024-04-01T12:00:00"},
        {"job_id": 102, "title": "Nurse", "description": "nursing patient care", "skills_text": "ignored", "listed_time": "2024-04-02T12:00:00"},
        {"job_id": 103, "title": "Unmatched Job", "description": "some text", "skills_text": "ignored", "listed_time": "2024-04-03T12:00:00"},
    ])

    candidates_df = pd.DataFrame([
        # Job 101 has preferred source/section, skills SQL and Python
        {"job_id": "101", "candidate_source": "li", "section_name": "requisitos", "skills_normalized": json.dumps(["SQL", "Python"])},
        # Job 101 has empty section candidate (should be ignored since preferred exists)
        {"job_id": "101", "candidate_source": "li", "section_name": "", "skills_normalized": json.dumps(["Java"])},
        # Job 102 has empty section candidate, but since no preferred exists, this is kept for coverage
        {"job_id": "102", "candidate_source": "paragraph", "section_name": "", "skills_normalized": json.dumps(["Nursing"])},
        # Job 102 has empty skills row (should be ignored)
        {"job_id": "102", "candidate_source": "paragraph", "section_name": "", "skills_normalized": json.dumps([])},
        # Job 103 has non-preferred source 'title', should be ignored
        {"job_id": "103", "candidate_source": "title", "section_name": "requisitos", "skills_normalized": json.dumps(["Management"])},
    ])

    jobs_file = tmp_path / "jobs.parquet"
    candidates_file = tmp_path / "candidates.parquet"
    outdir = tmp_path / "skill_evolution"

    jobs_df.to_parquet(jobs_file, index=False)
    candidates_df.to_parquet(candidates_file, index=False)

    # First run: with --candidates-path
    result = run_skill_evolution(
        input_path=jobs_file,
        outdir=outdir,
        bin_size="W",
        max_rows=10,
        candidates_path=candidates_file,
    )

    # Check that skills parsed correctly
    skill_long = pd.read_parquet(outdir / "skill_long.parquet")
    assert set(skill_long["job_id"]) == {"101", "102"}
    assert set(skill_long[skill_long["job_id"] == "101"]["skill"]) == {"SQL", "Python"}
    assert set(skill_long[skill_long["job_id"] == "102"]["skill"]) == {"Nursing"}

    # Check manifest/report
    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["skill_source"] == "candidates"
    assert manifest["candidates_available"] is True
    assert "Regex skills cover named hard skills only" in manifest["caveat"]

    report_text = (outdir / "report.md").read_text(encoding="utf-8")
    assert "Skill source used: `candidates`" in report_text
    assert "Deterministic candidate skills available: `True`" in report_text

    # Second run: without --candidates-path (existing behavior works)
    outdir_legacy = tmp_path / "skill_evolution_legacy"
    result_legacy = run_skill_evolution(
        input_path=jobs_file,
        outdir=outdir_legacy,
        bin_size="W",
        max_rows=10,
    )
    # legacy reads skills_text from jobs (which is "ignored" -> "Ignored")
    skill_long_legacy = pd.read_parquet(outdir_legacy / "skill_long.parquet")
    assert set(skill_long_legacy["job_id"]) == {"101", "102", "103"}
    assert set(skill_long_legacy["skill"]) == {"Ignored"}

    manifest_legacy = json.loads((outdir_legacy / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_legacy["skill_source"] == "skills_text"
    assert manifest_legacy["candidates_available"] is False

    report_text_legacy = (outdir_legacy / "report.md").read_text(encoding="utf-8")
    assert "Skill evolution is using the legacy `skills_text` source" in report_text_legacy
