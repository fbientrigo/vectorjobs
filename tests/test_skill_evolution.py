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
    domain_df = pd.DataFrame(
        {
            "domain": ["health"] * 6,
            "time_bin": ["2024-03", "2024-03", "2024-03", "2024-04", "2024-04", "2024-04"],
            "skill": ["Nursing", "Patient Care", "SQL", "Nursing", "Patient Care", "Excel"],
            "skill_job_count": [3, 1, 1, 1, 2, 1],
            "job_count": [5, 5, 5, 4, 4, 4],
            "share_pct": [60, 20, 20, 25, 50, 25],
        }
    )

    composition, ordered = prepare_skill_composition_plot_data(domain_df, top_n=2, low_support_threshold=5)

    sums = composition.groupby("time_bin")["composition_share"].sum()
    np.testing.assert_allclose(sums.to_numpy(), np.ones(len(sums)))
    assert ordered == ["Nursing", "Patient Care", "Other"]
    assert composition.loc[composition["time_bin"] == "2024-04", "low_support"].all()
