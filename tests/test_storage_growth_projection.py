from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from scripts.storage_growth_projection import (
    PLOT_FILES,
    SCENARIOS,
    StorageAnchors,
    future_base_monthly_jobs,
    project_storage,
    read_bronze_metrics,
    scenario_total_horizon,
    write_outputs,
)


def make_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        create table jobs (
            id text primary key,
            title text,
            company text,
            location text,
            description text,
            status text
        );
        create table crawl_runs (
            id integer primary key,
            started_at timestamp
        );
        create table job_observations (
            job_id text,
            crawl_id integer,
            seen_at timestamp,
            primary key (job_id, crawl_id)
        );
        """
    )
    con.executemany("insert into jobs(id) values (?)", [("a",), ("b",), ("c",)])
    con.executemany("insert into crawl_runs(id, started_at) values (?, ?)", [(1, "2026-03-01"), (2, "2026-04-01")])
    con.executemany(
        "insert into job_observations(job_id, crawl_id, seen_at) values (?, ?, ?)",
        [
            ("a", 1, "2026-03-01 10:00:00"),
            ("a", 2, "2026-04-01 10:00:00"),
            ("b", 2, "2026-04-02 10:00:00"),
            ("c", 2, "2026-04-03 10:00:00"),
        ],
    )
    con.commit()
    con.close()


def test_storage_projection_formulas_and_outputs(tmp_path: Path) -> None:
    db_path = tmp_path / "jobs.db"
    make_db(db_path)
    bronze = read_bronze_metrics(db_path)
    anchors = StorageAnchors(
        bronze_bytes_per_job=10.0,
        silver_bytes_per_job=5.0,
        fixed_gold_bytes=100.0,
        variable_gold_bytes_per_job=2.0,
        measured={},
    )

    assert bronze.jobs_count == 3
    assert bronze.observations_count == 4
    assert bronze.crawl_runs_count == 2
    assert bronze.first_seen_monthly == {"2026-03": 1, "2026-04": 2}
    assert future_base_monthly_jobs(bronze.first_seen_monthly) == 2

    rows = project_storage(
        bronze=bronze,
        anchors=anchors,
        horizon_months=2,
        scenarios={"0%/month": 0.0},
    )

    assert [row["month"] for row in rows] == ["2026-05", "2026-06"]
    assert [row["new_jobs"] for row in rows] == [2, 2]
    assert [row["cumulative_jobs"] for row in rows] == [5, 7]
    assert [row["gold_bundles_added"] for row in rows] == [1, 1]
    assert rows[0]["bronze_bytes"] == 50.0
    assert rows[0]["silver_bytes"] == 25.0
    assert rows[0]["gold_bytes"] == 110.0
    assert rows[0]["total_bytes"] == 185.0
    assert rows[0]["total_added_bytes"] == 140.0
    assert rows[1]["gold_bytes"] == 224.0
    assert rows[1]["total_bytes"] == 329.0
    assert rows[1]["total_added_bytes"] == 144.0

    scenario_rows = project_storage(
        bronze=bronze,
        anchors=anchors,
        horizon_months=2,
        scenarios=SCENARIOS,
    )
    horizon = scenario_total_horizon(
        bronze,
        anchors,
        target_gb=scenario_rows[5]["total_gb"],
        max_months=5,
    )
    assert horizon["months"] == 2
    assert scenario_rows[4]["total_gb"] < horizon["final_worst_case_gb"]
    assert horizon["final_worst_case_gb"] >= horizon["target_gb"]

    outputs = write_outputs(rows, anchors, tmp_path / "projection", scenario_rows=scenario_rows)
    csv = pd.read_csv(outputs["csv"])
    manifest = json.loads(Path(outputs["manifest"]).read_text(encoding="utf-8"))
    assert csv["total_bytes"].tolist() == [185.0, 329.0]
    assert manifest["total_scenario"]["label"] == "12%/month"
    assert "12% more new data" in manifest["total_scenario"]["meaning"]
    assert manifest["scenario_total_horizon"]["target_gb"] == 50.0
    for name in PLOT_FILES:
        assert Path(outputs[name]).exists()
    assert not (tmp_path / "projection" / "storage_growth_mb_log.png").exists()
    assert not (tmp_path / "projection" / "storage_growth_gb_log.png").exists()
