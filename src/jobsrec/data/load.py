"""SQLite bronze reader and silver Parquet writer."""

from __future__ import annotations

import ast
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd

from jobsrec.data.schema import BRONZE_REQUIRED_TABLES, SILVER_REQUIRED, SILVER_SCHEMA_VERSION, assert_columns
from jobsrec.text.job_card import build_job_card_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SilverResult:
    """Outcome of :func:`build_silver`."""

    output_path: Path
    manifest_path: Path
    input_rows: int
    output_rows: int


class _HTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)

    def text(self) -> str:
        text = re.sub(r"\s+", " ", " ".join(self.parts)).strip()
        return re.sub(r"\s+([.,;:!?])", r"\1", text)


@dataclass(frozen=True)
class ParsedCompany:
    """Structured result of parsing a scraper company field."""

    name: str
    confidential: bool | None
    raw: str
    city: str
    region: str
    industry: str
    parse_error: bool


def strip_html(value: object) -> str:
    """Convert scraper HTML descriptions into readable plain text."""
    if value is None or pd.isna(value):
        return ""
    parser = _HTMLTextParser()
    parser.feed(str(value))
    parser.close()
    return parser.text()


def parse_company(value: object) -> ParsedCompany:
    """Parse a scraper company field (Python repr dict) into structured fields."""
    _empty = ParsedCompany("", None, "", "", "", "", False)
    if value is None or pd.isna(value):
        return _empty
    raw = str(value).strip()
    if not raw:
        return _empty
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return ParsedCompany(raw, None, raw, "", "", "", True)
    if not isinstance(parsed, dict):
        return ParsedCompany(raw, None, raw, "", "", "", True)
    confidential = parsed.get("confidencial")
    name = parsed.get("denominacion") or parsed.get("nombre") or parsed.get("name") or ""
    if not name and confidential is True:
        name = "Confidencial"
    return ParsedCompany(
        name=str(name).strip(),
        confidential=confidential if isinstance(confidential, bool) else None,
        raw=raw,
        city=str(parsed.get("ciudad") or "").strip(),
        region=str(parsed.get("provincia") or "").strip(),
        industry=str(parsed.get("industria") or "").strip(),
        parse_error=False,
    )


def load_bronze_sqlite(input_db: Path | str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load official scraper tables and compute observation aggregates."""
    input_db = Path(input_db)
    if not input_db.exists():
        raise FileNotFoundError(f"SQLite bronze DB not found at {input_db}")

    with sqlite3.connect(input_db) as con:
        tables = {
            row[0]
            for row in con.execute(
                "select name from sqlite_master where type='table'"
            ).fetchall()
        }
        missing = sorted(set(BRONZE_REQUIRED_TABLES) - tables)
        if missing:
            raise ValueError(f"Bronze DB is missing required tables: {missing}")

        jobs = pd.read_sql_query(
            "select id as job_id, title, company, location, description as description_html, status from jobs",
            con,
        )
        observations = pd.read_sql_query(
            """
            select
                job_id,
                min(seen_at) as first_seen_at,
                max(seen_at) as last_seen_at,
                count(*) as times_seen,
                count(distinct crawl_id) as crawl_count
            from job_observations
            group by job_id
            """,
            con,
        )
        stats = {
            "jobs": int(pd.read_sql_query("select count(*) as n from jobs", con)["n"][0]),
            "job_observations": int(
                pd.read_sql_query("select count(*) as n from job_observations", con)["n"][0]
            ),
            "crawl_runs": int(
                pd.read_sql_query("select count(*) as n from crawl_runs", con)["n"][0]
            ),
        }

    jobs["job_id"] = jobs["job_id"].astype(str)
    observations["job_id"] = observations["job_id"].astype(str)
    return jobs.merge(observations, on="job_id", how="left"), stats


def build_silver(
    input_db: Path | str,
    output_dir: Path | str,
    config: dict[str, Any] | None = None,
) -> SilverResult:
    """Build silver Parquet from the official scraper SQLite bronze DB."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, source_stats = load_bronze_sqlite(input_db)
    input_rows = len(rows)

    rows["title"] = rows["title"].fillna("").astype(str).str.strip()
    rows["description_html"] = rows["description_html"].fillna("").astype(str).str.strip()
    rows["description_text"] = rows["description_html"].map(strip_html)
    company = rows["company"].map(parse_company)
    rows["company_name"] = [c.name for c in company]
    rows["company_confidential"] = [c.confidential for c in company]
    rows["company_raw"] = [c.raw for c in company]
    rows["company_city"] = [c.city for c in company]
    rows["company_region"] = [c.region for c in company]
    rows["company_industry"] = [c.industry for c in company]
    rows["company_parse_error"] = [c.parse_error for c in company]
    rows["location"] = rows["location"].where(rows["location"].notna(), None)
    rows["skills_text"] = ""
    rows["job_card_text"] = rows.apply(
        lambda row: build_job_card_text(
            title=row["title"],
            description=row["description_text"],
            location=str(row["location"] or ""),
            skills_text="",
        ),
        axis=1,
    )
    rows = rows[
        rows["title"].str.strip().ne("") | rows["description_text"].str.strip().ne("")
    ].copy()

    for col in ("times_seen", "crawl_count"):
        rows[col] = rows[col].fillna(0).astype("int64")

    silver = rows[list(SILVER_REQUIRED)].copy()
    assert_columns(silver, SILVER_REQUIRED, source="silver jobs.parquet")

    output_path = output_dir / "jobs.parquet"
    silver.to_parquet(output_path, index=False)

    manifest = _make_manifest(
        input_rows=input_rows,
        output_rows=len(silver),
        input_path=str(input_db),
        output_path=str(output_path),
        config=config or {},
        source_stats=source_stats,
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote silver Parquet: %s (%d rows)", output_path, len(silver))
    return SilverResult(output_path, manifest_path, input_rows, len(silver))


def _make_manifest(
    *,
    input_rows: int,
    output_rows: int,
    input_path: str,
    output_path: str,
    config: dict[str, Any],
    source_stats: dict[str, int],
) -> dict[str, Any]:
    return {
        "stage": "build-silver",
        "bronze_format": "sqlite",
        "silver_schema_version": SILVER_SCHEMA_VERSION,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "input_rows": input_rows,
        "output_rows": output_rows,
        "input_path": input_path,
        "output_path": output_path,
        "config": config,
        "source_tables": source_stats,
        "skills_source": "none",
        "skills_note": "Bronze source has no skills table yet; skills_text is kept empty for downstream compatibility.",
        "silver_columns": list(SILVER_REQUIRED),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write *manifest* dict as pretty-printed JSON to *path*."""
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
