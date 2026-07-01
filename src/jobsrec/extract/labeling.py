"""M1: Build a balanced labeling seed CSV for manual annotation."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LABEL_CLASSES: tuple[str, ...] = (
    "HARD_SKILL",
    "DOMAIN_SKILL",
    "SOFT_SKILL",
    "EDUCATION",
    "EXPERIENCE",
    "RESPONSIBILITY",
    "BENEFIT",
    "LOCATION",
    "SCHEDULE",
    "CONTRACT",
    "IGNORE",
    "UNCERTAIN",
)

OUTPUT_COLUMNS: tuple[str, ...] = (
    "label",
    "job_id",
    "candidate_index",
    "candidate_text",
    "candidate_source",
    "section_name",
    "skills_normalized",
    "title_clean",
    "company_name",
    "company_industry",
    "notes",
)

_SECTIONS = ("requisitos", "habilidades", "conocimientos", "funciones", "responsabilidades")
_SOURCES = ("title", "li", "paragraph")
_INDUSTRIES = ("Salud", "Retail", "Telecomunicaciones", "Banca", "Financiera",
               "Educación", "Construcción", "Servicios", "Consultoría")


def build_labeling_seed(
    silver: pd.DataFrame,
    candidates: pd.DataFrame,
    sample_size: int = 500,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Return a deterministic balanced labeling seed DataFrame."""
    from jobsrec.extract.candidates import get_stratified_partitions

    parts = get_stratified_partitions(
        silver=silver,
        candidates=candidates,
        sample_size=sample_size,
        sources=_SOURCES,
        sections=_SECTIONS,
        industries=_INDUSTRIES,
        random_seed=random_seed,
    )

    pool = pd.concat(parts).drop_duplicates(subset=["job_id", "candidate_index"])

    if len(pool) < sample_size:
        remainder = candidates[
            ~candidates.index.isin(pool.index)
        ].sample(frac=1, random_state=random_seed)
        pool = pd.concat([pool, remainder]).drop_duplicates(subset=["job_id", "candidate_index"])

    seed = pool.sample(frac=1, random_state=random_seed).head(sample_size).reset_index(drop=True)

    # join title / company cols from silver
    meta = silver[["job_id", "title", "company_name", "company_industry"]].copy()
    meta["job_id"] = meta["job_id"].astype(str)
    meta = meta.rename(columns={"title": "title_clean"})
    meta = meta.drop_duplicates(subset=["job_id"])

    seed["job_id"] = seed["job_id"].astype(str)
    seed = seed.merge(meta, on="job_id", how="left")

    seed["label"] = ""
    seed["notes"] = ""

    return seed[list(OUTPUT_COLUMNS)]


def write_labeling_seed(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
