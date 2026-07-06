"""Module to build agent labeling pack."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from jobsrec.extract.labeling import LABEL_CLASSES

AGENT_LABELING_SCHEMA_VERSION = "agent_labeling_v1.0"

JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "AgentLabelingRow",
    "description": "Schema for a single candidate fragment labeling task.",
    "type": "object",
    "required": [
        "task_id",
        "job_id",
        "candidate_index",
        "candidate_text",
        "candidate_source",
        "section_name",
        "title_clean",
        "company_name",
        "company_industry",
        "skills_regex_raw",
        "skills_normalized",
        "context",
        "baseline",
        "label_options",
        "expected_response_schema"
    ],
    "properties": {
        "task_id": {
            "type": "string",
            "description": "Unique stable identifier for this task (usually job_id_candidate_index)"
        },
        "job_id": {
            "type": "string",
            "description": "Original job posting ID"
        },
        "candidate_index": {
            "type": "integer",
            "description": "Index of candidate fragment within the job posting"
        },
        "candidate_text": {
            "type": "string",
            "description": "The candidate text fragment to be labeled"
        },
        "candidate_source": {
            "type": "string",
            "enum": ["title", "paragraph", "li"],
            "description": "Source structure type of the candidate"
        },
        "section_name": {
            "type": "string",
            "description": "Detected section name in the job posting (e.g., requisitos, funciones)"
        },
        "title_clean": {
            "type": "string",
            "description": "Cleaned job title"
        },
        "company_name": {
            "type": "string",
            "description": "Name of the hiring company"
        },
        "company_industry": {
            "type": "string",
            "description": "Industry of the hiring company"
        },
        "skills_regex_raw": {
            "type": "array",
            "items": { "type": "string" },
            "description": "Raw regex-matched skills"
        },
        "skills_normalized": {
            "type": "array",
            "items": { "type": "string" },
            "description": "Normalized regex-matched skills"
        },
        "context": {
            "type": "object",
            "required": ["job_card_text", "description_text_excerpt"],
            "properties": {
                "job_card_text": {
                    "type": "string",
                    "description": "Short truncated context from the full job card"
                },
                "description_text_excerpt": {
                    "type": "string",
                    "description": "Optional short description excerpt"
                }
            }
        },
        "baseline": {
            "type": "object",
            "required": ["regex_has_skill", "regex_skills"],
            "properties": {
                "regex_has_skill": {
                    "type": "boolean",
                    "description": "Whether the regex baseline detected any skills in this candidate text"
                },
                "regex_skills": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of normalized skills detected by the regex baseline"
                }
            }
        },
        "label_options": {
            "type": "array",
            "items": { "type": "string" },
            "description": "The exact list of valid labeling classes"
        },
        "expected_response_schema": {
            "type": "object",
            "required": ["label", "normalized_value", "evidence", "confidence", "notes"],
            "properties": {
                "label": {
                    "type": "string",
                    "description": "One value selected from label_options"
                },
                "normalized_value": {
                    "type": "string",
                    "description": "Normalized representation of the labeled value (if applicable) or empty string"
                },
                "evidence": {
                    "type": "string",
                    "description": "Literal substring from candidate_text showing evidence for the label"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence score from 0.0 to 1.0"
                },
                "notes": {
                    "type": "string",
                    "description": "Short explanation for the labeling decision"
                }
            }
        }
    }
}

PROMPT_MARKDOWN = """# Agent Labeling Prompt & Guidelines

You are an expert annotator tasked with labeling candidate text fragments from job postings into structured categories.
Your labels will be used to train and fine-tune `apolo-slm`, a small language model for job posting parsing.

## Labeling Rules
1. **Single Label**: You must assign exactly one label from the `label_options` list.
2. **Literal Evidence**: The `evidence` field must be copied verbatim (case-sensitive, exact substring) from `candidate_text`.
   - If the label is `IGNORE` or there is no specific evidence, or if the entire text is the evidence, you can copy the whole `candidate_text` or use an empty string as appropriate, but copying the verbatim evidence is highly preferred.
3. **No Unjustified Inference**: Do not infer skills or other attributes that are not explicitly present in the `candidate_text`.
4. **No Invented Normalized Values**: Do not invent a `normalized_value`. If there is a standard normalization (e.g. "python" for "Python", "2 years" for "2 años", "hybrid" for "híbrido"), output it. If there is no clear normalization, set it to an empty string.
5. **Handle Ambiguity**: If you are unsure of the correct class, use `UNCERTAIN`.

## Label Classification Guidelines

- **HARD_SKILL**: Concrete tools, technologies, software, programming languages, databases, framework, hardware, technical methods, or certifications (e.g., Python, SQL, Git, AWS, Scrum, Excel, SAP, Docker, CCNA).
- **DOMAIN_SKILL**: Domain capabilities, business processes, or area-specific expertise (e.g., gestión de inventario, análisis financiero, atención clínica, control de calidad, marketing digital, contabilidad general, liquidación de sueldos).
- **SOFT_SKILL**: Communication, interpersonal skills, teamwork, leadership, autonomy, problem-solving, work ethic, customer orientation, adaptability (e.g., trabajo en equipo, proactividad, buenas relaciones interpersonales, liderazgo).
- **EDUCATION**: Degrees, academic titles, educational levels, fields of study, or university/technical requirements (e.g., Ingeniero Civil, Licenciatura en Administración, técnico nivel superior, universitario completo).
- **EXPERIENCE**: Years of experience, level of seniority, or prior role experience required (e.g., 3 años de experiencia, junior, al menos 1 año trabajando en cargos similares).
- **RESPONSIBILITY**: Duties, tasks, goals, or functions to be performed in the role (e.g., diseñar la arquitectura de software, redactar informes semanales, atender clientes presencialmente).
- **BENEFIT**: Employer perks, benefits, insurance, vouchers, or compensation packages (e.g., seguro de salud complementario, bonos por desempeño, flexibilidad horaria, tickets de almuerzo).
- **LOCATION**: Physical place, region, city, or work modality location details (e.g., Santiago, Las Condes, presencial, híbrido, teletrabajo, remoto).
- **SCHEDULE**: Working hours, shift details, weekly distribution, or day/night details (e.g., Lunes a Viernes de 9:00 a 18:00, turnos rotativos, 45 horas semanales).
- **CONTRACT**: Type of contract (e.g., contrato indefinido, a plazo fijo, boleta de honorarios, por proyecto).
- **IGNORE**: Boilerplate, legal notices, empty fragments, job search navigation, unrelated advertising, or completely irrelevant text fragments.
- **UNCERTAIN**: Used for highly ambiguous, mixed, or unclear cases where classification is not possible.

## JSON Response Format
Your response for each candidate fragment must conform exactly to the following JSON schema:
```json
{
  "label": "<one_label_option>",
  "normalized_value": "<string_or_empty>",
  "evidence": "<literal_substring_from_candidate_text>",
  "confidence": <float_between_0_and_1>,
  "notes": "<short_explanation>"
}
```
"""


def safe_decode_json_list(val: Any) -> list[str]:
    """Safely decode JSON list from candidate fields."""
    if not val:
        return []
    if isinstance(val, str):
        try:
            loaded = json.loads(val)
            if isinstance(loaded, list):
                return [str(item) for item in loaded]
        except (json.JSONDecodeError, TypeError):
            pass
    elif isinstance(val, (list, tuple, set)):
        return [str(item) for item in val]
    return []


def build_agent_labeling_pack(
    silver: pd.DataFrame,
    candidates: pd.DataFrame,
    sample_size: int = 800,
    random_seed: int = 3407,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build representative, deterministic sample for agent labeling.

    Returns:
        tuple containing:
            - list of row dictionaries for agent_labeling_pack.jsonl
            - dictionary for agent_labeling_manifest.json
    """
    # 1. Filter out empty or extremely long candidates
    cand_df = candidates.copy()
    cand_df["text_len"] = cand_df["candidate_text"].str.len().fillna(0)

    # Filter for useful text length (e.g., between 5 and 600 characters)
    is_useful_len = (cand_df["text_len"] >= 5) & (cand_df["text_len"] <= 600)
    if is_useful_len.sum() >= sample_size:
        filtered_df = cand_df[is_useful_len].copy()
    else:
        filtered_df = cand_df.copy()

    # 2. Deduplicate candidate text to ensure diversity
    deduped_df = filtered_df.drop_duplicates(subset=["candidate_text"])
    if len(deduped_df) < sample_size:
        deduped_df = filtered_df.drop_duplicates(subset=["job_id", "candidate_index"])

    # Reset index for clean lookup
    deduped_df = deduped_df.reset_index(drop=True)

    # 3. Merge silver metadata
    meta = silver[["job_id", "title", "company_name", "company_industry", "job_card_text", "description_text"]].copy()
    meta["job_id"] = meta["job_id"].astype(str)
    meta = meta.rename(columns={"title": "title_clean"})
    meta = meta.drop_duplicates(subset=["job_id"])

    deduped_df["job_id_str"] = deduped_df["job_id"].astype(str)
    merged_df = deduped_df.merge(meta, left_on="job_id_str", right_on="job_id", how="left", suffixes=("", "_silver"))

    # 4. Set up strata
    has_skills = merged_df["skills_normalized"].apply(
        lambda x: bool(safe_decode_json_list(x))
    )

    # Top industries in the current subset
    top_industries = []
    if "company_industry" in merged_df.columns:
        top_industries = list(merged_df["company_industry"].dropna().value_counts().index[:5])

    # Dynamic target per stratum
    parts = []

    def add_sample(df_sub: pd.DataFrame, n: int):
        if df_sub.empty:
            return
        parts.append(df_sub.sample(n=min(n, len(df_sub)), random_state=random_seed))

    # Regex skills vs no regex skills
    add_sample(merged_df[has_skills], 100)
    add_sample(merged_df[~has_skills], 100)

    # Sources
    for src in ["title", "paragraph", "li"]:
        add_sample(merged_df[merged_df["candidate_source"] == src], 60)

    # Sections
    sections_list = [
        "requisitos", "habilidades", "conocimientos", "funciones",
        "responsabilidades", "beneficios", "horario", "modalidad", "contrato"
    ]
    for sec in sections_list:
        add_sample(merged_df[merged_df["section_name"] == sec], 40)

    # Industries
    for ind in top_industries:
        add_sample(merged_df[merged_df["company_industry"] == ind], 40)

    # Combine strata
    if parts:
        pool = pd.concat(parts).drop_duplicates(subset=["job_id", "candidate_index"])
    else:
        pool = pd.DataFrame(columns=merged_df.columns)

    # Fill remainder if pool is too small
    target_n = min(sample_size, len(merged_df))
    if len(pool) < target_n:
        remaining = merged_df[~merged_df.index.isin(pool.index)]
        needed = target_n - len(pool)
        if not remaining.empty:
            extra = remaining.sample(n=min(needed, len(remaining)), random_state=random_seed)
            pool = pd.concat([pool, extra]).drop_duplicates(subset=["job_id", "candidate_index"])

    # Final shuffle and selection of exactly target_n rows
    if not pool.empty:
        final_sample = pool.sample(n=target_n, random_state=random_seed).reset_index(drop=True)
    else:
        final_sample = pd.DataFrame(columns=merged_df.columns)

    # 5. Format JSONL output
    rows: list[dict[str, Any]] = []
    for _, row in final_sample.iterrows():
        job_id = str(row["job_id"])
        cand_idx = int(row["candidate_index"])
        raw_skills = safe_decode_json_list(row.get("skills_regex_raw"))
        norm_skills = safe_decode_json_list(row.get("skills_normalized"))

        job_card = str(row.get("job_card_text") or "")
        desc_text = str(row.get("description_text") or "")

        formatted_row = {
            "task_id": f"{job_id}_{cand_idx}",
            "job_id": job_id,
            "candidate_index": cand_idx,
            "candidate_text": str(row["candidate_text"]),
            "candidate_source": str(row["candidate_source"]),
            "section_name": str(row.get("section_name") or ""),
            "title_clean": str(row.get("title_clean") or ""),
            "company_name": str(row.get("company_name") or ""),
            "company_industry": str(row.get("company_industry") or ""),
            "skills_regex_raw": raw_skills,
            "skills_normalized": norm_skills,
            "context": {
                "job_card_text": job_card[:1000],
                "description_text_excerpt": desc_text[:500],
            },
            "baseline": {
                "regex_has_skill": len(norm_skills) > 0,
                "regex_skills": norm_skills,
            },
            "label_options": list(LABEL_CLASSES),
            "expected_response_schema": {
                "label": "one label option",
                "normalized_value": "string or empty",
                "evidence": "literal substring from candidate_text",
                "confidence": "float from 0 to 1",
                "notes": "short explanation",
            }
        }
        rows.append(formatted_row)

    # 6. Compute manifest statistics
    strata_sources: dict[str, int] = {}
    strata_sections: dict[str, int] = {}
    strata_has_skill: dict[str, int] = {"true": 0, "false": 0}

    for r in rows:
        src = r["candidate_source"]
        strata_sources[src] = strata_sources.get(src, 0) + 1

        sec = r["section_name"]
        if sec:
            strata_sections[sec] = strata_sections.get(sec, 0) + 1

        has_sk = "true" if r["baseline"]["regex_has_skill"] else "false"
        strata_has_skill[has_sk] += 1

    manifest = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "schema_version": AGENT_LABELING_SCHEMA_VERSION,
        "random_seed": random_seed,
        "sample_size": sample_size,
        "command_used": " ".join(sys.argv),
        "total_candidates_processed": len(candidates),
        "output_rows_count": len(rows),
        "strata_counts": {
            "sources": strata_sources,
            "sections": strata_sections,
            "regex_has_skill": strata_has_skill,
        }
    }

    return rows, manifest
