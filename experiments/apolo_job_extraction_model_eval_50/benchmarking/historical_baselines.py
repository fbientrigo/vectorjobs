"""Read-only adapter for the historical Gemini-era baseline in
data/silver/agent_labeling/. Never writes to, rewrites, or regenerates that
directory, and never calls any model.

Provenance is investigated, not assumed. What the repo actually shows:

- No manifest under data/silver/agent_labeling/ names a generating model,
  a reasoning-effort mode, or per-row request/retry/latency/cost metadata.
- scripts/llm_labeler.py calls the Gemini API, but for a DIFFERENT schema
  (fragment-level label/normalized_value/evidence/confidence/notes, written
  to model_labeled_candidates*.jsonl) than the job-level ApoloExtraction
  schema (schema_version/document_type/skills/education_requirements/
  seniority) used by job_level_apolo_extractions_*.jsonl. No script in this
  repository produces the job-level file's exact schema.
- The larger manifests (1100, 21100, all) add the note "Model-assisted weak
  labels and rule-based extraction... Prefix of 100 rows manually labeled
  with high-quality controls" — evidence the generation method is mixed
  and/or changed between batch sizes, not a single uniform LLM pass.
- git history has no commits touching data/silver/agent_labeling/ (the
  directory is generated data, not version-controlled), so commit messages
  cannot corroborate provenance either.

Given this, provenance_status/model_configuration_status/reasoning_mode are
"unknown" and generation_method is "mixed" (per the manifest's own note).
display_name uses the task's suggested fallback convention for an unverified
Gemini-family baseline; it is NOT evidence of model identity.
"""

import json
from pathlib import Path

from . import loaders

DATA_DIR = Path(loaders.EXPERIMENT_DIR).parent.parent / "data" / "silver" / "agent_labeling"
HISTORICAL_JSONL = DATA_DIR / "job_level_apolo_extractions_all.jsonl"
HISTORICAL_MANIFEST = DATA_DIR / "job_level_apolo_extractions_all_manifest.json"
AGENT_LABELING_PROMPT = DATA_DIR / "agent_labeling_prompt.md"

BASELINE_ID = "gemini_historical_all"
DISPLAY_NAME = "Gemini 3.5 Flash historical baseline"


def load_historical_manifest() -> dict | None:
    if not HISTORICAL_MANIFEST.exists():
        return None
    return json.loads(HISTORICAL_MANIFEST.read_text(encoding="utf-8"))


def load_overlap_rows(frozen_job_ids: list[str]) -> dict[str, dict]:
    """Stream the 35,716-row historical file; keep only rows whose job_id is
    in the frozen 50. Never loads/copies the full file into another artifact.
    """
    wanted = set(str(j) for j in frozen_job_ids)
    overlap: dict[str, dict] = {}
    if not HISTORICAL_JSONL.exists():
        return overlap
    with HISTORICAL_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            job_id = str(row.get("job_id"))
            if job_id in wanted:
                overlap[job_id] = row
            if len(overlap) == len(wanted):
                break
    return overlap


def structural_diagnostics(overlap_rows: dict[str, dict], frozen_sample: dict[str, dict]) -> dict:
    """Extraction-behavior diagnostics computed only on the overlap rows.

    Mirrors the shape of the OpenRouter runs' manifest diagnostics so the
    two are comparable, but this is a completeness join over an imported
    dataset, not a reliability run — there is no attempted/failed framing.
    """
    n_overlap = len(overlap_rows)
    if n_overlap == 0:
        return {
            "n_overlap_rows": 0, "total_skills": None, "skills_per_job_mean": None,
            "empty_skill_rows": None, "broad_domain_skill_rate": None,
            "confidence_mean": None, "evidence_grounded_rate": None,
        }

    all_skills = []
    empty_skill_rows = 0
    evidence_grounded = 0
    for job_id, row in overlap_rows.items():
        skills = row.get("apolo_extraction", {}).get("skills", [])
        all_skills.extend(skills)
        if not skills:
            empty_skill_rows += 1
        job_card_text = frozen_sample.get(job_id, {}).get("job_card_text", "")
        if all((s.get("evidence", "") or "") in job_card_text for s in skills):
            evidence_grounded += 1

    total_skills = len(all_skills)
    confidences = [s.get("confidence") for s in all_skills if isinstance(s.get("confidence"), (int, float))]
    domain_skills = sum(1 for s in all_skills if s.get("category") == "domain")

    return {
        "n_overlap_rows": n_overlap,
        "total_skills": total_skills,
        "skills_per_job_mean": total_skills / n_overlap,
        "empty_skill_rows": empty_skill_rows,
        "broad_domain_skill_rate": (domain_skills / total_skills) if total_skills else None,
        "confidence_mean": (sum(confidences) / len(confidences)) if confidences else None,
        "evidence_grounded_rate": evidence_grounded / n_overlap,
    }


def build_baseline_report(frozen_job_ids: list[str], frozen_sample: dict[str, dict] | None = None) -> dict:
    """The normalized aggregate report — no raw historical rows attached."""
    manifest = load_historical_manifest()
    overlap_rows = load_overlap_rows(frozen_job_ids)
    frozen_sample = frozen_sample if frozen_sample is not None else loaders.load_frozen_sample()
    diagnostics = structural_diagnostics(overlap_rows, frozen_sample)

    missing_job_ids = sorted(set(str(j) for j in frozen_job_ids) - set(overlap_rows))
    schema_versions = {
        row.get("apolo_extraction", {}).get("schema_version") for row in overlap_rows.values()
    }
    source_schema_version = schema_versions.pop() if len(schema_versions) == 1 else None

    return {
        "baseline_id": BASELINE_ID,
        "display_name": DISPLAY_NAME,
        "source_path": str(HISTORICAL_JSONL),
        "source_manifest_path": str(HISTORICAL_MANIFEST),
        "n_source_rows": (manifest or {}).get("n_rows"),
        "frozen_50_overlap": len(overlap_rows),
        "frozen_50_missing_job_ids": missing_job_ids,
        "source_prompt_hash": None,
        "source_schema_version": source_schema_version,
        "model_name": None,
        "reasoning_mode": "unknown",
        "generation_method": "mixed",
        "provenance_status": "unknown",
        "model_configuration_status": "unknown",
        "directly_comparable": False,
        "comparability_notes": (
            "display_name follows the task's suggested fallback naming convention "
            "for an unverified Gemini-family baseline; no artifact in this repo "
            "names the generating model or reasoning mode, so this is NOT evidence "
            "of model identity. agent_labeling_prompt.md documents a different task "
            "(candidate-fragment classification) than the job-level ApoloExtraction "
            "schema found in job_level_apolo_extractions_*.jsonl, so the prompt "
            "actually used to generate this baseline was not found in this "
            "repository. The source manifest's own note ('Model-assisted weak "
            "labels and rule-based extraction... Prefix of 100 rows manually "
            "labeled with high-quality controls') indicates a mixed generation "
            "method, not a single uniform LLM pass. Treat this baseline as a "
            "historical reference, not a controlled benchmark competitor."
        ),
        "cost_available": False,
        "latency_available": False,
        "retry_metadata_available": False,
        **diagnostics,
    }
