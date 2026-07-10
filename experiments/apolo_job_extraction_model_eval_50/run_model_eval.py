#!/usr/bin/env python
"""Run one OpenRouter model over the frozen 50-job sample and score its output.

Usage:
    python experiments/apolo_job_extraction_model_eval_50/run_model_eval.py --model deepseek/deepseek-v4-flash
    python experiments/apolo_job_extraction_model_eval_50/run_model_eval.py --model deepseek/deepseek-v4-flash --resume
    python experiments/apolo_job_extraction_model_eval_50/run_model_eval.py --model deepseek/deepseek-v4-flash --only-job-id 1118271814 --resume

Writes, under outputs/<safe_model_name>/:
    extractions.jsonl           - rows that passed validation
    raw_responses.jsonl         - every raw API response/error, one per job
    failures.jsonl              - rejected calls or outputs
    validation_failures.jsonl   - field-level validation failures
    manifest.json               - run statistics and quality metrics

This is a weak-label evaluation harness, not ground truth. See README.md.
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("run_model_eval")

EXPERIMENT_DIR = Path(__file__).parent
SAMPLE_PATH = EXPERIMENT_DIR / "input" / "jobs_sample_50.jsonl"
PROMPT_PATH = EXPERIMENT_DIR / "prompts" / "apolo_extraction_base_prompt.md"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

ALLOWED_LANGUAGE = {"es", "en", "mixed", "unknown"}
ALLOWED_CATEGORY = {"technical", "soft", "tool", "language", "domain", "methodology", "certification", "other"}
ALLOWED_LEVEL = {"basic", "intermediate", "advanced", "expert", "unknown"}
ALLOWED_SENIORITY = {"intern", "junior", "semi_senior", "senior", "lead", "unknown"}
ALLOWED_WARNINGS = {
    "missing_dates", "ambiguous_skill", "low_information_input",
    "ambiguous_education_requirement", "mixed_language_input",
    "truncated_input", "possible_pii", "schema_uncertain",
}
REQUIRED_EXTRACTION_FIELDS = (
    "schema_version", "document_type", "language", "skills",
    "education_requirements", "seniority", "warnings",
)

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


def safe_model_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=EXPERIMENT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(file_obj, row: dict) -> None:
    file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")
    file_obj.flush()


def call_openrouter(api_key: str, model: str, system_prompt: str, user_prompt: str) -> dict:
    """Call OpenRouter chat completions. Returns the raw parsed JSON response body."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "usage": {"include": True},
    }

    backoff = RETRY_BACKOFF_SECONDS
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            if response.status_code == 200:
                return response.json()
            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            logger.warning(f"OpenRouter call failed (attempt {attempt + 1}/{MAX_RETRIES}): {last_error}")
        except Exception as e:  # noqa: BLE001 - network/timeout errors are all retryable here
            last_error = str(e)
            logger.warning(f"OpenRouter call raised (attempt {attempt + 1}/{MAX_RETRIES}): {last_error}")
        time.sleep(backoff)
        backoff *= 2

    return {"error": last_error}


def schema_failure(path: str, reason: str) -> dict:
    return {"field_path": path, "evidence": None, "reason": reason}


def validate_extraction(apolo_extraction: dict, job_card_text: str) -> list[dict]:
    """Return field-level failures. Empty list means the extraction is valid."""
    failures = []
    if not isinstance(apolo_extraction, dict):
        return [schema_failure("apolo_extraction", "expected object")]

    for field in REQUIRED_EXTRACTION_FIELDS:
        if field not in apolo_extraction:
            failures.append(schema_failure(f"apolo_extraction.{field}", "missing required field"))

    if failures:
        return failures

    if apolo_extraction.get("document_type") != "job_posting":
        failures.append(schema_failure("apolo_extraction.document_type", "expected job_posting"))
    if apolo_extraction.get("language") not in ALLOWED_LANGUAGE:
        failures.append(schema_failure("apolo_extraction.language", "invalid enum"))
    if apolo_extraction.get("seniority") not in ALLOWED_SENIORITY:
        failures.append(schema_failure("apolo_extraction.seniority", "invalid enum"))

    warnings = apolo_extraction.get("warnings")
    if not isinstance(warnings, list):
        failures.append(schema_failure("apolo_extraction.warnings", "expected list"))
    else:
        for i, warning in enumerate(warnings):
            if warning not in ALLOWED_WARNINGS:
                failures.append(schema_failure(f"apolo_extraction.warnings[{i}]", "invalid enum"))

    skills = apolo_extraction.get("skills")
    if not isinstance(skills, list):
        failures.append(schema_failure("apolo_extraction.skills", "expected list"))
    else:
        for i, skill in enumerate(skills):
            path = f"apolo_extraction.skills[{i}]"
            if not isinstance(skill, dict):
                failures.append(schema_failure(path, "expected object"))
                continue
            if skill.get("category") not in ALLOWED_CATEGORY:
                failures.append(schema_failure(f"{path}.category", "invalid enum"))
            if skill.get("level") not in ALLOWED_LEVEL:
                failures.append(schema_failure(f"{path}.level", "invalid enum"))
            confidence = skill.get("confidence")
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                failures.append(schema_failure(f"{path}.confidence", "expected number in [0, 1]"))
            evidence = skill.get("evidence", "")
            if evidence and evidence not in job_card_text:
                failures.append({
                    "field_path": f"{path}.evidence",
                    "evidence": evidence,
                    "reason": "evidence is not a substring of job_card_text",
                })

    education_requirements = apolo_extraction.get("education_requirements")
    if not isinstance(education_requirements, list):
        failures.append(schema_failure("apolo_extraction.education_requirements", "expected list"))
    else:
        for i, edu in enumerate(education_requirements):
            path = f"apolo_extraction.education_requirements[{i}]"
            if not isinstance(edu, dict):
                failures.append(schema_failure(path, "expected object"))
                continue
            if not isinstance(edu.get("required"), bool):
                failures.append(schema_failure(f"{path}.required", "expected bool"))
            evidence = edu.get("evidence", "")
            if evidence and evidence not in job_card_text:
                failures.append({
                    "field_path": f"{path}.evidence",
                    "evidence": evidence,
                    "reason": "evidence is not a substring of job_card_text",
                })

    return failures


def failure_kind(failures: list[dict]) -> str:
    if any("substring" in failure["reason"] for failure in failures):
        return "validation_failure"
    return "schema_failure"


def build_user_prompt(job: dict) -> str:
    return (
        f"job_id: {job['job_id']}\n"
        f"title: {job['title']}\n\n"
        f"job_card_text:\n{job['job_card_text']}\n\n"
        f"description_text (supplementary context only, not an evidence source):\n{job['description_text']}\n"
    )


def aggregate_manifest(
    args: argparse.Namespace,
    jobs: list[dict],
    extraction_rows: list[dict],
    failures: list[dict],
    prior_manifest: dict,
    started_at: str,
    finished_at: str,
    wall_time_seconds: float,
    prompt_tokens_delta: int,
    completion_tokens_delta: int,
    cost_delta: float,
    cost_available: bool,
    latencies: list[float],
    sample_path: Path,
) -> dict:
    n_rows_completed = len(extraction_rows)
    completed_job_ids = {str(row["job_id"]) for row in extraction_rows}
    all_skills = [s for row in extraction_rows for s in row["apolo_extraction"]["skills"]]
    total_skills = len(all_skills)
    total_education_requirements = sum(len(row["apolo_extraction"]["education_requirements"]) for row in extraction_rows)

    seniority_counts = {}
    warning_counts = {}
    for row in extraction_rows:
        ext = row["apolo_extraction"]
        seniority_counts[ext["seniority"]] = seniority_counts.get(ext["seniority"], 0) + 1
        for warning in ext["warnings"]:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1

    confidences = [s["confidence"] for s in all_skills]
    confidence_mean = sum(confidences) / len(confidences) if confidences else None
    confidence_1_rate = (sum(1 for c in confidences if c == 1.0) / len(confidences)) if confidences else None
    domain_skills = sum(1 for s in all_skills if s["category"] == "domain")
    broad_domain_skill_rate = (domain_skills / total_skills) if total_skills else None
    empty_skill_rows = sum(1 for row in extraction_rows if len(row["apolo_extraction"]["skills"]) == 0)
    skills_per_job_mean = (total_skills / n_rows_completed) if n_rows_completed else None

    prior_prompt_tokens = prior_manifest.get("prompt_tokens", 0) or 0
    prior_completion_tokens = prior_manifest.get("completion_tokens", 0) or 0
    prior_cost = prior_manifest.get("estimated_cost_usd")
    estimated_cost = None
    if cost_available or prior_cost is not None:
        estimated_cost = (prior_cost or 0.0) + cost_delta

    unresolved_failures = [failure for failure in failures if str(failure.get("job_id")) not in completed_job_ids]
    counts = {}
    for failure in unresolved_failures:
        counts[failure["failure_type"]] = counts.get(failure["failure_type"], 0) + 1

    validation_failure_count = counts.get("validation_failure", 0)
    schema_failures = counts.get("schema_failure", 0)
    json_parse_failures = counts.get("non_json_output", 0)
    empty_output_failures = counts.get("empty_output", 0)
    request_failures = counts.get("api_error", 0)

    return {
        "model": args.model,
        "n_rows_requested": prior_manifest.get("n_rows_requested") or len(jobs),
        "n_rows_completed": n_rows_completed,
        "request_failures": request_failures,
        "empty_output_failures": empty_output_failures,
        "json_parse_failures": json_parse_failures,
        "schema_failures": schema_failures,
        "validation_failure_count": validation_failure_count,
        "evidence_substring_failures": validation_failure_count,
        "total_skills": total_skills,
        "skills_per_job_mean": skills_per_job_mean,
        "total_education_requirements": total_education_requirements,
        "seniority_counts": seniority_counts,
        "warning_counts": warning_counts,
        "confidence_mean": confidence_mean,
        "confidence_1_rate": confidence_1_rate,
        "broad_domain_skill_rate": broad_domain_skill_rate,
        "empty_skill_rows": empty_skill_rows,
        "prompt_tokens": prior_prompt_tokens + prompt_tokens_delta,
        "completion_tokens": prior_completion_tokens + completion_tokens_delta,
        "total_tokens": prior_prompt_tokens + prior_completion_tokens + prompt_tokens_delta + completion_tokens_delta,
        "estimated_cost_usd": estimated_cost,
        "latency_seconds": (sum(latencies) / len(latencies)) if latencies else prior_manifest.get("latency_seconds"),
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_time_seconds": wall_time_seconds,
        "prompt_hash": sha256_file(PROMPT_PATH),
        "sample_hash": sha256_file(sample_path),
        "git_commit": git_commit(),
        "limitations": [
            "Weak-label evaluation only; not ground truth (see README.md).",
            "estimated_cost_usd relies on OpenRouter's optional usage.cost field; null if the provider never reported pricing.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one OpenRouter model over the frozen 50-job sample.")
    parser.add_argument("--model", type=str, required=True, help="OpenRouter model slug, e.g. deepseek/deepseek-v4-flash")
    parser.add_argument("--sample", type=str, default=str(SAMPLE_PATH), help="Path to jobs_sample_50.jsonl")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay in seconds between API requests")
    parser.add_argument("--resume", action="store_true", help="Skip job_ids already present in extractions.jsonl")
    parser.add_argument("--only-job-id", type=str, help="Run exactly one job_id from the sample")
    parser.add_argument("--max-rows", type=int, help="Debug limit after --only-job-id and --resume filtering")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY is not set in the environment.")
        sys.exit(1)

    sample_path = Path(args.sample)
    if not sample_path.exists():
        logger.error(f"Sample file not found at {sample_path}. Run build_sample.py first.")
        sys.exit(1)
    if not PROMPT_PATH.exists():
        logger.error(f"Prompt file not found at {PROMPT_PATH}")
        sys.exit(1)
    if args.max_rows is not None and args.max_rows < 0:
        logger.error("--max-rows must be non-negative.")
        sys.exit(1)

    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    jobs = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.only_job_id:
        jobs = [job for job in jobs if str(job["job_id"]) == args.only_job_id]
        if not jobs:
            logger.error(f"job_id={args.only_job_id} not found in {sample_path}")
            sys.exit(1)

    output_dir = EXPERIMENT_DIR / "outputs" / safe_model_name(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    extractions_path = output_dir / "extractions.jsonl"
    raw_responses_path = output_dir / "raw_responses.jsonl"
    failures_path = output_dir / "failures.jsonl"
    validation_failures_path = output_dir / "validation_failures.jsonl"
    manifest_path = output_dir / "manifest.json"

    existing_outputs = [path for path in (extractions_path, raw_responses_path, failures_path, validation_failures_path) if path.exists()]
    if existing_outputs and not args.resume:
        logger.error(f"Output files already exist under {output_dir}; use --resume to append without discarding raw responses.")
        sys.exit(1)

    prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    completed_job_ids = {str(row["job_id"]) for row in read_jsonl(extractions_path)}
    jobs_requested = jobs
    jobs_to_run = [job for job in jobs_requested if not (args.resume and str(job["job_id"]) in completed_job_ids)]
    if args.max_rows is not None:
        jobs_to_run = jobs_to_run[:args.max_rows]

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    latencies = []
    prompt_tokens_delta = 0
    completion_tokens_delta = 0
    cost_delta = 0.0
    cost_available = False

    with extractions_path.open("a", encoding="utf-8") as f_extractions, \
         raw_responses_path.open("a", encoding="utf-8") as f_raw, \
         failures_path.open("a", encoding="utf-8") as f_failures, \
         validation_failures_path.open("a", encoding="utf-8") as f_validation_failures:
        for i, job in enumerate(jobs_to_run):
            logger.info(f"[{i + 1}/{len(jobs_to_run)}] Calling {args.model} for job_id={job['job_id']}")
            user_prompt = build_user_prompt(job)

            start = time.monotonic()
            response = call_openrouter(api_key, args.model, system_prompt, user_prompt)
            latency = time.monotonic() - start
            latencies.append(latency)

            write_jsonl(f_raw, {"job_id": job["job_id"], "response": response})

            usage = response.get("usage", {}) if isinstance(response, dict) else {}
            prompt_tokens_delta += usage.get("prompt_tokens", 0) or 0
            completion_tokens_delta += usage.get("completion_tokens", 0) or 0
            if usage.get("cost") is not None:
                cost_available = True
                cost_delta += usage["cost"]

            if not isinstance(response, dict) or response.get("error"):
                write_jsonl(f_failures, {"job_id": job["job_id"], "failure_type": "api_error", "error": response.get("error") if isinstance(response, dict) else response})
                continue

            try:
                content = response["choices"][0]["message"]["content"]
            except Exception as exc:
                write_jsonl(f_failures, {"job_id": job["job_id"], "failure_type": "empty_output", "error": str(exc)})
                continue

            if not content:
                write_jsonl(f_failures, {"job_id": job["job_id"], "failure_type": "empty_output", "error": "empty message content"})
                continue

            try:
                apolo_extraction = json.loads(content)
            except json.JSONDecodeError as exc:
                write_jsonl(f_failures, {"job_id": job["job_id"], "failure_type": "non_json_output", "error": str(exc), "content": content})
                continue

            validation_failures = validate_extraction(apolo_extraction, job["job_card_text"])
            if validation_failures:
                kind = failure_kind(validation_failures)
                write_jsonl(f_failures, {"job_id": job["job_id"], "failure_type": kind, "failure_count": len(validation_failures)})
                for failure in validation_failures:
                    write_jsonl(f_validation_failures, {"job_id": job["job_id"], **failure})
                continue

            row = {
                "task_id": f"job_{job['job_id']}",
                "job_id": job["job_id"],
                "title": job["title"],
                "model": args.model,
                "apolo_extraction": apolo_extraction,
            }
            write_jsonl(f_extractions, row)

            if args.delay > 0:
                time.sleep(args.delay)

    finished_at = datetime.now(UTC).isoformat()
    wall_time_seconds = time.monotonic() - started
    extraction_rows = read_jsonl(extractions_path)
    failures = read_jsonl(failures_path)
    manifest = aggregate_manifest(
        args=args,
        jobs=jobs_requested,
        extraction_rows=extraction_rows,
        failures=failures,
        prior_manifest=prior_manifest,
        started_at=started_at,
        finished_at=finished_at,
        wall_time_seconds=wall_time_seconds,
        prompt_tokens_delta=prompt_tokens_delta,
        completion_tokens_delta=completion_tokens_delta,
        cost_delta=cost_delta,
        cost_available=cost_available,
        latencies=latencies,
        sample_path=sample_path,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Completed {manifest['n_rows_completed']}/{manifest['n_rows_requested']} rows for {args.model}")
    logger.info(f"Wrote {extractions_path}, {raw_responses_path}, {failures_path}, {validation_failures_path}, {manifest_path}")


if __name__ == "__main__":
    main()
