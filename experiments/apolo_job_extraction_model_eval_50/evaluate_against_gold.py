#!/usr/bin/env python
"""Score model extractions.jsonl against reviewed human gold labels.

Produces STRICT, deterministic baseline metrics only (exact / normalized /
strict-type string matching, plus an explicitly-optional partial token-overlap
score). No embeddings, no LLM judge. See evaluation_protocol.md for what these
metrics can and cannot support.

Exits cleanly (no traceback) with a message if no reviewed gold exists yet.

Usage:
    python experiments/apolo_job_extraction_model_eval_50/evaluate_against_gold.py
    python experiments/apolo_job_extraction_model_eval_50/evaluate_against_gold.py --model deepseek/deepseek-v4-flash
    python experiments/apolo_job_extraction_model_eval_50/evaluate_against_gold.py --gold gold/human_gold_50.jsonl
"""

import argparse
import csv
import json
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
SAMPLE_PATH = EXPERIMENT_DIR / "input" / "jobs_sample_50.jsonl"
DEFAULT_GOLD_PATH = EXPERIMENT_DIR / "gold" / "human_gold_50.jsonl"

FIELDS = [
    "model", "n_gold_reviewed", "n_jobs_evaluated",
    "skill_name_exact_precision", "skill_name_exact_recall", "skill_name_exact_f1",
    "skill_name_normalized_precision", "skill_name_normalized_recall", "skill_name_normalized_f1",
    "skill_strict_type_precision", "skill_strict_type_recall", "skill_strict_type_f1",
    "skill_partial_token_overlap_mean",
    "education_exact_precision", "education_exact_recall", "education_exact_f1",
    "seniority_accuracy", "seniority_unknown_rate_pred", "seniority_unknown_rate_gold",
    "evidence_substring_failure_count", "evidence_substring_failure_rate",
]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def load_gold(gold_path: Path) -> list[dict] | None:
    if not gold_path.exists():
        print(f"No gold file at {gold_path}. Nothing to evaluate against yet.")
        print(
            "Create/copy gold/human_gold_template.jsonl to that path and have a "
            "reviewer complete it first (see gold/annotation_guidelines.md)."
        )
        return None
    rows = read_jsonl(gold_path)
    reviewed = [r for r in rows if r.get("review_status") != "pending"]
    if not reviewed:
        print(f"{gold_path} exists but every row is still review_status=pending.")
        print("No reviewed gold labels yet; complete human annotation before evaluating.")
        return None
    return rows


def prf1(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def skill_key(skill: dict, mode: str) -> str:
    if mode == "exact":
        return norm(skill.get("name", ""))
    if mode == "normalized":
        return norm(skill.get("normalized_name") or skill.get("name", ""))
    if mode == "strict_type":
        return f"{norm(skill.get('normalized_name') or skill.get('name', ''))}|{skill.get('category')}"
    raise ValueError(mode)


def token_overlap(a: str, b: str) -> float:
    ta, tb = set(norm(a).split()), set(norm(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def evaluate(model: str, gold_rows: list[dict]) -> dict:
    output_dir = OUTPUTS_DIR / model.replace("/", "__").replace(":", "_")
    extractions = {str(r["job_id"]): r for r in read_jsonl(output_dir / "extractions.jsonl")}
    sample = {str(r["job_id"]): r for r in read_jsonl(SAMPLE_PATH)}
    reviewed_gold = {str(r["job_id"]): r for r in gold_rows if r.get("review_status") != "pending"}

    skill_counts = {mode: {"tp": 0, "fp": 0, "fn": 0} for mode in ("exact", "normalized", "strict_type")}
    edu_counts = {"tp": 0, "fp": 0, "fn": 0}
    seniority_correct = 0
    seniority_total = 0
    pred_unknown = 0
    gold_unknown = 0
    evidence_failures = 0
    n_jobs_evaluated = 0
    partial_overlap_scores = []

    for job_id, gold in reviewed_gold.items():
        if job_id not in extractions:
            continue
        n_jobs_evaluated += 1
        extraction = extractions[job_id]["apolo_extraction"]
        job_card_text = sample.get(job_id, {}).get("job_card_text", "")

        pred_skills = extraction.get("skills", [])
        for skill in pred_skills:
            evidence = skill.get("evidence", "")
            if evidence and evidence not in job_card_text:
                evidence_failures += 1

        gold_skills = gold.get("gold_skills", [])
        for mode in skill_counts:
            pred_set = {skill_key(s, mode) for s in pred_skills}
            gold_set = {skill_key(s, mode) for s in gold_skills}
            skill_counts[mode]["tp"] += len(pred_set & gold_set)
            skill_counts[mode]["fp"] += len(pred_set - gold_set)
            skill_counts[mode]["fn"] += len(gold_set - pred_set)

        for p in pred_skills:
            best = max((token_overlap(p.get("name", ""), g.get("name", "")) for g in gold_skills), default=0.0)
            partial_overlap_scores.append(best)

        pred_edu_set = {
            (norm(e.get("degree", "")), bool(e.get("required"))) for e in extraction.get("education_requirements", [])
        }
        gold_edu_set = {
            (norm(e.get("degree", "")), bool(e.get("required")))
            for e in gold.get("gold_education_requirements", [])
        }
        edu_counts["tp"] += len(pred_edu_set & gold_edu_set)
        edu_counts["fp"] += len(pred_edu_set - gold_edu_set)
        edu_counts["fn"] += len(gold_edu_set - pred_edu_set)

        pred_seniority = extraction.get("seniority", "unknown")
        gold_seniority = gold.get("seniority_gold", "unknown")
        seniority_total += 1
        if pred_seniority == gold_seniority:
            seniority_correct += 1
        if pred_seniority == "unknown":
            pred_unknown += 1
        if gold_seniority == "unknown":
            gold_unknown += 1

    p_exact, r_exact, f1_exact = prf1(**skill_counts["exact"])
    p_norm, r_norm, f1_norm = prf1(**skill_counts["normalized"])
    p_strict, r_strict, f1_strict = prf1(**skill_counts["strict_type"])
    p_edu, r_edu, f1_edu = prf1(**edu_counts)

    return {
        "model": model,
        "n_gold_reviewed": len(reviewed_gold),
        "n_jobs_evaluated": n_jobs_evaluated,
        "skill_name_exact_precision": p_exact,
        "skill_name_exact_recall": r_exact,
        "skill_name_exact_f1": f1_exact,
        "skill_name_normalized_precision": p_norm,
        "skill_name_normalized_recall": r_norm,
        "skill_name_normalized_f1": f1_norm,
        "skill_strict_type_precision": p_strict,
        "skill_strict_type_recall": r_strict,
        "skill_strict_type_f1": f1_strict,
        "skill_partial_token_overlap_mean": (
            sum(partial_overlap_scores) / len(partial_overlap_scores) if partial_overlap_scores else None
        ),
        "education_exact_precision": p_edu,
        "education_exact_recall": r_edu,
        "education_exact_f1": f1_edu,
        "seniority_accuracy": (seniority_correct / seniority_total) if seniority_total else None,
        "seniority_unknown_rate_pred": (pred_unknown / seniority_total) if seniority_total else None,
        "seniority_unknown_rate_gold": (gold_unknown / seniority_total) if seniority_total else None,
        "evidence_substring_failure_count": evidence_failures,
        "evidence_substring_failure_rate": (evidence_failures / n_jobs_evaluated) if n_jobs_evaluated else None,
    }


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


def write_reports(results: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "model_vs_gold.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Wrote {csv_path}")

    md_path = REPORTS_DIR / "model_vs_gold.md"
    lines = [
        "# Model vs. Human Gold — apolo_job_extraction_model_eval_50",
        "",
        "Exact/normalized/strict-type matching is deterministic string",
        "comparison, not a taxonomy-aware match. `skill_partial_token_overlap_mean`",
        "is an optional, flexible signal — not the primary metric.",
        "",
        "| " + " | ".join(FIELDS) + " |",
        "| " + " | ".join("---" for _ in FIELDS) + " |",
    ]
    for row in results:
        lines.append("| " + " | ".join(fmt(row[f]) for f in FIELDS) + " |")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model extractions against reviewed human gold.")
    parser.add_argument("--model", type=str, help="Single model slug to evaluate (default: all models with a manifest)")
    parser.add_argument("--gold", type=str, default=str(DEFAULT_GOLD_PATH), help="Path to reviewed gold jsonl")
    args = parser.parse_args()

    gold_rows = load_gold(Path(args.gold))
    if gold_rows is None:
        sys.exit(0)

    if args.model:
        models = [args.model]
    else:
        models = sorted(
            json.loads(p.read_text(encoding="utf-8"))["model"] for p in OUTPUTS_DIR.glob("*/manifest.json")
        )
    if not models:
        print(f"No manifests found under {OUTPUTS_DIR}. Run run_model_eval.py for at least one model first.")
        sys.exit(0)

    results = [evaluate(model, gold_rows) for model in models]
    write_reports(results)


if __name__ == "__main__":
    main()
