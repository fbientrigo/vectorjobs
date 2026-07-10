#!/usr/bin/env python
"""Build a multi-model comparison report from outputs/*/manifest.json.

Produces, under reports/:
    model_comparison.csv           - one row per model, all metric groups
    model_comparison.md            - the same, as grouped tables
    model_comparison_ranked.md     - models ranked by operational_candidate_score
    model_comparison_pareto.md     - Pareto front on (cost, latency) among gate-passers

Metric groups (see evaluation_protocol.md for what each group can/cannot support):
    1. Structural validity / hard gates  - a failure here means the model is not
                                            usable yet, not "scored low."
    2. Weak-label diagnostics            - descriptive only; NOT accuracy metrics.
    3. Efficiency                        - cost / latency / throughput.
    4. operational_candidate_score       - deterministic heuristic over 1-3.
                                            NOT a scientific accuracy score.

Usage:
    python experiments/apolo_job_extraction_model_eval_50/compare_models.py
"""

import csv
import json
import statistics
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"
REPORTS_DIR = EXPERIMENT_DIR / "reports"

GATE_COLUMNS = [
    "n_rows_requested", "n_rows_completed", "request_failures",
    "json_parse_failures", "schema_failures", "validation_failure_count",
    "evidence_substring_failures", "prompt_hash", "sample_hash",
    "raw_responses_preserved", "extraction_file_exists", "manifest_file_exists",
]
DIAGNOSTIC_COLUMNS = [
    "total_skills", "skills_per_job_mean", "skills_per_job_median",
    "empty_skill_rows", "broad_domain_skill_rate",
    "total_education_requirements", "seniority_unknown_rate",
    "confidence_mean", "confidence_1_rate",
]
EFFICIENCY_COLUMNS = [
    "total_tokens", "estimated_cost_usd", "wall_time_seconds", "latency_seconds",
    "cost_per_completed_row", "tokens_per_completed_row",
    "seconds_per_completed_row", "valid_rows_per_usd",
]
SCORE_COLUMNS = ["hard_gate_pass", "operational_candidate_score"]
ALL_COLUMNS = ["model"] + GATE_COLUMNS + DIAGNOSTIC_COLUMNS + EFFICIENCY_COLUMNS + SCORE_COLUMNS

COST_PER_ROW_BASELINE_USD = 0.01
SECONDS_PER_ROW_BASELINE = 2.0
BROAD_DOMAIN_RATE_THRESHOLD = 0.5
CONFIDENCE_1_RATE_THRESHOLD = 0.2

SCORE_FORMULA_DOC = """\
operational_candidate_score (0-100, heuristic, NOT scientific accuracy):

  0 if any hard-gate failure (missing rows, request/parse/schema/validation/
  evidence failures). Otherwise start at 100 and subtract:
    - 40 * empty_skill_row_rate
    - 30 * max(0, broad_domain_skill_rate - 0.5)
    - 30 * max(0, confidence_1_rate - 0.2)
    - 10 * min(1, cost_per_completed_row / 0.01)
    - 10 * min(1, seconds_per_completed_row / 2.0)
  clipped to [0, 100].

This never rewards extracting more skills. Structural failures dominate the
score; cost and latency only matter once structural validity is satisfied."""


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def hard_gate_pass(manifest: dict) -> bool:
    n_req = manifest.get("n_rows_requested") or 0
    return (
        manifest.get("n_rows_completed") == n_req
        and n_req > 0
        and (manifest.get("request_failures") or 0) == 0
        and (manifest.get("json_parse_failures") or 0) == 0
        and (manifest.get("schema_failures") or 0) == 0
        and (manifest.get("validation_failure_count") or 0) == 0
        and (manifest.get("evidence_substring_failures") or 0) == 0
    )


def operational_candidate_score(row: dict) -> float:
    if not row["hard_gate_pass"]:
        return 0.0

    n_completed = row.get("n_rows_completed") or 0
    empty_rate = (row.get("empty_skill_rows") or 0) / n_completed if n_completed else 0.0
    broad_rate = row.get("broad_domain_skill_rate") or 0.0
    conf1_rate = row.get("confidence_1_rate") or 0.0
    cost_per_row = row.get("cost_per_completed_row")
    seconds_per_row = row.get("seconds_per_completed_row")

    score = 100.0
    score -= 40 * empty_rate
    score -= 30 * max(0.0, broad_rate - BROAD_DOMAIN_RATE_THRESHOLD)
    score -= 30 * max(0.0, conf1_rate - CONFIDENCE_1_RATE_THRESHOLD)
    if cost_per_row is not None:
        score -= 10 * min(1.0, cost_per_row / COST_PER_ROW_BASELINE_USD)
    if seconds_per_row is not None:
        score -= 10 * min(1.0, seconds_per_row / SECONDS_PER_ROW_BASELINE)
    return max(0.0, min(100.0, score))


def load_row(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_dir = manifest_path.parent
    extractions_path = model_dir / "extractions.jsonl"
    raw_responses_path = model_dir / "raw_responses.jsonl"

    n_completed = manifest.get("n_rows_completed") or 0
    seniority_counts = manifest.get("seniority_counts", {}) or {}
    seniority_unknown_rate = (seniority_counts.get("unknown", 0) / n_completed) if n_completed else None

    extraction_rows = read_jsonl(extractions_path)
    skill_counts = [len(r["apolo_extraction"]["skills"]) for r in extraction_rows]
    skills_per_job_median = statistics.median(skill_counts) if skill_counts else None

    total_tokens = manifest.get("total_tokens")
    cost = manifest.get("estimated_cost_usd")
    wall_time = manifest.get("wall_time_seconds")

    row = {
        "model": manifest.get("display_name") or manifest.get("model"),
        "n_rows_requested": manifest.get("n_rows_requested"),
        "n_rows_completed": n_completed,
        "request_failures": manifest.get("request_failures", 0),
        "json_parse_failures": manifest.get("json_parse_failures", 0),
        "schema_failures": manifest.get("schema_failures", 0),
        "validation_failure_count": manifest.get("validation_failure_count", 0),
        "evidence_substring_failures": manifest.get(
            "evidence_substring_failures", manifest.get("validation_failure_count", 0)
        ),
        "prompt_hash": manifest.get("prompt_hash"),
        "sample_hash": manifest.get("sample_hash"),
        "raw_responses_preserved": raw_responses_path.exists() and raw_responses_path.stat().st_size > 0,
        "extraction_file_exists": extractions_path.exists(),
        "manifest_file_exists": True,
        "total_skills": manifest.get("total_skills"),
        "skills_per_job_mean": manifest.get("skills_per_job_mean"),
        "skills_per_job_median": skills_per_job_median,
        "empty_skill_rows": manifest.get("empty_skill_rows"),
        "broad_domain_skill_rate": manifest.get("broad_domain_skill_rate"),
        "total_education_requirements": manifest.get("total_education_requirements"),
        "seniority_unknown_rate": seniority_unknown_rate,
        "confidence_mean": manifest.get("confidence_mean"),
        "confidence_1_rate": manifest.get("confidence_1_rate"),
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost,
        "wall_time_seconds": wall_time,
        "latency_seconds": manifest.get("latency_seconds"),
        "cost_per_completed_row": (cost / n_completed) if (cost is not None and n_completed) else None,
        "tokens_per_completed_row": (total_tokens / n_completed) if (total_tokens is not None and n_completed) else None,
        "seconds_per_completed_row": (wall_time / n_completed) if (wall_time is not None and n_completed) else None,
        "valid_rows_per_usd": (n_completed / cost) if cost else None,
    }
    row["hard_gate_pass"] = hard_gate_pass(manifest)
    row["operational_candidate_score"] = operational_candidate_score(row)
    return row


def load_rows() -> list[dict]:
    return [load_row(p) for p in sorted(OUTPUTS_DIR.glob("*/manifest.json"))]


def fmt(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


def write_csv(rows: list[dict]) -> Path:
    path = REPORTS_DIR / "model_comparison.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _table(rows: list[dict], columns: list[str]) -> list[str]:
    lines = ["| model | " + " | ".join(columns) + " |", "| --- | " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        cells = [fmt(row[c]) for c in columns]
        lines.append(f"| {row['model']} | " + " | ".join(cells) + " |")
    return lines


def write_md(rows: list[dict]) -> Path:
    path = REPORTS_DIR / "model_comparison.md"
    lines = [
        "# Model Comparison — apolo_job_extraction_model_eval_50",
        "",
        "These are **weak-label diagnostics and operational metrics**, not a",
        "SOTA-style accuracy evaluation. No human gold set has been applied yet",
        "(see evaluation_protocol.md and gold/annotation_guidelines.md).",
        "",
        "## 1. Structural validity / hard gates",
        "",
        "A failure here means the model is not usable yet — treat it as",
        "failing, not as a low score.",
        "",
    ]
    lines += _table(rows, GATE_COLUMNS)
    lines += [
        "",
        "## 2. Weak-label diagnostics (descriptive only, not quality metrics)",
        "",
        "More skills, higher confidence, or a lower unknown-rate are not",
        "automatically better. `broad_domain_skill_rate` is a risk signal, not a",
        "quality signal — more \"domain\" skills usually means vaguer labels",
        "(see evaluation_protocol.md).",
        "",
    ]
    lines += _table(rows, DIAGNOSTIC_COLUMNS)
    lines += ["", "## 3. Efficiency", ""]
    lines += _table(rows, EFFICIENCY_COLUMNS)
    lines += ["", "## 4. operational_candidate_score (heuristic, NOT scientific accuracy)", "", SCORE_FORMULA_DOC, ""]
    lines += _table(rows, SCORE_COLUMNS)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_ranked_md(rows: list[dict]) -> Path:
    path = REPORTS_DIR / "model_comparison_ranked.md"
    ranked = sorted(rows, key=lambda r: r["operational_candidate_score"], reverse=True)
    lines = [
        "# Model Comparison — Ranked by operational_candidate_score",
        "",
        "This ranks operational readiness (structural validity, diagnostics",
        "risk, cost, latency). It is NOT an accuracy ranking — see",
        "evaluation_protocol.md.",
        "",
    ]
    lines += _table(
        ranked,
        [
            "hard_gate_pass", "operational_candidate_score", "estimated_cost_usd",
            "seconds_per_completed_row", "broad_domain_skill_rate", "confidence_1_rate",
        ],
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def is_pareto_optimal(row: dict, rows: list[dict]) -> bool:
    cost = row["cost_per_completed_row"]
    seconds = row["seconds_per_completed_row"]
    if cost is None or seconds is None:
        return True
    for other in rows:
        if other is row:
            continue
        other_cost, other_seconds = other["cost_per_completed_row"], other["seconds_per_completed_row"]
        if other_cost is None or other_seconds is None:
            continue
        dominates = other_cost <= cost and other_seconds <= seconds and (other_cost < cost or other_seconds < seconds)
        if dominates:
            return False
    return True


def write_pareto_md(rows: list[dict]) -> Path | None:
    gate_passers = [r for r in rows if r["hard_gate_pass"]]
    if len(gate_passers) < 2:
        return None
    path = REPORTS_DIR / "model_comparison_pareto.md"
    pareto_rows = [r for r in gate_passers if is_pareto_optimal(r, gate_passers)]
    lines = [
        "# Model Comparison — Pareto Front (cost vs latency, gate-passers only)",
        "",
        "A model is Pareto-optimal here if no other gate-passing model has both",
        "lower-or-equal cost_per_completed_row and lower-or-equal",
        "seconds_per_completed_row. This is a tradeoff view, not a ranking.",
        "",
    ]
    lines += _table(pareto_rows, ["cost_per_completed_row", "seconds_per_completed_row", "operational_candidate_score"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    if not OUTPUTS_DIR.exists() or not any(OUTPUTS_DIR.glob("*/manifest.json")):
        print(f"No manifests found under {OUTPUTS_DIR}. Run run_model_eval.py for at least one model first.")
        return

    rows = load_rows()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Wrote {write_csv(rows)}")
    print(f"Wrote {write_md(rows)}")
    print(f"Wrote {write_ranked_md(rows)}")
    pareto_path = write_pareto_md(rows)
    if pareto_path:
        print(f"Wrote {pareto_path}")
    else:
        print("Skipped model_comparison_pareto.md (need >= 2 gate-passing models).")


if __name__ == "__main__":
    main()
