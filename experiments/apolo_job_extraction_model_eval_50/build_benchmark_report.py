#!/usr/bin/env python
"""Build the normalized benchmark report from existing run outputs.

Reads outputs/*/manifest.json (never calls a model), and read-only imports
the historical Gemini-era baseline from data/silver/agent_labeling/.

Writes, under reports/:
    benchmark_table.csv / .json       - one row per model, all four metric
                                         families, flattened with prefixes
                                         reliability_/structural_/efficiency_/
                                         quality_.
    reliability_table.csv             - family 1 only (full 50-job target
                                         denominator, cohort A).
    historical_baseline_comparison.csv - the imported Gemini-era baseline's
                                         normalized aggregate metrics.
    benchmark_methodology.md          - metric/denominator/cohort docs.

Usage:
    python experiments/apolo_job_extraction_model_eval_50/build_benchmark_report.py
"""

import csv
import json
from pathlib import Path

from benchmarking import cohorts, historical_baselines, loaders, metrics, reliability

EXPERIMENT_DIR = Path(__file__).parent
REPORTS_DIR = EXPERIMENT_DIR / "reports"


def build_model_row(model: str, manifest: dict, gold_row: dict | None, n_jobs_target: int) -> dict:
    reliability_metrics = reliability.reliability_metrics(manifest, n_jobs_target)
    structural = metrics.structural_correctness(manifest)
    efficiency = metrics.efficiency_metrics(manifest)
    quality = metrics.semantic_quality(model, manifest, gold_row, n_jobs_target)

    row = {"model": model}
    row.update({f"reliability_{k}": v for k, v in reliability_metrics.items()})
    row.update({f"structural_{k}": v for k, v in structural.items()})
    row.update({f"efficiency_{k}": v for k, v in efficiency.items()})
    row.update({f"quality_{k}": v for k, v in quality.items()})
    return row


def build_all_model_rows() -> list[dict]:
    frozen_job_ids = loaders.load_frozen_job_ids()
    n_jobs_target = len(frozen_job_ids)
    manifests = loaders.load_all_run_manifests()
    gold_by_model = loaders.load_gold_vs_model() or {}
    return [
        build_model_row(model, manifest, gold_by_model.get(model), n_jobs_target)
        for model, manifest in manifests.items()
    ]


EFFICIENCY_FIELDS = [
    "timing_basis", "cost_basis", "total_input_tokens", "total_output_tokens",
    "total_tokens", "observed_cost_usd", "estimated_cost_usd",
    "cost_for_reporting_usd", "cost_per_eventually_valid_job",
    "estimated_inference_time_seconds", "seconds_per_valid_job",
    "completed_jobs_per_minute", "latency_safety_factor",
]


def build_efficiency_rows(rows: list[dict]) -> list[dict]:
    """Flat efficiency view: measured and estimated runs together, basis explicit."""
    out = []
    for r in rows:
        out.append({"model": r["model"], **{f: r.get(f"efficiency_{f}") for f in EFFICIENCY_FIELDS}})
    return out


def write_efficiency_reports(rows: list[dict]) -> None:
    eff_rows = build_efficiency_rows(rows)
    write_csv(eff_rows, REPORTS_DIR / "model_efficiency_comparison.csv")
    lines = [
        "# Model Efficiency Comparison — timing & cost",
        "",
        "Timing/cost basis is explicit per model. `measured` = provider/API",
        "telemetry captured during execution. `openrouter_equivalent_estimate`",
        "= reconstructed from OpenRouter latency/throughput with a conservative",
        "1.20 (+20%) time overhead (cost is list-price, NOT multiplied by 1.20).",
        "Subscription/manual runs keep `observed_cost_usd` null and report an",
        "`estimated_cost_usd` (cost_basis=openrouter_list_price_estimate).",
        "",
    ]
    lines += _md_table(eff_rows, ["model"] + EFFICIENCY_FIELDS)
    lines.append("")
    (REPORTS_DIR / "model_efficiency_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORTS_DIR / 'model_efficiency_comparison.csv'}")
    print(f"Wrote {REPORTS_DIR / 'model_efficiency_comparison.md'}")


def write_historical_efficiency_report(record: dict) -> None:
    fields = [
        "display_name", "n_rows", "frozen_50_overlap", "directly_comparable",
        "quality_comparable_to_frozen_50", "total_input_tokens", "total_output_tokens",
        "total_tokens", "token_count_is_measured", "cost_basis", "observed_cost_usd",
        "estimated_cost_usd", "estimated_cost_per_completed_job_usd",
        "estimated_cost_per_1000_jobs_usd", "timing_basis",
        "openrouter_performance_provider", "openrouter_latency_seconds",
        "openrouter_throughput_tokens_per_second", "latency_safety_factor",
        "base_estimated_runtime_seconds", "estimated_inference_time_seconds",
        "estimated_seconds_per_completed_job", "estimated_completed_jobs_per_minute",
    ]
    row = {f: record.get(f) for f in fields}
    write_csv([row], REPORTS_DIR / "historical_model_efficiency.csv")
    lines = [
        f"# Historical Model Efficiency — {record.get('display_name')}",
        "",
        "Efficiency/cost of the recovered historical Gemini dataset, aggregated",
        f"over its ACTUAL recovered row count ({record.get('n_rows')} rows), NOT an",
        "assumed 50. Timing is an `openrouter_equivalent_estimate` with a 1.20",
        "(+20%) overhead; cost is OpenRouter list-price (not multiplied by 1.20).",
        "",
        f"> {record.get('comparability_notes', '')}",
        "",
        f"> token method: {record.get('token_count_method', '')}",
        "",
    ]
    lines += _md_table([row], fields)
    lines.append("")
    (REPORTS_DIR / "historical_model_efficiency.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORTS_DIR / 'historical_model_efficiency.csv'}")
    print(f"Wrote {REPORTS_DIR / 'historical_model_efficiency.md'}")


def _md_table(rows: list[dict], columns: list[str]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for r in rows:
        body.append("| " + " | ".join(str(fmt(r.get(c))) for c in columns) + " |")
    return [header, sep] + body


def build_common_semantic_cohort(rows: list[dict]) -> dict:
    model_job_ids = {
        row["model"]: {str(r["job_id"]) for r in loaders.load_extractions(row["model"])}
        for row in rows
    }
    common, info = cohorts.common_semantic_cohort(model_job_ids)
    info["job_ids"] = sorted(common)
    return info


def fmt(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, list):
        return ";".join(value)
    return "" if value is None else value


def write_csv(rows: list[dict], path: Path) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(v) for k, v in row.items()})


def write_json(rows: list[dict], path: Path) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def write_reliability_table(rows: list[dict], path: Path) -> None:
    reliability_rows = [
        {"model": r["model"], **{k[len("reliability_"):]: v for k, v in r.items() if k.startswith("reliability_")}}
        for r in rows
    ]
    write_csv(reliability_rows, path)


METHODOLOGY_TEMPLATE = """# Benchmark Methodology — apolo_job_extraction_model_eval_50

Generated by build_benchmark_report.py. This document defines every metric
family, its denominator, and which comparisons are controlled vs. historical.

## Metric families (kept scientifically separate)

1. **Operational reliability** — always evaluated over all
   `n_jobs_target` = {n_jobs_target} frozen jobs (cohort A). Never restricted
   to a successful-job intersection. `run_model_eval.py` retries transport
   failures internally but does not persist per-attempt records, so
   first-pass vs. eventual cannot be distinguished for the runs in this
   report — `single_run_valid_count` / `single_run_valid_rate` are reported
   instead and documented as a single-run completion measure, not proven
   first-attempt success. `reliability.py` still exposes first-pass/eventual
   fields (null here) so a future resume/recovery segment can populate them
   without changing this module's shape or overwriting the initial segment.

2. **Structural correctness** — every rate's denominator is `n_attempted`
   (== `n_rows_requested`), except `empty_skill_row_rate` (denominator
   `n_completed`). Transport, JSON-parse, schema, and evidence-substring
   failures are counted separately, never collapsed into one failure count.

3. **Semantic extraction quality** — hierarchy: human-gold metrics (from
   `evaluate_against_gold.py`'s `model_vs_gold.csv`) when a full reviewed
   gold set exists; reviewed-subset metrics when partial gold exists;
   otherwise `operational_quality_proxy`, an explicitly-labeled [0, 1]
   heuristic from structural/diagnostic signals only (no cost or speed
   mixed in). No gold review exists yet for this experiment (see
   `gold/human_gold_template.jsonl` — all rows `review_status: pending`), so
   every model in this report uses the proxy. It is NOT an accuracy score;
   do not cite it as such.

4. **Efficiency** — cost/latency fields are `null`, never `0`, when neither
   measured nor a defensible estimate exists. Each has an explicit basis:
   measured API runs report `observed_cost_usd` (from OpenRouter `usage.cost`)
   and measured per-job latency (`timing_basis=measured`). Subscription/manual
   and recovered-historical runs report `observed_cost_usd=null` plus an
   `estimated_cost_usd` at OpenRouter list price
   (`cost_basis=openrouter_list_price_estimate`) and an
   `estimated_inference_time_seconds` reconstructed from OpenRouter
   latency/throughput with a conservative 1.20 (+20%) time overhead
   (`timing_basis=openrouter_equivalent_estimate`). The 1.20 factor is applied
   to time only, never to cost. Estimates never populate
   `provider_measured_latency_seconds`. See `model_efficiency_comparison.md`
   and `historical_model_efficiency.md`.

## Cohorts

- **A. Full-target reliability cohort** — all {n_jobs_target} frozen job_ids.
  Used for every reliability metric. Unsuccessful rows remain meaningful
  observations; this cohort is never reduced to a successful intersection.
- **B. Common semantic cohort** — intersection of job_ids each compared
  model successfully extracted. {common_semantic_note}
- **C. Gold-reviewed cohort** — exact job_ids with `review_status !=
  "pending"` in a reviewed gold file. Currently empty (template only, no
  reviewer has completed annotation yet).

## Run-status terminology

A model finishing its planned run with fewer than {n_jobs_target} valid
rows is `process_finished_with_invalid_or_missing_rows`, not "failed to
finish" — both facts (the process completed, and reliability is below
100%) are preserved. See `reliability.run_status()`.

## Historical Gemini-era baseline

{historical_summary}

Full provenance investigation is in `benchmarking/historical_baselines.py`'s
module docstring. Because provenance is unresolved, the baseline is
labeled a **historical reference**, not a controlled benchmark competitor,
and is excluded from every cost/latency/retry/throughput figure — none of
those metrics exist for it. It may appear in extraction-diagnostics or
semantic-quality figures only where the required metric is actually
supported by the {frozen_50_overlap}-row overlap with the frozen 50.

## Missing-metric handling

Every function in `benchmarking/` returns `None` (not `0`, not an omitted
key) for a metric it cannot compute from available data. Plotting functions
in `benchmarking/plots.py` skip a series (with a returned skip note) rather
than rendering a fabricated zero.
"""


def build_methodology(rows: list[dict], baseline: dict, common_semantic: dict) -> str:
    n_jobs_target = rows[0]["reliability_n_jobs_target"] if rows else 50
    common_note = (
        f"{common_semantic['n_common_jobs']} job_ids common to {common_semantic['models']}."
        if common_semantic.get("models") else "No models available to compute an intersection yet."
    )
    historical_summary = (
        f"`{baseline['display_name']}` ({baseline['baseline_id']}) — "
        f"provenance_status={baseline['provenance_status']}, "
        f"model_configuration_status={baseline['model_configuration_status']}, "
        f"reasoning_mode={baseline['reasoning_mode']}, "
        f"generation_method={baseline['generation_method']}. "
        f"{baseline['frozen_50_overlap']}/{n_jobs_target} frozen job_ids overlap "
        f"with the {baseline['n_source_rows']}-row historical dataset."
    )
    return METHODOLOGY_TEMPLATE.format(
        n_jobs_target=n_jobs_target,
        common_semantic_note=common_note,
        historical_summary=historical_summary,
        frozen_50_overlap=baseline["frozen_50_overlap"],
    )


def main() -> None:
    rows = build_all_model_rows()
    if not rows:
        print(f"No manifests found under {loaders.OUTPUTS_DIR}. Run run_model_eval.py for at least one model first.")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(rows, REPORTS_DIR / "benchmark_table.csv")
    write_json(rows, REPORTS_DIR / "benchmark_table.json")
    print(f"Wrote {REPORTS_DIR / 'benchmark_table.csv'}")
    print(f"Wrote {REPORTS_DIR / 'benchmark_table.json'}")

    write_reliability_table(rows, REPORTS_DIR / "reliability_table.csv")
    print(f"Wrote {REPORTS_DIR / 'reliability_table.csv'}")

    write_efficiency_reports(rows)

    common_semantic = build_common_semantic_cohort(rows)

    frozen_job_ids = loaders.load_frozen_job_ids()
    baseline = historical_baselines.build_baseline_report(frozen_job_ids)
    write_csv([baseline], REPORTS_DIR / "historical_baseline_comparison.csv")
    print(f"Wrote {REPORTS_DIR / 'historical_baseline_comparison.csv'}")

    historical_efficiency = historical_baselines.historical_efficiency_report(frozen_job_ids)
    write_historical_efficiency_report(historical_efficiency)

    methodology_path = REPORTS_DIR / "benchmark_methodology.md"
    methodology_path.write_text(build_methodology(rows, baseline, common_semantic), encoding="utf-8")
    print(f"Wrote {methodology_path}")


if __name__ == "__main__":
    main()
