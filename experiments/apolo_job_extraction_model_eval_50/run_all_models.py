#!/usr/bin/env python
"""Plan and run run_model_eval.py across configs/models.json.

This is an orchestration layer only: it never calls OpenRouter itself. Each
selected model is executed as a subprocess: `run_model_eval.py --model <model>`.

Usage:
    python experiments/apolo_job_extraction_model_eval_50/run_all_models.py --dry-run
    python experiments/apolo_job_extraction_model_eval_50/run_all_models.py \
        --resume --skip-existing-valid --budget-usd 0.50
    python experiments/apolo_job_extraction_model_eval_50/run_all_models.py \
        --models deepseek/deepseek-v4-pro --resume
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
MODELS_CONFIG_PATH = EXPERIMENT_DIR / "configs" / "models.json"
SAMPLE_MANIFEST_PATH = EXPERIMENT_DIR / "input" / "sample_manifest.json"
OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
RUN_MODEL_EVAL = EXPERIMENT_DIR / "run_model_eval.py"


def safe_model_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def sample_size() -> int:
    if SAMPLE_MANIFEST_PATH.exists():
        return json.loads(SAMPLE_MANIFEST_PATH.read_text(encoding="utf-8")).get("sample_size", 50)
    return 50


def load_manifest(model: str) -> dict | None:
    path = OUTPUTS_DIR / safe_model_name(model) / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def is_valid_complete(manifest: dict) -> bool:
    n_req = manifest.get("n_rows_requested") or 0
    return (
        manifest.get("n_rows_completed") == n_req
        and n_req > 0
        and (manifest.get("validation_failure_count") or 0) == 0
        and (manifest.get("schema_failures") or 0) == 0
        and (manifest.get("json_parse_failures") or 0) == 0
        and (manifest.get("request_failures") or 0) == 0
    )


def estimate_cost_per_row() -> float | None:
    """Average cost/completed-row across any existing manifest with cost data.

    ponytail: this is a rough cross-model proxy (not a per-model price table),
    used only to gate an approximate budget before a model has ever run.
    """
    samples = []
    for manifest_path in OUTPUTS_DIR.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        n = manifest.get("n_rows_completed") or 0
        cost = manifest.get("estimated_cost_usd")
        if n and cost is not None:
            samples.append(cost / n)
    if not samples:
        return None
    return sum(samples) / len(samples)


def build_plan(candidates: list[str], args: argparse.Namespace) -> list[dict]:
    cost_per_row = estimate_cost_per_row()
    accumulated_cost = sum(
        (load_manifest(m) or {}).get("estimated_cost_usd") or 0.0
        for m in json.loads(MODELS_CONFIG_PATH.read_text(encoding="utf-8"))
    )

    plan = []
    projected_cost = accumulated_cost
    n_to_run = 0
    for model in candidates:
        manifest = load_manifest(model)
        entry = {"model": model, "action": "run", "reason": "", "estimated_cost_usd": None}

        if args.skip_existing_valid and manifest and is_valid_complete(manifest):
            entry["action"] = "skip"
            entry["reason"] = "existing valid manifest (all rows completed, zero failures)"
            plan.append(entry)
            continue

        if args.max_models is not None and n_to_run >= args.max_models:
            entry["action"] = "skip"
            entry["reason"] = f"--max-models {args.max_models} reached"
            plan.append(entry)
            continue

        rows_remaining = sample_size()
        if manifest:
            rows_remaining = (manifest.get("n_rows_requested") or sample_size()) - (manifest.get("n_rows_completed") or 0)
        model_cost_estimate = (cost_per_row * rows_remaining) if cost_per_row is not None else None

        if args.budget_usd is not None and model_cost_estimate is not None:
            if projected_cost + model_cost_estimate > args.budget_usd:
                entry["action"] = "stop_budget"
                entry["reason"] = (
                    f"projected accumulated cost {projected_cost + model_cost_estimate:.4f} USD "
                    f"would exceed --budget-usd {args.budget_usd:.4f}"
                )
                plan.append(entry)
                break

        entry["estimated_cost_usd"] = model_cost_estimate
        plan.append(entry)
        projected_cost += model_cost_estimate or 0.0
        n_to_run += 1

    return plan


def write_plan_report(candidates: list[str], cost_per_row: float | None, plan: list[dict]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = REPORTS_DIR / "run_all_models_plan.json"
    plan_path.write_text(
        json.dumps(
            {"candidates": candidates, "cost_per_row_estimate": cost_per_row, "plan": plan},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return plan_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan and run multiple OpenRouter models over the frozen sample.")
    parser.add_argument("--models", nargs="+", help="Subset of model slugs (default: all in configs/models.json)")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; never invoke run_model_eval.py")
    parser.add_argument("--resume", action="store_true", help="Pass --resume through to run_model_eval.py")
    parser.add_argument("--skip-existing-valid", action="store_true", help="Skip models with a complete, failure-free manifest")
    parser.add_argument("--max-models", type=int, help="Run at most N models this invocation")
    parser.add_argument("--budget-usd", type=float, help="Stop before a model would push projected accumulated cost over this budget")
    args = parser.parse_args()

    all_models = json.loads(MODELS_CONFIG_PATH.read_text(encoding="utf-8"))
    candidates = args.models if args.models else all_models
    unknown = [m for m in candidates if m not in all_models]
    if unknown:
        print(f"Warning: {unknown} not present in configs/models.json")

    plan = build_plan(candidates, args)
    plan_path = write_plan_report(candidates, estimate_cost_per_row(), plan)
    print(f"Wrote plan to {plan_path}")
    for entry in plan:
        suffix = f" ({entry['reason']})" if entry["reason"] else ""
        print(f"  {entry['model']}: {entry['action']}{suffix}")

    if args.dry_run:
        print("\n--dry-run: no models were executed.")
        print(f"Next: python {EXPERIMENT_DIR / 'compare_models.py'}")
        return

    for entry in plan:
        if entry["action"] != "run":
            continue
        cmd = [sys.executable, str(RUN_MODEL_EVAL), "--model", entry["model"]]
        if args.resume:
            cmd.append("--resume")
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"run_model_eval.py failed for {entry['model']} (exit {result.returncode}); stopping.")
            sys.exit(result.returncode)

    print("\nAll planned models finished.")
    print(f"Next: python {EXPERIMENT_DIR / 'compare_models.py'}")


if __name__ == "__main__":
    main()
