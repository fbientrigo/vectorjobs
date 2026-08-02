#!/usr/bin/env python
"""One-shot, deterministic, network-free updater that stamps
OpenRouter-equivalent cost + timing estimates onto the subscription/manual
run manifests (GPT-5.5 High, Claude Sonnet 5).

Reruns are idempotent: it recomputes from the manifest's existing recovered
token counts and the static snapshot in benchmarking/api_equivalent.py. It
never touches extractions, raw responses, tokens, or quality metrics.

    python apply_api_equivalent_estimates.py
"""

import json
from pathlib import Path

from benchmarking import api_equivalent

EXPERIMENT_DIR = Path(__file__).parent

# manifest dir -> api_equivalent model key
SUBSCRIPTION_RUNS = {
    "openai__gpt-5.5__high": "gpt-5.5",
    "anthropic__claude-sonnet-5": "claude-sonnet-5",
}

# Legacy fields derived from local loop wall-time — impossible/misleading as
# provider speed. Removed so no wall-time-derived speed survives.
STALE_KEYS = ("latency_seconds", "latency_estimation_factor")


def update_manifest(slug: str, model_key: str) -> None:
    path = EXPERIMENT_DIR / "outputs" / slug / "manifest.json"
    m = json.loads(path.read_text(encoding="utf-8"))

    ti = m["total_input_tokens"]
    to = m["total_output_tokens"]
    n_req = m.get("n_rows_requested") or m.get("n_rows_completed")
    n_att = m.get("n_rows_attempted") or n_req
    n_comp = m.get("n_rows_completed")

    m["token_count_method"] = (
        "input: deterministic reconstruction of the logical request "
        "(exact extraction prompt + exact job payload), utf8_bytes/4; "
        "output: exact stored visible JSON response, utf8_bytes/4"
    )
    m["token_count_is_measured"] = False
    m["token_count_limitations"] = (
        "No provider usage was captured (no API call). Token counts are the "
        "deterministic utf8_bytes/4 fallback, not measured provider usage."
    )

    for k in STALE_KEYS:
        m.pop(k, None)

    estimate = api_equivalent.full_estimate(ti, to, model_key, n_req, n_att, n_comp)
    m.update(estimate)

    # Overwrite legacy per-job speed fields with the honest estimate so no
    # impossible wall-time-derived speed remains in the artifact.
    m["seconds_per_requested_job"] = estimate["estimated_seconds_per_attempted_job"]
    m["seconds_per_completed_job"] = estimate["estimated_seconds_per_completed_job"]
    m["completed_jobs_per_minute"] = estimate["estimated_completed_jobs_per_minute"]

    m["limitations"] = [
        lim for lim in m.get("limitations", [])
        if "wall_time_seconds * 0.80" not in lim and "Comparable latency uses" not in lim
    ]
    for extra in (
        "wall_time_seconds is the local in-session loop time, not inference "
        "latency; speed/cost here are OpenRouter-equivalent estimates "
        "(timing_basis=openrouter_equivalent_estimate, +20% overhead).",
        "observed_cost_usd is null: this run had no billed API call; "
        "estimated_cost_usd is the OpenRouter list-price equivalent.",
    ):
        if extra not in m["limitations"]:
            m["limitations"].append(extra)

    path.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated {path.relative_to(EXPERIMENT_DIR)}: "
          f"est_cost=${estimate['estimated_cost_usd']:.4f}, "
          f"est_inference={estimate['estimated_inference_time_seconds']:.1f}s, "
          f"s/completed={estimate['estimated_seconds_per_completed_job']:.2f}")


def main() -> None:
    for slug, model_key in SUBSCRIPTION_RUNS.items():
        update_manifest(slug, model_key)


if __name__ == "__main__":
    main()
