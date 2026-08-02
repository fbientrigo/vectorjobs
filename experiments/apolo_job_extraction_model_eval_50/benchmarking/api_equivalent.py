"""OpenRouter list-price cost and OpenRouter-equivalent timing estimates for
runs that were genuinely executed but NOT called through an inference API
(subscription / manual / first-party model execution, or a recovered
historical dataset).

Nothing here calls a model or a pricing API. Every number is either derived
deterministically from recovered token counts or read from the static
OpenRouter snapshot constants below (snapshot date 2026-07-10).

Honesty contract (see the experiment task brief):
  * These are ESTIMATES, never measured provider telemetry.
  * cost_basis      == "openrouter_list_price_estimate"
  * timing_basis    == "openrouter_equivalent_estimate"
  * observed_cost_usd stays null; estimated_cost_usd carries the value.
  * provider_measured_latency_seconds is never populated from an estimate.
"""

import math

SNAPSHOT_DATE = "2026-07-10"
# Conservative +20% overhead on the pure inference-time model. Applied to
# TIME only, never to cost.
LATENCY_SAFETY_FACTOR = 1.20

# --- OpenRouter list prices, USD per million tokens (uncached) ---------------
PRICES = {
    "claude-sonnet-5": {
        "openrouter_model_id": "anthropic/claude-sonnet-5",
        "input_price_per_million_usd": 2.0,
        "output_price_per_million_usd": 10.0,
    },
    "gpt-5.5": {
        "openrouter_model_id": "openai/gpt-5.5",
        "input_price_per_million_usd": 5.0,
        "output_price_per_million_usd": 30.0,
    },
    "gemini-3.5-flash": {
        "openrouter_model_id": "google/gemini-3.5-flash",
        "input_price_per_million_usd": 1.5,
        "output_price_per_million_usd": 9.0,
    },
}

# --- OpenRouter performance snapshot ----------------------------------------
# latency = seconds to first token per request; throughput = output tok/sec.
PERFORMANCE = {
    "claude-sonnet-5": {
        "openrouter_performance_provider": "Google Vertex (Global)",
        "openrouter_latency_seconds": 2.43,
        "openrouter_throughput_tokens_per_second": 68.0,
        "note": "Official-provider latency/throughput pair from the same provider.",
    },
    "gpt-5.5": {
        "openrouter_performance_provider": "OpenRouter model-page composite",
        "openrouter_latency_seconds": 3.23,
        "openrouter_throughput_tokens_per_second": 39.0,
        "note": (
            "Model-page composite latency (3.23s) and throughput (39 tok/s); the "
            "static snapshot did not expose a verified same-provider pair."
        ),
    },
    "gemini-3.5-flash": {
        "openrouter_performance_provider": "Google Vertex",
        "openrouter_latency_seconds": 1.26,
        "openrouter_throughput_tokens_per_second": 78.0,
        "note": (
            "Conservative official-provider pair. Model pages are not "
            "effort-specific: 'high' reasoning may differ from model-level "
            "averages; the 1.20 multiplier is the conservative adjustment."
        ),
    },
}

TIMING_FORMULA = (
    "base = n_rows_attempted * openrouter_latency_seconds "
    "+ total_output_tokens / openrouter_throughput_tokens_per_second; "
    "estimated_inference_time_seconds = base * 1.20"
)


def utf8_bytes_div4(text: str) -> int:
    """Deterministic token fallback: ceil(UTF-8 bytes / 4)."""
    return math.ceil(len(text.encode("utf-8")) / 4)


def estimate_cost(total_input_tokens, total_output_tokens, model_key):
    """OpenRouter uncached list-price cost. No 1.20 factor on cost."""
    p = PRICES[model_key]
    return (
        total_input_tokens / 1_000_000 * p["input_price_per_million_usd"]
        + total_output_tokens / 1_000_000 * p["output_price_per_million_usd"]
    )


def cost_block(total_input_tokens, total_output_tokens, model_key,
               n_requested, n_completed):
    p = PRICES[model_key]
    cost = estimate_cost(total_input_tokens, total_output_tokens, model_key)
    per_completed = cost / n_completed if n_completed else None
    return {
        "openrouter_cost_model_id": p["openrouter_model_id"],
        "input_price_per_million_usd": p["input_price_per_million_usd"],
        "output_price_per_million_usd": p["output_price_per_million_usd"],
        "price_snapshot_date": SNAPSHOT_DATE,
        "observed_cost_usd": None,
        "estimated_cost_usd": cost,
        "estimated_api_equivalent_cost_usd": cost,
        "cost_basis": "openrouter_list_price_estimate",
        "cost_is_measured": False,
        "estimated_cost_per_requested_job_usd": cost / n_requested if n_requested else None,
        "estimated_cost_per_completed_job_usd": per_completed,
        "estimated_cost_per_1000_jobs_usd": per_completed * 1000 if per_completed is not None else None,
    }


def timing_block(total_output_tokens, model_key, n_attempted, n_completed):
    perf = PERFORMANCE[model_key]
    latency = perf["openrouter_latency_seconds"]
    throughput = perf["openrouter_throughput_tokens_per_second"]
    base = n_attempted * latency + total_output_tokens / throughput
    estimated = base * LATENCY_SAFETY_FACTOR
    sec_per_completed = estimated / n_completed if n_completed else None
    return {
        "timing_basis": "openrouter_equivalent_estimate",
        "timing_is_measured": False,
        "provider_measured_latency_seconds": None,
        "openrouter_model_id": PRICES[model_key]["openrouter_model_id"],
        "openrouter_performance_provider": perf["openrouter_performance_provider"],
        "openrouter_latency_seconds": latency,
        "openrouter_throughput_tokens_per_second": throughput,
        "openrouter_snapshot_date": SNAPSHOT_DATE,
        "latency_safety_factor": LATENCY_SAFETY_FACTOR,
        "base_estimated_runtime_seconds": base,
        "estimated_inference_time_seconds": estimated,
        "estimated_seconds_per_attempted_job": estimated / n_attempted if n_attempted else None,
        "estimated_seconds_per_completed_job": sec_per_completed,
        "estimated_completed_jobs_per_minute": 60 / sec_per_completed if sec_per_completed else None,
        "timing_estimation_formula": TIMING_FORMULA,
        "timing_limitations": [
            "OpenRouter-equivalent estimate, not measured provider telemetry.",
            perf["note"],
            "Do not treat as provider_measured_latency_seconds.",
        ],
    }


def full_estimate(total_input_tokens, total_output_tokens, model_key,
                  n_requested, n_attempted, n_completed):
    """Combined cost + timing estimate block for one run."""
    block = cost_block(total_input_tokens, total_output_tokens, model_key,
                       n_requested, n_completed)
    block.update(timing_block(total_output_tokens, model_key, n_attempted, n_completed))
    return block


if __name__ == "__main__":
    # self-check: 1.20 applied once to time, never to cost
    b = full_estimate(95179, 17528, "gpt-5.5", 50, 50, 50)
    assert abs(b["estimated_cost_usd"] - 1.001735) < 1e-9, b["estimated_cost_usd"]
    base = 50 * 3.23 + 17528 / 39
    assert abs(b["base_estimated_runtime_seconds"] - base) < 1e-6
    assert abs(b["estimated_inference_time_seconds"] - base * 1.20) < 1e-6
    assert b["observed_cost_usd"] is None
    assert b["provider_measured_latency_seconds"] is None
    print("api_equivalent self-check OK", round(b["estimated_inference_time_seconds"], 2), "s")
