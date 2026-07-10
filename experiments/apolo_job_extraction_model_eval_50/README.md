# Apolo Job Extraction — Model Eval (50 jobs)

## Purpose

Compare several OpenRouter models on the exact same 50 job postings, using
one shared extraction prompt, to see which model produces the most usable
`ApoloExtraction` output (skills, education requirements, seniority) before
picking one for real labeling work. This is a model-selection experiment,
not a training run and not a source of ground truth.

## The frozen 50-row sample

`build_sample.py` reads `data/silver/jobs.parquet` (read-only), filters to
rows where `job_card_text` is 300-5000 characters and `description_text` is
at least 200 characters, then samples exactly 50 rows with
`random_state=3407`. That seed makes the sample reproducible — re-running
`build_sample.py` yields the same 50 `job_id`s every time, so all models are
scored against the identical inputs.

Note: `data/silver/jobs.parquet` has no `company`, `posted_date`, or
`source` columns. `build_sample.py` maps `company_name -> company`,
`first_seen_at -> posted_date`, and fills `source` with the constant
`"linkedin"` (the dataset has a single source). This mapping is recorded in
`input/sample_manifest.json`.

Outputs:
- `input/jobs_sample_50.jsonl` — the frozen sample.
- `input/sample_manifest.json` — filter counts, seed, and the exact
  `job_id` list, so you can verify a later run used the same sample.

## Running one model

```bash
export OPENROUTER_API_KEY=sk-...
python experiments/apolo_job_extraction_model_eval_50/run_model_eval.py --model deepseek/deepseek-v4-flash
```

This calls the OpenRouter Chat Completions API once per job (50 calls),
using `prompts/apolo_extraction_base_prompt.md` as the system prompt, and
writes to `outputs/<safe_model_name>/`:
- `extractions.jsonl` — rows that passed validation (JSON parse, required
  fields, allowed enums, confidence in `[0, 1]`, evidence is a literal
  substring of `job_card_text`).
- `raw_responses.jsonl` — every raw API response, one per job, including
  failures.
- `failures.jsonl` — API, empty-output, non-JSON, schema, and validation
  failures.
- `validation_failures.jsonl` — field-level validation failures with
  `job_id`, `field_path`, `evidence`, and `reason`.
- `manifest.json` — run statistics: completion/failure counts, skill and
  confidence distributions, token usage, cost (if OpenRouter reports it),
  latency, hashes, git commit, and wall-clock timing.

## Resume and quality gate

Do not compare a model until it has `50/50` completed rows and zero evidence
substring failures. A missing row changes aggregate rates, and a non-substring
evidence value means the model copied or paraphrased text that cannot be
audited against the frozen input.

Resume a failed run with `--resume`; completed `job_id`s in
`extractions.jsonl` are skipped and new responses are appended:

```bash
python experiments/apolo_job_extraction_model_eval_50/run_model_eval.py --model deepseek/deepseek-v4-flash --resume
```

To retry one row:

```bash
python experiments/apolo_job_extraction_model_eval_50/run_model_eval.py --model deepseek/deepseek-v4-flash --resume --only-job-id 1118271814
```

Keep `raw_responses.jsonl` append-only. Later SOTA-style evaluation may need
to re-score raw model outputs with stricter parsers, new validators, or human
review; deleting failed responses destroys that provenance.

## Running all models with `run_all_models.py`

`run_all_models.py` is an orchestration layer only — it never calls
OpenRouter itself. It reads `configs/models.json`, decides which models to
run, writes `reports/run_all_models_plan.json`, then (unless `--dry-run`)
invokes `run_model_eval.py --model <model>` once per selected model as a
subprocess.

Preview the plan without calling anything (safe, no network, no API key
touched):

```bash
python experiments/apolo_job_extraction_model_eval_50/run_all_models.py --dry-run
```

Run only the models that don't already have a complete, failure-free
manifest, with a hard budget cap on projected accumulated spend:

```bash
export OPENROUTER_API_KEY=sk-...
python experiments/apolo_job_extraction_model_eval_50/run_all_models.py \
  --resume --skip-existing-valid --budget-usd 0.50
```

Run specific models, or cap how many run in one invocation:

```bash
python experiments/apolo_job_extraction_model_eval_50/run_all_models.py --models deepseek/deepseek-v4-pro --resume
python experiments/apolo_job_extraction_model_eval_50/run_all_models.py --max-models 2 --resume
```

The budget check uses the average cost/completed-row across whatever
manifests already exist as a rough cross-model proxy (see
`estimate_cost_per_row()` in `run_all_models.py`) — it is an approximation,
not a per-model price table.

## Comparing models

```bash
python experiments/apolo_job_extraction_model_eval_50/compare_models.py
```

Reads every `outputs/*/manifest.json` and writes, under `reports/`:
- `model_comparison.csv` / `model_comparison.md` — all metrics, grouped into
  (1) structural validity/hard gates, (2) weak-label diagnostics, (3)
  efficiency, (4) `operational_candidate_score`.
- `model_comparison_ranked.md` — models ranked by
  `operational_candidate_score` (an operational-readiness heuristic, **not**
  an accuracy score — see `evaluation_protocol.md`).
- `model_comparison_pareto.md` — Pareto front on (cost, latency) among
  gate-passing models (written only once 2+ models pass hard gates).

## Inspecting cost

Cost/latency are in every model's `manifest.json` (`estimated_cost_usd`,
`total_tokens`, `wall_time_seconds`) and aggregated per-row in
`reports/model_comparison.csv` (`cost_per_completed_row`,
`tokens_per_completed_row`, `seconds_per_completed_row`,
`valid_rows_per_usd`). `run_all_models.py --dry-run` also reports a
projected-cost plan before any model runs.

## Creating human gold

`gold/human_gold_template.jsonl` is a blind 50-row template (same `job_id`s
and order as `input/jobs_sample_50.jsonl`, no model predictions) generated by
`gold/build_gold_template.py`. To create a reviewed gold set:

1. Copy it: `cp gold/human_gold_template.jsonl gold/human_gold_50.jsonl`
   (or annotate a copy under a reviewer-specific name).
2. Follow `gold/annotation_guidelines.md` to fill in `gold_skills`,
   `gold_education_requirements`, `seniority_gold`, and flip
   `review_status` away from `"pending"` per row.
3. See the guidelines for the minimum adjudication protocol (single-reviewer
   pilot vs. double-reviewed subset with disagreement reporting).

## Evaluating against gold

```bash
python experiments/apolo_job_extraction_model_eval_50/evaluate_against_gold.py
python experiments/apolo_job_extraction_model_eval_50/evaluate_against_gold.py --model deepseek/deepseek-v4-flash
python experiments/apolo_job_extraction_model_eval_50/evaluate_against_gold.py --gold gold/human_gold_50.jsonl
```

Exits cleanly (no traceback, exit code 0) with a message if
`gold/human_gold_50.jsonl` doesn't exist yet, or if every row is still
`review_status: pending`. Otherwise writes `reports/model_vs_gold.csv` and
`reports/model_vs_gold.md` with deterministic exact/normalized/strict-type
skill precision/recall/F1, education F1, seniority accuracy, and evidence
failure rate — see `evaluation_protocol.md` for what these can and cannot
support, and note that exact match is a strict baseline (see that file for
notes on future taxonomy-aware matching).

## Why these outputs are weak labels, not ground truth

Every extraction here is a single LLM pass with no cross-checking,
adjudication, or domain-expert review. Even a model with perfect schema
compliance can still hallucinate a skill that isn't in the text, miscategorize
a domain capability as a technical one, or over/under-assign confidence.
The evidence-substring check catches copy-paste evidence but not incorrect
*interpretation* of that evidence. These outputs are useful for **ranking
models against each other**, not for treating any single model's extraction
as correct.

## Why human review is required before using results in Apolo

Nothing here has been checked by a person. Before any of this feeds Apolo's
skill taxonomy, matching logic, or user-facing data, a human should sample
and review actual extractions (not just the aggregate manifest numbers),
confirm the winning model's error modes are acceptable for the intended use,
and only then decide whether to scale up labeling with it. This experiment
answers "which model is worth trying at scale," not "is this data ready to
ship."

**Do not scale any model to 500+ rows before at least a minimal human gold
review exists** (see "Creating human gold" above and
`gold/annotation_guidelines.md`). Diagnostics alone cannot tell you whether a
model's extractions are correct, only what they look like in aggregate.

## What you can and cannot call these results

- **"Operational model comparison"** — comparing `n_rows_completed`,
  failure counts, cost, and latency across models. Always valid once
  `compare_models.py` has run.
- **"Weak-label diagnostic"** — comparing `total_skills`,
  `confidence_mean`, `broad_domain_skill_rate`, etc. across models. Valid as
  a description of model behavior; never cite these as accuracy.
- **"Human-gold evaluation"** — only once `evaluate_against_gold.py` has run
  against a `gold/human_gold_50.jsonl` with `review_status` no longer
  `pending`, following `gold/annotation_guidelines.md`.
- **"SOTA-style evaluation"** — only once human-gold evaluation exists *and*
  adjudication/agreement has been reported per the guidelines' minimum
  protocol (or the result is explicitly labeled provisional/single-annotator).
  See `evaluation_protocol.md` for the full checklist and Go/Iterate/Stop
  thresholds.
