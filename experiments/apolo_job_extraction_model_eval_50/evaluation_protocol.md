# Evaluation Protocol — apolo_job_extraction_model_eval_50

## What weak-label diagnostics can tell you

The manifest/diagnostic metrics (`total_skills`, `skills_per_job_mean`,
`confidence_mean`, `seniority_unknown_rate`, etc.) describe what a model
*produced*: how much it extracted, how confident it claimed to be, how often
it fell back to `unknown`. They are useful for:

- Sanity-checking a model isn't degenerate (e.g. always empty, always
  identical confidence, always `unknown` seniority).
- Comparing models' *behavior* against each other under the same prompt.
- Flagging risk signals worth a human look (high `broad_domain_skill_rate`,
  high `confidence_1_rate`, many `empty_skill_rows`).

## What they cannot tell you

They cannot tell you whether any individual skill, education requirement, or
seniority label is *correct*. A model can be schema-perfect, evidence-grounded,
and still hallucinate a plausible-sounding skill, misclassify a category, or
assign confidence that has no relationship to actual correctness. Weak-label
diagnostics have no ground truth to check against — treat every ranking based
on them as "which model behaves most usably," not "which model is most
accurate."

## What qualifies as SOTA-style evaluation here

Only this qualifies:

1. A reviewed gold set exists (`gold/human_gold_50.jsonl`, not the pending
   template), with `review_status` no longer `pending` for the rows used.
2. Annotation followed `gold/annotation_guidelines.md`.
3. Inter-annotator agreement/adjudication is reported for at least a subset,
   per the guidelines' minimum protocol — or it is explicitly stated that
   this run is single-annotator and therefore a provisional gold set, not a
   validated one.
4. `evaluate_against_gold.py` was run and its deterministic metrics
   (`skill_name_exact_f1`, `skill_strict_type_f1`, `education_exact_f1`,
   `seniority_accuracy`, etc.) are what's being reported — not the weak-label
   diagnostics.

Anything short of this is a model-selection comparison, not a SOTA
evaluation, and should not be described as one.

## Why "more skills" is not a quality metric

A model that extracts more skills per job is not necessarily better — it may
simply have a lower bar for what counts as a skill, padding the count with
vague or duplicate items. Skill count only becomes informative once you know
the *precision* of those skills against gold. Until then, a high
`skills_per_job_mean` is exactly as likely to indicate noise as coverage.

## Why broad-domain extraction is risky

Skills categorized as `domain` (e.g. "ventas", "atención al cliente",
"administración") read as capabilities but often function as job-category
restatements. They are easy to extract (the words are usually right there in
the text) and hard to act on (they don't tell you a specific ability). A model
with a high `broad_domain_skill_rate` may look productive on paper while
adding little decision-useful signal. This is why the operational score
penalizes it and why gold annotation asks reviewers to think carefully before
accepting a bare domain term as a skill.

## Why evidence grounding matters

`evidence` fields exist so every extracted skill or requirement can be
audited against the literal source text. A model that fails the
evidence-substring check has paraphrased, translated, or invented text —
which means its claims cannot be checked at all, model output or human
review. Evidence failures are a hard gate, not a diagnostic, because without
them there is no way to distinguish a correct extraction from a
plausible-sounding hallucination later.

## How cost/latency are compared

Efficiency metrics (`cost_per_completed_row`, `seconds_per_completed_row`,
`tokens_per_completed_row`, `valid_rows_per_usd`) are only meaningful for
models that already pass the structural hard gates. Comparing cost/latency
across models with different failure rates is misleading — a cheap model that
fails half its rows is not "cheap per usable extraction." `compare_models.py`
reports a Pareto front on (cost, latency) restricted to gate-passing models
for exactly this reason.

## Go / Iterate / Stop thresholds

**Go** — safe to scale a model to more rows or feed it into further
pipeline work:
- Hard gates all pass (0 request/parse/schema/validation/evidence failures,
  50/50 rows completed).
- Reviewed gold evaluation exists and `skill_strict_type_f1` /
  `seniority_accuracy` meet whatever bar the team sets for the next stage.

**Iterate** — worth a prompt or model change before scaling:
- Hard gates pass but weak-label diagnostics show risk signals (high
  `broad_domain_skill_rate`, high `confidence_1_rate`, many
  `empty_skill_rows`), or gold evaluation shows low precision/recall on a
  specific category (e.g. seniority confusion, domain-skill over-extraction).

**Stop** — do not scale this model:
- Any hard-gate failure that isn't a transient API error (persistent
  schema/validation/evidence failures).
- Gold evaluation (once available) shows near-chance seniority accuracy or
  near-zero strict-type skill F1.

No result in this experiment currently meets the "Go" bar for scaling past 50
rows, because no reviewed gold evaluation exists yet — only hard gates and
weak-label diagnostics have been checked so far.
