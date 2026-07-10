# Human Gold Annotation Guidelines — apolo_job_extraction_model_eval_50

## Purpose

This gold set exists to let us **measure** model extraction quality against a
human-reviewed reference, instead of only comparing models against each
other's weak labels. It covers exactly the 50 frozen job postings in
`input/jobs_sample_50.jsonl` — the same sample every model is scored against.

**Model outputs are weak labels, not ground truth.** A model's extraction —
however confident, however schema-valid — is a guess. Nothing in
`outputs/*/extractions.jsonl` should be copied into this gold set. Annotate
directly from the source text in `gold/human_gold_template.jsonl` (or the raw
`input/jobs_sample_50.jsonl`), blind to any model's output.

This gold set is for **model evaluation only**. Do not use it directly to
drive Apolo's user-facing skill recommendations, taxonomy, or matching logic
— it is a small (n=50), single-annotator-by-default sample, not a validated
production label source.

## What counts as an explicit skill

A skill is a capability, tool, technical ability, or requirement that the
posting **explicitly states** the candidate should have. If you have to guess
or infer from the job title or industry alone, it is not a skill — do not add
it.

**Does NOT count as a skill:**
- Job title fragments repeated as if they were skills (e.g. "Vendedor" is not
  a skill unless the text separately requires a sales-related capability).
- Personality traits with no concrete referent ("buena persona", "con ganas
  de trabajar").
- Company benefits, schedule, contract type, or location.
- Anything not present in the text — do not infer standard skills for a role
  just because they're typical of that job.

## Skill categories

Assign exactly one category per skill:

| Category | Use for |
|---|---|
| `tool` | Named software, platform, or physical tool/equipment (Excel, SAP, torno CNC). |
| `technical` | A concrete technical procedure or ability, not tied to one named tool. |
| `soft` | Communication, teamwork, leadership, autonomy, adaptability. |
| `language` | A spoken/written language (English, Portuguese). |
| `domain` | A broad capability tied to an industry/function rather than a specific procedure (see "broad domains" below). |
| `methodology` | A named process/framework (Scrum, Lean, ISO 9001, 5S). |
| `certification` | A named license, certificate, or credential requirement. |
| `other` | Doesn't fit any of the above; use sparingly. |

## Education requirements

Mark `required: true` only when the text uses language equivalent to
"requisito", "excluyente", "debe contar con", "título requerido". Mark
`required: false` for "deseable", "ideal", "se valorará", "no excluyente".
If the text is silent on whether it's mandatory, prefer `false` and note the
ambiguity in `ambiguous_cases`.

## Seniority

Use the same enum the models use: `intern`, `junior`, `semi_senior`,
`senior`, `lead`, `unknown`. Base the label only on explicit signals (years of
experience stated, explicit level words, explicit supervisory/leadership
scope). If the posting gives no signal, use `unknown` — do not infer
seniority from job title alone (a "Jefe de Bodega" title alone is a `lead`
signal only if the text also describes supervisory scope; if it's silent,
still lean `lead` from the title but flag it in `ambiguous_cases`).

## Literal evidence

When you record a gold skill or education requirement, prefer copying the
literal supporting phrase from `job_card_text` (not `description_text`) so
future automated evidence checks can be run against gold too, the same way
they're run against model output.

## Ambiguity

If a case is genuinely unclear — the text could support two categories, two
seniority levels, or you're unsure whether something counts as a skill — do
not silently pick one. Add a short entry to `ambiguous_cases` describing the
disagreement so it can be adjudicated, and still make your best-effort single
label in the main fields.

## Broad domains ("ventas", "atención al cliente", "administración", etc.)

These are risky because they read as skills but function more like job
categories. Rule of thumb:
- If the text only names the broad area ("experiencia en ventas") with no
  further specificity, record it as a single `domain` skill (don't invent
  sub-skills it doesn't state) and consider flagging it in
  `ambiguous_cases` if it seems too vague to be useful.
- If the text gives a specific capability within that area ("manejo de caja",
  "atención de reclamos telefónicos"), record that specific phrase instead of
  (or in addition to) the broad domain term.

## Missing or low-information postings

Some postings in the sample are short or vague (see the model's
`low_information_input` warning as a hint, not gospel — verify against the
actual text). For these:
- It is valid for `gold_skills` to be a short list or empty.
- Do not pad the gold set with inferred skills to make it "look complete."
- Note briefly in `notes` if the posting is too thin to extract much at all.

## Adjudication and reporting

- **M0 pilot**: at least one reviewer labels all 50 rows.
- **Ideal**: a second reviewer independently labels a subset (e.g. 10-15
  rows) blind to the first reviewer's labels.
- Before claiming a SOTA-style evaluation (see `evaluation_protocol.md`),
  report: how many rows were double-reviewed, the disagreement rate on that
  subset, and the adjudicated (agreed-upon) labels used for scoring. A gold
  set with no disagreement reporting is a single-annotator weak-gold set, not
  a validated reference.

## Relationship to `human_review_template.jsonl`

If a file named `human_review_template.jsonl` already exists in this
experiment with fields like `accepted_skills` / `rejected_model_skills`, that
is a **model-error review template** — it exists to accept/reject a
*specific model's* predicted skills, not to author independent gold labels.
It is not a substitute for this blind annotation protocol or for
`gold/human_gold_template.jsonl`, and the two should not be merged silently.
