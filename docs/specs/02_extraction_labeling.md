# Extraction Labeling Specification

This document defines the semantic contract and guidelines for the manual annotation seed generated in Milestone 1.

## 1. Purpose
* This CSV is a manual annotation seed for candidate-level classification.
* It is not model training yet. Its role is to establish a high-quality human baseline and validate classification boundaries before building a training dataset.

## 2. Command
To build the labeling seed, run the following CLI command:

```powershell
.\.venv\Scripts\python.exe -m jobsrec.cli build-labeling-seed `
  --silver-path data/silver/jobs.parquet `
  --candidates-path data/silver/job_extraction_candidates.parquet `
  --output-path reports/baseline_labeling_seed.csv `
  --sample-size 500 `
  --random-seed 42
```

## 3. CSV Columns
The output CSV contains the following columns:
* `label`: The manual annotation label (starts empty).
* `job_id`: Unique identifier for the job posting.
* `candidate_index`: The index of the text unit within the job posting.
* `candidate_text`: The raw text content of the candidate unit.
* `candidate_source`: The source of the text (`title`, `li`, or `paragraph`).
* `section_name`: The detected section header anchor or empty.
* `skills_normalized`: JSON array of canonical skills matching the regex.
* `title_clean`: Cleaned job title from the silver dataset.
* `company_name`: Normalized company name from the silver dataset.
* `company_industry`: Normalized company industry.
* `notes`: Free-text field for annotations/exceptions/ambiguities.

## 4. Allowed Labels and Meanings
Annotators must use only the following 12 pre-defined labels:

* **`HARD_SKILL`**: Explicit tool, software, language, platform, certification, or named technical capability.
  * *Examples:* Excel, SAP, SQL, Power BI, Licencia clase B.
* **`DOMAIN_SKILL`**: Sector/domain-specific knowledge or practice.
  * *Examples:* instrumental quirúrgico, gestión de inventario, mantención preventiva, evaluación crediticia.
* **`SOFT_SKILL`**: Interpersonal or behavioral competency.
  * *Examples:* comunicación, liderazgo, trabajo en equipo, proactividad.
* **`EDUCATION`**: Degree, career, academic title, or formal study area.
  * *Examples:* Técnico en Enfermería, Ingeniería Comercial, título profesional.
* **`EXPERIENCE`**: Years or explicit prior experience requirement.
  * *Examples:* 2 años de experiencia, experiencia en retail.
* **`RESPONSIBILITY`**: Task, duty, or action the person will perform.
  * *Examples:* gestionar contratos, atender clientes, coordinar turnos.
* **`BENEFIT`**: Compensation, perks, company offers.
  * *Examples:* seguro complementario, bonos, casino.
* **`LOCATION`**: City, region, worksite, route, or geographic requirement.
* **`SCHEDULE`**: Shift, working hours, modality of time.
  * *Examples:* 4x4, lunes a viernes, turnos rotativos.
* **`CONTRACT`**: Contract type or employment condition.
  * *Examples:* plazo fijo, indefinido, honorarios.
* **`IGNORE`**: Boilerplate, empty, duplicated, spam, vague, or not useful text.
* **`UNCERTAIN`**: Genuinely ambiguous after reading title and context.

## 5. Annotation Guidelines
* **Single Label:** Use one primary label only.
* **Dominant Intent:** If a row contains several ideas, label the dominant one.
* **Duties/Actions:** If it is a duty/action, prefer `RESPONSIBILITY`.
* **Knowledge/Capabilities:** If it is phrased as knowledge/capability, prefer `DOMAIN_SKILL` or `HARD_SKILL`.
* **Uncertainty:** If unsure between two labels, use `UNCERTAIN` and write a short note in the `notes` column.
* **No Custom Labels:** Do not invent labels outside the allowed set.

## 6. First-Pass Protocol
To ensure high annotation alignment and catch edge cases early:
1. **Pilot Phase:** Annotate only the first 100 rows first.
2. **Ambiguity Review:** Review ambiguities, border cases, and common patterns before labeling the remaining 400 rows.
3. **Distribution Audit:** After 100 rows, inspect the label distribution and read all common `UNCERTAIN` notes to align definitions.
