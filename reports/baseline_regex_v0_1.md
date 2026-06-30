# Baseline Extraction Quality Audit — baseline_regex_v0_1

## Summary

| Metric | Value |
|--------|-------|
| Total jobs | 35,915 |
| Total candidate rows | 653,233 |
| Mean candidates / job | 18.19 |
| Median candidates / job | 18.0 |
| Jobs with ≥1 candidate | 35,915 |
| Jobs with ≥1 regex skill | 10,346 (28.8%) |
| Company parse errors | 0 |
| Silver schema version | 0.2.0 |
| Extraction schema version | extraction_v0.1 |
| Skill dictionary version | v0.1 |

## Top Skills

| Skill | Count |
|-------|-------|
| Excel | 5,404 |
| Microsoft Office | 3,697 |
| Inglés | 2,540 |
| SAP | 2,408 |
| ERP | 2,128 |
| CRM | 1,245 |
| Power BI | 1,071 |
| Word | 726 |
| SQL | 687 |
| Python | 421 |
| AutoCAD | 352 |
| PowerPoint | 311 |
| AWS | 244 |
| Salesforce | 196 |
| Java | 184 |
| Google Workspace | 136 |
| Tableau | 136 |
| GCP | 130 |
| Git | 100 |
| JavaScript | 85 |

## Candidate Sources

| Source | Count |
|--------|-------|
| paragraph | 313,816 |
| li | 303,502 |
| title | 35,915 |

## Skill Counts by Section

| Section | Skill Hits |
|---------|------------|
| requisitos | 7,170 |
| conocimientos | 5,135 |
| (none) | 3,323 |
| funciones | 2,857 |
| habilidades | 1,107 |
| responsabilidades | 873 |
| contrato | 584 |
| beneficios | 476 |
| horario | 391 |
| modalidad | 285 |

## Industry Coverage

| Industry | Total Jobs | With Skills | % |
|----------|-----------|-------------|---|
|  | 11,583 | 3,409 | 29.4% |
| Servicios | 4,222 | 1,388 | 32.9% |
| Consultoría | 2,824 | 1,206 | 42.7% |
| Comercio | 1,483 | 293 | 19.8% |
| Retail | 1,454 | 174 | 12.0% |
| Educación | 1,298 | 611 | 47.1% |
| Otra | 1,247 | 448 | 35.9% |
| Banca / Financiera | 1,024 | 271 | 26.5% |
| Consultora de Recursos Humanos | 996 | 397 | 39.9% |
| Construcción | 956 | 140 | 14.6% |
| Salud | 863 | 168 | 19.5% |
| Supermercado / Hipermercado | 849 | 100 | 11.8% |
| Gastronomía | 697 | 61 | 8.8% |
| Telecomunicaciones | 677 | 128 | 18.9% |
| Consumo masivo | 520 | 172 | 33.1% |
| Farmacéutica | 428 | 30 | 7.0% |
| Transporte | 407 | 96 | 23.6% |
| Automotriz | 396 | 172 | 43.4% |
| Administración | 331 | 150 | 45.3% |
| ONGs | 326 | 16 | 4.9% |
| Alimenticia | 316 | 39 | 12.3% |
| Minería / Petróleo / Gas | 304 | 99 | 32.6% |
| Industrial | 266 | 66 | 24.8% |
| Inmobiliaria | 252 | 86 | 34.1% |
| Manufactura | 240 | 47 | 19.6% |

## Warnings / Caveats

- `skills_regex_raw` and `skills_normalized` are JSON-encoded `list[str]`; always parse with `json.loads`.
- 28.8% of jobs have at least one regex skill — deterministic baseline before ML.
- Industry labels come from scraped `company_industry` and may be noisy.
