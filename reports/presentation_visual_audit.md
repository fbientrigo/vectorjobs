# Presentation Visual Audit

Date: 2026-06-19

Scope:
- Beamer source: `presentations/estado_actual/main.tex`
- Overleaf package: `dist/estado_actual_overleaf`
- Figure folder: `dist/estado_actual_overleaf/figs`
- Manifest: `dist/estado_actual_overleaf/asset_manifest.json`

No code was edited, no figures were regenerated, and no files were deleted.

## Build Metadata Observed

`asset_manifest.json` says the current Overleaf folder was created from `data\silver\jobs.parquet` with:

- `sample_size`: 20000
- `max_rows`: 50000
- `time_bin`: `D`
- `time_column`: `original_listed_time`
- `k`: 12
- `random_state`: 42
- every listed figure has status `copied`

Important discrepancy to resolve before presenting: the stated context says the verified folder used `--max-rows 20000`, but the manifest and downstream cluster/skill manifests say `max_rows=50000`; only the temporal TF-IDF sample is 20000 rows.

Observed support is very uneven:

- Temporal drift bins: 61 daily bins, `jobs_in_month` min/median/max = 1 / 5 / 5601.
- Salary coverage bins: 42 daily bins, coverage min/median/max = 0.111 / 0.303 / 1.0, with 100% bins often at `n_jobs=1`.
- Cluster time bins: daily total support min/median/max = 1 / 19 / 13497.
- Skill domain-bin support: min/median/max = 1 / 9 / 3734.
- Overall salary coverage in temporal TF-IDF: 5769 salary rows out of 20000 selected rows, 28.845%.

## Slide Usage

All 11 PNGs in `dist/estado_actual_overleaf/figs` are referenced by `presentations/estado_actual/main.tex`.

| Figure | Slide context |
| --- | --- |
| `centroid_drift_by_month.png` | Centroides: que son y por que importan |
| `job_cluster_map_svd.png` | Centroides en el espacio semantico |
| `cluster_semantic_trajectory.png` | Centroides en el espacio semantico |
| `cluster_share_timeseries.png` | Clusters interpretables |
| `market_value_by_sector.png` | Market cap laboral |
| `salary_coverage_by_month.png` | El eje salarial en los datos |
| `centroid_drift_salary_weighted_by_month.png` | El eje salarial en los datos |
| `skill_evolution_tech.png` | Evolucion de skills por dominio |
| `skill_evolution_health.png` | Evolucion de skills por dominio |
| `storage_growth_tb.png` | Crecimiento de almacenamiento |
| `aws_cost_projection.png` | Proyeccion de costo mensual |

## Figure-by-Figure Audit

### `centroid_drift_by_month.png`

- Source: `scripts/build_presentation_assets.py::run_analytics` -> `jobsrec temporal-demo` -> `src/jobsrec/trends/temporal.py::_write_required_plots` -> `_plot_bar`.
- Current issue: filename says month, title says day, and the chart is a bar chart of cosine distance between consecutive daily centroids. The concept is not self-contained, and the plot hides denominator size. Because median daily support is only 5 jobs, many spikes are likely low-support artifacts rather than meaningful market movement.
- Recommended replacement: reliability-aware drift plot. Use a line or lollipop chart of cosine distance between consecutive time bins, with point size or labels for `jobs_in_month`, muted color for low-support bins, and a visible note such as "cosine distance between consecutive TF-IDF/SVD daily centroids; low-n bins shaded". Prefer weekly bins if the deck goal is trend communication rather than showing raw diagnostic noise.
- Filename/main.tex: filename can be preserved to avoid touching `main.tex`, but the in-figure title must stop saying ambiguous "Centroid Drift by Day" without definition. A cleaner later rename would be `centroid_drift_by_day_support.png`, which would require `main.tex` updates.
- Exact tests/checks needed:
  - Verify source table includes `month`, `previous_month`, `centroid_drift`, `jobs_in_month`.
  - Assert every plotted point has a visible support cue (`n`, marker size, alpha, or low-support shading).
  - Assert bins below the chosen threshold, for example `n < 100` or `n < 1000`, are visually differentiated.
  - Assert title/subtitle defines drift as cosine distance between consecutive centroids.
  - Visual QA at Beamer size: x labels readable and no label clipping.

### `job_cluster_map_svd.png`

- Source: `scripts/build_presentation_assets.py::run_analytics` -> `jobsrec temporal-demo` -> `src/jobsrec/trends/temporal.py::_optional_cluster_outputs`.
- Current issue: visually readable, but semantically thin. It shows only cluster IDs 0-7 from the temporal-demo optional KMeans run, while the rest of the deck discusses 12 fixed temporal clusters with semantic labels. It can be mistaken for the same clustering system used in `temporal-clusters`.
- Recommended replacement: either keep it as a compact "2D SVD diagnostic map" with a caption explicitly saying it is a projection and uses temporal-demo clusters, or replace it with the 12-cluster assignment map from the same clustering run used by the temporal cluster figures. Add semantic labels or a small label table instead of numeric color IDs only.
- Filename/main.tex: can be preserved if the image remains a 2D cluster map. If replacing it with a different clustering source, keep the filename only if the caption and title disclose the source.
- Exact tests/checks needed:
  - Verify plotted cluster count matches the slide text or the figure caption.
  - Assert title or caption contains "2D projection" and does not imply clustering was performed in 2D.
  - Confirm color legend maps to semantic labels or a nearby table, not only raw IDs.
  - Visual QA: points remain visible, legend/colorbar does not dominate the slide.

### `cluster_semantic_trajectory.png`

- Source: `scripts/build_presentation_assets.py::run_analytics` -> `jobsrec temporal-clusters` -> `src/jobsrec/trends/temporal_clusters.py::run_temporal_clusters` -> `_write_cluster_semantic_trajectory`.
- Current issue: this is the weakest figure. It draws many jagged paths in a 2D SVD projection, marks global centroids as black Xs, and annotates only the final date repeatedly. The title admits it is not clustering space, but the visual still invites geometric interpretation. It does not answer "what changed with time?"
- Recommended replacement: replace with a time-composition or change-summary figure, not another projected path plot. Best options:
  - 100% stacked area chart of cluster share over time with top clusters plus `Other`.
  - Heatmap of cluster share by time bin, sorted by overall volume or change.
  - Dumbbell/slope chart comparing first reliable window vs last reliable window, with endpoint `n`.
- Filename/main.tex: main.tex should change if the replacement is not a semantic trajectory. Recommended new filename: `cluster_share_composition.png` or `cluster_share_change_summary.png`. Preserving `cluster_semantic_trajectory.png` for a share-composition chart would keep a misleading filename.
- Exact tests/checks needed:
  - Require reliable endpoint selection, excluding bins below a support threshold.
  - If using 100% stacked area, assert each plotted time interval sums to 100% after top-N plus `Other`.
  - If using heatmap/slope chart, assert denominator `n_jobs` is visible for endpoints or encoded by alpha/hatching.
  - Assert title states "share of postings by fixed cluster", not "semantic trajectory".
  - Visual QA: no more than 6-8 colors unless using a heatmap; legend readable in Beamer two-column layout.

### `cluster_share_timeseries.png`

- Source: `scripts/build_presentation_assets.py::run_analytics` -> `jobsrec temporal-clusters` -> `src/jobsrec/trends/temporal_clusters.py::_write_cluster_share_timeseries`.
- Current issue: closer to the right story than the trajectory plot, but still noisy. Top-8 lines overlap, daily bins create many 100% spikes from tiny denominators, and the chart does not make clear that omitted clusters mean the visible lines need not sum to 100%.
- Recommended replacement: 100% stacked area over reliable time bins for top clusters plus `Other`, or a small-multiple line chart limited to 4-5 focus clusters with a support histogram underneath.
- Filename/main.tex: can be preserved if the replacement remains a cluster share time series.
- Exact tests/checks needed:
  - Assert top-N plus `Other` shares sum to 1.0 per time bin for stacked area.
  - Assert low-support bins are dropped, shaded, or visibly flagged.
  - Assert labels use `cluster_label`, not raw IDs only.
  - Visual QA: legend fits, no overlapping labels, y-axis formatted as percent.

### `market_value_by_sector.png`

- Source: `scripts/build_presentation_assets.py::generate_direct_figures` -> `scripts/market_value.py::_sector_rows_from_metrics` -> `plot_market_value`.
- Current issue: the footnote is clipped in the PNG. The graphic is semantically useful but still fragile: it uses first/last endpoint windows, salary imputation, and a proxy formula while large endpoint changes are shown without endpoint denominators on the chart.
- Recommended replacement: horizontal dumbbell/slope chart with first vs last reliable endpoint, endpoint dates, `n_jobs`, and a short subtitle: "proxy = posting share x median salary; salary coverage limited". Move the long source note into the slide text or a compact wrapped caption.
- Filename/main.tex: can be preserved.
- Exact tests/checks needed:
  - Assert no footnote/caption text is clipped in the saved PNG.
  - Assert endpoint bins pass `min_bin_jobs` or are explicitly marked as low-support.
  - Assert salary fallback/imputation is disclosed.
  - Assert values in the plot match `cluster_time_metrics.parquet` for the chosen endpoints.

### `salary_coverage_by_month.png`

- Source: `scripts/build_presentation_assets.py::run_analytics` -> `jobsrec temporal-demo` -> `src/jobsrec/trends/temporal.py::_write_salary_weighted_plots` -> `_plot_bar`.
- Current issue: filename says month, title says day. The y-axis shows coverage rate only, so 100% bars look strong even when `n_jobs=1`. It does not show how many rows had salary data or how many total rows were in the bin.
- Recommended replacement: support-aware coverage chart. Prefer stacked bars of `n_salary_jobs` vs missing salary per time bin with a coverage line on a secondary axis, or a coverage line with marker size proportional to `n_jobs`. Include overall coverage (28.845% for the temporal sample) in subtitle.
- Filename/main.tex: can be preserved if the figure remains salary coverage. A later rename to `salary_coverage_by_day_support.png` would be more honest but requires `main.tex`.
- Exact tests/checks needed:
  - Verify source table includes `n_jobs`, `n_salary_jobs`, and `salary_coverage`.
  - Assert any 100% coverage bin displays its `n_jobs` or is low-support flagged.
  - Assert y-axis uses percent formatting.
  - Assert subtitle includes total selected rows and overall salary coverage.
  - Visual QA: no clipped x labels in two-column Beamer placement.

### `centroid_drift_salary_weighted_by_month.png`

- Source: `scripts/build_presentation_assets.py::run_analytics` -> `jobsrec temporal-demo` -> `src/jobsrec/trends/temporal.py::_write_salary_weighted_plots` -> `_plot_bar`.
- Current issue: same interpretability problem as unweighted drift, with extra sample-selection risk. It describes only the salary-disclosed USD subset; many bars compare bins with very small salary-supported counts, but the chart does not show `n_salary_from`, `n_salary_to`, or coverage.
- Recommended replacement: support-aware salary-weighted drift chart, ideally paired with salary coverage. Show cosine distance with marker size/min support based on salary-supported rows and include subtitle "salary-disclosed USD subset only".
- Filename/main.tex: can be preserved.
- Exact tests/checks needed:
  - Verify source table includes `n_from`, `n_to`, `n_salary_from`, `n_salary_to`, `salary_coverage_from`, `salary_coverage_to`.
  - Assert low salary-supported comparisons are shaded or omitted.
  - Assert figure text says "salary-disclosed USD subset".
  - Assert y-axis and title define the metric as cosine distance.

### `skill_evolution_tech.png`

- Source: `scripts/build_presentation_assets.py::run_analytics` -> `jobsrec skill-evolution` -> `src/jobsrec/trends/skill_evolution.py::run_skill_evolution` -> `_write_skill_evolution_plot`.
- Current issue: readable, but semantically easy to misread. Each line is "share of tech-domain postings requiring skill", so lines are independent and can sum above 100%. Daily bins create 100% spikes from small denominators, and 12 lines plus legend is busy.
- Recommended replacement: 100% stacked area composition for the Tech domain. For each time bin, normalize selected top skills plus `Other` so the stack sums to 100%. Use top 5-7 stable skills, aggregate the rest into `Other`, and add a support cue for domain `job_count`.
- Filename/main.tex: can be preserved because the figure is still Tech skill evolution.
- Exact tests/checks needed:
  - Assert per-time-bin stacked shares sum to 100% within a tight tolerance.
  - Assert `Other` is included when top skills do not cover all skill counts.
  - Assert low-support domain bins are dropped or hatched.
  - Assert subtitle explains the denominator: "within Tech skill mentions/postings, normalized per interval".
  - Visual QA: no more than 8 legend entries; colors are distinguishable in the slide.

### `skill_evolution_health.png`

- Source: `scripts/build_presentation_assets.py::run_analytics` -> `jobsrec skill-evolution` -> `src/jobsrec/trends/skill_evolution.py::run_skill_evolution` -> `_write_skill_evolution_plot`.
- Current issue: best current plot for presentation readability, but it is still independent skill coverage, not composition. Health Care Provider dominates, while other lines spike to 100% in sparse days. It does not answer "what is the mix of skills over time?" cleanly.
- Recommended replacement: 100% stacked area composition for the Health domain. Same logic as Tech: top skills plus `Other`, each interval sums to 100%, with low-support bins visually marked or omitted.
- Filename/main.tex: can be preserved.
- Exact tests/checks needed:
  - Assert per-time-bin stacked shares sum to 100%.
  - Assert Health Care Provider dominance does not hide smaller categories; use ordering and alpha/legend to keep it readable.
  - Assert low-support bins are not shown as reliable 100% spikes.
  - Assert title/subtitle distinguishes "composition share" from "coverage share".

### `storage_growth_tb.png`

- Source: `scripts/build_presentation_assets.py::generate_direct_figures` -> `scripts/aws_cost_projection.py::plot_storage_growth`.
- Current issue: visually clear and honest enough. Main limitation is semantic scope: it models stored TB from configured KB/job assumptions and ingestion growth, not actual measured storage growth. The footnote is small but not clipped.
- Recommended replacement: keep with small refinements. Add endpoint labels for 36-month TB under each scenario and a short subtitle "parametric estimate, not observed usage".
- Filename/main.tex: can be preserved.
- Exact tests/checks needed:
  - Assert footnote fits inside the rendered PNG.
  - Assert scenario assumptions match script constants and slide text.
  - Assert y-axis is TB and starts near zero.
  - Optional: check final values are annotated for quick reading.

### `aws_cost_projection.png`

- Source: `scripts/build_presentation_assets.py::generate_direct_figures` -> `scripts/aws_cost_projection.py::plot_cost_projection`.
- Current issue: visually clean, but semantically easy to overstate. It models only S3 storage plus embedding compute under fixed throughput and price assumptions; it does not include orchestration, retrieval serving, monitoring, networking, retries, or reprocessing. The "cambios reales" claim should be source-checked before final delivery.
- Recommended replacement: keep the two-panel structure, but add an explicit model-scope subtitle and show assumptions in a compact caption. If the talk audience is nontechnical, add "not full production TCO".
- Filename/main.tex: can be preserved.
- Exact tests/checks needed:
  - Assert caption/footnote discloses S3 + embedding compute only.
  - Assert price-change dates and prices are source-checked before presentation.
  - Assert stacked cost values match the script formulas for the base scenario.
  - Visual QA: top panel labels do not overlap and bottom y-axis remains readable.

## Recommended Priority

1. Replace `cluster_semantic_trajectory.png` with a cluster share composition/change figure and update `main.tex`.
2. Replace `skill_evolution_health.png` and `skill_evolution_tech.png` with 100% stacked area composition plots.
3. Replace `salary_coverage_by_month.png`, `centroid_drift_by_month.png`, and `centroid_drift_salary_weighted_by_month.png` with denominator-aware versions.
4. Fix `market_value_by_sector.png` footnote clipping and endpoint denominator disclosure.
5. Keep `storage_growth_tb.png` and `aws_cost_projection.png` with scope/assumption captions.

## Cross-Cutting Checks Before Final Deck

- Manifest consistency: `asset_manifest.json` and source manifests must agree with the described verified row cap.
- Figure existence: every `\includegraphics` target in `main.tex` exists in `dist/estado_actual_overleaf/figs`.
- Filename/title consistency: no `*_by_month.png` figure should display daily bins without an explicit explanation.
- Denominator visibility: every temporal rate/share/drift plot should expose `n` or visibly flag low-support bins.
- Projection honesty: every 2D projection should say it is a projection and not the clustering space.
- Composition math: every 100% stacked area replacement should sum to 100% per interval after top-N plus `Other`.
- Render QA: compile or render the Beamer deck and inspect the slides at presentation size for clipped captions, unreadable legends, and overcrowded axes.
