#!/usr/bin/env python
"""Generate the required benchmark figures (PNG + SVG) under reports/figures/.

Never calls a model; reads only outputs/*/manifest.json and the historical
Gemini-era baseline. A figure that needs data no model has (e.g. retry
attempts, cost) is skipped with a note in reports/figures/skip_notes.md
rather than rendered as a misleading empty/zero chart.

Usage:
    python experiments/apolo_job_extraction_model_eval_50/plot_model_benchmarks.py
"""

from pathlib import Path

from benchmarking import historical_baselines, loaders, plots
from build_benchmark_report import build_all_model_rows

EXPERIMENT_DIR = Path(__file__).parent
FIGURES_DIR = EXPERIMENT_DIR / "reports" / "figures"


def main() -> None:
    rows = build_all_model_rows()
    if not rows:
        print(f"No manifests found under {loaders.OUTPUTS_DIR}. Run run_model_eval.py for at least one model first.")
        return

    frozen_job_ids = loaders.load_frozen_job_ids()
    historical = historical_baselines.build_baseline_report(frozen_job_ids)

    all_notes = []
    written = []
    for plot_fn in plots.ALL_PLOTS:
        paths, notes = plot_fn(rows, FIGURES_DIR)
        written.extend(paths)
        all_notes.extend(notes)

    paths, notes = plots.plot_reliability_vs_quality(rows, FIGURES_DIR, historical=historical)
    written.extend(paths)
    all_notes.extend(notes)

    for path in written:
        print(f"Wrote {path}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    notes_path = FIGURES_DIR / "skip_notes.md"
    if all_notes:
        notes_path.write_text(
            "# Skipped figures / series\n\n" + "\n".join(f"- {n}" for n in all_notes) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {notes_path} ({len(all_notes)} skip note(s))")
    elif notes_path.exists():
        notes_path.unlink()


if __name__ == "__main__":
    main()
