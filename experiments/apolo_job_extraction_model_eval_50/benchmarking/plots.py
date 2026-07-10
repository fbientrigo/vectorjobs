"""The 10 required figures + 3 Pareto figures, as PNG and SVG.

Every function takes `rows` — flattened per-model dicts produced by
build_benchmark_report.build_model_row() (keys prefixed reliability_/
structural_/efficiency_/quality_) — and an optional `historical` dict from
historical_baselines.build_baseline_report(). The historical baseline is
only overlaid where its required metric actually exists; otherwise the
function returns a skip note instead of plotting a fabricated zero.

Each function returns (paths_written: list[Path], skip_notes: list[str]).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from adjustText import adjust_text
except ModuleNotFoundError:
    def adjust_text(*args, **kwargs):
        return args[0] if args else None
from matplotlib.lines import Line2D

HISTORICAL_MARKER = "D"
HISTORICAL_COLOR = "#999999"

MODEL_PALETTE = plt.get_cmap("tab10").colors


def _save(fig, out_dir, name) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        paths.append(path)
    plt.close(fig)
    return paths


def _labels(rows):
    return [r["model"] for r in rows]


def _model_color(model, all_models):
    return MODEL_PALETTE[sorted(set(all_models)).index(model) % len(MODEL_PALETTE)]


def _place_labels(ax, rows_xy, expand=(1.3, 1.5), force_text=(0.3, 0.5), max_move=(80, 80)):
    texts = [ax.text(x, y, label, fontsize=8) for label, x, y in rows_xy]
    if texts:
        adjust_text(texts, ax=ax, expand=expand, force_text=force_text, max_move=max_move,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))


def _model_legend(ax, models, all_models):
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=_model_color(m, all_models), markersize=7, label=m)
        for m in sorted(set(models))
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, title="model")


def plot_model_valid_completion_rate(rows, out_dir):
    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.4), 4.5))
    target = rows[0]["reliability_n_jobs_target"] if rows else 50
    counts = [r["reliability_single_run_valid_count"] for r in rows]
    labels = _labels(rows)
    bars = ax.bar(labels, counts, color="#4C72B0")
    ax.axhline(target, color="black", linewidth=1, linestyle="--", label=f"target = {target}")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.5, f"{count}/{target}", ha="center", fontsize=9)
    ax.set_ylim(0, target * 1.15)
    ax.set_ylabel(f"valid completed jobs (of {target} requested)")
    ax.set_title("Valid completion reliability — full 50-job target denominator")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, out_dir, "model_valid_completion_rate"), []


FAILURE_CATEGORIES = [
    ("structural_n_request_failures", "request"),
    ("structural_n_json_parse_failures", "json"),
    ("structural_n_schema_failures", "schema"),
    ("structural_n_evidence_substring_failures", "evidence"),
]


def plot_model_failure_profile(rows, out_dir):
    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.4), 4.5))
    labels = _labels(rows)
    bottom = [0] * len(rows)
    for key, name in FAILURE_CATEGORIES:
        values = [r.get(key) or 0 for r in rows]
        ax.bar(labels, values, bottom=bottom, label=name)
        bottom = [b + v for b, v in zip(bottom, values)]
    missing = [r.get("structural_n_missing_final_rows") or 0 for r in rows]
    ax.bar(labels, [0] * len(rows), label=f"missing_final_rows total={sum(missing)}", color="none")
    ax.set_ylabel("failure count (out of n_attempted)")
    ax.set_title("Failure profile by category (not collapsed into one count)")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, out_dir, "model_failure_profile"), []


def plot_model_retry_burden(rows, out_dir):
    have_data = [r for r in rows if r.get("reliability_attempt_level_data_available")]
    if not have_data:
        note = (
            "model_retry_burden skipped: no model has attempt-level retry data "
            "(run_model_eval.py does not persist per-attempt records). "
            "See reliability_attempt_level_data_available=False for all models."
        )
        return [], [note]

    fig, ax = plt.subplots(figsize=(max(6, len(have_data) * 1.4), 4.5))
    labels = _labels(have_data)
    total = [r["reliability_total_request_attempts"] or 0 for r in have_data]
    retries = [r["reliability_total_retry_attempts"] or 0 for r in have_data]
    recovered = [r["reliability_retry_recovered_jobs"] or 0 for r in have_data]
    permanent = [r["reliability_n_failed_final"] or 0 for r in have_data]
    x = range(len(labels))
    width = 0.2
    ax.bar([i - 1.5 * width for i in x], total, width, label="total attempts")
    ax.bar([i - 0.5 * width for i in x], retries, width, label="retry attempts")
    ax.bar([i + 0.5 * width for i in x], recovered, width, label="recovered jobs")
    ax.bar([i + 1.5 * width for i in x], permanent, width, label="permanent failures")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Retry burden")
    ax.legend()
    return _save(fig, out_dir, "model_retry_burden"), []


def plot_model_cost_50_requested_jobs(rows, out_dir):
    priced = [r for r in rows if r.get("efficiency_observed_cost_usd") is not None]
    skip_notes = []
    unpriced = [r["model"] for r in rows if r not in priced]
    if unpriced:
        skip_notes.append(f"models excluded (no observed cost, not shown as 0): {unpriced}")
    if not priced:
        return [], skip_notes + ["model_cost_50_requested_jobs skipped: no model has observed cost."]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(8, len(priced) * 2), 4.5))
    labels = _labels(priced)
    total_cost = [r["efficiency_observed_cost_usd"] for r in priced]
    ax1.bar(labels, total_cost, color="#55A868")
    ax1.set_title("Total observed cost (full attempted run)")
    ax1.set_ylabel("USD")
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")

    cost_per_attempted = [r["efficiency_cost_per_attempted_job"] for r in priced]
    cost_per_valid = [r["efficiency_cost_per_eventually_valid_job"] for r in priced]
    x = range(len(labels))
    width = 0.35
    ax2.bar([i - width / 2 for i in x], cost_per_attempted, width, label="cost / attempted job")
    ax2.bar([i + width / 2 for i in x], cost_per_valid, width, label="cost / valid job")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("USD")
    ax2.set_title("Cost normalized two ways")
    ax2.legend()
    return _save(fig, out_dir, "model_cost_50_requested_jobs"), skip_notes


def plot_model_speed_comparison(rows, out_dir):
    timed = [r for r in rows if r.get("efficiency_wall_time_seconds") is not None]
    if not timed:
        return [], ["model_speed_comparison skipped: no model has wall_time_seconds."]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(8, len(timed) * 2), 4.5))
    labels = _labels(timed)
    ax1.bar(labels, [r["efficiency_completed_jobs_per_minute"] for r in timed], color="#C44E52")
    ax1.set_title("Valid jobs per minute")
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")

    ax2.bar(labels, [r["efficiency_seconds_per_valid_job"] for r in timed], color="#8172B2")
    ax2.set_title("Seconds per valid job")
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
    for r in timed:
        wt = r["efficiency_wall_time_seconds"]
        ax2.annotate(f"wall={wt:.0f}s", (r["model"], r["efficiency_seconds_per_valid_job"]),
                     textcoords="offset points", xytext=(0, 5), fontsize=7, ha="center")
    return _save(fig, out_dir, "model_speed_comparison"), []


def _cost_bubble_sizes(costs, scale=8000.0, min_size=40.0):
    return [min_size + scale * c for c in costs]


def plot_reliability_vs_quality(rows, out_dir, historical=None):
    priced = [r for r in rows if r.get("efficiency_observed_cost_usd") is not None
              and r.get("quality_quality_value") is not None]
    skip_notes = []
    if not priced:
        return [], ["reliability_vs_quality skipped: need observed cost and a quality value per model."]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    all_models = _labels(rows)
    x = [r["reliability_single_run_valid_rate"] for r in priced]
    y = [r["quality_quality_value"] for r in priced]
    sizes = _cost_bubble_sizes([r["efficiency_observed_cost_usd"] for r in priced])
    colors = [_model_color(r["model"], all_models) for r in priced]
    ax.scatter(x, y, s=sizes, alpha=0.6, color=colors, edgecolors="black")
    ax.margins(0.2)
    _place_labels(ax, [(r["model"], xi, yi) for r, xi, yi in zip(priced, x, y)],
                  expand=(1.8, 2.2), force_text=(0.8, 1.5), max_move=(150, 150))
    _model_legend(ax, [r["model"] for r in priced], all_models)

    if historical and historical.get("evidence_grounded_rate") is not None:
        skip_notes.append("historical baseline omitted: no comparable single-run valid rate or observed cost.")
    ax.set_xlabel("single-run valid rate (x)")
    ax.set_ylabel(f"quality ({priced[0]['quality_quality_source']})")
    ax.set_title("Reliability vs. quality — bubble area = observed cost")
    return _save(fig, out_dir, "reliability_vs_quality"), skip_notes


def plot_cost_vs_quality(rows, out_dir):
    priced = [r for r in rows if r.get("efficiency_cost_per_eventually_valid_job") is not None
              and r.get("quality_quality_value") is not None]
    if not priced:
        return [], ["cost_vs_quality skipped: need cost_per_eventually_valid_job and a quality value per model."]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    all_models = _labels(rows)
    x = [r["efficiency_cost_per_eventually_valid_job"] for r in priced]
    y = [r["quality_quality_value"] for r in priced]
    colors = [_model_color(r["model"], all_models) for r in priced]
    ax.scatter(x, y, color=colors, edgecolors="black")
    _place_labels(ax, [(r["model"], xi, yi) for r, xi, yi in zip(priced, x, y)])
    _model_legend(ax, [r["model"] for r in priced], all_models)
    ax.set_xlabel("cost per valid output (USD)")
    ax.set_ylabel(f"quality ({priced[0]['quality_quality_source']})")
    ax.set_title("Cost vs. quality")
    return _save(fig, out_dir, "cost_vs_quality"), []


def plot_speed_quality_cost_bubble(rows, out_dir):
    ready = [r for r in rows if r.get("efficiency_completed_jobs_per_minute") is not None
             and r.get("quality_quality_value") is not None
             and r.get("efficiency_observed_cost_usd") is not None]
    if not ready:
        return [], ["speed_quality_cost_bubble skipped: need speed, quality, and observed cost per model."]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    all_models = _labels(rows)
    x = [r["efficiency_completed_jobs_per_minute"] for r in ready]
    y = [r["quality_quality_value"] for r in ready]
    sizes = _cost_bubble_sizes([r["efficiency_observed_cost_usd"] for r in ready])
    alphas = [0.3 + 0.6 * (r["reliability_single_run_valid_rate"] or 0) for r in ready]
    for r, xi, yi, s, a in zip(ready, x, y, sizes, alphas):
        ax.scatter(xi, yi, s=s, alpha=a, color=_model_color(r["model"], all_models), edgecolors="black")
    ax.margins(0.2)
    _place_labels(ax, [(r["model"], xi, yi) for r, xi, yi in zip(ready, x, y)],
                  expand=(1.8, 2.2), force_text=(0.8, 1.5), max_move=(150, 150))
    _model_legend(ax, [r["model"] for r in ready], all_models)
    ax.set_xlabel("valid jobs per minute")
    ax.set_ylabel(f"quality ({ready[0]['quality_quality_source']})")
    ax.set_title("Speed vs. quality — bubble area = cost, opacity = single-run valid rate")
    return _save(fig, out_dir, "speed_quality_cost_bubble"), []


def plot_cost_vs_reliability(rows, out_dir):
    priced = [r for r in rows if r.get("efficiency_cost_per_attempted_job") is not None]
    if not priced:
        return [], ["cost_vs_reliability skipped: need cost_per_attempted_job per model."]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    all_models = _labels(rows)
    x = [r["efficiency_cost_per_attempted_job"] for r in priced]
    y = [r["reliability_single_run_valid_rate"] for r in priced]
    colors = [_model_color(r["model"], all_models) for r in priced]
    ax.scatter(x, y, color=colors, edgecolors="black")
    _place_labels(ax, [(r["model"], xi, yi) for r, xi, yi in zip(priced, x, y)])
    _model_legend(ax, [r["model"] for r in priced], all_models)
    ax.set_xlabel("cost per attempted job (USD)")
    ax.set_ylabel("single-run valid rate")
    ax.set_title("Cost vs. reliability")
    return _save(fig, out_dir, "cost_vs_reliability"), []


def pareto_front(rows, x_key, y_key, minimize_x=True, maximize_y=True):
    candidates = [r for r in rows if r.get(x_key) is not None and r.get(y_key) is not None]
    front = []
    for r in candidates:
        rx, ry = r[x_key], r[y_key]
        dominated = False
        for o in candidates:
            if o is r:
                continue
            ox, oy = o[x_key], o[y_key]
            x_ok = (ox <= rx) if minimize_x else (ox >= rx)
            y_ok = (oy >= ry) if maximize_y else (oy <= ry)
            strictly_better = (ox < rx if minimize_x else ox > rx) or (oy > ry if maximize_y else oy < ry)
            if x_ok and y_ok and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(r)
    return front, candidates


def _plot_pareto(rows, x_key, y_key, minimize_x, maximize_y, x_label, y_label, title, name, out_dir):
    front, candidates = pareto_front(rows, x_key, y_key, minimize_x, maximize_y)
    if not candidates:
        return [], [f"{name} skipped: no model has both {x_key} and {y_key}."]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    all_models = _labels(rows)
    front_ids = {id(r) for r in front}

    front_sorted = sorted(front, key=lambda r: r[x_key])
    fx = [r[x_key] for r in front_sorted]
    fy = [r[y_key] for r in front_sorted]
    if len(front_sorted) > 1:
        ax.step(fx, fy, where="post" if minimize_x else "pre", linestyle=(0, (5, 3)),
                color="#444444", linewidth=1, alpha=0.6, zorder=1)

    for r in candidates:
        is_front = id(r) in front_ids
        ax.scatter(r[x_key], r[y_key], color=_model_color(r["model"], all_models),
                   marker="*" if is_front else "o", s=200 if is_front else 60,
                   edgecolors="black" if is_front else "none", zorder=2)
    ax.margins(0.15)
    _place_labels(ax, [(r["model"], r[x_key], r[y_key]) for r in candidates],
                  expand=(1.8, 2.2), force_text=(0.8, 1.5), max_move=(150, 150))
    _model_legend(ax, [r["model"] for r in candidates], all_models)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title + " (star = Pareto-optimal)")
    return _save(fig, out_dir, name), []


def plot_pareto_quality_cost(rows, out_dir):
    return _plot_pareto(
        rows, "efficiency_cost_per_eventually_valid_job", "quality_quality_value",
        minimize_x=True, maximize_y=True,
        x_label="cost per valid job (USD)", y_label="quality",
        title="Pareto: quality vs. cost", name="pareto_quality_cost", out_dir=out_dir,
    )


def plot_pareto_quality_speed(rows, out_dir):
    return _plot_pareto(
        rows, "efficiency_completed_jobs_per_minute", "quality_quality_value",
        minimize_x=False, maximize_y=True,
        x_label="valid jobs per minute", y_label="quality",
        title="Pareto: quality vs. speed", name="pareto_quality_speed", out_dir=out_dir,
    )


def plot_pareto_reliability_cost(rows, out_dir):
    return _plot_pareto(
        rows, "efficiency_observed_cost_usd", "reliability_single_run_valid_rate",
        minimize_x=True, maximize_y=True,
        x_label="total observed cost (USD)", y_label="valid completion rate",
        title="Pareto: reliability vs. cost", name="pareto_reliability_cost", out_dir=out_dir,
    )


ALL_PLOTS = [
    plot_model_valid_completion_rate,
    plot_model_failure_profile,
    plot_model_retry_burden,
    plot_model_cost_50_requested_jobs,
    plot_model_speed_comparison,
    plot_cost_vs_quality,
    plot_speed_quality_cost_bubble,
    plot_cost_vs_reliability,
    plot_pareto_quality_cost,
    plot_pareto_quality_speed,
    plot_pareto_reliability_cost,
]
