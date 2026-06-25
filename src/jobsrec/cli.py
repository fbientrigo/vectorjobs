"""
Click CLI entry-points for the jobsrec pipeline.

Commands
--------
build-silver      Load raw CSVs → build silver Parquet.
build-tfidf       Fit TF-IDF vectoriser on silver data → write gold artefacts.
recommend         Load gold artefacts → return top-k JSON recommendations.
profile-silver    Profile a silver Parquet and write a JSON data report.
temporal-demo     Build a fast temporal trend demo report and plots.
temporal-clusters Build fixed temporal cluster analytics report and plots.
skill-evolution   Build offline skill-share evolution analytics report and plots.
populares-validate Validate populares-scraper Parquet outputs.
populares-build-gold Build draft gold Parquet outputs for apolo-rag.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config(config_path: Optional[str]) -> dict:
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise click.BadParameter(f"Config file not found: {path}")
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="jobsrec")
def main() -> None:
    """jobsrec — LinkedIn job-skill recommendation pipeline."""


# ---------------------------------------------------------------------------
# build-silver
# ---------------------------------------------------------------------------

@main.command("build-silver")
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing postings.csv, jobs/, and mappings/.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Destination for jobs.parquet and manifest.json.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="YAML config file (e.g. configs/local_6gb.yaml).",
)
@click.option("--log-level", default="INFO", show_default=True)
def build_silver_cmd(
    input_dir: Path,
    output_dir: Path,
    config_path: Optional[str],
    log_level: str,
) -> None:
    """Load raw CSVs, build job_card_text, write silver Parquet."""
    _setup_logging(log_level)
    config = _load_config(config_path)

    from jobsrec.data.load import build_silver

    result = build_silver(
        input_dir=input_dir,
        output_dir=output_dir,
        config=config,
    )
    click.echo(
        json.dumps(
            {
                "output_path": str(result.output_path),
                "input_rows": result.input_rows,
                "output_rows": result.output_rows,
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# build-tfidf
# ---------------------------------------------------------------------------

@main.command("build-tfidf")
@click.option(
    "--silver-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the silver jobs.parquet file.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Destination for TF-IDF artefacts.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="YAML config file.",
)
@click.option("--log-level", default="INFO", show_default=True)
def build_tfidf_cmd(
    silver_path: Path,
    output_dir: Path,
    config_path: Optional[str],
    log_level: str,
) -> None:
    """Fit TF-IDF on silver data and write gold artefacts."""
    _setup_logging(log_level)
    config = _load_config(config_path)

    import pandas as pd

    from jobsrec.embeddings.tfidf import fit_and_save

    df = pd.read_parquet(silver_path)
    if "job_card_text" not in df.columns:
        raise click.UsageError(
            "silver Parquet does not contain 'job_card_text' column. "
            "Re-run build-silver."
        )

    documents: list[str] = df["job_card_text"].tolist()
    job_ids: list[int] = df["job_id"].tolist()

    result = fit_and_save(
        documents=documents,
        job_ids=job_ids,
        job_card_texts=documents,
        output_dir=output_dir,
        input_path=silver_path,
        config=config,
    )
    click.echo(
        json.dumps(
            {
                "vectorizer_path": str(result.vectorizer_path),
                "matrix_path": str(result.matrix_path),
                "n_docs": result.n_docs,
                "vocab_size": result.vocab_size,
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# recommend
# ---------------------------------------------------------------------------

@main.command("recommend")
@click.option(
    "--job-id",
    required=True,
    type=int,
    help="job_id of the query posting.",
)
@click.option(
    "--top-k",
    default=5,
    show_default=True,
    type=int,
    help="Number of recommendations to return.",
)
@click.option(
    "--gold-dir",
    default="data/gold",
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory containing TF-IDF gold artefacts.",
)
@click.option("--log-level", default="WARNING", show_default=True)
def recommend_cmd(
    job_id: int,
    top_k: int,
    gold_dir: Path,
    log_level: str,
) -> None:
    """Return top-k similar jobs as JSON."""
    _setup_logging(log_level)

    from jobsrec.recommend.retrieval import TfidfRetriever

    retriever = TfidfRetriever.from_dir(gold_dir)
    result = retriever.recommend(query_job_id=job_id, top_k=top_k)

    output = {
        "query_job_id": result.query_job_id,
        "results": [
            {"rank": r.rank, "job_id": r.job_id, "score": round(r.score, 6)}
            for r in result.results
        ],
    }
    click.echo(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# build-embeddings
# ---------------------------------------------------------------------------

@main.command("build-embeddings")
@click.option("--silver-path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--backend", required=True, type=str, help="Backend name, e.g. qwen3")
@click.option("--model-name", default="Qwen/Qwen3-Embedding-0.6B", show_default=True, type=str)
@click.option("--batch-size", default=32, show_default=True, type=int)
@click.option("--device", default="auto", show_default=True, type=str)
@click.option("--sample-size", default=None, type=int, help="Optional sample size for testing.")
@click.option("--log-level", default="INFO", show_default=True)
def build_embeddings_cmd(
    silver_path: Path,
    output_dir: Path,
    backend: str,
    model_name: str,
    batch_size: int,
    device: str,
    sample_size: Optional[int],
    log_level: str,
) -> None:
    """Build dense embeddings from silver data."""
    _setup_logging(log_level)
    import pandas as pd

    from jobsrec.embeddings.dense_store import build_and_save_dense

    df = pd.read_parquet(silver_path)
    if "job_card_text" not in df.columns:
        raise click.UsageError("silver Parquet does not contain 'job_card_text' column.")

    if sample_size is not None:
        df = df.head(sample_size)

    documents = df["job_card_text"].tolist()
    job_ids = df["job_id"].tolist()

    if backend == "qwen3":
        from jobsrec.embeddings.qwen3 import Qwen3EmbeddingBackend
        backend_instance = Qwen3EmbeddingBackend(model_name=model_name, device=device)
    elif backend == "fake":
        from tests.test_qwen3_backend_contract import FakeDenseEmbeddingBackend
        backend_instance = FakeDenseEmbeddingBackend(model_name=model_name)
    else:
        raise click.BadParameter(f"Unknown backend: {backend}")

    result = build_and_save_dense(
        backend=backend_instance,
        documents=documents,
        job_ids=job_ids,
        output_dir=output_dir,
        input_path=silver_path,
        batch_size=batch_size,
        sample_size=sample_size,
    )
    click.echo(
        json.dumps(
            {
                "embeddings_path": str(result.embeddings_path),
                "index_path": str(result.index_path),
                "manifest_path": str(result.manifest_path),
                "n_rows": result.n_rows,
                "embedding_dim": result.embedding_dim,
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# recommend-dense
# ---------------------------------------------------------------------------

@main.command("recommend-dense")
@click.option("--job-id", required=True, type=int)
@click.option("--top-k", default=5, show_default=True, type=int)
@click.option("--embeddings-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--silver-path", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Optional silver path.")
@click.option("--log-level", default="WARNING", show_default=True)
def recommend_dense_cmd(
    job_id: int,
    top_k: int,
    embeddings_dir: Path,
    silver_path: Optional[Path],
    log_level: str,
) -> None:
    """Return top-k similar jobs using dense embeddings."""
    _setup_logging(log_level)

    from jobsrec.recommend.dense_retrieval import DenseRetriever

    retriever = DenseRetriever.from_dir(embeddings_dir)
    result = retriever.recommend(query_job_id=job_id, top_k=top_k)

    output = {
        "query_job_id": result.query_job_id,
        "results": [
            {"rank": r.rank, "job_id": r.job_id, "score": round(r.score, 6)}
            for r in result.results
        ],
    }
    click.echo(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# temporal-audit
# ---------------------------------------------------------------------------

@main.command("temporal-audit")
@click.option("--input", "--silver-path", "silver_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-dir", default=Path("reports/temporal_audit"), show_default=True, type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--time-column",
    default="listed_time",
    show_default=True,
    type=click.Choice(["listed_time", "original_listed_time", "expiry", "closed_time"]),
)
@click.option("--log-level", default="INFO", show_default=True)
def temporal_audit_cmd(
    silver_path: Path,
    output_dir: Path,
    time_column: str,
    log_level: str,
) -> None:
    """Audit temporal coverage in a silver jobs Parquet."""
    _setup_logging(log_level)

    from jobsrec.trends.temporal import run_temporal_audit

    command_used = (
        "python -m jobsrec.cli temporal-audit "
        f"--input {silver_path} "
        f"--output-dir {output_dir} "
        f"--time-column {time_column}"
    )
    result = run_temporal_audit(
        silver_path=silver_path,
        output_dir=output_dir,
        command_used=command_used,
        time_column=time_column,
    )
    click.echo(
        json.dumps(
            {
                "report_path": str(result.report_path),
                "summary_path": str(result.summary_path),
                "monthly_counts_path": str(result.monthly_counts_path),
                "weekly_counts_path": str(result.weekly_counts_path),
                "daily_counts_path": str(result.daily_counts_path),
                "temporal_coverage_path": str(result.temporal_coverage_path),
                "total_rows": result.summary["total_rows"],
                "valid_time_rows": result.summary["valid_time_rows"],
                "valid_listed_time_rows": result.summary["valid_listed_time_rows"],
                "number_of_months": result.summary["number_of_months"],
                "time_column": result.summary["time_column"],
                "reliability_label": result.summary["reliability_label"],
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# temporal-demo
# ---------------------------------------------------------------------------

@main.command("temporal-demo")
@click.option("--input", "--silver-path", "silver_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--figures-dir", default=None, type=click.Path(file_okay=False, path_type=Path))
@click.option("--report-path", default=None, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--sample-size", default=10000, show_default=True, type=int)
@click.option(
    "--sampling-mode",
    default="temporal-stride",
    show_default=True,
    type=click.Choice(["temporal-stride", "random", "head"]),
)
@click.option("--representation", default="tfidf_svd", show_default=True, type=click.Choice(["tfidf_svd", "semantic_embeddings"]))
@click.option("--embedding-backend", default="mock", show_default=True, type=click.Choice(["mock", "existing_qwen3", "qwen3"]))
@click.option("--embedding-model", default="deterministic-mock", show_default=True, type=str)
@click.option("--embedding-batch-size", default=16, show_default=True, type=int)
@click.option("--embedding-cache-dir", default=Path("data/cache/embeddings"), show_default=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--device", default="cpu", show_default=True, type=click.Choice(["cpu", "cuda", "auto"]))
@click.option("--max-embedding-rows", default=1000, show_default=True, type=int)
@click.option(
    "--time-column",
    default="listed_time",
    show_default=True,
    type=click.Choice(["listed_time", "original_listed_time", "expiry", "closed_time"]),
)
@click.option("--time-bin", default="M", show_default=True, type=click.Choice(["H", "D", "W", "M"], case_sensitive=False))
@click.option(
    "--centroid-weighting",
    default="unweighted",
    show_default=True,
    type=click.Choice(["unweighted", "salary", "both"]),
)
@click.option("--log-level", default="INFO", show_default=True)
def temporal_demo_cmd(
    silver_path: Path,
    output_dir: Path,
    figures_dir: Optional[Path],
    report_path: Optional[Path],
    sample_size: int,
    sampling_mode: str,
    representation: str,
    embedding_backend: str,
    embedding_model: str,
    embedding_batch_size: int,
    embedding_cache_dir: Path,
    device: str,
    max_embedding_rows: int,
    time_column: str,
    time_bin: str,
    centroid_weighting: str,
    log_level: str,
) -> None:
    """Build a fast temporal analytics demo over silver job postings."""
    _setup_logging(log_level)

    from jobsrec.trends.temporal import run_temporal_demo

    figures_dir = figures_dir or (output_dir / "figures")
    report_path = report_path or (output_dir / "report.md")
    command_used = (
        "python -m jobsrec.cli temporal-demo "
        f"--input {silver_path} "
        f"--output-dir {output_dir} "
        f"--figures-dir {figures_dir} "
        f"--report-path {report_path} "
        f"--sample-size {sample_size} "
        f"--sampling-mode {sampling_mode} "
        f"--representation {representation} "
        f"--embedding-backend {embedding_backend} "
        f"--embedding-model {embedding_model} "
        f"--embedding-batch-size {embedding_batch_size} "
        f"--embedding-cache-dir {embedding_cache_dir} "
        f"--device {device} "
        f"--max-embedding-rows {max_embedding_rows} "
        f"--time-column {time_column} "
        f"--time-bin {time_bin} "
        f"--centroid-weighting {centroid_weighting}"
    )
    result = run_temporal_demo(
        silver_path=silver_path,
        output_dir=output_dir,
        figures_dir=figures_dir,
        report_path=report_path,
        sample_size=sample_size,
        sampling_mode=sampling_mode,
        command_used=command_used,
        representation=representation,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        embedding_batch_size=embedding_batch_size,
        embedding_cache_dir=embedding_cache_dir,
        device=device,
        max_embedding_rows=max_embedding_rows,
        time_column=time_column,
        time_bin=time_bin.upper(),
        centroid_weighting=centroid_weighting,
    )
    click.echo(
        json.dumps(
            {
                "monthly_drift_path": str(result.monthly_drift_path),
                "skill_growth_path": str(result.skill_growth_path),
                "manifest_path": str(result.manifest_path),
                "report_path": str(result.report_path),
                "n_rows_selected": result.manifest["n_rows_selected"],
                "n_months": result.manifest["n_months"],
                "representation": result.manifest["representation"],
                "time_column": result.manifest["time_column"],
                "time_bin": result.manifest["time_bin"],
                "centroid_weighting": result.manifest["centroid_weighting"],
                "salary_weighted_drift_path": result.manifest["salary_weighted_drift_path"],
                "reliability_label": result.manifest["reliability_label"],
            },
            indent=2,
        )
    )

# ---------------------------------------------------------------------------
# temporal-clusters
# ---------------------------------------------------------------------------

@main.command("temporal-clusters")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--outdir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--bin", "bin_size", default="D", show_default=True, type=click.Choice(["D", "W", "M"], case_sensitive=False))
@click.option("--k", default=12, show_default=True, type=int)
@click.option("--embedding", default="tfidf_svd", show_default=True, type=click.Choice(["tfidf_svd", "sentence_transformers"]))
@click.option("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2", show_default=True, type=str)
@click.option("--embedding-batch-size", default=8, show_default=True, type=int)
@click.option("--max-rows", default=100000, show_default=True, type=int)
@click.option("--random-state", default=42, show_default=True, type=int)
@click.option("--time-column", default=None, type=str, help="Optional temporal column override.")
@click.option("--log-level", default="INFO", show_default=True)
def temporal_clusters_cmd(
    input_path: Path,
    outdir: Path,
    bin_size: str,
    k: int,
    embedding: str,
    embedding_model: str,
    embedding_batch_size: int,
    max_rows: int,
    random_state: int,
    time_column: Optional[str],
    log_level: str,
) -> None:
    """Build fixed temporal cluster analytics over silver job postings."""
    _setup_logging(log_level)

    from jobsrec.trends.temporal_clusters import run_temporal_clusters

    command_parts = [
        "jobsrec temporal-clusters",
        f"--input {input_path}",
        f"--outdir {outdir}",
        f"--bin {bin_size}",
        f"--k {k}",
        f"--embedding {embedding}",
        f"--embedding-model {embedding_model}",
        f"--embedding-batch-size {embedding_batch_size}",
        f"--max-rows {max_rows}",
        f"--random-state {random_state}",
    ]
    if time_column:
        command_parts.append(f"--time-column {time_column}")
    command_used = " ".join(command_parts)
    result = run_temporal_clusters(
        input_path=input_path,
        outdir=outdir,
        bin_size=bin_size.upper(),
        k=k,
        embedding=embedding,
        max_rows=max_rows,
        random_state=random_state,
        command_used=command_used,
        embedding_model=embedding_model,
        embedding_batch_size=embedding_batch_size,
        time_column=time_column,
    )
    click.echo(
        json.dumps(
            {
                "metrics_path": str(result.metrics_path),
                "manifest_path": str(result.manifest_path),
                "report_path": str(result.report_path),
                "selected_row_count": result.manifest["selected_row_count"],
                "k_effective": result.manifest["k_effective"],
                "salary_available": result.manifest["salary_available"],
                "decay_available": result.manifest["decay_available"],
                "generated_files": result.generated_files,
            },
            indent=2,
        )
    )

# ---------------------------------------------------------------------------
# profile-silver
# ---------------------------------------------------------------------------

@main.command("profile-silver")
@click.option(
    "--silver-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the silver jobs.parquet file.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Destination JSON file for the data profile.",
)
@click.option("--log-level", default="INFO", show_default=True)
def profile_silver_cmd(
    silver_path: Path,
    output: Path,
    log_level: str,
) -> None:
    """Profile a silver Parquet and write a JSON data report."""
    _setup_logging(log_level)

    from jobsrec.data.profile import profile_silver_from_path

    profile = profile_silver_from_path(silver_path)
    profile_dict = profile.to_dict()

    # Add metadata
    from datetime import datetime, timezone

    profile_dict["_meta"] = {
        "silver_path": str(silver_path),
        "profiled_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile_dict, indent=2))
    click.echo(f"Profile written to: {output}")
    click.echo(f"  n_postings          : {profile.n_postings}")
    click.echo(f"  n_unique_job_ids    : {profile.n_unique_job_ids}")
    click.echo(f"  n_missing_titles    : {profile.n_missing_titles}")
    click.echo(f"  n_jobs_without_skills: {profile.n_jobs_without_skills}")
    click.echo(f"  n_unique_skills     : {profile.n_unique_skills}")
    if profile.listed_time_parse_rate is not None:
        rate_pct = round(profile.listed_time_parse_rate * 100, 1)
        click.echo(f"  listed_time parse rate: {rate_pct}%")


# ---------------------------------------------------------------------------
# skill-evolution
# ---------------------------------------------------------------------------

@main.command("skill-evolution")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--outdir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--bin", "bin_size", default="W", show_default=True, type=click.Choice(["D", "W", "M"], case_sensitive=False))
@click.option("--top-n", "top_n_skills", default=12, show_default=True, type=int)
@click.option("--max-rows", default=100000, show_default=True, type=int)
@click.option("--random-state", default=42, show_default=True, type=int)
@click.option("--time-column", default=None, type=str, help="Optional temporal column override.")
@click.option("--confidence-threshold", default=0.05, show_default=True, type=float)
@click.option("--margin-threshold", default=0.01, show_default=True, type=float)
@click.option("--log-level", default="INFO", show_default=True)
def skill_evolution_cmd(
    input_path: Path,
    outdir: Path,
    bin_size: str,
    top_n_skills: int,
    max_rows: int,
    random_state: int,
    time_column: Optional[str],
    confidence_threshold: float,
    margin_threshold: float,
    log_level: str,
) -> None:
    """Build offline skill-share evolution analytics report and plots."""
    _setup_logging(log_level)

    from jobsrec.trends.skill_evolution import run_skill_evolution

    command_parts = [
        "jobsrec skill-evolution",
        f"--input {input_path}",
        f"--outdir {outdir}",
        f"--bin {bin_size}",
        f"--top-n {top_n_skills}",
        f"--max-rows {max_rows}",
        f"--random-state {random_state}",
        f"--confidence-threshold {confidence_threshold}",
        f"--margin-threshold {margin_threshold}",
    ]
    if time_column:
        command_parts.insert(-2, f"--time-column {time_column}")
    command_used = " ".join(command_parts)
    result = run_skill_evolution(
        input_path=input_path,
        outdir=outdir,
        bin_size=bin_size.upper(),
        top_n_skills=top_n_skills,
        max_rows=max_rows,
        random_state=random_state,
        time_column=time_column,
        confidence_threshold=confidence_threshold,
        margin_threshold=margin_threshold,
        command_used=command_used,
    )
    click.echo(
        json.dumps(
            {
                "manifest_path": str(result.manifest_path),
                "report_path": str(result.report_path),
                "domain_skill_monthly_path": str(result.domain_skill_monthly_path),
                "generated_files": result.generated_files,
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# populares-validate
# ---------------------------------------------------------------------------

@main.command("populares-validate")
@click.option(
    "--documents",
    "documents_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to documents.parquet from populares-scraper.",
)
@click.option(
    "--chunks",
    "chunks_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to chunks.parquet from populares-scraper.",
)
@click.option(
    "--manifest",
    "manifest_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional embedding_manifest.json contract file.",
)
def populares_validate_cmd(
    documents_path: Path,
    chunks_path: Path,
    manifest_path: Optional[Path],
) -> None:
    """Validate populares-scraper inputs and print a JSON summary."""
    from jobsrec.ingest.populares import summarize_populares_inputs

    summary = summarize_populares_inputs(
        documents_path=documents_path,
        chunks_path=chunks_path,
        manifest_path=manifest_path,
    )
    click.echo(json.dumps(summary, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# populares-build-gold
# ---------------------------------------------------------------------------

@main.command("populares-build-gold")
@click.option(
    "--documents",
    "documents_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to documents.parquet from populares-scraper.",
)
@click.option(
    "--chunks",
    "chunks_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to chunks.parquet from populares-scraper.",
)
@click.option(
    "--manifest",
    "manifest_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Optional embedding_manifest.json contract file.",
)
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Destination gold dataset directory.",
)
def populares_build_gold_cmd(
    documents_path: Path,
    chunks_path: Path,
    manifest_path: Optional[Path],
    out_dir: Path,
) -> None:
    """Build draft gold dataset files for apolo-rag."""
    from jobsrec.ingest.populares import build_populares_gold

    manifest = build_populares_gold(
        documents_path=documents_path,
        chunks_path=chunks_path,
        manifest_path=manifest_path,
        out_dir=out_dir,
    )
    click.echo(json.dumps(manifest, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Entry-point guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
