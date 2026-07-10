#!/usr/bin/env python
"""Modular Python pipeline script to build silver, extract candidates, and build agent labeling pack."""

import argparse
import logging
import sys
import yaml
from pathlib import Path
import pandas as pd

from jobsrec.data.load import build_silver
from jobsrec.extract.candidates import (
    build_extraction_candidates,
    build_extraction_report,
    write_extraction_manifest,
)
from jobsrec.extract.agent_pack import (
    build_agent_labeling_pack,
    JSON_SCHEMA,
    PROMPT_MARKDOWN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("run_pipeline")

def main():
    parser = argparse.ArgumentParser(description="Run vectorjobs pipeline end-to-end.")
    parser.add_argument("--input-db", type=str, default="data/bronze/jobs.db", help="Path to LinkedIn scraped jobs.db SQLite database")
    parser.add_argument("--output-dir", type=str, default="data/silver", help="Output directory for processed silver data and candidates")
    parser.add_argument("--config", type=str, default="configs/local_6gb.yaml", help="Path to config YAML")
    parser.add_argument("--sample-size", type=int, default=800, help="Sample size for agent labeling pack")
    parser.add_argument("--random-seed", type=int, default=3407, help="Random seed for deterministic stratification")
    
    args = parser.parse_args()
    
    input_db = Path(args.input_db)
    output_dir = Path(args.output_dir)
    config_path = Path(args.config)
    
    if not input_db.exists():
        logger.error(f"Input DB not found at {input_db}")
        sys.exit(1)
        
    config = {}
    if config_path.exists():
        logger.info(f"Loading config from {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
    # Step 1: Run build-silver
    logger.info(f"Step 1: Running build_silver from {input_db} -> {output_dir}")
    silver_result = build_silver(input_db=input_db, output_dir=output_dir, config=config)
    logger.info(f"Successfully wrote silver jobs.parquet: {silver_result.output_path} ({silver_result.output_rows} rows)")
    
    # Step 2: Run extract-candidates
    logger.info("Step 2: Extracting candidates from silver Parquet")
    silver_df = pd.read_parquet(silver_result.output_path)
    parse_error_count = int(silver_df["company_parse_error"].sum()) if "company_parse_error" in silver_df.columns else 0
    candidates_df = build_extraction_candidates(silver_df)
    
    candidates_path = output_dir / "job_extraction_candidates.parquet"
    candidates_df.to_parquet(candidates_path, index=False)
    logger.info(f"Successfully wrote candidates: {candidates_path} ({len(candidates_df)} rows)")
    
    report = build_extraction_report(candidates_df, silver_df, parse_error_count)
    manifest_path = write_extraction_manifest(
        output_dir=output_dir,
        report=report,
        silver_path=str(silver_result.output_path),
        output_path=str(candidates_path),
    )
    logger.info(f"Wrote extraction manifest: {manifest_path}")
    
    # Step 3: Run build-agent-labeling-pack
    logger.info("Step 3: Building agent labeling pack")
    agent_output_dir = output_dir / "agent_labeling"
    agent_output_dir.mkdir(parents=True, exist_ok=True)
    
    rows, manifest = build_agent_labeling_pack(
        silver=silver_df,
        candidates=candidates_df,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
    )
    
    import json
    
    jsonl_path = agent_output_dir / "agent_labeling_pack.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Wrote labeling pack: {jsonl_path} ({len(rows)} rows)")
    
    manifest_path = agent_output_dir / "agent_labeling_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote labeling manifest: {manifest_path}")
    
    prompt_path = agent_output_dir / "agent_labeling_prompt.md"
    prompt_path.write_text(PROMPT_MARKDOWN, encoding="utf-8")
    logger.info(f"Wrote prompt guidelines: {prompt_path}")
    
    schema_path = agent_output_dir / "agent_labeling_schema.json"
    with schema_path.open("w", encoding="utf-8") as f:
        json.dump(JSON_SCHEMA, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote schema JSON: {schema_path}")
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
