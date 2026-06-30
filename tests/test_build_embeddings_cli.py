import json
from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from jobsrec.cli import main


def test_build_embeddings_and_recommend_dense_cli(tmp_path: Path):
    runner = CliRunner()
    
    # Create fake silver data
    silver_path = tmp_path / "jobs.parquet"
    df = pd.DataFrame({
        "job_id": ["1", "2", "3"],
        "job_card_text": ["Job A", "Job B", "Job C"]
    })
    df.to_parquet(silver_path)
    
    output_dir = tmp_path / "gold_dense"
    
    # Test build-embeddings
    result = runner.invoke(main, [
        "build-embeddings",
        "--silver-path", str(silver_path),
        "--output-dir", str(output_dir),
        "--backend", "fake"
    ])
    
    assert result.exit_code == 0
    
    assert (output_dir / "job_embeddings.npy").exists()
    assert (output_dir / "job_ids.parquet").exists()
    assert (output_dir / "embedding_manifest.json").exists()
    
    # Test recommend-dense
    res2 = runner.invoke(main, [
        "recommend-dense",
        "--job-id", "1",
        "--embeddings-dir", str(output_dir),
        "--top-k", "2"
    ])
    
    assert res2.exit_code == 0
    output_json = json.loads(res2.output)
    assert output_json["query_job_id"] == "1"
    assert len(output_json["results"]) <= 2
    for r in output_json["results"]:
        assert r["job_id"] != "1"
