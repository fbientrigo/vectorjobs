import json

import compare_models


def base_manifest(**overrides) -> dict:
    manifest = {
        "model": "fake/model-a",
        "n_rows_requested": 2,
        "n_rows_completed": 2,
        "request_failures": 0,
        "json_parse_failures": 0,
        "schema_failures": 0,
        "validation_failure_count": 0,
        "evidence_substring_failures": 0,
        "total_skills": 4,
        "skills_per_job_mean": 2.0,
        "total_education_requirements": 1,
        "seniority_counts": {"unknown": 1, "senior": 1},
        "confidence_mean": 0.9,
        "confidence_1_rate": 0.1,
        "broad_domain_skill_rate": 0.2,
        "empty_skill_rows": 0,
        "total_tokens": 1000,
        "estimated_cost_usd": 0.001,
        "wall_time_seconds": 4.0,
        "latency_seconds": 2.0,
        "prompt_hash": "abc",
        "sample_hash": "def",
    }
    manifest.update(overrides)
    return manifest


def write_model_output(outputs_dir, model_dirname, manifest, skills_per_row=(2, 2)):
    model_dir = outputs_dir / model_dirname
    model_dir.mkdir(parents=True)
    (model_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (model_dir / "extractions.jsonl").open("w", encoding="utf-8") as f:
        for i, n_skills in enumerate(skills_per_row):
            row = {
                "job_id": str(i),
                "apolo_extraction": {"skills": [{"name": f"s{j}"} for j in range(n_skills)]},
            }
            f.write(json.dumps(row) + "\n")
    (model_dir / "raw_responses.jsonl").write_text('{"job_id": "0"}\n', encoding="utf-8")


def test_compare_models_does_not_fail_with_one_model(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "outputs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(compare_models, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(compare_models, "REPORTS_DIR", reports_dir)

    write_model_output(outputs_dir, "fake__model-a", base_manifest())

    compare_models.main()

    assert (reports_dir / "model_comparison.csv").exists()
    assert (reports_dir / "model_comparison.md").exists()
    assert (reports_dir / "model_comparison_ranked.md").exists()
    # Pareto report is skipped with < 2 gate-passing models, not an error.
    assert not (reports_dir / "model_comparison_pareto.md").exists()


def test_score_penalizes_evidence_failures():
    passing = base_manifest()
    failing = base_manifest(evidence_substring_failures=1)

    def to_row(manifest):
        row = {
            "n_rows_completed": manifest["n_rows_completed"],
            "empty_skill_rows": manifest["empty_skill_rows"],
            "broad_domain_skill_rate": manifest["broad_domain_skill_rate"],
            "confidence_1_rate": manifest["confidence_1_rate"],
            "cost_per_completed_row": manifest["estimated_cost_usd"] / manifest["n_rows_completed"],
            "seconds_per_completed_row": manifest["wall_time_seconds"] / manifest["n_rows_completed"],
        }
        row["hard_gate_pass"] = compare_models.hard_gate_pass(manifest)
        return row

    passing_score = compare_models.operational_candidate_score(to_row(passing))
    failing_score = compare_models.operational_candidate_score(to_row(failing))

    assert passing_score > 0
    assert failing_score == 0.0
