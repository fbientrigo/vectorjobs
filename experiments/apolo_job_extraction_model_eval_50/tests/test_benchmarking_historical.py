import hashlib
import inspect
import json

from benchmarking import historical_baselines, loaders


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_historical_rows_are_matched_by_job_id(tmp_path):
    jsonl_path = tmp_path / "hist.jsonl"
    rows = [
        {"job_id": "1", "apolo_extraction": {"skills": [], "schema_version": "0.1.0"}},
        {"job_id": "2", "apolo_extraction": {"skills": [], "schema_version": "0.1.0"}},
        {"job_id": "999", "apolo_extraction": {"skills": [], "schema_version": "0.1.0"}},
    ]
    jsonl_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    original = historical_baselines.HISTORICAL_JSONL
    historical_baselines.HISTORICAL_JSONL = jsonl_path
    try:
        overlap = historical_baselines.load_overlap_rows(["1", "2", "3"])
    finally:
        historical_baselines.HISTORICAL_JSONL = original

    assert set(overlap) == {"1", "2"}
    assert overlap["1"]["job_id"] == "1"


def test_gemini_is_never_executed():
    src = inspect.getsource(historical_baselines)
    assert "requests" not in src
    assert "generativelanguage.googleapis.com" not in src
    assert "openrouter.ai" not in src


def test_unknown_gemini_provenance_is_clearly_labelled():
    report = historical_baselines.build_baseline_report(loaders.load_frozen_job_ids())
    assert report["provenance_status"] == "unknown"
    assert report["reasoning_mode"] == "unknown"
    assert report["model_configuration_status"] == "unknown"
    assert report["directly_comparable"] is False


def test_inspection_model_identity_not_assumed_as_generator_identity():
    report = historical_baselines.build_baseline_report(loaders.load_frozen_job_ids())
    assert "not evidence of model identity" in report["comparability_notes"].lower()


def test_frozen_50_overlap_matches_real_dataset():
    frozen_job_ids = loaders.load_frozen_job_ids()
    report = historical_baselines.build_baseline_report(frozen_job_ids)
    assert report["frozen_50_overlap"] == 50
    assert report["frozen_50_missing_job_ids"] == []


def test_historical_files_remain_unchanged_after_read(tmp_path):
    before_jsonl = sha256_file(historical_baselines.HISTORICAL_JSONL)
    before_manifest = sha256_file(historical_baselines.HISTORICAL_MANIFEST)

    historical_baselines.build_baseline_report(loaders.load_frozen_job_ids())

    assert sha256_file(historical_baselines.HISTORICAL_JSONL) == before_jsonl
    assert sha256_file(historical_baselines.HISTORICAL_MANIFEST) == before_manifest
