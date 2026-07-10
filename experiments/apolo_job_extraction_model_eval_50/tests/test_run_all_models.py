import json
import subprocess

import pytest

import run_all_models


def _block_subprocess(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called during --dry-run")

    monkeypatch.setattr(subprocess, "run", _boom)


def test_dry_run_never_calls_subprocess(tmp_path, monkeypatch):
    models_config = tmp_path / "models.json"
    models_config.write_text(json.dumps(["fake/model-a", "fake/model-b"]), encoding="utf-8")
    outputs_dir = tmp_path / "outputs"
    reports_dir = tmp_path / "reports"
    outputs_dir.mkdir()

    monkeypatch.setattr(run_all_models, "MODELS_CONFIG_PATH", models_config)
    monkeypatch.setattr(run_all_models, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(run_all_models, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(run_all_models, "SAMPLE_MANIFEST_PATH", tmp_path / "does_not_exist.json")
    monkeypatch.setattr("sys.argv", ["run_all_models.py", "--dry-run"])
    _block_subprocess(monkeypatch)

    run_all_models.main()

    plan = json.loads((reports_dir / "run_all_models_plan.json").read_text(encoding="utf-8"))
    assert [entry["model"] for entry in plan["plan"]] == ["fake/model-a", "fake/model-b"]
    assert all(entry["action"] == "run" for entry in plan["plan"])


def test_budget_stops_before_exceeding(tmp_path, monkeypatch):
    models_config = tmp_path / "models.json"
    models_config.write_text(json.dumps(["fake/model-a", "fake/model-b"]), encoding="utf-8")
    outputs_dir = tmp_path / "outputs"
    reports_dir = tmp_path / "reports"
    outputs_dir.mkdir()
    # A prior completed model establishes a cost-per-row estimate of $1/row.
    prior_dir = outputs_dir / "fake__model-prior"
    prior_dir.mkdir()
    (prior_dir / "manifest.json").write_text(
        json.dumps({"model": "fake/model-prior", "n_rows_completed": 1, "n_rows_requested": 1, "estimated_cost_usd": 1.0}),
        encoding="utf-8",
    )

    monkeypatch.setattr(run_all_models, "MODELS_CONFIG_PATH", models_config)
    monkeypatch.setattr(run_all_models, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(run_all_models, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(run_all_models, "SAMPLE_MANIFEST_PATH", tmp_path / "does_not_exist.json")
    monkeypatch.setattr("sys.argv", ["run_all_models.py", "--dry-run", "--budget-usd", "1.0"])
    _block_subprocess(monkeypatch)

    run_all_models.main()

    plan = json.loads((reports_dir / "run_all_models_plan.json").read_text(encoding="utf-8"))
    actions = {entry["model"]: entry["action"] for entry in plan["plan"]}
    assert actions["fake/model-a"] == "stop_budget"
