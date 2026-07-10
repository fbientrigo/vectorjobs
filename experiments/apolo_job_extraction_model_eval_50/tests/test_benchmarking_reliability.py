from benchmarking import reliability


def test_reliability_uses_all_50_target_jobs_regardless_of_manifest():
    # Manifest doesn't even mention 50; n_jobs_target still drives every rate.
    manifest = {"n_rows_completed": 46, "finished_at": "t"}
    metrics = reliability.reliability_metrics(manifest, n_jobs_target=50)
    assert metrics["n_jobs_target"] == 50
    assert metrics["single_run_valid_rate"] == 46 / 50


def test_46_of_50_gives_092_single_run_valid_rate():
    manifest = {"n_rows_requested": 50, "n_rows_completed": 46, "finished_at": "t"}
    metrics = reliability.reliability_metrics(manifest, n_jobs_target=50)
    assert metrics["single_run_valid_count"] == 46
    assert metrics["single_run_valid_rate"] == 0.92
    assert metrics["run_status"] == reliability.RUN_STATUS_PARTIAL


def test_45_of_50_gives_090_single_run_valid_rate():
    manifest = {"n_rows_requested": 50, "n_rows_completed": 45, "finished_at": "t"}
    metrics = reliability.reliability_metrics(manifest, n_jobs_target=50)
    assert metrics["single_run_valid_rate"] == 0.90


def test_50_of_50_is_process_finished_valid_50():
    manifest = {"n_rows_requested": 50, "n_rows_completed": 50, "finished_at": "t"}
    metrics = reliability.reliability_metrics(manifest, n_jobs_target=50)
    assert metrics["run_status"] == reliability.RUN_STATUS_VALID_ALL
    assert metrics["single_run_valid_rate"] == 1.0


def test_no_attempt_log_leaves_attempt_level_fields_null():
    manifest = {"n_rows_requested": 50, "n_rows_completed": 46, "finished_at": "t"}
    metrics = reliability.reliability_metrics(manifest, n_jobs_target=50)
    assert metrics["attempt_level_data_available"] is False
    for key in ("n_valid_first_pass", "total_request_attempts", "total_retry_attempts", "retry_recovered_jobs"):
        assert metrics[key] is None


def test_retry_recovery_does_not_overwrite_first_pass_metrics():
    manifest = {"n_rows_requested": 50, "n_rows_completed": 46, "finished_at": "t"}
    attempt_log = {
        "initial_segment_valid_count": 46,
        "cumulative_valid_count": 48,
        "total_request_attempts": 58,
        "total_retry_attempts": 8,
        "retry_recovered_jobs": 2,
    }
    metrics = reliability.reliability_metrics(manifest, n_jobs_target=50, attempt_log=attempt_log)

    assert metrics["n_valid_first_pass"] == 46
    assert metrics["first_pass_valid_rate"] == 0.92
    assert metrics["n_eventually_valid"] == 48
    assert metrics["eventual_valid_rate"] == 0.96
    assert metrics["attempt_level_data_available"] is True


def test_attempts_per_valid_job_includes_retries():
    manifest = {"n_rows_requested": 50, "n_rows_completed": 46, "finished_at": "t"}
    attempt_log = {
        "initial_segment_valid_count": 46,
        "cumulative_valid_count": 48,
        "total_request_attempts": 58,
        "total_retry_attempts": 8,
        "retry_recovered_jobs": 2,
    }
    metrics = reliability.reliability_metrics(manifest, n_jobs_target=50, attempt_log=attempt_log)
    assert metrics["attempts_per_valid_job"] == 58 / 48


def test_run_status_interrupted_when_unfinished():
    manifest = {"n_rows_requested": 50, "n_rows_completed": 10}
    assert reliability.run_status(manifest, 50) == reliability.RUN_STATUS_INTERRUPTED


def test_run_status_budget_stopped_when_fewer_requested_than_target():
    manifest = {"n_rows_requested": 20, "n_rows_completed": 20, "finished_at": "t"}
    assert reliability.run_status(manifest, 50) == reliability.RUN_STATUS_BUDGET_STOPPED
