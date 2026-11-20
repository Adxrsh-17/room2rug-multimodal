from pathlib import Path

from src.utils.tracking import ExperimentTracker


def test_experiment_tracker_local(tmp_path: Path):
    tracker = ExperimentTracker(experiment_name="test_experiment", tracking_uri=str(tmp_path / "mlruns"))
    tracker.start_run(run_name="smoke_test_run")
    tracker.log_param("learning_rate", 0.001)
    tracker.log_params({"batch_size": 32, "model": "clip"})
    tracker.log_metric("ndcg@10", 0.75)
    tracker.log_metrics({"mrr@10": 0.88, "precision@5": 0.90})

    run_output_file = tmp_path / "run_summary.json"
    data = tracker.end_run(save_local_path=run_output_file)

    assert data["params"]["learning_rate"] == 0.001
    assert data["params"]["batch_size"] == 32
    assert data["metrics"]["ndcg@10"] == 0.75
    assert run_output_file.exists()
