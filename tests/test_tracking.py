import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from pooling_experiment.tracking import (
    log_epoch_metrics,
    log_run_results,
    log_step_metrics,
    start_run,
)


def test_tracking_logs_params_metrics_and_artifact(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")

    with start_run("test_experiment", tracking_uri, "test_run"):
        log_step_metrics(0.6, step=3)
        log_epoch_metrics({"train_loss": 0.5}, epoch=1)
        log_run_results(
            {"pooling_type": "max", "seed": 0},
            [{"epoch": 1.0, "train_loss": 0.5}],
            {"loss": 0.4, "accuracy": 0.8},
            checkpoint,
        )
        run_id = mlflow.active_run().info.run_id

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    assert run.data.params["pooling_type"] == "max"
    assert run.data.metrics["test_accuracy"] == 0.8
    assert run.data.metrics["step_loss"] == 0.6
    assert run.data.metrics["train_loss"] == 0.5
    assert client.list_artifacts(run_id, "checkpoints")
