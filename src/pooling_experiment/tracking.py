from pathlib import Path
from typing import Any

import mlflow


def start_run(
    experiment_name: str,
    tracking_uri: str = "sqlite:///mlflow.db",
    run_name: str | None = None,
) -> Any:
    """Start an MLflow run in the configured tracking store."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def log_run_results(
    params: dict[str, Any],
    history: list[dict[str, float]],
    test_metrics: dict[str, float],
    checkpoint_path: str | Path | None = None,
) -> None:
    """Log experiment parameters, epoch metrics, final metrics, and checkpoint."""
    mlflow.log_params({key: str(value) for key, value in params.items()})
    mlflow.log_metrics(
        {f"test_{key}": value for key, value in test_metrics.items()}
    )
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")


def log_step_metrics(loss: float, step: int) -> None:
    """Log the current batch loss using a global training step."""
    if mlflow.active_run() is None:
        return
    mlflow.log_metric("step_loss", loss, step=step)


def log_epoch_metrics(metrics: dict[str, float], epoch: int) -> None:
    """Log aggregate train or validation metrics for one epoch."""
    if mlflow.active_run() is None:
        return
    mlflow.log_metrics(metrics, step=epoch)
