from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .metrics import accuracy, macro_f1
from .tracking import log_epoch_metrics, log_step_metrics


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer | None,
    step_offset: int = 0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    all_predictions: list[Tensor] = []
    all_targets: list[Tensor] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if training:
                loss.backward()
                optimizer.step()
                log_step_metrics(loss.item(), step_offset + batch_index + 1)

            sample_count = targets.size(0)
            total_loss += loss.item() * sample_count
            total_samples += sample_count
            all_predictions.append(outputs.detach().cpu())
            all_targets.append(targets.detach().cpu())

    if total_samples == 0:
        raise ValueError("loader must not be empty")
    predictions = torch.cat(all_predictions)
    collected_targets = torch.cat(all_targets)
    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy(predictions, collected_targets),
        "macro_f1": macro_f1(
            predictions, collected_targets, predictions.size(1)
        ),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    step_offset: int = 0,
) -> dict[str, float]:
    """Train the model for one epoch and return aggregate metrics."""
    return _run_epoch(
        model,
        loader,
        criterion,
        device,
        optimizer,
        step_offset=step_offset,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the model without updating its parameters."""
    return _run_epoch(model, loader, criterion, device, optimizer=None)


def fit(
    model: nn.Module,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: Optimizer,
    epochs: int,
    device: torch.device,
    scheduler: Any = None,
    best_model_path: str | Path | None = None,
) -> list[dict[str, float]]:
    """Train a model and optionally save the checkpoint with best validation loss."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    step_offset = 0
    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            step_offset=step_offset,
        )
        step_offset += len(train_loader)
        validation_metrics = evaluate(
            model, validation_loader, criterion, device
        )
        if scheduler is not None:
            scheduler.step()

        record = {
            "epoch": float(epoch + 1),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            },
        }
        history.append(record)
        log_epoch_metrics(
            {f"train_{key}": value for key, value in train_metrics.items()},
            epoch + 1,
        )
        log_epoch_metrics(
            {
                f"validation_{key}": value
                for key, value in validation_metrics.items()
            },
            epoch + 1,
        )
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_state = deepcopy(model.state_dict())
            if best_model_path is not None:
                path = Path(best_model_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, path)

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
