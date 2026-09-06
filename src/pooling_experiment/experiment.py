from pathlib import Path
from typing import Any

import torch
from torch import nn

from .data import create_cifar10_dataloaders
from .models import CIFAR10CNN
from .reproducibility import set_seed
from .tracking import log_run_results, start_run
from .training import evaluate, fit


def run_experiment(
    pooling_type: str,
    seed: int,
    data_root: str | Path = "data",
    checkpoint_path: str | Path | None = None,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
    device: str | torch.device | None = None,
    experiment_name: str = "cifar10_pooling",
    tracking_uri: str = "sqlite:///mlflow.db",
) -> dict[str, Any]:
    """Train one pooling variant on CIFAR-10 and return its results."""
    set_seed(seed)
    with start_run(experiment_name, tracking_uri, f"{pooling_type}_seed_{seed}") as run:
        selected_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        train_loader, validation_loader, test_loader = create_cifar10_dataloaders(
            root=data_root,
            batch_size=batch_size,
            seed=seed,
            num_workers=num_workers,
            pin_memory=selected_device.type == "cuda",
        )
        model = CIFAR10CNN(pooling_type=pooling_type).to(selected_device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        history = fit(
            model,
            train_loader,
            validation_loader,
            criterion,
            optimizer,
            epochs=epochs,
            device=selected_device,
            best_model_path=checkpoint_path,
        )
        test_metrics = evaluate(model, test_loader, criterion, selected_device)
        params = {
            "pooling_type": pooling_type,
            "seed": seed,
            "device": str(selected_device),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        }
        log_run_results(params, history, test_metrics, checkpoint_path)
        return {
            **params,
            "run_id": run.info.run_id,
            "history": history,
            "test_metrics": test_metrics,
        }
