import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from pooling_experiment.training import evaluate, fit, train_one_epoch


def _loader() -> DataLoader[TensorDataset]:
    inputs = torch.randn(12, 4)
    targets = (inputs[:, 0] > 0).long()
    return DataLoader(TensorDataset(inputs, targets), batch_size=4)


def _model() -> nn.Module:
    return nn.Linear(4, 2)


def test_train_one_epoch_returns_metrics() -> None:
    model = _model()
    metrics = train_one_epoch(
        model,
        _loader(),
        nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        torch.device("cpu"),
    )

    assert set(metrics) == {"loss", "accuracy", "macro_f1"}
    assert metrics["loss"] >= 0
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1


def test_evaluate_does_not_update_parameters() -> None:
    model = _model()
    before = [parameter.detach().clone() for parameter in model.parameters()]

    evaluate(model, _loader(), nn.CrossEntropyLoss(), torch.device("cpu"))

    assert all(
        torch.equal(parameter, previous)
        for parameter, previous in zip(model.parameters(), before)
    )


def test_fit_returns_history_and_saves_best_checkpoint(tmp_path: Path) -> None:
    model = _model()
    checkpoint = tmp_path / "best.pt"
    history = fit(
        model,
        _loader(),
        _loader(),
        nn.CrossEntropyLoss(),
        torch.optim.SGD(model.parameters(), lr=0.1),
        epochs=2,
        device=torch.device("cpu"),
        best_model_path=checkpoint,
    )

    assert len(history) == 2
    assert history[0]["epoch"] == 1
    assert checkpoint.exists()
