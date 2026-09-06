# POOLING-EXPERIMENT

Experiments for comparing pooling operators in a CIFAR-10 convolutional neural network.

## Setup

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -e ".[dev]"
```

## Run an experiment

```powershell
python scripts/run_experiment.py --pooling rank_max --epochs 20
```

Available pooling variants: `max`, `avg`, `scaled_sum`, `ratio_mean`, `softmax`, and `rank_max`.

## Tests

```powershell
pytest
```

Downloaded datasets, experiment logs, MLflow state, and checkpoints are kept out of version control.
