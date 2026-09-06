# POOLING-EXPERIMENT

`POOLING-EXPERIMENT` is a research-oriented project for studying how different pooling operations affect the quality and training behaviour of a convolutional neural network on the CIFAR-10 image-classification task.

The project trains one common CNN architecture while changing only the pooling layer. This makes it possible to compare approaches under the same training settings, random seed, dataset split, optimiser, and evaluation metrics.

## What is being compared

The implementation includes standard pooling layers and several alternative aggregation rules:

- `max` — max pooling, which keeps the strongest activation in each local window;
- `avg` — average pooling, which aggregates all activations equally;
- `scaled_sum` — a scaled sum of values in the local window;
- `ratio_mean` — a pooling operation based on relative activation values;
- `softmax` — softmax-weighted pooling, where larger activations receive more weight;
- `rank_max` — rank-aware max pooling.

Each experiment reports training and validation history together with final test metrics. The code records accuracy and macro F1-score, so performance can be assessed even when class-level behaviour differs.

## Project structure

```text
src/pooling_experiment/   Main Python package
  pooling.py              Custom pooling-layer implementations
  models.py               CIFAR-10 CNN architecture
  data.py                 Dataset loading and deterministic data split
  training.py             Training and evaluation loops
  tracking.py             MLflow logging helpers
  experiment.py           End-to-end experiment entry point
scripts/
  run_experiment.py       CLI for a single experiment
  run_rank_max_sweep.ps1  PowerShell sweep for RankMax across seeds
tests/                    Unit tests
```

## Requirements and installation

Python 3.10 or newer is required. Create a virtual environment and install the package with development dependencies:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

The principal dependencies are PyTorch, TorchVision, NumPy, MLflow, and Pytest. The first run downloads CIFAR-10 to the selected data directory.

## Running an experiment

Run a baseline using max pooling:

```powershell
python scripts/run_experiment.py --pooling max --epochs 10
```

Run a 20-epoch RankMax experiment, save its best checkpoint, and use a fixed seed:

```powershell
python scripts/run_experiment.py `
  --pooling rank_max `
  --seed 0 `
  --epochs 20 `
  --checkpoint artifacts/checkpoints/rank_max_seed_0.pt
```

Useful command-line options include:

- `--pooling`: pooling method to evaluate;
- `--seed`: random seed used for deterministic experiment setup;
- `--epochs`, `--batch-size`, `--learning-rate`, and `--weight-decay`: training hyperparameters;
- `--data-root`: local directory for CIFAR-10;
- `--checkpoint`: destination for the best model checkpoint;
- `--device`: explicitly select `cpu` or `cuda` when needed.

To reproduce the included RankMax seed sweep, run:

```powershell
.\\scripts\\run_rank_max_sweep.ps1
```

## Experiment tracking

Runs are tracked locally with MLflow. By default, metadata and metrics are stored in `mlflow.db`, while run artifacts are written to `mlruns/`. Start the MLflow UI after running experiments:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open the local address shown by MLflow in a browser to compare parameters, metrics, and saved checkpoints across runs.

## Reproducibility

Every run accepts a seed and configures Python, NumPy, and PyTorch random generators. Dataset splitting also uses the supplied seed. For a fair comparison, run each pooling method with the same hyperparameters and a shared collection of seeds, then compare the aggregate results rather than relying on a single run.

## Tests

Run the test suite with:

```powershell
pytest
```

## Version-control policy

Source code, scripts, tests, and project metadata are tracked. Downloaded datasets, virtual environments, checkpoints, logs, and MLflow databases are intentionally excluded through `.gitignore`, because they are local/generated files and can be large.
