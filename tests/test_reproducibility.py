import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from pooling_experiment import set_seed


def test_set_seed_reproduces_random_values() -> None:
    set_seed(123)
    first = (random.random(), np.random.rand(), torch.rand(1).item())

    set_seed(123)
    second = (random.random(), np.random.rand(), torch.rand(1).item())

    assert first == second
