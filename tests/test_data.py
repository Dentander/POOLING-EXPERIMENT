import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from pooling_experiment import split_indices


def test_split_indices_is_reproducible() -> None:
    first_train, first_validation = split_indices(100, 20, seed=7)
    second_train, second_validation = split_indices(100, 20, seed=7)

    assert first_train == second_train
    assert first_validation == second_validation


def test_split_indices_are_disjoint_and_cover_dataset() -> None:
    train_indices, validation_indices = split_indices(100, 20, seed=7)

    assert len(train_indices) == 80
    assert len(validation_indices) == 20
    assert set(train_indices).isdisjoint(validation_indices)
    assert set(train_indices) | set(validation_indices) == set(range(100))


@pytest.mark.parametrize(
    ("dataset_size", "validation_size"),
    [(0, 1), (10, 0), (10, 10), (10, 11)],
)
def test_split_indices_reject_invalid_sizes(
    dataset_size: int, validation_size: int
) -> None:
    with pytest.raises(ValueError):
        split_indices(dataset_size, validation_size, seed=0)
