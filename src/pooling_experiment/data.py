from pathlib import Path

import torch
from torch import Generator
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def split_indices(
    dataset_size: int,
    validation_size: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Create a reproducible, non-overlapping train/validation split."""
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if validation_size <= 0 or validation_size >= dataset_size:
        raise ValueError("validation_size must be between 1 and dataset_size - 1")

    generator = Generator().manual_seed(seed)
    train_size = dataset_size - validation_size
    shuffled_indices = torch.randperm(dataset_size, generator=generator).tolist()
    return shuffled_indices[:train_size], shuffled_indices[train_size:]


def _transforms() -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation_transform = transforms.Compose(
        [transforms.ToTensor(), normalize]
    )
    return train_transform, evaluation_transform


def create_cifar10_dataloaders(
    root: str | Path = "data",
    batch_size: int = 128,
    validation_size: int = 5_000,
    seed: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader[Dataset], DataLoader[Dataset], DataLoader[Dataset]]:
    """Download CIFAR-10 and return train, validation, and test loaders."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    train_transform, evaluation_transform = _transforms()
    train_dataset = datasets.CIFAR10(
        root=Path(root), train=True, transform=train_transform, download=True
    )
    validation_dataset = datasets.CIFAR10(
        root=Path(root), train=True, transform=evaluation_transform, download=False
    )
    test_dataset = datasets.CIFAR10(
        root=Path(root), train=False, transform=evaluation_transform, download=True
    )
    train_indices, validation_indices = split_indices(
        len(train_dataset), validation_size, seed
    )

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    return (
        DataLoader(Subset(train_dataset, train_indices), shuffle=True, **loader_options),
        DataLoader(
            Subset(validation_dataset, validation_indices),
            shuffle=False,
            **loader_options,
        ),
        DataLoader(test_dataset, shuffle=False, **loader_options),
    )
