import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from pooling_experiment import RankMaxPool2d, RatioMeanPool2d, ScaledSumPool2d, SoftmaxPool2d


def test_forward_matches_max_pool_for_positive_inputs() -> None:
    torch.manual_seed(0)
    inputs = torch.rand(2, 3, 8, 9)
    pooling = ScaledSumPool2d(kernel_size=3, stride=2)

    actual = pooling(inputs)
    expected = nn.MaxPool2d(kernel_size=3, stride=2)(inputs)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected)


def test_zero_window_returns_zero() -> None:
    inputs = torch.zeros(1, 1, 4, 4)

    actual = ScaledSumPool2d(kernel_size=2)(inputs)

    assert torch.equal(actual, torch.zeros(1, 1, 2, 2))


def test_gradient_is_scaled_sum_gradient() -> None:
    inputs = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)

    ScaledSumPool2d(kernel_size=2)(inputs).sum().backward()

    gradient = inputs.grad
    assert gradient is not None
    assert torch.allclose(gradient, torch.full_like(inputs, 0.4))


def test_gradient_differs_from_max_pool_gradient() -> None:
    custom_inputs = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True
    )
    max_inputs = custom_inputs.detach().clone().requires_grad_()

    ScaledSumPool2d(kernel_size=2)(custom_inputs).sum().backward()
    nn.MaxPool2d(kernel_size=2)(max_inputs).sum().backward()

    custom_gradient = custom_inputs.grad
    max_gradient = max_inputs.grad
    assert custom_gradient is not None
    assert max_gradient is not None
    assert not torch.equal(custom_gradient, max_gradient)
    assert torch.equal(max_gradient, torch.tensor([[[[0.0, 0.0], [0.0, 1.0]]]]))


def test_padding_matches_max_pool_for_nonnegative_inputs() -> None:
    torch.manual_seed(1)
    inputs = torch.rand(1, 2, 5, 6)
    pooling = ScaledSumPool2d(kernel_size=3, stride=2, padding=1)

    actual = pooling(inputs)
    expected = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(inputs)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected)


def test_ratio_mean_matches_max_pool_for_positive_inputs() -> None:
    inputs = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]], [[4.0, 3.0], [2.0, 1.0]]]]
    )

    actual = RatioMeanPool2d(kernel_size=2)(inputs)
    expected = nn.MaxPool2d(kernel_size=2)(inputs)

    assert torch.allclose(actual, expected)


def test_ratio_mean_keeps_zero_values_zero() -> None:
    inputs = torch.tensor([[[[0.0, 2.0], [3.0, 4.0]]]])

    actual = RatioMeanPool2d(kernel_size=2)(inputs)

    assert torch.equal(actual, torch.tensor([[[[3.0]]]]))


def test_ratio_mean_uses_detached_ratios_for_gradient() -> None:
    inputs = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)

    RatioMeanPool2d(kernel_size=2)(inputs).sum().backward()

    gradient = inputs.grad
    assert gradient is not None
    assert torch.allclose(
        gradient,
        torch.tensor([[[[1.0, 0.5], [1 / 3, 0.25]]]]),
    )


def test_softmax_pool_matches_max_pool_as_alpha_increases() -> None:
    inputs = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

    actual = SoftmaxPool2d(kernel_size=2, alpha=100.0)(inputs)
    expected = nn.MaxPool2d(kernel_size=2)(inputs)

    assert torch.allclose(actual, expected, atol=1e-3)


def test_softmax_pool_with_zero_alpha_is_average_pool() -> None:
    inputs = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

    actual = SoftmaxPool2d(kernel_size=2, alpha=0.0)(inputs)
    expected = nn.AvgPool2d(kernel_size=2)(inputs)

    assert torch.allclose(actual, expected)


def test_rank_max_pool_forward_matches_max_pool() -> None:
    inputs = torch.tensor([[[[-3.0, -2.0], [-1.0, -4.0]]]])

    actual = RankMaxPool2d(kernel_size=2)(inputs)
    expected = nn.MaxPool2d(kernel_size=2)(inputs)

    assert torch.equal(actual, expected)


def test_rank_max_pool_adds_rank_weighted_gradients() -> None:
    inputs = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)

    RankMaxPool2d(kernel_size=2)(inputs).sum().backward()

    # Rank weights are (1, 2, 3, 4) / 10 in input order; max pooling adds
    # one more unit of gradient to the element with value 4.
    assert inputs.grad is not None
    assert torch.allclose(
        inputs.grad,
        torch.tensor([[[[0.1, 0.2], [0.3, 1.4]]]]),
    )


def test_rank_max_pool_padding_matches_max_pool_for_negative_inputs() -> None:
    inputs = -torch.arange(1.0, 17.0).view(1, 1, 4, 4)

    actual = RankMaxPool2d(kernel_size=3, stride=2, padding=1)(inputs)
    expected = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(inputs)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("kernel_size", "stride", "padding"),
    [
        (0, None, 0),
        ((2, 0), None, 0),
        (2, 0, 0),
        (2, None, -1),
    ],
)
def test_invalid_pooling_arguments_raise(
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None,
    padding: int | tuple[int, int],
) -> None:
    with pytest.raises(ValueError):
        ScaledSumPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
