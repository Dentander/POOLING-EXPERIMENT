from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


IntOrPair = int | tuple[int, int]


def _as_pair(
    value: IntOrPair,
    name: str,
    min_value: int = 1,
) -> tuple[int, int]:
    if isinstance(value, int):
        result = (value, value)
    elif isinstance(value, Sequence) and len(value) == 2:
        result = (int(value[0]), int(value[1]))
    else:
        raise ValueError(f"{name} must be an int or a pair of two ints")

    if result[0] < min_value or result[1] < min_value:
        raise ValueError(f"{name} must contain values at least {min_value}")
    return result


class ScaledSumPool2d(nn.Module):
    """Pooling with max-pooling output and sum-pooling backward path."""

    def __init__(
        self,
        kernel_size: IntOrPair,
        stride: IntOrPair | None = None,
        padding: IntOrPair = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = _as_pair(kernel_size, "kernel_size")
        self.stride = _as_pair(
            kernel_size if stride is None else stride,
            "stride",
        )
        self.padding = _as_pair(padding, "padding", min_value=0)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("Expected a tensor with shape (N, C, H, W)")

        kernel_height, kernel_width = self.kernel_size
        sums = F.avg_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            count_include_pad=True,
        ) * (kernel_height * kernel_width)
        maxima = F.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        safe_sums = torch.where(sums == 0, torch.ones_like(sums), sums)
        scale = (maxima / safe_sums).detach()
        return torch.where(sums == 0, torch.zeros_like(sums), sums * scale)


class RatioMeanPool2d(nn.Module):
    """Mean pooling after scaling each value toward the window maximum."""

    def __init__(
        self,
        kernel_size: IntOrPair,
        stride: IntOrPair | None = None,
        padding: IntOrPair = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = _as_pair(kernel_size, "kernel_size")
        self.stride = _as_pair(
            kernel_size if stride is None else stride,
            "stride",
        )
        self.padding = _as_pair(padding, "padding", min_value=0)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("Expected a tensor with shape (N, C, H, W)")

        windows = F.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=1,
            padding=self.padding,
            stride=self.stride,
        )
        maxima = F.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        window_area = self.kernel_size[0] * self.kernel_size[1]
        window_maxima = maxima.flatten(2).unsqueeze(2).expand(
            -1, -1, window_area, -1
        )
        window_maxima = window_maxima.reshape(
            x.shape[0], x.shape[1] * window_area, -1
        )
        safe_values = torch.where(windows == 0, torch.ones_like(windows), windows)
        ratios = (window_maxima / safe_values).detach()
        scaled_values = torch.where(windows == 0, torch.zeros_like(windows), windows * ratios)
        output_height = (x.shape[-2] + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        output_width = (x.shape[-1] + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        scaled_values = scaled_values.view(
            x.shape[0], x.shape[1], window_area, -1
        )
        return scaled_values.mean(dim=2).view(
            x.shape[0], x.shape[1], output_height, output_width
        )


class SoftmaxPool2d(nn.Module):
    """Weighted mean pooling with softmax weights inside each window."""

    def __init__(
        self,
        kernel_size: IntOrPair,
        stride: IntOrPair | None = None,
        padding: IntOrPair = 0,
        alpha: float = 10.0,
    ) -> None:
        super().__init__()
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.kernel_size = _as_pair(kernel_size, "kernel_size")
        self.stride = _as_pair(
            kernel_size if stride is None else stride,
            "stride",
        )
        self.padding = _as_pair(padding, "padding", min_value=0)
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("Expected a tensor with shape (N, C, H, W)")

        windows = F.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=1,
            padding=self.padding,
            stride=self.stride,
        )
        window_height = (x.shape[-2] + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        window_width = (x.shape[-1] + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        window_area = self.kernel_size[0] * self.kernel_size[1]
        windows = windows.view(x.shape[0], x.shape[1], window_area, -1)
        weights = torch.softmax(self.alpha * windows, dim=2)
        pooled = (weights * windows).sum(dim=2)
        return pooled.view(x.shape[0], x.shape[1], window_height, window_width)


class RankMaxPool2d(nn.Module):
    """Max pooling whose backward path also uses normalized rank weights.

    The largest valid value in each window receives the largest rank weight.
    The forward value is exactly max pooling; rank weights affect gradients only.
    """

    def __init__(
        self,
        kernel_size: IntOrPair,
        stride: IntOrPair | None = None,
        padding: IntOrPair = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = _as_pair(kernel_size, "kernel_size")
        self.stride = _as_pair(
            kernel_size if stride is None else stride,
            "stride",
        )
        self.padding = _as_pair(padding, "padding", min_value=0)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("Expected a tensor with shape (N, C, H, W)")

        maxima = F.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        window_area = self.kernel_size[0] * self.kernel_size[1]
        windows = F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        ).view(x.shape[0], x.shape[1], window_area, -1)

        # Exclude artificial zero-padding from the ranks. MaxPool2d treats it
        # as negative infinity, so this also preserves its behavior for inputs
        # containing negative values.
        valid = F.unfold(
            torch.ones_like(x[:, :1]),
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        ).bool().view(x.shape[0], 1, window_area, -1)
        rankable = windows.masked_fill(~valid, -torch.inf)
        sorted_indices = rankable.argsort(dim=2, descending=True)
        valid_count = valid.sum(dim=2, keepdim=True)
        rank_positions = torch.arange(
            window_area, device=x.device, dtype=x.dtype
        ).view(1, 1, window_area, 1)
        rank_values = (valid_count - rank_positions).clamp_min(0)
        rank_values = rank_values / (
            valid_count * (valid_count + 1) / 2
        )
        rank_values = rank_values.expand_as(windows)
        weights = torch.zeros_like(windows).scatter_(
            2, sorted_indices, rank_values
        )
        weights = weights * valid
        weighted_sum = (weights * windows).sum(dim=2)

        output_height, output_width = maxima.shape[-2:]
        weighted_sum = weighted_sum.view(
            x.shape[0], x.shape[1], output_height, output_width
        )
        return maxima + (weighted_sum - weighted_sum.detach())
