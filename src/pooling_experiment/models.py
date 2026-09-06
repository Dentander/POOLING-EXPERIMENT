import torch
from torch import Tensor, nn

from .pooling import RankMaxPool2d, RatioMeanPool2d, ScaledSumPool2d, SoftmaxPool2d


class CIFAR10CNN(nn.Module):
    """Small CNN for CIFAR-10 with interchangeable pooling layers."""

    def __init__(self, pooling_type: str = "max", num_classes: int = 10) -> None:
        super().__init__()
        if pooling_type not in {"max", "avg", "scaled_sum", "ratio_mean", "softmax", "rank_max"}:
            raise ValueError(
            "pooling_type must be 'max', 'avg', 'scaled_sum', 'ratio_mean', 'softmax', or 'rank_max'"
            )
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        pool = self._make_pool(pooling_type)
        self.pooling_type = pooling_type
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            pool,
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            self._make_pool(pooling_type),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            self._make_pool(pooling_type),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _make_pool(pooling_type: str) -> nn.Module:
        if pooling_type == "max":
            return nn.MaxPool2d(kernel_size=3, stride=2)
        if pooling_type == "avg":
            return nn.AvgPool2d(kernel_size=3, stride=2)
        if pooling_type == "ratio_mean":
            return RatioMeanPool2d(kernel_size=3, stride=2)
        if pooling_type == "softmax":
            return SoftmaxPool2d(kernel_size=3, stride=2)
        if pooling_type == "rank_max":
            return RankMaxPool2d(kernel_size=3, stride=2)
        return ScaledSumPool2d(kernel_size=3, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))


if __name__ == "__main__":
    model = CIFAR10CNN(pooling_type="scaled_sum")
    sample = torch.randn(4, 3, 32, 32)
    print(model(sample).shape)
