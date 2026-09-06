import torch
from torch import nn


class WindowLinear1D(nn.Module):
	"""Linear layer with separate weights for each local window."""

	def __init__(self, in_features: int, out_features: int, bias: bool = True):
		super().__init__()
		if in_features < out_features:
			raise ValueError("in_features должно быть не меньше out_features")

		self.in_features = in_features
		self.out_features = out_features
		# For 10 inputs and 7 outputs, the window size is 4.
		self.kernel_size = in_features - out_features + 1
		self.weight = nn.Parameter(torch.empty(out_features, self.kernel_size))
		self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
		self.reset_parameters()

	def reset_parameters(self) -> None:
		nn.init.kaiming_uniform_(self.weight, a=5**0.5)
		if self.bias is not None:
			bound = self.kernel_size**-0.5
			nn.init.uniform_(self.bias, -bound, bound)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.shape[-1] != self.in_features:
			raise ValueError(
				f"Ожидалось {self.in_features} входов, получено {x.shape[-1]}"
			)

		# For 10 -> 7, the windows are [1..4], [2..5], ..., [7..10].
		windows = x.unfold(dimension=-1, size=self.kernel_size, step=1)
		result = (windows * self.weight).sum(dim=-1)
		return result + self.bias if self.bias is not None else result


if __name__ == "__main__":
	layer = WindowLinear1D(in_features=10, out_features=7)
	x = torch.randn(32, 10)
	print(layer(x).shape)  # Expected: torch.Size([32, 7])
