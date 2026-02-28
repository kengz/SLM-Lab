"""Weight Normalization linear layer (Salimans & Kingma, 2016).

Decouples weight magnitude from direction: w = g * (v / ||v||).
Smoother optimization landscape without normalizing activations.
"""

import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as P


class WeightNormLinear(nn.Module):
    """Linear layer with weight normalization applied."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = P.weight_norm(nn.Linear(in_features, out_features, bias=bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LazyWeightNormLinear(nn.Module):
    """Lazy version of WeightNormLinear -- infers in_features from first input."""

    def __init__(self, out_features: int, bias: bool = True):
        super().__init__()
        self.out_features = out_features
        self.bias = bias
        self._linear = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._linear is None:
            self._linear = P.weight_norm(
                nn.Linear(x.shape[-1], self.out_features, bias=self.bias)
            ).to(x.device)
        return self._linear(x)


# Register in torch.nn so TorchArc can resolve from YAML specs
if not hasattr(nn, "WeightNormLinear"):
    setattr(nn, "WeightNormLinear", WeightNormLinear)
if not hasattr(nn, "LazyWeightNormLinear"):
    setattr(nn, "LazyWeightNormLinear", LazyWeightNormLinear)
