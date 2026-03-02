"""Batch Renormalization (Ioffe 2017) for CrossQ.

Standard BatchNorm uses noisy minibatch statistics during training but
running statistics during inference — this mismatch causes instability
in off-policy RL (CrossQ). Batch Renormalization smoothly transitions
from minibatch to running statistics via clamped correction factors.

Reference: https://arxiv.org/abs/1702.03275
"""

import torch
import torch.nn as nn
from torch.nn.modules.lazy import LazyModuleMixin


class BatchRenorm1d(nn.Module):
    """Batch Renormalization layer.

    During training, normalizes using minibatch statistics but applies
    correction factors (r, d) that gradually align output with running
    statistics. During warmup, behaves identically to standard BatchNorm.

    Args:
        num_features: Number of features (channels).
        eps: Numerical stability constant.
        momentum: Running stats update rate (PyTorch convention: new = old * (1-m) + batch * m).
        r_max: Maximum scale correction factor after warmup.
        d_max: Maximum shift correction factor after warmup.
        warmup_steps: Training steps before full BRN correction is active.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.01,
        r_max: float = 3.0,
        d_max: float = 5.0,
        warmup_steps: int = 10000,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.r_max_limit = r_max
        self.d_max_limit = d_max
        self.warmup_steps = warmup_steps

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    def _reshape_for_broadcast(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Reshape 1D (C,) tensor to broadcast with x of shape (B, C, ...) ."""
        if x.dim() == 2:
            return v
        shape = [1, -1] + [1] * (x.dim() - 2)  # e.g. (1, C, 1, 1) for 4D
        return v.view(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            rm = self._reshape_for_broadcast(self.running_mean, x)
            rv = self._reshape_for_broadcast(self.running_var, x)
            w = self._reshape_for_broadcast(self.weight, x)
            b = self._reshape_for_broadcast(self.bias, x)
            x_hat = (x - rm) / (rv + self.eps).sqrt()
            return w * x_hat + b

        # Compute batch statistics over all dims except features
        dims = [0] + list(range(2, x.dim()))
        batch_mean = x.mean(dims)
        batch_var = x.var(dims, unbiased=False)
        batch_std = (batch_var + self.eps).sqrt()
        running_std = (self.running_var + self.eps).sqrt()

        # Warmup schedule: linearly increase r_max 1->limit, d_max 0->limit
        t = min(self.step.item() / max(self.warmup_steps, 1), 1.0)
        r_max = 1.0 + t * (self.r_max_limit - 1.0)
        d_max = t * self.d_max_limit

        # Correction factors (detached — no gradient through r, d)
        r = (batch_std.detach() / running_std).clamp(1.0 / r_max, r_max)
        d = ((batch_mean.detach() - self.running_mean) / running_std).clamp(-d_max, d_max)

        # Reshape for broadcasting with arbitrary-dim input
        bm = self._reshape_for_broadcast(batch_mean, x)
        bs = self._reshape_for_broadcast(batch_std, x)
        r = self._reshape_for_broadcast(r, x)
        d = self._reshape_for_broadcast(d, x)
        w = self._reshape_for_broadcast(self.weight, x)
        b = self._reshape_for_broadcast(self.bias, x)

        # Normalize with batch stats, correct toward running stats
        x_hat = (x - bm) / bs * r + d

        # Update running statistics
        with torch.no_grad():
            self.running_mean.lerp_(batch_mean, self.momentum)
            self.running_var.lerp_(batch_var, self.momentum)
            self.step += 1

        return w * x_hat + b

    def extra_repr(self) -> str:
        return (
            f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"r_max={self.r_max_limit}, d_max={self.d_max_limit}, "
            f"warmup_steps={self.warmup_steps}"
        )


class LazyBatchRenorm1d(LazyModuleMixin, BatchRenorm1d):
    """Lazy version that infers num_features from first input.

    Use in TorchArc YAML specs where input dimensions are unknown:
        - LazyBatchRenorm1d:
            momentum: 0.01
            eps: 0.001
            warmup_steps: 10000
    """

    cls_to_become = BatchRenorm1d
    weight: nn.UninitializedParameter
    bias: nn.UninitializedParameter

    def __init__(
        self,
        eps: float = 1e-5,
        momentum: float = 0.01,
        r_max: float = 3.0,
        d_max: float = 5.0,
        warmup_steps: int = 10000,
    ):
        super().__init__(0, eps=eps, momentum=momentum, r_max=r_max, d_max=d_max, warmup_steps=warmup_steps)
        self.weight = nn.UninitializedParameter()
        self.bias = nn.UninitializedParameter()

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params():
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def initialize_parameters(self, input: torch.Tensor) -> None:
        if self.has_uninitialized_params():
            with torch.no_grad():
                num_features = input.shape[1]
                self.num_features = num_features
                self.weight.materialize((num_features,))
                self.bias.materialize((num_features,))
                self.register_buffer("running_mean", torch.zeros(num_features, device=input.device))
                self.register_buffer("running_var", torch.ones(num_features, device=input.device))
                self.reset_parameters()


# Register in torch.nn so TorchArc can resolve from YAML specs
if not hasattr(nn, "BatchRenorm1d"):
    setattr(nn, "BatchRenorm1d", BatchRenorm1d)
if not hasattr(nn, "LazyBatchRenorm1d"):
    setattr(nn, "LazyBatchRenorm1d", LazyBatchRenorm1d)
