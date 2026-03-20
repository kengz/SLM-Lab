"""L0 Vision pipeline: DINOv2 backbone + LoRA + stereo fusion → 512-dim embedding.

Architecture (from L0-perception.md §2):
  - DINOv2 ViT-L/14 (304M), frozen. Loaded from HuggingFace via torch.hub.
  - LoRA (rank 16, alpha 32) at Q/V projections in layers 4,8,12,16,20,24. ~300K trainable.
  - Chirality encoding: 1-dim flag broadcast to each patch, projected 1025→1024.
  - Multi-scale features extracted at layers 8, 16, 24.
  - StereoFusionModule: 2-layer cross-attention with QK-Norm (RMSNorm on Q/K), 8 heads.
    Input 3072 per patch → 1024 → pool → 512.
  - FiLM conditioning (L3 mood→vision) deferred to vision_film.py (§2.7).
  - Dual-rate caching: vision runs at 5-10 Hz; cache reused at 25 Hz control rate.

Output: 512-dim visual embedding consumed by L1 channel attention.

Phase 3.2b. Spec: notes/layers/L0-perception.md §2.1–2.7.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Drop-in LoRA wrapper for nn.Linear.

    Freezes the original weight. Adds low-rank update: W' = W + (alpha/rank) * B @ A.

    Args:
        linear:  the nn.Linear to wrap (its weight is frozen in-place)
        rank:    LoRA rank r
        alpha:   LoRA scaling alpha (effective scale = alpha / rank)
    """

    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: float = 32.0) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.scale = alpha / rank

        # Frozen original weight (and optional bias)
        self.weight = linear.weight  # reference — already frozen by caller
        self.bias = linear.bias      # may be None

        # Low-rank matrices: A initialized as Gaussian, B as zeros (standard LoRA init)
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base + lora


def _inject_lora(
    module: nn.Module,
    target_layers: list[int],
    rank: int = 16,
    alpha: float = 32.0,
) -> None:
    """Inject LoRA into Q and V projections of specified transformer layers in-place.

    Assumes DINOv2 ViT block structure where each block has an `attn` sub-module
    with `qkv` as a single fused linear (as in timm/dinov2).

    DINOv2 uses a fused qkv projection. We split it into three separate projections
    and replace Q and V with LoRA-wrapped versions. K remains frozen.

    Args:
        module:        the DINOv2 model
        target_layers: 1-indexed layer numbers to inject LoRA into
        rank:          LoRA rank
        alpha:         LoRA alpha
    """
    blocks = module.blocks  # nn.ModuleList of transformer blocks (0-indexed)
    for layer_idx in target_layers:
        block = blocks[layer_idx - 1]  # convert 1-indexed → 0-indexed
        attn = block.attn

        # DINOv2 uses fused qkv: nn.Linear(d_model, 3*d_model)
        # Replace with split Q, K, V projections where Q and V get LoRA
        qkv_weight = attn.qkv.weight.data       # (3*D, D)
        qkv_bias = attn.qkv.bias.data if attn.qkv.bias is not None else None

        d = attn.qkv.in_features  # d_model
        d3 = attn.qkv.out_features  # 3 * d_model

        assert d3 == 3 * d, f"Expected 3*d_model fused qkv, got {d3} for d={d}"

        # Split fused weights: [Q_w | K_w | V_w]
        q_w = qkv_weight[:d].clone()
        k_w = qkv_weight[d:2*d].clone()
        v_w = qkv_weight[2*d:].clone()

        q_b = qkv_bias[:d].clone() if qkv_bias is not None else None
        k_b = qkv_bias[d:2*d].clone() if qkv_bias is not None else None
        v_b = qkv_bias[2*d:].clone() if qkv_bias is not None else None

        # Build frozen linears for Q, K, V
        q_linear = nn.Linear(d, d, bias=q_b is not None)
        q_linear.weight = nn.Parameter(q_w, requires_grad=False)
        if q_b is not None:
            q_linear.bias = nn.Parameter(q_b, requires_grad=False)

        k_linear = nn.Linear(d, d, bias=k_b is not None)
        k_linear.weight = nn.Parameter(k_w, requires_grad=False)
        if k_b is not None:
            k_linear.bias = nn.Parameter(k_b, requires_grad=False)

        v_linear = nn.Linear(d, d, bias=v_b is not None)
        v_linear.weight = nn.Parameter(v_w, requires_grad=False)
        if v_b is not None:
            v_linear.bias = nn.Parameter(v_b, requires_grad=False)

        # Wrap Q and V with LoRA
        attn.q_lora = LoRALinear(q_linear, rank=rank, alpha=alpha)
        attn.k_proj = k_linear
        attn.v_lora = LoRALinear(v_linear, rank=rank, alpha=alpha)

        # Remove fused qkv — replaced by split projections above
        # We must also patch the forward method of attn to use the split projections
        _patch_attn_forward(attn)


def _patch_attn_forward(attn: nn.Module) -> None:
    """Patch attn.forward to use split q_lora/k_proj/v_lora instead of fused qkv.

    DINOv2 MemEffAttention (or Attention) forward calls self.qkv(x) to get
    (B, N, 3*D) then reshapes. We replace with split projections.
    """
    import types

    def new_forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
        B, N, D = x.shape
        q = self.q_lora(x)  # (B, N, D)
        k = self.k_proj(x)  # (B, N, D)
        v = self.v_lora(x)  # (B, N, D)

        num_heads = self.num_heads
        head_dim = D // num_heads
        scale = head_dim ** -0.5

        # Reshape: (B, N, D) → (B, num_heads, N, head_dim)
        q = q.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)

        # Standard scaled dot-product attention
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = attn_weights @ v  # (B, num_heads, N, head_dim)

        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return self.proj(out)

    attn.forward = types.MethodType(new_forward, attn)


# ---------------------------------------------------------------------------
# Multi-scale feature hook
# ---------------------------------------------------------------------------

class _MultiScaleHook:
    """Register forward hooks on DINOv2 blocks to capture intermediate features."""

    def __init__(self, model: nn.Module, layers: list[int]) -> None:
        self._features: dict[int, torch.Tensor] = {}
        self._hooks = []
        blocks = model.blocks
        for layer_idx in layers:
            block = blocks[layer_idx - 1]  # 1-indexed → 0-indexed
            hook = block.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            self._features[layer_idx] = output
        return hook

    def get(self) -> dict[int, torch.Tensor]:
        return dict(self._features)

    def clear(self) -> None:
        self._features.clear()

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()


# ---------------------------------------------------------------------------
# QK-Norm helper
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias term)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


# ---------------------------------------------------------------------------
# StereoFusionModule
# ---------------------------------------------------------------------------

class StereoFusionModule(nn.Module):
    """Cross-attention stereo fusion with QK-Norm.

    Fuses left and right multi-scale features into a single 512-dim embedding.
    QK-Norm (RMSNorm on Q/K before dot product) prevents attention logit blow-up
    when stereo features have heterogeneous activation scales.

    Input:
        left:  (B, 3, N_patches, D)  — 3 scales, N_patches patch tokens, D=1024
        right: (B, 3, N_patches, D)

    Output: (B, 512) visual embedding

    Spec: L0-perception.md §2.5, §2.7 (QK-Norm note).
    """

    def __init__(
        self,
        d_model: int = 1024,
        d_out: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        n_scales: int = 3,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Concat 3 scales per patch: 3*1024 → 1024
        self.scale_proj = nn.Linear(n_scales * d_model, d_model)

        # Per-layer cross-attention projections (QKV for each layer)
        self.q_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.k_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.v_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.out_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])

        # QK-Norm: RMSNorm on Q and K per layer (applied per head)
        self.q_norms = nn.ModuleList([RMSNorm(self.head_dim) for _ in range(n_layers)])
        self.k_norms = nn.ModuleList([RMSNorm(self.head_dim) for _ in range(n_layers)])

        # Post-attention layer norms
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # Output projection
        self.out_proj = nn.Linear(d_model, d_out)

    def _cross_attn(
        self,
        layer_idx: int,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
    ) -> torch.Tensor:
        """Single cross-attention layer with QK-Norm.

        Args:
            q_x:  (B, N, D) — query source
            kv_x: (B, M, D) — key/value source

        Returns:
            (B, N, D) attended output
        """
        B, N, D = q_x.shape
        _, M, _ = kv_x.shape
        H = self.n_heads
        Dh = self.head_dim

        q = self.q_projs[layer_idx](q_x).reshape(B, N, H, Dh).permute(0, 2, 1, 3)  # (B,H,N,Dh)
        k = self.k_projs[layer_idx](kv_x).reshape(B, M, H, Dh).permute(0, 2, 1, 3)  # (B,H,M,Dh)
        v = self.v_projs[layer_idx](kv_x).reshape(B, M, H, Dh).permute(0, 2, 1, 3)  # (B,H,M,Dh)

        # QK-Norm: normalize Q and K per head
        q = self.q_norms[layer_idx](q)  # RMSNorm broadcasts over (B,H,N,Dh)
        k = self.k_norms[layer_idx](k)

        scale = Dh ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale   # (B,H,N,M)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v                              # (B,H,N,Dh)

        out = out.permute(0, 2, 1, 3).reshape(B, N, D)
        return self.out_projs[layer_idx](out)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """
        Args:
            left:  (B, 3, N, 1024)
            right: (B, 3, N, 1024)

        Returns:
            (B, 512)
        """
        B, S, N, D = left.shape

        # Concat scales per patch: (B, N, 3*D)
        left_cat = left.permute(0, 2, 1, 3).reshape(B, N, S * D)
        right_cat = right.permute(0, 2, 1, 3).reshape(B, N, S * D)

        # Project to d_model: (B, N, D)
        lf = self.scale_proj(left_cat)
        rf = self.scale_proj(right_cat)

        # Cross-attention layers: left queries right
        for i in range(self.n_layers):
            attn_out = self._cross_attn(i, lf, rf)
            lf = self.norms[i](lf + attn_out)

        # Mean pool patch dimension: (B, D)
        pooled = lf.mean(dim=1)

        return self.out_proj(pooled)  # (B, 512)


# ---------------------------------------------------------------------------
# DINOv2Backbone
# ---------------------------------------------------------------------------

# LoRA target layers (1-indexed, per spec §2.6)
_LORA_LAYERS = [4, 8, 12, 16, 20, 24]
# Multi-scale extraction layers (1-indexed, per spec §2.4)
_SCALE_LAYERS = [8, 16, 24]

# Dual-rate config
_VISION_HZ = 10      # vision runs at up to 10 Hz
_CONTROL_HZ = 25     # control rate
_CACHE_STEPS = _CONTROL_HZ // _VISION_HZ  # reuse cached embedding for this many steps


class DINOv2Backbone(nn.Module):
    """DINOv2 ViT-L/14 with LoRA adapters, chirality encoding, and dual-rate caching.

    Loads DINOv2 ViT-L from HuggingFace via torch.hub (facebookresearch/dinov2).
    Freezes all backbone parameters. Injects LoRA at Q/V in layers 4,8,12,16,20,24.
    Chirality encoding: 1-dim flag appended per patch, projected 1025→1024.
    Multi-scale features extracted from layers 8, 16, 24.
    Dual-rate: caches output for `cache_steps` control steps.

    Args:
        pretrained:   if True, load from HuggingFace; if False, use random weights (for tests)
        lora_rank:    LoRA rank (default 16)
        lora_alpha:   LoRA alpha (default 32.0)
        cache_steps:  number of control steps to reuse cached visual features
        _mock_model:  optional pre-built model to use instead of HuggingFace (for tests)
    """

    def __init__(
        self,
        pretrained: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        cache_steps: int = _CACHE_STEPS,
        _mock_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.cache_steps = cache_steps

        # Load or accept backbone
        if _mock_model is not None:
            self.backbone = _mock_model
        elif pretrained:
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitl14", pretrained=True
            )
        else:
            # Random ViT-L (for offline testing without HF download)
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitl14", pretrained=False
            )

        # Freeze all backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Inject LoRA at Q/V in specified layers
        _inject_lora(self.backbone, _LORA_LAYERS, rank=lora_rank, alpha=lora_alpha)

        # Chirality projection: 1025 → 1024 (patch_dim + 1 flag → patch_dim)
        d_model = self.backbone.embed_dim  # 1024 for ViT-L
        self.chirality_proj = nn.Linear(d_model + 1, d_model)

        # Multi-scale hook (registered after model is ready)
        self._hook = _MultiScaleHook(self.backbone, _SCALE_LAYERS)

        # Dual-rate cache
        self._cache: tuple[torch.Tensor, torch.Tensor] | None = None  # (left_feats, right_feats)
        self._step_count: int = 0

    def _get_patch_tokens(
        self, image: torch.Tensor, chirality: float
    ) -> dict[int, torch.Tensor]:
        """Forward one eye through DINOv2, injecting chirality, return multi-scale features.

        Args:
            image:     (B, 3, H, W) float32, values in [0, 1]
            chirality: 0.0 for left eye, 1.0 for right eye

        Returns:
            dict mapping layer_idx → (B, N_patches, 1024) patch token features
        """
        B = image.shape[0]
        self._hook.clear()

        # Chirality injected via pre-hook on blocks[0]: appends 1-dim flag to each token,
        # then projects D+1 → D via chirality_proj. Hooks capture intermediate block outputs.

        chirality_tensor = torch.full(
            (B, 1), chirality, dtype=image.dtype, device=image.device
        )

        def _chirality_hook(module, args):
            """Pre-hook on blocks[0]: receive (x,) where x is (B, N_tokens, D).
            Append chirality flag to each token, project back to D, then pass on."""
            x = args[0]
            # Expand chirality to all tokens
            flag = chirality_tensor.unsqueeze(1).expand(B, x.shape[1], 1)
            x_aug = torch.cat([x, flag], dim=-1)  # (B, N, D+1)
            x_proj = self.chirality_proj(x_aug)    # (B, N, D)
            return (x_proj,)

        pre_hook = self.backbone.blocks[0].register_forward_pre_hook(_chirality_hook)
        try:
            _ = self.backbone(image)
        finally:
            pre_hook.remove()

        features = self._hook.get()

        # DINOv2 token layout: [CLS, patch_0, ..., patch_N, reg_0..3]
        # Strip CLS (index 0) and 4 trailing register tokens.
        patch_features = {}
        for layer_idx, feats in features.items():
            # feats: (B, N_tokens, D) where N_tokens = 1 + N_patches + 4
            patch_features[layer_idx] = feats[:, 1:-4, :]  # (B, N_patches, D)

        return patch_features

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract multi-scale features from stereo images with dual-rate caching.

        Args:
            left:  (B, 3, H, W) — left eye image, float32 [0,1]
            right: (B, 3, H, W) — right eye image, float32 [0,1]

        Returns:
            left_feats:  (B, 3, N_patches, 1024)
            right_feats: (B, 3, N_patches, 1024)
        """
        # Dual-rate: reuse cache if within cache window
        if self._cache is not None and self._step_count % self.cache_steps != 0:
            self._step_count += 1
            return self._cache

        # Extract multi-scale for each eye (shared weights, different chirality)
        left_scales = self._get_patch_tokens(left, chirality=0.0)
        right_scales = self._get_patch_tokens(right, chirality=1.0)

        # Stack scales: (B, 3, N_patches, 1024) ordered by _SCALE_LAYERS
        def _stack(scale_dict: dict[int, torch.Tensor]) -> torch.Tensor:
            tensors = [scale_dict[layer] for layer in _SCALE_LAYERS]
            return torch.stack(tensors, dim=1)  # (B, 3, N, D)

        left_feats = _stack(left_scales)
        right_feats = _stack(right_scales)

        self._cache = (left_feats, right_feats)
        self._step_count += 1
        return left_feats, right_feats

    def reset_cache(self) -> None:
        """Reset dual-rate cache (call at episode start)."""
        self._cache = None
        self._step_count = 0

    @property
    def d_model(self) -> int:
        return self.backbone.embed_dim


# ---------------------------------------------------------------------------
# VisionEncoder (full pipeline)
# ---------------------------------------------------------------------------

class VisionEncoder(nn.Module):
    """Full L0 vision pipeline: stereo images → 512-dim visual embedding.

    Combines DINOv2Backbone (multi-scale stereo features) with StereoFusionModule.

    Args:
        pretrained:   if True, load DINOv2 weights from HuggingFace
        lora_rank:    LoRA rank (default 16)
        lora_alpha:   LoRA alpha (default 32.0)
        cache_steps:  dual-rate cache window (default 2, for ~10 Hz vision at 25 Hz control)
        _mock_model:  bypass HF download (tests only)

    Input:
        left:  (B, 3, H, W) — left eye, float32 [0,1]
        right: (B, 3, H, W) — right eye, float32 [0,1]

    Output:
        (B, 512) visual embedding
    """

    def __init__(
        self,
        pretrained: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        cache_steps: int = _CACHE_STEPS,
        _mock_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.backbone = DINOv2Backbone(
            pretrained=pretrained,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            cache_steps=cache_steps,
            _mock_model=_mock_model,
        )
        self.fusion = StereoFusionModule(
            d_model=self.backbone.d_model,
            d_out=512,
            n_heads=8,
            n_layers=2,
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """
        Args:
            left:  (B, 3, H, W)
            right: (B, 3, H, W)

        Returns:
            (B, 512) visual embedding
        """
        left_feats, right_feats = self.backbone(left, right)
        return self.fusion(left_feats, right_feats)

    def reset_cache(self) -> None:
        self.backbone.reset_cache()
