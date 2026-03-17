"""Tests for L0 vision pipeline: DINOv2Backbone, StereoFusionModule, VisionEncoder.

Uses MockDINOv2 — no HuggingFace download. Covers:
  - LoRA: only LoRA params are trainable, backbone frozen
  - Multi-scale shapes
  - StereoFusionModule: shapes, QK-Norm present, gradients flow
  - Dual-rate cache: cache hit / miss behavior
  - VisionEncoder end-to-end: stereo → 512
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from slm_lab.agent.net.vision import (
    DINOv2Backbone,
    LoRALinear,
    RMSNorm,
    StereoFusionModule,
    VisionEncoder,
    _SCALE_LAYERS,
    _inject_lora,
)


# ---------------------------------------------------------------------------
# MockDINOv2
# ---------------------------------------------------------------------------

class MockAttention(nn.Module):
    """Minimal multi-head attention matching DINOv2 ViT-L interface."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # Fused QKV projection — same layout as real DINOv2
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, D)
        return self.proj(out)


class MockBlock(nn.Module):
    """Minimal transformer block matching DINOv2 block interface."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MockAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MockDINOv2(nn.Module):
    """Minimal DINOv2-compatible ViT with ~1M params.

    Matches the interface expected by DINOv2Backbone:
      - .blocks: nn.ModuleList of transformer blocks (0-indexed, each has .attn.qkv)
      - .embed_dim: int
      - .forward(images) → (B, N_tokens, D)  [same as DINOv2]
      - Token layout: [CLS, patch_0, ..., patch_N, reg_0, reg_1, reg_2, reg_3]

    Dimensions chosen small (~1M params) for fast CPU tests.
    """

    PATCH_SIZE = 16
    N_REGISTERS = 4

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 24,  # must match spec layer indices (up to layer 24)
        n_heads: int = 8,
        img_size: int = 128,
    ) -> None:
        super().__init__()
        self.embed_dim = d_model
        self.n_patches = (img_size // self.PATCH_SIZE) ** 2  # 64 for 128×128

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=self.PATCH_SIZE, stride=self.PATCH_SIZE)

        # Learnable tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_tokens = nn.Parameter(torch.zeros(1, self.N_REGISTERS, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.n_patches + self.N_REGISTERS, d_model))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            MockBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            (B, N_tokens, D) where N_tokens = 1 + N_patches + N_registers
        """
        B = x.shape[0]

        # Patch embedding
        patches = self.patch_embed(x)           # (B, D, H/P, W/P)
        patches = patches.flatten(2).transpose(1, 2)  # (B, N_patches, D)

        # CLS + patches + registers
        cls = self.cls_token.expand(B, -1, -1)
        regs = self.register_tokens.expand(B, -1, -1)
        tokens = torch.cat([cls, patches, regs], dim=1)  # (B, 1+N+4, D)
        tokens = tokens + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        return self.norm(tokens)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B = 2       # batch size
H = W = 128  # training resolution
D = 128      # mock d_model (small for speed)
N_PATCHES = (H // MockDINOv2.PATCH_SIZE) ** 2  # 64


@pytest.fixture(scope="module")
def mock_dinov2() -> MockDINOv2:
    return MockDINOv2(d_model=D, n_layers=24, n_heads=8, img_size=H)


@pytest.fixture(scope="module")
def backbone(mock_dinov2: MockDINOv2) -> DINOv2Backbone:
    return DINOv2Backbone(pretrained=False, _mock_model=mock_dinov2, cache_steps=2)


@pytest.fixture(scope="module")
def encoder(mock_dinov2: MockDINOv2) -> VisionEncoder:
    return VisionEncoder(pretrained=False, _mock_model=mock_dinov2, cache_steps=2)


@pytest.fixture
def stereo_pair():
    left = torch.rand(B, 3, H, W)
    right = torch.rand(B, 3, H, W)
    return left, right


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------

class TestLoRALinear:
    def test_output_shape(self):
        linear = nn.Linear(64, 64)
        linear.weight.requires_grad_(False)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        x = torch.randn(2, 64)
        out = lora(x)
        assert out.shape == (2, 64)

    def test_trainable_params(self):
        linear = nn.Linear(64, 64)
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        trainable = [n for n, p in lora.named_parameters() if p.requires_grad]
        assert "lora_A" in trainable
        assert "lora_B" in trainable
        # Original weight must NOT be trainable
        frozen = [n for n, p in lora.named_parameters() if not p.requires_grad]
        assert any("weight" in n for n in frozen)

    def test_lora_B_zero_init(self):
        """At init, lora_B=0, so output == base linear output."""
        linear = nn.Linear(32, 32, bias=False)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        x = torch.randn(3, 32)
        base_out = nn.functional.linear(x, linear.weight)
        lora_out = lora(x)
        assert torch.allclose(base_out, lora_out, atol=1e-6)

    def test_gradient_flows_through_lora(self):
        linear = nn.Linear(32, 32, bias=False)
        linear.weight.requires_grad_(False)
        lora = LoRALinear(linear, rank=4, alpha=8.0)
        # Perturb B so output differs from base
        with torch.no_grad():
            lora.lora_B.fill_(0.01)
        x = torch.randn(2, 32)
        out = lora(x).sum()
        out.backward()
        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None


# ---------------------------------------------------------------------------
# _inject_lora
# ---------------------------------------------------------------------------

class TestInjectLoRA:
    def test_backbone_frozen_after_inject(self, mock_dinov2: MockDINOv2):
        """Backbone params stay frozen; only LoRA params are trainable."""
        model = MockDINOv2(d_model=D, n_layers=24, n_heads=8)
        for p in model.parameters():
            p.requires_grad_(False)
        _inject_lora(model, target_layers=[4, 8, 12, 16, 20, 24], rank=4, alpha=8.0)

        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        frozen_names = [n for n, p in model.named_parameters() if not p.requires_grad]

        # LoRA params should be trainable
        assert any("lora_A" in n for n in trainable_names)
        assert any("lora_B" in n for n in trainable_names)
        # No backbone qkv weight should be trainable
        assert not any("qkv.weight" in n for n in trainable_names)

    def test_trainable_param_count_reasonable(self):
        """LoRA params should be << full backbone params."""
        model = MockDINOv2(d_model=D, n_layers=24, n_heads=8)
        total_before = sum(p.numel() for p in model.parameters())
        for p in model.parameters():
            p.requires_grad_(False)
        _inject_lora(model, target_layers=[4, 8, 12, 16, 20, 24], rank=4, alpha=8.0)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable < total_before * 0.05  # LoRA << 5% of backbone


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == (2, 10, 64)

    def test_normalizes_scale(self):
        """RMSNorm output RMS should be ~1."""
        norm = RMSNorm(64)
        x = torch.randn(4, 64) * 100  # large scale
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        # After norm with learned weight=1, RMS ≈ 1
        assert (rms - 1.0).abs().max().item() < 0.5

    def test_gradient_flows(self):
        norm = RMSNorm(32)
        x = torch.randn(2, 32, requires_grad=True)
        out = norm(x).sum()
        out.backward()
        assert x.grad is not None
        assert norm.weight.grad is not None


# ---------------------------------------------------------------------------
# StereoFusionModule
# ---------------------------------------------------------------------------

class TestStereoFusion:
    @pytest.fixture
    def fusion(self):
        return StereoFusionModule(d_model=D, d_out=512, n_heads=8, n_layers=2)

    @pytest.fixture
    def stereo_feats(self):
        left = torch.randn(B, 3, N_PATCHES, D)
        right = torch.randn(B, 3, N_PATCHES, D)
        return left, right

    def test_output_shape(self, fusion: StereoFusionModule, stereo_feats):
        left, right = stereo_feats
        out = fusion(left, right)
        assert out.shape == (B, 512), f"Expected ({B}, 512), got {out.shape}"

    def test_qk_norm_present(self, fusion: StereoFusionModule):
        """QK-Norm modules exist for each layer."""
        assert len(fusion.q_norms) == 2
        assert len(fusion.k_norms) == 2
        for qn, kn in zip(fusion.q_norms, fusion.k_norms):
            assert isinstance(qn, RMSNorm)
            assert isinstance(kn, RMSNorm)

    def test_qk_norm_dims(self, fusion: StereoFusionModule):
        """QK-Norm operates on head_dim, not d_model."""
        expected_head_dim = D // 8  # d_model // n_heads
        assert fusion.q_norms[0].weight.shape == (expected_head_dim,)
        assert fusion.k_norms[0].weight.shape == (expected_head_dim,)

    def test_gradients_flow(self, fusion: StereoFusionModule, stereo_feats):
        left, right = stereo_feats
        left = left.requires_grad_(True)
        right = right.requires_grad_(True)
        out = fusion(left, right).sum()
        out.backward()
        assert left.grad is not None
        assert right.grad is not None

    def test_scale_proj_reduces_3x(self, fusion: StereoFusionModule):
        """scale_proj takes 3*D → D."""
        assert fusion.scale_proj.in_features == 3 * D
        assert fusion.scale_proj.out_features == D

    def test_out_proj_to_512(self, fusion: StereoFusionModule):
        assert fusion.out_proj.out_features == 512

    def test_different_stereo_gives_different_output(self, fusion: StereoFusionModule):
        """Left-only vs right-only content should produce different embeddings."""
        fusion.eval()
        left = torch.randn(1, 3, N_PATCHES, D)
        right_same = left.clone()
        right_diff = torch.randn(1, 3, N_PATCHES, D)
        with torch.no_grad():
            out_same = fusion(left, right_same)
            out_diff = fusion(left, right_diff)
        assert not torch.allclose(out_same, out_diff)


# ---------------------------------------------------------------------------
# DINOv2Backbone
# ---------------------------------------------------------------------------

class TestDINOv2Backbone:
    def test_forward_shapes(self, backbone: DINOv2Backbone, stereo_pair):
        left, right = stereo_pair
        left_feats, right_feats = backbone(left, right)
        assert left_feats.shape == (B, 3, N_PATCHES, D), f"Got {left_feats.shape}"
        assert right_feats.shape == (B, 3, N_PATCHES, D), f"Got {right_feats.shape}"

    def test_three_scale_layers(self, backbone: DINOv2Backbone, stereo_pair):
        """Exactly 3 scales extracted."""
        left, right = stereo_pair
        left_feats, right_feats = backbone(left, right)
        assert left_feats.shape[1] == 3
        assert right_feats.shape[1] == 3

    def test_backbone_frozen(self, backbone: DINOv2Backbone):
        """All backbone params (except LoRA) must be frozen."""
        frozen = [
            n for n, p in backbone.backbone.named_parameters()
            if not p.requires_grad and "lora" not in n
        ]
        # There should be many frozen params
        assert len(frozen) > 0

    def test_lora_trainable(self, backbone: DINOv2Backbone):
        """LoRA adapters must have requires_grad=True."""
        trainable = [
            n for n, p in backbone.backbone.named_parameters()
            if p.requires_grad
        ]
        assert any("lora_A" in n for n in trainable), "No lora_A found trainable"
        assert any("lora_B" in n for n in trainable), "No lora_B found trainable"

    def test_chirality_proj_shape(self, backbone: DINOv2Backbone):
        assert backbone.chirality_proj.in_features == D + 1
        assert backbone.chirality_proj.out_features == D

    def test_chirality_trainable(self, backbone: DINOv2Backbone):
        assert backbone.chirality_proj.weight.requires_grad

    def test_left_right_differ(self, backbone: DINOv2Backbone, stereo_pair):
        """Different images → different features (chirality + content)."""
        left, right = stereo_pair
        backbone.reset_cache()
        left_feats, right_feats = backbone(left, right)
        assert not torch.allclose(left_feats, right_feats)

    def test_dual_rate_cache_hit(self, backbone: DINOv2Backbone, stereo_pair):
        """Second call within cache window returns cached result."""
        left, right = stereo_pair
        backbone.reset_cache()
        backbone.cache_steps = 2  # cache for 2 steps

        f1_left, f1_right = backbone(left, right)
        # Modify inputs — cache should still return previous result
        left2 = torch.zeros_like(left)
        right2 = torch.zeros_like(right)
        f2_left, f2_right = backbone(left2, right2)

        assert torch.allclose(f1_left, f2_left), "Cache miss on step 2 (should hit)"
        assert torch.allclose(f1_right, f2_right)

    def test_dual_rate_cache_miss(self, backbone: DINOv2Backbone, stereo_pair):
        """After cache_steps, cache expires and fresh features are computed."""
        left, right = stereo_pair
        backbone.reset_cache()
        backbone.cache_steps = 2

        f1_left, _ = backbone(left, right)   # step 0 — compute
        backbone(left, right)                 # step 1 — cache hit
        # Step 2: cache expires, must recompute with new (zero) input
        zero_left = torch.zeros_like(left)
        zero_right = torch.zeros_like(right)
        f3_left, _ = backbone(zero_left, zero_right)  # step 2 — fresh compute

        assert not torch.allclose(f1_left, f3_left), "Cache still hit after expiry"

    def test_reset_cache_clears(self, backbone: DINOv2Backbone, stereo_pair):
        left, right = stereo_pair
        backbone(left, right)  # populate cache
        backbone.reset_cache()
        assert backbone._cache is None
        assert backbone._step_count == 0


# ---------------------------------------------------------------------------
# VisionEncoder (end-to-end)
# ---------------------------------------------------------------------------

class TestVisionEncoder:
    def test_output_shape(self, encoder: VisionEncoder, stereo_pair):
        left, right = stereo_pair
        encoder.reset_cache()
        out = encoder(left, right)
        assert out.shape == (B, 512), f"Expected ({B}, 512), got {out.shape}"

    def test_output_dtype_float32(self, encoder: VisionEncoder, stereo_pair):
        left, right = stereo_pair
        encoder.reset_cache()
        out = encoder(left, right)
        assert out.dtype == torch.float32

    def test_gradients_flow_through_fusion(self, encoder: VisionEncoder):
        """Gradients must reach LoRA params and fusion params."""
        encoder.reset_cache()
        left = torch.rand(1, 3, H, W)
        right = torch.rand(1, 3, H, W)
        out = encoder(left, right).sum()
        out.backward()

        # LoRA grads
        lora_grads = [
            (n, p.grad)
            for n, p in encoder.backbone.backbone.named_parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert len(lora_grads) > 0, "No LoRA gradients found"

        # Fusion grads
        fusion_grads = [
            (n, p.grad)
            for n, p in encoder.fusion.named_parameters()
            if p.grad is not None
        ]
        assert len(fusion_grads) > 0, "No fusion gradients found"

    def test_different_inputs_different_outputs(self, encoder: VisionEncoder):
        encoder.reset_cache()
        left = torch.rand(1, 3, H, W)
        right = torch.rand(1, 3, H, W)
        left2 = torch.rand(1, 3, H, W)
        right2 = torch.rand(1, 3, H, W)
        with torch.no_grad():
            encoder.reset_cache()
            out1 = encoder(left, right)
            encoder.reset_cache()
            out2 = encoder(left2, right2)
        assert not torch.allclose(out1, out2)

    def test_only_trained_params_have_grad(self, encoder: VisionEncoder):
        """Backbone frozen weights must have no grad after backward."""
        encoder.reset_cache()
        left = torch.rand(1, 3, H, W)
        right = torch.rand(1, 3, H, W)
        out = encoder(left, right).sum()
        out.backward()

        # No backbone weights (non-LoRA) should accumulate grad
        for n, p in encoder.backbone.backbone.named_parameters():
            if not p.requires_grad:
                # Frozen params should have no gradient
                assert p.grad is None, f"Frozen param {n} has grad"
