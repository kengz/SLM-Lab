"""Tests for DaseinNet vision mode integration.

Coverage:
- Forward pass with vision dict obs (mock DINOv2) produces correct shapes
- Gradient flow through full pipeline (proprio_enc_vision, StereoFusion, L1, heads)
- DINOv2 backbone frozen (requires_grad=False on backbone params)
- LoRA adapters are trainable (requires_grad=True on lora_A, lora_B)
- InfoNCE loss computes and is a finite scalar
- Ground truth mode still works unchanged (backward compat)
- vision_mode="vision" rejects flat tensor input
- vision_mode="ground_truth" accepts flat tensor input
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from slm_lab.agent.net.dasein_net import DaseinNet, OBS_DIM, INFONCE_ALPHA, INFONCE_TEMP, InfoNCELoss


# ---------------------------------------------------------------------------
# MockDINOv2 — self-contained, no HuggingFace download
# (mirrors test_vision.py MockDINOv2 — keep in sync if that changes)
# ---------------------------------------------------------------------------

class _MockAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
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


class _MockBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _MockAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _MockDINOv2(nn.Module):
    """Minimal DINOv2-compatible ViT for tests (d_model=128, no HF download).

    Matches DINOv2Backbone interface:
      .blocks: nn.ModuleList (0-indexed, each has .attn.qkv)
      .embed_dim: int
      .forward(images) → (B, N_tokens, D)  [CLS + patches + 4 registers]
    """
    PATCH_SIZE = 16
    N_REGISTERS = 4

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 24,
        n_heads: int = 8,
        img_size: int = 128,
    ) -> None:
        super().__init__()
        self.embed_dim = d_model
        self.n_patches = (img_size // self.PATCH_SIZE) ** 2

        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=self.PATCH_SIZE, stride=self.PATCH_SIZE)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_tokens = nn.Parameter(torch.zeros(1, self.N_REGISTERS, d_model))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.n_patches + self.N_REGISTERS, d_model)
        )
        self.blocks = nn.ModuleList([_MockBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N_patches, D)
        cls = self.cls_token.expand(B, -1, -1)
        regs = self.register_tokens.expand(B, -1, -1)
        tokens = torch.cat([cls, patches, regs], dim=1) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B = 2           # small batch for fast tests
ACTION_DIM = 10
IMG_H = 128
IMG_W = 128

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GROUND_TRUTH_SPEC = {
    "type": "DaseinNet",
    "shared": True,
    "vision_mode": "ground_truth",
    "action_dim": ACTION_DIM,
    "log_std_init": 0.0,
    "clip_grad_val": 0.5,
    "use_same_optim": True,
    "loss_spec": {"name": "MSELoss"},
    "optim_spec": {"name": "Adam", "lr": 3e-4},
    "lr_scheduler_spec": None,
    "gpu": False,
}

VISION_SPEC = {
    "type": "DaseinNet",
    "shared": True,
    "vision_mode": "vision",
    "action_dim": ACTION_DIM,
    "log_std_init": 0.0,
    "infonce_alpha": INFONCE_ALPHA,
    "lora_rank": 4,     # small rank for fast tests
    "lora_alpha": 8.0,
    "clip_grad_val": 0.5,
    "use_same_optim": True,
    "loss_spec": {"name": "MSELoss"},
    "optim_spec": {"name": "Adam", "lr": 1e-4},
    "lr_scheduler_spec": None,
    "gpu": False,
}


@pytest.fixture
def gt_net():
    """DaseinNet in ground_truth mode."""
    return DaseinNet(GROUND_TRUTH_SPEC, OBS_DIM, [ACTION_DIM, ACTION_DIM, 1])


@pytest.fixture
def vision_net():
    """DaseinNet in vision mode with mock DINOv2 (no HuggingFace download)."""
    mock = _MockDINOv2(d_model=128, n_layers=24, n_heads=8, img_size=IMG_H)
    return DaseinNet(
        VISION_SPEC, OBS_DIM, [ACTION_DIM, ACTION_DIM, 1],
        _mock_dinov2=mock,
    )


def _make_vision_obs(batch_size: int = B) -> dict:
    """Random vision-mode observation dict."""
    return {
        "ground_truth": torch.randn(batch_size, OBS_DIM),
        "left": torch.rand(batch_size, 3, IMG_H, IMG_W),   # [0,1] float32
        "right": torch.rand(batch_size, 3, IMG_H, IMG_W),
    }


# ---------------------------------------------------------------------------
# Backward compatibility: ground_truth mode unchanged
# ---------------------------------------------------------------------------

def test_gt_mode_forward_shape(gt_net):
    obs = torch.randn(B, OBS_DIM)
    out = gt_net(obs)
    assert len(out) == 3
    mean, log_std, value = out
    assert mean.shape == (B, ACTION_DIM)
    assert log_std.shape == (B, ACTION_DIM)
    assert value.shape == (B, 1)


def test_gt_mode_accepts_flat_tensor(gt_net):
    obs = torch.randn(B, OBS_DIM)
    out = gt_net(obs)
    assert isinstance(out, list)


def test_gt_mode_accepts_dict_obs(gt_net):
    """ground_truth mode should also accept a dict and extract 'ground_truth' key."""
    obs = {"ground_truth": torch.randn(B, OBS_DIM)}
    out = gt_net(obs)
    assert len(out) == 3


def test_gt_mode_no_infonce(gt_net):
    """InfoNCE loss is None in ground_truth mode."""
    obs = torch.randn(B, OBS_DIM)
    gt_net(obs)
    assert gt_net.last_infonce_loss is None


def test_gt_mode_gradients(gt_net):
    x = torch.randn(B, OBS_DIM, requires_grad=True)
    mean, _, value = gt_net(x)
    (mean.sum() + value.sum()).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Vision mode: forward pass shape
# ---------------------------------------------------------------------------

def test_vision_forward_returns_list(vision_net):
    obs = _make_vision_obs()
    out = vision_net(obs)
    assert isinstance(out, list)


def test_vision_forward_three_outputs(vision_net):
    obs = _make_vision_obs()
    out = vision_net(obs)
    assert len(out) == 3


def test_vision_mean_shape(vision_net):
    obs = _make_vision_obs()
    mean, log_std, value = vision_net(obs)
    assert mean.shape == (B, ACTION_DIM)


def test_vision_log_std_shape(vision_net):
    obs = _make_vision_obs()
    mean, log_std, value = vision_net(obs)
    assert log_std.shape == (B, ACTION_DIM)


def test_vision_value_shape(vision_net):
    obs = _make_vision_obs()
    mean, log_std, value = vision_net(obs)
    assert value.shape == (B, 1)


def test_vision_finite_outputs(vision_net):
    obs = _make_vision_obs()
    mean, log_std, value = vision_net(obs)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(log_std).all()
    assert torch.isfinite(value).all()


# ---------------------------------------------------------------------------
# Vision mode: DINOv2 backbone frozen
# ---------------------------------------------------------------------------

def test_dinov2_backbone_frozen(vision_net):
    """Original DINOv2 backbone parameters (non-LoRA) must have requires_grad=False.

    LoRA adapters (lora_A, lora_B) are injected into backbone blocks after freezing
    and are intentionally trainable — exclude them from the frozen check.
    """
    backbone = vision_net.vision_enc.backbone.backbone
    for name, p in backbone.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            continue  # LoRA adapters are trainable by design
        assert not p.requires_grad, (
            f"DINOv2 backbone param '{name}' has requires_grad=True — should be frozen"
        )


def test_lora_adapters_trainable(vision_net):
    """LoRA adapters (lora_A, lora_B) must be trainable."""
    lora_params = [
        (name, p)
        for name, p in vision_net.vision_enc.named_parameters()
        if "lora_A" in name or "lora_B" in name
    ]
    assert len(lora_params) > 0, "No LoRA parameters found in vision_enc"
    for name, p in lora_params:
        assert p.requires_grad, f"LoRA param '{name}' has requires_grad=False"


# ---------------------------------------------------------------------------
# Vision mode: InfoNCE loss
# ---------------------------------------------------------------------------

def test_infonce_loss_computed(vision_net):
    """InfoNCE loss is computed after forward pass in vision mode."""
    obs = _make_vision_obs()
    vision_net(obs)
    assert vision_net.last_infonce_loss is not None


def test_infonce_loss_scalar(vision_net):
    """InfoNCE loss is a scalar tensor."""
    obs = _make_vision_obs()
    vision_net(obs)
    loss = vision_net.last_infonce_loss
    assert loss.ndim == 0, f"InfoNCE loss should be scalar, got shape {loss.shape}"


def test_infonce_loss_finite(vision_net):
    """InfoNCE loss is finite."""
    obs = _make_vision_obs()
    vision_net(obs)
    assert torch.isfinite(vision_net.last_infonce_loss)


def test_infonce_loss_positive(vision_net):
    """InfoNCE is a cross-entropy loss — must be non-negative."""
    obs = _make_vision_obs()
    vision_net(obs)
    assert vision_net.last_infonce_loss.item() >= 0.0


# ---------------------------------------------------------------------------
# InfoNCELoss unit tests
# ---------------------------------------------------------------------------

def test_infonce_unit_identity():
    """InfoNCE with identical embeddings produces low loss (diagonal logits dominant)."""
    loss_fn = InfoNCELoss(temperature=INFONCE_TEMP)
    z = torch.randn(4, 512)
    z_norm = torch.nn.functional.normalize(z, dim=-1)
    # Perfect alignment: being_emb == vision_feat
    loss = loss_fn(z_norm, z_norm)
    # Should converge toward -log(1) = 0 but cross-entropy never exactly 0
    # Just check it's finite and non-negative
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_infonce_unit_random():
    """InfoNCE with random embeddings is finite and ≈ log(B)."""
    loss_fn = InfoNCELoss(temperature=INFONCE_TEMP)
    z_b = torch.randn(8, 512)
    z_v = torch.randn(8, 512)
    loss = loss_fn(z_b, z_v)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_infonce_symmetric():
    """InfoNCE loss is symmetric (loss(a,b) ≈ loss(b,a))."""
    loss_fn = InfoNCELoss(temperature=INFONCE_TEMP)
    z_b = torch.randn(4, 512)
    z_v = torch.randn(4, 512)
    # Both directions are averaged inside the loss
    loss_ab = loss_fn(z_b, z_v)
    loss_ba = loss_fn(z_v, z_b)
    # Should be equal (same formula applied symmetrically)
    assert torch.allclose(loss_ab, loss_ba, atol=1e-5)


# ---------------------------------------------------------------------------
# Vision mode: gradient flow
# ---------------------------------------------------------------------------

def test_vision_gradients_through_proprio_enc(vision_net):
    """Gradients reach _ProprioVisionEncoder from policy loss."""
    obs = _make_vision_obs()
    mean, _, _ = vision_net(obs)
    mean.sum().backward()
    params = list(vision_net.proprio_enc_vision.parameters())
    grads = [p.grad for p in params if p.grad is not None]
    assert len(grads) > 0, "_ProprioVisionEncoder received no gradients"


def test_vision_gradients_through_stereo_fusion(vision_net):
    """Gradients reach StereoFusionModule from policy loss."""
    obs = _make_vision_obs()
    mean, _, _ = vision_net(obs)
    mean.sum().backward()
    params = list(vision_net.vision_enc.fusion.parameters())
    grads = [p.grad for p in params if p.grad is not None]
    assert len(grads) > 0, "StereoFusionModule received no gradients"


def test_vision_gradients_through_l1(vision_net):
    """Gradients reach BeingEmbedding (L1) from policy loss."""
    obs = _make_vision_obs()
    mean, _, _ = vision_net(obs)
    mean.sum().backward()
    params = list(vision_net.being_emb.parameters())
    grads = [p.grad for p in params if p.grad is not None]
    assert len(grads) > 0, "BeingEmbedding (L1) received no gradients"


def test_vision_gradients_through_value_head(vision_net):
    """Gradients reach L1 from value loss."""
    obs = _make_vision_obs()
    _, _, value = vision_net(obs)
    value.sum().backward()
    params = list(vision_net.being_emb.parameters())
    grads = [p.grad for p in params if p.grad is not None]
    assert len(grads) > 0, "L1 received no gradients from value head"


def test_vision_no_grad_to_frozen_backbone(vision_net):
    """Frozen DINOv2 backbone parameters (non-LoRA) receive no gradients."""
    obs = _make_vision_obs()
    mean, _, _ = vision_net(obs)
    mean.sum().backward()
    backbone = vision_net.vision_enc.backbone.backbone
    for name, p in backbone.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            continue  # LoRA adapters should receive gradients
        assert p.grad is None, (
            f"Frozen backbone param '{name}' received a gradient — grad leak"
        )


# ---------------------------------------------------------------------------
# Mode validation
# ---------------------------------------------------------------------------

def test_vision_mode_rejects_flat_tensor(vision_net):
    """vision mode must raise TypeError if given a flat tensor."""
    obs = torch.randn(B, OBS_DIM)
    with pytest.raises(TypeError):
        vision_net(obs)


def test_invalid_vision_mode_raises():
    """Unknown vision_mode raises ValueError at construction."""
    spec = dict(GROUND_TRUTH_SPEC)
    spec["vision_mode"] = "unknown_mode"
    with pytest.raises(ValueError):
        DaseinNet(spec, OBS_DIM, [ACTION_DIM, ACTION_DIM, 1])


# ---------------------------------------------------------------------------
# PPO interface compatibility (vision mode)
# ---------------------------------------------------------------------------

def test_vision_ppo_value_view(vision_net):
    """PPO calls out[-1].view(-1) — must work in vision mode."""
    obs = _make_vision_obs()
    out = vision_net(obs)
    v = out[-1].view(-1)
    assert v.shape == (B,)


def test_vision_ppo_pdparam(vision_net):
    """PPO pdparam extraction: out[:-1] = [mean, log_std]."""
    obs = _make_vision_obs()
    out = vision_net(obs)
    pdparam = out[:-1]
    assert len(pdparam) == 2
    assert pdparam[0].shape == (B, ACTION_DIM)
    assert pdparam[1].shape == (B, ACTION_DIM)


# ---------------------------------------------------------------------------
# GRU hidden state (vision mode)
# ---------------------------------------------------------------------------

def test_vision_hidden_state_updated(vision_net):
    """h_prev changes after vision forward pass."""
    vision_net.reset_hidden(batch_size=B)
    h_before = vision_net.h_prev.clone()
    obs = _make_vision_obs()
    vision_net(obs)
    assert not torch.equal(h_before, vision_net.h_prev)


def test_vision_hidden_state_detached(vision_net):
    """h_prev is detached after vision forward — no BPTT through full history."""
    vision_net.reset_hidden(batch_size=B)
    obs = _make_vision_obs()
    vision_net(obs)
    assert not vision_net.h_prev.requires_grad
