"""Tests for DaseinNet — L0 + L1 + PPO-compatible policy/value heads.

Coverage:
- Forward pass with 56-dim obs produces correct output shapes
- act: produces valid 10-dim actions via Gaussian sampling
- Value estimate is scalar per sample
- Gradients flow through L0 → L1 → policy and value branches
- Log prob computation works (needed for PPO ratio)
- GRU hidden state management (reset, carry-forward)
- output list format compatible with PPO (shared=True convention)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.distributions import Normal

from slm_lab.agent.net.dasein_net import DaseinNet, OBS_DIM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B = 4           # batch size
ACTION_DIM = 10 # sensorimotor action dim


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NET_SPEC = {
    "type": "DaseinNet",
    "shared": True,
    "action_dim": ACTION_DIM,
    "log_std_init": 0.0,
    "clip_grad_val": 0.5,
    "use_same_optim": True,
    "loss_spec": {"name": "MSELoss"},
    "optim_spec": {"name": "Adam", "lr": 3e-4},
    "lr_scheduler_spec": None,
    "gpu": False,
}


@pytest.fixture
def net():
    """DaseinNet with sensorimotor action dim."""
    in_dim = OBS_DIM
    out_dim = [ACTION_DIM, ACTION_DIM, 1]   # [mean_dim, log_std_dim, value_dim]
    return DaseinNet(NET_SPEC, in_dim, out_dim)


@pytest.fixture
def obs():
    """Random 56-dim batch observation."""
    return torch.randn(B, OBS_DIM)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_forward_returns_list(net, obs):
    out = net(obs)
    assert isinstance(out, list), f"forward must return list, got {type(out)}"


def test_forward_three_outputs(net, obs):
    out = net(obs)
    assert len(out) == 3, f"expected 3 outputs [mean, log_std, value], got {len(out)}"


def test_mean_shape(net, obs):
    mean, log_std, value = net(obs)
    assert mean.shape == (B, ACTION_DIM), f"mean shape {mean.shape} != ({B}, {ACTION_DIM})"


def test_log_std_shape(net, obs):
    mean, log_std, value = net(obs)
    assert log_std.shape == (B, ACTION_DIM), f"log_std shape {log_std.shape} != ({B}, {ACTION_DIM})"


def test_value_shape(net, obs):
    mean, log_std, value = net(obs)
    # PPO calls out[-1].view(-1), so must be (B, 1) or broadcastable
    assert value.shape == (B, 1), f"value shape {value.shape} != ({B}, 1)"


# ---------------------------------------------------------------------------
# PPO interface compatibility
# ---------------------------------------------------------------------------

def test_ppo_pdparam_extraction(net, obs):
    """PPO calc_pdparam: out[:-1] = [mean, log_std], out[-1] = value."""
    out = net(obs)
    pdparam = out[:-1]
    v = out[-1].view(-1)
    assert len(pdparam) == 2
    assert pdparam[0].shape == (B, ACTION_DIM)
    assert pdparam[1].shape == (B, ACTION_DIM)
    assert v.shape == (B,)


def test_value_view_minus1(net, obs):
    """PPO calls net(x)[-1].view(-1) for value."""
    out = net(obs)
    v = out[-1].view(-1)
    assert v.shape == (B,)


# ---------------------------------------------------------------------------
# Action sampling
# ---------------------------------------------------------------------------

def test_act_shape_via_distribution(net, obs):
    """Actions sampled from Gaussian on mean/log_std are 10-dim."""
    mean, log_std, _ = net(obs)
    std = log_std.exp()
    dist = Normal(mean, std)
    action = dist.sample()
    assert action.shape == (B, ACTION_DIM)


def test_act_finite(net, obs):
    """Sampled actions must be finite."""
    mean, log_std, _ = net(obs)
    std = log_std.exp().clamp(min=1e-6)
    dist = Normal(mean, std)
    action = dist.sample()
    assert torch.isfinite(action).all()


# ---------------------------------------------------------------------------
# Log prob computation (required for PPO ratio)
# ---------------------------------------------------------------------------

def test_log_prob_shape(net, obs):
    """log_prob per action element: (B, A)."""
    mean, log_std, _ = net(obs)
    std = log_std.exp().clamp(min=1e-6)
    dist = Normal(mean, std)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)
    assert log_probs.shape == (B, ACTION_DIM)


def test_log_prob_finite(net, obs):
    """log_prob values must be finite."""
    mean, log_std, _ = net(obs)
    std = log_std.exp().clamp(min=1e-6)
    dist = Normal(mean, std)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)
    assert torch.isfinite(log_probs).all()


def test_log_prob_reduced_shape(net, obs):
    """Sum-reduced log_prob → (B,) for PPO."""
    mean, log_std, _ = net(obs)
    std = log_std.exp().clamp(min=1e-6)
    dist = Normal(mean, std)
    actions = dist.sample()
    log_probs = dist.log_prob(actions).sum(dim=-1)
    assert log_probs.shape == (B,)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradients_flow_through_policy_head(net, obs):
    """Gradients reach L0 ProprioceptionEncoder from policy loss."""
    x = obs.requires_grad_(True)
    mean, log_std, _ = net(x)
    loss = mean.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_gradients_flow_through_value_head(net, obs):
    """Gradients reach L0 from value loss."""
    x = obs.requires_grad_(True)
    _, _, value = net(x)
    loss = value.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_gradients_flow_through_l1(net, obs):
    """Policy loss propagates through BeingEmbedding (L1)."""
    # Verify L1 parameters have gradients after backward
    x = obs.requires_grad_(True)
    mean, _, _ = net(x)
    mean.sum().backward()
    l1_params = list(net.being_emb.parameters())
    assert len(l1_params) > 0
    grads = [p.grad for p in l1_params if p.grad is not None]
    assert len(grads) > 0, "No L1 parameters received gradients"


def test_gradients_flow_through_l0_proprio(net, obs):
    """Value loss propagates through ProprioceptionEncoder (L0)."""
    x = obs.requires_grad_(True)
    _, _, value = net(x)
    value.sum().backward()
    l0_params = list(net.proprio_enc.parameters())
    grads = [p.grad for p in l0_params if p.grad is not None]
    assert len(grads) > 0, "No L0 proprio encoder parameters received gradients"


def test_gradients_flow_through_l0_obj(net, obs):
    """Value loss propagates through ObjectStateEncoder (L0)."""
    x = obs.requires_grad_(True)
    _, _, value = net(x)
    value.sum().backward()
    l0_params = list(net.obj_enc.parameters())
    grads = [p.grad for p in l0_params if p.grad is not None]
    assert len(grads) > 0, "No L0 object encoder parameters received gradients"


# ---------------------------------------------------------------------------
# No NaN / inf in outputs
# ---------------------------------------------------------------------------

def test_no_nan_in_mean(net, obs):
    mean, _, _ = net(obs)
    assert not torch.isnan(mean).any()


def test_no_nan_in_value(net, obs):
    _, _, value = net(obs)
    assert not torch.isnan(value).any()


def test_no_inf_in_outputs(net, obs):
    mean, log_std, value = net(obs)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(log_std).all()
    assert torch.isfinite(value).all()


# ---------------------------------------------------------------------------
# GRU hidden state management
# ---------------------------------------------------------------------------

def test_reset_hidden_zeroes(net):
    """reset_hidden zeroes the GRU hidden state."""
    net.h_prev = torch.ones(1, 1024)
    net.reset_hidden(batch_size=1)
    assert net.h_prev.abs().max().item() == 0.0


def test_hidden_state_updated_on_forward(net, obs):
    """h_prev changes after a forward pass (GRU updated)."""
    net.reset_hidden(batch_size=B)
    h_before = net.h_prev.clone()
    net(obs)
    h_after = net.h_prev
    # h_prev should change from zero after forward
    assert not torch.equal(h_before, h_after)


def test_hidden_state_detached(net, obs):
    """h_prev must be detached — no grad accumulation between steps."""
    net.reset_hidden(batch_size=B)
    net(obs)
    assert not net.h_prev.requires_grad


# ---------------------------------------------------------------------------
# Module structure
# ---------------------------------------------------------------------------

def test_is_nn_module(net):
    assert isinstance(net, nn.Module)


def test_has_l0_encoders(net):
    assert hasattr(net, "proprio_enc")
    assert hasattr(net, "obj_enc")


def test_has_l1_being_emb(net):
    assert hasattr(net, "being_emb")


def test_has_policy_and_value_heads(net):
    assert hasattr(net, "mean_head")
    assert hasattr(net, "value_head")
    assert hasattr(net, "log_std")
    assert isinstance(net.log_std, nn.Parameter)


def test_log_std_is_learnable(net):
    assert net.log_std.requires_grad


def test_param_count_reasonable(net):
    """DaseinNet should have ~20M params (L1 dominates at ~19.2M)."""
    n = sum(p.numel() for p in net.parameters())
    assert 10_000_000 < n < 100_000_000, f"param count {n} outside expected range"


# ---------------------------------------------------------------------------
# Batch independence
# ---------------------------------------------------------------------------

def test_batch_independence(net, obs):
    """Different batch elements produce independent outputs."""
    out_full = net(obs)
    mean_full = out_full[0]

    # Single element
    obs_single = obs[0:1]
    net.reset_hidden(batch_size=1)
    out_single = net(obs_single)
    mean_single = out_single[0]

    assert torch.allclose(mean_full[0:1], mean_single, atol=1e-4), (
        "Batch element 0 should match single forward pass"
    )
