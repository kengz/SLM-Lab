"""Tests for L0 perception encoders (Phase 3.2a ground-truth mode).

Covers: shapes, forward pass, gradients, L0Output channel stack.
"""

import pytest
import torch
import torch.nn as nn

from slm_lab.agent.net.perception import (
    L0Output,
    ObjectStateEncoder,
    ProprioceptionEncoder,
    _encode_flat,
    scientific_encode,
)

B = 4          # batch size
N_OBJ = 5     # default max_objects


# ---------------------------------------------------------------------------
# scientific_encode
# ---------------------------------------------------------------------------

def test_scientific_encode_shape():
    x = torch.randn(B, 10)
    out = scientific_encode(x)
    assert out.shape == (B, 10, 2)


def test_scientific_encode_mantissa_range():
    x = torch.randn(B, 10) * 10
    out = scientific_encode(x)
    mantissa = out[..., 0]
    assert mantissa.min() >= -1.0 - 1e-6
    assert mantissa.max() <= 1.0 + 1e-6


def test_scientific_encode_exponent_range():
    x = torch.randn(B, 10) * 10
    out = scientific_encode(x)
    exponent = out[..., 1]
    assert exponent.min() >= 0.0 - 1e-6
    assert exponent.max() <= 1.0 + 1e-6


def test_encode_flat_shape():
    x = torch.randn(B, 25)
    out = _encode_flat(x)
    assert out.shape == (B, 50)


# ---------------------------------------------------------------------------
# ProprioceptionEncoder
# ---------------------------------------------------------------------------

@pytest.fixture
def proprio_inputs():
    proprio  = torch.randn(B, 25)
    tactile  = torch.rand(B, 2)        # binary-ish [0,1]
    ee       = torch.randn(B, 6)
    internal = torch.randn(B, 2)
    return proprio, tactile, ee, internal


@pytest.fixture
def proprio_encoder():
    return ProprioceptionEncoder()


def test_proprio_encoder_is_module(proprio_encoder):
    assert isinstance(proprio_encoder, nn.Module)


def test_proprio_encoder_output_shape(proprio_encoder, proprio_inputs):
    out = proprio_encoder(*proprio_inputs)
    assert out.shape == (B, 512)


def test_proprio_encoder_output_dtype(proprio_encoder, proprio_inputs):
    out = proprio_encoder(*proprio_inputs)
    assert out.dtype == torch.float32


def test_proprio_encoder_no_nan(proprio_encoder, proprio_inputs):
    out = proprio_encoder(*proprio_inputs)
    assert not torch.isnan(out).any()


def test_proprio_encoder_gradients(proprio_encoder, proprio_inputs):
    inputs = [x.requires_grad_(True) for x in proprio_inputs]
    out = proprio_encoder(*inputs)
    loss = out.sum()
    loss.backward()
    for inp in inputs:
        assert inp.grad is not None
        assert not torch.isnan(inp.grad).any()


def test_proprio_encoder_batch_independence(proprio_encoder, proprio_inputs):
    """Different batch elements should produce independent outputs."""
    out_full = proprio_encoder(*proprio_inputs)
    # Run only first element
    single_inputs = [x[0:1] for x in proprio_inputs]
    out_single = proprio_encoder(*single_inputs)
    assert torch.allclose(out_full[0:1], out_single, atol=1e-5)


def test_proprio_encoder_param_count(proprio_encoder):
    n_params = sum(p.numel() for p in proprio_encoder.parameters())
    # Spec says ~0.5M — allow generous range
    assert 100_000 < n_params < 2_000_000, f"param count {n_params} out of expected range"


# ---------------------------------------------------------------------------
# ObjectStateEncoder
# ---------------------------------------------------------------------------

@pytest.fixture
def obj_encoder():
    return ObjectStateEncoder(max_objects=N_OBJ)


@pytest.fixture
def obj_input():
    return torch.randn(B, 7 * N_OBJ)


def test_obj_encoder_is_module(obj_encoder):
    assert isinstance(obj_encoder, nn.Module)


def test_obj_encoder_output_shape(obj_encoder, obj_input):
    out = obj_encoder(obj_input)
    assert out.shape == (B, 512)


def test_obj_encoder_output_dtype(obj_encoder, obj_input):
    out = obj_encoder(obj_input)
    assert out.dtype == torch.float32


def test_obj_encoder_no_nan(obj_encoder, obj_input):
    out = obj_encoder(obj_input)
    assert not torch.isnan(out).any()


def test_obj_encoder_gradients(obj_encoder, obj_input):
    x = obj_input.requires_grad_(True)
    out = obj_encoder(x)
    out.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_obj_encoder_custom_n_objects():
    enc = ObjectStateEncoder(max_objects=3)
    x = torch.randn(B, 7 * 3)
    out = enc(x)
    assert out.shape == (B, 512)


def test_obj_encoder_permutation_invariant(obj_encoder):
    """Max-pool should be permutation invariant over objects."""
    x = torch.randn(B, 7 * N_OBJ)
    out_orig = obj_encoder(x)

    # Shuffle object order along dim-1 in groups of 7
    x_reshaped = x.view(B, N_OBJ, 7)
    # Reverse object order
    idx = torch.arange(N_OBJ - 1, -1, -1)
    x_perm = x_reshaped[:, idx, :].reshape(B, 7 * N_OBJ)
    out_perm = obj_encoder(x_perm)

    assert torch.allclose(out_orig, out_perm, atol=1e-5)


# ---------------------------------------------------------------------------
# L0Output
# ---------------------------------------------------------------------------

def _make_feat(batch=B, dim=512):
    return torch.randn(batch, dim)


def test_l0output_channel_stack_proprio_only():
    out = L0Output(proprioception=_make_feat())
    stack = out.to_channel_stack()
    assert stack.shape == (B, 1, 512)


def test_l0output_channel_stack_with_object_state():
    out = L0Output(
        proprioception=_make_feat(),
        object_state=_make_feat(),
    )
    stack = out.to_channel_stack()
    assert stack.shape == (B, 2, 512)


def test_l0output_channel_stack_phase_32a():
    """Phase 3.2a: proprio + object_state, no vision/audio."""
    out = L0Output(
        proprioception=_make_feat(),
        object_state=_make_feat(),
    )
    stack = out.to_channel_stack()
    assert stack.shape == (B, 2, 512)
    # First channel is proprioception
    assert torch.equal(stack[:, 0, :], out.proprioception)
    # Second is object state
    assert torch.equal(stack[:, 1, :], out.object_state)


def test_l0output_channel_stack_all_channels():
    out = L0Output(
        proprioception=_make_feat(),
        vision=_make_feat(),
        audio=_make_feat(),
        object_state=_make_feat(),
    )
    stack = out.to_channel_stack()
    assert stack.shape == (B, 4, 512)


def test_l0output_channel_stack_phase_32b():
    """Phase 3.2b+: proprio + vision + audio, no object_state."""
    out = L0Output(
        proprioception=_make_feat(),
        vision=_make_feat(),
        audio=_make_feat(),
    )
    stack = out.to_channel_stack()
    assert stack.shape == (B, 3, 512)


def test_l0output_channel_stack_last_dim():
    out = L0Output(
        proprioception=_make_feat(),
        object_state=_make_feat(),
    )
    stack = out.to_channel_stack()
    assert stack.shape[-1] == 512


def test_l0output_channel_stack_no_nan():
    out = L0Output(
        proprioception=_make_feat(),
        object_state=_make_feat(),
    )
    assert not torch.isnan(out.to_channel_stack()).any()


# ---------------------------------------------------------------------------
# Integration: ProprioceptionEncoder → L0Output → to_channel_stack
# ---------------------------------------------------------------------------

def test_integration_proprio_to_l0output(proprio_encoder, proprio_inputs, obj_encoder, obj_input):
    proprio_feat = proprio_encoder(*proprio_inputs)
    obj_feat = obj_encoder(obj_input)

    l0 = L0Output(proprioception=proprio_feat, object_state=obj_feat)
    stack = l0.to_channel_stack()

    assert stack.shape == (B, 2, 512)
    assert not torch.isnan(stack).any()


def test_integration_gradients_flow_through_stack(proprio_encoder, proprio_inputs, obj_encoder, obj_input):
    """Gradients flow from channel stack back through both encoders."""
    inputs_grad = [x.requires_grad_(True) for x in proprio_inputs]
    obj_input_grad = obj_input.requires_grad_(True)

    proprio_feat = proprio_encoder(*inputs_grad)
    obj_feat = obj_encoder(obj_input_grad)

    l0 = L0Output(proprioception=proprio_feat, object_state=obj_feat)
    stack = l0.to_channel_stack()
    stack.sum().backward()

    for inp in inputs_grad:
        assert inp.grad is not None
    assert obj_input_grad.grad is not None
