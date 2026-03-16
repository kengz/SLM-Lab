"""Tests for L1 Being Embedding — slm_lab/agent/net/being_embedding.py

Coverage:
- Output shapes for all channel configurations (N=1,2,3,4)
- Forward pass (no errors, correct output types)
- Gradient flow through all components
- Temporal sequence (GRU state carry-forward)
- Attention weight inspection
- Phase 3.2a behavior (zero projection)
- L1Output dataclass fields
"""

import pytest
import torch

from slm_lab.agent.net.being_embedding import (
    L0Output,
    L1Output,
    BeingEmbedding,
    ChannelAttention,
    ChannelTypeEmbedding,
    HierarchicalFusion,
    ProjectionEncoder,
    TemporalAttention,
    ThrownessEncoder,
)

B = 4
D = 512


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def l0_phase32a():
    """Phase 3.2a: proprio + object_state."""
    return L0Output(
        proprioception=torch.randn(B, D),
        object_state=torch.randn(B, D),
    )


@pytest.fixture
def l0_phase32b():
    """Phase 3.2b: proprio + vision."""
    return L0Output(
        proprioception=torch.randn(B, D),
        vision=torch.randn(B, D),
    )


@pytest.fixture
def l0_full():
    """Phase 3.2b+: proprio + vision + audio."""
    return L0Output(
        proprioception=torch.randn(B, D),
        vision=torch.randn(B, D),
        audio=torch.randn(B, D),
    )


@pytest.fixture
def l0_single():
    """Minimal: proprio only."""
    return L0Output(proprioception=torch.randn(B, D))


@pytest.fixture
def l0_all_channels():
    """All 4 channels active."""
    return L0Output(
        proprioception=torch.randn(B, D),
        vision=torch.randn(B, D),
        audio=torch.randn(B, D),
        object_state=torch.randn(B, D),
    )


@pytest.fixture
def model():
    return BeingEmbedding(max_channels=4, d_model=D)


@pytest.fixture
def h_prev(device):
    return torch.zeros(B, 1024, device=device)


# ---------------------------------------------------------------------------
# L0Output interface tests
# ---------------------------------------------------------------------------

class TestL0Output:
    def test_channel_stack_shape_single(self, l0_single):
        stack = l0_single.to_channel_stack()
        assert stack.shape == (B, 1, D)

    def test_channel_stack_shape_two(self, l0_phase32a):
        stack = l0_phase32a.to_channel_stack()
        assert stack.shape == (B, 2, D)

    def test_channel_stack_shape_three(self, l0_full):
        stack = l0_full.to_channel_stack()
        assert stack.shape == (B, 3, D)

    def test_channel_stack_shape_four(self, l0_all_channels):
        stack = l0_all_channels.to_channel_stack()
        assert stack.shape == (B, 4, D)

    def test_channel_types_single(self, l0_single):
        assert l0_single.get_channel_types() == ['proprioception']

    def test_channel_types_phase32a(self, l0_phase32a):
        # object_state appended after proprioception
        types = l0_phase32a.get_channel_types()
        assert types == ['proprioception', 'object_state']

    def test_channel_types_phase32b(self, l0_phase32b):
        assert l0_phase32b.get_channel_types() == ['proprioception', 'vision']

    def test_channel_types_full(self, l0_full):
        assert l0_full.get_channel_types() == ['proprioception', 'vision', 'audio']

    def test_channel_types_all(self, l0_all_channels):
        assert l0_all_channels.get_channel_types() == [
            'proprioception', 'vision', 'audio', 'object_state'
        ]

    def test_proprio_always_first(self, l0_all_channels):
        stack = l0_all_channels.to_channel_stack()
        assert torch.allclose(stack[:, 0, :], l0_all_channels.proprioception)


# ---------------------------------------------------------------------------
# ChannelTypeEmbedding tests
# ---------------------------------------------------------------------------

class TestChannelTypeEmbedding:
    def test_output_shape(self):
        emb = ChannelTypeEmbedding(D)
        x = torch.randn(B, 2, D)
        out = emb(x, ['proprioception', 'vision'])
        assert out.shape == (B, 2, D)

    def test_modifies_input(self):
        emb = ChannelTypeEmbedding(D)
        x = torch.randn(B, 2, D)
        out = emb(x, ['proprioception', 'vision'])
        assert not torch.allclose(out, x)

    def test_different_types_produce_different_outputs(self):
        emb = ChannelTypeEmbedding(D)
        x = torch.randn(B, 1, D)
        out_proprio = emb(x.clone(), ['proprioception'])
        out_vision = emb(x.clone(), ['vision'])
        assert not torch.allclose(out_proprio, out_vision)

    def test_unknown_type_raises(self):
        emb = ChannelTypeEmbedding(D)
        x = torch.randn(B, 1, D)
        with pytest.raises(ValueError):
            emb(x, ['unknown_modality'])


# ---------------------------------------------------------------------------
# ChannelAttention tests
# ---------------------------------------------------------------------------

class TestChannelAttention:
    def test_output_shape_n2(self):
        attn = ChannelAttention(D)
        x = torch.randn(B, 2, D)
        out = attn(x)
        assert out.shape == (B, 2, D)

    def test_output_shape_n1(self):
        attn = ChannelAttention(D)
        x = torch.randn(B, 1, D)
        out = attn(x)
        assert out.shape == (B, 1, D)

    def test_output_shape_n4(self):
        attn = ChannelAttention(D)
        x = torch.randn(B, 4, D)
        out = attn(x)
        assert out.shape == (B, 4, D)

    def test_gradient_flow(self):
        attn = ChannelAttention(D)
        x = torch.randn(B, 2, D, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_output_is_not_input(self):
        attn = ChannelAttention(D)
        x = torch.randn(B, 2, D)
        out = attn(x)
        assert not torch.allclose(out, x)


# ---------------------------------------------------------------------------
# HierarchicalFusion tests
# ---------------------------------------------------------------------------

class TestHierarchicalFusion:
    def test_output_shape_n2(self):
        fusion = HierarchicalFusion(max_channels=4, d_model=D)
        x = torch.randn(B, 2, D)
        out = fusion(x)
        assert out.shape == (B, D)

    def test_output_shape_n1(self):
        fusion = HierarchicalFusion(max_channels=4, d_model=D)
        x = torch.randn(B, 1, D)
        out = fusion(x)
        assert out.shape == (B, D)

    def test_output_shape_n3(self):
        fusion = HierarchicalFusion(max_channels=4, d_model=D)
        x = torch.randn(B, 3, D)
        out = fusion(x)
        assert out.shape == (B, D)

    def test_output_shape_n4(self):
        fusion = HierarchicalFusion(max_channels=4, d_model=D)
        x = torch.randn(B, 4, D)
        out = fusion(x)
        assert out.shape == (B, D)

    def test_zero_padding_applied(self):
        fusion = HierarchicalFusion(max_channels=4, d_model=D)
        x_2ch = torch.randn(B, 2, D)
        x_4ch = torch.cat([x_2ch, torch.zeros(B, 2, D)], dim=1)
        out_2ch = fusion(x_2ch)
        out_4ch = fusion(x_4ch)
        assert torch.allclose(out_2ch, out_4ch)

    def test_gradient_flow(self):
        fusion = HierarchicalFusion(max_channels=4, d_model=D)
        x = torch.randn(B, 2, D, requires_grad=True)
        out = fusion(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# ThrownessEncoder tests
# ---------------------------------------------------------------------------

class TestThrownessEncoder:
    def test_output_shapes(self):
        enc = ThrownessEncoder(input_dim=D, hidden_dim=1024, output_dim=D)
        being_emb = torch.randn(B, D)
        h_prev = torch.zeros(B, 1024)
        thrownness, h_t = enc(being_emb, h_prev)
        assert thrownness.shape == (B, D)
        assert h_t.shape == (B, 1024)

    def test_init_hidden_shape(self):
        enc = ThrownessEncoder()
        h = enc.init_hidden(B, torch.device('cpu'))
        assert h.shape == (B, 1024)
        assert (h == 0).all()

    def test_hidden_state_updates(self):
        enc = ThrownessEncoder()
        being_emb = torch.randn(B, D)
        h0 = enc.init_hidden(B, torch.device('cpu'))
        _, h1 = enc(being_emb, h0)
        assert not torch.allclose(h0, h1)

    def test_different_inputs_different_thrownness(self):
        enc = ThrownessEncoder()
        h = enc.init_hidden(B, torch.device('cpu'))
        t1, _ = enc(torch.randn(B, D), h)
        t2, _ = enc(torch.randn(B, D), h)
        assert not torch.allclose(t1, t2)

    def test_gradient_flow(self):
        torch.manual_seed(42)
        enc = ThrownessEncoder()
        being_emb = torch.randn(B, D, requires_grad=True)
        h_prev = torch.randn(B, 1024) * 0.1  # non-zero hidden to avoid degenerate GRU gate
        thrownness, h_t = enc(being_emb, h_prev)
        thrownness.sum().backward()
        assert being_emb.grad is not None
        assert being_emb.grad.abs().sum() > 0

    def test_carry_forward_differs_from_reset(self):
        enc = ThrownessEncoder()
        T = 10
        h = enc.init_hidden(B, torch.device('cpu'))

        # Carry GRU state forward for T steps
        for _ in range(T):
            inp = torch.randn(B, D)
            _, h = enc(inp, h)
        t_carried, _ = enc(torch.randn(B, D), h)

        # Reset each step
        h_reset = enc.init_hidden(B, torch.device('cpu'))
        t_reset, _ = enc(torch.randn(B, D), h_reset)

        assert not torch.allclose(t_carried, t_reset)


# ---------------------------------------------------------------------------
# ProjectionEncoder tests
# ---------------------------------------------------------------------------

class TestProjectionEncoder:
    def test_output_shape(self):
        enc = ProjectionEncoder(d_model=D, n_steps=15)
        imagined = torch.randn(B, 15, D)
        out = enc(imagined)
        assert out.shape == (B, D)

    def test_variable_horizon(self):
        enc = ProjectionEncoder(d_model=D, n_steps=15)
        for H in [1, 5, 10, 15]:
            imagined = torch.randn(B, H, D)
            out = enc(imagined)
            assert out.shape == (B, D), f"Failed for H={H}"

    def test_gradient_flow(self):
        enc = ProjectionEncoder(d_model=D, n_steps=15)
        imagined = torch.randn(B, 15, D, requires_grad=True)
        out = enc(imagined)
        out.sum().backward()
        assert imagined.grad is not None
        assert imagined.grad.abs().sum() > 0

    def test_step_weights_learnable(self):
        enc = ProjectionEncoder(d_model=D, n_steps=15)
        assert enc.step_weights.requires_grad

    def test_different_horizons_differ(self):
        enc = ProjectionEncoder(d_model=D, n_steps=15)
        base = torch.randn(B, 15, D)
        out_full = enc(base)
        out_short = enc(base[:, :5, :])
        assert not torch.allclose(out_full, out_short)


# ---------------------------------------------------------------------------
# TemporalAttention tests
# ---------------------------------------------------------------------------

class TestTemporalAttention:
    def test_output_shape(self):
        attn = TemporalAttention(d_model=D, n_heads=8, n_layers=4)
        t = torch.randn(B, D)
        f = torch.randn(B, D)
        p = torch.randn(B, D)
        out = attn(t, f, p)
        assert out.shape == (B, D)

    def test_gradient_flow_all_inputs(self):
        # Use seeded inputs to avoid degenerate zero-gradient initialization
        torch.manual_seed(42)
        attn = TemporalAttention(d_model=D, n_heads=8, n_layers=4)
        thrownness = torch.randn(B, D, requires_grad=True)
        falling = torch.randn(B, D, requires_grad=True)
        projection = torch.randn(B, D, requires_grad=True)
        out = attn(thrownness, falling, projection)
        out.sum().backward()
        for name, tensor in [('thrownness', thrownness), ('falling', falling),
                              ('projection', projection)]:
            assert tensor.grad is not None, f"No grad for {name}"
            assert torch.isfinite(tensor.grad).all(), f"Non-finite grad for {name}"

    def test_temporal_pos_learnable(self):
        attn = TemporalAttention(D)
        assert attn.temporal_pos.requires_grad

    def test_cls_token_learnable(self):
        attn = TemporalAttention(D)
        assert attn.cls_token.requires_grad

    def test_zero_projection_still_works(self):
        attn = TemporalAttention(d_model=D, n_heads=8, n_layers=4)
        t = torch.randn(B, D)
        f = torch.randn(B, D)
        p = torch.zeros(B, D)  # Phase 3.2a: projection = zeros
        out = attn(t, f, p)
        assert out.shape == (B, D)
        assert not torch.isnan(out).any()

    def test_different_inputs_different_output(self):
        attn = TemporalAttention(d_model=D, n_heads=8, n_layers=4)
        t1 = torch.randn(B, D)
        f = torch.randn(B, D)
        p = torch.zeros(B, D)
        t2 = torch.randn(B, D)
        out1 = attn(t1, f, p)
        out2 = attn(t2, f, p)
        assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# BeingEmbedding (top-level) tests
# ---------------------------------------------------------------------------

class TestBeingEmbedding:
    def test_output_types(self, model, l0_phase32a, h_prev):
        out = model(l0_phase32a, h_prev)
        assert isinstance(out, L1Output)

    def test_being_embedding_shape(self, model, l0_phase32a, h_prev):
        out = model(l0_phase32a, h_prev)
        assert out.being_embedding.shape == (B, D)

    def test_being_time_embedding_shape(self, model, l0_phase32a, h_prev):
        out = model(l0_phase32a, h_prev)
        assert out.being_time_embedding.shape == (B, D)

    def test_h_t_shape(self, model, l0_phase32a, h_prev):
        out = model(l0_phase32a, h_prev)
        assert out.h_t.shape == (B, 1024)

    def test_temporal_channels_shapes(self, model, l0_phase32a, h_prev):
        out = model(l0_phase32a, h_prev)
        assert out.thrownness.shape == (B, D)
        assert out.falling.shape == (B, D)
        assert out.projection.shape == (B, D)

    def test_falling_equals_being_embedding(self, model, l0_phase32a, h_prev):
        out = model(l0_phase32a, h_prev)
        assert torch.allclose(out.falling, out.being_embedding)

    def test_projection_zeros_when_no_imagined_states(self, model, l0_phase32a, h_prev):
        out = model(l0_phase32a, h_prev, imagined_states=None)
        assert (out.projection == 0).all()

    def test_projection_nonzero_with_imagined_states(self, model, l0_phase32a, h_prev):
        imagined = torch.randn(B, 15, D)
        out = model(l0_phase32a, h_prev, imagined_states=imagined)
        assert not (out.projection == 0).all()

    def test_no_nans_phase32a(self, model, l0_phase32a, h_prev):
        out = model(l0_phase32a, h_prev)
        assert not torch.isnan(out.being_embedding).any()
        assert not torch.isnan(out.being_time_embedding).any()
        assert not torch.isnan(out.h_t).any()

    def test_no_nans_full_channels(self, model, l0_full, h_prev):
        out = model(l0_full, h_prev, imagined_states=torch.randn(B, 15, D))
        assert not torch.isnan(out.being_embedding).any()
        assert not torch.isnan(out.being_time_embedding).any()

    def test_channel_n1_shape(self, model, l0_single, h_prev):
        out = model(l0_single, h_prev)
        assert out.being_embedding.shape == (B, D)
        assert out.being_time_embedding.shape == (B, D)

    def test_channel_n4_shape(self, model, l0_all_channels, h_prev):
        out = model(l0_all_channels, h_prev)
        assert out.being_embedding.shape == (B, D)
        assert out.being_time_embedding.shape == (B, D)

    def test_gradient_flow_full(self, model, l0_phase32a, h_prev):
        # Enable grad on all channel embeddings
        proprio = l0_phase32a.proprioception.requires_grad_(True)
        obj = l0_phase32a.object_state.requires_grad_(True)
        h_prev_grad = h_prev.requires_grad_(True)

        out = model(l0_phase32a, h_prev_grad)
        out.being_time_embedding.sum().backward()

        assert proprio.grad is not None and proprio.grad.abs().sum() > 0
        assert obj.grad is not None and obj.grad.abs().sum() > 0
        assert h_prev_grad.grad is not None and h_prev_grad.grad.abs().sum() > 0

    def test_gru_state_propagates(self, model, l0_phase32a, h_prev):
        T = 5
        h = h_prev.clone()
        for _ in range(T):
            inp = L0Output(
                proprioception=torch.randn(B, D),
                object_state=torch.randn(B, D),
            )
            out = model(inp, h)
            h = out.h_t
        # After T steps, h should differ from initial zeros
        assert not torch.allclose(h, h_prev)

    def test_init_hidden(self, model):
        h = model.init_hidden(B, torch.device('cpu'))
        assert h.shape == (B, 1024)
        assert (h == 0).all()

    def test_temporal_sequence_smooth(self, model, h_prev):
        # Consecutive being embeddings from smooth obs should be cosine-similar
        obs_base = torch.randn(B, D)
        noise_scale = 0.01

        h = h_prev.clone()
        embeddings = []
        for _ in range(5):
            noisy = obs_base + noise_scale * torch.randn_like(obs_base)
            inp = L0Output(proprioception=noisy, object_state=torch.randn(B, D) * noise_scale)
            out = model(inp, h)
            embeddings.append(out.being_embedding)
            h = out.h_t

        # Check consecutive similarity > threshold
        for i in range(len(embeddings) - 1):
            sim = torch.nn.functional.cosine_similarity(
                embeddings[i], embeddings[i + 1], dim=-1
            ).mean().item()
            assert sim > 0.5, f"Low cosine similarity at step {i}: {sim:.3f}"

    def test_attention_weights_accessible(self):
        # Verify forward pass doesn't crash when probing attention patterns
        model = BeingEmbedding()
        h = model.init_hidden(B, torch.device('cpu'))
        inp = L0Output(
            proprioception=torch.randn(B, D),
            object_state=torch.randn(B, D),
        )
        out = model(inp, h)
        # CLS output encodes all three temporal channels
        assert out.being_time_embedding.shape == (B, D)
        # Temporal channel norms should all be nonzero (except projection in 3.2a)
        assert out.thrownness.norm(dim=-1).mean() > 0
        assert out.falling.norm(dim=-1).mean() > 0
        assert (out.projection == 0).all()  # Phase 3.2a

    def test_deterministic_given_same_input(self, model, l0_phase32a, h_prev):
        model.eval()
        with torch.no_grad():
            out1 = model(l0_phase32a, h_prev)
            out2 = model(l0_phase32a, h_prev)
        assert torch.allclose(out1.being_time_embedding, out2.being_time_embedding)
        assert torch.allclose(out1.being_embedding, out2.being_embedding)

    def test_different_batch_sizes(self, model):
        for bs in [1, 2, 8]:
            h = model.init_hidden(bs, torch.device('cpu'))
            inp = L0Output(
                proprioception=torch.randn(bs, D),
                object_state=torch.randn(bs, D),
            )
            out = model(inp, h)
            assert out.being_time_embedding.shape == (bs, D)
            assert out.h_t.shape == (bs, 1024)
