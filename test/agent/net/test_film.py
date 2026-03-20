"""Tests for FiLM conditioning layers — Phase 3.2b.

Covers: identity init, forward shapes, gradient flow, mood/emotion
differentiation, somatic marker retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from slm_lab.agent.net.emotion import EmotionTag
from slm_lab.agent.net.film import (
    EmotionFiLMLayer,
    FiLMLayer,
    MoodFiLMLayer,
    SomaticMarkerSystem,
)

B = 4
N_TOKENS = 64
D_DINO = 1024
D_POLICY = 512
D_MOOD = 16
D_EMOTION = 7
D_COND = 8


# ---------------------------------------------------------------------------
# FiLMLayer — identity init and basic forward
# ---------------------------------------------------------------------------

@pytest.fixture
def film():
    return FiLMLayer(feature_dim=D_POLICY, cond_dim=D_COND)


def test_identity_init_output_equals_input(film):
    """At construction γ=1, β=0 → output must equal input exactly."""
    x = torch.randn(B, D_POLICY)
    cond = torch.randn(B, D_COND)
    out = film(x, cond)
    assert torch.allclose(out, x, atol=1e-6), "Identity init violated: output != input"


def test_identity_init_gamma_weights_zero(film):
    assert film.gamma.weight.abs().max().item() == 0.0
    assert film.gamma.bias.abs().max().item() == 0.0


def test_identity_init_beta_weights_zero(film):
    assert film.beta.weight.abs().max().item() == 0.0
    assert film.beta.bias.abs().max().item() == 0.0


def test_output_shape_2d(film):
    x = torch.randn(B, D_POLICY)
    cond = torch.randn(B, D_COND)
    out = film(x, cond)
    assert out.shape == (B, D_POLICY)


def test_output_shape_3d_sequence():
    """FiLM should broadcast over (B, N_tokens, D) patch sequences."""
    layer = FiLMLayer(feature_dim=D_DINO, cond_dim=D_MOOD)
    x = torch.randn(B, N_TOKENS, D_DINO)
    cond = torch.randn(B, D_MOOD)
    out = layer(x, cond)
    assert out.shape == (B, N_TOKENS, D_DINO)


def test_gradient_flows_through_film(film):
    x = torch.randn(B, D_POLICY, requires_grad=True)
    cond = torch.randn(B, D_COND, requires_grad=True)
    out = film(x, cond)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum().item() > 0
    assert cond.grad is not None


def test_gradient_flows_through_film_params(film):
    """After training, gamma/beta weights receive gradients."""
    # First do one forward with identity init; then perturb weights to allow grad
    x = torch.randn(B, D_POLICY)
    cond = torch.randn(B, D_COND)
    # Manually set non-zero weights so gradient is non-trivially testable
    with torch.no_grad():
        film.gamma.weight.fill_(0.01)
        film.beta.weight.fill_(0.01)
    out = film(x, cond)
    out.sum().backward()
    assert film.gamma.weight.grad is not None
    assert film.beta.weight.grad is not None


def test_different_conds_produce_different_outputs():
    """After learning (non-zero weights), different conds must differ."""
    layer = FiLMLayer(feature_dim=D_POLICY, cond_dim=D_COND)
    with torch.no_grad():
        layer.gamma.weight.fill_(0.1)
        layer.beta.weight.fill_(0.1)
    x = torch.randn(B, D_POLICY)
    cond_a = torch.zeros(B, D_COND)
    cond_b = torch.ones(B, D_COND)
    out_a = layer(x, cond_a)
    out_b = layer(x, cond_b)
    assert not torch.allclose(out_a, out_b)


# ---------------------------------------------------------------------------
# MoodFiLMLayer
# ---------------------------------------------------------------------------

@pytest.fixture
def mood_film():
    return MoodFiLMLayer()


def test_mood_film_identity_init_block8(mood_film):
    h = torch.randn(B, N_TOKENS, D_DINO)
    mood = torch.randn(B, D_MOOD)
    out = mood_film(h, mood, block=8)
    assert torch.allclose(out, h, atol=1e-6)


def test_mood_film_identity_init_block16(mood_film):
    h = torch.randn(B, N_TOKENS, D_DINO)
    mood = torch.randn(B, D_MOOD)
    out = mood_film(h, mood, block=16)
    assert torch.allclose(out, h, atol=1e-6)


def test_mood_film_identity_init_block24(mood_film):
    h = torch.randn(B, N_TOKENS, D_DINO)
    mood = torch.randn(B, D_MOOD)
    out = mood_film(h, mood, block=24)
    assert torch.allclose(out, h, atol=1e-6)


def test_mood_film_output_shape(mood_film):
    h = torch.randn(B, N_TOKENS, D_DINO)
    mood = torch.randn(B, D_MOOD)
    for block in (8, 16, 24):
        out = mood_film(h, mood, block=block)
        assert out.shape == (B, N_TOKENS, D_DINO)


def test_mood_film_invalid_block(mood_film):
    h = torch.randn(B, N_TOKENS, D_DINO)
    mood = torch.randn(B, D_MOOD)
    with pytest.raises(ValueError):
        mood_film(h, mood, block=12)


def test_mood_film_three_independent_layers(mood_film):
    """Each block has its own FiLM — perturbing one should not affect others."""
    # Perturb block-8 weights only
    with torch.no_grad():
        mood_film.film_block8.gamma.weight.fill_(0.5)

    h = torch.randn(B, N_TOKENS, D_DINO)
    mood = torch.randn(B, D_MOOD)

    out8 = mood_film(h, mood, block=8)
    out16 = mood_film(h, mood, block=16)

    # block16 still identity → should equal h; block8 should differ
    assert torch.allclose(out16, h, atol=1e-6)
    assert not torch.allclose(out8, h, atol=1e-6)


def test_mood_film_different_moods_different_outputs():
    """After learning, distinct mood vectors must produce distinct outputs."""
    layer = MoodFiLMLayer()
    with torch.no_grad():
        layer.film_block8.gamma.weight.fill_(0.1)

    h = torch.randn(B, N_TOKENS, D_DINO)
    mood_a = torch.zeros(B, D_MOOD)
    mood_b = torch.ones(B, D_MOOD)

    out_a = layer(h, mood_a, block=8)
    out_b = layer(h, mood_b, block=8)
    assert not torch.allclose(out_a, out_b)


def test_mood_film_gradient_flow(mood_film):
    h = torch.randn(B, N_TOKENS, D_DINO, requires_grad=True)
    mood = torch.randn(B, D_MOOD, requires_grad=True)
    out = mood_film(h, mood, block=16)
    out.sum().backward()
    assert h.grad is not None and h.grad.abs().sum().item() > 0
    assert mood.grad is not None


# ---------------------------------------------------------------------------
# EmotionFiLMLayer
# ---------------------------------------------------------------------------

@pytest.fixture
def emotion_film():
    return EmotionFiLMLayer()


def test_emotion_film_identity_init(emotion_film):
    h = torch.randn(B, D_POLICY)
    emo_vec = torch.randn(B, D_EMOTION)
    out = emotion_film(h, emo_vec)
    assert torch.allclose(out, h, atol=1e-6)


def test_emotion_film_output_shape(emotion_film):
    h = torch.randn(B, D_POLICY)
    emo_vec = torch.randn(B, D_EMOTION)
    out = emotion_film(h, emo_vec)
    assert out.shape == (B, D_POLICY)


def test_emotion_film_1d_vec_broadcasts(emotion_film):
    """A (7,) emotion vector should broadcast over the batch."""
    h = torch.randn(B, D_POLICY)
    emo_vec = torch.randn(D_EMOTION)
    out = emotion_film(h, emo_vec)
    assert out.shape == (B, D_POLICY)


def test_emotion_film_gradient_flow(emotion_film):
    h = torch.randn(B, D_POLICY, requires_grad=True)
    emo_vec = torch.randn(B, D_EMOTION, requires_grad=True)
    out = emotion_film(h, emo_vec)
    out.sum().backward()
    assert h.grad is not None and h.grad.abs().sum().item() > 0
    assert emo_vec.grad is not None


def test_emotion_film_different_emotions_different_outputs():
    torch.manual_seed(42)
    layer = EmotionFiLMLayer()
    with torch.no_grad():
        # Use random non-uniform weights so different one-hot positions map differently
        layer.film.gamma.weight.copy_(torch.randn_like(layer.film.gamma.weight))
        layer.film.beta.weight.copy_(torch.randn_like(layer.film.beta.weight))

    h = torch.randn(B, D_POLICY)
    fear_vec = EmotionFiLMLayer.encode(EmotionTag("fear", 0.9)).unsqueeze(0).expand(B, -1)
    satis_vec = EmotionFiLMLayer.encode(EmotionTag("satisfaction", 0.9)).unsqueeze(0).expand(B, -1)

    out_fear = layer(h, fear_vec)
    out_satis = layer(h, satis_vec)
    assert not torch.allclose(out_fear, out_satis)


def test_emotion_encode_shape():
    vec = EmotionFiLMLayer.encode(EmotionTag("fear", 0.8))
    assert vec.shape == (7,)


def test_emotion_encode_one_hot_fear():
    vec = EmotionFiLMLayer.encode(EmotionTag("fear", 0.8))
    assert vec[0].item() == 1.0           # fear is index 0
    assert abs(vec[6].item() - 0.8) < 1e-5
    assert vec[1:6].sum().item() == 0.0


def test_emotion_encode_neutral_all_zero():
    vec = EmotionFiLMLayer.encode(EmotionTag("neutral", 0.0))
    assert vec.sum().item() == 0.0


# ---------------------------------------------------------------------------
# SomaticMarkerSystem
# ---------------------------------------------------------------------------

@dataclass
class FakeTransition:
    state: torch.Tensor
    emotion_type: str
    emotion_magnitude: float


def make_buffer(transitions: list[FakeTransition]) -> MagicMock:
    buf = MagicMock()
    buf.sample_recent.return_value = transitions
    return buf


def test_somatic_empty_buffer():
    buf = make_buffer([])
    sms = SomaticMarkerSystem(buf)
    bias = sms.query(torch.randn(512))
    assert bias == 0.0


def test_somatic_no_similar_transitions():
    """All similarities below threshold → bias = 0."""
    # Orthogonal vectors have cosine similarity 0 < threshold 0.7
    state = torch.zeros(512)
    state[0] = 1.0
    transitions = [FakeTransition(state=state, emotion_type="fear", emotion_magnitude=0.9)]
    current_be = torch.zeros(512)
    current_be[1] = 1.0  # orthogonal to state
    buf = make_buffer(transitions)
    sms = SomaticMarkerSystem(buf, similarity_threshold=0.7)
    bias = sms.query(current_be)
    assert bias == 0.0


def test_somatic_identical_fear_gives_negative_bias():
    """Identical state + fear emotion → negative somatic bias."""
    state = torch.randn(512)
    state = F.normalize(state, dim=0)
    transitions = [
        FakeTransition(state=state.clone(), emotion_type="fear", emotion_magnitude=1.0)
    ]
    buf = make_buffer(transitions)
    sms = SomaticMarkerSystem(buf, similarity_threshold=0.5)
    bias = sms.query(state.clone())
    assert bias < 0.0


def test_somatic_identical_satisfaction_gives_positive_bias():
    """Identical state + satisfaction → positive somatic bias."""
    state = torch.randn(512)
    state = F.normalize(state, dim=0)
    transitions = [
        FakeTransition(state=state.clone(), emotion_type="satisfaction", emotion_magnitude=1.0)
    ]
    buf = make_buffer(transitions)
    sms = SomaticMarkerSystem(buf, similarity_threshold=0.5)
    bias = sms.query(state.clone())
    assert bias > 0.0


def test_somatic_bias_in_range():
    """Somatic bias must stay in [-1, 1]."""
    state = torch.randn(512)
    state = F.normalize(state, dim=0)
    transitions = [
        FakeTransition(state=state.clone(), emotion_type=etype, emotion_magnitude=1.0)
        for etype in ("fear", "satisfaction", "curiosity", "surprise")
    ]
    buf = make_buffer(transitions)
    sms = SomaticMarkerSystem(buf, similarity_threshold=0.5)
    bias = sms.query(state.clone())
    assert -1.0 <= bias <= 1.0


def test_somatic_top_k_respected():
    """Only top-k=2 transitions should be used."""
    base = torch.randn(512)
    base = F.normalize(base, dim=0)

    # 5 identical transitions — all should be above threshold
    # top-k=2 means only 2 are used
    transitions = [
        FakeTransition(state=base.clone(), emotion_type="fear", emotion_magnitude=1.0)
        for _ in range(5)
    ]
    buf_k2 = make_buffer(transitions)
    buf_k5 = make_buffer(transitions)
    sms_k2 = SomaticMarkerSystem(buf_k2, top_k=2, similarity_threshold=0.5)
    sms_k5 = SomaticMarkerSystem(buf_k5, top_k=5, similarity_threshold=0.5)

    # Both should give same bias since all transitions are identical
    bias_k2 = sms_k2.query(base.clone())
    bias_k5 = sms_k5.query(base.clone())
    assert abs(bias_k2 - bias_k5) < 1e-5


def test_somatic_2d_being_embedding_handled():
    """(1, 512) shaped being embedding should work (squeeze applied)."""
    state = torch.randn(512)
    state = F.normalize(state, dim=0)
    transitions = [
        FakeTransition(state=state.clone(), emotion_type="satisfaction", emotion_magnitude=0.8)
    ]
    buf = make_buffer(transitions)
    sms = SomaticMarkerSystem(buf, similarity_threshold=0.5)
    be_2d = state.clone().unsqueeze(0)  # (1, 512)
    bias = sms.query(be_2d)
    assert isinstance(bias, float)


# F imported at top of file
