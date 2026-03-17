"""Tests for L3 emotion module — Phase 3.2a subset.

Covers: shapes, forward pass, gradients, emotion dynamics,
intrinsic reward sanity, mood slow update.
"""

import pytest
import torch
from collections import deque

from slm_lab.agent.net.emotion import (
    PHASE_EMOTIONS,
    EmotionModule,
    EmotionTag,
    FrustrationAccumulator,
    IntrinsicMotivation,
    InteroceptionModule,
    LearningProgressReward,
    MaximumGripReward,
    MoodVector,
    NoveltyReward,
    get_active_emotions,
)

B = 4   # batch size
D = 32  # latent dim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pe_history(vals: list[float]) -> deque:
    d = deque(maxlen=100)
    d.extend(vals)
    return d


def make_action_history(norms: list[float]) -> deque:
    d = deque(maxlen=100)
    d.extend(norms)
    return d


# ---------------------------------------------------------------------------
# Phase activation
# ---------------------------------------------------------------------------

def test_phase_322a_active_set():
    active = get_active_emotions("3.2a")
    assert active == {"fear", "surprise", "satisfaction"}


def test_phase_322c_active_set():
    active = get_active_emotions("3.2c")
    assert "curiosity" in active
    assert "frustration" in active


def test_unknown_phase_returns_all():
    active = get_active_emotions("unknown")
    assert "social_approval" in active


# ---------------------------------------------------------------------------
# InteroceptionModule — shapes
# ---------------------------------------------------------------------------

@pytest.fixture
def intero():
    return InteroceptionModule()


def test_intero_output_shape(intero):
    energy = torch.rand(B)
    pe_hist = make_pe_history([0.2] * 50)
    action_hist = make_action_history([0.5] * 50)
    teacher = torch.zeros(B, 2)
    out = intero(energy, pe_hist, 0.5, 0.6, teacher, action_hist)
    assert out.shape == (B, 5)


def test_intero_energy_channel_passthrough(intero):
    energy = torch.tensor([0.3, 0.7, 1.0, 0.0])
    pe_hist = make_pe_history([0.1] * 10)
    action_hist = make_action_history([])
    teacher = torch.zeros(B, 2)
    out = intero(energy, pe_hist, 0.5, 0.5, teacher, action_hist)
    assert torch.allclose(out[:, 0], energy)


def test_intero_social_channel(intero):
    energy = torch.ones(B)
    pe_hist = make_pe_history([])
    action_hist = make_action_history([])
    teacher = torch.zeros(B, 2)
    teacher[0] = torch.tensor([0.8, 0.9])  # valence=0.8, magnitude=0.9 → 0.72
    out = intero(energy, pe_hist, 0.5, 0.5, teacher, action_hist)
    assert abs(out[0, 3].item() - 0.72) < 1e-4


def test_intero_social_clamped(intero):
    energy = torch.ones(B)
    pe_hist = make_pe_history([])
    action_hist = make_action_history([])
    teacher = torch.full((B, 2), 10.0)
    out = intero(energy, pe_hist, 0.5, 0.5, teacher, action_hist)
    assert out[:, 3].max().item() <= 1.0 + 1e-5


def test_intero_pe_trend_bounds(intero):
    energy = torch.ones(B)
    pe_hist = make_pe_history([2.0] * 100)  # high PE values
    action_hist = make_action_history([])
    teacher = torch.zeros(B, 2)
    out = intero(energy, pe_hist, 0.5, 0.5, teacher, action_hist)
    assert 0.0 <= out[0, 1].item() <= 1.0


def test_intero_lp_range(intero):
    energy = torch.ones(B)
    pe_hist = make_pe_history([])
    action_hist = make_action_history([])
    teacher = torch.zeros(B, 2)
    # learning_progress = accuracy_curr - accuracy_prev = 0.9 - 0.5 = 0.4
    out = intero(energy, pe_hist, 0.5, 0.9, teacher, action_hist)
    assert -1.0 <= out[0, 2].item() <= 1.0
    assert abs(out[0, 2].item() - 0.4) < 1e-4


def test_intero_motor_fatigue_bounds(intero):
    energy = torch.ones(B)
    pe_hist = make_pe_history([])
    action_hist = make_action_history([5.0] * 100)  # large norms
    teacher = torch.zeros(B, 2)
    out = intero(energy, pe_hist, 0.5, 0.5, teacher, action_hist)
    assert 0.0 <= out[0, 4].item() <= 1.0


# ---------------------------------------------------------------------------
# MoodVector — shapes and EMA dynamics
# ---------------------------------------------------------------------------

@pytest.fixture
def mood_net():
    return MoodVector()


def test_mood_output_shapes(mood_net):
    intero = torch.randn(B, 5)
    ema = torch.zeros(B, 16)
    mv, new_ema = mood_net(intero, ema)
    assert mv.shape == (B, 16)
    assert new_ema.shape == (B, 16)


def test_mood_ema_is_slow(mood_net):
    """EMA with 0.99 momentum should barely move from zero after one step."""
    intero = torch.ones(B, 5)
    ema = torch.zeros(B, 16)
    mv, new_ema = mood_net(intero, ema)
    # Raw output may be significant, but EMA is 0.99*0 + 0.01*raw
    assert new_ema.abs().max().item() < 1.0  # stays small after one step


def test_mood_ema_accumulates_over_steps(mood_net):
    """Mood should grow across repeated identical inputs."""
    intero = torch.ones(B, 5) * 2.0
    ema = torch.zeros(B, 16)
    for _ in range(200):
        _, ema = mood_net(intero, ema)
    # After many steps mood should be non-trivially different from zero
    assert ema.abs().max().item() > 0.01


def test_mood_init(mood_net):
    mv, ema = mood_net.init_mood(B, torch.device("cpu"))
    assert mv.shape == (B, 16)
    assert mv.sum().item() == 0.0


def test_mood_gradients(mood_net):
    intero = torch.randn(B, 5, requires_grad=True)
    ema = torch.zeros(B, 16)
    mv, _ = mood_net(intero, ema)
    loss = mv.sum()
    loss.backward()
    assert intero.grad is not None
    assert intero.grad.abs().sum().item() > 0


def test_mood_exploration_temperature_range(mood_net):
    mv = torch.randn(B, 16) * 2.0
    temp = mood_net.exploration_temperature(mv)
    assert temp.shape == (B,)
    assert temp.min().item() >= 0.5 - 1e-5
    assert temp.max().item() <= 2.0 + 1e-5


# ---------------------------------------------------------------------------
# EmotionModule — trigger conditions
# ---------------------------------------------------------------------------

@pytest.fixture
def emo():
    return EmotionModule(phase="3.2a")


def test_fear_triggered(emo):
    tag = emo.compute(pe=0.3, reward=-0.8)
    assert tag.emotion_type == "fear"
    assert 0.0 < tag.magnitude <= 1.0


def test_surprise_triggered(emo):
    tag = emo.compute(pe=0.7, reward=0.0)
    assert tag.emotion_type == "surprise"


def test_satisfaction_triggered(emo):
    tag = emo.compute(pe=0.05, reward=0.8)
    assert tag.emotion_type == "satisfaction"


def test_neutral_when_no_trigger(emo):
    tag = emo.compute(pe=0.05, reward=0.1)
    assert tag.emotion_type == "neutral"
    assert tag.magnitude == 0.0


def test_fear_priority_over_surprise(emo):
    """fear (reward<-0.5 and pe>0.1) fires before surprise (pe>0.5)."""
    tag = emo.compute(pe=0.8, reward=-0.9)
    assert tag.emotion_type == "fear"


def test_frustration_not_active_in_322a(emo):
    """frustration not in 3.2a active set — should fall through."""
    tag = emo.compute(pe=0.05, reward=-0.8, failure_count=10)
    # reward<-0.5, pe<0.1, failures>=3 → would be frustration in 3.2c
    # In 3.2a → neutral (no other triggers fire)
    assert tag.emotion_type == "neutral"


def test_frustration_active_in_322c():
    emo_c = EmotionModule(phase="3.2c")
    tag = emo_c.compute(pe=0.05, reward=-0.8, failure_count=10)
    assert tag.emotion_type == "frustration"


def test_fear_magnitude_capped(emo):
    tag = emo.compute(pe=10.0, reward=-100.0)
    assert tag.magnitude <= 1.0


def test_surprise_magnitude_capped(emo):
    tag = emo.compute(pe=100.0, reward=0.0)
    assert tag.magnitude <= 1.0


def test_satisfaction_magnitude_capped(emo):
    tag = emo.compute(pe=0.0, reward=100.0)
    assert tag.magnitude <= 1.0


# ---------------------------------------------------------------------------
# EmotionModule — modulation outputs
# ---------------------------------------------------------------------------

def test_lr_modulation_fear(emo):
    tag = EmotionTag("fear", 1.0)
    factor = emo.lr_modulation(tag)
    assert abs(factor - 1.5) < 1e-5


def test_lr_modulation_surprise(emo):
    tag = EmotionTag("surprise", 0.5)
    factor = emo.lr_modulation(tag)
    assert abs(factor - 1.25) < 1e-5


def test_lr_modulation_satisfaction(emo):
    tag = EmotionTag("satisfaction", 1.0)
    factor = emo.lr_modulation(tag)
    assert abs(factor - 0.7) < 1e-5


def test_lr_modulation_neutral(emo):
    tag = EmotionTag("neutral", 0.0)
    factor = emo.lr_modulation(tag)
    assert factor == 1.0


def test_per_priority_positive(emo):
    tag = EmotionTag("fear", 0.8)
    p = emo.per_priority(tag)
    assert p > 0.0


def test_per_priority_neutral_zero(emo):
    tag = EmotionTag("neutral", 0.0)
    p = emo.per_priority(tag)
    assert p == 0.0


def test_encode_emotion_vector_shape(emo):
    tag = EmotionTag("fear", 0.8)
    vec = emo.encode_emotion_vector(tag)
    assert vec.shape == (7,)
    assert vec[0].item() == 1.0  # fear is index 0
    assert abs(vec[6].item() - 0.8) < 1e-5


def test_encode_neutral_vector(emo):
    tag = EmotionTag("neutral", 0.0)
    vec = emo.encode_emotion_vector(tag)
    assert vec[:6].sum().item() == 0.0
    assert vec[6].item() == 0.0


# ---------------------------------------------------------------------------
# FrustrationAccumulator
# ---------------------------------------------------------------------------

def test_frustration_accumulates():
    acc = FrustrationAccumulator(threshold=5.0)
    for _ in range(6):
        acc.update(EmotionTag("frustration", 1.0), reward=-1.0)
    assert acc.cumulative >= 6.0
    assert acc.should_switch


def test_frustration_decays_on_reward():
    acc = FrustrationAccumulator(threshold=5.0, decay=0.95)
    for _ in range(3):
        acc.update(EmotionTag("frustration", 1.0), reward=-1.0)
    before = acc.cumulative
    acc.update(EmotionTag("neutral", 0.0), reward=1.0)
    assert acc.cumulative < before


def test_frustration_reset():
    acc = FrustrationAccumulator()
    acc.update(EmotionTag("frustration", 0.9), reward=-1.0)
    acc.reset()
    assert acc.cumulative == 0.0


def test_frustration_no_switch_below_threshold():
    acc = FrustrationAccumulator(threshold=5.0)
    acc.update(EmotionTag("frustration", 0.5), reward=-1.0)
    assert not acc.should_switch


# ---------------------------------------------------------------------------
# NoveltyReward
# ---------------------------------------------------------------------------

def test_novelty_shape():
    nr = NoveltyReward()
    z_pred = torch.randn(B, D)
    z_actual = torch.randn(B, D)
    out = nr.compute(z_pred, z_actual)
    assert out.shape == (B,)


def test_novelty_zero_on_identical():
    nr = NoveltyReward()
    z = torch.randn(B, D)
    out = nr.compute(z, z)
    assert out.abs().max().item() < 1e-6


def test_novelty_positive():
    nr = NoveltyReward()
    z_pred = torch.zeros(B, D)
    z_actual = torch.ones(B, D)
    out = nr.compute(z_pred, z_actual)
    assert (out > 0).all()


# ---------------------------------------------------------------------------
# LearningProgressReward
# ---------------------------------------------------------------------------

def test_lp_zero_initially():
    lp = LearningProgressReward(window=10)
    assert lp.update(0.5) == 0.0


def test_lp_positive_when_improving():
    lp = LearningProgressReward(window=5)
    # Steps 1-5: high PE. At step 5 the swap fires: prev←[0.8]*5, curr reset.
    for _ in range(5):
        lp.update(0.8)
    # Step 6: low PE. prev=[0.8]*5, curr=[0.2] → acc_prev=0.2, acc_curr=0.8 → lp=0.6
    val = lp.update(0.2)
    assert val > 0.0


def test_lp_floor_at_zero():
    lp = LearningProgressReward(window=5)
    for _ in range(5):
        lp.update(0.1)
    # Second window worse (deterioration)
    val = 0.0
    for _ in range(5):
        val = lp.update(0.9)
    # accuracy got worse → clamp to 0
    assert val == 0.0


def test_lp_reset():
    lp = LearningProgressReward(window=5)
    for _ in range(10):
        lp.update(0.5)
    lp.reset()
    assert lp.step_count == 0
    assert len(lp.pe_buffer_curr) == 0


# ---------------------------------------------------------------------------
# MaximumGripReward
# ---------------------------------------------------------------------------

def test_grip_zero_when_pe_below_threshold():
    mg = MaximumGripReward(novelty_threshold=0.15)
    # PE is low — not in novel region
    mg.pe_ema = 0.05
    reward = mg.compute(0.04)
    assert reward == 0.0


def test_grip_positive_on_pe_drop_in_novel_region():
    mg = MaximumGripReward(novelty_threshold=0.15)
    mg.pe_ema = 0.5  # recently high PE (novel region)
    reward = mg.compute(0.1)  # PE dropped
    assert reward > 0.0


def test_grip_zero_on_pe_increase():
    mg = MaximumGripReward(novelty_threshold=0.15)
    mg.pe_ema = 0.5
    reward = mg.compute(0.8)  # PE went up
    assert reward == 0.0


def test_grip_reset():
    mg = MaximumGripReward()
    mg.pe_ema = 0.9
    mg.reset()
    assert mg.pe_ema == 0.5


# ---------------------------------------------------------------------------
# IntrinsicMotivation — combined
# ---------------------------------------------------------------------------

@pytest.fixture
def intrinsic():
    return IntrinsicMotivation(phase="3.2a")


def test_intrinsic_output_shape_with_latents(intrinsic):
    z_pred = torch.randn(B, D)
    z_actual = torch.randn(B, D)
    r_int, lp = intrinsic.compute(pe=0.3, z_predicted=z_pred, z_actual=z_actual)
    assert r_int.shape == (B,)


def test_intrinsic_output_scalar_without_latents(intrinsic):
    r_int, lp = intrinsic.compute(pe=0.3)
    assert r_int.shape == (1,)


def test_intrinsic_322a_novelty_only():
    """In 3.2a, only novelty component active — LP and grip weights not applied."""
    m = IntrinsicMotivation(phase="3.2a")
    z_pred = torch.zeros(B, D)
    z_actual = torch.ones(B, D)
    r_int, _ = m.compute(pe=0.3, z_predicted=z_pred, z_actual=z_actual)
    # Should be non-zero (novelty active)
    assert r_int.sum().item() > 0.0


def test_intrinsic_322c_includes_all_components():
    m = IntrinsicMotivation(phase="3.2c")
    # Fill LP window first
    for _ in range(m.lp_reward.window):
        m.lp_reward.update(0.5)
    m.grip_reward.pe_ema = 0.5
    z_pred = torch.zeros(B, D)
    z_actual = torch.ones(B, D)
    r_int, lp = m.compute(pe=0.1, z_predicted=z_pred, z_actual=z_actual)
    assert r_int.shape == (B,)


def test_intrinsic_lambda_annealing():
    """Lambda should decrease from start to end over training."""
    lam_start = IntrinsicMotivation._lambda(0, 1000)
    lam_end = IntrinsicMotivation._lambda(1000, 1000)
    assert abs(lam_start - 1.0) < 1e-5
    assert abs(lam_end - 0.1) < 1e-5


def test_intrinsic_lambda_monotone_decreasing():
    lams = [IntrinsicMotivation._lambda(s, 100) for s in range(0, 101, 10)]
    for i in range(len(lams) - 1):
        assert lams[i] >= lams[i + 1] - 1e-6


def test_intrinsic_reset(intrinsic):
    for _ in range(50):
        intrinsic.lp_reward.update(0.4)
    intrinsic.grip_reward.pe_ema = 0.9
    intrinsic.reset()
    assert intrinsic.lp_reward.step_count == 0
    assert intrinsic.grip_reward.pe_ema == 0.5


def test_intrinsic_reward_non_negative_with_latents(intrinsic):
    z_pred = torch.randn(B, D)
    z_actual = torch.randn(B, D)
    r_int, _ = intrinsic.compute(pe=0.3, z_predicted=z_pred, z_actual=z_actual)
    assert (r_int >= 0).all()


# ---------------------------------------------------------------------------
# Mood slow-update integration
# ---------------------------------------------------------------------------

def test_mood_slow_update_pipeline():
    """Full slow-update pipeline: interoception → mood → EMA."""
    intero_mod = InteroceptionModule()
    mood_mod = MoodVector()

    energy = torch.rand(B)
    pe_hist = make_pe_history([0.3] * 50)
    action_hist = make_action_history([0.5] * 50)
    teacher = torch.zeros(B, 2)

    mv, ema = mood_mod.init_mood(B, torch.device("cpu"))

    for _ in range(10):
        intero = intero_mod(energy, pe_hist, 0.5, 0.55, teacher, action_hist)
        mv, ema = mood_mod(intero, ema)

    assert mv.shape == (B, 16)
    # Mood should have drifted from zero after 10 slow updates
    assert mv.abs().max().item() > 1e-4


def test_mood_different_for_different_energy():
    """Two agents with different energy levels should get different mood vectors."""
    intero_mod = InteroceptionModule()
    mood_mod = MoodVector()

    ema = torch.zeros(B, 16)
    pe_hist = make_pe_history([0.2] * 50)
    action_hist = make_action_history([0.3] * 30)
    teacher = torch.zeros(B, 2)

    # High energy batch
    energy_high = torch.ones(B)
    intero_h = intero_mod(energy_high, pe_hist, 0.5, 0.5, teacher, action_hist)
    mv_h, _ = mood_mod(intero_h, ema)

    # Low energy batch
    energy_low = torch.zeros(B)
    intero_l = intero_mod(energy_low, pe_hist, 0.5, 0.5, teacher, action_hist)
    mv_l, _ = mood_mod(intero_l, ema)

    assert not torch.allclose(mv_h, mv_l)
