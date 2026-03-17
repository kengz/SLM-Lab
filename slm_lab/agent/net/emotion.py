"""L3 Mood, Emotion, and Intrinsic Motivation — Phase 3.2a subset.

Implements the emotional/motivational layer of the SLM agent:
    - InteroceptionModule: raw signals → 5-dim interoceptive vector
    - MoodVector: 16-dim slow EMA mood, influences exploration temperature
    - EmotionModule: 3 basic emotions (fear, surprise, satisfaction) with trigger/intensity/decay
    - IntrinsicMotivation: novelty (η=0.1), learning progress (η=0.2), maximum grip (η=0.1)

Phase 3.2a active set: {fear, surprise, satisfaction} + novelty only.
Frustration, curiosity, social_approval activated in Phase 3.2c+.

Source: notes/layers/L3-mood-emotion.md
Traceability: Ax4 → Th13 → DR18 → IS14 | Ax5 → Th14 → DR17 → IS13 | Ax15 → DR19 → IS15
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Phase activation
# ---------------------------------------------------------------------------

PHASE_EMOTIONS: dict[str, set[str]] = {
    "3.2a": {"fear", "surprise", "satisfaction"},
    "3.2b": {"fear", "surprise", "satisfaction"},
    "3.2c": {"fear", "surprise", "satisfaction", "frustration", "curiosity"},
    "3.2d": {"fear", "surprise", "satisfaction", "frustration", "curiosity"},
    "3.3":  {"fear", "surprise", "satisfaction", "frustration", "curiosity", "social_approval"},
    "3.4":  {"fear", "surprise", "satisfaction", "frustration", "curiosity", "social_approval"},
}

EMOTION_TYPES = ("fear", "surprise", "satisfaction", "frustration", "curiosity", "social_approval")


def get_active_emotions(phase: str) -> set[str]:
    """Return the set of active emotion types for a training phase."""
    return PHASE_EMOTIONS.get(phase, set(EMOTION_TYPES))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EmotionTag:
    """Tagged emotion for a single timestep."""
    emotion_type: str   # one of EMOTION_TYPES or "neutral"
    magnitude: float    # [0, 1]


@dataclass
class L3Output:
    """Per-step outputs from L3 consumed by L2 and above."""
    emotion_tag: EmotionTag
    intrinsic_reward: torch.Tensor   # (B,) scalar per batch element
    lr_modulation: float             # scalar: 1.0 baseline, >1 boost, <1 dampen
    frustration_delta: float         # cumulative frustration increment (0 if no frustration)
    mood_vector: torch.Tensor        # (B, 16) current mood (updated every 10 steps)


# ---------------------------------------------------------------------------
# InteroceptionModule
# ---------------------------------------------------------------------------

class InteroceptionModule(nn.Module):
    """Compute 5-dim interoceptive signal from raw RL inputs.

    Signals:
        [0] energy        — environment survival metric          [0, 1]
        [1] pe_trend      — EMA of prediction error history      [0, 1]
        [2] learning_prog — Δ(prediction accuracy) over window   [-1, 1]
        [3] social        — teacher valence × magnitude          [-1, 1]
        [4] motor_fatigue — EMA of ‖action‖₂ history            [0, 1]

    Updates at 2.5 Hz (every 10 control steps). No learned parameters.
    Traceability: Ax5 → Th14 → DR17 → IS13
    """

    def __init__(self, ema_momentum: float = 0.95):
        super().__init__()
        self.ema_momentum = ema_momentum

    def _ema_over_deque(self, history: deque, init: float = 0.0) -> float:
        """Compute EMA over a deque (oldest → newest)."""
        if len(history) == 0:
            return init
        ema = float(list(history)[0])
        for val in list(history)[1:]:
            ema = self.ema_momentum * ema + (1.0 - self.ema_momentum) * float(val)
        return ema

    def compute_pe_trend(self, pe_history: deque) -> float:
        """EMA of PE over history window. Returns [0, 1]."""
        return min(1.0, max(0.0, self._ema_over_deque(pe_history, init=0.5)))

    def compute_learning_progress(self, accuracy_prev: float, accuracy_curr: float) -> float:
        """Δ(accuracy) clamped to [-1, 1]."""
        return max(-1.0, min(1.0, accuracy_curr - accuracy_prev))

    def compute_motor_fatigue(self, action_history: deque) -> float:
        """EMA of action norms, clamped to [0, 1]."""
        return min(1.0, max(0.0, self._ema_over_deque(action_history, init=0.0)))

    def forward(
        self,
        energy: torch.Tensor,           # (B,) or scalar
        pe_history: deque,
        accuracy_prev: float,
        accuracy_curr: float,
        teacher_emotion: torch.Tensor,  # (B, 2) — [valence, magnitude]
        action_history: deque,
    ) -> torch.Tensor:
        """Compute (B, 5) interoceptive vector.

        Args:
            energy: energy level per batch element, [0, 1]
            pe_history: deque of past PE scalars
            accuracy_prev: accuracy in previous 100-step window
            accuracy_curr: accuracy in current 100-step window
            teacher_emotion: (B, 2) teacher [valence, magnitude] or zeros
            action_history: deque of past action norms

        Returns:
            (B, 5) interoceptive signals
        """
        energy = energy.float()
        B = energy.shape[0]

        pe_trend = self.compute_pe_trend(pe_history)
        lp = self.compute_learning_progress(accuracy_prev, accuracy_curr)
        fatigue = self.compute_motor_fatigue(action_history)

        social = (teacher_emotion[:, 0] * teacher_emotion[:, 1]).clamp(-1.0, 1.0)  # (B,)

        pe_col = torch.full((B,), pe_trend, dtype=torch.float32, device=energy.device)
        lp_col = torch.full((B,), lp, dtype=torch.float32, device=energy.device)
        fat_col = torch.full((B,), fatigue, dtype=torch.float32, device=energy.device)

        return torch.stack([energy, pe_col, lp_col, social, fat_col], dim=-1)  # (B, 5)


# ---------------------------------------------------------------------------
# MoodVector
# ---------------------------------------------------------------------------

class MoodVector(nn.Module):
    """16-dim mood vector: interoceptive (5) → MLP (5→32→16) → EMA (0.99).

    Slow dynamics (update every 10 steps). Influences exploration temperature
    and feeds into L0 FiLM conditioning (Phase 3.2b+).

    Traceability: Ax5 → Th14 → DR17 → IS13 → VT22
    """

    def __init__(self, intero_dim: int = 5, mood_dim: int = 16, ema_momentum: float = 0.99):
        super().__init__()
        self.mood_dim = mood_dim
        self.ema_momentum = ema_momentum
        self.mlp = nn.Sequential(
            nn.Linear(intero_dim, 32),
            nn.ReLU(),
            nn.Linear(32, mood_dim),
        )

    def forward(
        self,
        interoceptive: torch.Tensor,  # (B, 5)
        mood_ema: torch.Tensor,       # (B, 16) running EMA accumulator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute new mood vector and updated EMA.

        Args:
            interoceptive: (B, 5) current interoceptive signals
            mood_ema: (B, 16) EMA accumulator from previous slow update

        Returns:
            mood_vector: (B, 16) smoothed mood for this timestep
            new_ema: (B, 16) updated EMA accumulator
        """
        raw = self.mlp(interoceptive)  # (B, 16)
        new_ema = self.ema_momentum * mood_ema + (1.0 - self.ema_momentum) * raw
        return new_ema, new_ema

    def init_mood(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialized (mood_vector, mood_ema) for episode start."""
        z = torch.zeros(batch_size, self.mood_dim, device=device)
        return z, z.clone()

    def exploration_temperature(self, mood_vector: torch.Tensor) -> torch.Tensor:
        """Scalar temperature modifier from mood norm. High arousal → higher temp.

        Returns (B,) values in [0.5, 2.0].
        """
        norm = mood_vector.norm(dim=-1)  # (B,)
        # normalise to [0, 1] assuming mood norms live in ~[0, 4]
        t = (norm / 4.0).clamp(0.0, 1.0)
        return 0.5 + 1.5 * t  # [0.5, 2.0]


# ---------------------------------------------------------------------------
# EmotionModule — Phase 3.2a: fear, surprise, satisfaction
# ---------------------------------------------------------------------------

class EmotionModule(nn.Module):
    """Appraisal-based emotion system. Phase 3.2a active set: fear, surprise, satisfaction.

    Trigger conditions (Barrett constructed emotion theory):
        fear        — reward < -0.5 AND pe > 0.1
        surprise    — pe > 0.5  (any reward valence)
        satisfaction — reward > 0.5 AND pe < 0.1

    Priority: fear > surprise > satisfaction > neutral.

    Traceability: Ax4 → Th13 → DR18 → IS14 → VT23
    """

    # Thresholds
    PE_HIGH: float = 0.1
    PE_VERY_HIGH: float = 0.5
    REWARD_POSITIVE: float = 0.5
    REWARD_NEGATIVE: float = -0.5
    LP_THRESHOLD: float = 0.05
    ENERGY_MIN: float = 0.3
    FAILURE_MIN: int = 3
    MAX_FAILURES: int = 20
    MAX_PE: float = 2.0

    def __init__(self, phase: str = "3.2a"):
        super().__init__()
        self.active = get_active_emotions(phase)

    def compute(
        self,
        pe: float,
        reward: float,
        learning_progress: float = 0.0,
        energy: float = 1.0,
        policy_entropy: float = 1.0,
        failure_count: int = 0,
        teacher_valence: float = 0.0,
        teacher_magnitude: float = 0.0,
    ) -> EmotionTag:
        """Compute emotion tag for one timestep.

        Priority order: fear > surprise > frustration > satisfaction > curiosity > social_approval.
        Only emotions in self.active are evaluated.

        Args:
            pe: prediction error scalar
            reward: extrinsic reward scalar
            learning_progress: Δ(accuracy) over window
            energy: agent energy level [0, 1]
            policy_entropy: policy distribution entropy
            failure_count: consecutive failures
            teacher_valence: teacher emotion valence [-1, 1]
            teacher_magnitude: teacher emotion magnitude [0, 1]

        Returns:
            EmotionTag with type and magnitude
        """
        if "fear" in self.active and reward < self.REWARD_NEGATIVE and pe > self.PE_HIGH:
            return EmotionTag("fear", min(1.0, abs(reward) * pe))

        if "surprise" in self.active and pe > self.PE_VERY_HIGH:
            return EmotionTag("surprise", min(1.0, pe / self.MAX_PE))

        if ("frustration" in self.active and reward < self.REWARD_NEGATIVE
                and pe < self.PE_HIGH and failure_count >= self.FAILURE_MIN):
            return EmotionTag("frustration", min(1.0, failure_count / self.MAX_FAILURES))

        if "satisfaction" in self.active and reward > self.REWARD_POSITIVE and pe < self.PE_HIGH:
            return EmotionTag("satisfaction", min(1.0, reward * (1.0 - pe)))

        if ("curiosity" in self.active and learning_progress > self.LP_THRESHOLD
                and energy > self.ENERGY_MIN):
            return EmotionTag("curiosity", min(1.0, learning_progress / 0.2))

        if "social_approval" in self.active and teacher_valence > 0.5:
            return EmotionTag("social_approval", min(1.0, teacher_valence * teacher_magnitude))

        return EmotionTag("neutral", 0.0)

    def lr_modulation(self, tag: EmotionTag, beta: float = 0.5) -> float:
        """Compute learning rate modulation factor from emotion.

        fear/surprise: boost up to 1.5×; satisfaction: dampen by 0.3×.
        Returns scalar factor (multiply current LR by this).
        """
        if tag.emotion_type in ("fear", "surprise"):
            return 1.0 + beta * tag.magnitude
        if tag.emotion_type == "satisfaction":
            return 1.0 - 0.3 * tag.magnitude
        return 1.0

    def per_priority(self, tag: EmotionTag, alpha: float = 0.6) -> float:
        """Compute PER sampling priority from emotion magnitude.

        P ∝ emotion_magnitude^alpha. Returns 0 for neutral.
        """
        return tag.magnitude ** alpha if tag.magnitude > 0 else 0.0

    def encode_emotion_vector(self, tag: EmotionTag) -> torch.Tensor:
        """Encode emotion as 7-dim vector: 6 one-hot + 1 magnitude.

        Used for EmotionFiLM conditioning at L2.
        """
        type_map = {t: i for i, t in enumerate(EMOTION_TYPES)}
        vec = torch.zeros(7)
        if tag.emotion_type in type_map:
            vec[type_map[tag.emotion_type]] = 1.0
        vec[6] = tag.magnitude
        return vec


# ---------------------------------------------------------------------------
# FrustrationAccumulator
# ---------------------------------------------------------------------------

class FrustrationAccumulator:
    """Cumulative frustration counter for dual-system switching.

    Accumulates frustration magnitude; decays on positive reward.
    Triggers model-free → model-based switch when threshold exceeded.

    Traceability: Ax2 → Th9/Th10 → DR11/DR12 → IS8
    """

    def __init__(self, threshold: float = 5.0, decay: float = 0.95):
        self.threshold = threshold
        self.decay = decay
        self.cumulative: float = 0.0

    def update(self, tag: EmotionTag, reward: float) -> float:
        """Update accumulator and return this step's frustration delta."""
        if tag.emotion_type == "frustration":
            delta = tag.magnitude
            self.cumulative += delta
            return delta
        if reward > 0.0:
            self.cumulative *= self.decay
        return 0.0

    @property
    def should_switch(self) -> bool:
        return self.cumulative > self.threshold

    def reset(self) -> None:
        self.cumulative = 0.0


# ---------------------------------------------------------------------------
# IntrinsicMotivation
# ---------------------------------------------------------------------------

class NoveltyReward(nn.Module):
    """ICM-style novelty: MSE between predicted and actual RSSM latent.

    Traceability: Ax15 → DR19 → IS15
    """

    def compute(
        self,
        z_predicted: torch.Tensor,  # (B, D)
        z_actual: torch.Tensor,     # (B, D)
    ) -> torch.Tensor:
        """Return (B,) novelty reward."""
        return ((z_predicted - z_actual) ** 2).mean(dim=-1)


class LearningProgressReward:
    """Oudeyer-style learning progress: Δ(accuracy) over sliding window.

    Robust to noisy TV because it measures improvement, not raw PE.
    Traceability: Ax15 → DR19 → IS15
    """

    def __init__(self, window: int = 100):
        self.window = window
        self.step_count: int = 0
        self.pe_buffer_prev: deque = deque(maxlen=window)
        self.pe_buffer_curr: deque = deque(maxlen=window)

    def update(self, pe: float) -> float:
        """Ingest one PE value; return current learning progress estimate."""
        self.pe_buffer_curr.append(pe)
        self.step_count += 1

        if self.step_count % self.window == 0:
            self.pe_buffer_prev = deque(self.pe_buffer_curr, maxlen=self.window)
            self.pe_buffer_curr = deque(maxlen=self.window)

        if len(self.pe_buffer_prev) == 0 or len(self.pe_buffer_curr) == 0:
            return 0.0

        acc_prev = 1.0 - sum(self.pe_buffer_prev) / len(self.pe_buffer_prev)
        acc_curr = 1.0 - sum(self.pe_buffer_curr) / len(self.pe_buffer_curr)
        return max(0.0, acc_curr - acc_prev)

    def reset(self) -> None:
        self.step_count = 0
        self.pe_buffer_prev = deque(maxlen=self.window)
        self.pe_buffer_curr = deque(maxlen=self.window)


class MaximumGripReward:
    """Merleau-Ponty maximum grip: reward low PE in recently high-PE regions.

    Tracks PE EMA; when current PE drops below recent EMA in a novel region,
    it signals successful grip (mastery of previously uncertain situation).

    Traceability: Ax15 → DR19 → IS15 → VT11
    """

    def __init__(self, novelty_threshold: float = 0.15, ema_momentum: float = 0.95):
        self.novelty_threshold = novelty_threshold
        self.ema_momentum = ema_momentum
        self.pe_ema: float = 0.5

    def compute(self, pe_current: float) -> float:
        """Return grip reward for this step."""
        pe_recent = self.pe_ema
        self.pe_ema = self.ema_momentum * self.pe_ema + (1.0 - self.ema_momentum) * pe_current
        if pe_recent > self.novelty_threshold:
            return max(0.0, pe_recent - pe_current)
        return 0.0

    def reset(self) -> None:
        self.pe_ema = 0.5


class IntrinsicMotivation(nn.Module):
    """Combined intrinsic reward: novelty + learning progress + maximum grip.

    Phase 3.2a: novelty only (η_PE=0.1). Full three-component in Phase 3.2c+.

    r_intrinsic = eta_PE * r_novelty + eta_LP * r_LP + eta_MG * r_grip
    r_total = r_extrinsic + lambda_intrinsic * r_intrinsic

    Traceability: Ax15 → DR19 → IS15
    """

    ETA_PE: float = 0.1   # novelty weight
    ETA_LP: float = 0.2   # learning progress weight
    ETA_MG: float = 0.1   # maximum grip weight

    def __init__(self, phase: str = "3.2a", lp_window: int = 100):
        super().__init__()
        self.phase = phase
        self.novelty = NoveltyReward()
        self.lp_reward = LearningProgressReward(window=lp_window)
        self.grip_reward = MaximumGripReward()

    def compute(
        self,
        pe: float,
        z_predicted: torch.Tensor | None = None,
        z_actual: torch.Tensor | None = None,
        step: int = 0,
        total_steps: int = 1_000_000,
    ) -> tuple[torch.Tensor, float]:
        """Compute intrinsic reward components.

        Args:
            pe: scalar prediction error for LP and grip
            z_predicted: (B, D) RSSM predicted latent (None → novelty=0)
            z_actual: (B, D) RSSM actual latent (None → novelty=0)
            step: current training step for lambda annealing
            total_steps: total training steps for lambda annealing

        Returns:
            r_intrinsic: (B,) or scalar(0) combined intrinsic reward
            lp: float learning progress for this step
        """
        lp = self.lp_reward.update(pe)
        r_grip = self.grip_reward.compute(pe)
        lambda_i = self._lambda(step, total_steps)

        if z_predicted is not None and z_actual is not None:
            r_novelty = self.novelty.compute(z_predicted, z_actual)  # (B,)
        else:
            r_novelty = torch.zeros(1)

        # Phase-gated: 3.2a uses novelty only; 3.2c+ adds LP and grip
        if self.phase in ("3.2a", "3.2b"):
            r_int = self.ETA_PE * r_novelty
        else:
            r_int = (self.ETA_PE * r_novelty
                     + self.ETA_LP * lp
                     + self.ETA_MG * r_grip)

        return lambda_i * r_int, lp

    @staticmethod
    def _lambda(step: int, total_steps: int,
                start: float = 1.0, end: float = 0.1) -> float:
        """Anneal lambda_intrinsic from 1.0 → 0.1 over training."""
        progress = min(1.0, step / max(1, total_steps))
        return start + progress * (end - start)

    def reset(self) -> None:
        self.lp_reward.reset()
        self.grip_reward.reset()
