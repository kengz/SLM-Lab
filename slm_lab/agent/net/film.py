"""FiLM (Feature-wise Linear Modulation) conditioning layers.

FiLMLayer:          generic conditioning vector → γ, β; identity init.
MoodFiLMLayer:      mood (16-dim) → 3 separate FiLM instances for DINOv2 blocks 8, 16, 24.
EmotionFiLMLayer:   emotion encoding (7-dim) → FiLM for L2 policy features.
SomaticMarkerSystem: cosine-similarity retrieval from replay buffer, top-k=5.

Source: notes/layers/L0-perception.md §2.7, notes/layers/L3-mood-emotion.md §3.3, §5.3, §7, §8.2
Traceability: Ax5 → Th14 → DR17 → IS13 → VT22 | Ax4 → Th13 → DR18 → IS14 → VT23
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm_lab.agent.net.emotion import EmotionTag


# ---------------------------------------------------------------------------
# FiLMLayer — generic base
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: h' = γ(cond) * h + β(cond).

    Identity init (Flamingo zero-gating): γ=1, β=0 at construction.
    At init, conditioning is a no-op; modulation is learned gradually.

    Args:
        feature_dim: dimensionality of the feature to modulate
        cond_dim:    dimensionality of the conditioning vector
    """

    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feature_dim)
        self.beta = nn.Linear(cond_dim, feature_dim)
        # Identity init: zeros → output 0, so 1.0 + 0 = 1 for gamma, 0 for beta
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM: h' = (1 + γ(cond)) * h + β(cond).

        Args:
            x:    (B, ..., feature_dim) features to modulate
            cond: (B, cond_dim) conditioning vector

        Returns:
            Modulated tensor of same shape as x.
        """
        # Broadcast over sequence/patch dimensions if present
        gamma = 1.0 + self.gamma(cond)  # (B, feature_dim)
        beta = self.beta(cond)           # (B, feature_dim)

        # Expand to x's shape for broadcasting: add dims between B and feature_dim
        for _ in range(x.dim() - 2):
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return gamma * x + beta


# ---------------------------------------------------------------------------
# MoodFiLMLayer — 3 FiLM instances for DINOv2 blocks 8, 16, 24
# ---------------------------------------------------------------------------

DINO_INSERTION_BLOCKS = (8, 16, 24)


class MoodFiLMLayer(nn.Module):
    """Mood (16-dim) → FiLM conditioning at DINOv2 ViT-L blocks 8, 16, 24.

    Three independent FiLM instances, one per insertion point. Each modulates
    the 1024-dim patch features after the corresponding transformer block.
    Updated every 10 control steps (2.5 Hz). Params: ~104K total.

    Traceability: Ax5 → Th14 → DR17 → IS13 → VT22
    """

    BLOCKS = DINO_INSERTION_BLOCKS
    FEATURE_DIM = 1024   # DINOv2 ViT-L hidden dim
    MOOD_DIM = 16

    def __init__(self, feature_dim: int = FEATURE_DIM, mood_dim: int = MOOD_DIM):
        super().__init__()
        self.film_block8 = FiLMLayer(feature_dim, mood_dim)
        self.film_block16 = FiLMLayer(feature_dim, mood_dim)
        self.film_block24 = FiLMLayer(feature_dim, mood_dim)
        self._layers = {8: self.film_block8, 16: self.film_block16, 24: self.film_block24}

    def forward(self, h: torch.Tensor, mood: torch.Tensor, block: int) -> torch.Tensor:
        """Apply mood FiLM at the specified DINOv2 block.

        Args:
            h:     (B, N_tokens, 1024) patch features after DINOv2 block
            mood:  (B, 16) current mood vector
            block: which insertion block (must be in {8, 16, 24})

        Returns:
            (B, N_tokens, 1024) modulated features
        """
        if block not in self._layers:
            raise ValueError(f"block must be one of {self.BLOCKS}, got {block}")
        return self._layers[block](h, mood)


# ---------------------------------------------------------------------------
# EmotionFiLMLayer — emotion encoding → FiLM on policy features
# ---------------------------------------------------------------------------

class EmotionFiLMLayer(nn.Module):
    """Emotion encoding (7-dim) → FiLM on L2 policy features (512-dim).

    Emotion vector: 6 one-hot (fear/surprise/satisfaction/frustration/curiosity/
    social_approval) + 1 magnitude scalar. Applied per-step at L2.
    Params: ~8.2K.

    Traceability: Ax4 → Th13 → DR18 → IS14 → VT23
    """

    FEATURE_DIM = 512
    EMOTION_DIM = 7

    def __init__(self, feature_dim: int = FEATURE_DIM, emotion_dim: int = EMOTION_DIM):
        super().__init__()
        self.film = FiLMLayer(feature_dim, emotion_dim)

    def forward(self, h: torch.Tensor, emotion_vec: torch.Tensor) -> torch.Tensor:
        """Apply emotion FiLM to policy features.

        Args:
            h:           (B, feature_dim) policy features from L2
            emotion_vec: (B, 7) or (7,) encoded emotion vector

        Returns:
            (B, feature_dim) modulated features
        """
        if emotion_vec.dim() == 1:
            emotion_vec = emotion_vec.unsqueeze(0).expand(h.shape[0], -1)
        return self.film(h, emotion_vec)

    @staticmethod
    def encode(tag: EmotionTag) -> torch.Tensor:
        """Encode EmotionTag → 7-dim vector (6 one-hot + 1 magnitude).

        Delegates to EmotionModule.encode_emotion_vector logic inline to avoid
        circular dependency (EmotionModule already has this method, but keeping
        it here for self-contained use in FiLM pipeline).
        """
        from slm_lab.agent.net.emotion import EMOTION_TYPES
        type_map = {t: i for i, t in enumerate(EMOTION_TYPES)}
        vec = torch.zeros(7)
        if tag.emotion_type in type_map:
            vec[type_map[tag.emotion_type]] = 1.0
        vec[6] = tag.magnitude
        return vec


# ---------------------------------------------------------------------------
# SomaticMarkerSystem — cosine-similarity retrieval from replay buffer
# ---------------------------------------------------------------------------

class SomaticMarkerSystem:
    """Damasio somatic marker hypothesis: emotion-tagged memories bias action.

    Retrieves top-k transitions from replay buffer by cosine similarity to
    the current being embedding. Returns a somatic_bias ∈ [-1, 1] that soft-
    biases the L2 value function.

    Traceability: L3-mood-emotion.md §7
    """

    VALENCE_MAP: dict[str, float] = {
        "fear": -1.0,
        "frustration": -0.5,
        "surprise": 0.0,
        "curiosity": 0.3,
        "satisfaction": 1.0,
        "social_approval": 0.7,
        "neutral": 0.0,
    }

    def __init__(
        self,
        replay_buffer,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ):
        self.buffer = replay_buffer
        self.top_k = top_k
        self.threshold = similarity_threshold

    def query(self, current_be: torch.Tensor) -> float:
        """Return somatic bias ∈ [-1, 1] for current being embedding.

        Args:
            current_be: (512,) or (1, 512) current being embedding

        Returns:
            Weighted-average valence of top-k similar emotion-tagged memories.
            Returns 0.0 if no memories exceed similarity threshold.
        """
        if current_be.dim() > 1:
            current_be = current_be.squeeze(0)

        transitions = self.buffer.sample_recent(n=1000)
        if not transitions:
            return 0.0

        candidates: list[tuple[float, object]] = []
        for t in transitions:
            state = t.state
            if state.dim() > 1:
                state = state.squeeze(0)
            sim = F.cosine_similarity(
                current_be.unsqueeze(0), state.unsqueeze(0)
            ).item()
            if sim > self.threshold:
                candidates.append((sim, t))

        if not candidates:
            return 0.0

        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[: self.top_k]

        total_weight = 0.0
        total_signal = 0.0
        for sim, trans in top:
            valence = self.VALENCE_MAP.get(trans.emotion_type, 0.0)
            weight = sim * trans.emotion_magnitude
            total_signal += weight * valence
            total_weight += weight

        return total_signal / total_weight if total_weight > 0.0 else 0.0
