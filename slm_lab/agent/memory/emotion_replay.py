"""EmotionTaggedReplayBuffer — PER replay with emotion tags and stage-aware sampling.

Axiom trace: Ax5 → Th14 → DR17 → IS13 → VT22
See: notes/layers/continual-learning.md §3
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# Valid emotion types from L3 (6 types + neutral)
EmotionType = Literal["fear", "surprise", "satisfaction", "curiosity", "frustration", "social_approval", "neutral"]

EMOTION_TYPES: tuple[str, ...] = ("fear", "surprise", "satisfaction", "curiosity", "frustration", "social_approval", "neutral")


@dataclass
class Transition:
    """A single agent transition with emotion metadata.

    Axiom trace: Ax5 → Th14 → DR17 → IS13 → VT22
    """
    state: np.ndarray           # observation vector
    action: np.ndarray          # action vector
    reward: float
    next_state: np.ndarray      # observation vector
    done: bool
    emotion_type: str           # one of EMOTION_TYPES
    emotion_magnitude: float    # [0, 1]
    prediction_error: float     # TD error or world-model surprise
    stage_name: str             # developmental stage name, e.g. "pavlovian"

    def __post_init__(self) -> None:
        if self.emotion_type not in EMOTION_TYPES:
            raise ValueError(f"emotion_type must be one of {EMOTION_TYPES}, got {self.emotion_type!r}")
        if not (0.0 <= self.emotion_magnitude <= 1.0):
            raise ValueError(f"emotion_magnitude must be in [0, 1], got {self.emotion_magnitude}")


class _SumTree:
    """Binary sum tree for O(log n) PER insertion and sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        # tree[0] = root (total sum); leaves at [capacity-1 .. 2*capacity-2]
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write = 0  # circular write pointer into leaves

    # ------------------------------------------------------------------
    def _propagate(self, leaf_idx: int, delta: float) -> None:
        idx = leaf_idx
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def _leaf_idx(self, pos: int) -> int:
        """Convert circular position → tree leaf index."""
        return pos + self.capacity - 1

    # ------------------------------------------------------------------
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, pos: int) -> None:
        """Insert priority at circular position pos."""
        leaf = self._leaf_idx(pos)
        delta = priority - self.tree[leaf]
        self.tree[leaf] = priority
        self._propagate(leaf, delta)

    def update(self, pos: int, priority: float) -> None:
        self.add(priority, pos)

    def get(self, s: float) -> int:
        """Return circular position for cumulative value s."""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                # idx is a leaf
                return idx - (self.capacity - 1)
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right

    def sample_batch(self, n: int) -> np.ndarray:
        """Sample n positions proportional to priority."""
        total = self.total()
        if total <= 0:
            return np.random.randint(0, self.capacity, size=n)
        segments = np.linspace(0, total, n + 1)
        positions = np.empty(n, dtype=np.int64)
        for i in range(n):
            s = np.random.uniform(segments[i], segments[i + 1])
            # clamp to avoid floating-point edge overrun
            positions[i] = self.get(min(s, total - 1e-12))
        return positions


class EmotionTaggedReplayBuffer:
    """Prioritized replay buffer with emotion tags and stage-aware old/new mixing.

    Capacity is split into two partitions:
      - current partition  : (1 - old_stage_reserve) × capacity  circular buffer
      - old partition      : old_stage_reserve × capacity         promoted at stage boundaries

    Priority: P(t) ∝ (emotion_magnitude + ε)^α, α=0.6.
    IS correction: weights = (N · P)^(-β) / max_weight, β anneals 0.4 → 1.0.

    Axiom trace: Ax5 → Th14 → DR17 → IS13 → VT22
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        old_stage_reserve: float = 0.10,
        priority_alpha: float = 0.6,
        is_beta_start: float = 0.4,
        is_beta_end: float = 1.0,
        is_beta_steps: int = 1_000_000,
        epsilon: float = 1e-6,
    ) -> None:
        if not (0.0 < old_stage_reserve < 1.0):
            raise ValueError("old_stage_reserve must be in (0, 1)")
        self.capacity = capacity
        self.alpha = priority_alpha
        self.epsilon = epsilon
        self.is_beta_start = is_beta_start
        self.is_beta_end = is_beta_end
        self.is_beta_steps = is_beta_steps
        self._step = 0  # global step counter for beta annealing

        # Partition sizes
        old_cap = int(capacity * old_stage_reserve)
        cur_cap = capacity - old_cap
        self.old_capacity = old_cap
        self.current_capacity = cur_cap

        # Current-stage circular buffer
        self._current: list[Transition | None] = [None] * cur_cap
        self._cur_tree = _SumTree(cur_cap)
        self._cur_head = -1    # write pointer
        self._cur_size = 0

        # Old-stage fixed buffer (sorted by emotion_magnitude descending)
        self._old: list[Transition] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._cur_size + len(self._old)

    @property
    def _beta(self) -> float:
        """IS beta annealed linearly from start to end over is_beta_steps."""
        frac = min(self._step / max(self.is_beta_steps, 1), 1.0)
        return self.is_beta_start + frac * (self.is_beta_end - self.is_beta_start)

    # ------------------------------------------------------------------
    # Core add / sample
    # ------------------------------------------------------------------

    def _priority(self, emotion_magnitude: float) -> float:
        return (emotion_magnitude + self.epsilon) ** self.alpha

    def add(self, transition: Transition) -> None:
        """Add transition to current-stage buffer with emotion-weighted priority."""
        self._cur_head = (self._cur_head + 1) % self.current_capacity
        self._current[self._cur_head] = transition
        p = self._priority(transition.emotion_magnitude)
        self._cur_tree.add(p, self._cur_head)
        if self._cur_size < self.current_capacity:
            self._cur_size += 1

    def _prioritized_sample_current(self, n: int) -> list[Transition]:
        """Sample n transitions from current partition via PER."""
        if self._cur_size == 0:
            return []
        n = min(n, self._cur_size)
        positions = self._cur_tree.sample_batch(n)
        return [self._current[int(pos)] for pos in positions if self._current[int(pos)] is not None]

    def _is_weights(self, positions: np.ndarray) -> np.ndarray:
        """Importance-sampling weights for current-partition samples."""
        beta = self._beta
        total = self._cur_tree.total()
        if total <= 0 or self._cur_size == 0:
            return np.ones(len(positions), dtype=np.float32)
        probs = np.array([self._cur_tree.tree[self._cur_tree._leaf_idx(int(p))] / total for p in positions])
        probs = np.clip(probs, 1e-12, None)
        weights = (self._cur_size * probs) ** (-beta)
        return (weights / weights.max()).astype(np.float32)

    def sample_batch(
        self,
        batch_size: int,
        old_ratio: float = 0.10,
    ) -> tuple[list[Transition], np.ndarray]:
        """Sample batch with old/new mixing.

        Args:
            batch_size: total transitions to return.
            old_ratio: fraction from old partition (0.10 default per spec §3.2).

        Returns:
            (transitions, is_weights) — is_weights are 1.0 for old-partition samples.
        """
        self._step += 1
        n_old = int(batch_size * old_ratio)
        n_new = batch_size - n_old

        # Old partition: uniform sample (already high-emotion curated)
        if self._old and n_old > 0:
            idxs = np.random.choice(len(self._old), size=min(n_old, len(self._old)), replace=False)
            old_samples = [self._old[i] for i in idxs]
        else:
            old_samples = []

        # Current partition: PER sample
        n_new = batch_size - len(old_samples)
        cur_positions = self._cur_tree.sample_batch(min(n_new, self._cur_size)) if self._cur_size > 0 else np.array([], dtype=np.int64)
        new_samples = [self._current[int(p)] for p in cur_positions if self._current[int(p)] is not None]

        # IS weights: 1.0 for old partition, computed for new
        old_weights = np.ones(len(old_samples), dtype=np.float32)
        new_weights = self._is_weights(cur_positions) if len(cur_positions) > 0 else np.array([], dtype=np.float32)

        transitions = old_samples + new_samples
        is_weights = np.concatenate([old_weights, new_weights])
        return transitions, is_weights

    # ------------------------------------------------------------------
    # Stage boundary: promote top-k to old partition
    # ------------------------------------------------------------------

    def promote_to_old(self, stage_name: str, n_samples: int | None = None) -> int:
        """Move top-k high-emotion transitions from current to old partition.

        Called at stage boundaries. If old partition is full, retains only
        the highest-emotion transitions up to old_capacity.

        Args:
            stage_name: name of completed stage (used for logging/filtering).
            n_samples: how many to promote (default: old_capacity // 4).

        Returns:
            Number of transitions promoted.
        """
        if self._cur_size == 0:
            return 0
        if n_samples is None:
            n_samples = max(1, self.old_capacity // 4)

        active = [t for t in self._current if t is not None and t.stage_name == stage_name]
        if not active:
            # fallback: any non-None
            active = [t for t in self._current if t is not None]
        if not active:
            return 0

        active.sort(key=lambda t: t.emotion_magnitude, reverse=True)
        promoted = active[:n_samples]
        self._old.extend(promoted)

        # Trim old partition: keep only highest-emotion up to old_capacity
        if len(self._old) > self.old_capacity:
            self._old.sort(key=lambda t: t.emotion_magnitude, reverse=True)
            self._old = self._old[: self.old_capacity]

        return len(promoted)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def old_size(self) -> int:
        return len(self._old)

    def current_size(self) -> int:
        return self._cur_size

    def stage_counts(self) -> dict[str, int]:
        """Count current-partition transitions by stage_name."""
        counts: dict[str, int] = {}
        for t in self._current:
            if t is not None:
                counts[t.stage_name] = counts.get(t.stage_name, 0) + 1
        return counts
