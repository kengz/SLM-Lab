"""Pavlovian conditioning environment for SLM-Lab.

2D kinematic arena for TC-01 through TC-10. No physics engine — kinematics only.
All classical conditioning tasks use a two-phase protocol: acquisition (shaped)
then probe (CS-alone, no reward). Operant tasks (TC-07 to TC-10) are single-phase.

Registered as SLM/Pavlovian-v0 in slm_lab/env/__init__.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARENA_SIZE: float = 10.0
DT: float = 1.0 / 30.0           # 30 Hz
CONTACT_RADIUS: float = 0.6       # metres
AGENT_RADIUS: float = 0.25
MAX_ENERGY: float = 100.0
ENERGY_DECAY: float = 0.1         # per step
FORWARD_COST: float = 0.01
ANGULAR_COST: float = 0.005
MAX_FORWARD: float = 1.0          # m/s
MAX_ANGULAR: float = math.pi / 2  # rad/s
FOV_RANGE: float = 15.0           # visibility radius (>= arena diagonal)
FOV_HALF_ANGLE: float = math.pi   # 360-degree FOV (egocentric frame)

# Object indices
OBJ_RED = 0    # red sphere  → reward target
OBJ_BLUE = 1   # blue cube   → penalty
OBJ_GREEN = 2  # green cyl   → neutral / secondary cue

OBS_DIM = 18
ACT_DIM = 2

# Phase names
PHASE_ACQUISITION = "acquisition"
PHASE_PROBE = "probe"
PHASE_EXTINCTION = "extinction"
PHASE_REST = "rest"

# Valid task names
TASKS = (
    "stimulus_response",      # TC-01
    "temporal_contingency",   # TC-02
    "extinction",             # TC-03
    "spontaneous_recovery",   # TC-04
    "generalization",         # TC-05
    "discrimination",         # TC-06
    "reward_contingency",     # TC-07
    "partial_reinforcement",  # TC-08
    "shaping",                # TC-09
    "chaining",               # TC-10
)


# ---------------------------------------------------------------------------
# Internal state dataclass
# ---------------------------------------------------------------------------

@dataclass
class _AgentState:
    x: float = 5.0
    y: float = 5.0
    heading: float = 0.0
    energy: float = MAX_ENERGY
    v_forward: float = 0.0
    v_angular: float = 0.0


@dataclass
class _ObjectState:
    x: float = 0.0
    y: float = 0.0
    active: bool = True           # whether the object should appear


@dataclass
class _TrialState:
    """Per-task trial tracking state."""
    phase: str = PHASE_ACQUISITION
    trial: int = 0                   # trial counter within current phase
    step_in_trial: int = 0           # step counter within current trial
    cs_active: bool = False
    cs_signal: float = 0.0           # obs[17] value
    prev_dist_to_red: float = 0.0
    # Acquisition-phase metrics
    acq_approaches: list[bool] = field(default_factory=list)
    # Probe-phase metrics
    probe_approaches: list[bool] = field(default_factory=list)
    probe_trial_approached: bool = False
    # Timing for TC-02
    approach_time: int | None = None
    # Discrimination: current trial type
    disc_cs_type: str = "plus"       # "plus" or "minus"
    # Chaining
    chain_step: int = 0              # 0=need green, 1=need blue, 2=need red
    chains_completed: int = 0
    chains_attempted: int = 0
    # Generalization: current test stimulus level
    gen_stimulus_level: float = 1.0
    responses_by_strength: dict = field(default_factory=dict)
    # TC-04 rest countdown
    rest_steps_remaining: int = 0
    # TC-09 shaping comparison
    condition: str = "shaped"        # "shaped" | "unshaped"
    shaped_successes: list[bool] = field(default_factory=list)
    unshaped_successes: list[bool] = field(default_factory=list)
    # ITI approach tracking (control metric)
    iti_approaches: list[bool] = field(default_factory=list)
    iti_trial_approached: bool = False
    # Operant: step counters
    total_steps: int = 0
    # Partial reinforcement state
    reward_this_step: bool = True


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

class PavlovianEnv(gym.Env):
    """2D kinematic Pavlovian conditioning arena.

    Args:
        task: One of the 10 TASKS strings.
        arena_size: Side length of the square arena in metres.
        dt: Simulation timestep in seconds.
        max_energy: Initial and maximum agent energy.
        energy_decay: Energy lost per step (before movement costs).
        contact_radius: Object contact detection radius.
        shaping_scale: Distance-shaping reward scale (acquisition phases only).
        seed: RNG seed.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task: str = "stimulus_response",
        arena_size: float = ARENA_SIZE,
        dt: float = DT,
        max_energy: float = MAX_ENERGY,
        energy_decay: float = ENERGY_DECAY,
        contact_radius: float = CONTACT_RADIUS,
        shaping_scale: float = 1.0,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        if task not in TASKS:
            raise ValueError(f"Unknown task '{task}'. Valid: {TASKS}")

        self.task = task
        self.arena_size = arena_size
        self.dt = dt
        self.max_energy = max_energy
        self.energy_decay = energy_decay
        self.contact_radius = contact_radius
        self.shaping_scale = shaping_scale
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        # action[0]: forward velocity [-1, 1] (negative = backward, clamped to 0)
        # action[1]: angular velocity [-1, 1] (rescaled to ±π/2)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self._agent = _AgentState()
        self._objects: list[_ObjectState] = [_ObjectState() for _ in range(3)]
        self._ts = _TrialState()
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._reset_agent()
        self._reset_objects()
        self._ts = _TrialState()
        self._ts.prev_dist_to_red = self._dist_to(OBJ_RED)
        self._init_task_state()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, -1.0, 1.0)
        v_forward = float(max(0.0, action[0])) * MAX_FORWARD
        v_angular = float(action[1]) * MAX_ANGULAR

        self._agent.v_forward = v_forward
        self._agent.v_angular = v_angular
        self._step_count += 1

        # Kinematics
        self._agent.heading += v_angular * self.dt
        self._agent.heading = _wrap_angle(self._agent.heading)
        self._agent.x += v_forward * math.cos(self._agent.heading) * self.dt
        self._agent.y += v_forward * math.sin(self._agent.heading) * self.dt
        self._agent.x = float(np.clip(self._agent.x, 0.0, self.arena_size))
        self._agent.y = float(np.clip(self._agent.y, 0.0, self.arena_size))

        # Energy
        self._agent.energy -= self.energy_decay
        self._agent.energy -= v_forward * FORWARD_COST
        self._agent.energy -= abs(v_angular) * ANGULAR_COST

        # Task-specific reward and state updates
        reward = self._step_task()

        self._ts.prev_dist_to_red = self._dist_to(OBJ_RED)

        terminated = self._agent.energy <= 0.0
        truncated = False
        obs = self._get_obs()
        info = self._get_info()

        return obs, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        return self._render_frame()

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _reset_agent(self):
        """Place agent near centre with random heading."""
        cx, cy = self.arena_size / 2, self.arena_size / 2
        r = self._rng.uniform(0.0, 1.5)
        theta = self._rng.uniform(0.0, 2 * math.pi)
        self._agent = _AgentState(
            x=float(np.clip(cx + r * math.cos(theta), 0.5, self.arena_size - 0.5)),
            y=float(np.clip(cy + r * math.sin(theta), 0.5, self.arena_size - 0.5)),
            heading=self._rng.uniform(-math.pi, math.pi),
            energy=self.max_energy,
        )

    def _reset_objects(self):
        """Place objects at fixed positions with some random jitter."""
        base_positions = [
            (7.5, 7.5),  # red
            (2.5, 7.5),  # blue
            (7.5, 2.5),  # green
        ]
        for i, (bx, by) in enumerate(base_positions):
            jx = self._rng.uniform(-0.5, 0.5)
            jy = self._rng.uniform(-0.5, 0.5)
            self._objects[i] = _ObjectState(
                x=float(np.clip(bx + jx, 0.5, self.arena_size - 0.5)),
                y=float(np.clip(by + jy, 0.5, self.arena_size - 0.5)),
                active=True,
            )

    def _init_task_state(self):
        """Task-specific initialisation of trial state."""
        ts = self._ts
        if self.task == "generalization":
            ts.gen_stimulus_level = 1.0
            ts.responses_by_strength = {1.0: [], 0.8: [], 0.6: [], 0.4: [], 0.2: []}
        elif self.task == "discrimination":
            ts.disc_cs_type = self._rng.choice(["plus", "minus"])
        elif self.task == "spontaneous_recovery":
            ts.rest_steps_remaining = 0
        elif self.task == "shaping":
            ts.condition = "shaped"
        elif self.task == "chaining":
            ts.chain_step = 0
            ts.chains_completed = 0
            ts.chains_attempted = 0

    # ------------------------------------------------------------------
    # Task-specific step logic
    # ------------------------------------------------------------------

    def _step_task(self) -> float:
        dispatch = {
            "stimulus_response": self._step_tc01,
            "temporal_contingency": self._step_tc02,
            "extinction": self._step_tc03,
            "spontaneous_recovery": self._step_tc04,
            "generalization": self._step_tc05,
            "discrimination": self._step_tc06,
            "reward_contingency": self._step_tc07,
            "partial_reinforcement": self._step_tc08,
            "shaping": self._step_tc09,
            "chaining": self._step_tc10,
        }
        return dispatch[self.task]()

    # ----- TC-01: Stimulus-Response Association -----

    def _step_tc01(self) -> float:
        """Two-phase: acquisition (shaped, rewarded) → probe (CS-alone)."""
        ts = self._ts
        reward = 0.0
        CS_DUR = 30
        ITI_DUR = 60
        ACQ_TRIALS = 40
        PROBE_TRIALS = 50

        # Advance trial counter
        cycle = CS_DUR + ITI_DUR
        ts.total_steps += 1
        step_in_cycle = (ts.total_steps - 1) % cycle

        # Determine phase
        current_trial = (ts.total_steps - 1) // cycle
        if ts.phase == PHASE_ACQUISITION and current_trial >= ACQ_TRIALS:
            ts.phase = PHASE_PROBE
        elif ts.phase == PHASE_PROBE and (current_trial - ACQ_TRIALS) >= PROBE_TRIALS:
            # Keep running; episode terminates via energy
            pass

        # Determine CS state
        in_cs = step_in_cycle >= ITI_DUR
        ts.cs_signal = 1.0 if in_cs else 0.0
        ts.cs_active = in_cs

        in_probe = ts.phase == PHASE_PROBE
        trial_index = current_trial - (ACQ_TRIALS if in_probe else 0)

        if in_cs:
            dist = self._dist_to(OBJ_RED)
            contacted_red = dist < self.contact_radius

            if contacted_red:
                if ts.phase == PHASE_ACQUISITION:
                    reward += 10.0
                    self._agent.energy += 10.0
                # Record approach for current trial
                if not ts.probe_trial_approached:
                    ts.probe_trial_approached = True

            # Distance-based shaping (acquisition only)
            if ts.phase == PHASE_ACQUISITION:
                shaping = self.shaping_scale * max(0.0, ts.prev_dist_to_red - dist)
                reward += shaping

            # Blue penalty
            if self._dist_to(OBJ_BLUE) < self.contact_radius:
                reward -= 5.0
                self._agent.energy -= 5.0
        else:
            # ITI: track undirected approaches
            if not ts.iti_trial_approached and self._dist_to(OBJ_RED) < self.contact_radius:
                ts.iti_trial_approached = True

        # Trial boundary: record and reset
        if step_in_cycle == cycle - 1:
            if in_probe:
                ts.probe_approaches.append(ts.probe_trial_approached)
            else:
                ts.acq_approaches.append(ts.probe_trial_approached)
            ts.iti_approaches.append(ts.iti_trial_approached)
            ts.probe_trial_approached = False
            ts.iti_trial_approached = False

        return reward

    # ----- TC-02: Temporal Contingency Learning -----

    def _step_tc02(self) -> float:
        """Acquisition: single delay (30 steps). Probe: multi-delay [15, 30, 60]."""
        ts = self._ts
        reward = 0.0
        ACQ_TRIALS = 40
        PROBE_TRIALS = 60  # 20 per delay
        ACQ_DELAY = 30
        PROBE_DELAYS = [15, 30, 60]
        ITI_DUR = 90

        ts.total_steps += 1

        # Select delay for current trial
        if ts.phase == PHASE_ACQUISITION:
            delay = ACQ_DELAY
            cs_dur = delay + 10  # window: delay ±20% + buffer
            cycle = ITI_DUR + cs_dur
            current_trial = ts.total_steps // cycle
            step_in_cycle = ts.total_steps % cycle
            if current_trial >= ACQ_TRIALS:
                ts.phase = PHASE_PROBE
                ts.trial = 0
                ts.approach_time = None
        else:
            delay_idx = (ts.trial % len(PROBE_DELAYS))
            delay = PROBE_DELAYS[delay_idx]
            cs_dur = delay + 10
            cycle = ITI_DUR + cs_dur
            current_trial = ts.trial
            step_in_cycle = ts.step_in_trial
            ts.step_in_trial += 1

        in_cs = ts.phase == PHASE_PROBE and ts.step_in_trial > 0 or (
            ts.phase == PHASE_ACQUISITION and (ts.total_steps % cycle) >= ITI_DUR
        )
        if ts.phase == PHASE_ACQUISITION:
            step_in_cycle = ts.total_steps % cycle
            in_cs = step_in_cycle >= ITI_DUR

        ts.cs_signal = 1.0 if in_cs else 0.0
        ts.cs_active = in_cs

        if in_cs:
            dist = self._dist_to(OBJ_RED)
            contacted = dist < self.contact_radius
            if ts.approach_time is None and contacted:
                if ts.phase == PHASE_ACQUISITION:
                    t_in_cs = (ts.total_steps % cycle) - ITI_DUR
                else:
                    t_in_cs = ts.step_in_trial
                ts.approach_time = t_in_cs

            if ts.phase == PHASE_ACQUISITION:
                t_in_cs = (ts.total_steps % cycle) - ITI_DUR
                in_us_window = int(0.8 * ACQ_DELAY) <= t_in_cs <= int(1.2 * ACQ_DELAY)
                if contacted and in_us_window:
                    reward += 10.0
                    self._agent.energy += 10.0
                # Shaping
                shaping = self.shaping_scale * max(0.0, ts.prev_dist_to_red - dist)
                reward += shaping

        # Trial boundary for probe
        if ts.phase == PHASE_PROBE:
            delay_idx = ts.trial % len(PROBE_DELAYS)
            delay = PROBE_DELAYS[delay_idx]
            cs_dur = delay + 10
            if ts.step_in_trial >= ITI_DUR + cs_dur:
                ts.probe_approaches.append({
                    "approach_time": ts.approach_time,
                    "trained_delay": ACQ_DELAY,
                    "test_delay": delay,
                })
                ts.trial += 1
                ts.step_in_trial = 0
                ts.approach_time = None
                if ts.trial >= PROBE_TRIALS:
                    pass  # completed; episode ends via energy

        # Acquisition trial boundary
        if ts.phase == PHASE_ACQUISITION:
            cycle_a = ITI_DUR + ACQ_DELAY + 10
            if (ts.total_steps % cycle_a) == 0:
                ts.acq_approaches.append({"approach_time": ts.approach_time})
                ts.approach_time = None

        return reward

    # ----- TC-03: Extinction -----

    def _step_tc03(self) -> float:
        """Acquisition (shaped) → extinction (CS-alone, no reward)."""
        ts = self._ts
        reward = 0.0
        CS_DUR = 30
        ITI_DUR = 60
        ACQ_TRIALS = 40
        EXT_TRIALS = 50
        cycle = CS_DUR + ITI_DUR

        ts.total_steps += 1
        step_in_cycle = (ts.total_steps - 1) % cycle
        current_trial = (ts.total_steps - 1) // cycle

        # Phase transitions
        if ts.phase == PHASE_ACQUISITION:
            if current_trial >= ACQ_TRIALS:
                # Acquisition gate: check last 10 trials
                last10 = ts.acq_approaches[-10:] if len(ts.acq_approaches) >= 10 else ts.acq_approaches
                acq_rate = sum(last10) / max(len(last10), 1)
                if acq_rate < 0.60:
                    ts.phase = "acquisition_failed"
                else:
                    ts.phase = PHASE_EXTINCTION
                    ts.trial = 0
        elif ts.phase == PHASE_EXTINCTION:
            ext_trial = current_trial - ACQ_TRIALS
            if ext_trial >= EXT_TRIALS:
                pass

        in_cs = step_in_cycle >= ITI_DUR
        ts.cs_signal = 1.0 if in_cs else 0.0
        ts.cs_active = in_cs

        if in_cs and ts.phase not in ("acquisition_failed",):
            dist = self._dist_to(OBJ_RED)
            contacted = dist < self.contact_radius
            if contacted:
                if ts.phase == PHASE_ACQUISITION:
                    reward += 10.0
                    self._agent.energy += 10.0
                if not ts.probe_trial_approached:
                    ts.probe_trial_approached = True
            if ts.phase == PHASE_ACQUISITION:
                reward += self.shaping_scale * max(0.0, ts.prev_dist_to_red - dist)
            # Blue penalty always active
            if self._dist_to(OBJ_BLUE) < self.contact_radius:
                reward -= 5.0
                self._agent.energy -= 5.0

        if step_in_cycle == cycle - 1:
            if ts.phase == PHASE_ACQUISITION:
                ts.acq_approaches.append(ts.probe_trial_approached)
            elif ts.phase == PHASE_EXTINCTION:
                ts.probe_approaches.append(ts.probe_trial_approached)
            ts.probe_trial_approached = False

        return reward

    # ----- TC-04: Spontaneous Recovery -----

    def _step_tc04(self) -> float:
        """Acq (30 trials) → extinction (30 trials) → rest (150 steps) → probe (10 trials)."""
        ts = self._ts
        reward = 0.0
        CS_DUR = 30
        ITI_DUR = 60
        ACQ_TRIALS = 30
        EXT_TRIALS = 30
        REST_STEPS = 150
        PROBE_TRIALS = 10
        cycle = CS_DUR + ITI_DUR

        ts.total_steps += 1
        step_in_cycle = (ts.total_steps - 1) % cycle
        current_trial = (ts.total_steps - 1) // cycle

        # Phase transitions
        if ts.phase == PHASE_ACQUISITION and current_trial >= ACQ_TRIALS:
            last10 = ts.acq_approaches[-10:] if len(ts.acq_approaches) >= 10 else ts.acq_approaches
            acq_rate = sum(last10) / max(len(last10), 1)
            if acq_rate < 0.50:
                ts.phase = "acquisition_failed"
            else:
                ts.phase = PHASE_EXTINCTION
                ts.trial = 0
        elif ts.phase == PHASE_EXTINCTION:
            ext_trial = current_trial - ACQ_TRIALS
            if ext_trial >= EXT_TRIALS:
                last10 = ts.probe_approaches[-10:] if len(ts.probe_approaches) >= 10 else ts.probe_approaches
                ext_rate = sum(last10) / max(len(last10), 1)
                last10_acq = ts.acq_approaches[-10:] if len(ts.acq_approaches) >= 10 else ts.acq_approaches
                acq_rate = sum(last10_acq) / max(len(last10_acq), 1)
                if ext_rate > 0.50 * acq_rate:
                    ts.phase = "extinction_failed"
                else:
                    ts.phase = PHASE_REST
                    ts.rest_steps_remaining = REST_STEPS
                    ts.probe_approaches.clear()
        elif ts.phase == PHASE_REST:
            ts.rest_steps_remaining -= 1
            if ts.rest_steps_remaining <= 0:
                ts.phase = PHASE_PROBE
                ts.trial = 0
        elif ts.phase == PHASE_PROBE:
            probe_trial = current_trial - ACQ_TRIALS - EXT_TRIALS - (REST_STEPS // cycle + 1)
            if len(ts.probe_approaches) >= PROBE_TRIALS:
                pass  # done

        if ts.phase in (PHASE_REST, "acquisition_failed", "extinction_failed"):
            ts.cs_signal = 0.0
            return 0.0

        in_cs = step_in_cycle >= ITI_DUR
        ts.cs_signal = 1.0 if in_cs else 0.0
        ts.cs_active = in_cs

        if in_cs:
            dist = self._dist_to(OBJ_RED)
            contacted = dist < self.contact_radius
            if contacted:
                if ts.phase == PHASE_ACQUISITION:
                    reward += 10.0
                    self._agent.energy += 10.0
                if not ts.probe_trial_approached:
                    ts.probe_trial_approached = True
            if ts.phase == PHASE_ACQUISITION:
                reward += self.shaping_scale * max(0.0, ts.prev_dist_to_red - dist)
            if self._dist_to(OBJ_BLUE) < self.contact_radius:
                reward -= 5.0
                self._agent.energy -= 5.0

        if step_in_cycle == cycle - 1:
            if ts.phase == PHASE_ACQUISITION:
                ts.acq_approaches.append(ts.probe_trial_approached)
            elif ts.phase in (PHASE_EXTINCTION, PHASE_PROBE):
                ts.probe_approaches.append(ts.probe_trial_approached)
            ts.probe_trial_approached = False

        return reward

    # ----- TC-05: Generalization -----

    def _step_tc05(self) -> float:
        """Train on CS=1.0 (shaped), test on [1.0, 0.8, 0.6, 0.4, 0.2] (probe)."""
        ts = self._ts
        reward = 0.0
        CS_DUR = 30
        ITI_DUR = 60
        ACQ_TRIALS = 30
        TRIALS_PER_LEVEL = 10
        TEST_LEVELS = [1.0, 0.8, 0.6, 0.4, 0.2]
        TOTAL_PROBE = len(TEST_LEVELS) * TRIALS_PER_LEVEL
        cycle = CS_DUR + ITI_DUR

        ts.total_steps += 1
        step_in_cycle = (ts.total_steps - 1) % cycle
        current_trial = (ts.total_steps - 1) // cycle

        if ts.phase == PHASE_ACQUISITION and current_trial >= ACQ_TRIALS:
            ts.phase = PHASE_PROBE
            ts.trial = 0
            # Build randomised probe order
            probe_order = []
            for level in TEST_LEVELS:
                probe_order.extend([level] * TRIALS_PER_LEVEL)
            self._rng.shuffle(probe_order)
            ts._probe_order = probe_order

        if ts.phase == PHASE_PROBE and not hasattr(ts, "_probe_order"):
            ts._probe_order = []

        # Set stimulus level
        if ts.phase == PHASE_ACQUISITION:
            ts.cs_signal = 1.0 if step_in_cycle >= ITI_DUR else 0.0
        elif ts.phase == PHASE_PROBE:
            probe_idx = ts.trial
            if probe_idx < len(getattr(ts, "_probe_order", [])):
                ts.gen_stimulus_level = ts._probe_order[probe_idx]
            ts.cs_signal = ts.gen_stimulus_level if step_in_cycle >= ITI_DUR else 0.0

        in_cs = step_in_cycle >= ITI_DUR
        ts.cs_active = in_cs

        if in_cs:
            dist = self._dist_to(OBJ_RED)
            contacted = dist < self.contact_radius
            if contacted:
                if ts.phase == PHASE_ACQUISITION:
                    reward += 10.0
                    self._agent.energy += 10.0
                if not ts.probe_trial_approached:
                    ts.probe_trial_approached = True
            if ts.phase == PHASE_ACQUISITION:
                reward += self.shaping_scale * max(0.0, ts.prev_dist_to_red - dist)
            if self._dist_to(OBJ_BLUE) < self.contact_radius:
                reward -= 5.0
                self._agent.energy -= 5.0

        if step_in_cycle == cycle - 1:
            if ts.phase == PHASE_ACQUISITION:
                ts.acq_approaches.append(ts.probe_trial_approached)
            elif ts.phase == PHASE_PROBE:
                level = ts.gen_stimulus_level
                if level not in ts.responses_by_strength:
                    ts.responses_by_strength[level] = []
                ts.responses_by_strength[level].append(ts.probe_trial_approached)
                ts.probe_approaches.append(ts.probe_trial_approached)
                ts.trial += 1
            ts.probe_trial_approached = False

        return reward

    # ----- TC-06: Discrimination -----

    def _step_tc06(self) -> float:
        """CS+ (green visible) = approach; CS- (blue visible) = avoid."""
        ts = self._ts
        reward = 0.0
        CS_DUR = 30
        ITI_DUR = 60
        DISC_TRIALS = 60   # 30 CS+ / 30 CS-
        PROBE_TRIALS = 50  # 25 CS+ / 25 CS-
        cycle = CS_DUR + ITI_DUR

        ts.total_steps += 1
        step_in_cycle = (ts.total_steps - 1) % cycle
        current_trial = (ts.total_steps - 1) // cycle

        if ts.phase == PHASE_ACQUISITION and current_trial >= DISC_TRIALS:
            ts.phase = PHASE_PROBE
            ts.trial = 0

        # Determine CS type at trial onset
        if step_in_cycle == 0:
            ts.disc_cs_type = self._rng.choice(["plus", "minus"])

        in_cs = step_in_cycle >= ITI_DUR
        if in_cs:
            # CS signal: both types set obs[17]=1.0; type encoded via object visibility
            ts.cs_signal = 1.0
            # Green active = CS+, Blue active = CS-
            self._objects[OBJ_GREEN].active = ts.disc_cs_type == "plus"
            self._objects[OBJ_BLUE].active = ts.disc_cs_type == "minus"
        else:
            ts.cs_signal = 0.0
            self._objects[OBJ_GREEN].active = True
            self._objects[OBJ_BLUE].active = True
        ts.cs_active = in_cs

        if in_cs:
            dist = self._dist_to(OBJ_RED)
            contacted = dist < self.contact_radius
            if contacted and not ts.probe_trial_approached:
                ts.probe_trial_approached = True

            if ts.phase == PHASE_ACQUISITION:
                if ts.disc_cs_type == "plus":
                    if contacted:
                        reward += 10.0
                        self._agent.energy += 10.0
                    reward += self.shaping_scale * max(0.0, ts.prev_dist_to_red - dist)
                else:  # minus
                    if contacted:
                        reward -= 1.0  # penalty for approaching on CS-
            # Blue contact always penalised
            if self._dist_to(OBJ_BLUE) < self.contact_radius and ts.disc_cs_type != "minus":
                reward -= 5.0
                self._agent.energy -= 5.0

        if step_in_cycle == cycle - 1:
            if ts.phase == PHASE_ACQUISITION:
                ts.acq_approaches.append((ts.disc_cs_type, ts.probe_trial_approached))
            elif ts.phase == PHASE_PROBE:
                ts.probe_approaches.append((ts.disc_cs_type, ts.probe_trial_approached))
            ts.probe_trial_approached = False

        return reward

    # ----- TC-07: Reward Contingency -----

    def _step_tc07(self) -> float:
        """Operant: forward movement → reward proportional to forward velocity."""
        v_f = self._agent.v_forward
        reward = max(0.0, v_f / MAX_FORWARD) * 0.5
        self._agent.energy += reward * 0.1
        ts = self._ts
        ts.total_steps += 1
        ts.cs_signal = 0.0
        return reward

    # ----- TC-08: Partial Reinforcement -----

    def _step_tc08(self) -> float:
        """Operant: 50% Bernoulli reward gating."""
        ts = self._ts
        ts.total_steps += 1
        ts.reward_this_step = self._rng.random() < 0.5
        if ts.reward_this_step:
            v_f = self._agent.v_forward
            reward = max(0.0, v_f / MAX_FORWARD) * 0.5
            self._agent.energy += reward * 0.1
        else:
            reward = 0.0
        ts.cs_signal = 0.0
        return reward

    # ----- TC-09: Shaping -----

    def _step_tc09(self) -> float:
        """Compare shaped vs. unshaped navigation.

        Condition 'shaped': distance shaping + milestone bonuses + contact reward.
        Condition 'unshaped': contact reward only.
        Episodes split 50/50 via ts.condition cycling.
        """
        ts = self._ts
        ts.total_steps += 1
        reward = 0.0
        dist = self._dist_to(OBJ_RED)
        contacted = dist < self.contact_radius

        if contacted:
            reward += 10.0
            self._agent.energy += 10.0
            if ts.condition == "shaped":
                ts.shaped_successes.append(True)
            else:
                ts.unshaped_successes.append(True)
            # Reposition agent to reset
            self._reset_agent()

        if ts.condition == "shaped":
            # Distance shaping
            reward += self.shaping_scale * max(0.0, ts.prev_dist_to_red - dist)
            # Milestone bonuses
            init_dist = math.sqrt(2) * self.arena_size * 0.5  # approx max
            for frac in (0.75, 0.50, 0.25):
                if dist < frac * init_dist and ts.prev_dist_to_red >= frac * init_dist:
                    reward += 1.0

        ts.cs_signal = 0.0
        return reward

    # ----- TC-10: Chaining -----

    def _step_tc10(self) -> float:
        """Sequential navigation: green → blue → red."""
        ts = self._ts
        ts.total_steps += 1
        reward = 0.0

        chain_targets = [OBJ_GREEN, OBJ_BLUE, OBJ_RED]
        target = chain_targets[ts.chain_step]
        dist = self._dist_to(target)
        contacted = dist < self.contact_radius

        if contacted:
            if ts.chain_step == 0:
                ts.chains_attempted += 1
                reward += 2.0
                ts.chain_step = 1
            elif ts.chain_step == 1:
                reward += 2.0
                ts.chain_step = 2
            elif ts.chain_step == 2:
                reward += 20.0
                self._agent.energy += 10.0
                ts.chains_completed += 1
                ts.chain_step = 0
        else:
            # Wrong object during chain → penalty and reset
            for other in chain_targets:
                if other != target and self._dist_to(other) < self.contact_radius:
                    reward -= 1.0
                    ts.chain_step = 0
                    break

        ts.cs_signal = 0.0
        return reward

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build 18-dim observation vector.

        [0-1]  agent (x, y)
        [2]    agent heading
        [3]    energy (normalised)
        [4-5]  agent velocity (v_forward, v_angular normalised)
        [6-7]  direction to red object (unit vector, egocentric)
        [8-10] object 0 features (dx, dy, visible)
        [11-13] object 1 features
        [14-16] object 2 features
        [17]   stimulus signal
        """
        a = self._agent
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0] = a.x / self.arena_size
        obs[1] = a.y / self.arena_size
        obs[2] = a.heading / math.pi
        obs[3] = a.energy / self.max_energy
        obs[4] = a.v_forward / MAX_FORWARD
        obs[5] = a.v_angular / MAX_ANGULAR

        # Direction to red object (egocentric unit vector)
        dx_red = self._objects[OBJ_RED].x - a.x
        dy_red = self._objects[OBJ_RED].y - a.y
        dist_red = math.sqrt(dx_red ** 2 + dy_red ** 2) + 1e-8
        angle_to_red = math.atan2(dy_red, dx_red) - a.heading
        obs[6] = math.cos(angle_to_red)
        obs[7] = math.sin(angle_to_red)

        # Per-object features
        for i, obj in enumerate(self._objects):
            base = 8 + i * 3
            if not obj.active:
                obs[base] = 0.0
                obs[base + 1] = 0.0
                obs[base + 2] = 0.0
                continue
            dx = obj.x - a.x
            dy = obj.y - a.y
            dist = math.sqrt(dx ** 2 + dy ** 2) + 1e-8
            # Check visibility (FOV)
            angle = abs(_wrap_angle(math.atan2(dy, dx) - a.heading))
            visible = float(dist <= FOV_RANGE and angle <= FOV_HALF_ANGLE)
            if visible:
                obs[base] = dx / self.arena_size
                obs[base + 1] = dy / self.arena_size
            else:
                obs[base] = 0.0
                obs[base + 1] = 0.0
            obs[base + 2] = visible

        obs[17] = float(self._ts.cs_signal)
        return obs

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def _get_info(self) -> dict[str, Any]:
        ts = self._ts
        info: dict[str, Any] = {
            "task": self.task,
            "phase": ts.phase,
            "cs_active": ts.cs_active,
            "cs_signal": ts.cs_signal,
            "energy": self._agent.energy,
            "step": self._step_count,
        }

        if self.task == "stimulus_response":
            info["probe_approaches"] = ts.probe_approaches.copy()
            info["acq_approaches"] = ts.acq_approaches.copy()
            info["iti_approaches"] = ts.iti_approaches.copy()
            # score: probe approach rate (0.0 if no probe trials yet)
            n = len(ts.probe_approaches)
            info["score"] = float(sum(ts.probe_approaches) / n) if n > 0 else 0.0
        elif self.task == "temporal_contingency":
            info["probe_trials"] = ts.probe_approaches.copy()
            info["acq_trials"] = ts.acq_approaches.copy()
            # score: fraction of probe trials with approach_time within ±20% of trained delay (30 steps)
            trained = 30
            timed = sum(
                1 for t in ts.probe_approaches
                if isinstance(t, dict) and t.get("approach_time") is not None
                and abs(t["approach_time"] - trained) <= 0.20 * trained
            )
            n = len(ts.probe_approaches)
            info["score"] = float(timed / n) if n > 0 else 0.0
        elif self.task == "extinction":
            info["acq_approaches"] = ts.acq_approaches.copy()
            info["ext_approaches"] = ts.probe_approaches.copy()
            info["acquisition_failed"] = ts.phase == "acquisition_failed"
            # score: 1 - (extinction_rate / acquisition_rate); nan → 0
            last10_acq = ts.acq_approaches[-10:] if len(ts.acq_approaches) >= 10 else ts.acq_approaches
            acq_rate = sum(last10_acq) / max(len(last10_acq), 1)
            n_ext = len(ts.probe_approaches)
            ext_rate = sum(ts.probe_approaches) / n_ext if n_ext > 0 else 0.0
            info["score"] = float(max(0.0, min(1.0, 1.0 - ext_rate / acq_rate))) if acq_rate > 0 else 0.0
        elif self.task == "spontaneous_recovery":
            info["acq_approaches"] = ts.acq_approaches.copy()
            info["ext_approaches"] = ts.probe_approaches.copy()
            info["acquisition_failed"] = ts.phase == "acquisition_failed"
            info["extinction_failed"] = ts.phase == "extinction_failed"
            n = len(ts.probe_approaches)
            info["score"] = float(sum(ts.probe_approaches) / n) if n > 0 else 0.0
        elif self.task == "generalization":
            info["responses_by_strength"] = {k: list(v) for k, v in ts.responses_by_strength.items()}
            # score: Pearson correlation between stimulus level and approach rate
            levels = sorted(ts.responses_by_strength.keys(), reverse=True)
            rates = [float(np.mean(ts.responses_by_strength[lv])) if ts.responses_by_strength[lv] else 0.0 for lv in levels]
            if len(levels) >= 3 and any(rates):
                corr = float(np.corrcoef(levels, rates)[0, 1])
                info["score"] = max(0.0, corr) if not np.isnan(corr) else 0.0
            else:
                info["score"] = 0.0
        elif self.task == "discrimination":
            cs_plus = [(app) for cs, app in ts.probe_approaches if cs == "plus"]
            cs_minus = [(app) for cs, app in ts.probe_approaches if cs == "minus"]
            info["cs_plus_approaches"] = cs_plus
            info["cs_minus_approaches"] = cs_minus
            plus_rate = float(sum(cs_plus) / len(cs_plus)) if cs_plus else 0.0
            minus_rate = float(sum(cs_minus) / len(cs_minus)) if cs_minus else 0.0
            info["score"] = float(max(0.0, min(1.0, plus_rate - minus_rate)))
        elif self.task == "reward_contingency":
            info["total_steps"] = ts.total_steps
            info["v_forward"] = self._agent.v_forward
            # score: normalised forward velocity [0, 1]
            info["score"] = float(self._agent.v_forward / MAX_FORWARD)
        elif self.task == "partial_reinforcement":
            info["total_steps"] = ts.total_steps
            info["reward_this_step"] = ts.reward_this_step
            info["score"] = float(self._agent.v_forward / MAX_FORWARD)
        elif self.task == "shaping":
            info["shaped_successes"] = ts.shaped_successes.copy()
            info["unshaped_successes"] = ts.unshaped_successes.copy()
            info["condition"] = ts.condition
            n = len(ts.shaped_successes)
            info["score"] = float(sum(ts.shaped_successes) / n) if n > 0 else 0.0
        elif self.task == "chaining":
            info["chains_completed"] = ts.chains_completed
            info["chains_attempted"] = ts.chains_attempted
            info["chain_step"] = ts.chain_step
            info["score"] = float(ts.chains_completed / ts.chains_attempted) if ts.chains_attempted > 0 else 0.0

        return info

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _dist_to(self, obj_idx: int) -> float:
        obj = self._objects[obj_idx]
        dx = obj.x - self._agent.x
        dy = obj.y - self._agent.y
        return math.sqrt(dx ** 2 + dy ** 2)

    def _render_frame(self) -> np.ndarray:
        """Simple 2D top-down render as RGB array (240x240)."""
        size = 240
        scale = size / self.arena_size
        frame = np.full((size, size, 3), 50, dtype=np.uint8)

        colours = {
            OBJ_RED: (220, 50, 50),
            OBJ_BLUE: (50, 50, 220),
            OBJ_GREEN: (50, 200, 50),
        }
        for i, obj in enumerate(self._objects):
            if not obj.active:
                continue
            cx = int(obj.x * scale)
            cy = int((self.arena_size - obj.y) * scale)
            r = max(2, int(self.contact_radius * scale))
            _draw_circle(frame, cx, cy, r, colours[i])

        # Agent
        ax = int(self._agent.x * scale)
        ay = int((self.arena_size - self._agent.y) * scale)
        _draw_circle(frame, ax, ay, max(2, int(AGENT_RADIUS * scale)), (255, 255, 0))

        return frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def _draw_circle(img: np.ndarray, cx: int, cy: int, r: int, colour: tuple):
    h, w = img.shape[:2]
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                px, py = cx + dx, cy + dy
                if 0 <= px < w and 0 <= py < h:
                    img[py, px] = colour


# ---------------------------------------------------------------------------
# Gymnasium registration
# ---------------------------------------------------------------------------

def _make_pavlovian(**kwargs) -> PavlovianEnv:
    return PavlovianEnv(**kwargs)
