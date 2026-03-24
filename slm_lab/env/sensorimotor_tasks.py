"""Per-task scene definitions and reward logic for the sensorimotor stage (TC-11 to TC-24).

Each task class owns:
  - scene_objects(): returns list of object IDs needed in this task's scene
  - reset(model, data, rng): places objects, returns initial task state dict
  - step(model, data, state, info): computes reward and updates task state
  - score(state): returns float score in [0, 1]

The parent SLMSensorimotor env calls these hooks each step/reset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# Constants (shared with sensorimotor.py)
# ---------------------------------------------------------------------------

TABLE_CENTER = np.array([2.5, 2.5, 0.75])   # table surface x, y, z
TABLE_HEIGHT = 0.75                           # z of table top surface
OBJ_Z = 0.775                                # default z for objects on table (half-height 0.025)
NEAR_X = (2.0, 2.4)
NEAR_Y = (2.2, 2.8)
MID_X = (2.4, 2.7)
MID_Y = (2.2, 2.8)
FAR_X = (2.7, 3.0)
FAR_Y = (2.2, 2.8)

# Workspace reachable by arm (approx hemisphere, shoulder at (1.5, 2.5, 1.10), radius 0.80)
REACH_RADIUS = 0.60  # safe inner radius with margin

# Reflex tracking constants for TC-11
VISUAL_TRACK_TOL_DEG = 15.0
TACTILE_CLOSE_THRESH = 0.02  # m — gripper gap when "closed"
PROPRIO_RETURN_TOL = 0.10    # rad


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _sample_pos(rng: np.random.Generator, x_range: tuple, y_range: tuple, z: float = OBJ_Z) -> np.ndarray:
    x = rng.uniform(*x_range)
    y = rng.uniform(*y_range)
    return np.array([x, y, z])


def _no_overlap(positions: list[np.ndarray], new_pos: np.ndarray, min_dist: float = 0.08) -> bool:
    for p in positions:
        if np.linalg.norm(p[:2] - new_pos[:2]) < min_dist:
            return False
    return True


def _sample_no_overlap(rng: np.random.Generator, x_range, y_range, existing: list[np.ndarray], z=OBJ_Z, max_tries=50) -> np.ndarray:
    for _ in range(max_tries):
        pos = _sample_pos(rng, x_range, y_range, z)
        if _no_overlap(existing, pos):
            return pos
    return _sample_pos(rng, x_range, y_range, z)


def _get_body_xpos(data: mujoco.MjData, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[body_id].copy()


def _set_body_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str, pos: np.ndarray):
    """Set a free-joint body position via qpos."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    jnt_adr = model.body_jntadr[body_id]
    if jnt_adr < 0:
        return  # fixed body — skip
    qpos_adr = model.jnt_qposadr[jnt_adr]
    data.qpos[qpos_adr:qpos_adr + 3] = pos
    # zero velocity
    jnt_dof_adr = model.jnt_dofadr[jnt_adr]
    data.qvel[jnt_dof_adr:jnt_dof_adr + 6] = 0.0


def _get_ee_pos(data: mujoco.MjData) -> np.ndarray:
    """End-effector position from wrist_roll link (approximate)."""
    return _get_body_xpos(data, "wrist_roll")


def _gripper_gap(data: mujoco.MjData, model: mujoco.MjModel) -> float:
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint")
    if jnt_id < 0:
        return 0.04
    qpos_adr = model.jnt_qposadr[jnt_id]
    return float(data.qpos[qpos_adr]) * 2.0  # symmetric: 2 * half-opening


def _is_grasped(data: mujoco.MjData, model: mujoco.MjModel) -> bool:
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "left_contact")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "right_contact")
    left = data.sensordata[left_id] > 0.5 if left_id >= 0 else False
    right = data.sensordata[right_id] > 0.5 if right_id >= 0 else False
    gap = _gripper_gap(data, model)
    return bool(left and right and gap < TACTILE_CLOSE_THRESH)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# Task protocol
# ---------------------------------------------------------------------------

class SensorimotorTask(Protocol):
    task_id: str

    def scene_objects(self) -> list[str]: ...
    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> dict: ...
    def step(self, model: mujoco.MjModel, data: mujoco.MjData, state: dict, info: dict) -> float: ...
    def score(self, state: dict) -> float: ...


# ---------------------------------------------------------------------------
# TC-11: Reflex Validation
# ---------------------------------------------------------------------------

class TC11ReflexValidation:
    task_id = "TC-11"
    STEPS_PER_TRIAL = 50
    ITI_STEPS = 25
    TRIALS_PER_TYPE = 20
    STIM_TYPES = ("visual", "tactile", "proprioceptive")

    def scene_objects(self) -> list[str]:
        return ["sphere_red"]

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> dict:
        # Sphere starts at edge for visual stimulus
        sphere_pos = np.array([1.8, 2.5, OBJ_Z + 0.005])
        _set_body_pos(model, data, "sphere_red", sphere_pos)
        return {
            "step": 0,
            "stim_type_idx": 0,          # cycles through STIM_TYPES
            "stim_phase": "iti",          # "iti" or "active"
            "stim_step": 0,
            "visual_trials": [],
            "tactile_trials": [],
            "proprio_trials": [],
            "home_qpos": data.qpos[:9].copy(),  # 7 joints + gripper + head
            "sphere_x": sphere_pos[0],
            "perturb_applied": False,
            "perturb_target_qpos": None,
            "trial_result": None,
            "rng": rng,
        }

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, state: dict, info: dict) -> float:
        state["step"] += 1
        state["stim_step"] += 1
        stim_type = self.STIM_TYPES[state["stim_type_idx"] % len(self.STIM_TYPES)]
        cycle = self.STEPS_PER_TRIAL + self.ITI_STEPS
        in_iti = (state["stim_step"] % cycle) < self.ITI_STEPS
        step_in_trial = (state["stim_step"] % cycle) - self.ITI_STEPS

        if in_iti:
            state["stim_phase"] = "iti"
            return 0.0

        state["stim_phase"] = "active"

        # Visual: move sphere across field, check head pan tracks
        if stim_type == "visual":
            # Advance sphere position
            sphere_x = state["sphere_x"] + 0.006  # 0.3 m/s * 0.04 s * target rate
            state["sphere_x"] = sphere_x
            sphere_pos = np.array([sphere_x, 2.5, OBJ_Z + 0.005])
            _set_body_pos(model, data, "sphere_red", sphere_pos)

            if step_in_trial == 24:  # midpoint measurement
                sphere_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "sphere_red")
                head_pan_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_pan")
                sphere_pos_now = data.xpos[sphere_body_id]
                head_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head_tilt")
                head_pos = data.xpos[head_body_id] if head_body_id >= 0 else np.array([1.5, 2.5, 1.4])
                dx = sphere_pos_now[0] - head_pos[0]
                dy = sphere_pos_now[1] - head_pos[1]
                target_pan = math.atan2(dy, dx)
                if head_pan_id >= 0:
                    qpos_adr = model.jnt_qposadr[head_pan_id]
                    actual_pan = data.qpos[qpos_adr]
                    pan_err_deg = abs(math.degrees(actual_pan - target_pan))
                    state["trial_result"] = pan_err_deg < VISUAL_TRACK_TOL_DEG

        # Tactile: cube teleported into gripper, check close reflex
        elif stim_type == "tactile":
            if step_in_trial == 0:
                # Teleport sphere to gripper
                ee = _get_ee_pos(data)
                _set_body_pos(model, data, "sphere_red", ee)
            if step_in_trial == 10:
                gap = _gripper_gap(data, model)
                state["trial_result"] = gap < TACTILE_CLOSE_THRESH

        # Proprioceptive: external perturbation, check return
        elif stim_type == "proprioceptive":
            if step_in_trial == 0:
                # Apply 10 N impulse by temporarily offsetting joint targets
                state["perturb_target_qpos"] = state["home_qpos"].copy()
                state["perturb_applied"] = True
                perturb = state["rng"].uniform(-0.5, 0.5, 7)
                state["perturb_target_qpos"][:7] += perturb

            if step_in_trial == 50:  # check within 50 steps
                home = state["home_qpos"][:7]
                current = data.qpos[:7].copy()
                max_err = np.max(np.abs(current - home))
                state["trial_result"] = max_err < PROPRIO_RETURN_TOL

        # Record trial result at end of trial
        if step_in_trial == self.STEPS_PER_TRIAL - 1:
            result = state.get("trial_result", False)
            if stim_type == "visual":
                state["visual_trials"].append(bool(result))
            elif stim_type == "tactile":
                state["tactile_trials"].append(bool(result))
            elif stim_type == "proprioceptive":
                state["proprio_trials"].append(bool(result))
            state["trial_result"] = None
            # Advance stim type after TRIALS_PER_TYPE
            trial_count = len(state["visual_trials"]) + len(state["tactile_trials"]) + len(state["proprio_trials"])
            state["stim_type_idx"] = trial_count // self.TRIALS_PER_TYPE

        info["tc11"] = {
            "visual_trials": state["visual_trials"],
            "tactile_trials": state["tactile_trials"],
            "proprio_trials": state["proprio_trials"],
            "stim_type": stim_type,
            "stim_phase": state["stim_phase"],
        }
        return 0.0

    def score(self, state: dict) -> float:
        v = state["visual_trials"]
        t = state["tactile_trials"]
        p = state["proprio_trials"]
        v_rate = sum(v) / len(v) if v else 0.0
        t_rate = sum(t) / len(t) if t else 0.0
        p_rate = sum(p) / len(p) if p else 0.0
        return 0.33 * v_rate + 0.33 * t_rate + 0.34 * p_rate


# ---------------------------------------------------------------------------
# TC-12: Action-Effect Discovery
# ---------------------------------------------------------------------------

class TC12ActionEffectDiscovery:
    task_id = "TC-12"
    BASELINE_EPS = 20
    CONTINGENCY_EPS = 40
    STEPS_PER_EP = 200
    HIGH_EFFECT_THRESH = 0.05  # m — EE displacement threshold
    REPEAT_TOL = 0.10          # rad — joint-space repetition radius

    def scene_objects(self) -> list[str]:
        return []

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> dict:
        return {
            "episode": 0,
            "phase": "baseline",  # "baseline" | "contingency" | "extinction"
            "ee_prev": _get_ee_pos(data).copy(),
            "baseline_displacements": [],
            "contingency_displacements": [],
            "high_effect_configs": [],  # list of qpos[:7] snapshots
            "repetition_flags": [],
        }

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, state: dict, info: dict) -> float:
        ee_now = _get_ee_pos(data)
        disp = _dist(ee_now, state["ee_prev"])
        state["ee_prev"] = ee_now.copy()

        reward = 0.0
        phase = state["phase"]

        if phase == "baseline":
            state["baseline_displacements"].append(disp)
        elif phase == "contingency":
            state["contingency_displacements"].append(disp)
            # Intrinsic reward: positive for large displacements
            reward = float(min(1.0, disp / self.HIGH_EFFECT_THRESH)) * 0.01

            if disp > self.HIGH_EFFECT_THRESH:
                state["high_effect_configs"].append(data.qpos[:7].copy())

            # Check repetition
            if state["high_effect_configs"]:
                qpos = data.qpos[:7].copy()
                is_repeat = any(
                    np.max(np.abs(qpos - ref)) < self.REPEAT_TOL
                    for ref in state["high_effect_configs"][-20:]
                )
                state["repetition_flags"].append(is_repeat)

        info["tc12"] = {
            "phase": phase,
            "disp": disp,
            "high_effect_count": len(state["high_effect_configs"]),
        }
        return reward

    def score(self, state: dict) -> float:
        b = state["baseline_displacements"]
        c = state["contingency_displacements"]
        r = state["repetition_flags"]
        baseline_mean = float(np.mean(b)) if b else 0.0
        contingency_mean = float(np.mean(c)) if c else 0.0
        if baseline_mean <= 0:
            movement_score = 1.0 if contingency_mean > 0 else 0.0
        else:
            movement_score = min(1.0, contingency_mean / (2.0 * baseline_mean))
        repetition_rate = float(sum(r) / len(r)) if r else 0.0
        return 0.5 * movement_score + 0.5 * repetition_rate


# ---------------------------------------------------------------------------
# TC-13: Motor Coordination (Reaching)
# ---------------------------------------------------------------------------

class TC13Reaching:
    task_id = "TC-13"
    REACH_THRESH = 0.03  # m
    # Curriculum: expand target radius as episodes accumulate
    EASY_RADIUS = 0.15   # m — initial radius around a near-home point
    FULL_RADIUS = 0.50   # m — full workspace radius
    CURRICULUM_EPISODES = 200  # episodes to reach full difficulty

    def scene_objects(self) -> list[str]:
        return ["target_disk"]

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> dict:
        # EE home projects onto table at approximately (1.55, 2.5, TABLE_HEIGHT)
        # but arm can reach forward — use a point slightly in front of home
        anchor = np.array([1.8, 2.5, TABLE_HEIGHT + 0.001])

        # Curriculum: expand sampling radius over episodes
        n_eps = len(getattr(self, '_successes_acc', []))
        frac = min(1.0, n_eps / self.CURRICULUM_EPISODES)
        radius = self.EASY_RADIUS + (self.FULL_RADIUS - self.EASY_RADIUS) * frac

        # Sample within radius on table surface
        angle = rng.uniform(-np.pi, np.pi)
        r = radius * np.sqrt(rng.uniform())
        pos = anchor + np.array([r * np.cos(angle), r * np.sin(angle), 0.0])
        # Clamp to reachable table region
        pos[0] = np.clip(pos[0], 1.6, 2.6)
        pos[1] = np.clip(pos[1], 2.1, 2.9)
        pos[2] = TABLE_HEIGHT + 0.001

        _set_body_pos(model, data, "target_disk", pos)
        return {
            "target_pos": pos.copy(),
            "successes": [],
            "completion_times": [],
            "reached_this_ep": False,
            "ep_step": 0,
        }

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, state: dict, info: dict) -> float:
        state["ep_step"] += 1
        ee = _get_ee_pos(data)
        target = state["target_pos"]
        dist = _dist(ee, target)
        reward = -dist  # dense distance penalty

        if dist < self.REACH_THRESH and not state["reached_this_ep"]:
            state["reached_this_ep"] = True
            reward += 10.0

        info["tc13"] = {
            "dist_to_target": dist,
            "reached": state["reached_this_ep"],
        }
        return float(reward)

    def on_episode_end(self, state: dict):
        state["successes"].append(state["reached_this_ep"])
        if state["reached_this_ep"]:
            state["completion_times"].append(state["ep_step"])
        # Track cumulative episode count for curriculum
        if not hasattr(self, '_successes_acc'):
            self._successes_acc = []
        self._successes_acc.append(state["reached_this_ep"])
        state["reached_this_ep"] = False
        state["ep_step"] = 0

    def score(self, state: dict) -> float:
        s = state["successes"]
        t = state["completion_times"]
        if not s:
            return 0.0
        success_rate = sum(s) / len(s)
        if not t:
            return 0.7 * success_rate
        efficiency = max(0.0, 1.0 - (sum(t) / len(t)) / 200)
        return 0.7 * success_rate + 0.3 * efficiency


# ---------------------------------------------------------------------------
# TC-14: Object Interaction
# ---------------------------------------------------------------------------

class TC14ObjectInteraction:
    task_id = "TC-14"
    CONTACT_THRESH = 0.04  # m — close enough to be in contact

    def scene_objects(self) -> list[str]:
        return ["cube_red"]

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> dict:
        pos = _sample_pos(rng, NEAR_X, NEAR_Y)
        _set_body_pos(model, data, "cube_red", pos)
        object_absent = rng.random() < 0.5  # interleaved absent-control trials
        if object_absent:
            _set_body_pos(model, data, "cube_red", np.array([5.0, 5.0, 0.0]))  # hide off-table
        return {
            "object_absent": object_absent,
            "cube_pos_init": pos.copy(),
            "contact_steps_present": 0,
            "contact_steps_absent": 0,
            "total_steps": 0,
            "action_modes": {"push": 0, "lift": 0, "rotate": 0},
            "prev_cube_pos": pos.copy(),
            "prev_cube_quat": np.zeros(4),
        }

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, state: dict, info: dict) -> float:
        state["total_steps"] += 1
        ee = _get_ee_pos(data)
        cube_pos = _get_body_xpos(data, "cube_red")
        dist = _dist(ee, cube_pos)
        in_contact = dist < self.CONTACT_THRESH

        reward = 0.0
        if not state["object_absent"] and in_contact:
            state["contact_steps_present"] += 1
            # Classify mode
            dz = cube_pos[2] - state["prev_cube_pos"][2]
            dxy = float(np.linalg.norm(cube_pos[:2] - state["prev_cube_pos"][:2]))
            if dz > 0.02:
                state["action_modes"]["lift"] += 1
            elif dxy > 0.01:
                state["action_modes"]["push"] += 1
            # rotation not easily computable without quat diff here — use lift/push proxy

            # Intrinsic reward
            cube_change = float(np.linalg.norm(cube_pos - state["prev_cube_pos"]))
            reward = min(1.0, cube_change / 0.05) * 0.01
        elif state["object_absent"] and dist < self.CONTACT_THRESH:
            state["contact_steps_absent"] += 1

        state["prev_cube_pos"] = cube_pos.copy()

        info["tc14"] = {
            "in_contact": in_contact,
            "object_absent": state["object_absent"],
            "action_modes": state["action_modes"],
        }
        return reward

    def score(self, state: dict) -> float:
        total = state["total_steps"]
        if total == 0:
            return 0.0
        rate_present = state["contact_steps_present"] / total
        rate_absent = state["contact_steps_absent"] / total
        if rate_absent <= 0:
            pref = 1.0 if rate_present > 0 else 0.0
        else:
            pref = min(1.0, rate_present / (2.0 * rate_absent))

        modes = state["action_modes"]
        total_actions = sum(modes.values())
        if total_actions == 0:
            variety = 0.0
        else:
            probs = [c / total_actions for c in modes.values() if c > 0]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            variety = entropy / math.log2(len(modes))
        return 0.5 * pref + 0.5 * variety


# ---------------------------------------------------------------------------
# TC-15: Means-End Precursor (String Pull)
# ---------------------------------------------------------------------------

class TC15MeansEnd:
    task_id = "TC-15"
    GRASP_THRESH = 0.02       # m — gripper gap when holding
    PULL_SUCCESS_DIST = 0.10  # m — target displacement for success

    def scene_objects(self) -> list[str]:
        return ["cube_red", "string"]

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> dict:
        # Cube at far table; string bridges near edge
        cube_y = rng.uniform(2.3, 2.7)
        cube_pos = np.array([2.85, cube_y, OBJ_Z])
        _set_body_pos(model, data, "cube_red", cube_pos)
        string_pos = np.array([2.35, cube_y, OBJ_Z])
        _set_body_pos(model, data, "string", string_pos)
        return {
            "cube_init_pos": cube_pos.copy(),
            "first_success_ep": None,
            "trials_after_first": [],
            "episode": 0,
            "success_this_ep": False,
            "cube_pulled": False,
        }

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, state: dict, info: dict) -> float:
        reward = 0.0
        cube_pos = _get_body_xpos(data, "cube_red")
        string_pos = _get_body_xpos(data, "string")
        ee = _get_ee_pos(data)

        # Check if agent is pulling string
        string_held = _dist(ee, string_pos) < 0.05 and _is_grasped(data, model)
        cube_displaced = _dist(cube_pos[:2], state["cube_init_pos"][:2]) > self.PULL_SUCCESS_DIST

        if cube_displaced and not state["success_this_ep"]:
            state["success_this_ep"] = True
            reward += 10.0

        # Shaping: reward for holding string
        if string_held:
            reward += 0.1

        info["tc15"] = {
            "string_held": string_held,
            "cube_displaced": float(np.linalg.norm(cube_pos[:2] - state["cube_init_pos"][:2])),
        }
        return reward

    def score(self, state: dict) -> float:
        if state["first_success_ep"] is None:
            return 0.0
        after = state["trials_after_first"][:10]
        if not after:
            return 0.0
        return sum(after) / len(after)


# ---------------------------------------------------------------------------
# TC-16: Object Permanence (A-not-B)
# ---------------------------------------------------------------------------

class TC16ObjectPermanence:
    task_id = "TC-16"
    SEARCH_THRESH = 0.05   # m — reach within this of screen base
    HIDING_STEPS = 10
    DELAY_STEPS = 5
    SEARCH_STEPS = 85

    def scene_objects(self) -> list[str]:
        return ["sphere_red", "screen_A", "screen_B"]

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> dict:
        sphere_pos = np.array([2.25, 2.3, OBJ_Z + 0.005])  # start visible
        _set_body_pos(model, data, "sphere_red", sphere_pos)
        return {
            "phase": "a_trials",
            "trial": 0,
            "trial_step": 0,
            "hiding_loc": "A",
            "a_trial_results": [],
            "b_trial_results": [],
            "search_recorded": False,
            "acq_gate_passed": False,
        }

    def _screen_pos(self, model: mujoco.MjModel, loc: str) -> np.ndarray:
        name = "screen_A" if loc == "A" else "screen_B"
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return model.body_pos[bid].copy()

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, state: dict, info: dict) -> float:
        state["trial_step"] += 1
        step = state["trial_step"]
        reward = 0.0
        cycle = self.HIDING_STEPS + self.DELAY_STEPS + self.SEARCH_STEPS

        # Hiding phase
        if step <= self.HIDING_STEPS:
            target_screen = self._screen_pos(model, state["hiding_loc"])
            sphere_pos = target_screen + np.array([0.0, -0.05, 0.0])
            _set_body_pos(model, data, "sphere_red", sphere_pos)

        # Search phase
        elif step > self.HIDING_STEPS + self.DELAY_STEPS:
            if not state["search_recorded"]:
                ee = _get_ee_pos(data)
                screen_A = self._screen_pos(model, "A")
                screen_B = self._screen_pos(model, "B")
                dist_A = _dist(ee, screen_A)
                dist_B = _dist(ee, screen_B)
                if dist_A < self.SEARCH_THRESH:
                    result = "A"
                    state["search_recorded"] = True
                elif dist_B < self.SEARCH_THRESH:
                    result = "B"
                    state["search_recorded"] = True
                else:
                    result = None

                if result is not None:
                    if state["phase"] == "a_trials":
                        state["a_trial_results"].append(result)
                        if result == "A":
                            reward += 5.0
                    else:
                        state["b_trial_results"].append(result)

        # Trial end
        if step >= cycle:
            if not state["search_recorded"]:
                result_logged = "none"
                if state["phase"] == "a_trials":
                    state["a_trial_results"].append(result_logged)
                else:
                    state["b_trial_results"].append(result_logged)
            state["trial_step"] = 0
            state["trial"] += 1
            state["search_recorded"] = False

            # Check acquisition gate after 5 A-trials
            if state["phase"] == "a_trials" and len(state["a_trial_results"]) >= 5:
                correct = sum(1 for r in state["a_trial_results"][-5:] if r == "A")
                if correct >= 4:
                    state["acq_gate_passed"] = True
                    state["phase"] = "b_trials"
                    state["hiding_loc"] = "B"

        info["tc16"] = {
            "phase": state["phase"],
            "hiding_loc": state["hiding_loc"],
            "a_results": state["a_trial_results"],
            "b_results": state["b_trial_results"],
            "acq_gate_passed": state["acq_gate_passed"],
        }
        return reward

    def score(self, state: dict) -> float:
        """Return stage-5 score (correct B searches)."""
        b = state["b_trial_results"]
        if not b:
            return 0.0
        return sum(1 for r in b if r == "B") / len(b)

    def score_stage4(self, state: dict) -> float:
        """Stage-4 score: A-not-B error expected."""
        b = state["b_trial_results"][:5]
        if not b:
            return 0.0
        return sum(1 for r in b if r == "A") / len(b)


# ---------------------------------------------------------------------------
# TC-17: Intentional Means-End (Obstacle Removal)
# ---------------------------------------------------------------------------

class TC17ObstacleRemoval:
    task_id = "TC-17"
    BARRIER_MOVE_THRESH = 0.10  # m

    def scene_objects(self) -> list[str]:
        return ["cube_red", "barrier"]

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> dict:
        placed: list[np.ndarray] = []
        barrier_pos = _sample_no_overlap(rng, MID_X, NEAR_Y, placed)
        barrier_pos[0] += rng.uniform(-0.05, 0.05)
        placed.append(barrier_pos)
        _set_body_pos(model, data, "barrier", barrier_pos)
        # Place cube just behind barrier
        cube_pos = barrier_pos.copy()
        cube_pos[0] += 0.10
        _set_body_pos(model, data, "cube_red", cube_pos)
        return {
            "barrier_init": barrier_pos.copy(),
            "trials": [],
            "barrier_removed": False,
            "target_grasped": False,
            "removal_step": None,
            "ep_step": 0,
            "correct_order": False,
        }

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, state: dict, info: dict) -> float:
        state["ep_step"] += 1
        reward = 0.0
        barrier_pos = _get_body_xpos(data, "barrier")
        cube_pos = _get_body_xpos(data, "cube_red")
        ee = _get_ee_pos(data)

        barrier_disp = _dist(barrier_pos, state["barrier_init"])
        if not state["barrier_removed"] and barrier_disp > self.BARRIER_MOVE_THRESH:
            state["barrier_removed"] = True
            state["removal_step"] = state["ep_step"]
            reward += 5.0

        # Barrier contact shaping
        if _dist(ee, barrier_pos) < 0.06:
            reward += 0.2

        if _is_grasped(data, model) and _dist(ee, cube_pos) < 0.06:
            if not state["target_grasped"]:
                state["target_grasped"] = True
                state["correct_order"] = state["barrier_removed"]
                reward += 10.0

        info["tc17"] = {
            "barrier_removed": state["barrier_removed"],
            "target_grasped": state["target_grasped"],
            "correct_order": state["correct_order"],
        }
        return reward

    def score(self, state: dict) -> float:
        trials = state["trials"]
        if not trials:
            # Single-episode score
            n = 1
            completion_rate = float(state["target_grasped"])
            order_rate = float(state["correct_order"])
            latency = state["ep_step"] if state["target_grasped"] else None
        else:
            n = len(trials)
            completion_rate = sum(1 for t in trials if t["target_grasped"]) / n
            order_rate = sum(1 for t in trials if t["correct_order"]) / n
            valid_latencies = [t["latency"] for t in trials if t.get("latency")]
            latency = sum(valid_latencies) / len(valid_latencies) if valid_latencies else None

        efficiency = max(0.0, 1.0 - latency / 300) if latency is not None else 0.0
        return 0.5 * completion_rate + 0.3 * order_rate + 0.2 * efficiency


# ---------------------------------------------------------------------------
# TC-18: Tool Use (Pull Cloth)
# ---------------------------------------------------------------------------

class TC18ClothPull:
    task_id = "TC-18"
    CLOTH_GRASP_THRESH = 0.05   # m — distance to cloth edge
    PULL_SUCCESS_Z_DELTA = 0.03  # unused — use XY displacement

    def scene_objects(self) -> list[str]:
        return ["cube_blue", "cloth"]

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> dict:
        # Cloth edge near agent; cube on far portion
        cube_pos = np.array([2.7, rng.uniform(2.3, 2.7), OBJ_Z + 0.01])
        _set_body_pos(model, data, "cube_blue", cube_pos)
        return {
            "cube_init_pos": cube_pos.copy(),
            "cloth_held": False,
            "success_this_ep": False,
            "standard_trials": [],
            "transfer_trials": [],
            "is_transfer": False,
            "ep_step": 0,
        }

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, state: dict, info: dict) -> float:
        state["ep_step"] += 1
        reward = 0.0
        ee = _get_ee_pos(data)
        cube_pos = _get_body_xpos(data, "cube_blue")

        # Approximate cloth edge position (near edge of cloth, x ~ 2.2, y ~ 2.5)
        cloth_edge_approx = np.array([2.2, 2.5, OBJ_Z])
        cloth_near = _dist(ee, cloth_edge_approx) < self.CLOTH_GRASP_THRESH and _is_grasped(data, model)

        if cloth_near and not state["cloth_held"]:
            state["cloth_held"] = True
            reward += 3.0

        if state["cloth_held"]:
            # Pull reward: EE moving toward agent (decreasing x)
            reward += max(0.0, 0.01 * (3.0 - ee[0]))

        # Success: cube pulled within reach
        cube_disp = _dist(cube_pos[:2], state["cube_init_pos"][:2])
        if cube_disp > 0.20 and not state["success_this_ep"]:
            state["success_this_ep"] = True
            reward += 10.0

        info["tc18"] = {
            "cloth_held": state["cloth_held"],
            "cube_disp": cube_disp,
            "success": state["success_this_ep"],
        }
        return reward

    def score(self, state: dict) -> float:
        s = state["standard_trials"]
        t = state["transfer_trials"]
        standard_rate = sum(s) / len(s) if s else float(state["success_this_ep"])
        transfer_rate = sum(t) / len(t) if t else 0.0
        return 0.6 * standard_rate + 0.4 * transfer_rate


# ---------------------------------------------------------------------------
# TC-19: Active Experimentation
# ---------------------------------------------------------------------------

class TC19ActiveExperimentation:
    task_id = "TC-19"
    LIFT_Z = TABLE_HEIGHT + 0.10  # z threshold for "lifted"
    DROP_Z = TABLE_HEIGHT + 0.05  # peak before drop

    def scene_objects(self) -> list[str]:
        return ["cube_red", "box_open"]

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> dict:
        placed: list[np.ndarray] = []
        cube_pos = _sample_no_overlap(rng, NEAR_X, NEAR_Y, placed)
        placed.append(cube_pos)
        box_pos = _sample_no_overlap(rng, MID_X, NEAR_Y, placed)
        _set_body_pos(model, data, "cube_red", cube_pos)
        _set_body_pos(model, data, "box_open", box_pos)
        return {
            "action_modes": {"push": 0, "lift": 0, "rotate": 0, "drop": 0, "place_in_box": 0},
            "cube_prev_pos": cube_pos.copy(),
            "cube_prev_quat": np.array([1, 0, 0, 0]),
            "was_lifted": False,
            "peak_z": cube_pos[2],
            "box_init_pos": box_pos.copy(),
        }

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, state: dict, info: dict) -> float:
        reward = 0.0
        cube_pos = _get_body_xpos(data, "cube_red")
        box_pos = _get_body_xpos(data, "box_open")
        ee = _get_ee_pos(data)
        in_contact = _dist(ee, cube_pos) < 0.06

        if in_contact:
            dz = cube_pos[2] - state["cube_prev_pos"][2]
            dxy = float(np.linalg.norm(cube_pos[:2] - state["cube_prev_pos"][:2]))

            if cube_pos[2] > self.LIFT_Z:
                state["was_lifted"] = True
                state["peak_z"] = max(state["peak_z"], cube_pos[2])
                state["action_modes"]["lift"] += 1
            elif dxy > 0.01 and not state["was_lifted"]:
                state["action_modes"]["push"] += 1

            # Drop detection
            if state["was_lifted"] and cube_pos[2] < TABLE_HEIGHT + 0.03:
                state["action_modes"]["drop"] += 1
                state["was_lifted"] = False

        # Place in box detection
        box_inner = 0.07
        in_box = (
            abs(cube_pos[0] - box_pos[0]) < box_inner and
            abs(cube_pos[1] - box_pos[1]) < box_inner and
            cube_pos[2] < box_pos[2] + 0.12
        )
        if in_box:
            state["action_modes"]["place_in_box"] += 1

        # Intrinsic reward
        change = float(np.linalg.norm(cube_pos - state["cube_prev_pos"]))
        reward = min(1.0, change / 0.05) * 0.01

        state["cube_prev_pos"] = cube_pos.copy()

        info["tc19"] = {"action_modes": dict(state["action_modes"])}
        return reward

    def score(self, state: dict) -> float:
        modes = state["action_modes"]
        coverage = sum(1 for c in modes.values() if c > 0) / len(modes)
        total = sum(modes.values())
        if total == 0:
            return 0.0
        probs = [c / total for c in modes.values() if c > 0]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        max_entropy = math.log2(len(modes))
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0.0
        return 0.5 * coverage + 0.5 * entropy_ratio


# ---------------------------------------------------------------------------
# TC-20: Novel Tool Use
# ---------------------------------------------------------------------------

class TC20ToolUse:
    task_id = "TC-20"
    RETRIEVAL_THRESH = 0.15  # m — sphere must move this far

    def scene_objects(self) -> list[str]:
        return ["sphere_red", "stick", "rake", "spoon"]

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> dict:
        sphere_pos = np.array([rng.uniform(2.75, 2.95), rng.uniform(2.3, 2.7), OBJ_Z + 0.005])
        _set_body_pos(model, data, "sphere_red", sphere_pos)
        placed = [sphere_pos]
        stick_pos = _sample_no_overlap(rng, NEAR_X, NEAR_Y, placed)
        placed.append(stick_pos)
        rake_pos = _sample_no_overlap(rng, NEAR_X, NEAR_Y, placed)
        placed.append(rake_pos)
        spoon_pos = _sample_no_overlap(rng, NEAR_X, NEAR_Y, placed)
        _set_body_pos(model, data, "stick", stick_pos)
        _set_body_pos(model, data, "rake", rake_pos)
        _set_body_pos(model, data, "spoon", spoon_pos)
        use_transfer = rng.random() < 0.33  # 1/3 episodes use L-bar
        return {
            "sphere_init": sphere_pos.copy(),
            "success_this_ep": False,
            "first_tool_touched": None,
            "known_trials": [],
            "known_selections": [],
            "transfer_trials": [],
            "is_transfer": use_transfer,
        }

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, state: dict, info: dict) -> float:
        reward = 0.0
        ee = _get_ee_pos(data)
        sphere_pos = _get_body_xpos(data, "sphere_red")

        # Track first tool touched
        if state["first_tool_touched"] is None:
            for tool in ("stick", "rake", "spoon"):
                tp = _get_body_xpos(data, tool)
                if _dist(ee, tp) < 0.06:
                    state["first_tool_touched"] = tool
                    break

        sphere_disp = _dist(sphere_pos[:2], state["sphere_init"][:2])
        if sphere_disp > self.RETRIEVAL_THRESH and not state["success_this_ep"]:
            state["success_this_ep"] = True
            reward += 10.0

        # Shaping: approach sphere
        reward += max(0.0, 0.01 * (1.0 - min(1.0, sphere_disp / self.RETRIEVAL_THRESH)))

        info["tc20"] = {
            "first_tool": state["first_tool_touched"],
            "sphere_disp": sphere_disp,
            "success": state["success_this_ep"],
        }
        return reward

    def score(self, state: dict) -> float:
        k = state["known_trials"]
        sel = state["known_selections"]
        tr = state["transfer_trials"]
        known_rate = sum(k) / len(k) if k else float(state["success_this_ep"])
        functional = {"stick", "rake"}
        selection_rate = sum(1 for s in sel if s in functional) / len(sel) if sel else 0.0
        transfer_rate = sum(tr) / len(tr) if tr else 0.0
        return 0.4 * known_rate + 0.3 * selection_rate + 0.3 * transfer_rate


# ---------------------------------------------------------------------------
# TC-21: Support Relations
# ---------------------------------------------------------------------------

class TC21SupportRelations:
    task_id = "TC-21"
    PRED_THRESH = 0.05  # m — EE must be within this of fall location

    def scene_objects(self) -> list[str]:
        return ["cube_green", "platform"]

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> dict:
        platform_pos = _sample_pos(rng, MID_X, NEAR_Y)
        platform_pos[0] = min(platform_pos[0] + rng.uniform(-0.05, 0.05), 2.65)
        _set_body_pos(model, data, "platform", platform_pos)
        # Place cube on platform
        cube_pos = platform_pos.copy()
        cube_pos[2] = platform_pos[2] + 0.025 + 0.025  # platform half-height + cube half-height
        _set_body_pos(model, data, "cube_green", cube_pos)
        return {
            "platform_init": platform_pos.copy(),
            "cube_init": cube_pos.copy(),
            "phase": "observation",
            "phase_step": 0,
            "prediction_trials": [],
            "catch_trials": [],
            "ee_prediction_pos": None,
        }

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, state: dict, info: dict) -> float:
        state["phase_step"] += 1
        reward = 0.0
        cube_pos = _get_body_xpos(data, "cube_green")
        ee = _get_ee_pos(data)

        if state["phase"] == "observation":
            # Passive observation; environment demonstrates platform removal
            if state["phase_step"] > 200:
                state["phase"] = "prediction"
                state["phase_step"] = 0
        elif state["phase"] == "prediction":
            # Agent positions EE before platform removed (first 50 steps)
            if state["phase_step"] <= 50:
                state["ee_prediction_pos"] = ee.copy()
            elif state["phase_step"] == 51:
                # Platform removed — let cube fall
                _set_body_pos(model, data, "platform", np.array([5.0, 5.0, 0.0]))
            elif state["phase_step"] == 80:
                # Measure cube landing
                if state["ee_prediction_pos"] is not None:
                    pred_err = _dist(state["ee_prediction_pos"][:2], cube_pos[:2])
                    state["prediction_trials"].append(pred_err < self.PRED_THRESH)
                state["phase_step"] = 0  # next trial
        elif state["phase"] == "catch":
            # Agent must catch cube before it falls off table
            if cube_pos[2] < TABLE_HEIGHT - 0.05:
                caught = _is_grasped(data, model) and _dist(ee, cube_pos) < 0.10
                state["catch_trials"].append(caught)
                state["phase_step"] = 0

        info["tc21"] = {
            "phase": state["phase"],
            "prediction_trials": state["prediction_trials"],
            "catch_trials": state["catch_trials"],
        }
        return reward

    def score(self, state: dict) -> float:
        p = state["prediction_trials"]
        c = state["catch_trials"]
        pred_rate = sum(p) / len(p) if p else 0.0
        catch_rate = sum(c) / len(c) if c else 0.0
        return 0.6 * pred_rate + 0.4 * catch_rate


# ---------------------------------------------------------------------------
# TC-22: Insightful Problem Solving
# ---------------------------------------------------------------------------

class TC22InsightfulProblemSolving:
    task_id = "TC-22"
    LATCH_UNLOCK_THRESH = 0.3  # rad

    def scene_objects(self) -> list[str]:
        return ["sphere_blue", "latch_box"]

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> dict:
        box_pos = np.array([2.5, rng.uniform(2.3, 2.7), OBJ_Z + 0.05])
        _set_body_pos(model, data, "latch_box", box_pos)
        sphere_pos = box_pos.copy()
        sphere_pos[2] += 0.02
        _set_body_pos(model, data, "sphere_blue", sphere_pos)
        return {
            "trials": [],
            "solved": False,
            "attempts": 0,
            "ep_step": 0,
            "first_move_step": None,
            "latch_unlocked": False,
            "lid_opened": False,
            "prev_ee": None,
            "attempt_active": False,
        }

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, state: dict, info: dict) -> float:
        state["ep_step"] += 1
        reward = 0.0
        ee = _get_ee_pos(data)

        if state["prev_ee"] is None:
            state["prev_ee"] = ee.copy()

        ee_moved = _dist(ee, state["prev_ee"]) > 0.005
        if ee_moved and state["first_move_step"] is None:
            state["first_move_step"] = state["ep_step"]

        # Track attempts (approach-retract cycles)
        box_pos = _get_body_xpos(data, "latch_box")
        near_box = _dist(ee, box_pos) < 0.15
        if near_box and not state["attempt_active"]:
            state["attempt_active"] = True
            state["attempts"] += 1
        elif not near_box and state["attempt_active"]:
            state["attempt_active"] = False

        # Check latch angle
        latch_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "latch_hinge")
        if latch_jnt_id >= 0:
            qpos_adr = model.jnt_qposadr[latch_jnt_id]
            latch_angle = abs(data.qpos[qpos_adr])
            if latch_angle > self.LATCH_UNLOCK_THRESH and not state["latch_unlocked"]:
                state["latch_unlocked"] = True
                reward += 3.0

        # Check lid
        lid_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lid_slide")
        if lid_jnt_id >= 0 and state["latch_unlocked"]:
            qpos_adr = model.jnt_qposadr[lid_jnt_id]
            if data.qpos[qpos_adr] > 0.05 and not state["lid_opened"]:
                state["lid_opened"] = True
                reward += 5.0

        # Solved: sphere grasped after lid open
        if state["lid_opened"] and _is_grasped(data, model):
            sphere_pos = _get_body_xpos(data, "sphere_blue")
            if _dist(ee, sphere_pos) < 0.06 and not state["solved"]:
                state["solved"] = True
                reward += 10.0

        state["prev_ee"] = ee.copy()

        info["tc22"] = {
            "solved": state["solved"],
            "attempts": state["attempts"],
            "latch_unlocked": state["latch_unlocked"],
            "lid_opened": state["lid_opened"],
        }
        return reward

    def score(self, state: dict) -> float:
        trials = state["trials"]
        if not trials:
            # single episode
            solve_rate = float(state["solved"])
            attempts = state["attempts"]
            efficiency = max(0.0, 1.0 - (attempts - 1) / 10.0)
            pause = state["first_move_step"] or 0
            if 20 <= pause <= 50:
                insight = 1.0
            elif pause < 20:
                insight = pause / 20.0
            else:
                insight = max(0.0, 1.0 - (pause - 50) / 100.0)
            return 0.4 * solve_rate + 0.3 * efficiency + 0.3 * insight

        n = len(trials)
        solve_rate = sum(1 for t in trials if t["solved"]) / n
        solved_trials = [t for t in trials if t["solved"]]
        if solved_trials:
            avg_attempts = sum(t["attempts"] for t in solved_trials) / len(solved_trials)
            efficiency = max(0.0, 1.0 - (avg_attempts - 1) / 10.0)
        else:
            efficiency = 0.0
        pauses = [t.get("pause", 0) for t in trials]
        avg_pause = sum(pauses) / len(pauses)
        if 20 <= avg_pause <= 50:
            insight = 1.0
        elif avg_pause < 20:
            insight = avg_pause / 20.0
        else:
            insight = max(0.0, 1.0 - (avg_pause - 50) / 100.0)
        return 0.4 * solve_rate + 0.3 * efficiency + 0.3 * insight


# ---------------------------------------------------------------------------
# TC-23: Deferred Imitation
# ---------------------------------------------------------------------------

class TC23DeferredImitation:
    task_id = "TC-23"
    PICK_Z = TABLE_HEIGHT + 0.05   # lifted above table
    BOX_INNER = 0.07               # box inner half-extent

    def scene_objects(self) -> list[str]:
        return ["cube_red", "cube_blue", "box_open"]

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> dict:
        # Fixed layout for demo consistency
        _set_body_pos(model, data, "cube_red", np.array([2.3, 2.5, OBJ_Z]))
        _set_body_pos(model, data, "cube_blue", np.array([2.5, 2.3, OBJ_Z]))
        _set_body_pos(model, data, "box_open", np.array([2.6, 2.5, OBJ_Z + 0.005]))
        return {
            "phase": "demo",   # "demo" | "short_delay" | "medium_delay" | "long_delay" | "repro"
            "demo_step": 0,
            "short_delay_trials": [],
            "medium_delay_trials": [],
            "long_delay_trials": [],
            "trial_state": {"step1": False, "step2": False, "step3": False},
            "delay_type": "short",
        }

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, state: dict, info: dict) -> float:
        reward = 0.0
        if state["phase"] == "demo":
            state["demo_step"] += 1
            return 0.0  # observation only

        # Reproduction phase
        ee = _get_ee_pos(data)
        cube_red_pos = _get_body_xpos(data, "cube_red")
        cube_blue_pos = _get_body_xpos(data, "cube_blue")
        box_pos = _get_body_xpos(data, "box_open")

        ts = state["trial_state"]

        # Step 1: pick up cube_red (grasp + lift)
        if not ts["step1"]:
            if _is_grasped(data, model) and _dist(ee, cube_red_pos) < 0.06 and cube_red_pos[2] > self.PICK_Z:
                ts["step1"] = True

        # Step 2: place cube_red in box (object inside box bounds)
        if ts["step1"] and not ts["step2"]:
            in_box = (
                abs(cube_red_pos[0] - box_pos[0]) < self.BOX_INNER and
                abs(cube_red_pos[1] - box_pos[1]) < self.BOX_INNER
            )
            if in_box:
                ts["step2"] = True

        # Step 3: push cube_blue off table
        if ts["step2"] and not ts["step3"]:
            if cube_blue_pos[2] < TABLE_HEIGHT - 0.05:
                ts["step3"] = True

        info["tc23"] = {"trial_state": dict(ts)}
        return reward

    def score(self, state: dict) -> float:
        def repro_rate(trials):
            if not trials:
                return 0.0
            return sum(
                1 for t in trials if t["step1"] and t["step2"] and t["step3"]
            ) / len(trials)
        s = repro_rate(state["short_delay_trials"])
        m = repro_rate(state["medium_delay_trials"])
        l = repro_rate(state["long_delay_trials"])
        return 0.40 * s + 0.40 * m + 0.20 * l


# ---------------------------------------------------------------------------
# TC-24: Invisible Displacement
# ---------------------------------------------------------------------------

class TC24InvisibleDisplacement:
    task_id = "TC-24"
    SEARCH_THRESH = 0.05  # m
    HIDING_STEPS = 20
    TRAVEL_STEPS = 15
    REVEAL_STEPS = 10

    def scene_objects(self) -> list[str]:
        return ["sphere_red", "box_open", "screen_A", "screen_B"]

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> dict:
        sphere_pos = np.array([2.4, 2.5, OBJ_Z + 0.005])
        _set_body_pos(model, data, "sphere_red", sphere_pos)
        box_pos = np.array([2.45, 2.5, OBJ_Z + 0.005])
        _set_body_pos(model, data, "box_open", box_pos)
        # Randomize which screen sphere is left behind
        deposit_screen = rng.choice(["A", "B"])
        return {
            "phase": "visible_warmup",  # → "invisible_test"
            "trial_step": 0,
            "trial": 0,
            "deposit_screen": deposit_screen,
            "disp_stage": 0,  # 0-5 stages of displacement sequence
            "visible_trials": [],
            "invisible_trials": [],
            "search_recorded": False,
        }

    def _screen_pos(self, model: mujoco.MjModel, loc: str) -> np.ndarray:
        name = f"screen_{loc}"
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return model.body_pos[bid].copy()

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, state: dict, info: dict) -> float:
        state["trial_step"] += 1
        reward = 0.0
        ee = _get_ee_pos(data)

        if state["phase"] == "visible_warmup":
            # Simple: sphere hidden directly behind one screen
            deposit = state["deposit_screen"]
            screen_pos = self._screen_pos(model, deposit)
            sphere_behind = screen_pos + np.array([0.0, -0.05, 0.0])
            _set_body_pos(model, data, "sphere_red", sphere_behind)

            if not state["search_recorded"] and state["trial_step"] > 20:
                dist_A = _dist(ee, self._screen_pos(model, "A"))
                dist_B = _dist(ee, self._screen_pos(model, "B"))
                if dist_A < self.SEARCH_THRESH or dist_B < self.SEARCH_THRESH:
                    result = "correct" if (
                        (deposit == "A" and dist_A < self.SEARCH_THRESH) or
                        (deposit == "B" and dist_B < self.SEARCH_THRESH)
                    ) else "incorrect"
                    state["visible_trials"].append(result)
                    state["search_recorded"] = True
                    reward += 5.0 if result == "correct" else 0.0

            if state["trial_step"] >= 60:
                state["trial"] += 1
                state["trial_step"] = 0
                state["search_recorded"] = False
                if state["trial"] >= 5:
                    state["phase"] = "invisible_test"
                    state["trial"] = 0

        elif state["phase"] == "invisible_test":
            # Displacement sequence: sphere→box→screen_A→empty→screen_B→empty
            deposit = state["deposit_screen"]
            seq_step = state["trial_step"]

            if seq_step <= 20:  # Put sphere in box (visible)
                box_pos = _get_body_xpos(data, "box_open")
                _set_body_pos(model, data, "sphere_red", box_pos + np.array([0, 0, 0.01]))
            elif seq_step <= 40:  # Move box behind screen_A
                screen_A_pos = self._screen_pos(model, "A")
                _set_body_pos(model, data, "box_open", screen_A_pos + np.array([0, -0.06, 0]))
                if deposit == "A":
                    # Leave sphere behind screen_A
                    _set_body_pos(model, data, "sphere_red", screen_A_pos + np.array([0, -0.05, 0]))
            elif seq_step <= 55:  # Box emerges empty from A
                _set_body_pos(model, data, "box_open", np.array([2.35, 2.5, OBJ_Z]))
            elif seq_step <= 75:  # Move box behind screen_B
                screen_B_pos = self._screen_pos(model, "B")
                _set_body_pos(model, data, "box_open", screen_B_pos + np.array([0, -0.06, 0]))
                if deposit == "B":
                    _set_body_pos(model, data, "sphere_red", screen_B_pos + np.array([0, -0.05, 0]))
            elif seq_step <= 90:  # Box emerges empty from B
                _set_body_pos(model, data, "box_open", np.array([2.35, 2.5, OBJ_Z]))

            # Search window
            elif not state["search_recorded"]:
                dist_A = _dist(ee, self._screen_pos(model, "A"))
                dist_B = _dist(ee, self._screen_pos(model, "B"))
                if dist_A < self.SEARCH_THRESH or dist_B < self.SEARCH_THRESH:
                    result = "correct" if (
                        (deposit == "A" and dist_A < self.SEARCH_THRESH) or
                        (deposit == "B" and dist_B < self.SEARCH_THRESH)
                    ) else "incorrect"
                    state["invisible_trials"].append(result)
                    state["search_recorded"] = True

            if state["trial_step"] >= 140:
                if not state["search_recorded"]:
                    state["invisible_trials"].append("none")
                state["trial"] += 1
                state["trial_step"] = 0
                state["search_recorded"] = False

        info["tc24"] = {
            "phase": state["phase"],
            "visible_trials": state["visible_trials"],
            "invisible_trials": state["invisible_trials"],
        }
        return reward

    def score(self, state: dict) -> float:
        vt = state["visible_trials"]
        it = state["invisible_trials"]
        visible_rate = sum(1 for t in vt if t == "correct") / len(vt) if vt else 0.0
        invisible_rate = sum(1 for t in it if t == "correct") / len(it) if it else 0.0
        return 0.3 * visible_rate + 0.7 * invisible_rate


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, SensorimotorTask] = {
    "TC-11": TC11ReflexValidation(),
    "TC-12": TC12ActionEffectDiscovery(),
    "TC-13": TC13Reaching(),
    "TC-14": TC14ObjectInteraction(),
    "TC-15": TC15MeansEnd(),
    "TC-16": TC16ObjectPermanence(),
    "TC-17": TC17ObstacleRemoval(),
    "TC-18": TC18ClothPull(),
    "TC-19": TC19ActiveExperimentation(),
    "TC-20": TC20ToolUse(),
    "TC-21": TC21SupportRelations(),
    "TC-22": TC22InsightfulProblemSolving(),
    "TC-23": TC23DeferredImitation(),
    "TC-24": TC24InvisibleDisplacement(),
}

VALID_TASK_IDS = tuple(TASK_REGISTRY.keys())
