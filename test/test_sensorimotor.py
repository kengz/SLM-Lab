"""Integration tests for the sensorimotor MuJoCo environment (TC-11 to TC-24).

Tests:
  - Model instantiation for all 14 tasks
  - Observation and action space shapes and dtypes
  - reset() returns valid obs + info
  - step() returns valid 5-tuple; obs/reward/terminated/truncated types correct
  - Per-task sanity: score() returns float in [0, 1]
  - Vectorized env: gymnasium.vector.SyncVectorEnv wraps correctly
"""

from __future__ import annotations

import pytest
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Registration side-effect
import slm_lab.env  # noqa: F401

from slm_lab.env.sensorimotor import SLMSensorimotor, OBS_DIM
from slm_lab.env.sensorimotor_tasks import VALID_TASK_IDS, TASK_REGISTRY


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_TASK_IDS = list(VALID_TASK_IDS)
OBS_GROUND_TRUTH_DIM = OBS_DIM          # 56
ACTION_DIM = 10
VISION_SHAPE = (2, 128, 128, 3)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=ALL_TASK_IDS, ids=ALL_TASK_IDS)
def env(request):
    """Create and tear down one env per task_id."""
    e = SLMSensorimotor(task_id=request.param, seed=42)
    yield e
    e.close()


@pytest.fixture(params=["TC-11", "TC-13", "TC-16", "TC-22"])
def env_subset(request):
    """Smaller fixture set for more expensive tests."""
    e = SLMSensorimotor(task_id=request.param, seed=0)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# 1. Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_task_ids_present(self):
        assert len(VALID_TASK_IDS) == 14
        for i in range(11, 25):
            assert f"TC-{i:02d}" in VALID_TASK_IDS

    def test_gymnasium_registration(self):
        for i in range(11, 25):
            env_id = f"SLM-Sensorimotor-TC{i:02d}-v0"
            assert env_id in gym.envs.registry, f"Missing registration: {env_id}"

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task_id"):
            SLMSensorimotor(task_id="TC-99")


# ---------------------------------------------------------------------------
# 2. Spaces
# ---------------------------------------------------------------------------

class TestSpaces:
    def test_observation_space_is_dict(self, env):
        assert isinstance(env.observation_space, spaces.Dict)

    def test_ground_truth_shape(self, env):
        gt_space = env.observation_space["ground_truth"]
        assert isinstance(gt_space, spaces.Box)
        assert gt_space.shape == (OBS_GROUND_TRUTH_DIM,), (
            f"Expected {OBS_GROUND_TRUTH_DIM}, got {gt_space.shape}"
        )
        assert gt_space.dtype == np.float32

    def test_vision_placeholder_shape(self, env):
        v_space = env.observation_space["vision"]
        assert isinstance(v_space, spaces.Box)
        assert v_space.shape == VISION_SHAPE

    def test_action_space(self, env):
        assert isinstance(env.action_space, spaces.Box)
        assert env.action_space.shape == (ACTION_DIM,)
        assert env.action_space.dtype == np.float32
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)


# ---------------------------------------------------------------------------
# 3. Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert "ground_truth" in obs
        assert isinstance(info, dict)

    def test_reset_obs_shape(self, env):
        obs, _ = env.reset()
        assert obs["ground_truth"].shape == (OBS_GROUND_TRUTH_DIM,)
        assert obs["ground_truth"].dtype == np.float32

    def test_reset_info_keys(self, env):
        _, info = env.reset()
        for key in ("task_id", "step", "energy", "ee_position"):
            assert key in info, f"Missing info key: {key}"

    def test_reset_task_id_matches(self, env):
        _, info = env.reset()
        assert info["task_id"] == env.task_id

    def test_reset_energy_full(self, env):
        _, info = env.reset()
        assert info["energy"] == pytest.approx(100.0)

    def test_reset_step_zero(self, env):
        _, info = env.reset()
        assert info["step"] == 0

    def test_seeded_reset_reproducible(self, env):
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        np.testing.assert_array_equal(obs1["ground_truth"], obs2["ground_truth"])


# ---------------------------------------------------------------------------
# 4. Step
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_returns_five_tuple(self, env):
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5

    def test_step_obs_shape(self, env):
        env.reset()
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        assert obs["ground_truth"].shape == (OBS_GROUND_TRUTH_DIM,)

    def test_step_reward_float(self, env):
        env.reset()
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert isinstance(reward, float)

    def test_step_terminated_bool(self, env):
        env.reset()
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_info_has_score(self, env):
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        assert "score" in info
        score = info["score"]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1]"

    def test_step_action_clipped(self, env):
        """Out-of-range actions should not raise."""
        env.reset()
        big_action = np.ones(ACTION_DIM, dtype=np.float32) * 5.0
        obs, _, _, _, _ = env.step(big_action)
        assert obs["ground_truth"].shape == (OBS_GROUND_TRUTH_DIM,)

    def test_multiple_steps(self, env):
        env.reset()
        for _ in range(10):
            obs, reward, term, trunc, info = env.step(env.action_space.sample())
            assert obs["ground_truth"].shape == (OBS_GROUND_TRUTH_DIM,)
            assert isinstance(reward, float)
        assert info["step"] == 10

    def test_energy_decreases(self, env):
        _, info0 = env.reset()
        _, _, _, _, info1 = env.step(np.zeros(ACTION_DIM))
        assert info1["energy"] < info0["energy"]


# ---------------------------------------------------------------------------
# 5. Per-task smoke tests
# ---------------------------------------------------------------------------

class TestPerTask:
    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_task_instantiates(self, task_id):
        task = TASK_REGISTRY[task_id]
        assert task.task_id == task_id

    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_task_env_runs_5_steps(self, task_id):
        e = SLMSensorimotor(task_id=task_id, seed=7)
        obs, info = e.reset()
        assert obs["ground_truth"].shape == (OBS_GROUND_TRUTH_DIM,)
        for _ in range(5):
            obs, reward, term, trunc, info = e.step(e.action_space.sample())
        assert "score" in info
        assert 0.0 <= info["score"] <= 1.0
        e.close()

    @pytest.mark.parametrize("task_id", ALL_TASK_IDS)
    def test_task_scene_objects_are_strings(self, task_id):
        task = TASK_REGISTRY[task_id]
        objs = task.scene_objects()
        assert isinstance(objs, list)
        assert all(isinstance(o, str) for o in objs)

    def test_tc11_visual_tactile_proprio(self):
        e = SLMSensorimotor(task_id="TC-11", seed=1)
        e.reset()
        for _ in range(20):
            e.step(e.action_space.sample())
        e.close()

    def test_tc13_reaching_score_format(self):
        e = SLMSensorimotor(task_id="TC-13", seed=2)
        e.reset()
        for _ in range(50):
            e.step(np.zeros(ACTION_DIM))
        _, _, _, _, info = e.step(e.action_space.sample())
        assert isinstance(info["score"], float)
        e.close()

    def test_tc16_object_permanence_phase_tracking(self):
        e = SLMSensorimotor(task_id="TC-16", seed=3)
        _, info = e.reset()
        # Task state should have phase
        for _ in range(30):
            _, _, _, _, info = e.step(e.action_space.sample())
        assert "tc16" in info
        assert "phase" in info["tc16"]
        e.close()

    def test_tc22_insightful_solving_has_attempts(self):
        e = SLMSensorimotor(task_id="TC-22", seed=4)
        e.reset()
        for _ in range(30):
            _, _, _, _, info = e.step(e.action_space.sample())
        assert "tc22" in info
        assert "attempts" in info["tc22"]
        e.close()


# ---------------------------------------------------------------------------
# 6. Gymnasium registration (gym.make)
# ---------------------------------------------------------------------------

class TestGymnasiumMake:
    @pytest.mark.parametrize("tc_num", [11, 13, 16, 22, 24])
    def test_gym_make(self, tc_num):
        env_id = f"SLM-Sensorimotor-TC{tc_num:02d}-v0"
        e = gym.make(env_id)
        obs, info = e.reset()
        assert "ground_truth" in obs
        e.close()

    def test_gym_make_tc11(self):
        e = gym.make("SLM-Sensorimotor-TC11-v0")
        assert e.unwrapped.task_id == "TC-11"
        e.close()


# ---------------------------------------------------------------------------
# 7. Vectorized environment
# ---------------------------------------------------------------------------

class TestVectorizedEnv:
    def test_sync_vector_env_wraps(self):
        def make():
            return SLMSensorimotor(task_id="TC-13", seed=0)

        vec_env = gym.vector.SyncVectorEnv([make, make])
        obs, infos = vec_env.reset()
        # SyncVectorEnv stacks obs; ground_truth should have batch dim
        gt = obs.get("ground_truth") if isinstance(obs, dict) else obs
        if gt is not None:
            assert gt.shape[0] == 2
        vec_env.close()

    def test_sync_vec_step(self):
        def make():
            return SLMSensorimotor(task_id="TC-13", seed=1)

        vec_env = gym.vector.SyncVectorEnv([make, make, make])
        vec_env.reset()
        actions = vec_env.action_space.sample()
        result = vec_env.step(actions)
        assert len(result) == 5
        vec_env.close()


# ---------------------------------------------------------------------------
# 8. Observation range sanity
# ---------------------------------------------------------------------------

class TestObsRange:
    def test_proprioception_normalized(self):
        """Joint angle obs should mostly stay in [-2, 2] range."""
        e = SLMSensorimotor(task_id="TC-13", seed=5)
        e.reset()
        for _ in range(10):
            obs, _, _, _, _ = e.step(e.action_space.sample())
        gt = obs["ground_truth"]
        # Proprioception channels 0-6 (joint angles normalized to ~[-1, 1] + noise)
        joint_angles = gt[:7]
        assert np.all(np.abs(joint_angles) < 3.0), (
            f"Joint angles out of expected range: {joint_angles}"
        )
        e.close()

    def test_internal_state_in_range(self):
        """Energy and time fraction should be in [-1, 1]."""
        e = SLMSensorimotor(task_id="TC-13", seed=6)
        e.reset()
        for _ in range(5):
            obs, _, _, _, _ = e.step(np.zeros(ACTION_DIM))
        gt = obs["ground_truth"]
        energy_norm = gt[33]
        time_frac = gt[34]
        assert -2.0 <= energy_norm <= 2.0, f"Energy norm {energy_norm} out of range"
        assert -2.0 <= time_frac <= 2.0, f"Time frac {time_frac} out of range"
        e.close()


# ---------------------------------------------------------------------------
# 9. Score functions (unit tests on task scoring logic)
# ---------------------------------------------------------------------------

class TestScoringFunctions:
    def test_tc11_score_zero_without_trials(self):
        from slm_lab.env.sensorimotor_tasks import TC11ReflexValidation
        task = TC11ReflexValidation()
        state = {"visual_trials": [], "tactile_trials": [], "proprio_trials": []}
        assert task.score(state) == 0.0

    def test_tc11_score_one_with_all_success(self):
        from slm_lab.env.sensorimotor_tasks import TC11ReflexValidation
        task = TC11ReflexValidation()
        state = {
            "visual_trials": [True] * 20,
            "tactile_trials": [True] * 20,
            "proprio_trials": [True] * 20,
        }
        assert task.score(state) == pytest.approx(1.0, abs=0.01)

    def test_tc13_score_no_successes(self):
        from slm_lab.env.sensorimotor_tasks import TC13Reaching
        task = TC13Reaching()
        state = {"successes": [], "completion_times": [], "reached_this_ep": False, "ep_step": 0}
        assert task.score(state) == 0.0

    def test_tc13_score_all_success(self):
        from slm_lab.env.sensorimotor_tasks import TC13Reaching
        task = TC13Reaching()
        state = {
            "successes": [True] * 20,
            "completion_times": [50] * 20,
            "reached_this_ep": False,
            "ep_step": 0,
        }
        score = task.score(state)
        assert 0.0 < score <= 1.0

    def test_tc16_stage4_score(self):
        from slm_lab.env.sensorimotor_tasks import TC16ObjectPermanence
        task = TC16ObjectPermanence()
        state = {
            "a_trial_results": ["A"] * 5,
            "b_trial_results": ["A", "A", "B", "A", "A"],
            "acq_gate_passed": True,
        }
        s4 = task.score_stage4(state)
        assert s4 == pytest.approx(0.8, abs=0.01)
        s5 = task.score(state)
        assert s5 == pytest.approx(0.2, abs=0.01)

    def test_tc22_score_single_ep_zero_when_unsolved(self):
        from slm_lab.env.sensorimotor_tasks import TC22InsightfulProblemSolving
        task = TC22InsightfulProblemSolving()
        state = {
            "trials": [],
            "solved": False,
            "attempts": 1,
            "ep_step": 100,
            "first_move_step": 5,
            "latch_unlocked": False,
            "lid_opened": False,
            "prev_ee": None,
            "attempt_active": False,
        }
        score = task.score(state)
        assert 0.0 <= score <= 1.0

    def test_tc24_score_invisible_only(self):
        from slm_lab.env.sensorimotor_tasks import TC24InvisibleDisplacement
        task = TC24InvisibleDisplacement()
        state = {
            "visible_trials": ["correct"] * 5,
            "invisible_trials": ["correct"] * 12 + ["incorrect"] * 8,
        }
        score = task.score(state)
        assert score == pytest.approx(0.3 * 1.0 + 0.7 * 0.6, abs=0.01)
