"""Integration tests for PavlovianMjxEnv (MuJoCo Playground native).

Tests use JAX CPU backend (impl='jax') for portability without a GPU.
"""
import jax
import jax.numpy as jp
import numpy as np
import pytest

from slm_lab.env.pavlovian_mjx import PavlovianMjxEnv, default_config

_OBS_DIM = 18  # 8 base + 3 objects × 3 dims + 1 cs_signal


@pytest.fixture
def env():
    config = default_config()
    config.impl = "jax"  # CPU-compatible backend
    return PavlovianMjxEnv(config=config, task_id=7)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


class TestModelInit:
    def test_mj_model_loaded(self, env):
        assert env.mj_model is not None

    def test_mjx_model_loaded(self, env):
        assert env.mjx_model is not None

    def test_action_size(self, env):
        assert env.action_size == 2


class TestReset:
    def test_obs_shape(self, env, rng):
        state = env.reset(rng)
        assert state.obs.shape == (_OBS_DIM,)

    def test_reward_zero_at_reset(self, env, rng):
        state = env.reset(rng)
        assert float(state.reward) == 0.0

    def test_done_zero_at_reset(self, env, rng):
        state = env.reset(rng)
        assert float(state.done) == 0.0

    def test_energy_normalized_at_reset(self, env, rng):
        state = env.reset(rng)
        obs = np.asarray(state.obs)
        # energy=100, normalized: (100-50)/50 = 1.0
        assert obs[6] == pytest.approx(1.0, abs=0.01)

    def test_time_normalized_at_reset(self, env, rng):
        state = env.reset(rng)
        obs = np.asarray(state.obs)
        # step_count=0, normalized: 2*0/1000 - 1 = -1.0
        assert obs[7] == pytest.approx(-1.0, abs=0.01)

    def test_object_positions_in_range(self, env, rng):
        state = env.reset(rng)
        obj_pos = np.asarray(state.info["obj_pos"])
        # Red sphere base (7, 7) ± 0.5
        assert 6.5 <= obj_pos[0, 0] <= 7.5
        assert 6.5 <= obj_pos[0, 1] <= 7.5
        # Blue cube base (3, 7) ± 0.5
        assert 2.5 <= obj_pos[1, 0] <= 3.5
        assert 6.5 <= obj_pos[1, 1] <= 7.5
        # Green cylinder base (5, 3) ± 0.5
        assert 4.5 <= obj_pos[2, 0] <= 5.5
        assert 2.5 <= obj_pos[2, 1] <= 3.5


class TestStep:
    def test_obs_shape_after_step(self, env, rng):
        state = env.reset(rng)
        state = env.step(state, jp.array([0.5, 0.1]))
        assert state.obs.shape == (_OBS_DIM,)

    def test_reward_scalar(self, env, rng):
        state = env.reset(rng)
        state = env.step(state, jp.array([0.5, 0.1]))
        assert state.reward.shape == ()

    def test_done_scalar(self, env, rng):
        state = env.reset(rng)
        state = env.step(state, jp.array([0.5, 0.1]))
        assert state.done.shape == ()

    def test_agent_moves_with_forward_action(self, env, rng):
        state = env.reset(rng)
        x_before = float(state.info["qpos"][0])
        y_before = float(state.info["qpos"][1])
        state = env.step(state, jp.array([1.0, 0.0]))
        x_after = float(state.info["qpos"][0])
        y_after = float(state.info["qpos"][1])
        displacement = np.sqrt((x_after - x_before) ** 2 + (y_after - y_before) ** 2)
        assert displacement > 0.0


class TestTC07Reward:
    def test_forward_action_gives_positive_reward(self, env, rng):
        state = env.reset(rng)
        state = env.step(state, jp.array([1.0, 0.0]))
        assert float(state.reward) > 0.0

    def test_zero_forward_gives_zero_reward(self, env, rng):
        state = env.reset(rng)
        state = env.step(state, jp.array([0.0, 0.5]))
        assert float(state.reward) == pytest.approx(0.0, abs=1e-6)

    def test_reward_proportional_to_forward(self, env, rng):
        state_half = env.reset(rng)
        state_full = env.reset(rng)
        state_half = env.step(state_half, jp.array([0.5, 0.0]))
        state_full = env.step(state_full, jp.array([1.0, 0.0]))
        # Full forward should give roughly 2x reward of half forward
        assert float(state_full.reward) == pytest.approx(
            float(state_half.reward) * 2.0, rel=0.01
        )


class TestEpisodeTermination:
    def test_episode_ends_after_max_steps_or_energy(self, env, rng):
        state = env.reset(rng)
        action = jp.array([1.0, 1.0])
        jit_step = jax.jit(env.step)
        for _ in range(1100):
            if float(state.done) == 1.0:
                break
            state = jit_step(state, action)
        assert float(state.done) == 1.0

    def test_wall_clamping_keeps_agent_in_bounds(self, env, rng):
        state = env.reset(rng)
        action = jp.array([1.0, 0.0])
        jit_step = jax.jit(env.step)
        for _ in range(500):
            if float(state.done) == 1.0:
                break
            state = jit_step(state, action)
            x = float(state.info["qpos"][0])
            y = float(state.info["qpos"][1])
            assert 0.25 <= x <= 9.75, f"x={x} out of bounds"
            assert 0.25 <= y <= 9.75, f"y={y} out of bounds"


class TestJITCompatibility:
    def test_step_is_jit_compilable(self, env, rng):
        state = env.reset(rng)
        jit_step = jax.jit(env.step)
        state2 = jit_step(state, jp.array([0.5, 0.0]))
        assert state2.obs.shape == (_OBS_DIM,)

    def test_reset_is_jit_compilable(self, env, rng):
        jit_reset = jax.jit(env.reset)
        state = jit_reset(rng)
        assert state.obs.shape == (_OBS_DIM,)
