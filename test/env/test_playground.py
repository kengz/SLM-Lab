"""Tests for MuJoCo Playground integration."""

from unittest.mock import MagicMock, patch

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest


# ============================================================================
# PlaygroundVecEnv tests (require mujoco_playground)
# ============================================================================


class TestPlaygroundVecEnv:
    """Tests for PlaygroundVecEnv with live mujoco_playground."""

    @pytest.fixture(autouse=True)
    def check_playground_available(self):
        pytest.importorskip("mujoco_playground")

    @pytest.fixture
    def env(self):
        from slm_lab.env.playground import PlaygroundVecEnv

        env = PlaygroundVecEnv("CartpoleBalance", num_envs=4)
        yield env
        env.close()

    def test_instantiation(self, env):
        assert env.num_envs == 4

    def test_spaces(self, env):
        assert env.single_observation_space is not None
        assert env.single_action_space is not None
        obs_dim = env.single_observation_space.shape[0]
        act_dim = env.single_action_space.shape[0]
        assert obs_dim > 0
        assert act_dim > 0
        # Batched spaces should have num_envs in first dim
        assert env.observation_space.shape == (4, obs_dim)
        assert env.action_space.shape == (4, act_dim)

    def test_reset(self, env):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4, env.single_observation_space.shape[0])
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_step(self, env):
        env.reset()
        actions = np.random.uniform(-1, 1, size=env.action_space.shape).astype(np.float32)
        obs, rewards, terminated, truncated, info = env.step(actions)

        assert obs.shape == (4, env.single_observation_space.shape[0])
        assert obs.dtype == np.float32
        assert rewards.shape == (4,)
        assert rewards.dtype == np.float32
        assert terminated.shape == (4,)
        assert terminated.dtype == bool
        assert truncated.shape == (4,)
        assert truncated.dtype == bool
        assert isinstance(info, dict)

    def test_reset_with_seed(self, env):
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_multiple_steps(self, env):
        env.reset()
        for _ in range(10):
            actions = np.random.uniform(-1, 1, size=env.action_space.shape).astype(np.float32)
            obs, rewards, terminated, truncated, info = env.step(actions)
            assert obs.shape[0] == 4


# ============================================================================
# make_env routing tests (mocked — no mujoco_playground needed)
# ============================================================================


class TestMakeEnvPlaygroundRouting:
    """Test that make_env routes playground/ envs to _make_playground_env."""

    def test_playground_prefix_routes_correctly(self):
        spec = {
            "agent": {"algorithm": {"gamma": 0.99}},
            "env": {
                "name": "playground/CartpoleBalance",
                "num_envs": 4,
                "max_frame": 100000,
            },
            "meta": {
                "distributed": False,
                "eval_frequency": 5000,
                "log_frequency": 5000,
                "max_session": 1,
            },
        }

        with patch("slm_lab.env._make_playground_env") as mock_pg:
            # Create a mock env with real gymnasium spaces
            obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
            act_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            mock_env = MagicMock(spec=gym.vector.VectorEnv)
            mock_env.num_envs = 4
            mock_env.is_venv = True
            mock_env.single_observation_space = obs_space
            mock_env.single_action_space = act_space
            mock_env.observation_space = obs_space
            mock_env.action_space = act_space
            mock_env.spec = None
            mock_pg.return_value = mock_env

            from slm_lab.env import make_env

            make_env(spec)
            mock_pg.assert_called_once()
            call_args = mock_pg.call_args
            assert call_args[0][0] == "playground/CartpoleBalance"
            assert call_args[0][1] == 4

    def test_non_playground_does_not_route(self):
        spec = {
            "agent": {"algorithm": {"gamma": 0.99}},
            "env": {
                "name": "CartPole-v1",
                "num_envs": 1,
                "max_frame": 1000,
            },
            "meta": {
                "distributed": False,
                "eval_frequency": 1000,
                "log_frequency": 1000,
                "max_session": 1,
            },
        }

        with patch("slm_lab.env._make_playground_env") as mock_pg:
            from slm_lab.env import make_env

            env = make_env(spec)
            mock_pg.assert_not_called()
            env.close()


# ============================================================================
# Import guard tests
# ============================================================================


class TestImportGuard:
    """Test that slm_lab.env imports cleanly without mujoco_playground."""

    def test_env_module_imports_without_playground(self):
        """Importing slm_lab.env should not fail if playground is missing.

        The playground import is lazy (inside _make_playground_env), so the
        env module should always import successfully.
        """
        import slm_lab.env

        assert hasattr(slm_lab.env, "make_env")
        assert hasattr(slm_lab.env, "_make_playground_env")
