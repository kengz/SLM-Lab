"""Tests for reward tracking wrappers and RecordEpisodeStatistics integration.

These tests verify critical reward tracking behavior quickly without requiring
full training runs. They serve as regression tests for the reward tracking system.
"""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.wrappers.vector import RecordEpisodeStatistics as VectorRecordEpisodeStatistics

from slm_lab.env import make_env
from slm_lab.env.wrappers import TrackReward, ClockMixin


# ============================================================================
# TrackReward (single env) tests
# ============================================================================

class TestTrackRewardSingleton:
    """Tests for single environment TrackReward wrapper."""

    def test_accumulates_rewards(self):
        """Verify rewards accumulate correctly within episode."""
        env = gym.make("CartPole-v1")
        env = TrackReward(env)
        env.reset()

        # Take a few steps
        for _ in range(5):
            env.step(0)

        # CartPole gives +1 per step
        assert env.ongoing_reward == 5.0
        env.close()

    def test_episode_end_records_total(self):
        """Verify total_reward is set when episode ends."""
        env = gym.make("CartPole-v1")
        env = TrackReward(env)
        env.reset()

        done = False
        while not done:
            _, _, term, trunc, info = env.step(env.action_space.sample())
            done = term or trunc

        assert env.total_reward > 0
        assert "episode_reward" in info
        assert info["episode_reward"] == env.total_reward
        env.close()

    def test_reset_clears_ongoing(self):
        """Verify reset clears ongoing reward."""
        env = gym.make("CartPole-v1")
        env = TrackReward(env)

        env.reset()
        env.step(0)
        assert env.ongoing_reward > 0

        env.reset()
        assert env.ongoing_reward == 0.0
        env.close()


# ============================================================================
# RecordEpisodeStatistics (vector env) integration tests
# ============================================================================

class TestVectorRecordEpisodeStatistics:
    """Tests for gymnasium RecordEpisodeStatistics integration."""

    def test_return_queue_exists(self):
        """Verify return_queue attribute exists after wrapping."""
        env = gym.make_vec("CartPole-v1", num_envs=2)
        env = VectorRecordEpisodeStatistics(env)

        assert hasattr(env, "return_queue")
        assert hasattr(env, "length_queue")
        env.close()

    def test_return_queue_populates_on_episode_end(self):
        """Verify return_queue gets populated when episodes complete."""
        env = gym.make_vec("CartPole-v1", num_envs=4)
        env = VectorRecordEpisodeStatistics(env)
        env.reset()

        initial_len = len(env.return_queue)

        # Run until at least one episode completes
        for _ in range(500):
            env.step(env.action_space.sample())
            if len(env.return_queue) > initial_len:
                break

        assert len(env.return_queue) > initial_len
        # CartPole should have positive returns
        assert list(env.return_queue)[-1] > 0
        env.close()

    def test_episode_info_format(self):
        """Verify episode info has correct format per gymnasium docs."""
        env = gym.make_vec("CartPole-v1", num_envs=2)
        env = VectorRecordEpisodeStatistics(env)
        env.reset()

        # Run until episode ends
        for _ in range(500):
            _, _, term, trunc, info = env.step(env.action_space.sample())
            if term.any() or trunc.any():
                break

        # Check info structure per gymnasium docs
        assert "episode" in info
        assert "r" in info["episode"]  # returns
        assert "l" in info["episode"]  # lengths
        assert "t" in info["episode"]  # times
        assert "_episode" in info  # boolean mask
        env.close()


# ============================================================================
# make_env integration tests
# ============================================================================

class TestMakeEnvIntegration:
    """Tests for make_env wrapper chain setup."""

    def test_vector_env_has_record_episode_statistics(self):
        """Verify RecordEpisodeStatistics is in vector env wrapper chain."""
        spec = {
            'agent': {'algorithm': {'gamma': 0.99}},
            'env': {'name': 'CartPole-v1', 'num_envs': 2, 'max_frame': 1e6},
            'meta': {'distributed': False, 'eval_frequency': 1000, 'log_frequency': 1000}
        }
        env = make_env(spec)

        # Traverse wrapper chain to find RecordEpisodeStatistics
        found = False
        current = env
        while current is not None:
            if isinstance(current, VectorRecordEpisodeStatistics):
                found = True
                break
            current = getattr(current, "env", None)

        assert found, "RecordEpisodeStatistics not found in wrapper chain"
        env.close()

    def test_single_env_has_track_reward(self):
        """Verify TrackReward is in single env wrapper chain."""
        spec = {
            'agent': {'algorithm': {'gamma': 0.99}},
            'env': {'name': 'CartPole-v1', 'num_envs': 1, 'max_frame': 1e6},
            'meta': {'distributed': False, 'eval_frequency': 1000, 'log_frequency': 1000}
        }
        env = make_env(spec)

        # Traverse wrapper chain to find TrackReward
        found = False
        current = env
        while current is not None:
            if isinstance(current, TrackReward):
                found = True
                break
            current = getattr(current, "env", None)

        assert found, "TrackReward not found in wrapper chain"
        env.close()


# ============================================================================
# ClockMixin.total_reward tests
# ============================================================================

class TestClockMixinTotalReward:
    """Tests for ClockMixin.total_reward property."""

    def test_returns_nan_before_episodes_complete(self):
        """Verify NaN returned when no episodes have completed."""
        spec = {
            'agent': {'algorithm': {'gamma': 0.99}},
            'env': {'name': 'CartPole-v1', 'num_envs': 2, 'max_frame': 1e6},
            'meta': {'distributed': False, 'eval_frequency': 1000, 'log_frequency': 1000}
        }
        env = make_env(spec)
        env.reset()

        # Take one step - no episode should complete
        env.step(np.array([0, 0]))

        assert np.isnan(env.total_reward)
        env.close()

    def test_returns_mean_after_episodes_complete(self):
        """Verify mean of return_queue returned after episodes complete."""
        spec = {
            'agent': {'algorithm': {'gamma': 0.99}},
            'env': {'name': 'CartPole-v1', 'num_envs': 4, 'max_frame': 1e6},
            'meta': {'distributed': False, 'eval_frequency': 1000, 'log_frequency': 1000}
        }
        env = make_env(spec)
        env.reset()

        # Run until episodes complete
        for _ in range(500):
            env.step(np.array([env.action_space.sample() for _ in range(4)]))
            if not np.isnan(env.total_reward):
                break

        reward = env.total_reward
        assert not np.isnan(reward)
        assert reward > 0  # CartPole has positive rewards
        env.close()

    def test_single_env_total_reward(self):
        """Verify total_reward works for single env."""
        spec = {
            'agent': {'algorithm': {'gamma': 0.99}},
            'env': {'name': 'CartPole-v1', 'num_envs': 1, 'max_frame': 1e6},
            'meta': {'distributed': False, 'eval_frequency': 1000, 'log_frequency': 1000}
        }
        env = make_env(spec)
        env.reset()

        # Run one episode
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            done = term or trunc

        assert env.total_reward > 0
        env.close()


# ============================================================================
# Atari-specific tests (require ale_py)
# ============================================================================

class TestAtariRewardTracking:
    """Tests for Atari-specific reward tracking."""

    @pytest.fixture(autouse=True)
    def check_atari_available(self):
        """Skip tests if ale_py not installed."""
        try:
            import ale_py
            gym.register_envs(ale_py)
        except ImportError:
            pytest.skip("ale_py not installed")

    def test_episodic_life_creates_per_life_episodes(self):
        """Verify episodic_life=true creates episodes on each life loss."""
        env = gym.make_vec(
            "ALE/Breakout-v5",
            num_envs=2,
            vectorization_mode="vector_entry_point",
            episodic_life=True
        )
        env = VectorRecordEpisodeStatistics(env)
        obs, info = env.reset()

        assert "lives" in info
        assert info["lives"][0] == 5  # Breakout starts with 5 lives

        # Run until we get episode completions (life losses)
        initial_queue_len = len(env.return_queue)
        for _ in range(500):
            # AtariVectorEnv expects 1D action array
            actions = np.array([env.single_action_space.sample(), env.single_action_space.sample()])
            env.step(actions)
            if len(env.return_queue) > initial_queue_len:
                break

        # Episodes should complete on life loss, not just game over
        assert len(env.return_queue) > initial_queue_len
        env.close()

    def test_life_loss_info_provides_lives(self):
        """Verify life_loss_info=true provides lives in info."""
        env = gym.make_vec(
            "ALE/Breakout-v5",
            num_envs=2,
            vectorization_mode="vector_entry_point",
            life_loss_info=True
        )
        env = VectorRecordEpisodeStatistics(env)
        _, info = env.reset()

        assert "lives" in info
        assert info["lives"][0] == 5
        env.close()

    def test_pong_has_no_lives(self):
        """Verify Pong reports lives=0 (no lives concept)."""
        env = gym.make_vec(
            "ALE/Pong-v5",
            num_envs=1,
            vectorization_mode="vector_entry_point",
            episodic_life=True
        )
        env = VectorRecordEpisodeStatistics(env)
        _, info = env.reset()

        assert "lives" in info
        assert info["lives"][0] == 0  # Pong has no lives
        env.close()

    def test_make_env_with_episodic_life(self):
        """Verify make_env works with episodic_life=true."""
        spec = {
            'agent': {'algorithm': {'gamma': 0.99}},
            'env': {
                'name': 'ALE/Breakout-v5',
                'num_envs': 2,
                'max_frame': 1e6,
                'episodic_life': True
            },
            'meta': {'distributed': False, 'eval_frequency': 1000, 'log_frequency': 1000}
        }
        env = make_env(spec)
        obs, info = env.reset()

        assert obs.shape == (2, 4, 84, 84)  # 2 envs, 4 frames, 84x84
        assert "lives" in info
        env.close()


# ============================================================================
# Regression tests for known bugs
# ============================================================================

class TestRegressions:
    """Regression tests for previously discovered bugs."""

    def test_no_infinite_reward_accumulation(self):
        """Regression: rewards must not accumulate infinitely.

        Bug: VectorTrackReward with episodic_life=true caused rewards to
        accumulate forever because lives never reached 0 (env resets on life loss).

        Fix: Use gymnasium's RecordEpisodeStatistics which correctly handles
        episode boundaries regardless of the episodic_life setting.
        """
        try:
            import ale_py
            gym.register_envs(ale_py)
        except ImportError:
            pytest.skip("ale_py not installed")

        spec = {
            'agent': {'algorithm': {'gamma': 0.99}},
            'env': {
                'name': 'ALE/Breakout-v5',
                'num_envs': 2,
                'max_frame': 1e6,
                'episodic_life': True
            },
            'meta': {'distributed': False, 'eval_frequency': 1000, 'log_frequency': 1000}
        }
        env = make_env(spec)
        env.reset()

        # Run for a while and collect rewards
        rewards_over_time = []
        for step in range(300):
            env.step(np.array([env.action_space.sample() for _ in range(2)]))
            if not np.isnan(env.total_reward):
                rewards_over_time.append(env.total_reward)

        env.close()

        # With the bug, rewards would grow unboundedly
        # With the fix, per-life rewards should be small (random policy)
        if len(rewards_over_time) > 0:
            max_reward = max(rewards_over_time)
            # Random Breakout policy gets maybe 1-5 points per life max
            # If rewards accumulated infinitely, we'd see 1000+
            assert max_reward < 100, f"Reward {max_reward} suspiciously high - possible accumulation bug"

    def test_wrapper_chain_traversal_with_extra_wrappers(self):
        """Regression: total_reward must work with extra wrappers in chain.

        Verify ClockMixin.total_reward correctly traverses wrapper chain
        even when normalize_obs, clip_obs etc. add extra wrappers.
        """
        spec = {
            'agent': {'algorithm': {'gamma': 0.99}},
            'env': {
                'name': 'CartPole-v1',
                'num_envs': 2,
                'max_frame': 1e6,
                'normalize_obs': True,
                'clip_obs': 10.0
            },
            'meta': {'distributed': False, 'eval_frequency': 1000, 'log_frequency': 1000}
        }
        env = make_env(spec)
        env.reset()

        # Run until episode completes
        for _ in range(500):
            env.step(np.array([env.action_space.sample() for _ in range(2)]))
            if not np.isnan(env.total_reward):
                break

        # Should still be able to get total_reward through wrapper chain
        assert not np.isnan(env.total_reward)
        env.close()
