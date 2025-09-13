"""
Gymnasium environment wrappers for SLM-Lab compatibility.
"""

import time
from typing import Any, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import pydash as ps

from slm_lab.lib import util


class TrackReward(gym.Wrapper):
    """Track cumulative reward for SLM-Lab compatibility"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.total_reward = 0.0
        self.episode_count = 0

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        # Reset total reward at episode start
        self.total_reward = 0.0
        state, info = self.env.reset(**kwargs)
        # Add total_reward to info for consistency
        info["total_reward"] = self.total_reward
        return state, info

    def step(
        self, action: Union[int, float, np.ndarray]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        state, reward, terminated, truncated, info = self.env.step(action)
        self.total_reward += reward

        # Always set total_reward for SLM-Lab compatibility
        info["total_reward"] = self.total_reward

        if terminated or truncated:
            info["episode_reward"] = self.total_reward
            info["episode_count"] = self.episode_count
            self.episode_count += 1

        return state, reward, terminated, truncated, info


class VectorTrackReward(gym.vector.VectorWrapper):
    """Track cumulative reward for vector environments"""

    def __init__(self, env: gym.vector.VectorEnv) -> None:
        super().__init__(env)
        self.total_rewards: np.ndarray = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_counts: np.ndarray = np.zeros(self.num_envs, dtype=int)

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        # Reset total rewards at episode start
        self.total_rewards.fill(0.0)
        observations, infos = self.env.reset(**kwargs)
        # Add total_reward to info for consistency
        if not isinstance(infos, dict):
            infos = {}
        infos["total_reward"] = self.total_rewards.copy()
        return observations, infos

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self.total_rewards += rewards

        # Ensure infos is a dict with total_reward for SLM-Lab compatibility
        if not isinstance(infos, dict):
            infos = {}
        infos["total_reward"] = self.total_rewards.copy()

        # Reset rewards for terminated/truncated environments and update episode counts
        dones = np.logical_or(terminations, truncations)
        self.episode_counts[dones] += 1
        self.total_rewards[dones] = 0.0

        return observations, rewards, terminations, truncations, infos


class ClockMixin:
    """Mixin class providing Clock timing functionality for environment wrappers"""

    def init_clock(self, max_frame: int = int(1e7)):
        """Initialize clock attributes"""
        self.max_frame = max_frame
        # num_envs already exists as env attribute
        self.reset_clock()

    def reset_clock(self) -> None:
        """Reset all clock counters"""
        self.t = 0
        self.frame = 0  # i.e. total_t
        self.epi = 0 if self.num_envs == 1 else None  # epi only for single envs
        self.start_wall_t = time.time()
        self.wall_t = 0
        self.batch_size = 1  # multiplier to accurately count opt steps
        self.opt_step = 0  # count the number of optimizer updates

    def load(self, train_df: pd.DataFrame) -> None:
        """Load clock from the last row of agent.mt.train_df"""
        last_row = train_df.iloc[-1]
        last_clock_vals = ps.pick(
            last_row, *["epi", "t", "wall_t", "opt_step", "frame"]
        )
        util.set_attr(self, last_clock_vals)
        self.start_wall_t -= self.wall_t  # offset elapsed wall_t

    def get(self, unit: str = "frame") -> int:
        """Get clock value for specified unit"""
        return getattr(self, unit)

    def get_elapsed_wall_t(self) -> int:
        """Calculate the elapsed wall time (int seconds) since self.start_wall_t"""
        return int(time.time() - self.start_wall_t)

    def set_batch_size(self, batch_size: int) -> None:
        """Set batch size for optimizer step counting"""
        self.batch_size = batch_size

    def tick_timestep(self) -> None:
        """Tick timestep - called automatically in step()"""
        self.t += 1  # timestep: invariant of num_envs
        self.frame += self.num_envs  # frame: exists by _set_env_attributes
        self.wall_t = self.get_elapsed_wall_t()

    def tick_episode(self) -> None:
        """Tick episode - called automatically in reset() when episode done (single env only)"""
        if self.t > 0:  # Only tick if we had a previous episode
            self.epi += 1
        self.t = 0

    def tick_opt_step(self) -> None:
        """Tick optimizer step - call this manually during training"""
        self.opt_step += self.batch_size

    @property
    def total_reward(self):
        """Delegate to TrackReward wrapper's total_reward"""
        # TrackReward is the immediate child wrapper
        if hasattr(self.env, "total_reward"):
            return self.env.total_reward  # Single env: TrackReward
        elif hasattr(self.env, "total_rewards"):
            return self.env.total_rewards  # Vector env: VectorTrackReward
        else:
            return np.nan


class ClockWrapper(ClockMixin, gym.Wrapper):
    """
    Environment wrapper that automatically handles Clock timing functionality.
    Eliminates the need for manual clock.tick() calls in the control loop.
    """

    def __init__(self, env: gym.Env, max_frame: int = int(1e7)):
        gym.Wrapper.__init__(self, env)
        self.init_clock(max_frame)

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment and handle episode clock logic"""
        self.tick_episode()
        return self.env.reset(**kwargs)

    def step(
        self, action
    ) -> tuple[
        np.ndarray,
        Union[float, np.ndarray],
        Union[bool, np.ndarray],
        Union[bool, np.ndarray],
        dict[str, Any],
    ]:
        """Step environment and automatically tick timestep"""
        result = self.env.step(action)
        self.tick_timestep()
        return result

    def __getattr__(self, name):
        """Pass through all attributes to the wrapped environment"""
        return getattr(self.env, name)


class VectorClockWrapper(ClockMixin, gym.vector.VectorWrapper):
    """
    Vector environment wrapper that automatically handles Clock timing functionality.
    """

    def __init__(self, env: gym.vector.VectorEnv, max_frame: int = int(1e7)):
        gym.vector.VectorWrapper.__init__(self, env)
        self.init_clock(max_frame)

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment - only called at initialization for vector envs"""
        # Note: Don't call tick_episode() because epi tracking not meaningful for vector envs
        return self.env.reset(**kwargs)

    def step(
        self, actions
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Step environment and automatically tick timestep"""
        result = self.env.step(actions)
        self.tick_timestep()
        return result

    def __getattr__(self, name):
        """Pass through all attributes to the wrapped environment"""
        return getattr(self.env, name)
