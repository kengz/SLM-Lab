"""Gymnasium environment wrappers for SLM-Lab."""

import math
import time
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import pydash as ps

from slm_lab.lib import util


class TrackReward(gym.Wrapper):
    """Track episode rewards for single environments."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.ongoing_reward = 0.0
        self.total_reward = 0.0
        self.episode_count = 0

    def reset(self, **kwargs):
        self.ongoing_reward = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        self.ongoing_reward += reward
        if terminated or truncated:
            self.total_reward = self.ongoing_reward
            info["episode_reward"] = self.total_reward
            info["episode_count"] = self.episode_count
            self.episode_count += 1
        return state, reward, terminated, truncated, info


class ClockMixin:
    """Mixin for timing and frame counting."""

    def init_clock(self, max_frame: int = int(1e7)):
        self.max_frame = max_frame
        self.reset_clock()

    def reset_clock(self):
        self.t = 0
        self.frame = 0
        self.epi = 0 if self.num_envs == 1 else None
        self.start_wall_t = time.time()
        self.wall_t = 0
        self.batch_size = 1
        self.opt_step = 0

    def load(self, train_df: pd.DataFrame):
        """Load clock state from training dataframe."""
        last_row = train_df.iloc[-1]
        last_clock_vals = ps.pick(last_row, *["epi", "t", "wall_t", "opt_step", "frame"])
        util.set_attr(self, last_clock_vals)
        self.start_wall_t -= self.wall_t

    def get(self, unit: str = "frame") -> int:
        return getattr(self, unit)

    def get_elapsed_wall_t(self) -> int:
        return int(time.time() - self.start_wall_t)

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def tick_timestep(self):
        self.t += 1
        self.frame += self.num_envs
        self.wall_t = self.get_elapsed_wall_t()

    def tick_episode(self):
        if self.t > 0:
            self.epi += 1
        self.t = 0

    def tick_opt_step(self):
        self.opt_step += 1

    @property
    def total_reward(self):
        """Get last episode reward from tracking wrappers.

        Priority: VectorFullGameStatistics > RecordEpisodeStatistics > TrackReward
        This ensures we report full-game scores for Atari with life_loss_info.
        """
        from gymnasium.wrappers.vector import RecordEpisodeStatistics as VectorRecordEpisodeStatistics

        env = self.env
        while env is not None:
            if isinstance(env, TrackReward):
                return env.total_reward
            # Prefer full-game statistics for accurate Atari benchmarking
            if isinstance(env, VectorFullGameStatistics):
                if len(env.return_queue) > 0:
                    return np.mean(env.return_queue)
                # Fall through to check for RecordEpisodeStatistics
            if isinstance(env, VectorRecordEpisodeStatistics):
                # return_queue is a deque of recent episode returns (per-life for Atari)
                if len(env.return_queue) > 0:
                    return np.mean(list(env.return_queue))
                return np.nan
            env = getattr(env, "env", None)
        return np.nan


class ClockWrapper(ClockMixin, gym.Wrapper):
    """Automatic timing for single environments."""

    def __init__(self, env: gym.Env, max_frame: int = int(1e7)):
        gym.Wrapper.__init__(self, env)
        self.init_clock(max_frame)

    def reset(self, **kwargs):
        self.tick_episode()
        return self.env.reset(**kwargs)

    def step(self, action):
        result = self.env.step(action)
        self.tick_timestep()
        return result

    def __getattr__(self, name):
        return getattr(self.env, name)


class VectorClockWrapper(ClockMixin, gym.vector.VectorWrapper):
    """Automatic timing for vector environments."""

    def __init__(self, env: gym.vector.VectorEnv, max_frame: int = int(1e7)):
        gym.vector.VectorWrapper.__init__(self, env)
        self.init_clock(max_frame)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, actions):
        result = self.env.step(actions)
        self.tick_timestep()
        return result

    def __getattr__(self, name):
        return getattr(self.env, name)


class ClipObservation(gym.ObservationWrapper):
    """Clip observations to [-bound, bound]."""

    def __init__(self, env: gym.Env, bound: float = 10.0):
        super().__init__(env)
        self.bound = bound

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.clip(observation, -self.bound, self.bound)


class VectorClipObservation(gym.vector.VectorObservationWrapper):
    """Clip observations to [-bound, bound] for vector environments."""

    def __init__(self, env: gym.vector.VectorEnv, bound: float = 10.0):
        super().__init__(env)
        self.bound = bound

    def observations(self, observations: np.ndarray) -> np.ndarray:
        return np.clip(observations, -self.bound, self.bound)


class VectorFullGameStatistics(gym.vector.VectorWrapper):
    """Track full-game statistics for Atari with life_loss_info.

    With life_loss_info=true, RecordEpisodeStatistics records per-life scores.
    This wrapper tracks full-game scores by accumulating across life losses
    and only recording when lives==0 (true game over).
    """

    def __init__(self, env: gym.vector.VectorEnv, buffer_length: int = 100):
        super().__init__(env)
        self.buffer_length = buffer_length
        self.return_queue = []  # Full-game returns
        self._ongoing_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._prev_lives = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._ongoing_returns.fill(0.0)
        self._prev_lives = info.get("lives", np.zeros(self.num_envs))
        return obs, info

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(actions)

        # Accumulate raw rewards (note: rewards here are already clipped for training)
        # We use the clipped rewards since AtariVectorEnv doesn't expose raw rewards easily
        self._ongoing_returns += rewards

        lives = info.get("lives", np.zeros(self.num_envs))

        # Check for true game-over (lives dropped to 0)
        # Only record when we transition TO 0 lives (not when already at 0)
        if self._prev_lives is not None:
            game_over = (lives == 0) & (self._prev_lives > 0)
            for i in range(self.num_envs):
                if game_over[i]:
                    self.return_queue.append(self._ongoing_returns[i])
                    if len(self.return_queue) > self.buffer_length:
                        self.return_queue.pop(0)
                    self._ongoing_returns[i] = 0.0

        # Also reset on truncation (time limit)
        for i in range(self.num_envs):
            if truncated[i] and not terminated[i]:
                self.return_queue.append(self._ongoing_returns[i])
                if len(self.return_queue) > self.buffer_length:
                    self.return_queue.pop(0)
                self._ongoing_returns[i] = 0.0

        self._prev_lives = lives.copy()
        return obs, rewards, terminated, truncated, info


class VectorRenderAll(gym.vector.VectorWrapper):
    """Render all vector envs in a grid using pygame."""

    def __init__(self, env: gym.vector.VectorEnv, render_freq: int = 32):
        super().__init__(env)
        self.render_freq = render_freq
        self.step_count = 0
        self.window = None
        self.clock = None
        self.grid_cols = int(math.ceil(math.sqrt(self.num_envs)))
        self.grid_rows = int(math.ceil(self.num_envs / self.grid_cols))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, actions):
        result = self.env.step(actions)
        self.step_count += 1
        if self.step_count % self.render_freq == 0:
            self._render_grid()
        return result

    def _get_base_env(self):
        """Find base env with call() method."""
        env = self.env
        while hasattr(env, 'env'):
            if hasattr(env, 'call'):
                return env
            env = env.env
        return env

    def _render_grid(self):
        try:
            import pygame
        except ImportError:
            return

        base_env = self._get_base_env()
        frames = base_env.call("render") if hasattr(base_env, 'call') else None
        if frames is None or frames[0] is None:
            return

        if self.window is None:
            pygame.init()
            frame_h, frame_w = frames[0].shape[:2]
            self.window = pygame.display.set_mode((frame_w * self.grid_cols, frame_h * self.grid_rows))
            pygame.display.set_caption(f"Vector Env ({self.num_envs} envs)")
            self.clock = pygame.time.Clock()

        frame_h, frame_w = frames[0].shape[:2]
        surface = pygame.Surface((frame_w * self.grid_cols, frame_h * self.grid_rows))

        for i, frame in enumerate(frames):
            if frame is None:
                continue
            row, col = i // self.grid_cols, i % self.grid_cols
            frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            surface.blit(frame_surface, (col * frame_w, row * frame_h))

        self.window.blit(surface, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.window = None
                raise KeyboardInterrupt("Render window closed")

    def close(self):
        if self.window is not None:
            import pygame
            pygame.quit()
            self.window = None
        return super().close()
