"""Gymnasium environment wrappers for SLM-Lab."""

import math
import time
from collections import deque
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
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
        for key in ("epi", "t", "wall_t", "opt_step", "frame"):
            if key in last_row.index:
                setattr(self, key, last_row[key])
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
        from gymnasium.wrappers.vector import (
            RecordEpisodeStatistics as VectorRecordEpisodeStatistics,
        )

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


class ExtractDictObs(gym.ObservationWrapper):
    """Extract a single key from Dict observation space, exposing it as a flat Box."""

    def __init__(self, env: gym.Env, key: str = "ground_truth"):
        super().__init__(env)
        self._key = key
        self.observation_space = env.observation_space[key]

    def observation(self, observation: dict) -> np.ndarray:
        return observation[self._key]


class VectorExtractDictObs(gym.vector.VectorObservationWrapper):
    """Extract a single key from Dict observation space for vector environments."""

    def __init__(self, env: gym.vector.VectorEnv, key: str = "ground_truth"):
        super().__init__(env)
        self._key = key
        inner_space = env.single_observation_space[key]
        self.single_observation_space = inner_space
        self.observation_space = gym.vector.utils.batch_space(inner_space, env.num_envs)

    def observations(self, observations: dict) -> np.ndarray:
        return observations[self._key]


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
        self.return_queue = deque(maxlen=buffer_length)  # Full-game returns
        self._ongoing_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._prev_lives = None
        self._zero_lives = np.zeros(self.num_envs)  # pre-allocate fallback

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._ongoing_returns.fill(0.0)
        self._prev_lives = info.get("lives", self._zero_lives)
        return obs, info

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(actions)

        self._ongoing_returns += rewards

        lives = info.get("lives", self._zero_lives)

        # Check for true game-over (lives dropped to 0) — vectorized
        if self._prev_lives is not None:
            game_over = (lives == 0) & (self._prev_lives > 0)
            done_idxs = np.flatnonzero(game_over)
            for i in done_idxs:
                self.return_queue.append(self._ongoing_returns[i])
            self._ongoing_returns[done_idxs] = 0.0

        # Also reset on truncation (time limit) — vectorized
        trunc_only = truncated & ~terminated
        trunc_idxs = np.flatnonzero(trunc_only)
        for i in trunc_idxs:
            self.return_queue.append(self._ongoing_returns[i])
        self._ongoing_returns[trunc_idxs] = 0.0

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
        while hasattr(env, "env"):
            if hasattr(env, "call"):
                return env
            env = env.env
        return env

    def _render_grid(self):
        try:
            import pygame
        except ImportError:
            return

        base_env = self._get_base_env()
        frames = base_env.call("render") if hasattr(base_env, "call") else None
        if frames is None or frames[0] is None:
            return

        if self.window is None:
            pygame.init()
            frame_h, frame_w = frames[0].shape[:2]
            self.window = pygame.display.set_mode(
                (frame_w * self.grid_cols, frame_h * self.grid_rows)
            )
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


class PlaygroundRenderWrapper(gym.vector.VectorWrapper):
    """Render MuJoCo Playground env[0] via pygame after each step."""

    def __init__(self, env: gym.vector.VectorEnv, render_freq: int = 1):
        super().__init__(env)
        self.render_freq = render_freq
        self.step_count = 0
        self.window = None
        self.clock = None

    def step(self, actions):
        result = self.env.step(actions)
        self.step_count += 1
        if self.step_count % self.render_freq == 0:
            self._show()
        return result

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._show()
        return result

    def _show(self):
        try:
            import pygame
        except ImportError:
            return
        frame = self.env.render()
        if frame is None:
            return
        if self.window is None:
            pygame.init()
            h, w = frame.shape[:2]
            self.window = pygame.display.set_mode((w, h))
            pygame.display.set_caption("MuJoCo Playground")
            self.clock = pygame.time.Clock()
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        self.window.blit(surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise KeyboardInterrupt("Render window closed")

    def close(self):
        if self.window is not None:
            import pygame

            pygame.quit()
            self.window = None
        return super().close()


class TorchNormalizeObservation(gym.vector.VectorWrapper):
    """Running-mean normalization for CUDA tensor observations (Welford algorithm)."""

    def __init__(self, env: gym.vector.VectorEnv, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self._mean = None
        self._var = None
        self._count = 0

    def _update_and_normalize(self, obs):
        if self._mean is None:
            self._mean = torch.zeros_like(obs[0])
            self._var = torch.ones_like(obs[0])
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0, unbiased=False)
        batch_count = obs.shape[0]
        # Welford parallel update
        total = self._count + batch_count
        delta = batch_mean - self._mean
        self._mean = self._mean + delta * batch_count / total
        self._var = (
            self._var * self._count
            + batch_var * batch_count
            + delta**2 * self._count * batch_count / total
        ) / total
        self._count = total
        return (obs - self._mean) / (self._var + self.epsilon).sqrt()

    def step(self, actions):
        obs, *rest = self.env.step(actions)
        return self._update_and_normalize(obs), *rest

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._update_and_normalize(obs), info
