from slm_lab.env.base import BaseEnv
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
from typing import Any, Optional, Union
import gymnasium as gym
import numpy as np
import os

# Register ALE environments immediately on module import
try:
    import ale_py
    os.environ.setdefault('ALE_PY_SILENCE', '1')
    gym.register_envs(ale_py)
except ImportError:
    pass  # Silent fail - will error later if Atari envs are actually needed


logger = logger.get_logger(__name__)


class TrackReward(gym.Wrapper):
    '''Track cumulative reward for SLM-Lab compatibility'''
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.total_reward = 0.0
        self.episode_count = 0
        
    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        # Reset total reward at episode start
        self.total_reward = 0.0
        state, info = self.env.reset(**kwargs)
        # Add total_reward to info for consistency
        info['total_reward'] = self.total_reward
        return state, info
        
    def step(self, action: Union[int, float, np.ndarray]) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        state, reward, terminated, truncated, info = self.env.step(action)
        self.total_reward += reward
        
        # Always set total_reward for SLM-Lab compatibility
        info['total_reward'] = self.total_reward
        
        if terminated or truncated:
            info['episode_reward'] = self.total_reward
            info['episode_count'] = self.episode_count
            self.episode_count += 1
            
        return state, reward, terminated, truncated, info


class VectorTrackReward(gym.vector.VectorWrapper):
    '''Track cumulative reward for vector environments'''
    
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
        infos['total_reward'] = self.total_rewards.copy()
        return observations, infos
        
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self.total_rewards += rewards
        
        # Ensure infos is a dict with total_reward for SLM-Lab compatibility
        if not isinstance(infos, dict):
            infos = {}
        infos['total_reward'] = self.total_rewards.copy()
        
        # Reset rewards for terminated/truncated environments and update episode counts
        dones = np.logical_or(terminations, truncations)
        self.episode_counts[dones] += 1
        self.total_rewards[dones] = 0.0
                
        return observations, rewards, terminations, truncations, infos


class GymEnv(BaseEnv):
    '''
    Wrapper for gymnasium environments to work with SLM-Lab.
    Uses gymnasium's optimal defaults for all preprocessing.

    e.g. env_spec
    "env": [{
        "name": "PongNoFrameskip-v4",
        "num_envs": 8,
        "vectorization_mode": "auto",  # "sync", "async", or "auto" (default)
        "max_t": null,
        "max_frame": 1e7
    }],
    '''

    def __init__(self, spec: dict[str, Any]) -> None:
        super().__init__(spec)

        render_mode = 'human' if util.to_render() else None

        if self.is_venv:  # make vector environment
            # Smart vectorization mode selection based on environment complexity and performance
            vectorization_mode = self._get_vectorization_mode()
            # Note: For Atari, gymnasium's make_vec automatically includes FrameStackObservation + AtariPreprocessing
            # See: https://ale.farama.org/vector-environment/
            # NOTE: render_mode is NOT a valid parameter for gym.make_vec - vector envs handle rendering differently
            self.u_env = gym.make_vec(self.name, num_envs=self.num_envs, vectorization_mode=vectorization_mode)
        else:
            # Use gymnasium's standard make which handles all preprocessing automatically
            # Note: For Atari, this includes frame stacking and preprocessing built-in
            self.u_env = gym.make(self.name, render_mode=render_mode)

        # Add reward tracking for SLM-Lab compatibility
        TrackRewardCls = VectorTrackReward if self.is_venv else TrackReward
        self.u_env = TrackRewardCls(self.u_env)

        self._set_attr_from_u_env(self.u_env)

        # Set max_t from environment spec
        self.max_t = self.max_t or self.u_env.spec.max_episode_steps or 108000

    def _get_vectorization_mode(self) -> str | None:
        """Select sync/async/vector_entry_point vectorization based on environment complexity and benchmark data."""
        # Manual override from spec
        manual_mode = getattr(self, 'vectorization_mode', 'auto')
        if manual_mode in ['sync', 'async']:
            logger.info(f'Using {manual_mode} vectorization for {self.name} (manual override)')
            return manual_mode
            
        # Detect complex environments that benefit from async parallelization
        entry_point = gym.envs.registry[self.name].entry_point.lower()
        # specific value for ALE environments (required for automatic preprocessing)
        if "ale_py" in entry_point:
            return "vector_entry_point"

        is_complex = not ("classic_control" in entry_point or "box2d" in entry_point)
        # Thresholds based on benchmark data: only async for complex envs
        mode = "async" if (is_complex and self.num_envs >= 8) else "sync"
        complexity = 'complex' if is_complex else 'simple'
        logger.info(f'Using {mode} vectorization for {self.name} ({complexity} env, {self.num_envs} envs)')
        return mode

    @lab_api
    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict[str, Any]]:
        state, info = self.u_env.reset(seed=seed)
        return state, info

    @lab_api
    def step(self, action: np.ndarray) -> tuple[np.ndarray, Union[float, np.ndarray], Union[bool, np.ndarray], Union[bool, np.ndarray], dict[str, Any]]:
        state, reward, term, trunc, info = self.u_env.step(action)
        self._update_total_reward(info)
        return state, reward, term, trunc, info

    @lab_api
    def close(self) -> None:
        self.u_env.close()
