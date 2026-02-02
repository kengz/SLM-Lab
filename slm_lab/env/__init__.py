# The environment module
import os
from typing import Any

import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import VectorEnv
import numpy as np
from gymnasium.wrappers import (
    AtariPreprocessing,
    ClipReward,
    FrameStackObservation,
    NormalizeObservation,
    NormalizeReward,
    RescaleAction,
)
from gymnasium.wrappers.vector import (
    ClipReward as VectorClipReward,
    NormalizeObservation as VectorNormalizeObservation,
    NormalizeReward as VectorNormalizeReward,
    RecordEpisodeStatistics as VectorRecordEpisodeStatistics,
    RescaleAction as VectorRescaleAction,
)

from slm_lab.env.wrappers import (
    ClipObservation,
    ClockWrapper,
    TrackReward,
    VectorClipObservation,
    VectorClockWrapper,
    VectorFullGameStatistics,
    VectorRenderAll,
)
from slm_lab.lib import logger, util
from slm_lab.lib.env_var import render

Clock = ClockWrapper  # Backward compatibility alias

# Register ALE environments on import
try:
    import ale_py
    os.environ.setdefault("ALE_PY_SILENCE", "1")
    gym.register_envs(ale_py)
except ImportError:
    pass

logger = logger.get_logger(__name__)

# Keys handled by make_env, not passed to gym.make
RESERVED_KEYS = {"name", "num_envs", "max_t", "max_frame", "normalize_obs", "normalize_reward", "clip_obs", "clip_reward"}


def _needs_action_rescaling(env: gym.Env) -> bool:
    """Check if action space needs rescaling to [-1, 1]."""
    action_space = getattr(env, 'single_action_space', env.action_space)
    if not isinstance(action_space, spaces.Box):
        return False
    return not (
        np.allclose(action_space.low, -1.0, atol=1e-6) and
        np.allclose(action_space.high, 1.0, atol=1e-6)
    )


def _get_vectorization_mode(name: str, num_envs: int, is_rendering: bool = False) -> str:
    """Select vectorization mode based on environment type."""
    entry_point = gym.envs.registry[name].entry_point.lower()

    # ALE: use AtariVectorEnv for speed, but sync mode when rendering
    # (AtariVectorEnv.render() not implemented)
    if "ale_py" in entry_point:
        return "sync" if is_rendering else "vector_entry_point"

    # Complex envs benefit from async parallelization
    is_simple = "classic_control" in entry_point or "box2d" in entry_point
    return "sync" if is_simple or num_envs < 8 else "async"


def _set_env_attributes(env: gym.Env, spec: dict[str, Any]) -> None:
    """Set SLM-Lab environment attributes."""
    env_spec = spec["env"]

    # Determine if vector env based on actual type, not spec
    env.is_venv = isinstance(env, VectorEnv)
    if not hasattr(env, "num_envs"):
        env.num_envs = env.num_envs if env.is_venv else 1
    util.set_attr(env, env_spec, ["name", "max_t", "max_frame"])

    # Logging config
    defaults = dict(eval_frequency=10000, log_frequency=10000)
    util.set_attr(env, defaults)
    util.set_attr(env, spec["meta"], ["eval_frequency", "log_frequency"])

    # Canonical spaces for consistent access
    if isinstance(env, VectorEnv):
        env.observation_space = env.single_observation_space
        env.action_space = env.single_action_space

    # State dimension
    obs_space = env.observation_space
    if isinstance(obs_space, spaces.Box):
        env.state_dim = obs_space.shape[0] if len(obs_space.shape) == 1 else obs_space.shape
    else:
        env.state_dim = getattr(obs_space, "n", obs_space.shape)

    # Action properties
    action_space = env.action_space
    if isinstance(action_space, spaces.Discrete):
        env.action_dim = action_space.n
        env.is_discrete = True
        env.is_multi = False
    elif isinstance(action_space, spaces.Box):
        env.action_dim = action_space.shape[0] if len(action_space.shape) == 1 else action_space.shape
        env.is_discrete = False
        env.is_multi = len(action_space.shape) > 1 or action_space.shape[0] > 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        env.action_dim = action_space.nvec.tolist()
        env.is_discrete = True
        env.is_multi = True
    elif isinstance(action_space, spaces.MultiBinary):
        env.action_dim = action_space.n
        env.is_discrete = True
        env.is_multi = True
    else:
        raise NotImplementedError(f"Action space {type(action_space)} not supported")

    # Timing config
    env.max_t = env_spec.get("max_t") or getattr(env.spec, "max_episode_steps", None) or 108000
    if spec["meta"]["distributed"] is not False:
        env.max_frame = int(env.max_frame / spec["meta"]["max_session"])
    env.done = False


def make_env(spec: dict[str, Any]) -> gym.Env:
    """Create a gymnasium environment.

    Gymnasium defaults are sensible - only override what's needed.
    For Atari (ALE/*), AtariVectorEnv handles all preprocessing natively.
    """
    env_spec = spec["env"]
    name = env_spec["name"]
    num_envs = env_spec.get("num_envs", 1)
    is_atari = name.startswith("ALE/")
    render_mode = "human" if render() else None

    # Pass through env kwargs (life_loss_info, repeat_action_probability, etc.)
    make_kwargs = {k: v for k, v in env_spec.items() if k not in RESERVED_KEYS}

    # Normalization options (for MuJoCo/continuous control)
    normalize_obs = env_spec.get("normalize_obs", False)
    normalize_reward = env_spec.get("normalize_reward", False)
    clip_obs = env_spec.get("clip_obs", 10.0 if normalize_obs else None)
    clip_reward = env_spec.get("clip_reward", 10.0 if normalize_reward else None)
    gamma = spec.get("agent", {}).get("algorithm", {}).get("gamma", 0.99)

    if num_envs > 1:
        env = _make_vector_env(name, num_envs, is_atari, render_mode, make_kwargs,
                               normalize_obs, normalize_reward, clip_obs, clip_reward, gamma)
    else:
        env = _make_single_env(name, is_atari, render_mode, make_kwargs,
                               normalize_obs, normalize_reward, clip_obs, clip_reward, gamma)

    _set_env_attributes(env, spec)
    ClockWrapperClass = VectorClockWrapper if env.is_venv else ClockWrapper
    return ClockWrapperClass(env, env.max_frame)


def _make_vector_env(name: str, num_envs: int, is_atari: bool, render_mode: str | None,
                     make_kwargs: dict, normalize_obs: bool, normalize_reward: bool,
                     clip_obs: float | None, clip_reward: float | None, gamma: float) -> gym.Env:
    """Create vector environment."""
    is_rendering = bool(render_mode)
    vectorization_mode = _get_vectorization_mode(name, num_envs, is_rendering)
    per_env_wrappers = None

    if is_atari:
        if vectorization_mode == "vector_entry_point":
            # AtariVectorEnv: native preprocessing, disable internal reward clipping
            make_kwargs["reward_clipping"] = False
            logger.info(f"AtariVectorEnv: {num_envs} envs, native preprocessing")
        else:
            # Sync mode for rendering - match AtariVectorEnv preprocessing
            make_kwargs.pop("life_loss_info", None)
            make_kwargs.pop("reward_clipping", None)
            make_kwargs["render_mode"] = "rgb_array"
            def preprocess(env):
                return FrameStackObservation(
                    AtariPreprocessing(env, frame_skip=1), stack_size=4, padding_type="zero"
                )
            per_env_wrappers = [preprocess]
            logger.info(f"Atari sync: {num_envs} envs with preprocessing wrappers")
    else:
        make_kwargs["render_mode"] = "rgb_array" if render_mode else None

    env = gym.make_vec(name, num_envs=num_envs, vectorization_mode=vectorization_mode,
                       wrappers=per_env_wrappers, **make_kwargs)

    if _needs_action_rescaling(env):
        action_space = env.single_action_space
        logger.info(f"Action rescaling: [{action_space.low.min():.1f}, {action_space.high.max():.1f}] → [-1, 1]")
        env = VectorRescaleAction(env, min_action=-1.0, max_action=1.0)

    env = VectorRecordEpisodeStatistics(env)
    if is_atari:
        env = VectorFullGameStatistics(env)  # Track full-game scores across life losses

    if normalize_obs:
        env = VectorNormalizeObservation(env)
    if clip_obs is not None:
        env = VectorClipObservation(env, bound=float(clip_obs))
    if normalize_reward:
        env = VectorNormalizeReward(env, gamma=gamma)
    if is_atari:
        env = VectorClipReward(env, min_reward=-1.0, max_reward=1.0)
    elif clip_reward is not None:
        if isinstance(clip_reward, (int, float)):
            env = VectorClipReward(env, min_reward=-clip_reward, max_reward=clip_reward)
        else:
            env = VectorClipReward(env, min_reward=clip_reward[0], max_reward=clip_reward[1])

    if render_mode:
        env = VectorRenderAll(env)

    return env


def _make_single_env(name: str, is_atari: bool, render_mode: str | None,
                     make_kwargs: dict, normalize_obs: bool, normalize_reward: bool,
                     clip_obs: float | None, clip_reward: float | None, gamma: float) -> gym.Env:
    """Create single environment."""
    if is_atari:
        make_kwargs.pop("life_loss_info", None)
        make_kwargs.pop("reward_clipping", None)

    make_kwargs["render_mode"] = render_mode
    env = gym.make(name, **make_kwargs)

    # Match AtariVectorEnv preprocessing
    if is_atari:
        env = FrameStackObservation(
            AtariPreprocessing(env, frame_skip=1), stack_size=4, padding_type="zero"
        )

    if _needs_action_rescaling(env):
        action_space = env.action_space
        logger.info(f"Action rescaling: [{action_space.low.min():.1f}, {action_space.high.max():.1f}] → [-1, 1]")
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)

    env = TrackReward(env)

    if normalize_obs:
        env = NormalizeObservation(env)
    if clip_obs is not None:
        env = ClipObservation(env, bound=float(clip_obs))
    if normalize_reward:
        env = NormalizeReward(env, gamma=gamma)
    if clip_reward is not None:
        if isinstance(clip_reward, (int, float)):
            env = ClipReward(env, min_reward=-clip_reward, max_reward=clip_reward)
        else:
            env = ClipReward(env, min_reward=clip_reward[0], max_reward=clip_reward[1])

    return env
