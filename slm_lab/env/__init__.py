# The environment module
import os
from typing import Any

import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import VectorEnv

from slm_lab.lib import logger, util
from slm_lab.env.wrappers import (
    ClockWrapper,
    TrackReward,
    VectorClockWrapper,
    VectorTrackReward,
)

# Alias ClockWrapper as Clock for backward compatibility
Clock = ClockWrapper

# Register ALE environments immediately on module import
try:
    import ale_py

    os.environ.setdefault("ALE_PY_SILENCE", "1")
    gym.register_envs(ale_py)
except ImportError:
    pass  # Silent fail - will error later if Atari envs are actually needed

logger = logger.get_logger(__name__)


def _get_vectorization_mode(name: str, num_envs: int) -> str | None:
    """Select sync/async/vector_entry_point vectorization based on environment complexity."""
    # Detect complex environments that benefit from async parallelization
    entry_point = gym.envs.registry[name].entry_point.lower()
    # specific value for ALE environments (required for automatic preprocessing)
    if "ale_py" in entry_point:
        return "vector_entry_point"

    is_complex = not ("classic_control" in entry_point or "box2d" in entry_point)
    # Thresholds based on benchmark data: only async for complex envs
    mode = "async" if (is_complex and num_envs >= 8) else "sync"
    complexity = "complex" if is_complex else "simple"
    logger.info(
        f"Using {mode} vectorization for {name} ({complexity} env, {num_envs} envs)"
    )
    return mode


def _set_logging_config(env: gym.Env, spec: dict[str, Any]) -> None:
    """Set logging and evaluation frequencies from spec"""
    defaults = dict(eval_frequency=10000, log_frequency=10000)
    util.set_attr(env, defaults)
    util.set_attr(env, spec["meta"], ["eval_frequency", "log_frequency"])


def _set_basic_attributes(env: gym.Env, spec: dict[str, Any]) -> None:
    """Set basic environment attributes from spec"""
    env_spec = spec["env"]

    # Set num_envs if not already set (vector envs already have this)
    if not hasattr(env, "num_envs"):
        env.num_envs = env_spec.get("num_envs", 1)

    # Set vectorization flag based on actual num_envs
    env.is_venv = env.num_envs > 1

    # Set basic attributes from env spec
    util.set_attr(env, env_spec, ["name", "max_t", "max_frame"])


def _set_canonical_spaces(env: gym.Env) -> None:
    """Set canonical observation/action spaces for consistent access"""
    if isinstance(env, VectorEnv):
        env.observation_space = env.single_observation_space
        env.action_space = env.single_action_space


def _extract_state_dim(env: gym.Env) -> None:
    """Extract state dimension from observation space"""
    obs_space = env.observation_space

    if isinstance(obs_space, spaces.Box):
        shape = obs_space.shape
        env.state_dim = shape[0] if len(shape) == 1 else shape
    else:
        env.state_dim = getattr(obs_space, "n", obs_space.shape)


def _extract_action_properties(env: gym.Env) -> None:
    """Extract action dimension and type properties from action space"""
    action_space = env.action_space

    if isinstance(action_space, spaces.Discrete):
        env.action_dim = action_space.n
        env.is_discrete = True
        env.is_multi = False
    elif isinstance(action_space, spaces.Box):
        shape = action_space.shape
        env.action_dim = shape[0] if len(shape) == 1 else shape
        env.is_discrete = False
        env.is_multi = len(shape) > 1 or shape[0] > 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        env.action_dim = action_space.nvec.tolist()
        env.is_discrete = True
        env.is_multi = True
    elif isinstance(action_space, spaces.MultiBinary):
        env.action_dim = action_space.n
        env.is_discrete = True
        env.is_multi = True
    else:
        raise NotImplementedError(
            f"Action space type {type(action_space)} not supported"
        )


def _set_timing_config(env: gym.Env, spec: dict[str, Any]) -> None:
    """Set timing and frame counting configuration"""
    # Set max_t from environment spec or defaults
    env.max_t = env.max_t or env.spec.max_episode_steps or 108000

    # Adjust max_frame for distributed training
    if spec["meta"]["distributed"] is not False:
        env.max_frame = int(env.max_frame / spec["meta"]["max_session"])

    # Clock speed accounts for vector environments
    env.clock_speed = env.num_envs

    # Initialize tracking attributes
    env.done = False


def _set_env_attributes(env: gym.Env, spec: dict[str, Any]) -> None:
    """Set environment attributes needed by SLM-Lab"""
    _set_logging_config(env, spec)
    _set_basic_attributes(env, spec)
    _set_canonical_spaces(env)
    _extract_state_dim(env)
    _extract_action_properties(env)
    _set_timing_config(env, spec)


def make_env(spec: dict[str, Any]) -> gym.Env:
    """Create a gymnasium environment with SLM-Lab compatibility"""
    env_spec = spec["env"]
    name = env_spec["name"]
    num_envs = env_spec.get("num_envs", 1)

    render_mode = "human" if util.to_render() else None

    if num_envs > 1:  # make vector environment
        vectorization_mode = _get_vectorization_mode(name, num_envs)
        # Note: For Atari, gymnasium's make_vec automatically includes FrameStackObservation + AtariPreprocessing
        env = gym.make_vec(
            name, num_envs=num_envs, vectorization_mode=vectorization_mode
        )
        # Add reward tracking for SLM-Lab compatibility
        env = VectorTrackReward(env)
    else:
        # Use gymnasium's standard make which handles all preprocessing automatically
        env = gym.make(name, render_mode=render_mode)
        # Add reward tracking for SLM-Lab compatibility
        env = TrackReward(env)

    # Set SLM-Lab attributes
    _set_env_attributes(env, spec)

    # Wrap with appropriate ClockWrapper for automatic timing
    # Use attributes set by _set_env_attributes instead of recomputing
    ClockWrapperClass = VectorClockWrapper if env.is_venv else ClockWrapper
    env = ClockWrapperClass(env, env.max_frame, env.clock_speed)

    return env
