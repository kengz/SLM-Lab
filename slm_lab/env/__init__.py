# The environment module
import os
from typing import Any

import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    NormalizeObservation,
    NormalizeReward,
)
from gymnasium.wrappers.vector import (
    NormalizeObservation as VectorNormalizeObservation,
    NormalizeReward as VectorNormalizeReward,
)

from slm_lab.env.wrappers import (
    ClockWrapper,
    TrackReward,
    VectorClockWrapper,
    VectorRenderAll,
    VectorTrackReward,
)
from slm_lab.lib import logger, util
from slm_lab.lib.env_var import render

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
    render_mode = "human" if render() else None

    # Build kwargs for gym.make() - pass through any extra env kwargs
    # Reserved keys that are handled separately by SLM-Lab, not passed to gym.make()
    reserved_keys = {"name", "num_envs", "max_t", "max_frame", "normalize_obs", "normalize_reward"}
    make_kwargs = {k: v for k, v in env_spec.items() if k not in reserved_keys}

    # Optional normalization (useful for MuJoCo and other continuous control envs)
    normalize_obs = env_spec.get("normalize_obs", False)
    normalize_reward = env_spec.get("normalize_reward", False)
    # Get gamma from agent spec for reward normalization (default 0.99)
    gamma = spec.get("agent", {}).get("algorithm", {}).get("gamma", 0.99)

    if num_envs > 1:  # make vector environment
        vectorization_mode = _get_vectorization_mode(name, num_envs)
        # For Atari vector envs, render_mode is not supported in kwargs
        if not name.startswith("ALE/"):
            make_kwargs["render_mode"] = "rgb_array" if render_mode else None

        # Note: For Atari, gymnasium's make_vec automatically includes FrameStackObservation + AtariPreprocessing
        env = gym.make_vec(
            name, num_envs=num_envs, vectorization_mode=vectorization_mode, **make_kwargs
        )
        # Add observation normalization first (normalizes inputs to policy network)
        if normalize_obs:
            env = VectorNormalizeObservation(env)
            logger.info("Observation normalization enabled")
        # Add reward normalization before tracking (so tracking sees normalized rewards)
        if normalize_reward:
            env = VectorNormalizeReward(env, gamma=gamma)
            logger.info(f"Reward normalization enabled (gamma={gamma})")
        # Add reward tracking for SLM-Lab compatibility
        env = VectorTrackReward(env)
        # Add grid rendering for all envs
        if render_mode:
            env = VectorRenderAll(env)
    else:
        make_kwargs["render_mode"] = render_mode
        env = gym.make(name, **make_kwargs)
        # gymnasium forgot to do this for single Atari env like in make_vec
        if name.startswith("ALE/"):
            env = FrameStackObservation(
                AtariPreprocessing(env, frame_skip=1), stack_size=4
            )
        # Add observation normalization first (normalizes inputs to policy network)
        if normalize_obs:
            env = NormalizeObservation(env)
            logger.info("Observation normalization enabled")
        # Add reward normalization before tracking
        if normalize_reward:
            env = NormalizeReward(env, gamma=gamma)
            logger.info(f"Reward normalization enabled (gamma={gamma})")
        # Add reward tracking for SLM-Lab compatibility
        env = TrackReward(env)

    # Set SLM-Lab attributes
    _set_env_attributes(env, spec)

    # Wrap with appropriate ClockWrapper for automatic timing
    # Use attributes set by _set_env_attributes instead of recomputing
    ClockWrapperClass = VectorClockWrapper if env.is_venv else ClockWrapper
    env = ClockWrapperClass(env, env.max_frame)

    return env
