"""MuJoCo Playground environment wrapper for SLM-Lab.

Wraps MuJoCo Playground (JAX/MJWarp) environments as gymnasium VectorEnv,
enabling use with SLM-Lab's training loop. BraxAutoResetWrapper handles
batched step/reset internally; arrays are converted to numpy at the boundary.

Uses MJWarp backend (Warp-accelerated MJX) uniformly for GPU simulation.
JAX is the dispatch/tracing layer; Warp CUDA kernels handle physics.
"""

import os
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

try:
    from mujoco_playground import registry as pg_registry
    from mujoco_playground import wrapper as pg_wrapper
    from mujoco_playground._src import mjx_env as _mjx_env_module
except ImportError:
    raise ImportError(
        "MuJoCo Playground is required for playground environments. "
        "Install with: uv sync --group playground"
    )

# Monkey-patch mjx_env.make_data to ensure naccdmax is set when missing.
# Some mujoco_warp versions default naccdmax=None to 0, causing CCD buffer
# overflow for envs with mesh/convex colliders. We resolve None to naconmax
# (the total active-contact buffer), which is always a safe upper bound.
_original_make_data = _mjx_env_module.make_data


def _patched_make_data(*args, **kwargs):
    naccdmax = kwargs.get("naccdmax")
    naconmax = kwargs.get("naconmax")
    if naccdmax is None and naconmax is not None:
        kwargs["naccdmax"] = naconmax
    return _original_make_data(*args, **kwargs)


_mjx_env_module.make_data = _patched_make_data

# Suppress MuJoCo C-level stderr warnings (ccd_iterations, nefc/broadphase overflow).
# These repeat every step for 100M frames, exploding log/output size on dstack.
# Suppressed permanently after first step — no per-call overhead or sync barriers.
_stderr_suppressed = False


# Per-env action_repeat from official dm_control_suite_params.py
# These match mujoco_playground's canonical training configs exactly.
_ACTION_REPEAT: dict[str, int] = {
    "PendulumSwingup": 4,
}

# SLM-native MjxEnv registry — bypasses mujoco_playground registry
_SLM_ENVS: dict[str, tuple[type, callable]] = {}


def register_slm_env(name: str, env_class: type, config_fn: callable) -> None:
    """Register an SLM-native MjxEnv for use with PlaygroundVecEnv."""
    _SLM_ENVS[name] = (env_class, config_fn)


def _build_config_overrides(env_name: str) -> dict:
    """Build config overrides for the given env.

    Sets impl='warp' for envs that support backend selection.
    When njmax is 0, sets None to trigger auto-detection via _default_njmax().
    """
    default_cfg = pg_registry.get_default_config(env_name)
    overrides = {"impl": "warp"} if hasattr(default_cfg, "impl") else {}
    njmax = getattr(default_cfg, "njmax", None)

    if njmax is not None and njmax == 0:
        overrides["njmax"] = None

    return overrides


class PlaygroundVecEnv(gym.vector.VectorEnv):
    """Vectorized wrapper for MuJoCo Playground environments.

    Uses MJWarp backend uniformly (impl='warp'). BraxAutoResetWrapper handles
    batched execution internally. Converts JAX arrays to numpy or torch tensors
    via DLPack at the API boundary for SLM-Lab's PyTorch training loop.
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        seed: int = 0,
        episode_length: int = 1000,
        device: str | None = None,
    ):
        self._env_name = env_name
        self._device = device
        if device is not None:
            import torch

            self._torch_device = torch.device(device)

        # Load the MJX environment and wrap for batched training
        # wrap_for_brax_training applies: VmapWrapper → EpisodeWrapper → BraxAutoResetWrapper
        # impl='warp' selects MJWarp (Warp-accelerated MJX) on CUDA; 'jax' on CPU
        if env_name in _SLM_ENVS:
            env_cls, config_fn = _SLM_ENVS[env_name]
            self._base_env = env_cls(config=config_fn())
        else:
            config_overrides = _build_config_overrides(env_name)
            self._base_env = pg_registry.load(
                env_name, config_overrides=config_overrides
            )  # kept for rendering
        base_env = self._base_env
        action_repeat = _ACTION_REPEAT.get(env_name, 1)
        self._env = pg_wrapper.wrap_for_brax_training(
            base_env, episode_length=episode_length, action_repeat=action_repeat
        )

        # Build observation and action spaces
        obs_size = base_env.observation_size
        if isinstance(obs_size, dict):
            if "state" in obs_size:
                # Use only "state" key — excludes privileged_state from actor input
                total_obs_dim = obs_size["state"] if not isinstance(obs_size["state"], tuple) else np.prod(obs_size["state"])
            else:
                total_obs_dim = sum(
                    np.prod(s) if isinstance(s, tuple) else s for s in obs_size.values()
                )
        else:
            total_obs_dim = obs_size
        act_size = base_env.action_size
        obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(int(total_obs_dim),), dtype=np.float32
        )
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(act_size,), dtype=np.float32)

        # Set VectorEnv attributes directly (gymnasium 1.x has no __init__)
        self.num_envs = num_envs
        self.single_observation_space = obs_space
        self.single_action_space = act_space
        self.observation_space = batch_space(obs_space, num_envs)
        self.action_space = batch_space(act_space, num_envs)

        # JIT-compile reset and step (BraxAutoResetWrapper handles batching internally)
        self._jit_reset = jax.jit(self._env.reset)
        self._jit_step = jax.jit(self._env.step)

        # Initialize RNG
        self._rng = jax.random.PRNGKey(seed)
        self._state = None

    def _to_output(self, x: jax.Array):
        """Convert JAX array to output format. DLPack zero-copy when JAX+PyTorch both on GPU."""
        if self._device is not None:
            import torch

            t = torch.from_dlpack(x)
            # If JAX is on CPU but device is cuda, move explicitly (CPU->GPU copy)
            return t if t.is_cuda else t.to(self._device)
        return np.asarray(x).astype(np.float32)

    def _get_obs(self, state):
        obs = state.obs
        if isinstance(obs, dict):
            # Use only "state" key when available — excludes privileged_state from actor
            obs = obs.get("state", jnp.concatenate([obs[k] for k in sorted(obs.keys())], axis=-1))
        return self._to_output(obs)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = jax.random.PRNGKey(seed)
        self._rng, *sub_keys = jax.random.split(self._rng, self.num_envs + 1)
        sub_keys = jnp.stack(sub_keys)
        self._state = self._jit_reset(sub_keys)
        obs = self._get_obs(self._state)
        return obs, {}

    def step(self, actions: np.ndarray):
        jax_actions = jnp.array(actions, dtype=jnp.float32)
        self._state = self._jit_step(self._state, jax_actions)
        # Suppress stderr permanently after first step — MuJoCo C warnings
        # repeat every step, but JAX async means we can't suppress per-call
        # without block_until_ready (which kills performance ~10x for slow envs).
        global _stderr_suppressed
        if not _stderr_suppressed:
            _stderr_suppressed = True
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
            os.close(devnull)

        obs = self._get_obs(self._state)
        # Rewards, dones, info always numpy (used for control flow and memory)
        rewards = np.asarray(self._state.reward).astype(np.float32)
        dones = np.asarray(self._state.done).astype(bool)

        # Brax EpisodeWrapper sets state.info['truncation'] (1 = time limit, 0 = not)
        truncation = self._state.info.get("truncation", None)
        if truncation is not None:
            truncated = np.asarray(truncation).astype(bool)
            terminated = dones & ~truncated
        else:
            terminated = dones
            truncated = np.zeros_like(dones, dtype=bool)

        # Extract metrics as info
        info = {}
        if self._state.metrics:
            for k, v in self._state.metrics.items():
                info[k] = np.asarray(v)

        return obs, rewards, terminated, truncated, info

    def close(self):
        self._state = None

    def render(self):
        """Render env[0] as an RGB array using MuJoCo renderer."""
        if self._state is None:
            return None
        # Extract first env's state from the batched pytree
        state_0 = jax.tree.map(lambda x: x[0], self._state)
        frames = self._base_env.render([state_0], height=240, width=320)
        return np.array(frames[0])
