"""MuJoCo Playground environment wrapper for SLM-Lab.

Wraps MuJoCo Playground (JAX-based MJX) environments as gymnasium VectorEnv,
enabling use with SLM-Lab's training loop. BraxAutoResetWrapper handles
batched step/reset internally; arrays are converted to numpy at the boundary.
"""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

try:
    from mujoco_playground import registry as pg_registry
    from mujoco_playground import wrapper as pg_wrapper
except ImportError:
    raise ImportError(
        "MuJoCo Playground is required for playground environments. "
        "Install with: uv sync --group playground"
    )

# Use MJWarp (Warp-accelerated MJX) on CUDA GPUs for ~3-5x faster simulation.
# Falls back to standard JAX/MJX on CPU.
_has_cuda = any(d.platform == "gpu" for d in jax.devices())
_impl = "warp" if _has_cuda else "jax"
_config_overrides = {"impl": _impl}


class PlaygroundVecEnv(gym.vector.VectorEnv):
    """Vectorized wrapper for MuJoCo Playground environments.

    BraxAutoResetWrapper handles batched execution internally.
    Converts JAX arrays to numpy at the API boundary for
    compatibility with SLM-Lab's PyTorch training loop.
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
        self._base_env = pg_registry.load(env_name, config_overrides=_config_overrides)  # kept for rendering
        base_env = self._base_env
        self._env = pg_wrapper.wrap_for_brax_training(
            base_env, episode_length=episode_length, action_repeat=1
        )

        # Build observation and action spaces
        obs_size = base_env.observation_size
        if isinstance(obs_size, dict):
            # Dict observations are flattened in _get_obs — compute total dim
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

            t = torch.from_dlpack(jax.dlpack.to_dlpack(x))
            # If JAX is on CPU but device is cuda, move explicitly (CPU->GPU copy)
            return t if t.is_cuda else t.to(self._device)
        return np.asarray(x).astype(np.float32)

    def _get_obs(self, state):
        obs = state.obs
        if isinstance(obs, dict):
            # Flatten dict observations by concatenating values
            arrays = [obs[k] for k in sorted(obs.keys())]
            obs = jnp.concatenate(arrays, axis=-1)
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

        obs = self._get_obs(self._state)
        # Rewards, dones, info always numpy (used for control flow and memory)
        rewards = np.asarray(self._state.reward).astype(np.float32)
        dones = np.asarray(self._state.done).astype(bool)

        # State may have a separate .truncation field (time limit, etc.)
        truncation = getattr(self._state, "truncation", None)
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
