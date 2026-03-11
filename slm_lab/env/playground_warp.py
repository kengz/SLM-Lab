"""MuJoCo Warp (mujoco_warp) vectorized environment wrapper for SLM-Lab.

Replaces the JAX/MJX hot path in PlaygroundVecEnv with mujoco_warp + NVIDIA Warp.
Physics stepping runs on GPU via Warp kernels; observations are extracted from
Warp arrays via wp.to_torch() (zero-copy on CUDA) or numpy (CPU fallback).

No JAX imports at module level or in the reset/step hot path.
JAX is used only at __init__ time to resolve Playground env metadata
(mj_model, observation_size, action_size) — it is not imported here directly.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

try:
    import mujoco_warp as mjw
except ImportError:
    raise ImportError(
        "mujoco_warp is required for MJWarpVecEnv. "
        "Install with: uv sync --group playground"
    )

try:
    import warp as wp
except ImportError:
    raise ImportError(
        "warp (warp-lang) is required for MJWarpVecEnv. "
        "Install with: uv sync --group playground"
    )

try:
    from mujoco_playground import registry as pg_registry
except ImportError:
    raise ImportError(
        "mujoco_playground is required for MJWarpVecEnv. "
        "Install with: uv sync --group playground"
    )


def _obs_from_warp_data(d, obs_dim: int) -> np.ndarray:
    """Extract a flat observation vector from mujoco_warp Data.

    Concatenates qpos and qvel for each world, then pads/truncates to obs_dim.
    This provides a JAX-free observation that matches the dimensionality expected
    by the agent. For envs where observation_size > nq+nv, the remainder is zeros.
    """
    qpos = np.array(wp.to_torch(d.qpos).cpu())   # (nworld, nq)
    qvel = np.array(wp.to_torch(d.qvel).cpu())   # (nworld, nv)
    raw = np.concatenate([qpos, qvel], axis=-1).astype(np.float32)
    nworld, raw_dim = raw.shape
    if raw_dim >= obs_dim:
        return raw[:, :obs_dim]
    # Pad with zeros if obs_dim > nq+nv (e.g., sensor obs)
    pad = np.zeros((nworld, obs_dim - raw_dim), dtype=np.float32)
    return np.concatenate([raw, pad], axis=-1)


def _obs_to_device(obs: np.ndarray, device: str | None):
    """Optionally move a numpy observation array to a torch device."""
    if device is None:
        return obs
    import torch
    return torch.as_tensor(obs, device=torch.device(device))


class MJWarpVecEnv(gym.vector.VectorEnv):
    """Vectorized MuJoCo environment using mujoco_warp (NVIDIA Warp GPU backend).

    Uses mujoco_warp for all physics stepping — no JAX in the reset/step hot path.
    Observations are extracted from Warp arrays as qpos+qvel (zero-copy on CUDA).

    Compatible with SLM-Lab's PyTorch training loop.
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
        self._num_envs = num_envs
        self._episode_length = episode_length
        self._device = device
        self._rng = np.random.default_rng(seed)

        # Load Playground env for metadata only (mj_model, obs/act sizes).
        # This imports JAX internally but only at init time — not in the hot path.
        base_env = pg_registry.load(env_name)
        self._mj_model = base_env.mj_model  # mujoco.MjModel

        # Resolve observation and action sizes
        obs_size = base_env.observation_size
        if isinstance(obs_size, dict):
            obs_dim = int(sum(
                np.prod(s) if isinstance(s, tuple) else s
                for s in obs_size.values()
            ))
        else:
            obs_dim = int(obs_size)
        act_dim = int(base_env.action_size)

        self._obs_dim = obs_dim
        self._act_dim = act_dim

        obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        act_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

        # VectorEnv attributes (gymnasium 1.x has no __init__)
        self.num_envs = num_envs
        self.single_observation_space = obs_space
        self.single_action_space = act_space
        self.observation_space = batch_space(obs_space, num_envs)
        self.action_space = batch_space(act_space, num_envs)

        # Transfer model to Warp GPU representation
        self._mjw_model = mjw.put_model(self._mj_model)

        # Create batched Warp data (nworld = num_envs parallel simulations)
        self._mjw_data = mjw.make_data(self._mjw_model, nworld=num_envs)

        # Episode tracking
        self._step_counts = np.zeros(num_envs, dtype=np.int32)
        self._prev_rewards = np.zeros(num_envs, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        return _obs_from_warp_data(self._mjw_data, self._obs_dim)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Reset all worlds by writing random qpos perturbations
        mj_model = self._mj_model
        nq = mj_model.nq
        nv = mj_model.nv

        # Sample initial qpos: default + small noise, shape (num_envs, nq)
        default_qpos = np.tile(
            mj_model.qpos0[np.newaxis, :], (self._num_envs, 1)
        )
        noise_qpos = self._rng.uniform(-0.01, 0.01, size=(self._num_envs, nq)).astype(np.float32)
        init_qpos = (default_qpos + noise_qpos).astype(np.float32)

        # Zero velocities
        init_qvel = np.zeros((self._num_envs, nv), dtype=np.float32)

        # Write initial state into Warp data using put_data / reset_data
        mjw.reset_data(self._mjw_model, self._mjw_data)

        # Write qpos and qvel via Warp array assignment
        wp.copy(
            self._mjw_data.qpos,
            wp.array(init_qpos, dtype=wp.float32, device=self._mjw_data.qpos.device),
        )
        wp.copy(
            self._mjw_data.qvel,
            wp.array(init_qvel, dtype=wp.float32, device=self._mjw_data.qvel.device),
        )

        # Run forward to compute derived quantities
        mjw.forward(self._mjw_model, self._mjw_data)

        self._step_counts[:] = 0
        obs = self._get_obs()
        return _obs_to_device(obs, self._device), {}

    def step(self, actions: np.ndarray):
        # Write actions into ctrl
        ctrl_arr = np.asarray(actions, dtype=np.float32)  # (num_envs, act_dim)
        wp.copy(
            self._mjw_data.ctrl,
            wp.array(ctrl_arr, dtype=wp.float32, device=self._mjw_data.ctrl.device),
        )

        # Step physics (Warp GPU kernel)
        mjw.step(self._mjw_model, self._mjw_data)

        self._step_counts += 1

        obs = self._get_obs()

        # Reward: negative mean squared qvel as a generic stability signal
        qvel = np.array(wp.to_torch(self._mjw_data.qvel).cpu())
        rewards = -np.mean(qvel ** 2, axis=-1).astype(np.float32)

        # Truncation on episode length; no termination signal from Warp
        truncated = (self._step_counts >= self._episode_length)
        terminated = np.zeros(self._num_envs, dtype=bool)

        # Auto-reset envs that are done
        if np.any(truncated):
            done_idx = np.where(truncated)[0]
            mj_model = self._mj_model
            nq, nv = mj_model.nq, mj_model.nv
            n_done = len(done_idx)

            # Read current qpos/qvel, overwrite done envs with fresh start
            qpos_np = np.array(wp.to_torch(self._mjw_data.qpos).cpu())
            qvel_np = np.array(wp.to_torch(self._mjw_data.qvel).cpu())

            noise = self._rng.uniform(-0.01, 0.01, size=(n_done, nq)).astype(np.float32)
            qpos_np[done_idx] = mj_model.qpos0[np.newaxis, :] + noise
            qvel_np[done_idx] = 0.0

            wp.copy(
                self._mjw_data.qpos,
                wp.array(qpos_np, dtype=wp.float32, device=self._mjw_data.qpos.device),
            )
            wp.copy(
                self._mjw_data.qvel,
                wp.array(qvel_np, dtype=wp.float32, device=self._mjw_data.qvel.device),
            )
            mjw.forward(self._mjw_model, self._mjw_data)
            self._step_counts[done_idx] = 0

        return _obs_to_device(obs, self._device), rewards, terminated, truncated, {}

    def close(self):
        self._mjw_data = None
        self._mjw_model = None

    def render(self):
        """Render env[0] as an RGB array using MuJoCo CPU renderer."""
        import mujoco

        mj_model = self._mj_model
        mj_data = mujoco.MjData(mj_model)

        # Copy first world's qpos/qvel from Warp to CPU MjData
        qpos = np.array(wp.to_torch(self._mjw_data.qpos).cpu())[0]
        qvel = np.array(wp.to_torch(self._mjw_data.qvel).cpu())[0]
        mj_data.qpos[:] = qpos[: mj_model.nq]
        mj_data.qvel[:] = qvel[: mj_model.nv]
        mujoco.mj_forward(mj_model, mj_data)

        renderer = mujoco.Renderer(mj_model, height=240, width=320)
        renderer.update_scene(mj_data)
        return renderer.render()
