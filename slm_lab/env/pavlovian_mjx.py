"""Native MuJoCo Playground environment for Pavlovian conditioning.

TC-01 through TC-10 tasks in a 10x10 arena with three landmark objects.
Pure JAX kinematics (MuJoCo model retained for rendering only).
Action space: 2-DOF (forward, turn).
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from slm_lab.env.pavlovian_mjx_tasks import TASK_FNS, TASK_METRIC_KEYS

_ARENA_XML = """
<mujoco model="pavlovian_arena">
  <option timestep="0.0333" iterations="1" ls_iterations="1">
    <flag contact="disable" gravity="disable"/>
  </option>
  <worldbody>
    <light pos="5 5 10"/>
    <camera name="top_down" pos="5 5 12" quat="1 0 0 0"/>
    <geom type="plane" pos="5 5 0" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="agent" pos="5 5 0.25">
      <joint name="slide_x" type="slide" axis="1 0 0" limited="true" range="0.25 9.75"/>
      <joint name="slide_y" type="slide" axis="0 1 0" limited="true" range="0.25 9.75"/>
      <joint name="hinge_z" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.25 0.1" rgba="0.5 0.5 0.5 1" mass="1"/>
    </body>
    <body name="red_sphere" mocap="true" pos="7 7 0.15">
      <geom type="sphere" size="0.15" rgba="1 0.2 0.2 1" contype="0" conaffinity="0"/>
    </body>
    <body name="blue_cube" mocap="true" pos="3 7 0.15">
      <geom type="box" size="0.15 0.15 0.15" rgba="0.2 0.2 1 1" contype="0" conaffinity="0"/>
    </body>
    <body name="green_cyl" mocap="true" pos="5 3 0.15">
      <geom type="cylinder" size="0.15 0.15" rgba="0.2 0.8 0.2 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

# Base mocap positions for the three objects [red, blue, green]
_OBJ_BASE_POS = jp.array([
    [7.0, 7.0, 0.15],
    [3.0, 7.0, 0.15],
    [5.0, 3.0, 0.15],
])

_MAX_STEPS = 1000
_INIT_ENERGY = 100.0
_ENERGY_DECAY = 0.1
_FORWARD_COST = 0.01
_ANGULAR_COST = 0.005


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.0333,
        sim_dt=0.0333,
        episode_length=_MAX_STEPS,
        action_repeat=1,
        impl="warp",
        naconmax=0,
        njmax=5,
    )


class PavlovianMjxEnv(mjx_env.MjxEnv):
    """MJX Pavlovian arena. Action: [forward (0-1), turn (-1 to 1)]."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        task_id: int = 7,
    ):
        super().__init__(config, config_overrides)
        self._task_id = task_id
        self._xml_path_str = "pavlovian_arena"
        self._mj_model = mujoco.MjModel.from_xml_string(_ARENA_XML)
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()

    def _post_init(self) -> None:
        self._qpos_x = 0
        self._qpos_y = 1
        self._qpos_heading = 2

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng_angle, rng_radius, rng_heading, rng_obj = jax.random.split(rng, 5)

        # Agent: uniform within 1.5m disc centered at (5, 5)
        angle = jax.random.uniform(rng_angle, (), minval=-jp.pi, maxval=jp.pi)
        radius = 1.5 * jp.sqrt(jax.random.uniform(rng_radius, ()))
        agent_x = 5.0 + radius * jp.cos(angle)
        agent_y = 5.0 + radius * jp.sin(angle)
        heading = jax.random.uniform(rng_heading, (), minval=-jp.pi, maxval=jp.pi)

        qpos = jp.array([agent_x, agent_y, heading])
        qvel = jp.zeros(3)

        # Objects: base positions + ±0.5m jitter in x,y
        jitter = jax.random.uniform(rng_obj, (3, 2), minval=-0.5, maxval=0.5)
        obj_xy = _OBJ_BASE_POS[:, :2] + jitter
        obj_pos = jp.concatenate([obj_xy, _OBJ_BASE_POS[:, 2:3]], axis=1)
        mocap_pos = obj_pos

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            mocap_pos=mocap_pos,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        # Compute initial distance to red sphere for delta-based rewards
        dx = obj_pos[0, 0] - agent_x
        dy = obj_pos[0, 1] - agent_y
        prev_dist_red = jp.sqrt(dx**2 + dy**2)

        info = {
            "rng": rng,
            "step_count": jp.float32(0),
            "energy": jp.float32(_INIT_ENERGY),
            "cs_signal": jp.float32(0.0),
            "prev_dist_red": prev_dist_red,
            "obj_pos": obj_pos,
            "trial_step": jp.int32(0),
            "trial_count": jp.int32(0),
            "phase": jp.int32(0),
            "chain_step": jp.int32(0),
            "reward_gate": jp.float32(1.0),
        }

        metrics = {"reward/task": jp.zeros(())}
        for key in TASK_METRIC_KEYS[self._task_id - 1]:
            metrics[key] = jp.zeros(())
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        heading = state.data.qpos[self._qpos_heading]
        dt = self._config.ctrl_dt

        # Map 2-DOF action → kinematics (pure JAX, no mjx.step)
        fwd = jp.clip(action[0], 0.0, 1.0)
        omega = action[1] * (jp.pi / 2)
        vx = fwd * jp.cos(heading)
        vy = fwd * jp.sin(heading)

        # Integrate and clamp to arena bounds
        new_x = jp.clip(state.data.qpos[self._qpos_x] + vx * dt, 0.25, 9.75)
        new_y = jp.clip(state.data.qpos[self._qpos_y] + vy * dt, 0.25, 9.75)
        new_heading = heading + omega * dt

        new_qpos = jp.array([new_x, new_y, new_heading])
        new_qvel = jp.array([vx, vy, omega])
        data = state.data.replace(qpos=new_qpos, qvel=new_qvel)

        # Base energy decay before reward function
        omega_abs = jp.abs(omega)
        energy = state.info["energy"] - _ENERGY_DECAY - fwd * _FORWARD_COST - omega_abs * _ANGULAR_COST
        step_count = state.info["step_count"] + 1

        info = {**state.info, "energy": energy, "step_count": step_count}

        # Task-specific reward (may further modify info["energy"])
        reward_fn = TASK_FNS[self._task_id - 1]
        reward, info, metrics = reward_fn(data, action, info, dict(state.metrics))

        metrics["reward/task"] = reward

        done = jp.where(
            (info["energy"] <= 0.0) | (step_count >= _MAX_STEPS),
            jp.float32(1.0),
            jp.float32(0.0),
        )

        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        x = data.qpos[self._qpos_x]
        y = data.qpos[self._qpos_y]
        heading = data.qpos[self._qpos_heading]
        vx = data.qvel[0]
        vy = data.qvel[1]
        omega = data.qvel[2]
        energy = info["energy"]
        step_count = info["step_count"]

        obs_parts = [
            (x - 5.0) / 5.0,
            (y - 5.0) / 5.0,
            jp.arctan2(jp.sin(heading), jp.cos(heading)) / jp.pi,
            vx,
            vy,
            omega / (jp.pi / 2),
            (energy - 50.0) / 50.0,
            2.0 * step_count / _MAX_STEPS - 1.0,
        ]

        obj_pos = info["obj_pos"]
        for i in range(3):
            ox = obj_pos[i, 0]
            oy = obj_pos[i, 1]
            dx = ox - x
            dy = oy - y
            dist = jp.sqrt(dx**2 + dy**2)
            ego_angle = jp.arctan2(dy, dx) - heading
            ego_angle = jp.arctan2(jp.sin(ego_angle), jp.cos(ego_angle))
            obs_parts.extend([
                dist / 15.0,
                ego_angle / jp.pi,
                jp.float32(1.0),
            ])

        obs_parts.append(info["cs_signal"])

        return jp.array(obs_parts)

    @property
    def xml_path(self) -> str:
        return self._xml_path_str

    @property
    def action_size(self) -> int:
        return 2

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
