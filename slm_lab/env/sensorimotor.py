"""Sensorimotor stage MuJoCo environment for SLM-Lab.

3D tabletop with Fetch-style 7-DOF arm for TC-11 through TC-24.

Physics: MuJoCo Python bindings, 500 Hz internal / 25 Hz control.
Observation (Phase 3.2a): 56-dim ground-truth vector (see env-detailed.md §6.7).
Action: 10-dim continuous [-1, 1] — 7 joint targets + gripper + head pan/tilt.
Controller: PD position control, kp=100, kd=10.

Registered as SLM-Sensorimotor-TC{11..24}-v0 in slm_lab/env/__init__.py.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from loguru import logger

from slm_lab.env.sensorimotor_tasks import TASK_REGISTRY, VALID_TASK_IDS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Physics
PHYSICS_DT = 0.002        # 500 Hz
CONTROL_DT = 0.04         # 25 Hz
SUBSTEPS = int(CONTROL_DT / PHYSICS_DT)  # 20
assert SUBSTEPS == 20, "SUBSTEPS must equal 20"

# Arm joints (7-DOF)
JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "shoulder_roll",
    "elbow", "wrist_yaw", "wrist_pitch", "wrist_roll",
]
JOINT_RANGES = np.array([
    [-1.57, 1.57],   # shoulder_pan
    [-1.57, 1.57],   # shoulder_lift
    [-0.785, 0.785], # shoulder_roll
    [0.0, 2.094],    # elbow
    [-1.57, 1.57],   # wrist_yaw
    [-1.57, 1.57],   # wrist_pitch
    [-1.57, 1.57],   # wrist_roll
], dtype=np.float32)

JOINT_MAX_VEL = np.array([2.0] * 7, dtype=np.float32)
JOINT_MAX_TORQUE = np.array([87, 87, 87, 87, 12, 12, 12], dtype=np.float32)
JOINT_MID = (JOINT_RANGES[:, 0] + JOINT_RANGES[:, 1]) / 2.0
JOINT_HALF = (JOINT_RANGES[:, 1] - JOINT_RANGES[:, 0]) / 2.0

HOME_QPOS = np.array([0.0, -0.5, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
HOME_GRIPPER = 0.04   # half-open (qpos for finger_joint)
HOME_HEAD_PAN = 0.0
HOME_HEAD_TILT = -0.3

# PD controller gains
KP = 100.0
KD = 10.0
KP_GRIPPER = 200.0
KD_GRIPPER = 40.0
KP_HEAD = 20.0
KD_HEAD = 4.0

# Energy
MAX_ENERGY = 100.0
ENERGY_DECAY = 0.05  # per control step

# Obs
OBS_DIM = 56       # Phase 3.2a ground-truth, 3-object scene
N_OBJECTS_MAX = 3  # pads to this many objects

# Observation noise (angles, velocities, torques)
NOISE_ANGLE = 0.01   # rad
NOISE_VEL = 0.02     # rad/s
NOISE_TORQUE = 0.05  # Nm

TABLE_CENTER = np.array([2.5, 2.5, 0.75], dtype=np.float32)
MAX_EPISODE_STEPS = 500  # control steps per episode (25 Hz × 20 s)


# ---------------------------------------------------------------------------
# MJCF builder
# ---------------------------------------------------------------------------

def _build_mjcf(include_objects: list[str]) -> str:
    """Generate MJCF XML for the sensorimotor environment.

    Only includes objects listed in include_objects. Object body definitions
    always present in model but placed off-table at z=-1 when inactive.
    """
    # All task objects always in model; tasks position them at reset.
    return r"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="slm_sensorimotor">
  <option timestep="0.002" gravity="0 0 -9.81" cone="pyramidal"
          solver="Newton" iterations="50" integrator="Euler"/>

  <default>
    <default class="robot">
      <joint armature="1" damping="50" frictionloss="0"/>
      <geom margin="0.001" solimp="0.99 0.99 0.01" solref="0.01 1"
            friction="1 0.005 0.0001" contype="1" conaffinity="1"/>
    </default>
    <default class="gripper">
      <joint armature="100" damping="1000"/>
      <geom friction="1.0 0.01 0.0001" condim="4" contype="1" conaffinity="1"/>
    </default>
    <default class="object">
      <geom condim="3" solref="0.02 1" solimp="0.95 0.99 0.001 0.5 2"
            margin="0.001" contype="1" conaffinity="1"/>
    </default>
    <default class="table">
      <geom friction="0.5 0.005 0.0001" condim="3"
            solref="0.02 1" solimp="0.99 0.99 0.01"
            contype="1" conaffinity="1"/>
    </default>
    <default class="visual">
      <geom contype="0" conaffinity="0"/>
    </default>
  </default>

  <asset>
    <texture name="floor_wood" type="2d" builtin="checker"
             width="512" height="512" rgb1="0.7 0.6 0.4" rgb2="0.6 0.5 0.35"/>
    <material name="floor_mat" texture="floor_wood" texrepeat="4 4"/>
    <material name="cloth_mat" rgba="0.8 0.2 0.2 0.9"/>
  </asset>

  <worldbody>
    <!-- Lighting -->
    <light name="key" directional="true" pos="2.5 2.5 3.0" dir="0 0 -1"
           diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>
    <light name="fill" directional="true" pos="0.5 2.5 2.5" dir="0.5 0 -0.5"
           diffuse="0.3 0.3 0.3" specular="0 0 0"/>

    <!-- Floor -->
    <geom name="floor" type="plane" size="2.5 2.5 0.01" material="floor_mat"
          class="table"/>

    <!-- Table -->
    <body name="table" pos="2.5 2.5 0.375">
      <geom name="table_top" type="box" size="0.5 0.5 0.375" mass="200"
            class="table" rgba="0.7 0.6 0.4 1"/>
    </body>

    <!-- Robot arm base at table edge -->
    <body name="robot_base" pos="1.5 2.5 0.75">
      <!-- Shoulder assembly -->
      <body name="shoulder" pos="0.0 0.0 0.35">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1"
               range="-1.57 1.57" class="robot"/>
        <geom type="cylinder" size="0.06 0.08" class="robot" rgba="0.3 0.3 0.3 1"/>

        <body name="upper_arm" pos="0.0 0.0 0.08">
          <joint name="shoulder_lift" type="hinge" axis="0 1 0"
                 range="-1.57 1.57" class="robot"/>
          <geom type="capsule" fromto="0 0 0 0.0 0.0 0.28" size="0.04"
                class="robot" rgba="0.5 0.5 0.5 1"/>

          <body name="shoulder_roll_link" pos="0.0 0.0 0.28">
            <joint name="shoulder_roll" type="hinge" axis="1 0 0"
                   range="-0.785 0.785" class="robot"/>
            <geom type="sphere" size="0.045" class="robot" rgba="0.3 0.3 0.3 1"/>

            <body name="forearm" pos="0.0 0.0 0.0">
              <joint name="elbow" type="hinge" axis="0 1 0"
                     range="0 2.094" class="robot"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.035"
                    class="robot" rgba="0.5 0.5 0.5 1"/>

              <body name="wrist_yaw_link" pos="0.0 0.0 0.25">
                <joint name="wrist_yaw" type="hinge" axis="0 0 1"
                       range="-1.57 1.57" class="robot"/>
                <geom type="sphere" size="0.04" class="robot" rgba="0.3 0.3 0.3 1"/>

                <body name="wrist_pitch_link" pos="0.0 0.0 0.0">
                  <joint name="wrist_pitch" type="hinge" axis="0 1 0"
                         range="-1.57 1.57" class="robot"/>
                  <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.03"
                        class="robot" rgba="0.5 0.5 0.5 1"/>

                  <body name="wrist_roll" pos="0.0 0.0 0.1">
                    <joint name="wrist_roll" type="hinge" axis="1 0 0"
                           range="-1.57 1.57" class="robot"/>
                    <geom type="sphere" size="0.035" class="robot" rgba="0.3 0.3 0.3 1"/>

                    <!-- Gripper palm -->
                    <body name="palm" pos="0.0 0.0 0.05">
                      <geom type="box" size="0.04 0.02 0.025" class="robot"
                            rgba="0.4 0.4 0.4 1"/>

                      <!-- Left finger -->
                      <body name="left_finger" pos="0.0 0.03 0.025">
                        <joint name="finger_joint" type="slide" axis="0 1 0"
                               range="0 0.04" damping="1000" armature="100"/>
                        <geom type="box" size="0.006 0.01 0.025" class="gripper"
                              friction="1.0 0.01 0.0001" rgba="0.6 0.6 0.6 1"/>
                        <site name="left_finger_tip" pos="0 0 0.025" size="0.005"/>
                      </body>

                      <!-- Right finger (mirrored) -->
                      <body name="right_finger" pos="0.0 -0.03 0.025">
                        <joint name="finger_joint_r" type="slide" axis="0 -1 0"
                               range="0 0.04" damping="1000" armature="100"/>
                        <geom type="box" size="0.006 0.01 0.025" class="gripper"
                              friction="1.0 0.01 0.0001" rgba="0.6 0.6 0.6 1"/>
                        <site name="right_finger_tip" pos="0 0 0.025" size="0.005"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Head (2-DOF, mounted above shoulder) -->
    <body name="head_base" pos="1.5 2.5 1.30">
      <body name="head_pan_link" pos="0 0 0">
        <joint name="head_pan" type="hinge" axis="0 0 1"
               range="-1.57 1.57" damping="10" armature="1"/>
        <geom type="sphere" size="0.01" mass="0.01" class="visual"/>
        <body name="head_tilt" pos="0 0 0">
          <joint name="head_tilt" type="hinge" axis="0 1 0"
                 range="-0.785 0.785" damping="10" armature="1"/>
          <geom type="sphere" size="0.04" rgba="0.4 0.4 0.4 1" class="visual"/>
          <!-- Stereo cameras (visual only) -->
          <camera name="stereo_left" pos="0.0 0.034 0.05" fovy="60" mode="fixed"/>
          <camera name="stereo_right" pos="0.0 -0.034 0.05" fovy="60" mode="fixed"/>
        </body>
      </body>
    </body>

    <!-- ====== Object Library ====== -->

    <!-- Cubes -->
    <body name="cube_red" pos="5.0 5.0 -1.0">
      <joint type="free"/>
      <geom type="box" size="0.025 0.025 0.025" mass="0.10" class="object"
            friction="0.35 0.005 0.0001" rgba="1.0 0.23 0.23 1"/>
    </body>
    <body name="cube_blue" pos="5.0 5.1 -1.0">
      <joint type="free"/>
      <geom type="box" size="0.025 0.025 0.025" mass="0.20" class="object"
            friction="0.35 0.005 0.0001" rgba="0.23 0.23 1.0 1"/>
    </body>
    <body name="cube_green" pos="5.0 5.2 -1.0">
      <joint type="free"/>
      <geom type="box" size="0.025 0.025 0.025" mass="0.30" class="object"
            friction="0.4 0.005 0.0001" rgba="0.23 0.78 0.23 1"/>
    </body>
    <body name="cube_yellow" pos="5.0 5.3 -1.0">
      <joint type="free"/>
      <geom type="box" size="0.025 0.025 0.025" mass="0.40" class="object"
            friction="0.4 0.005 0.0001" rgba="1.0 1.0 0.23 1"/>
    </body>
    <body name="cube_heavy" pos="5.0 5.4 -1.0">
      <joint type="free"/>
      <geom type="box" size="0.025 0.025 0.025" mass="0.50" class="object"
            friction="0.2 0.002 0.0001" rgba="1.0 0.65 0.0 1"/>
    </body>

    <!-- Spheres -->
    <body name="sphere_red" pos="5.1 5.0 -1.0">
      <joint type="free"/>
      <geom type="sphere" size="0.03" mass="0.15" class="object"
            friction="0.8 0.01 0.001" condim="6" rgba="1.0 0.2 0.2 1"/>
    </body>
    <body name="sphere_blue" pos="5.1 5.1 -1.0">
      <joint type="free"/>
      <geom type="sphere" size="0.03" mass="0.25" class="object"
            friction="0.8 0.01 0.001" condim="6" rgba="0.2 0.2 1.0 1"/>
    </body>

    <!-- Occluder screens (visual only, fixed) -->
    <body name="screen_A" pos="2.25 2.7 0.875">
      <geom type="box" size="0.10 0.005 0.125" class="visual" rgba="0.3 0.3 0.3 1"/>
    </body>
    <body name="screen_B" pos="2.75 2.7 0.875">
      <geom type="box" size="0.10 0.005 0.125" class="visual" rgba="0.3 0.3 0.3 1"/>
    </body>

    <!-- Open container -->
    <body name="box_open" pos="5.2 5.0 -1.0">
      <joint type="free"/>
      <geom type="box" size="0.075 0.075 0.0025" pos="0 0 0" mass="0.04"
            class="object" rgba="0.6 0.4 0.2 1"/>
      <geom type="box" size="0.075 0.0025 0.05" pos="0 0.0725 0.05" mass="0.04"
            class="object" rgba="0.6 0.4 0.2 1"/>
      <geom type="box" size="0.075 0.0025 0.05" pos="0 -0.0725 0.05" mass="0.04"
            class="object" rgba="0.6 0.4 0.2 1"/>
      <geom type="box" size="0.0025 0.075 0.05" pos="0.0725 0 0.05" mass="0.04"
            class="object" rgba="0.6 0.4 0.2 1"/>
      <geom type="box" size="0.0025 0.075 0.05" pos="-0.0725 0 0.05" mass="0.04"
            class="object" rgba="0.6 0.4 0.2 1"/>
    </body>

    <!-- Tools -->
    <body name="stick" pos="5.3 5.0 -1.0">
      <joint type="free"/>
      <geom type="cylinder" size="0.01 0.15" mass="0.05" class="object"
            friction="0.7 0.005 0.0001" rgba="0.6 0.4 0.2 1"/>
    </body>
    <body name="rake" pos="5.3 5.1 -1.0">
      <joint type="free"/>
      <geom name="rake_handle" type="cylinder" size="0.01 0.15"
            pos="0 0 0" mass="0.05" friction="0.7 0.005 0.0001"
            rgba="0.5 0.35 0.15 1" class="object"/>
      <geom name="rake_bar" type="cylinder" size="0.008 0.05"
            pos="0.15 0 0" euler="0 1.5708 0" mass="0.03"
            friction="0.7 0.005 0.0001" rgba="0.5 0.35 0.15 1" class="object"/>
    </body>
    <body name="spoon" pos="5.3 5.2 -1.0">
      <joint type="free"/>
      <geom name="spoon_handle" type="cylinder" size="0.008 0.075"
            pos="0 0 0" mass="0.025" friction="0.7 0.005 0.0001"
            rgba="0.7 0.7 0.7 1" class="object"/>
      <geom name="spoon_bowl" type="sphere" size="0.025"
            pos="0.075 0 0" mass="0.015" friction="0.7 0.005 0.0001"
            rgba="0.7 0.7 0.7 1" class="object"/>
    </body>
    <body name="l_bar" pos="5.3 5.3 -1.0">
      <joint type="free"/>
      <geom name="lbar_shaft" type="box" size="0.15 0.01 0.01"
            pos="0 0 0" mass="0.05" friction="0.7 0.005 0.0001"
            rgba="0.2 0.6 0.2 1" class="object"/>
      <geom name="lbar_foot" type="box" size="0.01 0.05 0.01"
            pos="0.15 0 0" mass="0.03" friction="0.7 0.005 0.0001"
            rgba="0.2 0.6 0.2 1" class="object"/>
    </body>

    <!-- String (TC-15) -->
    <body name="string" pos="5.4 5.0 -1.0">
      <joint type="free"/>
      <geom type="cylinder" size="0.005 0.10" mass="0.02"
            friction="0.5 0.005 0.0001" rgba="0.9 0.9 0.2 1" class="object"/>
    </body>

    <!-- Special objects -->
    <body name="target_disk" pos="5.5 5.0 -1.0">
      <geom type="cylinder" size="0.03 0.001" class="visual" rgba="1 0 0 0.8"/>
    </body>
    <body name="platform" pos="5.5 5.1 -1.0">
      <joint type="free"/>
      <geom type="box" size="0.05 0.05 0.025" mass="0.30"
            class="object" rgba="0.5 0.5 0.5 1"/>
    </body>
    <body name="barrier" pos="5.5 5.2 -1.0">
      <joint type="free"/>
      <geom type="box" size="0.075 0.05 0.05" mass="0.30"
            class="object" friction="0.5 0.005 0.0001" rgba="0.4 0.4 0.4 1"/>
    </body>

    <!-- Latch box (TC-22) -->
    <body name="latch_box" pos="5.6 5.0 -1.0">
      <geom name="box_floor" type="box" size="0.05 0.05 0.002"
            pos="0 0 0" rgba="0.8 0.9 1.0 0.3" class="object"/>
      <geom name="box_back" type="box" size="0.05 0.002 0.04"
            pos="0 -0.048 0.04" rgba="0.8 0.9 1.0 0.3" class="object"/>
      <geom name="box_left" type="box" size="0.002 0.05 0.04"
            pos="-0.048 0 0.04" rgba="0.8 0.9 1.0 0.3" class="object"/>
      <geom name="box_right" type="box" size="0.002 0.05 0.04"
            pos="0.048 0 0.04" rgba="0.8 0.9 1.0 0.3" class="object"/>
      <body name="lid" pos="0 0.048 0.08">
        <joint name="lid_slide" type="slide" axis="0 1 0" range="0 0.10"
               damping="5" stiffness="0" armature="0.1"/>
        <geom type="box" size="0.05 0.05 0.002" rgba="0.8 0.9 1.0 0.3"
              class="object"/>
      </body>
      <body name="latch" pos="0.048 0 0.04">
        <joint name="latch_hinge" type="hinge" axis="0 0 1"
               range="-0.5 0.5" damping="2" stiffness="5" armature="0.05"/>
        <geom type="box" size="0.015 0.005 0.02" rgba="0.8 0.2 0.2 1"
              class="object"/>
      </body>
    </body>

    <!-- Cloth (TC-18) — composite body approximation using geom chain -->
    <!-- Note: full MuJoCo composite syntax; simplified if parser rejects -->
  </worldbody>

  <!-- String weld equality (TC-15) — string far end welded to cube_red -->
  <equality>
    <weld body1="string" body2="cube_red" relpose="0.10 0 0 1 0 0 0"
          solref="0.01 1" solimp="0.9 0.95 0.001"/>
  </equality>

  <!-- Tactile sensors -->
  <sensor>
    <touch name="left_contact" site="left_finger_tip"/>
    <touch name="right_contact" site="right_finger_tip"/>
  </sensor>

  <!-- Actuators: position control for each joint -->
  <actuator>
    <position name="act_shoulder_pan"   joint="shoulder_pan"   kp="100" gear="1"/>
    <position name="act_shoulder_lift"  joint="shoulder_lift"  kp="100" gear="1"/>
    <position name="act_shoulder_roll"  joint="shoulder_roll"  kp="100" gear="1"/>
    <position name="act_elbow"          joint="elbow"          kp="100" gear="1"/>
    <position name="act_wrist_yaw"      joint="wrist_yaw"      kp="100" gear="1"/>
    <position name="act_wrist_pitch"    joint="wrist_pitch"    kp="100" gear="1"/>
    <position name="act_wrist_roll"     joint="wrist_roll"     kp="100" gear="1"/>
    <position name="act_finger"         joint="finger_joint"   kp="200" gear="1"/>
    <position name="act_finger_r"       joint="finger_joint_r" kp="200" gear="1"/>
    <position name="act_head_pan"       joint="head_pan"       kp="20"  gear="1"/>
    <position name="act_head_tilt"      joint="head_tilt"      kp="20"  gear="1"/>
  </actuator>
</mujoco>
"""


# ---------------------------------------------------------------------------
# Object type IDs (for obs encoding)
# ---------------------------------------------------------------------------

OBJECT_TYPE_IDS: dict[str, float] = {
    "cube_red": 0.0 / 19,
    "cube_blue": 1.0 / 19,
    "cube_green": 2.0 / 19,
    "cube_yellow": 3.0 / 19,
    "cube_heavy": 4.0 / 19,
    "sphere_red": 5.0 / 19,
    "sphere_blue": 6.0 / 19,
    "box_open": 7.0 / 19,
    "stick": 8.0 / 19,
    "rake": 9.0 / 19,
    "spoon": 10.0 / 19,
    "l_bar": 11.0 / 19,
    "string": 12.0 / 19,
    "target_disk": 13.0 / 19,
    "platform": 14.0 / 19,
    "barrier": 15.0 / 19,
    "latch_box": 16.0 / 19,
    "screen_A": 17.0 / 19,
    "screen_B": 18.0 / 19,
    "cloth": 19.0 / 19,
}

OBJECT_MASSES: dict[str, float] = {
    "cube_red": 0.10,
    "cube_blue": 0.20,
    "cube_green": 0.30,
    "cube_yellow": 0.40,
    "cube_heavy": 0.50,
    "sphere_red": 0.15,
    "sphere_blue": 0.25,
    "box_open": 0.20,
    "stick": 0.05,
    "rake": 0.08,
    "spoon": 0.04,
    "l_bar": 0.08,
    "string": 0.02,
    "target_disk": 0.0,
    "platform": 0.30,
    "barrier": 0.30,
    "latch_box": 0.15,
}


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class SLMSensorimotor(gym.Env):
    """MuJoCo tabletop environment for TC-11 through TC-24.

    Args:
        task_id: Task identifier, e.g. "TC-11" through "TC-24".
        render_mode: "human" or "rgb_array" for visual rendering, or None.
        vision_mode: If True, provides vision placeholder in observation.
        seed: RNG seed.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        task_id: str = "TC-13",
        render_mode: str | None = None,
        vision_mode: bool = False,
        seed: int | None = None,
    ):
        super().__init__()

        if task_id not in VALID_TASK_IDS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {VALID_TASK_IDS}")

        self.task_id = task_id
        self.render_mode = render_mode
        self.vision_mode = vision_mode

        self._task = TASK_REGISTRY[task_id]
        self._rng = np.random.default_rng(seed)

        # Build and compile model
        mjcf_xml = _build_mjcf(self._task.scene_objects())
        self._model = mujoco.MjModel.from_xml_string(mjcf_xml)
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = PHYSICS_DT

        # Cache joint IDs
        self._joint_ids = np.array([
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ], dtype=np.int32)
        self._gripper_jnt_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint"
        )
        self._gripper_jnt_r_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint_r"
        )
        self._head_pan_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, "head_pan"
        )
        self._head_tilt_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, "head_tilt"
        )

        # Cache actuator IDs
        self._act_joint_ids = np.array([
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{name}")
            for name in JOINT_NAMES
        ], dtype=np.int32)
        self._act_finger_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_finger"
        )
        self._act_finger_r_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_finger_r"
        )
        self._act_head_pan_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_head_pan"
        )
        self._act_head_tilt_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_head_tilt"
        )

        # Cache wrist body ID for EE position
        self._ee_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "wrist_roll"
        )

        # Objects in this task's scene (for obs encoding)
        self._scene_objects = self._task.scene_objects()

        # Spaces
        # Always Dict: "ground_truth" (56-dim Box) + "vision" placeholder.
        # Agents in Phase 3.2a extract obs["ground_truth"] before passing to DaseinNet.
        # Phase 3.2b+: vision populated with real stereo frames.
        obs_dim = OBS_DIM
        self.observation_space = spaces.Dict({
            "ground_truth": spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
            "vision": spaces.Box(
                low=0, high=255, shape=(2, 128, 128, 3), dtype=np.uint8
            ),
        })
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )

        # Episode state
        self._step_count: int = 0
        self._energy: float = MAX_ENERGY
        self._task_state: dict = {}
        self._renderer: mujoco.Renderer | None = None

        logger.debug(f"SLMSensorimotor initialized: task={task_id}, obs_dim={obs_dim}")

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self._model, self._data)
        self._set_home_position()
        self._step_count = 0
        self._energy = MAX_ENERGY

        # Let task place objects
        self._task_state = self._task.reset(self._model, self._data, self._rng)

        # Run a few physics steps to settle
        for _ in range(10):
            mujoco.mj_step(self._model, self._data)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict, float, bool, bool, dict]:
        action = np.clip(action.astype(np.float32), -1.0, 1.0)

        # Map action → joint position targets
        joint_targets = JOINT_MID + action[:7] * JOINT_HALF
        gripper_target = (action[7] + 1.0) / 2.0 * 0.04  # [-1,1] → [0, 0.04]
        head_pan_target = action[8] * 1.57
        head_tilt_target = action[9] * 0.785

        # Apply position control targets to actuators
        for i, act_id in enumerate(self._act_joint_ids):
            self._data.ctrl[act_id] = joint_targets[i]
        self._data.ctrl[self._act_finger_id] = gripper_target
        self._data.ctrl[self._act_finger_r_id] = gripper_target
        self._data.ctrl[self._act_head_pan_id] = head_pan_target
        self._data.ctrl[self._act_head_tilt_id] = head_tilt_target

        # Step physics (20 substeps)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self._model, self._data)

        self._step_count += 1
        self._energy -= ENERGY_DECAY

        # Task-specific reward
        info: dict[str, Any] = {}
        reward = self._task.step(self._model, self._data, self._task_state, info)

        # Termination
        terminated = self._energy <= 0.0
        truncated = False
        obs = self._get_obs()
        info.update(self._get_info())
        info["score"] = self._task.score(self._task_state)

        return obs, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=240, width=320)
        self._renderer.update_scene(self._data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if hasattr(self, "_stereo_renderer") and self._stereo_renderer is not None:
            self._stereo_renderer.close()
            self._stereo_renderer = None

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict:
        gt = self._build_ground_truth_obs()
        if self.vision_mode:
            left, right = self._render_stereo()
            return {"ground_truth": gt, "vision": np.stack([left, right], axis=0)}
        return {"ground_truth": gt, "vision": np.zeros((2, 128, 128, 3), dtype=np.uint8)}

    def _render_stereo(self) -> tuple[np.ndarray, np.ndarray]:
        """Render 128×128 RGB images from stereo_left and stereo_right cameras.

        Returns:
            left:  (128, 128, 3) uint8
            right: (128, 128, 3) uint8
        """
        if not hasattr(self, "_stereo_renderer") or self._stereo_renderer is None:
            self._stereo_renderer = mujoco.Renderer(self._model, height=128, width=128)

        renderer = self._stereo_renderer
        renderer.update_scene(self._data, camera="stereo_left")
        left = renderer.render().copy()

        renderer.update_scene(self._data, camera="stereo_right")
        right = renderer.render().copy()

        return left, right

    def _build_ground_truth_obs(self) -> np.ndarray:
        """Build 56-dim ground-truth observation per env-detailed.md §6.7."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # --- Proprioception (25 channels, idx 0-24) ---
        rng = self._rng
        for i, jnt_id in enumerate(self._joint_ids):
            if jnt_id < 0:
                continue
            qpos_adr = self._model.jnt_qposadr[jnt_id]
            dof_adr = self._model.jnt_dofadr[jnt_id]
            angle = self._data.qpos[qpos_adr]
            vel = self._data.qvel[dof_adr]
            torque = self._data.actuator_force[self._act_joint_ids[i]]
            # Normalize + noise
            obs[i] = float((angle - JOINT_MID[i]) / JOINT_HALF[i])
            obs[i] += rng.normal(0.0, NOISE_ANGLE / JOINT_HALF[i])
            obs[7 + i] = float(np.clip(vel / JOINT_MAX_VEL[i], -1, 1))
            obs[7 + i] += rng.normal(0.0, NOISE_VEL / JOINT_MAX_VEL[i])
            obs[14 + i] = float(np.clip(torque / JOINT_MAX_TORQUE[i], -1, 1))
            obs[14 + i] += rng.normal(0.0, NOISE_TORQUE / JOINT_MAX_TORQUE[i])

        # Gripper
        g_qpos_adr = self._model.jnt_qposadr[self._gripper_jnt_id]
        gripper_pos = float(self._data.qpos[g_qpos_adr])  # half-opening [0, 0.04]
        obs[21] = gripper_pos / 0.04  # normalized [0, 1]
        g_dof_adr = self._model.jnt_dofadr[self._gripper_jnt_id]
        gripper_vel = float(self._data.qvel[g_dof_adr])
        obs[22] = float(np.clip(gripper_vel / 0.5, -1, 1))

        # Head
        if self._head_pan_id >= 0:
            obs[23] = float(self._data.qpos[self._model.jnt_qposadr[self._head_pan_id]]) / 1.57
        if self._head_tilt_id >= 0:
            obs[24] = float(self._data.qpos[self._model.jnt_qposadr[self._head_tilt_id]]) / 0.785

        # --- Tactile (2 channels, idx 25-26) ---
        left_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "left_contact")
        right_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "right_contact")
        obs[25] = float(self._data.sensordata[left_id] > 0.0) if left_id >= 0 else 0.0
        obs[26] = float(self._data.sensordata[right_id] > 0.0) if right_id >= 0 else 0.0

        # --- EE state (6 channels, idx 27-32) ---
        ee_pos = self._data.xpos[self._ee_body_id]
        obs[27:30] = (ee_pos - TABLE_CENTER[:3]) / 0.5

        # EE orientation (euler angles from rotation matrix)
        ee_mat = self._data.xmat[self._ee_body_id].reshape(3, 3)
        # Approximate euler from rotation matrix (roll, pitch, yaw)
        sy = math.sqrt(ee_mat[0, 0] ** 2 + ee_mat[1, 0] ** 2)
        if sy > 1e-6:
            roll = math.atan2(ee_mat[2, 1], ee_mat[2, 2])
            pitch = math.atan2(-ee_mat[2, 0], sy)
            yaw = math.atan2(ee_mat[1, 0], ee_mat[0, 0])
        else:
            roll = math.atan2(-ee_mat[1, 2], ee_mat[1, 1])
            pitch = math.atan2(-ee_mat[2, 0], sy)
            yaw = 0.0
        obs[30] = roll / math.pi
        obs[31] = pitch / math.pi
        obs[32] = yaw / math.pi

        # --- Internal state (2 channels, idx 33-34) ---
        obs[33] = (self._energy - 50.0) / 50.0
        obs[34] = 2.0 * self._step_count / MAX_EPISODE_STEPS - 1.0

        # --- Object state (21 channels = 7 * 3, idx 35-55) ---
        objects_to_encode = self._scene_objects[:N_OBJECTS_MAX]
        for k, obj_name in enumerate(objects_to_encode):
            base = 35 + k * 7
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            if bid < 0:
                continue
            pos = self._data.xpos[bid]
            obs[base + 0] = (pos[0] - 2.5) / 0.5
            obs[base + 1] = (pos[1] - 2.5) / 0.5
            obs[base + 2] = (pos[2] - 0.75) / 0.5

            # Visibility: simple occlusion check vs screens
            visible = self._check_visibility(pos)
            obs[base + 3] = float(visible)

            # Grasped: object near EE and gripper closed
            gap = float(self._data.qpos[g_qpos_adr]) * 2.0
            grasped = bool(
                float(np.linalg.norm(pos - ee_pos)) < 0.06 and
                obs[25] > 0.5 and obs[26] > 0.5 and gap < 0.02
            )
            obs[base + 4] = float(grasped)

            obs[base + 5] = OBJECT_TYPE_IDS.get(obj_name, 0.0)
            obs[base + 6] = OBJECT_MASSES.get(obj_name, 0.0) / 0.5

        return obs.astype(np.float32)

    def _check_visibility(self, obj_pos: np.ndarray) -> bool:
        """Ray-cast occlusion check vs screen_A and screen_B."""
        head_bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "head_tilt")
        if head_bid < 0:
            return True
        cam_pos = self._data.xpos[head_bid]

        for screen_name in ("screen_A", "screen_B"):
            sbid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, screen_name)
            if sbid < 0:
                continue
            screen_center = self._model.body_pos[sbid]
            # Simplified: check if object is directly behind screen (Y > screen Y)
            # and within screen X bounds (±0.10 m of screen center)
            if obj_pos[1] > screen_center[1] - 0.01:
                if abs(obj_pos[0] - screen_center[0]) < 0.10:
                    # Ray from camera to object crosses screen plane
                    if cam_pos[1] < screen_center[1] < obj_pos[1]:
                        return False
        return True

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def _get_info(self) -> dict[str, Any]:
        ee_pos = self._data.xpos[self._ee_body_id].copy()
        g_qpos_adr = self._model.jnt_qposadr[self._gripper_jnt_id]
        gripper_gap = float(self._data.qpos[g_qpos_adr]) * 2.0
        left_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "left_contact")
        right_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "right_contact")
        contacts = {
            "left": float(self._data.sensordata[left_id]) if left_id >= 0 else 0.0,
            "right": float(self._data.sensordata[right_id]) if right_id >= 0 else 0.0,
        }
        obj_positions = {}
        for obj_name in self._scene_objects:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            if bid >= 0:
                obj_positions[obj_name] = self._data.xpos[bid].copy()
        return {
            "task_id": self.task_id,
            "step": self._step_count,
            "energy": self._energy,
            "ee_position": ee_pos,
            "gripper_gap": gripper_gap,
            "contacts": contacts,
            "object_positions": obj_positions,
            "grasp_state": bool(
                contacts["left"] > 0 and contacts["right"] > 0 and gripper_gap < 0.02
            ),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_home_position(self):
        """Set arm and head to home configuration."""
        for i, jnt_id in enumerate(self._joint_ids):
            if jnt_id < 0:
                continue
            qpos_adr = self._model.jnt_qposadr[jnt_id]
            self._data.qpos[qpos_adr] = HOME_QPOS[i]

        if self._gripper_jnt_id >= 0:
            self._data.qpos[self._model.jnt_qposadr[self._gripper_jnt_id]] = HOME_GRIPPER / 2.0
        if self._gripper_jnt_r_id >= 0:
            self._data.qpos[self._model.jnt_qposadr[self._gripper_jnt_r_id]] = HOME_GRIPPER / 2.0
        if self._head_pan_id >= 0:
            self._data.qpos[self._model.jnt_qposadr[self._head_pan_id]] = HOME_HEAD_PAN
        if self._head_tilt_id >= 0:
            self._data.qpos[self._model.jnt_qposadr[self._head_tilt_id]] = HOME_HEAD_TILT

        # Set actuator targets to match
        for i, act_id in enumerate(self._act_joint_ids):
            self._data.ctrl[act_id] = HOME_QPOS[i]
        self._data.ctrl[self._act_finger_id] = HOME_GRIPPER / 2.0
        self._data.ctrl[self._act_finger_r_id] = HOME_GRIPPER / 2.0
        self._data.ctrl[self._act_head_pan_id] = HOME_HEAD_PAN
        self._data.ctrl[self._act_head_tilt_id] = HOME_HEAD_TILT

        mujoco.mj_forward(self._model, self._data)
