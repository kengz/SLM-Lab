"""L0 Perception encoders: ProprioceptionEncoder, ObjectStateEncoder.

Phase 3.2a ground-truth mode — no vision or audio. These modules produce
512-dim channel embeddings consumed by L1 channel attention.

L0Output is the canonical interface dataclass — defined in being_embedding.py
and re-exported here for convenience.

Input layout (from L0-perception.md §1):
  proprio  (B, 25): channels 0-24
    0-6   joint angles (arm, 7)
    7-13  joint velocities (7)
    14-20 joint torques (7)
    21    gripper position
    22    gripper velocity
    23    head pan
    24    head tilt
  tactile  (B, 2):  left/right fingertip contact
  ee       (B, 6):  EE position (3) + orientation Euler (3)
  internal (B, 2):  energy + time fraction
"""

from __future__ import annotations

import torch
import torch.nn as nn

from slm_lab.agent.net.being_embedding import L0Output  # canonical definition  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scientific_encode(x: torch.Tensor, x0: float = 1.0) -> torch.Tensor:
    """Map scalar tensor to (mantissa, exponent) pairs.

    Each input value becomes two values:
      mantissa = tanh(x / x0)          — sign + magnitude in [-1, 1]
      exponent = sigmoid(log|x| + eps) — scale magnitude in (0, 1)

    Args:
        x: (..., D) tensor
        x0: reference scale (default 1.0)

    Returns:
        (..., D, 2) tensor, last dim = [mantissa, exponent]
    """
    mantissa = torch.tanh(x / x0)
    exponent = torch.sigmoid(torch.log(torch.abs(x) + 1e-8))
    return torch.stack([mantissa, exponent], dim=-1)


def _encode_flat(x: torch.Tensor) -> torch.Tensor:
    """scientific_encode then flatten last two dims: (..., D) → (..., 2D)."""
    enc = scientific_encode(x)          # (..., D, 2)
    return enc.flatten(start_dim=-2)    # (..., 2D)


# ---------------------------------------------------------------------------
# ProprioceptionEncoder
# ---------------------------------------------------------------------------

class ProprioceptionEncoder(nn.Module):
    """Hierarchical MLP: 35 proprio dims → 512-dim embedding.

    Args:
        proprio:  (B, 25) — joint angles/velocities/torques, gripper, head
        tactile:  (B, 2)  — fingertip contact sensors
        ee:       (B, 6)  — end-effector position + Euler orientation
        internal: (B, 2)  — energy + time fraction

    Returns:
        (B, 512) proprioception embedding
    """

    def __init__(self) -> None:
        super().__init__()

        # Group encoders: encoded_dim → hidden → group_out
        # Finger: gripper_pos(1) + gripper_vel(1) + tactile(2) = 4 scalars → 8 encoded
        self.finger_enc = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
        )
        # Wrist: joints 4-6 angles(3) + vels(3) + torques(3) = 9 scalars → 18 encoded
        self.wrist_enc = nn.Sequential(
            nn.Linear(18, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        # Arm: joints 0-3 angles(4) + vels(4) + torques(4) = 12 scalars → 24 encoded
        self.arm_enc = nn.Sequential(
            nn.Linear(24, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
        )
        # Head: pan(1) + tilt(1) = 2 scalars → 4 encoded
        self.head_enc = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
        )
        # EE: pos(3) + ori(3) = 6 scalars → 12 encoded
        self.ee_enc = nn.Sequential(
            nn.Linear(12, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
        )
        # Internal: energy(1) + time(1) = 2 scalars → 4 encoded
        self.internal_enc = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
        )

        # Fusion: concat(64+64+128+32+64+32=384) → 512
        self.fusion = nn.Sequential(
            nn.Linear(384, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.LayerNorm(512),
        )

    def forward(
        self,
        proprio: torch.Tensor,
        tactile: torch.Tensor,
        ee: torch.Tensor,
        internal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            proprio:  (B, 25)
            tactile:  (B, 2)
            ee:       (B, 6)
            internal: (B, 2)

        Returns:
            (B, 512)
        """
        # Scientific-encode all inputs
        p = _encode_flat(proprio)   # (B, 50)
        t = _encode_flat(tactile)   # (B, 4)
        e = _encode_flat(ee)        # (B, 12)
        i = _encode_flat(internal)  # (B, 4)

        # Split proprio encoded channels — each original channel becomes 2 consecutive values
        # angles:   ch 0-6   → encoded  0-13  (7*2=14)
        # vels:     ch 7-13  → encoded 14-27  (7*2=14)
        # torques:  ch 14-20 → encoded 28-41  (7*2=14)
        # gripper pos: ch 21 → encoded 42-43  (1*2=2)
        # gripper vel: ch 22 → encoded 44-45  (1*2=2)
        # head pan:    ch 23 → encoded 46-47  (1*2=2)
        # head tilt:   ch 24 → encoded 48-49  (1*2=2)
        angles  = p[:, 0:14]    # (B, 14)  joints 0-6 angles
        vels    = p[:, 14:28]   # (B, 14)  joints 0-6 velocities
        torques = p[:, 28:42]   # (B, 14)  joints 0-6 torques

        gripper_pos = p[:, 42:44]   # (B, 2)
        gripper_vel = p[:, 44:46]   # (B, 2)
        head_pan    = p[:, 46:48]   # (B, 2)
        head_tilt   = p[:, 48:50]   # (B, 2)

        # Arm group: joints 0-3 → 8 angle + 8 vel + 8 torque = 24
        arm_group = torch.cat([
            angles[:, 0:8], vels[:, 0:8], torques[:, 0:8]
        ], dim=-1)   # (B, 24)

        # Wrist group: joints 4-6 → 6 angle + 6 vel + 6 torque = 18
        wrist_group = torch.cat([
            angles[:, 8:14], vels[:, 8:14], torques[:, 8:14]
        ], dim=-1)   # (B, 18)

        # Finger group: gripper_pos(2) + gripper_vel(2) + tactile(4) = 8
        finger_group = torch.cat([gripper_pos, gripper_vel, t], dim=-1)  # (B, 8)

        # Head group: pan(2) + tilt(2) = 4
        head_group = torch.cat([head_pan, head_tilt], dim=-1)  # (B, 4)

        # EE and internal already encoded
        ee_group       = e  # (B, 12)
        internal_group = i  # (B, 4)

        # Encode each group
        f = self.finger_enc(finger_group)    # (B, 64)
        w = self.wrist_enc(wrist_group)      # (B, 64)
        a = self.arm_enc(arm_group)          # (B, 128)
        h = self.head_enc(head_group)        # (B, 32)
        ee_feat = self.ee_enc(ee_group)      # (B, 64)
        int_feat = self.internal_enc(internal_group)  # (B, 32)

        # Fuse
        fused = torch.cat([f, w, a, h, ee_feat, int_feat], dim=-1)  # (B, 384)
        return self.fusion(fused)  # (B, 512)


# ---------------------------------------------------------------------------
# ObjectStateEncoder
# ---------------------------------------------------------------------------

class ObjectStateEncoder(nn.Module):
    """Flat-concat MLP → 512-dim embedding (Phase 3.2a bridge).

    Each object has 7 features: position(3), visible(1), grasped(1),
    type_id(1), mass(1). All objects concatenated then projected.

    Args:
        max_objects: N_obj (default 5)

    Input:  (B, 7 * N_obj) flattened object state
    Output: (B, 512)

    Discarded in Phase 3.2b when vision replaces ground-truth state.

    Architecture per L0-perception.md §6.3.
    """

    OBJ_DIM = 7  # dims per object

    def __init__(self, max_objects: int = 5) -> None:
        super().__init__()
        self.max_objects = max_objects

        # Flat projection: 7*N_obj → 256 → 512
        self.proj = nn.Sequential(
            nn.Linear(self.OBJ_DIM * max_objects, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.LayerNorm(512),
        )

    def forward(self, obj_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obj_state: (B, 7 * N_obj)

        Returns:
            (B, 512)
        """
        return self.proj(obj_state)  # (B, 512)


# Register for TorchArc YAML spec usage
import torch.nn as nn  # noqa: E402
if not hasattr(nn, "ProprioceptionEncoder"):
    setattr(nn, "ProprioceptionEncoder", ProprioceptionEncoder)
if not hasattr(nn, "ObjectStateEncoder"):
    setattr(nn, "ObjectStateEncoder", ObjectStateEncoder)

