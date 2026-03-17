"""DaseinNet — L0 + L1 + policy/value heads for sensorimotor PPO.

Integrates the full L0 (ProprioceptionEncoder + ObjectStateEncoder) and L1
(BeingEmbedding) pipeline with a shared policy/value head compatible with
the PPO actor-critic interface.

Input: 56-dim flat ground-truth observation from SLM-Sensorimotor-TC*-v0.

Observation layout (from sensorimotor.py _build_ground_truth_obs):
  [0:25]  proprio   — joint angles/vels/torques (7 each), gripper pos/vel, head pan/tilt
  [25:27] tactile   — left/right fingertip contact
  [27:33] ee        — end-effector position (3) + Euler orientation (3)
  [33:35] internal  — energy + time fraction
  [35:56] object    — 3 objects × 7 features (position, visible, grasped, type_id, mass)

Output (shared=True, continuous action): [mean (B, A), log_std (B, A), value (B, 1)]
  Compatible with PPO's calc_pdparam → out[-1] is value, out[:-1] is [mean, log_std].

GRU hidden state: managed as a module buffer (h_prev). Reset at episode start via
reset_hidden(). For batched training, h_prev held constant across minibatch passes
(stateless forward for PPO — GRU only used for thrownness computation).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from slm_lab.agent.net.base import Net
from slm_lab.agent.net import net_util
from slm_lab.agent.net.being_embedding import BeingEmbedding, L0Output
from slm_lab.agent.net.perception import ObjectStateEncoder, ProprioceptionEncoder
from slm_lab.lib import util


# Observation slice indices
_PROPRIO_SLICE = slice(0, 25)
_TACTILE_SLICE = slice(25, 27)
_EE_SLICE = slice(27, 33)
_INTERNAL_SLICE = slice(33, 35)
_OBJ_SLICE = slice(35, 56)

OBS_DIM = 56
N_OBJECTS = 3           # 3 objects × 7 features = 21 dims
D_MODEL = 512           # channel embedding dim, must match BeingEmbedding d_model
GRU_HIDDEN_DIM = 1024   # must match BeingEmbedding.thrownness_enc.hidden_dim


class DaseinNet(Net, nn.Module):
    """L0 + L1 perception pipeline with policy and value heads for PPO.

    Implements the SLM agent architecture for Phase 3.2a ground-truth mode.
    Compatible with ActorCritic/PPO's shared network interface: forward() returns
    [mean, log_std, value] where value is used by calc_v() and mean/log_std by
    the policy distribution.

    net_spec keys (beyond standard Net):
        action_dim:       int, action space dimension (default 10 for sensorimotor)
        log_std_init:     float, initial log_std value (default 0.0)
        clip_grad_val:    float | None
        optim_spec:       optimizer spec dict
        lr_scheduler_spec: lr scheduler spec dict | None
        gpu:              bool | str

    Args:
        net_spec: spec dict from experiment YAML
        in_dim:   must equal OBS_DIM (56)
        out_dim:  [action_dim, action_dim, 1] — set by ActorCritic.init_nets
    """

    def __init__(self, net_spec: dict, in_dim: int, out_dim: list[int]) -> None:
        nn.Module.__init__(self)
        Net.__init__(self, net_spec, in_dim, out_dim)

        util.set_attr(
            self,
            dict(
                action_dim=10,
                log_std_init=0.0,
                clip_grad_val=0.5,
                loss_spec={"name": "MSELoss"},
                optim_spec={"name": "Adam", "lr": 3e-4},
                lr_scheduler_spec=None,
                update_type="replace",
                update_frequency=1,
                polyak_coef=0.0,
                gpu=False,
                shared=True,
            ),
        )
        util.set_attr(
            self,
            self.net_spec,
            [
                "action_dim",
                "log_std_init",
                "clip_grad_val",
                "loss_spec",
                "optim_spec",
                "lr_scheduler_spec",
                "update_type",
                "update_frequency",
                "polyak_coef",
                "gpu",
                "shared",
            ],
        )

        # Infer action_dim from out_dim if provided as list [A, A, 1]
        if isinstance(out_dim, list) and len(out_dim) >= 2:
            self.action_dim = out_dim[0]

        # L0: perception encoders
        self.proprio_enc = ProprioceptionEncoder()
        self.obj_enc = ObjectStateEncoder(max_objects=N_OBJECTS)

        # L1: being embedding (channel attention + GRU + temporal transformer)
        self.being_emb = BeingEmbedding(max_channels=4, d_model=D_MODEL)

        # Shared backbone (policy + value share first two layers)
        self.shared_backbone = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL), nn.ReLU(),
            nn.Linear(D_MODEL, D_MODEL), nn.ReLU(),
        )

        # Policy head: additional layer + mean output
        self.policy_fc = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.ReLU())
        self.mean_head = nn.Linear(D_MODEL, self.action_dim)
        self.log_std = nn.Parameter(torch.ones(self.action_dim) * self.log_std_init)

        # Value head: additional layer + scalar
        self.value_fc = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.ReLU())
        self.value_head = nn.Linear(D_MODEL, 1)

        # GRU hidden state buffer — (1, GRU_HIDDEN_DIM), expanded at runtime
        self.register_buffer(
            "h_prev", torch.zeros(1, GRU_HIDDEN_DIM), persistent=False
        )

        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.to(self.device)
        self.train()

    # ------------------------------------------------------------------
    # Hidden state management
    # ------------------------------------------------------------------

    def reset_hidden(self, batch_size: int = 1) -> None:
        """Reset GRU hidden state for new episodes."""
        self.h_prev = torch.zeros(batch_size, GRU_HIDDEN_DIM, device=self.device)

    def _get_h_prev(self, batch_size: int) -> torch.Tensor:
        """Return h_prev expanded to batch_size, reinitializing if needed."""
        if self.h_prev.shape[0] != batch_size:
            return torch.zeros(batch_size, GRU_HIDDEN_DIM, device=self.device)
        return self.h_prev

    # ------------------------------------------------------------------
    # Obs splitting
    # ------------------------------------------------------------------

    def _split_obs(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split 56-dim flat obs into component tensors."""
        return (
            x[:, _PROPRIO_SLICE],   # (B, 25)
            x[:, _TACTILE_SLICE],   # (B, 2)
            x[:, _EE_SLICE],        # (B, 6)
            x[:, _INTERNAL_SLICE],  # (B, 2)
            x[:, _OBJ_SLICE],       # (B, 21)
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Full forward pass: obs → [mean, log_std, value].

        Compatible with PPO's shared network convention:
          out[-1]  = value  (B, 1)
          out[:-1] = [mean (B, A), log_std expanded (B, A)]

        Args:
            x: (B, 56) flat ground-truth observation

        Returns:
            [mean, log_std_expanded, value]
        """
        B = x.shape[0]
        proprio, tactile, ee, internal, obj_state = self._split_obs(x)

        # L0: encode channels
        proprio_feat = self.proprio_enc(proprio, tactile, ee, internal)  # (B, 512)
        obj_feat = self.obj_enc(obj_state)                                # (B, 512)

        l0_out = L0Output(proprioception=proprio_feat, object_state=obj_feat)

        # L1: being-time embedding
        h_prev = self._get_h_prev(B)
        l1_out = self.being_emb(l0_out, h_prev)       # L1Output
        # Update hidden state (detach to prevent BPTT through full history)
        self.h_prev = l1_out.h_t.detach()

        bte = l1_out.being_time_embedding  # (B, 512)

        # Shared backbone
        shared = self.shared_backbone(bte)  # (B, 512)

        # Policy branch
        policy_feat = self.policy_fc(shared)           # (B, 512)
        mean = self.mean_head(policy_feat)             # (B, A)
        log_std_expanded = self.log_std.expand_as(mean)  # (B, A)

        # Value branch
        value_feat = self.value_fc(shared)             # (B, 512)
        value = self.value_head(value_feat)            # (B, 1)

        return [mean, log_std_expanded, value]
