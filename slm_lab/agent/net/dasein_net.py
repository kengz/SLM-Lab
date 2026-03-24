"""DaseinNet — L0 + L1 + policy/value heads for sensorimotor PPO.

Two modes (vision_mode parameter):

  ground_truth (default, Phase 3.2a):
    Input: 56-dim flat vector from SLM-Sensorimotor-TC*-v0.
    L0 channels: proprio (512) + object_state (512).

  vision (Phase 3.2b):
    Input: dict with keys:
      "ground_truth"  — (B, 35) proprio slice (indices 0-34, no object_state)
      "left"          — (B, 3, H, W) left eye image, float32 [0,1]
      "right"         — (B, 3, H, W) right eye image, float32 [0,1]
    L0 channels: proprio (512) + vision (512).
    InfoNCE loss (α=0.1, τ=0.07) aligns being embedding with DINOv2 features.

Observation layout (ground_truth 56-dim, from sensorimotor.py _build_ground_truth_obs):
  [0:25]  proprio   — joint angles/vels/torques (7 each), gripper pos/vel, head pan/tilt
  [25:27] tactile   — left/right fingertip contact
  [27:33] ee        — end-effector position (3) + Euler orientation (3)
  [33:35] internal  — energy + time fraction
  [35:56] object    — 3 objects × 7 features (position, visible, grasped, type_id, mass)

In vision mode, proprio slice covers [0:35] (proprio + tactile + ee + internal);
object_state is replaced by visual features from DINOv2 → StereoFusionModule.

Output (shared=True, continuous action): [mean (B, A), log_std (B, A), value (B, 1)]
  Compatible with PPO's calc_pdparam → out[-1] is value, out[:-1] is [mean, log_std].

GRU hidden state: managed as a module buffer (h_prev). Reset at episode start via
reset_hidden(). For batched training, h_prev held constant across minibatch passes
(stateless forward for PPO — GRU only used for thrownness computation).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from slm_lab.agent.net.base import Net
from slm_lab.agent.net import net_util
from slm_lab.agent.net.being_embedding import BeingEmbedding, L0Output
from slm_lab.agent.net.perception import ObjectStateEncoder, ProprioceptionEncoder
from slm_lab.lib import util


# Observation slice indices (ground_truth mode — full 56-dim)
_PROPRIO_SLICE = slice(0, 25)
_TACTILE_SLICE = slice(25, 27)
_EE_SLICE = slice(27, 33)
_INTERNAL_SLICE = slice(33, 35)
_OBJ_SLICE = slice(35, 56)

# Vision mode: proprio covers [0:35] (no object_state in flat obs)
_VISION_PROPRIO_SLICE = slice(0, 35)  # proprio + tactile + ee + internal

OBS_DIM = 56
N_OBJECTS = 3           # 3 objects × 7 features = 21 dims
D_MODEL = 512           # channel embedding dim, must match BeingEmbedding d_model
GRU_HIDDEN_DIM = 1024   # must match BeingEmbedding.thrownness_enc.hidden_dim

# InfoNCE hyperparameters
INFONCE_ALPHA = 0.1     # weight of InfoNCE loss relative to PPO loss
INFONCE_TEMP = 0.07     # temperature τ


class InfoNCELoss(nn.Module):
    """Contrastive loss aligning being embedding with DINOv2 visual features.

    Aligns the being embedding (from L1) with DINOv2 vision features (from L0)
    using the NT-Xent / InfoNCE objective. Encourages the being embedding to
    be grounded in visual perception.

    τ = 0.07 (standard from SimCLR/MoCo). Applied with weight α=0.1.

    Spec: notes/layers/L1-being-embedding.md §7.1
    """

    def __init__(self, temperature: float = INFONCE_TEMP) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        being_emb: torch.Tensor,
        vision_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE loss between being embedding and vision features.

        Args:
            being_emb:   (B, 512) L1 being embedding
            vision_feat: (B, 512) DINOv2 stereo fusion output

        Returns:
            Scalar InfoNCE loss
        """
        B = being_emb.shape[0]

        # L2-normalize both views
        z_b = F.normalize(being_emb, dim=-1)   # (B, 512)
        z_v = F.normalize(vision_feat, dim=-1)  # (B, 512)

        # Cosine similarity matrix (B, B), scaled by temperature
        logits = (z_b @ z_v.T) / self.temperature  # (B, B)

        # Positive pairs: diagonal (same sample)
        labels = torch.arange(B, device=being_emb.device)

        # Symmetric loss: being→vision and vision→being
        loss_bv = F.cross_entropy(logits, labels)
        loss_vb = F.cross_entropy(logits.T, labels)

        return (loss_bv + loss_vb) / 2.0


class _ProprioVisionEncoder(nn.Module):
    """Encode [0:35] flat obs into 512-dim proprio embedding (vision mode).

    In vision mode the proprio slice covers indices 0-35 (proprio + tactile +
    ee + internal). ObjectStateEncoder is unused. We project the 35-dim vector
    through two layers to match ProprioceptionEncoder's 512-dim output.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(35, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 35) — proprio + tactile + ee + internal
        Returns:
            (B, 512)
        """
        return self.net(x)


class DaseinNet(Net, nn.Module):
    """L0 + L1 perception pipeline with policy and value heads for PPO.

    Supports two modes via vision_mode parameter:
      "ground_truth": Phase 3.2a, 56-dim flat obs, proprio + object_state channels.
      "vision": Phase 3.2b, dict obs with stereo images, proprio + DINOv2 channels.

    net_spec keys (beyond standard Net):
        vision_mode:      str, "ground_truth" or "vision" (default "ground_truth")
        action_dim:       int, action space dimension (default 10 for sensorimotor)
        log_std_init:     float, initial log_std value (default 0.0)
        infonce_alpha:    float, InfoNCE loss weight (default 0.1, vision mode only)
        clip_grad_val:    float | None
        optim_spec:       optimizer spec dict
        lr_scheduler_spec: lr scheduler spec dict | None
        gpu:              bool | str
        lora_rank:        int, LoRA rank for DINOv2 (default 16, vision mode only)
        lora_alpha:       float, LoRA alpha (default 32.0, vision mode only)

    Args:
        net_spec: spec dict from experiment YAML
        in_dim:   must equal OBS_DIM (56) in ground_truth mode; ignored in vision mode
        out_dim:  [action_dim, action_dim, 1] — set by ActorCritic.init_nets
    """

    def __init__(
        self,
        net_spec: dict,
        in_dim: int,
        out_dim: list[int],
        _mock_dinov2: nn.Module | None = None,
    ) -> None:
        """
        Args:
            net_spec:      spec dict from experiment YAML
            in_dim:        must equal OBS_DIM (56) in ground_truth mode; ignored in vision mode
            out_dim:       [action_dim, action_dim, 1]
            _mock_dinov2:  optional pre-built DINOv2 model for unit tests (bypasses torch.hub)
        """
        nn.Module.__init__(self)
        Net.__init__(self, net_spec, in_dim, out_dim)

        util.set_attr(
            self,
            dict(
                vision_mode="ground_truth",
                action_dim=10,
                log_std_init=0.0,
                infonce_alpha=INFONCE_ALPHA,
                clip_grad_val=0.5,
                loss_spec={"name": "MSELoss"},
                optim_spec={"name": "Adam", "lr": 3e-4},
                lr_scheduler_spec=None,
                update_type="replace",
                update_frequency=1,
                polyak_coef=0.0,
                gpu=False,
                shared=True,
                lora_rank=16,
                lora_alpha=32.0,
            ),
        )
        util.set_attr(
            self,
            self.net_spec,
            [
                "vision_mode",
                "action_dim",
                "log_std_init",
                "infonce_alpha",
                "clip_grad_val",
                "loss_spec",
                "optim_spec",
                "lr_scheduler_spec",
                "update_type",
                "update_frequency",
                "polyak_coef",
                "gpu",
                "shared",
                "lora_rank",
                "lora_alpha",
            ],
        )

        if self.vision_mode not in ("ground_truth", "vision"):
            raise ValueError(
                f"vision_mode must be 'ground_truth' or 'vision', got '{self.vision_mode}'"
            )

        # Infer action_dim from out_dim if provided as list [A, A, 1]
        if isinstance(out_dim, list) and len(out_dim) >= 2:
            self.action_dim = out_dim[0]

        # L0: perception encoders — mode-dependent
        if self.vision_mode == "ground_truth":
            self.proprio_enc = ProprioceptionEncoder()
            self.obj_enc = ObjectStateEncoder(max_objects=N_OBJECTS)
            self.vision_enc = None
            self.infonce = None
        else:
            # vision mode: lazy-import to avoid HF download at import time
            from slm_lab.agent.net.vision import VisionEncoder
            self.proprio_enc_vision = _ProprioVisionEncoder()
            self.vision_enc = VisionEncoder(
                pretrained=False,   # loaded by caller; pretrained weight optional
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                _mock_model=_mock_dinov2,  # None in production; mock in tests
            )
            # DINOv2 backbone frozen by DINOv2Backbone before LoRA injection.
            # LoRA adapters (lora_A, lora_B) are trainable by design.
            # No assertion needed here — VisionEncoder enforces this internally.
            self.obj_enc = None
            self.proprio_enc = None
            self.infonce = InfoNCELoss(temperature=INFONCE_TEMP)

        # MoodFiLMLayer: conditions DINOv2 features with mood (vision mode only)
        # Instantiated unconditionally so state_dict is stable; only used in vision mode.
        from slm_lab.agent.net.film import MoodFiLMLayer
        self.mood_film = MoodFiLMLayer()

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

        # Last InfoNCE loss for external logging (None in ground_truth mode)
        self._last_infonce_loss: torch.Tensor | None = None

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

    def _split_obs_ground_truth(
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
    # Forward — ground_truth mode
    # ------------------------------------------------------------------

    def _forward_ground_truth(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Ground-truth forward: 56-dim obs → [mean, log_std, value]."""
        B = x.shape[0]
        proprio, tactile, ee, internal, obj_state = self._split_obs_ground_truth(x)

        # L0: encode channels
        proprio_feat = self.proprio_enc(proprio, tactile, ee, internal)  # (B, 512)
        obj_feat = self.obj_enc(obj_state)                                # (B, 512)

        l0_out = L0Output(proprioception=proprio_feat, object_state=obj_feat)

        # L1: being-time embedding
        h_prev = self._get_h_prev(B)
        l1_out = self.being_emb(l0_out, h_prev)
        self.h_prev = l1_out.h_t.detach()

        return self._heads(l1_out.being_time_embedding)

    # ------------------------------------------------------------------
    # Forward — vision mode
    # ------------------------------------------------------------------

    def _forward_vision(self, obs: dict) -> list[torch.Tensor]:
        """Vision forward: dict obs with stereo images → [mean, log_std, value].

        Args:
            obs: dict with keys:
              "ground_truth" — (B, 35) or (B, 56) flat obs (only [0:35] used)
              "left"         — (B, 3, H, W) or (B, H, W, 3) left eye, float32 [0,1]
              "right"        — (B, 3, H, W) or (B, H, W, 3) right eye, float32 [0,1]

        Returns:
            [mean, log_std, value]
        """
        gt = obs["ground_truth"]  # (B, 35) or (B, 56)
        left = obs["left"]        # stereo images
        right = obs["right"]

        B = gt.shape[0]

        # Proprio features from [0:35]
        proprio_35 = gt[:, _VISION_PROPRIO_SLICE]  # (B, 35)
        proprio_feat = self.proprio_enc_vision(proprio_35)  # (B, 512)

        # Vision features: DINOv2 → StereoFusion
        # vision_enc.forward(left, right) → (B, 512)
        # MoodFiLM deferred: mood tensor not available in base forward;
        # callers with mood context should use forward_with_mood().
        vision_feat = self.vision_enc(left, right)  # (B, 512)

        # InfoNCE: align being embedding with visual features
        # Computed post-L1 below; store vision_feat for loss computation.

        l0_out = L0Output(proprioception=proprio_feat, vision=vision_feat)

        # L1: being-time embedding
        h_prev = self._get_h_prev(B)
        l1_out = self.being_emb(l0_out, h_prev)
        self.h_prev = l1_out.h_t.detach()

        # InfoNCE: align being_embedding (L1 spatial) with DINOv2 vision features
        self._last_infonce_loss = self.infonce(l1_out.being_embedding, vision_feat)

        return self._heads(l1_out.being_time_embedding)

    # ------------------------------------------------------------------
    # Shared heads
    # ------------------------------------------------------------------

    def _heads(self, bte: torch.Tensor) -> list[torch.Tensor]:
        """Shared policy + value heads.

        Args:
            bte: (B, 512) being-time embedding

        Returns:
            [mean (B, A), log_std_expanded (B, A), value (B, 1)]
        """
        shared = self.shared_backbone(bte)

        policy_feat = self.policy_fc(shared)
        mean = self.mean_head(policy_feat)
        log_std_expanded = self.log_std.expand_as(mean)

        value_feat = self.value_fc(shared)
        value = self.value_head(value_feat)

        return [mean, log_std_expanded, value]

    # ------------------------------------------------------------------
    # Forward (dispatch)
    # ------------------------------------------------------------------

    def forward(self, x) -> list[torch.Tensor]:
        """Full forward pass: obs → [mean, log_std, value].

        Compatible with PPO's shared network convention:
          out[-1]  = value  (B, 1)
          out[:-1] = [mean (B, A), log_std expanded (B, A)]

        Args:
            x: (B, 56) flat tensor (ground_truth mode)
               OR dict with "ground_truth", "left", "right" (vision mode)

        Returns:
            [mean, log_std_expanded, value]
        """
        if self.vision_mode == "vision":
            if not isinstance(x, dict):
                raise TypeError(
                    "vision mode requires dict obs with 'ground_truth', 'left', 'right'"
                )
            return self._forward_vision(x)
        else:
            if isinstance(x, dict):
                x = x["ground_truth"]
            return self._forward_ground_truth(x)

    # ------------------------------------------------------------------
    # InfoNCE loss accessor
    # ------------------------------------------------------------------

    @property
    def last_infonce_loss(self) -> torch.Tensor | None:
        """Last InfoNCE loss computed during forward (vision mode only).

        Callers (e.g., PPO training loop) add: total_loss += α * net.last_infonce_loss
        α = net.infonce_alpha
        """
        return self._last_infonce_loss


# Register for TorchArc YAML spec usage
if not hasattr(nn, "DaseinNet"):
    setattr(nn, "DaseinNet", DaseinNet)
