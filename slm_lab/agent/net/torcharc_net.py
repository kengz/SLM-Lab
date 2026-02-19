import numpy as np
import torcharc
import torch
import torch.nn as nn

from slm_lab.agent.net import net_util
from slm_lab.agent.net.base import Net
from slm_lab.lib import util


class TorchArcNet(Net, nn.Module):
    """Network built from torcharc YAML architecture specs.

    The torcharc spec defines the body/feature extractor.
    Output tails are added automatically based on out_dim (same as MLPNet).

    net_spec example:
        net:
          type: TorchArcNet
          arc:  # inline torcharc spec
            modules:
              body:
                Sequential:
                  - LazyLinear:
                      out_features: 64
                  - Tanh:
                  - LazyLinear:
                      out_features: 64
                  - Tanh:
            graph:
              input: x
              modules:
                body: [x]
              output: body
          # Standard SLM-Lab params
          clip_grad_val: 0.5
          loss_spec: {name: MSELoss}
          optim_spec: {name: Adam, lr: 0.001}
          gpu: auto
    """

    def __init__(self, net_spec, in_dim, out_dim):
        nn.Module.__init__(self)
        super().__init__(net_spec, in_dim, out_dim)
        # set defaults (same pattern as MLPNet)
        util.set_attr(
            self,
            dict(
                out_layer_activation=None,
                init_fn=None,
                shared=False,
                normalize=False,
                hid_layers_activation="relu",
                clip_grad_val=None,
                loss_spec={"name": "MSELoss"},
                optim_spec={"name": "Adam"},
                lr_scheduler_spec=None,
                update_type="replace",
                update_frequency=1,
                polyak_coef=0.0,
                gpu=False,
                log_std_init=None,
                actor_init_std=None,
                critic_init_std=None,
            ),
        )
        util.set_attr(
            self,
            self.net_spec,
            [
                "out_layer_activation",
                "init_fn",
                "shared",
                "normalize",
                "hid_layers_activation",
                "clip_grad_val",
                "loss_spec",
                "optim_spec",
                "lr_scheduler_spec",
                "update_type",
                "update_frequency",
                "polyak_coef",
                "gpu",
                "log_std_init",
                "actor_init_std",
                "critic_init_std",
            ],
        )

        # Build body from torcharc spec (inline dict or file path)
        arc_spec = net_spec.get("arc")
        arc_file = net_spec.get("arc_file")
        if arc_spec:
            self.body = torcharc.build(arc_spec)
        elif arc_file:
            self.body = torcharc.build(arc_file)
        else:
            raise ValueError(
                "TorchArcNet requires 'arc' (inline spec) or 'arc_file' (path) in net_spec"
            )

        # Move body to device before dummy forward so LazyLinear initializes on correct device
        self.body.to(self.device)
        # Compute body output dim via dummy forward pass (triggers LazyLinear init on device)
        body_out_dim = self._get_body_out_dim()

        # Build output tails (same as MLPNet)
        self.tails, self.log_std = net_util.build_tails(
            body_out_dim, self.out_dim, self.out_layer_activation, self.log_std_init
        )

        net_util.init_layers(self, self.init_fn)
        net_util.init_tails(self, self.actor_init_std, self.critic_init_std)
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.to(self.device)
        self.train()

    def _get_body_out_dim(self):
        """Compute body output dimension via dummy forward pass."""
        with torch.no_grad():
            if isinstance(self.in_dim, (int, np.integer)):
                dummy = torch.ones(1, self.in_dim, device=self.device)
            elif isinstance(self.in_dim, (list, tuple)):
                dummy = torch.ones(1, *self.in_dim, device=self.device)
            else:
                dummy = torch.ones(1, self.in_dim, device=self.device)
            out = self.body(dummy)
            if isinstance(out, (list, tuple)):
                out = out[0]
            # Flatten if needed (for conv outputs)
            if out.dim() > 2:
                return out.view(out.size(0), -1).shape[-1]
            return out.shape[-1]

    def forward(self, x):
        """Forward pass: body -> tails"""
        if self.normalize:
            x = x / 255.0
        x = self.body(x)
        if isinstance(x, (list, tuple)):
            x = x[0]
        # Flatten if needed (for conv body outputs)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return net_util.forward_tails(x, self.tails, self.log_std)
