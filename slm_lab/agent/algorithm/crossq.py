import numpy as np
import torch

from slm_lab.agent import net
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.sac import SoftActorCritic
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, math_util
from slm_lab.lib.decorator import lab_api

logger = logger.get_logger(__name__)


class CrossQ(SoftActorCritic):
    """CrossQ: Batch Normalization in Deep RL (Bhatt et al., ICLR 2024).

    Eliminates target networks via cross batch normalization in critics.
    Key differences from SAC:
    - No target networks (BatchNorm provides sufficient regularization)
    - Cross batch norm: current (s,a) and next (s',a') share BN statistics
    - UTD=1 (training_iter=1) — 20x fewer gradient steps for same performance
    """

    @lab_api
    def init_nets(self, global_nets=None):
        self.shared = False
        # steps_per_schedule: frames processed per scheduler.step() call
        steps_per_schedule = self.training_frequency * self.agent.env.num_envs

        # Actor network (identical to SAC)
        ActorNetClass = getattr(net, self.net_spec["type"])
        self.net = ActorNetClass(
            self.net_spec, self.agent.state_dim, net_util.get_out_dim(self.agent)
        )
        self.optim = net_util.get_optim(self.net, self.net.optim_spec)
        self.lr_scheduler = net_util.get_lr_scheduler(
            self.optim, self.net.lr_scheduler_spec, steps_per_schedule
        )

        # Critic networks — use critic_net_spec if provided, else net_spec
        critic_net_spec = self.agent.agent_spec.get("critic_net", self.net_spec)
        CriticNetClass = getattr(net, critic_net_spec["type"])

        if self.agent.is_discrete:
            q_in_dim, q_out_dim = self.agent.state_dim, self.agent.action_dim
        else:
            q_in_dim = self.agent.state_dim + self.agent.action_dim
            q_out_dim = 1

        self.q1_net = CriticNetClass(critic_net_spec, q_in_dim, q_out_dim)
        self.q2_net = CriticNetClass(critic_net_spec, q_in_dim, q_out_dim)
        if self.spectral_norm:
            net_util.apply_spectral_norm_penultimate(self.q1_net)
            net_util.apply_spectral_norm_penultimate(self.q2_net)
        self.q1_optim = net_util.get_optim(self.q1_net, self.q1_net.optim_spec)
        self.q1_lr_scheduler = net_util.get_lr_scheduler(
            self.q1_optim, self.q1_net.lr_scheduler_spec, steps_per_schedule
        )
        self.q2_optim = net_util.get_optim(self.q2_net, self.q2_net.optim_spec)
        self.q2_lr_scheduler = net_util.get_lr_scheduler(
            self.q2_optim, self.q2_net.lr_scheduler_spec, steps_per_schedule
        )

        # No target networks — this is CrossQ's key distinction
        self.net_names = ["net", "q1_net", "q2_net"]

        self._init_entropy_tuning(steps_per_schedule)

        net_util.set_global_nets(self, global_nets)
        self.end_init_nets()

    def calc_q_cross(self, states, actions, next_states, next_actions, q_net):
        """Cross batch normalization forward pass.

        Concatenates current (s,a) and next (s',a') into a single batch,
        forwards through the critic, then splits. This shares BatchNorm
        statistics between current and next state batches — the core
        innovation that eliminates the need for target networks.
        """
        current = torch.cat([states, actions], dim=-1)
        future = torch.cat([next_states, next_actions], dim=-1)
        batch = torch.cat([current, future], dim=0)
        q_all = q_net(batch)
        q_current, q_next = q_all.chunk(2, dim=0)
        return q_current.view(-1), q_next.view(-1)

    def calc_q_cross_discrete(self, states, next_states, q_net):
        """Cross batch norm forward for discrete actions.

        For discrete actions, Q-network takes only states (outputs Q for all actions).
        """
        batch = torch.cat([states, next_states], dim=0)
        q_all = q_net(batch)
        q_current, q_next = q_all.chunk(2, dim=0)
        return q_current, q_next

    def train(self):
        """Override SAC's train to use cross batch norm forward pass.

        One cross forward in train mode processes current (s,a) and next (s',a')
        together through each critic, sharing BatchNorm statistics. Q_next values
        from this same forward are detached for target computation — no separate
        eval-mode forward needed. This is CrossQ's core mechanism.
        """
        if self.to_train == 1:
            self._anneal_target_entropy()
            self._anneal_alpha()
            for _ in range(self.training_iter):
                batch = self.sample()
                self.agent.env.set_batch_size(len(batch))

                states = batch["states"]
                actions = batch["actions"]
                next_states = batch["next_states"]

                # Get next actions from policy (no gradient through policy for critic update)
                with torch.no_grad():
                    next_pdparams = self.calc_pdparam(next_states)
                    next_action_pd = policy_util.init_action_pd(
                        self.agent.ActionPD, next_pdparams
                    )

                # Cross batch norm forward: one pass through each critic with both
                # current and next batches concatenated. BN statistics are shared.
                self.q1_net.train()
                self.q2_net.train()

                if self.agent.is_discrete:
                    q1_current, q1_next = self.calc_q_cross_discrete(
                        states, next_states, self.q1_net
                    )
                    q2_current, q2_next = self.calc_q_cross_discrete(
                        states, next_states, self.q2_net
                    )
                    q1_preds = q1_current.gather(
                        1, actions.long().unsqueeze(1)
                    ).squeeze(1)
                    q2_preds = q2_current.gather(
                        1, actions.long().unsqueeze(1)
                    ).squeeze(1)
                    q1_all, q2_all = q1_current, q2_current

                    # V(s') from cross forward Q_next (detached — no gradient into targets)
                    with torch.no_grad():
                        next_log_probs = torch.nn.functional.log_softmax(
                            next_action_pd.logits, dim=-1
                        )
                        next_probs = next_log_probs.exp()
                        avg_q_next = (q1_next.detach() + q2_next.detach()) / 2
                        v_next = (
                            next_probs * (avg_q_next - self.alpha * next_log_probs)
                        ).sum(dim=-1)
                else:
                    with torch.no_grad():
                        next_log_probs, next_actions = self.calc_log_prob_action(
                            next_action_pd
                        )
                    q1_preds, q1_next = self.calc_q_cross(
                        states, actions, next_states, next_actions, self.q1_net
                    )
                    q2_preds, q2_next = self.calc_q_cross(
                        states, actions, next_states, next_actions, self.q2_net
                    )
                    q1_all, q2_all = None, None

                    # V(s') from cross forward Q_next (detached)
                    with torch.no_grad():
                        min_q_next = torch.min(q1_next.detach(), q2_next.detach())
                        v_next = min_q_next - self.alpha * next_log_probs

                # Compute targets from cross forward Q_next values
                with torch.no_grad():
                    q_targets = (
                        batch["rewards"]
                        + self.gamma * (1 - batch["terminateds"]) * v_next
                    )

                # Apply symlog compression to Q-values if enabled
                if self.symlog:
                    symlog_targets = math_util.symlog(q_targets)
                    q1_loss = self.net.loss_fn(
                        math_util.symlog(q1_preds), symlog_targets
                    )
                    q2_loss = self.net.loss_fn(
                        math_util.symlog(q2_preds), symlog_targets
                    )
                else:
                    q1_loss = self.net.loss_fn(q1_preds, q_targets)
                    q2_loss = self.net.loss_fn(q2_preds, q_targets)
                self.q1_net.train_step(
                    q1_loss,
                    self.q1_optim,
                    self.q1_lr_scheduler,
                    global_net=self.global_q1_net,
                )

                self.q2_net.train_step(
                    q2_loss,
                    self.q2_optim,
                    self.q2_lr_scheduler,
                    global_net=self.global_q2_net,
                )

                self._train_step += 1
                loss = q1_loss + q2_loss

                # Policy and alpha updates with optional delay
                if self._train_step % self.policy_delay == 0:
                    # Critics in eval mode for policy update: use frozen running stats,
                    # don't corrupt BN statistics with policy-only batch (paper requirement)
                    self.q1_net.eval()
                    self.q2_net.eval()
                    action_pd = policy_util.init_action_pd(
                        self.agent.ActionPD, self.calc_pdparam(states)
                    )
                    policy_loss = self.calc_policy_loss(
                        states, action_pd, q1_all, q2_all
                    )
                    self.net.train_step(
                        policy_loss,
                        self.optim,
                        self.lr_scheduler,
                        global_net=self.global_net,
                    )

                    # Alpha update: skip when using fixed alpha
                    if self.fixed_alpha is None:
                        alpha_loss = self.calc_alpha_loss(action_pd)
                        self.train_alpha(alpha_loss)
                        loss = loss + alpha_loss

                    loss = loss + policy_loss

                self.agent.env.tick_opt_step()
                self.try_update_per(torch.min(q1_preds, q2_preds), q_targets)

            # Step LR schedulers once per training iteration
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.q1_lr_scheduler is not None:
                self.q1_lr_scheduler.step()
            if self.q2_lr_scheduler is not None:
                self.q2_lr_scheduler.step()
            if self.alpha_lr_scheduler is not None:
                self.alpha_lr_scheduler.step()
            # reset
            self.to_train = 0
            logger.debug(
                f"Trained {self.name} at epi: {self.agent.env.get('epi')}, "
                f"frame: {self.agent.env.get('frame')}, t: {self.agent.env.get('t')}, "
                f"total_reward so far: {self.agent.env.total_reward}, loss: {loss.item():g}"
            )
            return loss.item()
        else:
            return np.nan

    def update_nets(self):
        """No-op: CrossQ has no target networks to update."""
        pass
