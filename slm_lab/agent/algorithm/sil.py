from slm_lab.agent import memory
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.actor_critic import ActorCritic
from slm_lab.agent.algorithm.ppo import PPO
from slm_lab.lib import logger, math_util, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import torch

logger = logger.get_logger(__name__)


class SIL(ActorCritic):
    '''
    Implementation of Self-Imitation Learning (SIL) https://arxiv.org/abs/1806.05635
    This is actually just A2C with an extra SIL loss function

    e.g. algorithm_spec
    "algorithm": {
        "name": "SIL",
        "action_pdtype": "default",
        "action_policy": "default",
        "explore_var_spec": null,
        "gamma": 0.99,
        "lam": 0.95,
        "num_step_returns": 100,
        "entropy_coef_spec": {
          "name": "linear_decay",
          "start_val": 0.01,
          "end_val": 0.001,
          "start_step": 100,
          "end_step": 5000,
        },
        "policy_loss_coef": 1.0,
        "val_loss_coef": 0.01,
        "sil_policy_loss_coef": 1.0,
        "sil_val_loss_coef": 0.01,
        "training_batch_iter": 8,
        "training_frequency": 1,
        "training_iter": 8,
    }

    e.g. special memory_spec
    "memory": {
        "name": "OnPolicyReplay",
        "sil_replay_name": "Replay",
        "batch_size": 32,
        "max_size": 10000,
        "use_cer": true
    }
    '''

    def __init__(self, agent, global_nets=None):
        super().__init__(agent, global_nets)
        # create the extra replay memory for SIL
        MemoryClass = getattr(memory, self.memory_spec['sil_replay_name'])
        self.replay_memory = MemoryClass(self.memory_spec, self.agent)

    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
        # set default
        util.set_attr(self, dict(
            action_pdtype='default',
            action_policy='default',
            explore_var_spec=None,
            entropy_coef_spec=None,
            policy_loss_coef=1.0,
            val_loss_coef=1.0,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            # theoretically, AC does not have policy update; but in this implementation we have such option
            'explore_var_spec',
            'gamma',  # the discount factor
            'lam',
            'num_step_returns',
            'entropy_coef_spec',
            'policy_loss_coef',
            'val_loss_coef',
            'sil_policy_loss_coef',
            'sil_val_loss_coef',
            'training_frequency',
            'training_batch_iter',
            'training_iter',
        ])
        super().init_algorithm_params()

    def sample(self):
        '''Modify the onpolicy sample to also append to replay'''
        batch = self.agent.memory.sample()
        if self.agent.memory.is_episodic:
            batch = {k: np.concatenate(v) for k, v in batch.items()}  # concat episodic memory
        # Convert lists to numpy arrays if needed (OnPolicyBatchReplay returns numpy arrays)
        batch = {k: np.array(v) if isinstance(v, list) else v for k, v in batch.items()}
        # Flatten venv dimension: reshape from (time_horizon, num_envs, ...) to (time_horizon * num_envs, ...)
        flat_batch = {}
        for k, v in batch.items():
            if self.agent.env.is_venv and v.ndim > 1:
                # Flatten first two dims (time_horizon, num_envs) into one
                flat_batch[k] = v.reshape(-1, *v.shape[2:]) if v.ndim > 2 else v.reshape(-1)
            else:
                flat_batch[k] = v
        # Map batch keys (plural) to add_experience keys (singular)
        key_map = {
            'states': 'state', 'actions': 'action', 'rewards': 'reward',
            'next_states': 'next_state', 'dones': 'done',
            'terminateds': 'terminated', 'truncateds': 'truncated',
        }
        for idx in range(len(flat_batch['dones'])):
            # Build kwargs dict from batch data, mapping plural to singular keys
            kwargs = {key_map.get(k, k): flat_batch[k][idx] for k in flat_batch.keys() if k in key_map}
            self.replay_memory.add_experience(**kwargs)
        batch = util.to_torch_batch(batch, self.net.device, self.replay_memory.is_episodic)
        return batch

    def replay_sample(self):
        '''Samples a batch from memory'''
        batch = self.replay_memory.sample()
        batch = util.to_torch_batch(batch, self.net.device, self.replay_memory.is_episodic)
        return batch

    def calc_pdparam_v_flat(self, batch):
        '''Like calc_pdparam_v but for flat replay data (no venv_unpack needed)'''
        states = batch['states']
        # Note: replay data is already flat (batch_size, state_dim), no venv_unpack needed
        pdparam = self.calc_pdparam(states)
        v_pred = self.calc_v(states, use_cache=False)
        return pdparam, v_pred

    def calc_sil_policy_val_loss(self, batch, pdparams):
        '''
        Calculate the SIL policy losses for actor and critic
        sil_policy_loss = -log_prob * max(R - v_pred, 0)
        sil_val_loss = (max(R - v_pred, 0)^2) / 2
        This is called on a randomly-sample batch from experience replay
        '''
        v_preds = self.calc_v(batch['states'], use_cache=False)
        rets = math_util.calc_returns(batch['rewards'], batch['terminateds'], self.gamma)
        clipped_advs = torch.clamp(rets - v_preds, min=0.0)

        action_pd = policy_util.init_action_pd(self.agent.ActionPD, pdparams)
        # Note: replay memory stores flat experiences (not venv-packed), so no venv_unpack needed
        actions = batch['actions']
        log_probs = policy_util.reduce_multi_action(action_pd.log_prob(actions))

        sil_policy_loss = - self.sil_policy_loss_coef * (log_probs * clipped_advs.detach()).mean()
        sil_val_loss = self.sil_val_loss_coef * clipped_advs.pow(2).mean() / 2
        logger.debug(f'SIL actor policy loss: {sil_policy_loss:g}')
        logger.debug(f'SIL critic value loss: {sil_val_loss:g}')
        return sil_policy_loss, sil_val_loss

    def train(self):
        if self.to_train == 1:
            # onpolicy update
            super_loss = super().train()
            # offpolicy sil update with random minibatch
            total_sil_loss = torch.tensor(0.0, device=self.net.device)
            for _ in range(self.training_iter):
                batch = self.replay_sample()
                for _ in range(self.training_batch_iter):
                    pdparams, _v_preds = self.calc_pdparam_v_flat(batch)
                    sil_policy_loss, sil_val_loss = self.calc_sil_policy_val_loss(batch, pdparams)
                    sil_loss = sil_policy_loss + sil_val_loss
                    self.net.train_step(sil_loss, self.optim, self.lr_scheduler, global_net=self.global_net)
                    self.agent.env.tick_opt_step()
                    total_sil_loss += sil_loss
            sil_loss = total_sil_loss / self.training_iter
            loss = super_loss + sil_loss
            logger.debug(f'Trained {self.name} at epi: {self.agent.env.get("epi")}, frame: {self.agent.env.get("frame")}, t: {self.agent.env.get("t")}, total_reward so far: {self.agent.env.total_reward}, loss: {loss:g}')
            return loss.item()
        else:
            return np.nan


class PPOSIL(SIL, PPO):
    '''
    SIL extended from PPO. This will call the SIL methods and use PPO as super().

    e.g. algorithm_spec
    "algorithm": {
        "name": "PPOSIL",
        "action_pdtype": "default",
        "action_policy": "default",
        "explore_var_spec": null,
        "gamma": 0.99,
        "lam": 0.95,
        "clip_eps_spec": {
          "name": "linear_decay",
          "start_val": 0.01,
          "end_val": 0.001,
          "start_step": 100,
          "end_step": 5000,
        },
        "entropy_coef_spec": {
          "name": "linear_decay",
          "start_val": 0.01,
          "end_val": 0.001,
          "start_step": 100,
          "end_step": 5000,
        },
        "sil_policy_loss_coef": 1.0,
        "sil_val_loss_coef": 0.01,
        "time_horizon": 32,
        "training_batch_iter": 8,
        "training_iter": 8,
        "training_epoch": 8,
    }

    e.g. special memory_spec
    "memory": {
        "name": "OnPolicyReplay",
        "sil_replay_name": "Replay",
        "batch_size": 32,
        "max_size": 10000,
        "use_cer": true
    }
    '''
    pass
