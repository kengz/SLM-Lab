from slm_lab.agent import net, memory
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.actor_critic import ActorCritic
from slm_lab.agent.algorithm.ppo import PPO
from slm_lab.lib import logger, math_util, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import pydash as ps
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
        "lam": 1.0,
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
        "training_batch_epoch": 8,
        "training_frequency": 1,
        "training_epoch": 8,
        "normalize_state": true
    }

    e.g. special memory_spec
    "memory": {
        "name": "OnPolicyReplay",
        "sil_replay_name": "SILReplay",
        "batch_size": 32,
        "max_size": 10000,
        "use_cer": true
    }
    '''

    def __init__(self, agent, global_nets=None):
        super(SIL, self).__init__(agent, global_nets)
        # create the extra replay memory for SIL
        MemoryClass = getattr(memory, self.memory_spec['sil_replay_name'])
        self.body.replay_memory = MemoryClass(self.memory_spec, self.body)

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
            'training_batch_epoch',
            'training_epoch',
            'normalize_state'
        ])
        super(SIL, self).init_algorithm_params()

    def sample(self):
        '''Modify the onpolicy sample to also append to replay'''
        batch = self.body.memory.sample()
        batch = {k: np.concatenate(v) for k, v in batch.items()}  # concat episodic memory
        batch['rets'] = math_util.calc_returns(batch, self.gamma)
        for idx in range(len(batch['dones'])):
            tuples = [batch[k][idx] for k in self.body.replay_memory.data_keys]
            self.body.replay_memory.add_experience(*tuples)
        if self.normalize_state:
            batch = policy_util.normalize_states_and_next_states(self.body, batch)
        batch = util.to_torch_batch(batch, self.net.device, self.body.replay_memory.is_episodic)
        return batch

    def replay_sample(self):
        '''Samples a batch from memory'''
        batch = self.body.replay_memory.sample()
        if self.normalize_state:
            batch = policy_util.normalize_states_and_next_states(
                self.body, batch, episodic_flag=self.body.replay_memory.is_episodic)
        batch = util.to_torch_batch(batch, self.net.device, self.body.replay_memory.is_episodic)
        assert not torch.isnan(batch['states']).any(), batch['states']
        return batch

    def calc_sil_policy_val_loss(self, batch):
        '''
        Calculate the SIL policy losses for actor and critic
        sil_policy_loss = -log_prob * max(R - v_pred, 0)
        sil_val_loss = (max(R - v_pred, 0)^2) / 2
        This is called on a randomly-sample batch from experience replay
        '''
        returns = batch['rets']
        v_preds = self.calc_v(batch['states'], evaluate=False)
        clipped_advs = torch.clamp(returns - v_preds, min=0.0)
        log_probs = policy_util.calc_log_probs(self, self.net, self.body, batch)

        sil_policy_loss = self.sil_policy_loss_coef * torch.mean(- log_probs * clipped_advs)
        sil_val_loss = self.sil_val_loss_coef * torch.pow(clipped_advs, 2) / 2
        sil_val_loss = torch.mean(sil_val_loss)
        logger.debug(f'SIL actor policy loss: {sil_policy_loss:.4f}')
        logger.debug(f'SIL critic value loss: {sil_val_loss:.4f}')
        return sil_policy_loss, sil_val_loss

    def train_shared(self):
        '''
        Trains the network when the actor and critic share parameters
        '''
        clock = self.body.env.clock
        if self.to_train == 1:
            # onpolicy update
            super_loss = super(SIL, self).train_shared()
            # offpolicy sil update with random minibatch
            total_sil_loss = torch.tensor(0.0, device=self.net.device)
            for _ in range(self.training_epoch):
                batch = self.replay_sample()
                for _ in range(self.training_batch_epoch):
                    sil_policy_loss, sil_val_loss = self.calc_sil_policy_val_loss(batch)
                    sil_loss = sil_policy_loss + sil_val_loss
                    self.net.training_step(loss=sil_loss, lr_clock=clock)
                    total_sil_loss += sil_loss
            sil_loss = total_sil_loss / self.training_epoch
            loss = super_loss + sil_loss
            logger.debug(f'Trained {self.name} at epi: {clock.epi}, total_t: {clock.total_t}, t: {clock.t}, total_reward so far: {self.body.memory.total_reward}, loss: {loss:.8f}')

            return loss.item()
        else:
            return np.nan

    def train_separate(self):
        '''
        Trains the network when the actor and critic are separate networks
        '''
        clock = self.body.env.clock
        if self.to_train == 1:
            # onpolicy update
            super_loss = super(SIL, self).train_separate()
            # offpolicy sil update with random minibatch
            total_sil_loss = torch.tensor(0.0, device=self.net.device)
            for _ in range(self.training_epoch):
                batch = self.replay_sample()
                for _ in range(self.training_batch_epoch):
                    sil_policy_loss, sil_val_loss = self.calc_sil_policy_val_loss(batch)
                    self.net.training_step(loss=sil_policy_loss, lr_clock=clock, retain_graph=True)
                    self.critic.training_step(loss=sil_val_loss, lr_clock=clock)
                    total_sil_loss += sil_policy_loss + sil_val_loss
            sil_loss = total_sil_loss / self.training_epoch
            loss = super_loss + sil_loss
            logger.debug(f'Trained {self.name} at epi: {clock.epi}, total_t: {clock.total_t}, t: {clock.t}, total_reward so far: {self.body.memory.total_reward}, loss: {loss:.8f}')

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
        "lam": 1.0,
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
        "training_frequency": 1,
        "training_batch_epoch": 8,
        "training_epoch": 8,
        "normalize_state": true
    }

    e.g. special memory_spec
    "memory": {
        "name": "OnPolicyReplay",
        "sil_replay_name": "SILReplay",
        "batch_size": 32,
        "max_size": 10000,
        "use_cer": true
    }
    '''
    pass
