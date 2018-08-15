from copy import deepcopy
from functools import partial
from slm_lab.agent import net, memory
from slm_lab.agent.algorithm import math_util, policy_util
from slm_lab.agent.algorithm.actor_critic import ActorCritic
from slm_lab.agent.algorithm.ppo import PPO
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import torch
import pydash as ps

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
        "action_policy_update": "no_update",
        "explore_var_start": null,
        "explore_var_end": null,
        "explore_anneal_epi": null,
        "gamma": 0.99,
        "use_gae": true,
        "lam": 1.0,
        "use_nstep": false,
        "num_step_returns": 100,
        "add_entropy": true,
        "entropy_coef": 0.01,
        "policy_loss_coef": 1.0,
        "val_loss_coef": 0.01,
        "sil_policy_loss_coef": 1.0,
        "sil_val_loss_coef": 0.01,
        "continuous_action_clip": 2.0,
        "training_batch_epoch": 8,
        "training_frequency": 1,
        "training_epoch": 8
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
    @lab_api
    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        self.body = self.agent.nanflat_body_a[0]  # single-body algo
        # create the extra replay memory for SIL
        memory_name = self.memory_spec['sil_replay_name']
        MemoryClass = getattr(memory, memory_name)
        self.body.replay_memory = MemoryClass(self.memory_spec, self, self.body)
        super(SIL, self).post_body_init()

    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
        # set default
        util.set_attr(self, dict(
            action_pdtype='default',
            action_policy='default',
            action_policy_update='no_update',
            explore_var_start=np.nan,
            explore_var_end=np.nan,
            explore_anneal_epi=np.nan,
            policy_loss_coef=1.0,
            val_loss_coef=1.0,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            # theoretically, AC does not have policy update; but in this implementation we have such option
            'action_policy_update',
            'explore_var_start',
            'explore_var_end',
            'explore_anneal_epi',
            'gamma',  # the discount factor
            'use_gae',
            'lam',
            'use_nstep',
            'num_step_returns',
            'add_entropy',
            'entropy_coef',
            'policy_loss_coef',
            'val_loss_coef',
            'sil_policy_loss_coef',
            'sil_val_loss_coef',
            'continuous_action_clip',
            'training_frequency',
            'training_batch_epoch',
            'training_epoch',
        ])
        super(SIL, self).init_algorithm_params()

    def sample(self):
        '''Modify the onpolicy sample to also append to replay'''
        batches = [body.memory.sample() for body in self.agent.nanflat_body_a]
        batch = util.concat_batches(batches)
        data_keys = self.body.replay_memory.data_keys
        for idx in range(len(batch['dones'])):
            tuples = [batch[k][idx] for k in data_keys]
            self.body.replay_memory.add_experience(*tuples)
        batch = util.to_torch_batch(batch, self.net.gpu)
        return batch

    def replay_sample(self):
        '''Samples a batch from memory'''
        batches = [body.replay_memory.sample() for body in self.agent.nanflat_body_a]
        batch = util.concat_batches(batches)
        batch = util.to_torch_batch(batch, self.net.gpu)
        assert not torch.isnan(batch['states']).any(), batch['states']
        return batch

    def calc_sil_policy_val_loss(self, batch):
        '''
        Calculate the SIL policy losses for actor and critic
        sil_policy_loss = -log_prob * max(R - v_pred, 0)
        sil_val_loss = (max(R - v_pred, 0)^2) / 2
        This is called on a randomly-sample batch from experience replay
        '''
        returns = math_util.calc_returns(batch, self.gamma)
        v_preds = self.calc_v(batch['states'], evaluate=False)
        clipped_advs = torch.clamp(returns - v_preds, min=0.0)
        log_probs = policy_util.calc_log_probs(self, self.net, self.body, batch)

        sil_policy_loss = self.sil_policy_loss_coef * torch.mean(- log_probs * clipped_advs)
        sil_val_loss = self.sil_val_loss_coef * torch.pow(clipped_advs, 2) / 2
        sil_val_loss = torch.mean(sil_val_loss)

        if torch.cuda.is_available() and self.net.gpu:
            sil_policy_loss = sil_policy_loss.cuda()
            sil_val_loss = sil_val_loss.cuda()
        logger.debug(f'SIL actor policy loss: {sil_policy_loss:.4f}')
        logger.debug(f'SIL critic value loss: {sil_val_loss:.4f}')
        return sil_policy_loss, sil_val_loss

    def train_shared(self):
        '''
        Trains the network when the actor and critic share parameters
        '''
        if self.to_train == 1:
            # onpolicy update
            super_loss = super(SIL, self).train_shared()
            # offpolicy sil update with random minibatch
            total_sil_loss = torch.tensor(0.0)
            for _ in range(self.training_epoch):
                batch = self.replay_sample()
                sil_policy_loss, sil_val_loss = self.calc_sil_policy_val_loss(batch)
                sil_loss = sil_policy_loss + sil_val_loss
                self.net.training_step(loss=sil_loss, global_net=self.global_nets.get('net'))
                total_sil_loss += sil_loss.cpu()
            sil_loss = total_sil_loss / self.training_epoch
            loss = super_loss + sil_loss
            logger.debug(f'Loss: {loss:.4f}')
            self.last_loss = loss.item()
        return self.last_loss

    def train_separate(self):
        '''
        Trains the network when the actor and critic are separate networks
        '''
        if self.to_train == 1:
            # onpolicy update
            super_loss = super(SIL, self).train_separate()
            # offpolicy sil update with random minibatch
            total_sil_loss = torch.tensor(0.0)
            for _ in range(self.training_epoch):
                batch = self.replay_sample()
                sil_policy_loss, sil_val_loss = self.calc_sil_policy_val_loss(batch)
                self.net.training_step(loss=sil_policy_loss, retain_graph=True, global_net=self.global_nets.get('net'))
                self.critic.training_step(loss=sil_val_loss, global_net=self.global_nets.get('critic'))
                total_sil_loss += sil_policy_loss + sil_val_loss
            sil_loss = total_sil_loss / self.training_epoch
            loss = super_loss + sil_loss
            logger.debug(f'Loss: {loss:.4f}')
            self.last_loss = loss.item()
        return self.last_loss


class PPOSIL(PPO):
    '''
    SIL extended from PPO

    e.g. algorithm_spec
    "algorithm": {
        "name": "PPOSIL",
        "action_pdtype": "default",
        "action_policy": "default",
        "action_policy_update": "no_update",
        "explore_var_start": null,
        "explore_var_end": null,
        "explore_anneal_epi": null,
        "gamma": 0.99,
        "lam": 1.0,
        "clip_eps": 0.10,
        "entropy_coef": 0.02,
        "sil_policy_loss_coef": 1.0,
        "sil_val_loss_coef": 0.01,
        "training_frequency": 1,
        "training_batch_epoch": 8,
        "training_epoch": 8
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
    @lab_api
    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        self.body = self.agent.nanflat_body_a[0]  # single-body algo
        # create the extra replay memory for SIL
        memory_name = self.memory_spec['sil_replay_name']
        MemoryClass = getattr(memory, memory_name)
        self.body.replay_memory = MemoryClass(self.memory_spec, self, self.body)
        super(PPOSIL, self).post_body_init()

    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
        # set default
        util.set_attr(self, dict(
            action_pdtype='default',
            action_policy='default',
            action_policy_update='no_update',
            explore_var_start=np.nan,
            explore_var_end=np.nan,
            explore_anneal_epi=np.nan,
            val_loss_coef=1.0,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            # theoretically, PPO does not have policy update; but in this implementation we have such option
            'action_policy_update',
            'explore_var_start',
            'explore_var_end',
            'explore_anneal_epi',
            'gamma',
            'lam',
            'clip_eps',
            'entropy_coef',
            'val_loss_coef',
            'sil_policy_loss_coef',
            'sil_val_loss_coef',
            'training_frequency',
            'training_batch_epoch',
            'training_epoch',
        ])
        super(PPOSIL, self).init_algorithm_params()

    def sample(self):
        '''Modify the onpolicy sample to also append to replay'''
        batches = [body.memory.sample() for body in self.agent.nanflat_body_a]
        batch = util.concat_batches(batches)
        data_keys = self.body.replay_memory.data_keys
        for idx in range(len(batch['dones'])):
            tuples = [batch[k][idx] for k in data_keys]
            self.body.replay_memory.add_experience(*tuples)
        batch = util.to_torch_batch(batch, self.net.gpu)
        return batch

    def replay_sample(self):
        '''Samples a batch from memory'''
        batches = [body.replay_memory.sample() for body in self.agent.nanflat_body_a]
        batch = util.concat_batches(batches)
        batch = util.to_torch_batch(batch, self.net.gpu)
        assert not torch.isnan(batch['states']).any(), batch['states']
        return batch

    def calc_sil_policy_val_loss(self, batch):
        '''
        Calculate the SIL policy losses for actor and critic
        sil_policy_loss = -log_prob * max(R - v_pred, 0)
        sil_val_loss = (max(R - v_pred, 0)^2) / 2
        This is called on a randomly-sample batch from experience replay
        '''
        returns = math_util.calc_returns(batch, self.gamma)
        v_preds = self.calc_v(batch['states'], evaluate=False)
        clipped_advs = torch.clamp(returns - v_preds, min=0.0)
        log_probs = policy_util.calc_log_probs(self, self.net, self.body, batch)

        sil_policy_loss = self.sil_policy_loss_coef * torch.mean(- log_probs * clipped_advs)
        sil_val_loss = self.sil_val_loss_coef * torch.pow(clipped_advs, 2) / 2
        sil_val_loss = torch.mean(sil_val_loss)

        if torch.cuda.is_available() and self.net.gpu:
            sil_policy_loss = sil_policy_loss.cuda()
            sil_val_loss = sil_val_loss.cuda()
        logger.debug(f'SIL actor policy loss: {sil_policy_loss:.4f}')
        logger.debug(f'SIL critic value loss: {sil_val_loss:.4f}')
        return sil_policy_loss, sil_val_loss

    def train_shared(self):
        '''
        Trains the network when the actor and critic share parameters
        '''
        if self.to_train == 1:
            # onpolicy update
            super_loss = super(PPOSIL, self).train_shared()
            # offpolicy sil update with random minibatch
            total_sil_loss = torch.tensor(0.0)
            for _ in range(self.training_epoch):
                batch = self.replay_sample()
                sil_policy_loss, sil_val_loss = self.calc_sil_policy_val_loss(batch)
                sil_loss = sil_policy_loss + sil_val_loss
                self.net.training_step(loss=sil_loss, global_net=self.global_nets.get('net'))
                total_sil_loss += sil_loss.cpu()
            sil_loss = total_sil_loss / self.training_epoch
            loss = super_loss + sil_loss
            logger.debug(f'Loss: {loss:.4f}')
            self.last_loss = loss.item()
        return self.last_loss

    def train_separate(self):
        '''
        Trains the network when the actor and critic are separate networks
        '''
        if self.to_train == 1:
            # onpolicy update
            super_loss = super(PPOSIL, self).train_separate()
            # offpolicy sil update with random minibatch
            total_sil_loss = torch.tensor(0.0)
            for _ in range(self.training_epoch):
                batch = self.replay_sample()
                sil_policy_loss, sil_val_loss = self.calc_sil_policy_val_loss(batch)
                self.net.training_step(loss=sil_policy_loss, retain_graph=True, global_net=self.global_nets.get('net'))
                self.critic.training_step(loss=sil_val_loss, global_net=self.global_nets.get('critic'))
                total_sil_loss += sil_policy_loss + sil_val_loss
            sil_loss = total_sil_loss / self.training_epoch
            loss = super_loss + sil_loss
            logger.debug(f'Loss: {loss:.4f}')
            self.last_loss = loss.item()
        return self.last_loss
