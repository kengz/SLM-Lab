from copy import deepcopy
from functools import partial
from slm_lab.agent import net, memory
from slm_lab.agent.algorithm import math_util, policy_util
from slm_lab.agent.algorithm.actor_critic import ActorCritic
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
        "val_loss_coef": 1.0,
        "continuous_action_clip": 2.0,
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
        self.init_algorithm_params()
        self.init_nets()
        logger.info(util.self_desc(self))

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
            'continuous_action_clip',
            'training_frequency',
            'training_epoch',
        ])
        self.to_train = 0
        self.action_policy = getattr(policy_util, self.action_policy)
        self.action_policy_update = getattr(policy_util, self.action_policy_update)
        for body in self.agent.nanflat_body_a:
            body.explore_var = self.explore_var_start
        self.calc_advs_v_targets = self.calc_gae_advs_v_targets

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
        assert not torch.isnan(batch['states']).any()
        return batch

    def calc_log_probs(self, batch):
        '''Helper method to calculate log_probs for a randomly sampled batch'''
        states, actions = batch['states'], batch['actions']
        # get ActionPD, don't append to state_buffer
        ActionPD, _pdparam, _body = policy_util.init_action_pd(states[0].cpu().numpy(), self, self.body, append=False)
        # construct log_probs for each state-action
        pdparams = self.calc_pdparam(states)
        log_probs = []
        for idx, pdparam in enumerate(pdparams):
            _action, action_pd = policy_util.sample_action_pd(ActionPD, pdparam, self.body)
            log_prob = action_pd.log_prob(actions[idx])
            log_probs.append(log_prob)
        log_probs = torch.tensor(log_probs)
        return log_probs

    def calc_sil_policy_val_loss(self, batch):
        '''
        Calculate the SIL policy losses for actor and critic
        sil_policy_loss = -log_prob * max(R - v_pred, 0)
        sil_val_loss = norm(max(R - v_pred, 0)) / 2
        This is called on a randomly-sample batch from experience replay
        '''
        returns = math_util.calc_returns(batch, self.gamma)
        v_preds = self.calc_v(batch['states'])
        clipped_advs = torch.clamp(returns - v_preds, min=0.0)
        log_probs = self.calc_log_probs(batch)

        sil_policy_loss = self.policy_loss_coef * torch.mean(- log_probs * v_preds)
        sil_val_loss = self.val_loss_coef * torch.norm(clipped_advs ** 2) / 2

        if torch.cuda.is_available() and self.net.gpu:
            sil_policy_loss = sil_policy_loss.cuda()
            sil_val_loss = sil_val_loss.cuda()
        return sil_policy_loss, sil_val_loss

    def train_shared(self):
        '''
        Trains the network when the actor and critic share parameters
        '''
        if self.to_train == 1:
            # onpolicy a2c update
            a2c_loss = super(SIL, self).train_shared()
            # offpolicy sil update with random minibatch
            total_sil_loss = torch.tensor(0.0)
            for _ in range(self.training_epoch):
                batch = self.replay_sample()
                sil_policy_loss, sil_val_loss = self.calc_sil_policy_val_loss(batch)
                sil_loss = sil_policy_loss + sil_val_loss
                self.net.training_step(loss=sil_loss)
                total_sil_loss += sil_loss.cpu()
            sil_loss = total_sil_loss / self.training_epoch
            loss = a2c_loss + sil_loss
            self.last_loss = loss.item()
        return self.last_loss

    def train_separate(self):
        '''
        Trains the network when the actor and critic are separate networks
        '''
        if self.to_train == 1:
            # onpolicy a2c update
            a2c_loss = super(SIL, self).train_separate()
            # offpolicy sil update with random minibatch
            total_sil_loss = torch.tensor(0.0)
            for _ in range(self.training_epoch):
                batch = self.replay_sample()
                sil_policy_loss, sil_val_loss = self.calc_sil_policy_val_loss(batch)
                self.net.training_step(loss=sil_policy_loss, retain_graph=True)
                self.critic.training_step(loss=sil_val_loss)
                total_sil_loss += sil_policy_loss + sil_val_loss
            sil_loss = total_sil_loss / self.training_epoch
            loss = a2c_loss + sil_loss
            self.last_loss = loss.item()
        return self.last_loss
