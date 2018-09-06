from slm_lab.agent import net
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.sarsa import SARSA
from slm_lab.agent.algorithm.dqn import DQN
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import torch

logger = logger.get_logger(__name__)


class MultitaskDQN(DQN):
    '''
    Simplest Multi-task DQN implementation.
    Multitask is for parallelizing bodies in the same env to get more data
    States and action dimensions are concatenated, and a single shared network is reponsible for processing concatenated states, and generating one action per environment from a single output layer.
    '''

    @lab_api
    def init_nets(self):
        '''Initialize nets with multi-task dimensions, and set net params'''
        self.state_dims = [body.state_dim for body in self.agent.nanflat_body_a]
        self.action_dims = [body.action_dim for body in self.agent.nanflat_body_a]
        in_dim = sum(self.state_dims)
        out_dim = sum(self.action_dims)
        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self.net_spec, in_dim, out_dim)
        self.target_net = NetClass(self.net_spec, in_dim, out_dim)
        self.net_names = ['net', 'target_net']
        self.post_init_nets()
        self.online_net = self.target_net
        self.eval_net = self.target_net

    @lab_api
    def calc_pdparam(self, x, evaluate=True, net=None):
        '''
        Calculate pdparams for multi-action by chunking the network logits output
        '''
        pdparam = super(MultitaskDQN, self).calc_pdparam(x, evaluate=evaluate, net=net)
        pdparam = torch.cat(torch.split(pdparam, self.action_dims, dim=1))
        logger.debug(f'pdparam: {pdparam}')
        return pdparam

    @lab_api
    def space_act(self, state_a):
        '''Non-atomizable act to override agent.act(), do a single pass on the entire state_a instead of composing act() via iteration'''
        # gather and flatten
        states = []
        for eb, body in util.ndenumerate_nonan(self.agent.body_a):
            state = state_a[eb]
            if self.normalize_state:
                state = policy_util.update_online_stats_and_normalize_state(body, state)
            states.append(state)
        state = torch.tensor(states, device=self.net.device).view(-1).unsqueeze_(0).float()
        pdparam = self.calc_pdparam(state, evaluate=False)
        # use multi-policy. note arg change
        action_a, action_pd_a = self.action_policy(states, self, self.agent.nanflat_body_a, pdparam)
        for idx, body in enumerate(self.agent.nanflat_body_a):
            action_pd = action_pd_a[idx]
            body.entropies.append(action_pd.entropy())
            body.log_probs.append(action_pd.log_prob(action_a[idx].float()))
            assert not torch.isnan(body.log_probs[-1])
        return action_a.cpu().numpy()

    @lab_api
    def space_sample(self):
        '''
        Samples a batch from memory.
        Note that multitask's bodies are parallelized copies with similar envs, just to get more batch sizes
        '''
        batches = []
        for body in self.agent.nanflat_body_a:
            body_batch = body.memory.sample()
            # one-hot actions to calc q_targets
            if body.is_discrete:
                body_batch['actions'] = util.to_one_hot(body_batch['actions'], body.action_space.high)
            if self.normalize_state:
                body_batch = policy_util.normalize_states_and_next_states(body, body_batch)
            body_batch = util.to_torch_batch(body_batch, self.net.device, body.memory.is_episodic)
            batches.append(body_batch)
        # Concat states at dim=1 for feedforward
        batch = {
            'states': torch.cat([body_batch['states'] for body_batch in batches], dim=1),
            'next_states': torch.cat([body_batch['next_states'] for body_batch in batches], dim=1),
        }
        # retain body-batches for body-wise q_targets calc
        batch['body_batches'] = batches
        return batch

    def calc_q_targets(self, batch):
        '''Compute the target Q values for multitask network by iterating through the slices corresponding to bodies, and computing the singleton function'''
        q_preds = self.net.wrap_eval(batch['states'])
        # Use online_net to select actions in next state
        online_next_q_preds = self.online_net.wrap_eval(
            batch['next_states'])
        next_q_preds = self.eval_net.wrap_eval(batch['next_states'])
        start_idx = 0
        multi_q_targets = []
        # iterate over body, use slice with proper idx offset
        for b, body_batch in enumerate(batch['body_batches']):
            body = self.agent.nanflat_body_a[b]
            end_idx = start_idx + body.action_dim
            _, action_idxs = torch.max(online_next_q_preds[:, start_idx:end_idx], dim=1)
            # Offset action index properly
            action_idxs += start_idx
            batch_size = len(body_batch['dones'])
            max_next_q_preds = next_q_preds[range(batch_size), action_idxs]
            max_q_targets = body_batch['rewards'] + self.gamma * (1 - body_batch['dones']) * max_next_q_preds
            max_q_targets.unsqueeze_(1)
            q_targets = (max_q_targets * body_batch['actions']) + (q_preds[:, start_idx:end_idx] * (1 - body_batch['actions']))
            multi_q_targets.append(q_targets)
            start_idx = end_idx
        q_targets = torch.cat(multi_q_targets, dim=1)
        logger.debug(f'q_targets: {q_targets}')
        return q_targets


class HydraDQN(MultitaskDQN):
    '''Multi-task DQN with separate state and action processors per environment'''

    @lab_api
    def init_nets(self):
        '''Initialize nets with multi-task dimensions, and set net params'''
        # NOTE: Separate init from MultitaskDQN despite similarities so that this implementation can support arbitrary sized state and action heads (e.g. multiple layers)
        self.state_dims = in_dims = [body.state_dim for body in self.agent.nanflat_body_a]
        self.action_dims = out_dims = [body.action_dim for body in self.agent.nanflat_body_a]
        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self.net_spec, in_dims, out_dims)
        self.target_net = NetClass(self.net_spec, in_dims, out_dims)
        self.net_names = ['net', 'target_net']
        self.post_init_nets()
        self.online_net = self.target_net
        self.eval_net = self.target_net

    @lab_api
    def calc_pdparam(self, x, evaluate=True, net=None):
        '''
        Calculate pdparams for multi-action by chunking the network logits output
        '''
        x = torch.cat(torch.split(x, self.state_dims, dim=1)).unsqueeze_(dim=1)
        pdparam = SARSA.calc_pdparam(self, x, evaluate=evaluate, net=net)
        return pdparam

    @lab_api
    def space_sample(self):
        '''Samples a batch per body, which may experience different environment'''
        batches = []
        for body in self.agent.nanflat_body_a:
            body_batch = body.memory.sample()
            # one-hot actions to calc q_targets
            if body.is_discrete:
                body_batch['actions'] = util.to_one_hot(body_batch['actions'], body.action_space.high)
            if self.normalize_state:
                body_batch = policy_util.normalize_states_and_next_states(body, body_batch)
            body_batch = util.to_torch_batch(body_batch, self.net.device, body.memory.is_episodic)
            batches.append(body_batch)
        # collect per body for feedforward to hydra heads
        batch = {
            'states': [body_batch['states'] for body_batch in batches],
            'next_states': [body_batch['next_states'] for body_batch in batches],
        }
        # retain body-batches for body-wise q_targets calc
        batch['body_batches'] = batches
        return batch

    def calc_q_targets(self, batch):
        '''Compute the target Q values for hydra network by iterating through the tails corresponding to bodies, and computing the singleton function'''
        q_preds = self.net.wrap_eval(batch['states'])
        online_next_q_preds = self.online_net.wrap_eval(batch['next_states'])
        next_q_preds = self.eval_net.wrap_eval(batch['next_states'])
        multi_q_targets = []
        # iterate over body, use proper output tail
        for b, body_batch in enumerate(batch['body_batches']):
            _, action_idxs = torch.max(online_next_q_preds[b], dim=1)
            batch_size = len(body_batch['dones'])
            max_next_q_preds = next_q_preds[b][range(batch_size), action_idxs]
            max_q_targets = body_batch['rewards'] + self.gamma * (1 - body_batch['dones']) * max_next_q_preds
            max_q_targets.unsqueeze_(1)
            q_targets = (max_q_targets * body_batch['actions']) + (q_preds[b] * (1 - body_batch['actions']))
            multi_q_targets.append(q_targets)
        # return as list for compatibility with net output in training_step
        q_targets = multi_q_targets
        logger.debug(f'q_targets: {q_targets}')
        return q_targets

    @lab_api
    def space_train(self):
        '''
        Completes one training step for the agent if it is time to train.
        i.e. the environment timestep is greater than the minimum training timestep and a multiple of the training_frequency.
        Each training step consists of sampling n batches from the agent's memory.
        For each of the batches, the target Q values (q_targets) are computed and a single training step is taken k times
        Otherwise this function does nothing.
        '''
        if util.get_lab_mode() == 'enjoy':
            return np.nan
        total_t = util.s_get(self, 'aeb_space.clock').get('total_t')
        self.to_train = (total_t > self.training_min_timestep and total_t % self.training_frequency == 0)
        is_per = util.get_class_name(self.agent.nanflat_body_a[0].memory) == 'PrioritizedReplay'
        if self.to_train == 1:
            total_loss = torch.tensor(0.0, device=self.net.device)
            for _ in range(self.training_epoch):
                batch = self.space_sample()
                for _ in range(self.training_batch_epoch):
                    with torch.no_grad():
                        q_targets = self.calc_q_targets(batch)
                        if is_per:
                            q_preds = self.net.wrap_eval(batch['states'])
                            errors = torch.abs(q_targets - q_preds)
                            errors = errors.sum(dim=1).unsqueeze_(dim=1)
                            for body in self.agent.nanflat_body_a:
                                body.memory.update_priorities(errors)
                    loss = self.net.training_step(batch['states'], q_targets, global_net=self.global_nets.get('net'))
                    total_loss += loss
            loss = total_loss / (self.training_epoch * self.training_batch_epoch)
            # reset
            self.to_train = 0
            for body in self.agent.nanflat_body_a:
                body.entropies = []
                body.log_probs = []
            logger.info(f'Trained {self.name}, loss: {loss:.4f}')
            return loss.item()
        else:
            return np.nan
