from slm_lab.agent import net
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.sarsa import SARSA
from slm_lab.agent.algorithm.dqn import DQN
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import torch

logger = logger.get_logger(__name__)


class HydraDQN(DQN):
    '''Multi-task DQN with separate state and action processors per environment'''

    @lab_api
    def init_nets(self, global_nets=None):
        '''Initialize nets with multi-task dimensions, and set net params'''
        # NOTE: Separate init from MultitaskDQN despite similarities so that this implementation can support arbitrary sized state and action heads (e.g. multiple layers)
        self.state_dims = in_dims = [body.state_dim for body in self.agent.nanflat_body_a]
        self.action_dims = out_dims = [body.action_dim for body in self.agent.nanflat_body_a]
        if global_nets is None:
            NetClass = getattr(net, self.net_spec['type'])
            self.net = NetClass(self.net_spec, in_dims, out_dims)
            self.target_net = NetClass(self.net_spec, in_dims, out_dims)
            self.net_names = ['net', 'target_net']
        else:
            util.set_attr(self, global_nets)
            self.net_names = list(global_nets.keys())
        self.post_init_nets()
        self.online_net = self.target_net
        self.eval_net = self.target_net

    @lab_api
    def calc_pdparam(self, xs, evaluate=True, net=None):
        '''
        Calculate pdparams for multi-action by chunking the network logits output
        '''
        pdparam = SARSA.calc_pdparam(self, xs, evaluate=evaluate, net=net)
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
        xs = [torch.from_numpy(state).float() for state in states]
        pdparam = self.calc_pdparam(xs, evaluate=False)
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
        '''Samples a batch per body, which may experience different environment'''
        batch = {k: [] for k in self.body.memory.data_keys}
        for body in self.agent.nanflat_body_a:
            body_batch = body.memory.sample()
            if self.normalize_state:
                body_batch = policy_util.normalize_states_and_next_states(body, body_batch)
            body_batch = util.to_torch_batch(body_batch, self.net.device, body.memory.is_episodic)
            for k, arr in batch.items():
                arr.append(body_batch[k])
        return batch

    def calc_q_loss(self, batch):
        '''Compute the Q value loss for Hydra network by apply the singleton logic on generalized aggregate.'''
        q_preds = torch.stack(self.net.wrap_eval(batch['states']))
        act_q_preds = q_preds.gather(-1, torch.stack(batch['actions']).long().unsqueeze(-1)).squeeze(-1)
        # Use online_net to select actions in next state
        online_next_q_preds = torch.stack(self.online_net.wrap_eval(batch['next_states']))
        # Use eval_net to calculate next_q_preds for actions chosen by online_net
        next_q_preds = torch.stack(self.eval_net.wrap_eval(batch['next_states']))
        max_next_q_preds = online_next_q_preds.gather(-1, next_q_preds.argmax(dim=-1, keepdim=True)).squeeze(-1)
        max_q_targets = torch.stack(batch['rewards']) + self.gamma * (1 - torch.stack(batch['dones'])) * max_next_q_preds
        q_loss = self.net.loss_fn(act_q_preds, max_q_targets)

        # TODO use the same loss_fn but do not reduce yet
        for body in self.agent.nanflat_body_a:
            if 'Prioritized' in util.get_class_name(body.memory):  # PER
                errors = torch.abs(max_q_targets - act_q_preds)
                body.memory.update_priorities(errors)
        return q_loss

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
        if self.to_train == 1:
            total_loss = torch.tensor(0.0, device=self.net.device)
            for _ in range(self.training_epoch):
                batch = self.space_sample()
                for _ in range(self.training_batch_epoch):
                    loss = self.calc_q_loss(batch)
                    self.net.training_step(loss=loss, lr_clock=self.body.env.clock)
                    total_loss += loss
            loss = total_loss / (self.training_epoch * self.training_batch_epoch)
            # reset
            self.to_train = 0
            for body in self.agent.nanflat_body_a:
                body.entropies = []
                body.log_probs = []
            logger.debug(f'Trained {self.name} at epi: {self.body.env.clock.get("epi")}, total_t: {self.body.env.clock.get("total_t")}, t: {self.body.env.clock.get("t")}, total_reward so far: {self.body.memory.total_reward}, loss: {loss:.8f}')

            return loss.item()
        else:
            return np.nan
