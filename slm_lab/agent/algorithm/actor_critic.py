from slm_lab.agent import memory
from slm_lab.agent import net
from slm_lab.agent.algorithm.algorithm_util import act_fns
from slm_lab.agent.algorithm.reinforce import ReinforceDiscrete
from slm_lab.agent.net import net_util
from slm_lab.lib import util
from torch.autograd import Variable
import numpy as np
import torch
import pydash as _


class ACDiscrete(ReinforceDiscrete):
    '''
    TODO
    Adapted from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
    '''

    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        # TODO generalize
        default_body = self.agent.bodies[0]
        # autoset net head and tail
        # TODO auto-architecture to handle multi-head, multi-tail nets
        state_dim = default_body.state_dim
        action_dim = default_body.action_dim
        net_spec = self.agent.spec['net']
        self.actor = getattr(net, net_spec['type'])(
            state_dim, net_spec['hid_layers'], action_dim,
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim_actor'),
            loss_param=_.get(net_spec, 'loss'), # Note: Not used for PG algos
            clamp_grad=_.get(net_spec, 'clamp_grad'),
        )
        print(self.actor)
        self.critic = getattr(net, net_spec['type'])(
            state_dim, net_spec['hid_layers'], 1,
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim_critic'),
            loss_param=_.get(net_spec, 'loss'), # Note: Not used for PG algos
            clamp_grad=_.get(net_spec, 'clamp_grad'),
        )
        print(self.critic)
        algorithm_spec = self.agent.spec['algorithm']
        self.action_policy = act_fns[algorithm_spec['action_policy']]
        self.training_frequency = algorithm_spec['training_frequency']
        self.training_iters_per_batch = algorithm_spec['training_iters_per_batch']
        self.gamma = algorithm_spec['gamma']
        # To save on a forward pass keep the log probs
        # from each action
        self.saved_log_probs = []
        self.to_train = 0

    def body_act_discrete(self, body, body_state):
        # TODO can handle identical bodies now; to use body_net for specific body.
        return self.action_policy(self.agent, body, body_state, self.actor)

    def train(self):
        if self.to_train == 1:
            # Only care about the rewards
            batch = self.agent.memory.get_batch()
            # print("Batch size: {}".format(len(batch['rewards'])))
            batch = self.convert_to_variables(batch)
            critic_loss = self.train_critic(batch)
            actor_loss = self.train_actor(batch)
            total_loss = critic_loss + actor_loss
            print("Losses: Critic: {:.2f}, Actor: {:.2f}, Total: {:.2f}".format(
                critic_loss, actor_loss, total_loss
            ))
            return total_loss
        else:
            return None

    def convert_to_variables(self, batch):
        # Package data into pytorch variables
        float_data_list = [
            'states', 'actions', 'rewards', 'dones', 'next_states']
        for k in float_data_list:
            if k == 'dones':
                batch[k] = Variable(torch.from_numpy(np.asarray(batch[k], dtype=int)).float())
            else:
                batch[k] = Variable(torch.from_numpy(np.asarray(batch[k])).float())
            if k == 'rewards' or k == 'dones':
                batch[k].data.unsqueeze_(dim=1)
            # print("{} dims: {}".format(k, batch[k].size()))
            # print(batch[k])
        return batch

    def train_critic(self, batch):
        state_vals = self.critic.wrap_eval(batch['states'])
        next_state_vals = self.critic.wrap_eval(batch['next_states'])
        # print("state: {}".format(state_vals.size()))
        # print("next state: {}".format(next_state_vals.size()))
        # Add reward and discount
        next_state_vals = batch['rewards'].data + self.gamma * \
            torch.mul((1 - batch['dones'].data), next_state_vals)
        # print("next state: {}".format(next_state_vals.size()))
        y = Variable(next_state_vals)
        # Train critic
        loss = 0
        for _i in range(self.training_iters_per_batch):
            loss = self.critic.training_step(batch['states'], y).data[0]
        return loss

    def calculate_advantage(self, batch):
        state_vals = self.critic.wrap_eval(batch['states'])
        next_state_vals = self.critic.wrap_eval(batch['next_states'])
        advantage = batch['rewards'].data + self.gamma * \
            torch.mul((1 - batch['dones'].data), next_state_vals) - state_vals
        advantage.squeeze_()
        # print("Advantage: {}".format(type(advantage)))
        # print("Shape: {}".format(advantage.size()))
        return advantage

    def train_actor(self, batch):
        advantage = self.calculate_advantage(batch)
        assert len(self.saved_log_probs) == advantage.size(0)
        policy_loss = []
        for log_prob, a in zip(self.saved_log_probs, advantage):
            policy_loss.append(-log_prob * a)
        self.actor.optim.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        loss = policy_loss.data[0]
        policy_loss.backward()
        self.actor.optim.step()
        self.to_train = 0
        self.saved_log_probs = []
        return loss
