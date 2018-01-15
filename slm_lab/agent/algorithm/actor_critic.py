from slm_lab.agent import memory
from slm_lab.agent import net
from slm_lab.agent.algorithm.algorithm_util import act_fns
from slm_lab.agent.algorithm.reinforce import ReinforceDiscrete
from slm_lab.agent.net import net_util
from slm_lab.lib import util, logger
from torch.autograd import Variable
import numpy as np
import torch
import pydash as _


class ACDiscrete(ReinforceDiscrete):
    '''
    Implementation of a simple actor-critic algorithm.
    Algorithm:
        1. Collect k examples
            - Train the critic network using these examples
            - Calculate the advantage of each example using the critic
            - Multiply the advantage by the negative of log probability of the action taken
        2. Sum all the values above.
        3. Calculate the gradient of this value with respect to all of the parameters of the actor network
        4. Update the actor network parameters using the gradient
    Separate networks with no shared parameters are used to approximate the actor and critic
    '''

    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        body = self.agent.flat_nonan_body_a[0]  # singleton algo
        state_dim = body.state_dim
        action_dim = body.action_dim
        net_spec = self.agent.spec['net']
        self.actor = getattr(net, net_spec['type'])(
            state_dim, net_spec['hid_layers'], action_dim,
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim_actor'),
            loss_param=_.get(net_spec, 'loss'),  # Note: Not used for PG algos
            clamp_grad=_.get(net_spec, 'clamp_grad'),
            clamp_grad_val=_.get(net_spec, 'clamp_grad_val'),
        )
        print(f'Actor: {self.actor}')
        self.critic = getattr(net, net_spec['type'])(
            state_dim, net_spec['hid_layers'], 1,
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim_critic'),
            loss_param=_.get(net_spec, 'loss'),  # Note: Not used for PG algos
            clamp_grad=_.get(net_spec, 'clamp_grad'),
            clamp_grad_val=_.get(net_spec, 'clamp_grad_val'),
        )
        print(f'Critic: {self.critic}')
        algorithm_spec = self.agent.spec['algorithm']
        self.action_policy = act_fns[algorithm_spec['action_policy']]
        self.training_frequency = algorithm_spec['training_frequency']
        self.training_iters_per_batch = algorithm_spec['training_iters_per_batch']
        self.gamma = algorithm_spec['gamma']
        # To save on a forward pass keep the log probs from each action
        self.saved_log_probs = []
        self.to_train = 0

    def body_act_discrete(self, body, state):
        return self.action_policy(self, state, self.actor)

    def sample(self):
        '''Samples a batch from memory'''
        batches = [body.memory.sample()
                   for body in self.agent.flat_nonan_body_a]
        batch = util.concat_dict(batches)
        util.to_torch_batch(batch)
        return batch

    def train(self):
        if self.to_train == 1:
            batch = self.sample()
            if len(batch['states']) < self.training_frequency:
                logger.debug(f'Small batch, {len(batch["states"])}')
            critic_loss = self.train_critic(batch)
            actor_loss = self.train_actor(batch)
            total_loss = critic_loss + abs(actor_loss)
            logger.debug("Losses: Critic: {:.2f}, Actor: {:.2f}, Total: {:.2f}".format(
                critic_loss, abs(actor_loss), total_loss
            ))
            return total_loss
        else:
            return None

    def train_critic(self, batch):
        loss = 0
        for _i in range(self.training_iters_per_batch):
            state_vals = self.critic.wrap_eval(batch['states'])
            next_state_vals = self.critic.wrap_eval(batch['next_states'])
            next_state_vals.squeeze_()
            # Add reward and discount
            next_state_vals = batch['rewards'].data + self.gamma * \
                torch.mul((1 - batch['dones'].data), next_state_vals)
            next_state_vals.unsqueeze_(1)
            y = Variable(next_state_vals)
            loss = self.critic.training_step(batch['states'], y).data[0]
            logger.debug(f'Critic grad norms: {self.critic.get_grad_norms()}')
        return loss

    def calculate_advantage(self, batch):
        state_vals = self.critic.wrap_eval(batch['states']).squeeze_()
        next_state_vals = self.critic.wrap_eval(
            batch['next_states']).squeeze_()
        advantage = batch['rewards'].data + self.gamma * \
            torch.mul((1 - batch['dones'].data), next_state_vals) - state_vals
        advantage.squeeze_()
        logger.debug(f'Advantage: {advantage.size()}')
        return advantage

    def train_actor(self, batch):
        advantage = self.calculate_advantage(batch)
        if len(self.saved_log_probs) != advantage.size(0):
            # Caused by first reward of episode being nan
            del self.saved_log_probs[0]
            logger.debug('Deleting first log prob in epi')
        assert len(self.saved_log_probs) == advantage.size(0)
        policy_loss = []
        for log_prob, a in zip(self.saved_log_probs, advantage):
            logger.debug(f'log prob: {log_prob.data[0]}, advantage: {a}')
            policy_loss.append(-log_prob * a)
        self.actor.optim.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        loss = policy_loss.data[0]
        policy_loss.backward()
        if self.actor.clamp_grad:
            logger.info("Clipping actor gradient...")
            torch.nn.utils.clip_grad_norm(
                self.actor.parameters(), self.actor.clamp_grad_val)
        logger.debug(f'Gradient norms: {self.actor.get_grad_norms()}')
        self.actor.optim.step()
        self.to_train = 0
        self.saved_log_probs = []
        logger.debug(f'Policy loss: {loss}')
        return loss


class ACDiscreteSimple(ACDiscrete):
    '''
    Implementation of a simple actor-critic algorithm.
    Similar to ACDiscrete, but uses a different approach to calculating the advantage which follows
    https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
    '''

    def train_critic(self, batch):
        loss = 0
        rewards = []
        raw_rewards = batch['rewards']
        R = 0
        for r in raw_rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards)
        if rewards.size(0) == 1:
            logger.info("Rewards of length one, no need to normalize")
        else:
            rewards = (rewards - rewards.mean()) / \
                (rewards.std() + np.finfo(np.float32).eps)
        self.current_rewards = rewards
        for _i in range(self.training_iters_per_batch):
            y = Variable(rewards)
            loss = self.critic.training_step(batch['states'], y).data[0]
            logger.debug(f'Normalized rewards: {y.data}')
            logger.debug(f'Critic grad norms: {self.critic.get_grad_norms()}')
        return loss

    def calculate_advantage(self, batch):
        critic_estimate = self.critic.wrap_eval(batch['states']).squeeze_()
        advantage = self.current_rewards - critic_estimate
        logger.debug(f'Advantage: {advantage.size()}')
        return advantage

    def sample(self):
        '''Samples a batch from memory'''
        batches = [body.memory.sample()
                   for body in self.agent.flat_nonan_body_a]
        batch = util.concat_dict(batches)
        util.to_torch_batch_ex_rewards(batch)
        return batch
