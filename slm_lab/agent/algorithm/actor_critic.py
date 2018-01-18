from slm_lab.agent import memory
from slm_lab.agent import net
from slm_lab.agent.algorithm.algorithm_util import act_fns
from slm_lab.agent.algorithm.reinforce import Reinforce
from slm_lab.agent.net import net_util
from slm_lab.lib import util, logger
from slm_lab.lib.decorator import lab_api
from torch.autograd import Variable
import numpy as np
import torch
import pydash as _


class ActorCritic(Reinforce):
    '''
    Implementation of a simple actor-critic algorithm.
    TODO - finish comments
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

    @lab_api
    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        self.init_nets()
        self.init_algo_params()
        self.actor.print_nets()  # Print the network architecture
        logger.info(util.self_desc(self))

    def init_nets(self):
        '''Initialize the neural network used to learn the Q function from the spec'''
        body = self.agent.nanflat_body_a[0]  # singleton algo
        state_dim = body.state_dim
        action_dim = body.action_dim
        # TODO: create separate net specs
        # TODO: automate actor switch between continuous and discrete
        net_spec = self.agent.spec['net']
        actor_kwargs = util.compact_dict(dict(
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim_actor'),
            loss_param=_.get(net_spec, 'loss'),  # Note: Not used for PG algos
            clamp_grad=_.get(net_spec, 'clamp_grad'),
            clamp_grad_val=_.get(net_spec, 'clamp_grad_val'),
        ))
        self.actor = getattr(net, net_spec['type'])(
            state_dim, net_spec['hid_layers'], action_dim, **actor_kwargs)
        critic_kwargs = util.compact_dict(dict(
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim_critic'),
            loss_param=_.get(net_spec, 'loss'),  # Note: Not used for PG algos
            clamp_grad=_.get(net_spec, 'clamp_grad'),
            clamp_grad_val=_.get(net_spec, 'clamp_grad_val'),
        ))
        self.critic = getattr(net, net_spec['type'])(
            state_dim, net_spec['hid_layers'], 1, **critic_kwargs)

    def init_algo_params(self):
        '''Initialize other algorithm parameters'''
        algorithm_spec = self.agent.spec['algorithm']
        # TODO: fix for auto continuous and discrete adjustment
        self.action_policy = act_fns[algorithm_spec['action_policy']]
        util.set_attr(self, _.pick(algorithm_spec, [
            'gamma',
            'training_frequency', 'training_iters_per_batch',
            'add_entropy', 'advantage_fn'
        ]))
        if self.advantage_fn == "gae_1":
            self.get_target = self.gae_1_target
        else:
            self.get_target = self.gae_0_target
        # To save on a forward pass keep the log probs from each action
        self.saved_log_probs = []
        self.entropy = []
        self.to_train = 0

    @lab_api
    def body_act_discrete(self, body, state):
        return self.action_policy(self, state, self.actor)

    @lab_api
    def body_act_continuous(self, body, state):
        return self.action_policy(self, state, self.net, body)

    def sample(self):
        '''Samples a batch from memory'''
        # TODO: Change sampling type depending on memory
        batches = [body.memory.sample()
                   for body in self.agent.nanflat_body_a]
        batch = util.concat_dict(batches)
        util.to_torch_batch(batch)
        return batch

    @lab_api
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
            return np.nan

    def train_critic(self, batch):
        loss = 0
        for _i in range(self.training_iters_per_batch):
            target = self.get_target(batch)
            y = Variable(target)
            loss = self.critic.training_step(batch['states'], y).data[0]
            logger.debug(f'Critic grad norms: {self.critic.get_grad_norms()}')
        return loss

    def calc_advantage(self, batch):
        target = self.get_target(batch)
        state_vals = self.critic.wrap_eval(batch['states']).squeeze_()
        advantage = target - state_vals
        advantage.squeeze_()
        logger.debug(f'Advantage: {advantage.size()}')
        return advantage

    def gae_0_target(self, batch):
        next_state_vals = self.critic.wrap_eval(
            batch['next_states']).squeeze_()
        target = batch['rewards'].data + self.gamma * \
            torch.mul((1 - batch['dones'].data), next_state_vals)
        logger.debug(f'Target: {target.size()}')
        return target

    def gae_1_target(self, batch):
        rewards = []
        epi_rewards = batch['rewards']
        big_r = 0
        for i in xrange(epi_rewards.size(0), 0, -1):
            r = epi_rewards[i]
            big_r = r + self.gamma * big_r
            rewards.insert(0, big_r)
        rewards = torch.Tensor(rewards)
        logger.debug(f'Target: {target.size()}')
        return target

    def train_actor(self, batch):
        advantage = self.calc_advantage(batch)
        # Check log probs, advantage, and entropy all have the same size
        # Occassionally they do not, this is caused by first reward of an episode being nan
        if len(self.saved_log_probs) != advantage.size(0):
            del self.saved_log_probs[0]
            logger.debug('Deleting first log prob in epi')
        if len(self.entropy) != advantage.size(0):
            del self.entropy[0]
            logger.debug('Deleting first entropy in epi')
        assert len(self.saved_log_probs) == advantage.size(0)
        assert len(self.entropy) == advantage.size(0)
        policy_loss = []
        for log_prob, a, e in zip(self.saved_log_probs, advantage, self.entropy):
            logger.debug(f'log prob: {log_prob.data[0]}, advantage: {a}, entropy: {e.data[0]}')
            if self.add_entropy:
                policy_loss.append(-log_prob * a - 0.1 * e)
            else:
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
        self.entropy = []
        logger.debug(f'Policy loss: {loss}')
        return loss
