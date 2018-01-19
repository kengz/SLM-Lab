from slm_lab.agent import memory
from slm_lab.agent import net
from slm_lab.agent.algorithm.algorithm_util import act_fns
from slm_lab.agent.algorithm.reinforce import Reinforce
from slm_lab.agent.net import net_util
from slm_lab.lib import util, logger
from slm_lab.lib.decorator import lab_api
from torch.autograd import Variable
import sys
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
        self.is_discrete = body.is_discrete
        net_spec = self.agent.spec['net']
        net_type = self.agent.spec['net']['type']
        actor_kwargs = util.compact_dict(dict(
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim_actor'),
            loss_param=_.get(net_spec, 'loss'),  # Note: Not used for training actor
            clamp_grad=_.get(net_spec, 'clamp_grad'),
            clamp_grad_val=_.get(net_spec, 'clamp_grad_val'),
        ))
        critic_kwargs = util.compact_dict(dict(
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim_critic'),
            loss_param=_.get(net_spec, 'loss'),
            clamp_grad=_.get(net_spec, 'clamp_grad'),
            clamp_grad_val=_.get(net_spec, 'clamp_grad_val'),
        ))
        # Below we automatically select an appropriate net based on two different conditions
        #   1. If the action space is discrete or continuous action
        #           - Networks for continuous action spaces have two heads and return two values, the first is a tensor containing the mean of the action policy, the second is a tensor containing the std deviation of the action policy. The distribution is assumed to be a Gaussian (Normal) distribution.
        #           - Networks for discrete action spaces have a single head and return the logits for a categorical probability distribution over the discrete actions
        #   2. If the actor and critic should be separate or share weights
        #           - If the networks share weights then the single network returns a tuple. The first element contains the output for the actor (policy). This will vary depending on the type of distribution (see point 1.). The second element contains the state-value estimated by the network.
        if net_type == 'MLPseparate':
            self.is_shared_architecture = False
            if self.is_discrete:
                self.actor = getattr(net, 'MLPNet')(
                    state_dim, net_spec['hid_layers'], action_dim, **actor_kwargs)
            else:
                self.actor = getattr(net, 'MLPHeterogenousHeads')(
                    state_dim, net_spec['hid_layers'], [action_dim, action_dim], **actor_kwargs)
            self.critic = getattr(net, 'MLPNet')(
                state_dim, net_spec['hid_layers'], 1, **critic_kwargs)
        else:
            self.is_shared_architecture = True
            # TODO fix for MLPshared - single heterogenous head
            logger.warn(f'Net type {net_type} not supported yet. Please use MLPseparate')
            sys.exit()

    def init_algo_params(self):
        '''Initialize other algorithm parameters'''
        body = self.agent.nanflat_body_a[0]
        algorithm_spec = self.agent.spec['algorithm']
        # Automatically selects appropriate discrete or continuous action policy under default setting
        action_fn = algorithm_spec['action_policy']
        if action_fn == 'default':
            if self.is_discrete:
                self.action_policy = act_fns['softmax']
            else:
                if body.action_dim > 1:
                    logger.warn(f'Action dim: {body.action_dim}. Continuous multidimensional action space not supported yet. Contact author')
                    sys.exit()
                else:
                    self.action_policy = act_fns['gaussian']
        else:
            self.action_policy = act_fns[action_fn]
        # Set other training parameters
        util.set_attr(self, _.pick(algorithm_spec, [
            'gamma',
            'training_frequency', 'training_iters_per_batch',
            'add_entropy', 'advantage_fn', 'num_epis_to_collect'
        ]))
        if self.advantage_fn == "gae_1":
            self.get_target = self.gae_1_target
        else:
            self.get_target = self.gae_0_target
        # Flag if memory is episodic or discrete.
        # This affects how the target and advantage functions are calculated
        memory = body.memory.__class__.__name__
        if memory.find('OnPolicyReplay') != -1:
            self.is_episodic = True
        elif memory.find('OnPolicyBatchReplay') != -1:
            self.is_episodic = False
        else:
            logger.warn(f'Error: Memory {memory} not recognised')
            sys.exit()
        # To save on a forward pass keep the log probs and entropy from each action
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
        batches = [body.memory.sample()
                   for body in self.agent.nanflat_body_a]
        batch = util.concat_dict(batches)
        if self.is_episodic:
            util.to_torch_nested_batch(batch)
        else:
            util.to_torch_batch(batch)
        return batch

    @lab_api
    def train(self):
        if self.is_shared_architecture:
            return self.train_shared()
        else:
            return self.train_separate()

    def train_shared(self):
        # TODO: implement train_shared
        pass

    def train_separate(self):
        if self.to_train == 1:
            batch = self.sample()
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
        if self.is_episodic:
            return self.train_critic_episodic(batch)
        else:
            return self.train_critic_batch(batch)

    def train_critic_batch(self, batch):
        loss = 0
        for _i in range(self.training_iters_per_batch):
            target = self.get_target(batch)
            y = Variable(target)
            loss = self.critic.training_step(batch['states'], y).data[0]
            logger.debug(f'Critic grad norms: {self.critic.get_grad_norms()}')
        return loss

    def train_critic_episodic(self, batch):
        loss = 0
        for _i in range(self.training_iters_per_batch):
            target = self.get_target(batch)
            for t in target:
                logger.debug(f'Target size: {t.size()}')
            target = torch.cat(target)
            logger.debug(f'Combined size: {target.size()}')
            x = []
            for state in batch['states']:
                x.append(state)
                logger.debug(f'states: {state.size()}')
            x = torch.cat(x, dim=0)
            logger.debug(f'Combined states: {x.size()}')
            y = Variable(target)
            loss = self.critic.training_step(x, y).data[0]
            logger.debug(f'Critic grad norms: {self.critic.get_grad_norms()}')
        return loss

    def calc_advantage(self, batch):
        ''' Calculates advantage = target - state_vals for each timestep
            state_vals are the current estimate using the critic
            Method for calculating target varies. It is either
                1. The discounted sum of rewards from the current timestep to the end of an episode. Equivalent to GAE(1) or TD(infinity) with discounts
                2. Bootstrapped estimate of the state-action value = r_t + gamma * V(s_(t+1)). Equivalent to GAE(0) or TD(0) with discounts.
        '''
        if self.is_episodic:
            return self.calc_advantage_episodic(batch)
        else:
            return self.calc_advantage_batch(batch)

    def calc_advantage_batch(self, batch):
        '''Calculates advantage when memory is batch based.
           target and state_vals are Tensors.
           returns advantage as a single Tensor'''
        target = self.get_target(batch)
        state_vals = self.critic.wrap_eval(batch['states']).squeeze_()
        advantage = target - state_vals
        advantage.squeeze_()
        logger.debug(f'Advantage: {advantage.size()}')
        return advantage

    def calc_advantage_episodic(self, batch):
        '''Calculates advantage when memory is batch based.
           target and state_vals are lists containing tensors per episode.
           returns advantage as a single tensor combined for all episodes'''
        target = self.get_target(batch)
        advantage = []
        states = batch['states']
        for s, t in zip(states, target):
            state_vals = self.critic.wrap_eval(batch['states']).squeeze_()
            a = t - state_vals
            a.squeeze_()
            logger.debug(f'Advantage: {a.size()}')
            advantage.append(a)
        advantage = torch.cat(advantage)
        return advantage

    def gae_0_target(self, batch):
        '''Calculates target = bootstrapped estimate of the state-action value = r_t + gamma * V(s_(t+1)). Equivalent to GAE(0) or TD(0) with discounts.
        In the episodic case it returns a list containing targets per episode
        In the batch case it returns a tensor containing the targets for the batch'''
        if self.is_episodic:
            return self.gae_0_target_episodic(batch)
        else:
            return self.gae_0_target_batch(batch)

    def gae_0_target_batch(self, batch):
        next_state_vals = self.critic.wrap_eval(
            batch['next_states']).squeeze_()
        target = batch['rewards'].data + self.gamma * \
            torch.mul((1 - batch['dones'].data), next_state_vals)
        logger.debug(f'Target: {target.size()}')
        return target

    def gae_0_target_episodic(self, batch):
        next_states = batch['next_states']
        rewards = batch['rewards']
        dones = batch['dones']
        targets = []
        for ns, r, d in zip(next_states, rewards, dones):
            r = torch.from_numpy(np.asarray(r))
            nsv = self.critic.wrap_eval(ns).squeeze_()
            target = r + self.gamma * torch.mul((1 - d.data), nsv)
            targets.append(target)
            logger.debug(f'Next state vals: {nsv}, rewards: {r}, dones: {d}')
            logger.debug(f'Target: {target.size()}')
        return targets

    def gae_1_target(self, batch):
        '''Calculates target = discounted sum of rewards from the current timestep to the end of an episode. Equivalent to GAE(1) or TD(infinity) with discounts. Returns a list containing targets per episode'''
        assert self.is_episodic is True
        raw_rewards = batch['rewards']
        epi_rewards = []
        for epi_raw_rewards in raw_rewards:
            rewards = []
            big_r = 0
            for r in epi_raw_rewards[::-1]:
                big_r = r + self.gamma * big_r
                rewards.insert(0, big_r)
            rewards = torch.Tensor(rewards)
            epi_rewards.append(rewards)
        return epi_rewards

    def train_actor(self, batch):
        advantage = self.calc_advantage(batch)
        advantage = self.check_sizes(advantage)
        policy_loss = []
        for log_prob, a, e in zip(self.saved_log_probs, advantage, self.entropy):
            logger.debug(
                f'log prob: {log_prob.data[0]}, advantage: {a}, entropy: {e.data[0]}')
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
