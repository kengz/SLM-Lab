from slm_lab.agent import memory
from slm_lab.agent import net
from slm_lab.agent.algorithm.algorithm_util import act_fns, decay_learning_rate
from slm_lab.agent.algorithm.reinforce import Reinforce
from slm_lab.agent.net import net_util
from slm_lab.lib import util, logger
from slm_lab.lib.decorator import lab_api
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch
import pydash as _


class ActorCritic(Reinforce):
    '''
    Implementation of single threaded Advantage Actor Critic
    Original paper: "Asynchronous Methods for Deep Reinforcement Learning"
    https://arxiv.org/abs/1602.01783
    Algorithm specific training options:
        - GAE:          @param: 'algorithm.use_GAE' option to use generalized advantage estimation
                        introduced in "High-Dimensional Continuous Control Using Generalized Advantage
                        Estimation https://arxiv.org/abs/1506.02438. The default option is to use n-step
                        returns as desribed in "Asynchronous Methods for Deep Reinforcement Learning"
        - entropy:      @param: 'algorithm.add_entropy' option to add entropy to policy during training to
                        encourage exploration as outlined in "Asynchronous Methods for Deep Reinforcement
                        Learning"
        - memory type:  @param: 'memory.name' batch (through OnPolicyBatchReplay memory class) or episodic
                        through (OnPolicyReplay memory class)
        - return steps: @param: 'algorithm.num_step_returns' how many forward step returns to use when
                        calculating the advantage target. Min = 0. Applied for standard advantage estimation.
                        Not used for GAE.
        - lamda:        @param: 'algorithm.lamda' controls the bias variance tradeoff when using GAE.
                        Floating point value between 0 and 1. Lower values correspond to more bias,
                        less variance. Higher values to more variance, less bias.
        - param sharing: @param: 'net.type' whether the actor and critic should share params (e.g. through
                        'MLPshared') or have separate params (e.g. through 'MLPseparate')
                        If param sharing is used then there is also the option to control the weight
                        given to the policy and value components of the loss function through
                        'policy_loss_weight' and 'val_loss_weight'
    Algorithm - separate actor and critic:
        Repeat:
            1. Collect k examples
            2. Train the critic network using these examples
            3. Calculate the advantage of each example using the critic
            4. Multiply the advantage by the negative of log probability of the action taken, and sum all the values. This is the policy loss.
            5. Calculate the gradient the parameters of the actor network with respect to the policy loss
            6. Update the actor network parameters using the gradient
    Algorithm - shared parameters:
        Repeat:
            1. Collect k examples
            2. Calculate the target for each example for the critic
            3. Compute current estimate of state-value for each example using the critic
            4. Calculate the critic loss using a regression loss (e.g. square loss) between the target and estimate of the state-value for each example
            5. Calculate the advantage of each example using the rewards and critic
            6. Multiply the advantage by the negative of log probability of the action taken, and sum all the values. This is the policy loss.
            7. Compute the total loss by summing the value and policy lossses
            8. Calculate the gradient of the parameters of shared network with respect to the total loss
            9. Update the shared network parameters using the gradient
    '''

    @lab_api
    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        self.init_nets()
        self.init_algo_params()
        logger.info(util.self_desc(self))

    @lab_api
    def body_act_discrete(self, body, state):
        return self.action_policy(self, state, body)

    @lab_api
    def body_act_continuous(self, body, state):
        return self.action_policy(self, state, body)

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
        '''Trains the algorithm'''
        if self.is_shared_architecture:
            return self.train_shared()
        else:
            return self.train_separate()

    def train_shared(self):
        '''Trains the network when the actor and critic share parameters'''
        if self.to_train == 1:
            batch = self.sample()
            '''Calculate policy loss (actor)'''
            policy_loss = self.get_policy_loss(batch)
            '''Calculate state-value loss (critic)'''
            target = self.get_target(batch, critic_specific=True)
            states = batch['states']
            if self.is_episodic:
                target = torch.cat(target)
                states = torch.cat(states)
            y = Variable(target.unsqueeze_(dim=-1))
            state_vals = self.get_critic_output(states, evaluate=False)
            assert state_vals.data.size() == y.data.size()
            val_loss = F.mse_loss(state_vals, y)
            '''Combine losses and train'''
            self.actorcritic.optim.zero_grad()
            total_loss = self.policy_loss_weight * policy_loss + self.val_loss_weight * val_loss
            loss = total_loss.data[0]
            total_loss.backward()
            if self.actorcritic.clamp_grad:
                logger.debug("Clipping actorcritic gradient...")
                torch.nn.utils.clip_grad_norm(
                    self.actorcritic.params, self.actorcritic.clamp_grad_val)
            logger.debug2(f'Combined AC gradient norms: {self.actorcritic.get_grad_norms()}')
            self.actorcritic.optim.step()
            self.to_train = 0
            self.saved_log_probs = []
            self.entropy = []
            logger.debug("Losses: Critic: {:.2f}, Actor: {:.2f}, Total: {:.2f}".format(
                val_loss.data[0], abs(policy_loss.data[0]), loss
            ))
            return loss
        else:
            return np.nan

    def train_separate(self):
        '''Trains the network when the actor and critic are separate networks'''
        if self.to_train == 1:
            batch = self.sample()
            logger.debug3(f'Batch states: {batch["states"]}')
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
        '''Trains the critic when the actor and critic are separate networks'''
        if self.is_episodic:
            return self.train_critic_episodic(batch)
        else:
            return self.train_critic_batch(batch)

    def train_actor(self, batch):
        '''Trains the actor when the actor and critic are separate networks'''
        self.actor.optim.zero_grad()
        policy_loss = self.get_policy_loss(batch)
        loss = policy_loss.data[0]
        policy_loss.backward()
        if self.actor.clamp_grad:
            logger.debug("Clipping actor gradient...")
            torch.nn.utils.clip_grad_norm(
                self.actor.params, self.actor.clamp_grad_val)
        logger.debug(f'Actor gradient norms: {self.actor.get_grad_norms()}')
        self.actor.optim.step()
        self.to_train = 0
        self.saved_log_probs = []
        self.entropy = []
        logger.debug(f'Policy loss: {loss}')
        return loss

    def get_policy_loss(self, batch):
        '''Returns the policy loss for a batch of data.'''
        return super(ActorCritic, self).get_policy_loss(batch)

    def train_critic_batch(self, batch):
        '''Trains the critic using batches of data. Algorithm doesn't wait until episode has ended to train'''
        loss = 0
        for _i in range(self.training_iters_per_batch):
            target = self.get_target(batch, critic_specific=True)
            y = Variable(target)
            loss = self.critic.training_step(batch['states'], y).data[0]
            logger.debug(f'Critic grad norms: {self.critic.get_grad_norms()}')
        return loss

    def train_critic_episodic(self, batch):
        '''Trains the critic using entire episodes of data. Algorithm waits until episode has ended to train'''
        loss = 0
        for _i in range(self.training_iters_per_batch):
            target = self.get_target(batch, critic_specific=True)
            target = torch.cat(target)
            logger.debug2(f'Combined size: {target.size()}')
            x = []
            for state in batch['states']:
                x.append(state)
                logger.debug2(f'states: {state.size()}')
            x = torch.cat(x, dim=0)
            logger.debug2(f'Combined states: {x.size()}')
            y = Variable(target)
            loss = self.critic.training_step(x, y).data[0]
            logger.debug2(f'Critic grad norms: {self.critic.get_grad_norms()}')
        return loss

    def calc_advantage(self, batch):
        ''' Calculates advantage = target - state_vals for each timestep
            state_vals are the current estimate using the critic
            Two options for calculating the advantage.
                1. n_step forward returns as in "Asynchronous Methods for Deep Reinforcement Learning"
                2. Generalized advantage estimation (GAE) as in "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
            Default is 1. To select GAE set use_GAE to true in the spec.
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
        state_vals = self.get_critic_output(batch['states']).squeeze_()
        advantage = target - state_vals
        advantage.squeeze_()
        logger.debug2(f'Advantage: {advantage.size()}')
        return advantage

    def calc_advantage_episodic(self, batch):
        '''Calculates advantage when memory is batch based.
           target and state_vals are lists containing tensors per episode.
           returns advantage as a single tensor combined for all episodes'''
        target = self.get_target(batch)
        advantage = []
        states = batch['states']
        for s, t in zip(states, target):
            state_vals = self.get_critic_output(s).squeeze_()
            a = t - state_vals
            a.squeeze_()
            logger.debug2(f'Advantage: {a.size()}')
            advantage.append(a)
        advantage = torch.cat(advantage)
        return advantage

    def get_nstep_target(self, batch, critic_specific=False):
        '''Estimates state-action value with n-step returns. Used as a target when training the critic and calculting the advantage. No critic specific target value for this method of calculating the advantage.
        In the episodic case it returns a list containing targets per episode
        In the batch case it returns a tensor containing the targets for the batch'''
        if self.is_episodic:
            return self.get_nstep_target_episodic(batch)
        else:
            return self.get_nstep_target_batch(batch)

    def get_gae_target(self, batch, critic_specific=False):
        '''Estimates the state-action value using generalized advantage estimation. Used as a target when training the critic and calculting the advantage.
        In the episodic case it returns a list containing targets per episode
        In the batch case it returns a tensor containing the targets for the batch'''
        if self.is_episodic:
            return self.get_gae_target_episodic(batch, critic_specific=critic_specific)
        else:
            return self.get_gae_target_batch(batch, critic_specific=critic_specific)

    def get_nstep_target_batch(self, batch):
        '''Returns a tensor containing the estimate of the state-action values using n-step returns'''
        nts = self.num_step_returns
        next_state_vals = self.get_critic_output(batch['next_states']).squeeze_(dim=1)
        rewards = batch['rewards'].data
        (R, next_state_gammas) = self.get_R_ex_state_val_estimate(next_state_vals, rewards)
        '''Complete for 0th step and add state-value estimate'''
        R = rewards + self.gamma * R
        next_state_gammas *= self.gamma
        logger.debug3(f'R: {R}')
        logger.debug3(f'next_state_gammas: {next_state_gammas}')
        logger.debug3(f'dones: {batch["dones"]}')
        '''Calculate appropriate state value accounting for terminal states and number of time steps'''
        discounted_state_val_estimate = torch.mul(next_state_vals, next_state_gammas)
        discounted_state_val_estimate = torch.mul(discounted_state_val_estimate, 1 - batch['dones'].data)
        R += discounted_state_val_estimate
        logger.debug3(f'discounted_state_val_estimate: {discounted_state_val_estimate}')
        logger.debug3(f'R: {R}')
        return R

    def get_nstep_target_episodic(self, batch):
        '''Returns a list of tensors containing the estimate of the state-action values per batch using n-step returns'''
        nts = self.num_step_returns
        targets = []
        dones = batch['dones']
        next_states = batch['next_states']
        rewards = batch['rewards']
        for d, ns, r in zip(dones, next_states, rewards):
            next_state_vals = self.get_critic_output(ns).squeeze_(dim=1)
            r = r.data
            (R, next_state_gammas) = self.get_R_ex_state_val_estimate(next_state_vals, r)
            '''Complete for 0th step and add state-value estimate'''
            R = r + self.gamma * R
            next_state_gammas *= self.gamma
            logger.debug3(f'R: {R}')
            logger.debug3(f'next_state_gammas: {next_state_gammas}')
            logger.debug3(f'dones: {d}')
            '''Calculate appropriate state value accounting for terminal states and number of time steps'''
            discounted_state_val_estimate = torch.mul(next_state_vals, next_state_gammas)
            discounted_state_val_estimate = torch.mul(discounted_state_val_estimate, 1 - d.data)
            if nts < next_state_vals.size(0):
                logger.debug2(f'N step returns less than episode length, adding boostrap')
                R += discounted_state_val_estimate
            logger.debug3(f'discounted_state_val_estimate: {discounted_state_val_estimate}')
            logger.debug3(f'R: {R}')
            targets.append(R)
        return targets

    def get_R_ex_state_val_estimate(self, next_state_vals, rewards):
        nts = self.num_step_returns
        R = torch.zeros_like(next_state_vals)
        curr_reward_step = torch.zeros_like(next_state_vals)
        next_state_gammas = torch.zeros_like(next_state_vals)
        if nts >= next_state_vals.size(0):
            logger.debug2(f'Num step returns {self.num_step_returns} greater than length batch {next_state_vals.size(0)}. Updating to batch length')
            nts = next_state_vals.size(0) - 1
        if nts == 0:
            next_state_gammas.fill_(1.0)
        else:
            j = -nts
            next_state_gammas[:j] = 1.0
        for i in range(nts, 0, -1):
            logger.debug(f'i: {i}, j: {j}')
            curr_reward_step[:j] = rewards[i:]
            next_state_gammas[:j] *= self.gamma
            R = curr_reward_step + self.gamma * R
            next_state_gammas[j] = 1.0
            j += 1
            logger.debug3(f'curr_reward_step: {curr_reward_step}')
            logger.debug3(f'next_state_gammas: {next_state_gammas}')
            logger.debug3(f'R: {R}')
        return (R, next_state_gammas)

    def get_gae_target_batch(self, batch, critic_specific):
        '''Returns a tensor containing the estimate of the state-action values using generalized advantage estimation'''
        rewards = batch['rewards'].data
        if critic_specific:
            logger.debug2(f'Using critic specific target')
            '''Target is the discounted sum of returns for training the critic'''
            target = self.get_gae_critic_target(rewards)
        else:
            logger.debug2(f'Using actor specific target')
            '''Target is the Generalized advantage estimate + current state-value estimate'''
            states = batch['states']
            next_states = batch['next_states']
            dones = batch['dones']
            target = self.get_gae_actor_target(rewards, states, next_states, dones)
        return target

    def get_gae_target_episodic(self, batch, critic_specific):
        '''Returns a list of tensors containing the estimate of the state-action values per batch using generalized advantage estimation'''
        rewards = batch['rewards']
        targets = []
        if critic_specific:
            logger.debug2(f'Using critic specific target')
            '''Target is the discounted sum of returns for training the critic'''
            for r in rewards:
                t = self.get_gae_critic_target(r.data)
                targets.append(t)
        else:
            logger.debug2(f'Using actor specific target')
            '''Target is the Generalized advantage estimate + current state-value estimate'''
            states = batch['states']
            next_states = batch['next_states']
            dones = batch['dones']
            for r, s, ns, d in zip(rewards, states, next_states, dones):
                t = self.get_gae_actor_target(r.data, s, ns, d)
                targets.append(t)
        return targets

    def get_gae_critic_target(self, rewards):
        '''Target is the discounted sum of returns for training the critic'''
        target = []
        big_r = 0
        for i in range(rewards.size(0) - 1, -1, -1):
            big_r = rewards[i] + self.gamma * big_r
            target.insert(0, big_r)
        target = torch.Tensor(target)
        logger.debug3(f'Target: {target}')
        return target

    def get_gae_actor_target(self, rewards, states, next_states, dones):
        '''Target is the Generalized advantage estimate + current state-value estimate'''
        '''First calculate the 1 step bootstrapped estimate of the advantage. Also described as the TD residual of V with discount self.gamma (Sutton & Barto, 1998)'''
        next_state_vals = self.get_critic_output(next_states).squeeze_(dim=1)
        next_state_vals = torch.mul(next_state_vals, 1 - dones.data)
        state_vals = self.get_critic_output(states).squeeze_(dim=1)
        deltas = rewards + self.gamma * next_state_vals - state_vals
        logger.debug3(f'State_vals: {state_vals}')
        logger.debug3(f'Next state_vals: {next_state_vals}')
        logger.debug3(f'Dones: {dones}')
        logger.debug3(f'Deltas: {deltas}')
        logger.debug3(f'Lamda: {self.lamda}, gamma: {self.gamma}')
        '''Then calculate GAE, the exponentially weighted average of the TD residuals'''
        advantage = []
        gae = 0
        for i in range(deltas.size(0) - 1, -1, -1):
            gae = deltas[i] + self.gamma * self.lamda * gae
            advantage.insert(0, gae)
        advantage = torch.Tensor(advantage)
        '''Add state_vals so that calc_advantage() api is preserved'''
        target = advantage + state_vals
        logger.debug3(f'Advantage: {advantage}')
        logger.debug3(f'Target: {target}')
        return target

    def init_nets(self):
        '''Initialize the neural networks used to learn the actor and critic from the spec'''
        body = self.agent.nanflat_body_a[0]  # singleton algo
        state_dim = body.state_dim
        action_dim = body.action_dim
        self.is_discrete = body.is_discrete
        net_spec = self.agent.spec['net']
        mem_spec = self.agent.spec['memory']
        net_type = self.agent.spec['net']['type']
        actor_kwargs = util.compact_dict(dict(
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim_actor'),
            loss_param=_.get(net_spec, 'loss'),  # Note: Not used for training actor
            clamp_grad=_.get(net_spec, 'clamp_grad'),
            clamp_grad_val=_.get(net_spec, 'clamp_grad_val'),
        ))
        if self.agent.spec['net']['use_same_optim']:
            logger.info('Using same optimizer for actor and critic')
            critic_kwargs = actor_kwargs
        else:
            logger.info('Using different optimizer for actor and critic')
            critic_kwargs = util.compact_dict(dict(
                hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
                optim_param=_.get(net_spec, 'optim_critic'),
                loss_param=_.get(net_spec, 'loss'),
                clamp_grad=_.get(net_spec, 'clamp_grad'),
                clamp_grad_val=_.get(net_spec, 'clamp_grad_val'),
            ))
        '''
         Below we automatically select an appropriate net based on two different conditions
           1. If the action space is discrete or continuous action
                   - Networks for continuous action spaces have two heads and return two values, the first is a tensor containing the mean of the action policy, the second is a tensor containing the std deviation of the action policy. The distribution is assumed to be a Gaussian (Normal) distribution.
                   - Networks for discrete action spaces have a single head and return the logits for a categorical probability distribution over the discrete actions
           2. If the actor and critic are separate or share weights
                   - If the networks share weights then the single network returns a list.
                        - Continuous action spaces: The return list contains 3 elements: The first element contains the mean output for the actor (policy), the second element the std dev of the policy, and the third element is the state-value estimated by the network.
                        - Discrete action spaces: The return list contains 2 element. The first element is a tensor containing the logits for a categorical probability distribution over the actions. The second element contains the state-value estimated by the network.
           3. If the network type is feedforward, convolutional, or recurrent
                    - Feedforward and convolutional networks take a single state as input and require an OnPolicyReplay or OnPolicyBatchReplay memory
                    - Recurrent networks take n states as input and require an OnPolicyNStepReplay or OnPolicyNStepBatchReplay memory
        '''
        if net_type == 'MLPseparate':
            self.is_shared_architecture = False
            self.is_recurrent = False
            if self.is_discrete:
                self.actor = getattr(net, 'MLPNet')(
                    state_dim, net_spec['hid_layers'], action_dim, **actor_kwargs)
                logger.info("Feedforward net, discrete action space, actor and critic are separate networks")
            else:
                self.actor = getattr(net, 'MLPHeterogenousHeads')(
                    state_dim, net_spec['hid_layers'], [action_dim, action_dim], **actor_kwargs)
                logger.info("Feedforward net, continuous action space, actor and critic are separate networks")
            self.critic = getattr(net, 'MLPNet')(
                state_dim, net_spec['hid_layers'], 1, **critic_kwargs)
        elif net_type == 'MLPshared':
            self.is_shared_architecture = True
            self.is_recurrent = False
            if self.is_discrete:
                self.actorcritic = getattr(net, 'MLPHeterogenousHeads')(
                    state_dim, net_spec['hid_layers'], [action_dim, 1], **actor_kwargs)
                logger.info("Feedforward net, discrete action space, actor and critic combined into single network, sharing params")
            else:
                self.actorcritic = getattr(net, 'MLPHeterogenousHeads')(
                    state_dim, net_spec['hid_layers'], [action_dim, action_dim, 1], **actor_kwargs)
                logger.info("Feedforward net, continuous action space, actor and critic combined into single network, sharing params")
        elif net_type == 'Convseparate':
            self.is_shared_architecture = False
            self.is_recurrent = False
            if self.is_discrete:
                self.actor = getattr(net, 'ConvNet')(
                    state_dim, net_spec['hid_layers'], action_dim, **actor_kwargs)
                logger.info("Convolutional net, discrete action space, actor and critic are separate networks")
            else:
                self.actor = getattr(net, 'ConvNet')(
                    state_dim, net_spec['hid_layers'], [action_dim, action_dim], **actor_kwargs)
                logger.info("Convolutional net, continuous action space, actor and critic are separate networks")
            self.critic = getattr(net, 'ConvNet')(
                state_dim, net_spec['hid_layers'], 1, **critic_kwargs)
        elif net_type == 'Convshared':
            self.is_shared_architecture = True
            self.is_recurrent = False
            if self.is_discrete:
                self.actorcritic = getattr(net, 'ConvNet')(
                    state_dim, net_spec['hid_layers'], [action_dim, 1], **actor_kwargs)
                logger.info("Convolutional net, discrete action space, actor and critic combined into single network, sharing params")
            else:
                self.actorcritic = getattr(net, 'ConvNet')(
                    state_dim, net_spec['hid_layers'], [action_dim, action_dim, 1], **actor_kwargs)
                logger.info("Convolutional net, continuous action space, actor and critic combined into single network, sharing params")
        elif net_type == 'Recurrentseparate':
            self.is_shared_architecture = False
            self.is_recurrent = True
            if self.is_discrete:
                self.actor = getattr(net, 'RecurrentNet')(
                    state_dim, net_spec['hid_layers'], action_dim, mem_spec['length_history'], **actor_kwargs)
                logger.info("Recurrent net, discrete action space, actor and critic are separate networks")
            else:
                self.actor = getattr(net, 'RecurrentNet')(
                    state_dim, net_spec['hid_layers'], [action_dim, action_dim], mem_spec['length_history'], **actor_kwargs)
                logger.info("Recurrent net, continuous action space, actor and critic are separate networks")
            self.critic = getattr(net, 'RecurrentNet')(
                state_dim, net_spec['hid_layers'], 1, mem_spec['length_history'], **critic_kwargs)
        elif net_type == 'Recurrentshared':
            self.is_shared_architecture = True
            self.is_recurrent = True
            if self.is_discrete:
                self.actorcritic = getattr(net, 'RecurrentNet')(
                    state_dim, net_spec['hid_layers'], [action_dim, 1], mem_spec['length_history'], **actor_kwargs)
                logger.info("Recurrent net, discrete action space, actor and critic combined into single network, sharing params")
            else:
                self.actorcritic = getattr(net, 'RecurrentNet')(
                    state_dim, net_spec['hid_layers'], [action_dim, action_dim, 1], mem_spec['length_history'], **actor_kwargs)
                logger.info("Recurrent net, continuous action space, actor and critic combined into single network, sharing params")
        else:
            logger.warn("Incorrect network type. Please use 'MLPshared', MLPseparate', Recurrentshared, or Recurrentseparate.")
            raise NotImplementedError

    def init_algo_params(self):
        '''Initialize other algorithm parameters'''
        algorithm_spec = self.agent.spec['algorithm']
        net_spec = self.agent.spec['net']
        self.set_action_fn()
        util.set_attr(self, _.pick(algorithm_spec, [
            'gamma',
            'num_epis_to_collect',
            'add_entropy', 'entropy_weight',
            'continuous_action_clip',
            'lamda', 'num_step_returns',
            'training_frequency', 'training_iters_per_batch',
            'use_GAE',
            'policy_loss_weight', 'val_loss_weight',

        ]))
        util.set_attr(self, _.pick(net_spec, [
            'decay_lr', 'decay_lr_frequency', 'decay_lr_min_timestep',
        ]))
        '''Select appropriate function for calculating state-action-value estimate (target)'''
        self.get_target = self.get_nstep_target
        if self.use_GAE:
            self.get_target = self.get_gae_target
        self.set_memory_flag()
        '''To save on a forward pass keep the log probs and entropy from each action'''
        self.saved_log_probs = []
        self.entropy = []
        self.to_train = 0

    def set_action_fn(self):
        '''Sets the function used to select actions. Automatically selects appropriate discrete or continuous action policy under default setting'''
        body = self.agent.nanflat_body_a[0]
        algorithm_spec = self.agent.spec['algorithm']
        action_fn = algorithm_spec['action_policy']
        if action_fn == 'default':
            if self.is_discrete:
                self.action_policy = act_fns['softmax']
            else:
                if body.action_dim > 1:
                    logger.warn(f'Action dim: {body.action_dim}. Continuous multidimensional action space not supported yet. Contact author')
                    raise NotImplementedError
                else:
                    self.action_policy = act_fns['gaussian']
        else:
            self.action_policy = act_fns[action_fn]

    def set_memory_flag(self):
        '''Flags if memory is episodic or discrete. This affects how the target and advantage functions are calculated'''
        body = self.agent.nanflat_body_a[0]
        memory = body.memory.__class__.__name__
        if (memory.find('OnPolicyReplay') != -1) or (memory.find('OnPolicyNStepReplay') != -1):
            self.is_episodic = True
        elif (memory.find('OnPolicyBatchReplay') != -1) or (memory.find('OnPolicyNStepBatchReplay') != -1):
            self.is_episodic = False
        else:
            logger.warn(f'Error: Memory {memory} not recognized')
            raise NotImplementedError

    def print_nets(self):
        '''Prints networks to stdout'''
        if self.is_shared_architecture:
            print(self.actorcritic)
        else:
            print(self.actor)
            print(self.critic)

    def get_actor_output(self, x, evaluate=True):
        '''Returns the output of the policy, regardless of the underlying network structure. This makes it easier to handle AC algorithms with shared or distinct params.
           Output will either be the logits for a categorical probability distribution over discrete actions (discrete action space) or the mean and std dev of the action policy (continuous action space)
        '''
        if self.is_shared_architecture:
            if evaluate:
                out = self.actorcritic.wrap_eval(x)
            else:
                self.actorcritic.train()
                out = self.actorcritic(x)
            return out[:-1]
        else:
            if evaluate:
                return self.actor.wrap_eval(x)
            else:
                self.actor.train()
                return self.actor(x)

    def get_critic_output(self, x, evaluate=True):
        '''Returns the estimated state-value regardless of the underlying network structure. This makes it easier to handle AC algorithms with shared or distinct params.'''
        if self.is_shared_architecture:
            if evaluate:
                out = self.actorcritic.wrap_eval(x)
            else:
                self.actorcritic.train()
                out = self.actorcritic(x)
            return out[-1]
        else:
            if evaluate:
                return self.critic.wrap_eval(x)
            else:
                self.critic.train()
                return self.critic(x)

    def update_learning_rate(self):
        if self.is_shared_architecture:
            decay_learning_rate(self, [self.actorcritic])
        else:
            decay_learning_rate(self, [self.actor, self.critic])
