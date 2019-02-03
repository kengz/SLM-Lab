from slm_lab.agent import net
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.reinforce import Reinforce
from slm_lab.agent.net import net_util
from slm_lab.lib import logger, math_util, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import pydash as ps
import torch

logger = logger.get_logger(__name__)


class ActorCritic(Reinforce):
    '''
    Implementation of single threaded Advantage Actor Critic
    Original paper: "Asynchronous Methods for Deep Reinforcement Learning"
    https://arxiv.org/abs/1602.01783
    Algorithm specific spec param:
    memory.name: batch (through OnPolicyBatchReplay memory class) or episodic through (OnPolicyReplay memory class)
    lam: if not null, used as the lambda value of generalized advantage estimation (GAE) introduced in "High-Dimensional Continuous Control Using Generalized Advantage Estimation https://arxiv.org/abs/1506.02438. The algorithm becomes A2C. This lambda controls the bias variance tradeoff for GAE. Floating point value between 0 and 1. Lower values correspond to more bias, less variance. Higher values to more variance, less bias.
    num_step_returns: if lam is null and this is not null, specifies the number of steps for N-step returns from "Asynchronous Methods for Deep Reinforcement Learning". The algorithm becomes A2C.
    If both lam and num_step_returns are null, use the default TD error. Then the algorithm stays as AC.
    net.type: whether the actor and critic should share params (e.g. through 'MLPNetShared') or have separate params (e.g. through 'MLPNetSeparate'). If param sharing is used then there is also the option to control the weight given to the policy and value components of the loss function through 'policy_loss_coef' and 'val_loss_coef'
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

    e.g. algorithm_spec
    "algorithm": {
        "name": "ActorCritic",
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
        "training_frequency": 1,
        "training_epoch": 8,
        "normalize_state": true
    }

    e.g. special net_spec param "shared" to share/separate Actor/Critic
    "net": {
        "type": "MLPNet",
        "shared": true,
        ...
    '''

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
            'training_frequency',
            'training_epoch',
            'normalize_state',
        ])
        self.to_train = 0
        self.action_policy = getattr(policy_util, self.action_policy)
        self.explore_var_scheduler = policy_util.VarScheduler(self.explore_var_spec)
        self.body.explore_var = self.explore_var_scheduler.start_val
        if self.entropy_coef_spec is not None:
            self.entropy_coef_scheduler = policy_util.VarScheduler(self.entropy_coef_spec)
            self.body.entropy_coef = self.entropy_coef_scheduler.start_val
        # Select appropriate methods to calculate adv_targets and v_targets for training
        if self.lam is not None:
            self.calc_advs_v_targets = self.calc_gae_advs_v_targets
        elif self.num_step_returns is not None:
            self.calc_advs_v_targets = self.calc_nstep_advs_v_targets
        else:
            self.calc_advs_v_targets = self.calc_td_advs_v_targets

    @lab_api
    def init_nets(self, global_nets=None):
        '''
        Initialize the neural networks used to learn the actor and critic from the spec
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
            - Recurrent networks take n states as input and require an OnPolicySeqReplay or OnPolicySeqBatchReplay memory
        '''
        assert 'shared' in self.net_spec, 'Specify "shared" for ActorCritic network in net_spec'
        self.shared = self.net_spec['shared']

        # create actor/critic specific specs
        actor_net_spec = self.net_spec.copy()
        critic_net_spec = self.net_spec.copy()
        for k in self.net_spec:
            if 'actor_' in k:
                actor_net_spec[k.replace('actor_', '')] = actor_net_spec.pop(k)
                critic_net_spec.pop(k)
            if 'critic_' in k:
                critic_net_spec[k.replace('critic_', '')] = critic_net_spec.pop(k)
                actor_net_spec.pop(k)
        if critic_net_spec['use_same_optim']:
            critic_net_spec = actor_net_spec

        if global_nets is None:
            in_dim = self.body.state_dim
            out_dim = net_util.get_out_dim(self.body, add_critic=self.shared)
            # main actor network, may contain out_dim self.shared == True
            NetClass = getattr(net, actor_net_spec['type'])
            self.net = NetClass(actor_net_spec, in_dim, out_dim)
            self.net_names = ['net']
            if not self.shared:  # add separate network for critic
                critic_out_dim = 1
                CriticNetClass = getattr(net, critic_net_spec['type'])
                self.critic = CriticNetClass(critic_net_spec, in_dim, critic_out_dim)
                self.net_names.append('critic')
        else:
            util.set_attr(self, global_nets)
            self.net_names = list(global_nets.keys())
        self.post_init_nets()

    @lab_api
    def calc_pdparam(self, x, evaluate=True, net=None):
        '''
        The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous prob. dist.
        '''
        pdparam = super(ActorCritic, self).calc_pdparam(x, evaluate=evaluate, net=net)
        if self.shared:  # output: policy, value
            if len(pdparam) == 2:  # single policy outputs, value
                pdparam = pdparam[0]
            else:  # multiple policy outputs, value
                pdparam = pdparam[:-1]
        logger.debug(f'pdparam: {pdparam}')
        return pdparam

    def calc_v(self, x, evaluate=True, net=None):
        '''
        Forward-pass to calculate the predicted state-value from critic.
        '''
        net = self.net if net is None else net
        if self.shared:  # output: policy, value
            if evaluate:
                out = net.wrap_eval(x)
            else:
                net.train()
                out = net(x)
            v = out[-1].squeeze_(dim=1)  # get value only
        else:
            if evaluate:
                out = self.critic.wrap_eval(x)
            else:
                self.critic.train()
                out = self.critic(x)
            v = out.squeeze_(dim=1)
        logger.debug(f'v: {v}')
        return v

    @lab_api
    def train(self):
        '''Trains the algorithm'''
        if util.get_lab_mode() in ('enjoy', 'eval'):
            self.body.flush()
            return np.nan
        if self.shared:
            return self.train_shared()
        else:
            return self.train_separate()

    def train_shared(self):
        '''
        Trains the network when the actor and critic share parameters
        loss = self.policy_loss_coef * policy_loss + self.val_loss_coef * val_loss
        '''
        clock = self.body.env.clock
        if self.to_train == 1:
            batch = self.sample()
            with torch.no_grad():
                advs, v_targets = self.calc_advs_v_targets(batch)
            policy_loss = self.calc_policy_loss(batch, advs)  # from actor
            val_loss = self.calc_val_loss(batch, v_targets)  # from critic
            loss = policy_loss + val_loss
            self.net.training_step(loss=loss, lr_clock=clock)
            # reset
            self.to_train = 0
            self.body.flush()
            logger.debug(f'Trained {self.name} at epi: {clock.epi}, total_t: {clock.total_t}, t: {clock.t}, total_reward so far: {self.body.memory.total_reward}, loss: {loss:g}')
            return loss.item()
        else:
            return np.nan

    def train_separate(self):
        '''
        Trains the network when the actor and critic are separate networks
        loss = val_loss + abs(policy_loss)
        '''
        if self.to_train == 1:
            batch = self.sample()
            policy_loss = self.train_actor(batch)
            val_loss = self.train_critic(batch)
            loss = val_loss + abs(policy_loss)
            # reset
            self.to_train = 0
            self.body.flush()
            logger.debug(f'Trained {self.name}, loss: {loss:g}')
            return loss.item()
        else:
            return np.nan

    def train_actor(self, batch):
        '''Trains the actor when the actor and critic are separate networks'''
        with torch.no_grad():
            advs, _v_targets = self.calc_advs_v_targets(batch)
        policy_loss = self.calc_policy_loss(batch, advs)
        self.net.training_step(loss=policy_loss, lr_clock=self.body.env.clock)
        return policy_loss

    def train_critic(self, batch):
        '''Trains the critic when the actor and critic are separate networks'''
        total_val_loss = torch.tensor(0.0, device=self.net.device)
        # training iters only applicable to separate critic network
        for _ in range(self.training_epoch):
            with torch.no_grad():
                _advs, v_targets = self.calc_advs_v_targets(batch)
            val_loss = self.calc_val_loss(batch, v_targets)
            self.critic.training_step(loss=val_loss, lr_clock=self.body.env.clock)
            total_val_loss += val_loss
        val_loss = total_val_loss / self.training_epoch
        return val_loss

    def calc_policy_loss(self, batch, advs):
        '''Calculate the actor's policy loss'''
        assert len(self.body.log_probs) == len(advs), f'batch_size of log_probs {len(self.body.log_probs)} vs advs: {len(advs)}'
        log_probs = torch.stack(self.body.log_probs)
        policy_loss = - self.policy_loss_coef * log_probs * advs
        if self.entropy_coef_spec is not None:
            entropies = torch.stack(self.body.entropies)
            policy_loss += (-self.body.entropy_coef * entropies)
        policy_loss = torch.mean(policy_loss)
        logger.debug(f'Actor policy loss: {policy_loss:g}')
        return policy_loss

    def calc_val_loss(self, batch, v_targets):
        '''Calculate the critic's value loss'''
        v_targets = v_targets.unsqueeze(dim=-1)
        v_preds = self.calc_v(batch['states'], evaluate=False).unsqueeze_(dim=-1)
        assert v_preds.shape == v_targets.shape
        val_loss = self.val_loss_coef * self.net.loss_fn(v_preds, v_targets)
        logger.debug(f'Critic value loss: {val_loss:g}')
        return val_loss

    def calc_gae_advs_v_targets(self, batch):
        '''
        Calculate the GAE advantages and value targets for training actor and critic respectively
        adv_targets = GAE (see math_util method)
        v_targets = adv_targets + v_preds
        before output, adv_targets is standardized (so v_targets used the unstandardized version)
        Used for training with GAE
        '''
        v_preds = self.calc_v(batch['states'])
        # calc next_state boundary value and concat with above for efficiency
        next_v_pred_tail = self.calc_v(batch['next_states'][-1:])
        next_v_preds = torch.cat([v_preds[1:], next_v_pred_tail], dim=0)
        # v targets = r_t + gamma * V(s_(t+1))
        v_targets = math_util.calc_nstep_returns(batch, self.gamma, 1, next_v_preds)
        # ensure val for next_state is 0 at done
        next_v_preds = next_v_preds * (1 - batch['dones'])
        adv_targets = math_util.calc_gaes(batch['rewards'], v_preds, next_v_preds, self.gamma, self.lam)
        adv_targets = math_util.standardize(adv_targets)
        logger.debug(f'adv_targets: {adv_targets}\nv_targets: {v_targets}')
        return adv_targets, v_targets

    def calc_nstep_advs_v_targets(self, batch):
        '''
        Calculate N-step returns advantage = nstep_returns - v_pred
        See n-step advantage under http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf.pdf
        Used for training with N-step (not GAE)
        Returns 2-tuple for API-consistency with GAE
        '''
        next_v_preds = self.calc_v(batch['next_states'])
        v_preds = self.calc_v(batch['states'])
        # v targets = r_t + gamma * V(s_(t+1))
        v_targets = math_util.calc_nstep_returns(batch, self.gamma, 1, next_v_preds)
        nstep_returns = math_util.calc_nstep_returns(batch, self.gamma, self.num_step_returns, next_v_preds)
        nstep_advs = nstep_returns - v_preds
        adv_targets = nstep_advs
        logger.debug(f'adv_targets: {adv_targets}\nv_targets: {v_targets}')
        return adv_targets, v_targets

    def calc_td_advs_v_targets(self, batch):
        '''
        Estimate Q(s_t, a_t) with r_t + gamma * V(s_t+1 ) for simplest AC algorithm
        '''
        next_v_preds = self.calc_v(batch['next_states'])
        # Equivalent to 1-step return
        # v targets = r_t + gamma * V(s_(t+1))
        v_targets = math_util.calc_nstep_returns(batch, self.gamma, 1, next_v_preds)
        adv_targets = v_targets  # Plain Q estimate, called adv for API consistency
        logger.debug(f'adv_targets: {adv_targets}\nv_targets: {v_targets}')
        return adv_targets, v_targets

    @lab_api
    def update(self):
        self.body.explore_var = self.explore_var_scheduler.update(self, self.body.env.clock)
        if self.entropy_coef_spec is not None:
            self.body.entropy_coef = self.entropy_coef_scheduler.update(self, self.body.env.clock)
        return self.body.explore_var
