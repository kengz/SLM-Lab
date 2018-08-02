from copy import deepcopy
from slm_lab.agent import net
from slm_lab.agent.algorithm import math_util, policy_util
from slm_lab.agent.algorithm.reinforce import Reinforce
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import torch
import pydash as ps

logger = logger.get_logger(__name__)


class ActorCritic(Reinforce):
    '''
    Implementation of single threaded Advantage Actor Critic
    Original paper: "Asynchronous Methods for Deep Reinforcement Learning"
    https://arxiv.org/abs/1602.01783
    Algorithm specific spec param:
    use_gae: If false, use the default TD error. Then the algorithm stays as AC. If True, use generalized advantage estimation (GAE) introduced in "High-Dimensional Continuous Control Using Generalized Advantage Estimation https://arxiv.org/abs/1506.02438. The algorithm becomes A2C.
    use_nstep: If false, use the default TD error. Then the algorithm stays as AC. If True, use n-step returns from "Asynchronous Methods for Deep Reinforcement Learning". The algorithm becomes A2C.
    add_entropy: option to add entropy to policy during training to encourage exploration as outlined in "Asynchronous Methods for Deep Reinforcement Learning"
    memory.name: batch (through OnPolicyBatchReplay memory class) or episodic through (OnPolicyReplay memory class)
    num_step_returns: if use_gae is false, this specifies the number of steps used for the N-step returns method.
    lam: is use_gae, this lambda controls the bias variance tradeoff for GAE. Floating point value between 0 and 1. Lower values correspond to more bias, less variance. Higher values to more variance, less bias.
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
        "action_policy_update": "no_update",
        "explore_var_start": null,
        "explore_var_end": null,
        "explore_anneal_epi": null,
        "gamma": 0.99,
        "use_gae": false,
        "lam": 1.0,
        "use_nstep": false,
        "num_step_returns": 100,
        "add_entropy": false,
        "entropy_coef": 0.01,
        "policy_loss_coef": 1.0,
        "val_loss_coef": 0.01,
        "continuous_action_clip": 2.0,
        "training_frequency": 1,
        "training_epoch": 8
    }
    '''

    @lab_api
    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        self.body = self.agent.nanflat_body_a[0]  # single-body algo
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
            'continuous_action_clip',
            'training_frequency',
            'training_epoch',
        ])
        self.to_train = 0
        self.action_policy = getattr(policy_util, self.action_policy)
        self.action_policy_update = getattr(policy_util, self.action_policy_update)
        for body in self.agent.nanflat_body_a:
            body.explore_var = self.explore_var_start
        # Select appropriate methods to calculate adv_targets and v_targets for training
        if self.use_gae:
            self.calc_advs_v_targets = self.calc_gae_advs_v_targets
        elif self.use_nstep:
            self.calc_advs_v_targets = self.calc_nstep_advs_v_targets
        else:
            self.calc_advs_v_targets = self.calc_td_advs_v_targets

    @lab_api
    def init_nets(self):
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
        net_type = self.net_spec['type']
        # options of net_type are {MLPNet, ConvNet, RecurrentNet} x {Shared, Separate}
        in_dim = self.body.state_dim
        if self.body.is_discrete:
            if 'Shared' in net_type:
                self.share_architecture = True
                out_dim = [self.body.action_dim, 1]
            else:
                assert 'Separate' in net_type
                self.share_architecture = False
                out_dim = self.body.action_dim
                critic_out_dim = 1
        else:
            if 'Shared' in net_type:
                self.share_architecture = True
                out_dim = [self.body.action_dim, self.body.action_dim, 1]
            else:
                assert 'Separate' in net_type
                self.share_architecture = False
                out_dim = [self.body.action_dim, self.body.action_dim]
                critic_out_dim = 1

        self.net_spec['type'] = net_type = net_type.replace('Shared', '').replace('Separate', '')
        if 'MLP' in net_type and ps.is_list(out_dim) and len(out_dim) > 1:
            self.net_spec['type'] = 'MLPHeterogenousTails'

        actor_net_spec = self.net_spec.copy()
        critic_net_spec = self.net_spec.copy()
        for k in self.net_spec:
            if 'actor_' in k:
                actor_net_spec[k.replace('actor_', '')] = actor_net_spec.pop(k)
                critic_net_spec.pop(k)
            if 'critic_' in k:
                critic_net_spec[k.replace('critic_', '')] = critic_net_spec.pop(k)
                actor_net_spec.pop(k)

        NetClass = getattr(net, self.net_spec['type'])
        # properly set net_spec and action_dim for actor, critic nets
        if self.share_architecture:
            # net = actor_critic as one
            self.net = NetClass(actor_net_spec, self, in_dim, out_dim)
            self.net_names = ['net']
        else:
            # main net = actor
            self.net = NetClass(actor_net_spec, self, in_dim, out_dim)
            if critic_net_spec['use_same_optim']:
                critic_net_spec = actor_net_spec
            # stand-alone critic does not use Heterogenous tails
            CriticNetClass = getattr(net, self.net_spec['type'].replace('HeterogenousTails', 'Net'))
            self.critic = CriticNetClass(critic_net_spec, self, in_dim, critic_out_dim)
            self.net_names = ['net', 'critic']
        self.post_init_nets()

    @lab_api
    def calc_pdparam(self, x, evaluate=True):
        '''
        The pdparam will be the logits for discrete prob. dist., or the mean and std for continuous prob. dist.
        '''
        if evaluate:
            pdparam = self.net.wrap_eval(x)
        else:
            self.net.train()
            pdparam = self.net(x)
        if self.share_architecture:
            # MLPHeterogenousTails, get front (no critic)
            if self.body.is_discrete:
                return pdparam[0]
            else:
                return pdparam[:-1]
        else:
            return pdparam

    def calc_v(self, x, evaluate=True):
        '''
        Forward-pass to calculate the predicted state-value from critic.
        '''
        if self.share_architecture:
            if evaluate:
                out = self.net.wrap_eval(x)
            else:
                self.net.train()
                out = self.net(x)
            # MLPHeterogenousTails, get last
            v = out[-1].squeeze_(dim=1)
        else:
            if evaluate:
                out = self.critic.wrap_eval(x)
            else:
                self.critic.train()
                out = self.critic(x)
            v = out.squeeze_(dim=1)
        return v

    @lab_api
    def train(self):
        '''Trains the algorithm'''
        if util.get_lab_mode() == 'enjoy':
            return np.nan
        if self.share_architecture:
            return self.train_shared()
        else:
            return self.train_separate()

    def train_shared(self):
        '''
        Trains the network when the actor and critic share parameters
        loss = self.policy_loss_coef * policy_loss + self.val_loss_coef * val_loss
        '''
        if self.to_train == 1:
            batch = self.sample()
            with torch.no_grad():
                advs, v_targets = self.calc_advs_v_targets(batch)
            policy_loss = self.calc_policy_loss(batch, advs)  # from actor
            val_loss = self.calc_val_loss(batch, v_targets)  # from critic
            loss = policy_loss + val_loss
            self.net.training_step(loss=loss)
            # reset
            self.to_train = 0
            self.body.log_probs = []
            self.body.entropies = []
            logger.debug(f'Total loss: {loss:.2f}')
            self.last_loss = loss.item()
        return self.last_loss

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
            self.body.entropies = []
            self.body.log_probs = []
            logger.debug(f'Total loss: {loss:.2f}')
            self.last_loss = loss.item()
        return self.last_loss

    def train_actor(self, batch):
        '''Trains the actor when the actor and critic are separate networks'''
        with torch.no_grad():
            advs, _v_targets = self.calc_advs_v_targets(batch)
        policy_loss = self.calc_policy_loss(batch, advs)
        self.net.training_step(loss=policy_loss)
        return policy_loss

    def train_critic(self, batch):
        '''Trains the critic when the actor and critic are separate networks'''
        total_val_loss = torch.tensor(0.0)
        # training iters only applicable to separate critic network
        for _ in range(self.training_epoch):
            with torch.no_grad():
                _advs, v_targets = self.calc_advs_v_targets(batch)
            val_loss = self.calc_val_loss(batch, v_targets)
            self.critic.training_step(loss=val_loss)
            total_val_loss += val_loss.cpu()
        val_loss = total_val_loss / self.training_epoch
        return val_loss

    def calc_policy_loss(self, batch, advs):
        '''Calculate the actor's policy loss'''
        assert len(self.body.log_probs) == len(advs), f'{len(self.body.log_probs)} vs {len(advs)}'
        log_probs = torch.stack(self.body.log_probs)
        policy_loss = - self.policy_loss_coef * log_probs * advs
        if self.add_entropy:
            entropies = torch.stack(self.body.entropies)
            policy_loss += (-self.entropy_coef * entropies)
        policy_loss = torch.mean(policy_loss)
        if torch.cuda.is_available() and self.net.gpu:
            policy_loss = policy_loss.cuda()
        logger.debug(f'Actor policy loss: {policy_loss:.2f}')
        return policy_loss

    def calc_val_loss(self, batch, v_targets):
        '''Calculate the critic's value loss'''
        v_targets = v_targets.unsqueeze(dim=-1)
        v_preds = self.calc_v(batch['states'], evaluate=False).unsqueeze_(dim=-1)
        assert v_preds.shape == v_targets.shape
        val_loss = self.val_loss_coef * self.net.loss_fn(v_preds, v_targets)
        if torch.cuda.is_available() and self.net.gpu:
            val_loss = val_loss.cuda()
        logger.debug(f'Critic value loss: {val_loss:.2f}')
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
        # ensure val for next_state is 0 at done
        next_v_preds = next_v_preds * (1 - batch['dones'])

        # v_targets = gae_targets + v_preds
        adv_targets = math_util.calc_gaes(batch['rewards'], v_preds, next_v_preds, self.gamma, self.lam)
        v_targets = adv_targets + v_preds
        if torch.cuda.is_available() and self.net.gpu:
            adv_targets = adv_targets.cuda()
            v_targets = v_targets.cuda()

        # standardization trick
        # guard nan std by setting to 0 and add small const
        adv_std = adv_targets.std()
        adv_std[adv_std != adv_std] = 0
        adv_std += 1e-08
        adv_targets = (adv_targets - adv_targets.mean()) / adv_std
        return adv_targets, v_targets

    def calc_nstep_advs_v_targets(self, batch):
        '''
        Calculate N-step returns advantage = nstep_returns - v_pred
        See n-step advantage under http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf.pdf
        Used for training with N-step (not GAE)
        Returns 2-tuple for API-consistency with GAE
        '''
        v_preds = self.calc_v(batch['states'])
        nstep_returns = math_util.calc_nstep_returns(batch, self.gamma, self.num_step_returns, v_preds)
        nstep_advs = nstep_returns - v_preds
        if torch.cuda.is_available() and self.net.gpu:
            nstep_advs = nstep_advs.cuda()
        adv_targets = v_targets = nstep_advs
        return adv_targets, v_targets

    def calc_td_advs_v_targets(self, batch):
        '''
        Calculate plain TD error and target for plain AC algorithm
        '''
        v_preds = self.calc_v(batch['states'])
        # TD is equivalent to 1-step return
        td_returns = math_util.calc_nstep_returns(batch, self.gamma, 1, v_preds)
        if torch.cuda.is_available() and self.net.gpu:
            td_returns = td_returns.cuda()
        v_targets = td_returns
        adv_targets = v_targets - v_preds  # TD error, but called adv for API consistency
        return adv_targets, v_targets

    @lab_api
    def update(self):
        nets = [self.net] if self.share_architecture else [self.net, self.critic]
        for net in nets:
            net.update_lr()
        explore_vars = [self.action_policy_update(self, body) for body in self.agent.nanflat_body_a]
        explore_var_a = self.nanflat_to_data_a('explore_var', explore_vars)
        return explore_var_a
