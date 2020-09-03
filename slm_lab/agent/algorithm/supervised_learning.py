from slm_lab.agent import net
from slm_lab.agent.algorithm import policy_util
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.lib import math_util, util
from slm_lab.agent.net import net_util
from slm_lab.lib.decorator import lab_api
import numpy as np
import torch

from slm_lab.lib import logger
logger = logger.get_logger(__name__)

# TODO change doc
class SupervisedLAPolicy(Algorithm):
    '''
    SpLActPolicy = Supervised learning the action policy

    e.g. algorithm_spec:
    "algorithm": {
        "name": "Reinforce",
        "action_pdtype": "default",
        "action_policy": "default",
        "explore_var_spec": null,
        "gamma": 0.99,
        "entropy_coef_spec": {
          "name": "linear_decay",
          "start_val": 0.01,
          "end_val": 0.001,
          "start_step": 100,
          "end_step": 5000,
        },
        "training_frequency": 1,
    }
    '''

    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
        # set default
        util.set_attr(self, dict(
            action_pdtype='default',
            action_policy='default',
            # center_return=False,
            # normalize_return=False,
            explore_var_spec=None,
            entropy_coef_spec=None,
            policy_loss_coef=1.0,
            targets='actions',
            inputs=['states'],
            normalize_inputs=False,
            training_batch_iter=1,  # how many gradient updates per batch
            training_iter=1,        # how many batch of data to sample
            training_start_step=1,
        ))
        util.set_attr(self, self.algorithm_spec, [
            'action_pdtype',
            'action_policy',
            # 'center_return',  # center by the mean
            # 'normalize_return', # divide by std
            'explore_var_spec',
            # 'gamma',  # the discount factor
            'entropy_coef_spec',
            'policy_loss_coef',
            'training_frequency',
            'targets',
            'inputs',
            'normalize_inputs',
            'training_batch_iter',  # how many gradient updates per batch
            'training_iter',
            'training_start_step',
        ])
        self.to_train = 0
        self.action_policy = getattr(policy_util, self.action_policy)
        self.explore_var_scheduler = policy_util.VarScheduler(self.body.env.clock, self.explore_var_spec)
        self.explore_var_scheduler.start_val
        if self.entropy_coef_spec is not None:
            self.entropy_coef_scheduler = policy_util.VarScheduler(self.body.env.clock, self.entropy_coef_spec)

    @lab_api
    def init_nets(self, global_nets=None):
        '''
        Initialize the neural network used to learn the policy function from the spec
        Below we automatically select an appropriate net for a discrete or continuous action space if the setting is of the form 'MLPNet'. Otherwise the correct type of network is assumed to be specified in the spec.
        Networks for continuous action spaces have two heads and return two values, the first is a tensor containing the mean of the action policy, the second is a tensor containing the std deviation of the action policy. The distribution is assumed to be a Gaussian (Normal) distribution.
        Networks for discrete action spaces have a single head and return the logits for a categorical probability distribution over the discrete actions
        '''
        # in_dim = self.body.observation_dim
        assert self.agent.body.action_space_is_discrete
        in_dim = [0]
        for input in self.inputs:
            if input == "actions":
                in_dim[0] += self.body.action_dim
            elif input == "states":
                tmp = list(self.body.observation_dim)
                tmp[0] += in_dim[0]
                in_dim = tmp
            elif input == "rewards":
                in_dim[0] += 1
            else:
                raise NotImplementedError()

        # out_dim = net_util.get_out_dim(self.body)
        if self.targets == "actions":
            out_dim = self.body.action_dim
        elif self.targets == "states":
            out_dim = self.body.observation_dim
        elif self.targets == "rewards":
            out_dim = [1]
        else:
            raise NotImplementedError()

        NetClass = getattr(net, self.net_spec['type'])
        self.net = NetClass(self.net_spec, in_dim, out_dim, self.body.env.clock,
                            name=f"agent_{self.agent.agent_idx}_algo_{self.algo_idx}_net")
        self.net_names = ['net']
        # init net optimizer and its lr scheduler
        self.optim = net_util.get_optim(self.net, self.net.optim_spec)
        self.lr_scheduler = net_util.get_lr_scheduler(self.optim, self.net.lr_scheduler_spec)
        net_util.set_global_nets(self, global_nets)
        self.post_init_nets()

    @lab_api
    def proba_distrib_params(self, x, net=None):
        '''The pdparam (proba distrib param) will be the logits for discrete prob. dist., or the mean and std for
        continuous prob. dist.'''
        net = self.net if net is None else net

        if self.normalize_inputs:
            # print("x", x.min(), x.max())
            assert x.min() >= 0.0
            assert x.max() <= 1.0
            x = (x - 0.5) / 0.5
            # print("x normalized", x.min(), x.max())
            assert x.min() >= -1.0
            assert x.max() <= 1.0

        pdparam = net(x)
        return pdparam

    @lab_api
    def act(self, state):
        # print("state act", state)
        body = self.body
        self.net.eval()
        with torch.no_grad():
            action, action_pd = self.action_policy(state, self, body)
        self.net.train()

        self.to_log["entropy_act"] = action_pd.entropy().mean().item()

        # print("act", action)
        # print("prob", action_pd.probs.tolist())
        return action.cpu().squeeze().numpy(), action_pd  # squeeze to handle scalar

    @lab_api
    def sample(self, reset=True):
        '''Samples a batch from memory'''
        batch = self.memory.sample(reset=reset)
        batch = util.to_torch_batch(batch, self.net.device, self.memory.is_episodic)
        return batch

    def proba_distrib_param_batch(self, batch):
        '''Efficiently forward to get pdparam and by batch for loss computation'''
        # states = batch['states']
        x = []
        for input in self.inputs:
            data = batch[input]
            if self.body.env.is_venv:
                data = math_util.venv_unpack(data)
            x.append(data)
        x = torch.cat(x, dim=-1)

        pdparam = self.proba_distrib_params(x)
        return pdparam

    def calc_ret_advs(self, batch):
        '''Calculate plain returns; which is generalized to advantage in ActorCritic'''
        rets = math_util.calc_returns(batch['rewards'], batch['dones'], self.gamma)
        if self.center_return:
            rets = math_util.center_mean(rets)
        if self.normalize_return:
            rets = math_util.normalize_var(rets)
        advs = rets
        if self.body.env.is_venv:
            advs = math_util.venv_unpack(advs)
        return advs

    def supervised_learning_loss(self, batch, pdparams):
        '''Calculate the actor's policy loss'''
        # print(f"pdparams {pdparams[0,...]}")
        action_pd = policy_util.init_action_pd(self.ActionPD, pdparams)
        targets = batch[self.targets]
        if self.body.env.is_venv:
            targets = math_util.venv_unpack(targets)

        preds = action_pd.probs # use proba as predictions, not the sampled actions
        if targets.dim() == 1:
            # targets = self.one_hot_embedding(targets.long(), self.agent.body.action_space[self.agent.agent_idx].n)
            targets = self.one_hot_embedding(targets.long(), self.agent.body.action_space.n)

        if isinstance(self.net.loss_fn, torch.nn.SmoothL1Loss):
            # Used with the SmoothL1Loss loss (Huber loss)  where err < 1 => MSE and err > 1 => MAE
            scaling = 2
            supervised_learning_loss = self.net.loss_fn(preds * scaling, targets * scaling) / scaling
            supervised_learning_loss = supervised_learning_loss.mean()
        else:
            supervised_learning_loss = self.net.loss_fn(preds, targets).mean()
        self.to_log["loss_policy"] = supervised_learning_loss

        if self.entropy_coef_spec:
            self.entropy = action_pd.entropy().mean().item()
            self.to_log["entropy_train"] = self.entropy
            self.to_log["entropy_coef"] = self.entropy_coef_scheduler.val
            entropy_loss = (-self.entropy_coef_scheduler.val * self.entropy)
            self.to_log["entropy_over_loss"] = (entropy_loss / supervised_learning_loss + self.episilon).clamp(
                    min=-100, max=100)
            supervised_learning_loss += entropy_loss
        # logger.debug(f'supervised_learning_loss: {supervised_learning_loss:g}')
        # print(f'self.algo_idx {self.algo_idx} supervised_learning_loss: {supervised_learning_loss:g} , '
        #       f'preds {preds[0,...]}')
        return supervised_learning_loss

    def one_hot_embedding(self, labels, num_classes):
        """Embedding labels to one-hot form.

        Args:
          labels: (LongTensor) class labels, sized [N,].
          num_classes: (int) number of classes.

        Returns:
          (tensor) encoded labels, sized [N, #classes].
        """
        y = torch.eye(num_classes)
        return y[labels]

    @lab_api
    def train(self, loss_penalty=None):
        if util.in_eval_lab_modes():
            return np.nan
        clock = self.body.env.clock

        if self.to_train == 1:
            for i in range(self.training_iter):

                batch = self.sample()
                clock.set_batch_size(len(batch))

                # TODO do something about the fact that only the last loss(and other var) is logged and returned
                for _ in range(self.training_batch_iter):

                    # Compute predictions
                    pdparams = self.proba_distrib_param_batch(batch)

                    loss = self.supervised_learning_loss(batch, pdparams)

                    # TODO Clean this mess !!
                    if loss_penalty is not None:
                        loss += loss_penalty
                        self.to_log["loss_penalty"] = loss_penalty
                        print("loss_penalty", loss_penalty)

                    if hasattr(self, "auxilary_loss"):
                        if not np.isnan(self.auxilary_loss):
                            print("loss auxilary_loss", loss, self.auxilary_loss)
                            loss += self.auxilary_loss
                            del self.auxilary_loss
                    if hasattr(self, "lr_overwritter"):
                        lr_source = self.lr_overwritter
                        self.to_log['lr'] = self.lr_overwritter
                    else:
                        lr_source = self.lr_scheduler

                    self.net.train_step(loss, self.optim, lr_source, clock=self.internal_clock, global_net=self.global_net)

                    if hasattr(self, "lr_overwritter"):
                        del self.lr_overwritter
                    else:
                        self.to_log['lr'] = np.mean(self.lr_scheduler.get_lr())

                    self.to_train = 0
                    logger.debug(f'Trained {self.name} at epi: {clock.epi}, frame: {clock.frame}, t: {clock.t}, total_reward so far: {self.body.env.total_reward}, loss: {loss:g}')
                    self.to_log["loss_tot"] = loss.item()
            return loss.item()
        else:
            return np.nan

    @lab_api
    def update(self):
        self.explore_var_scheduler.update(self, self.body.env.clock)
        if self.entropy_coef_spec is not None:
            self.entropy_coef_scheduler.update(self, self.body.env.clock)
        return self.explore_var_scheduler.val
