from mpi4py import MPI
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.agent.net.mlp_policy import MLPPolicy
from slm_lab.lib import logger, math_util, tf_util, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import pydash as ps
import tensorflow as tf

logger = logger.get_logger(__name__)


class PPO(Algorithm):
    '''
    Implementation of PPO
    Original paper: "Proximal Policy Optimization Algorithms"
    https://arxiv.org/pdf/1707.06347.pdf

    Adapted from OpenAI baselines, CPU version https://github.com/openai/baselines/tree/master/baselines/ppo1
    Algorithm:
    for iteration = 1, 2, 3, ... do
        for actor = 1, 2, 3, ..., N do
            run policy pi_old in env for T timesteps
            compute advantage A_1, ..., A_T
        end for
        optimize surrogate L wrt theta, with K epochs and minibatch size M <= NT
    end for
    '''

    @lab_api
    def post_body_init(self):
        '''Initializes the part of algorithm needing a body to exist first.'''
        self.comm = MPI.Comm.Clone(MPI.COMM_WORLD)
        # TODO fix access to info_space then set below
        tf_util.make_session(num_cpus=1).__enter__()
        # TODO close when done to clear
        self.cur_lr_mult = 1.0
        self.body = self.agent.nanflat_body_a[0]  # singleton algo
        self.body.memory = self.body.memory
        self.init_algorithm_params()
        self.init_nets()
        logger.info(util.self_desc(self))

    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
        util.set_attr(self, self.algorithm_spec, [
            'clip_eps',
            'ent_coef',
            'adam_epsilon',
            'gamma',
            'lam',
            'horizon',
            'epoch',
            'lr',
            'max_frame',
            'schedule',
        ])

    @lab_api
    def init_nets(self):
        '''
        Init policy network
        Construct the loss function for PPO which tf can minimize:
        (subscript t is omitted)
        L^{CLIP+VF+S} = E[ L^CLIP - c1 * L^VF + c2 * S[pi](s) ]

        Breakdown piecewise,
        1. L^CLIP = E[ min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) ]
        where ratio = pi(a|s) / pi_old(a|s)

        2. L^VF = E[ (V(s_t) - V^target)^2 ]

        3. S = E[ entropy ]
        '''
        self.pi = pi = MLPPolicy(self.net_spec, self, self.body, 'pi')
        self.pi_old = pi_old = MLPPolicy(self.net_spec, self, self.body, 'pi_old')
        ob = pi.ob
        ac = pi.pdtype.sample_placeholder([None])
        # target advantage function
        adv_target = tf.placeholder(name='adv_target', dtype=tf.float32, shape=[None])
        # empirical return
        v_target = tf.placeholder(name='v_target', dtype=tf.float32, shape=[None])
        # learning rate multiplier for schedule
        lr_mult = tf.placeholder(name='lr_mult', dtype=tf.float32, shape=[])
        # annealed clipping param epsilon for L^CLIP
        clip_eps = self.clip_eps * lr_mult

        # L^CLIP
        ratio = tf.exp(pi.pd.logp(ac) - pi_old.pd.logp(ac))
        sur_1 = ratio * adv_target
        sur_2 = tf.clip_by_value(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_target
        # flip sign because need to maximize
        loss_clip = -tf.reduce_mean(tf.minimum(sur_1, sur_2))

        # L^VF
        loss_vf = tf.losses.mean_squared_error(v_target, pi.v_pred)

        # S entropy bonus, some variables for diagnosis
        kl_mean = tf.reduce_mean(pi_old.pd.kl(pi.pd))
        ent_mean = tf.reduce_mean(pi.pd.entropy())
        ent_penalty = -self.ent_coef * ent_mean

        total_loss = loss_clip + loss_vf + ent_penalty

        # variables
        self.losses = [loss_clip, loss_vf, kl_mean, ent_mean, ent_penalty]
        self.loss_names = ['loss_clip', 'loss_vf', 'kl_mean', 'ent_mean', 'ent_penalty']
        var_list = pi.get_trainable_variables()
        inputs = [ob, ac, adv_target, v_target, lr_mult]
        outputs = self.losses + [tf_util.flat_grad(total_loss, var_list)]

        # compute functions
        self.adam = tf_util.MpiAdam(var_list, epsilon=self.adam_epsilon, comm=self.comm)
        self.compute_loss_grad = tf_util.function(inputs, outputs)
        self.compute_losses = tf_util.function(inputs, self.losses)
        self.update_pi = tf_util.function([], [], updates=[
            tf.assign(old_w, new_w) for (old_w, new_w) in zip(pi_old.get_variables(), pi.get_variables())
        ])
        tf_util.init()
        self.adam.sync()

    @lab_api
    def body_act_discrete(self, body, state):
        action, self.body.memory.v_pred = self.pi.act(state)
        return action

    @lab_api
    def body_act_continuous(self, body, state):
        action, self.body.memory.v_pred = self.pi.act(state)
        return action

    @lab_api
    def sample(self):
        '''Samples a batch from memory'''
        seg = self.body.memory.sample()
        return seg

    @lab_api
    def train(self):
        total_t = self.body.env.clock.get('total_t')
        if total_t > 0 and (total_t % self.horizon == 0):
            mean_losses = self._train()
            loss = mean_losses.sum()
        else:
            loss = np.nan
        return loss

    def _train(self):
        seg = self.sample()  # sample a segment
        self.add_v_target_and_adv(seg)
        obs, acs, adv_targets, v_targets = seg['obs'], seg['acs'], seg['advs'], seg['tdlamrets']

        # standardized advantage function estimate
        adv_targets = (adv_targets - adv_targets.mean()) / adv_targets.std()
        data = {'obs': obs, 'acs': acs, 'adv_targets': adv_targets, 'v_targets': v_targets}
        dataset = math_util.Dataset(data, shuffle=True)

        self.pi.ob_rms.update(obs)  # only if is not image
        self.update_pi()

        # compute gradient
        for _i in range(self.epoch):
            losses = []
            for batch in dataset.iterate_once(self.body.memory.batch_size):
                inputs = [batch[k] for k in ['obs', 'acs', 'adv_targets', 'v_targets']]
                inputs.append(self.cur_lr_mult)
                outputs = self.compute_loss_grad(*inputs)
                g = outputs.pop()
                assert not np.isnan(g).any(), f'grad has nan: {g}'
                self.adam.update(g, self.cur_lr_mult * self.lr)
                losses.append(outputs)
            mean_losses = np.mean(losses, axis=0)

        # compute losses
        losses = []
        for batch in dataset.iterate_once(self.body.memory.batch_size):
            inputs = [batch[k] for k in ['obs', 'acs', 'adv_targets', 'v_targets']]
            inputs.append(self.cur_lr_mult)
            new_losses = self.compute_losses(*inputs)
            losses.append(new_losses)
        mean_losses, _std, _count = tf_util.mpi_moments(losses, axis=0, comm=self.comm)
        logger.debug(f'Training losses {list(zip(self.loss_names, mean_losses))}')
        return mean_losses

    @lab_api
    def update(self):
        if self.schedule == 'constant':
            self.cur_lr_mult = 1.0
        elif self.schedule == 'linear':
            self.cur_lr_mult = max(1.0 - self.body.env.clock.get('total_t') / self.max_frame, 0.0)
        else:
            raise NotImplementedError

        explore_var = np.nan
        return explore_var

    def add_v_target_and_adv(self, seg):
        '''Compute advantage given a segment of trajectory'''
        news = np.append(seg['news'], 0)
        v_preds = np.append(seg['v_preds'], seg['next_v_pred'])
        rews = seg['rews']
        T = len(rews)
        seg['advs'] = gaelams = np.empty(T, 'float32')
        last_gaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - news[t + 1]
            delta = rews[t] + self.gamma * v_preds[t + 1] * nonterminal - v_preds[t]
            gaelams[t] = last_gaelam = delta + self.gamma * self.lam * nonterminal * last_gaelam
        assert not np.isnan(gaelams).any(), f'GAE has nan: {gaelams}'
        seg['tdlamrets'] = seg['advs'] + seg['v_preds']
