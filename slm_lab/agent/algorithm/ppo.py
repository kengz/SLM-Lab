from slm_lab.agent import memory
from slm_lab.agent import net
from slm_lab.agent.algorithm.algorithm_util import act_fns, act_update_fns, decay_learning_rate
from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.agent.net import net_util
from slm_lab.lib import distribution, logger, util
from slm_lab.lib.decorator import lab_api
from torch.autograd import Variable
import numpy as np
import torch
import pydash as _
import tensorflow as tf

# TODO may need to switch to non-MPI version


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * \
            batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


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
        self.init_nets()
        self.init_algo_params()
        logger.info(util.self_desc(self))

    @lab_api
    def init_nets(self):
        '''Initialize the neural network used to learn the Q function from the spec'''
        body = self.agent.nanflat_body_a[0]  # singleton algo
        state_dim = body.state_dim
        action_dim = body.action_dim
        self.is_discrete = body.is_discrete
        net_spec = self.agent.spec['net']
        net_kwargs = util.compact_dict(dict(
            hid_layers_activation=_.get(net_spec, 'hid_layers_activation'),
            optim_param=_.get(net_spec, 'optim'),
            loss_param=_.get(net_spec, 'loss'),
            clamp_grad=_.get(net_spec, 'clamp_grad'),
            clamp_grad_val=_.get(net_spec, 'clamp_grad_val'),
        ))

        # NOTE OpenAI: init tf net
        self.pdtype = distribution.make_pdtype(body)

        # observation placeholder
        self.ob = tf.placeholder(name='ob', dtype=tf.float32, shape=(state_dim,))

        # TODO externalize into preprocessor, standardize into z-score
        with tf.variable_scope('obfilter'):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value(
                (self.ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = obz
            for i, hid_size in enumerate(net_spec['hid_layers']):
                # TODO dont hard code activation
                last_out = tf.nn.tanh(tf.layers.dense(
                    last_out, hid_size, name=f'fc_{i+1}', kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(
                last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:, 0]

        with tf.variable_scope('pol'):
            last_out = obz
            for i, hid_size in enumerate(net_spec['hid_layers']):
                last_out = tf.nn.tanh(tf.layers.dense(
                    last_out, hid_size, name=f'fc_{i+1}', kernel_initializer=U.normc_initializer(1.0)))
            # TODO restore param gaussian_fixed_var=True
            # continuous action output layer
            if gaussian_fixed_var and body.action_space == 'Box':
                mean = tf.layers.dense(
                    last_out, self.pdtype.param_shape()[0] // 2, name='final', kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(
                    name='logstd', shape=[1, self.pdtype.param_shape()[0] // 2], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(
                    last_out, self.pdtype.param_shape()[0], name='final', kernel_initializer=U.normc_initializer(0.01))

        self.pd = self.pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        # switcher to sample or use mode
        # stochastic = tf.placeholder(dtype=tf.bool, shape=())
        # action placeholder to sample or use mode
        # ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        # stochastic is always true anyway
        self.ac = tf.cond(True, lambda x: self.pd.sample(), lambda x: self.pd.mode())
        # action function to output action and its value
        # self._act = U.function([ob], [ac, self.vpred])

    @lab_api
    def init_algo_params(self):
        '''Initialize other algorithm parameters'''
        algorithm_spec = self.agent.spec['algorithm']
        net_spec = self.agent.spec['net']
        # Automatically selects appropriate discrete or continuous action policy if setting is default
        action_fn = algorithm_spec['action_policy']
        if action_fn == 'default':
            if self.is_discrete:
                self.action_policy = act_fns['softmax']
            else:
                self.action_policy = act_fns['gaussian']
        else:
            self.action_policy = act_fns[action_fn]
        util.set_attr(self, _.pick(algorithm_spec, [
            'gamma',
            'num_epis_to_collect',
            'add_entropy', 'entropy_weight',
            'continuous_action_clip'
        ]))
        util.set_attr(self, _.pick(net_spec, [
            'decay_lr', 'decay_lr_frequency', 'decay_lr_min_timestep',
        ]))
        # To save on a forward pass keep the log probs from each action
        self.saved_log_probs = []
        self.entropy = []

        self.to_train = 0

    @lab_api
    def body_act_discrete(self, body, state):
        # TODO uhh auto discrete or cont?
        # ac1, vpred1 = self._act(ob[None])
        ac1, vpred1 = tf.get_default_session.run([self.ac, self.vpred], feed_dict={self.ob: state})
        return ac1[0], vpred1[0]
        # return self.action_policy(self, state, body)

    @lab_api
    def body_act_continuous(self, body, state):
        # TODO uhh auto discrete or cont?
        # ac1, vpred1 = self._act(stochastic, ob[None])
        ac1, vpred1 = tf.get_default_session.run([self.ac, self.vpred], feed_dict={self.ob: state})
        return ac1[0], vpred1[0]
        # return self.action_policy(self, state, body)

    def generate_traj_segment():
        '''It is online anyway, so run per episode to generate trajectory'''
        return

    def add_vtarg_and_adv(self, seg, gamma, lam):
        """
        Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
        """
        new = np.append(
            seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
        vpred = np.append(seg["vpred"], seg["nextvpred"])
        T = len(seg["rew"])
        seg["adv"] = gaelam = np.empty(T, 'float32')
        rew = seg["rew"]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - new[t + 1]
            delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam = delta + \
                gamma * lam * nonterminal * lastgaelam
        seg["tdlamret"] = seg["adv"] + seg["vpred"]

    def ppo_loss():
        '''
        Construct the loss function for PPO which tf can minimize:
        (subscript t is omitted)
        L^{CLIP+VF+S} = E[ L^CLIP - c1 * L^VF + c2 * S[pi](s) ]

        Breakdown piecewise,
        1. L^CLIP = E[ min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) ]
        where ratio = pi(a|s) / pi_old(a|s)

        2. L^VF = E[ (V(s_t) - V^target)^2 ]

        3. S = E[ entropy ]
        '''

        # L^CLIP
        self.val_net
        self.pi_net
        self.pi_old_net
        adv_target  # target advantage
        pi_output  # = output layer of pi_net
        pi_old_output  # = output layer of pi_old_net
        pi_pd = pdtype.pdfromflat(pi_output)
        pi_old_pd = pdtype.pdfromflat(pi_old_output)
        ratio = tf.exp(pi_pd.logp(action) - pi_old_pd.logp(action))
        sur_1 = ratio * adv_target
        sur_2 = tf.clip_by_value(
            ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_target
        # flip sign cuz need to maximize
        loss_clip = -tf.reduce_mean(tf.minimum(sur_1, sur_2))

        # L^VF
        # where v_emp = discounted rewards
        loss_vf = tf.reduce_mean(tf.square(val_net.v_pred - v_emp))

        # S
        loss_ent = -entropy_weight * tf.reduce_mean(pi_pd.entropy())

        loss = loss_clip + loss_vf + loss_ent

        return loss

    def learn():
        # Construct network for new policy
        pi = policy_fn("pi", ob_space, ac_space)
        # Network for old policy
        oldpi = policy_fn("oldpi", ob_space, ac_space)
        # Target advantage function (if applicable)
        atarg = tf.placeholder(dtype=tf.float32, shape=[None])
        ret = tf.placeholder(dtype=tf.float32, shape=[
                             None])  # Empirical return

        # learning rate multiplier, updated with schedule
        lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])
        clip_param = clip_param * lrmult  # Annealed cliping parameter epislon

        ob = U.get_placeholder_cached(name="ob")
        ac = pi.pdtype.sample_placeholder([None])

        kloldnew = oldpi.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        pol_entpen = (-entcoeff) * meanent

        ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # pnew / pold
        surr1 = ratio * atarg  # surrogate from conservative policy iteration
        surr2 = tf.clip_by_value(
            ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg
        # maximize, so sign flip
        # PPO's pessimistic surrogate (L^CLIP)
        pol_surr = -tf.reduce_mean(tf.minimum(surr1, surr2))

        # rms loss
        vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
        total_loss = pol_surr + pol_entpen + vf_loss
        losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
        loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

        var_list = pi.get_trainable_variables()
        lossandgrad = U.function(
            [ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
        adam = MpiAdam(var_list, epsilon=adam_epsilon)

        assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
        compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

        U.initialize()
        adam.sync()

        # Prepare for rollouts
        # ----------------------------------------
        seg_gen = traj_segment_generator(
            pi, env, timesteps_per_actorbatch, stochastic=True)

        episodes_so_far = 0
        timesteps_so_far = 0
        iters_so_far = 0
        tstart = time.time()
        lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
        rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards

        assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                    max_seconds > 0]) == 1, "Only one time constraint permitted"

        while True:
            if callback:
                callback(locals(), globals())
            if max_timesteps and timesteps_so_far >= max_timesteps:
                break
            elif max_episodes and episodes_so_far >= max_episodes:
                break
            elif max_iters and iters_so_far >= max_iters:
                break
            elif max_seconds and time.time() - tstart >= max_seconds:
                break

            if schedule == 'constant':
                cur_lrmult = 1.0
            elif schedule == 'linear':
                cur_lrmult = max(
                    1.0 - float(timesteps_so_far) / max_timesteps, 0)
            else:
                raise NotImplementedError

            logger.log("********** Iteration %i ************" % iters_so_far)

            seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, gamma, lam)

            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            # predicted value function before udpate
            vpredbefore = seg["vpred"]
            # standardized advantage function estimate
            atarg = (atarg - atarg.mean()) / atarg.std()
            d = Dataset(dict(ob=ob, ac=ac, atarg=atarg,
                             vtarg=tdlamret), shuffle=not pi.recurrent)
            optim_batchsize = optim_batchsize or ob.shape[0]

            if hasattr(pi, "ob_rms"):
                pi.ob_rms.update(ob)  # update running mean/std for policy

            assign_old_eq_new()  # set old parameter values to new parameter values
            logger.log("Optimizing...")
            logger.log(fmt_row(13, loss_names))
            # Here we do a bunch of optimization epochs over the data
            for _ in range(optim_epochs):
                losses = []  # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize):
                    *newlosses, g = lossandgrad(
                        batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    adam.update(g, optim_stepsize * cur_lrmult)
                    losses.append(newlosses)
                logger.log(fmt_row(13, np.mean(losses, axis=0)))

            logger.log("Evaluating losses...")
            losses = []
            for batch in d.iterate_once(optim_batchsize):
                newlosses = compute_losses(
                    batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                losses.append(newlosses)
            meanlosses, _, _ = mpi_moments(losses, axis=0)
            logger.log(fmt_row(13, meanlosses))
            for (lossval, name) in zipsame(meanlosses, loss_names):
                logger.record_tabular("loss_" + name, lossval)
            logger.record_tabular(
                "ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
            lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
            lens, rews = map(flatten_lists, zip(*listoflrpairs))
            lenbuffer.extend(lens)
            rewbuffer.extend(rews)
            logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            logger.record_tabular("EpRewMean", np.mean(rewbuffer))
            logger.record_tabular("EpThisIter", len(lens))
            episodes_so_far += len(lens)
            timesteps_so_far += sum(lens)
            iters_so_far += 1
            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)
            if MPI.COMM_WORLD.Get_rank() == 0:
                logger.dump_tabular()

    # ok at this stage it shd be sufficient to port functions over directly

    @lab_api
    def sample(self):
        '''Samples a batch from memory'''
        batches = [body.memory.sample()
                   for body in self.agent.nanflat_body_a]
        batch = util.concat_dict(batches)
        batch = util.to_torch_nested_batch_ex_rewards(batch)
        return batch

    @lab_api
    def train(self):
        if self.to_train == 1:
            logger.debug2(f'Training...')
            # We only care about the rewards from the batch
            rewards = self.sample()['rewards']
            logger.debug3(f'Length first epi: {len(rewards[0])}')
            logger.debug3(f'Len log probs: {len(self.saved_log_probs)}')
            self.net.optim.zero_grad()
            policy_loss = self.get_policy_loss(rewards)
            loss = policy_loss.data[0]
            policy_loss.backward()
            if self.net.clamp_grad:
                logger.debug("Clipping gradient...")
                torch.nn.utils.clip_grad_norm(
                    self.net.parameters(), self.net.clamp_grad_val)
            logger.debug2(f'Gradient norms: {self.net.get_grad_norms()}')
            self.net.optim.step()
            self.to_train = 0
            self.saved_log_probs = []
            self.entropy = []
            logger.debug(f'Policy loss: {loss}')
            return loss
        else:
            return np.nan

    @lab_api
    def update(self):
        self.update_learning_rate()
        '''No update needed to explore var'''
        explore_var = np.nan
        return explore_var
