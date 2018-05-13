from slm_lab.agent.net.base import Net
from slm_lab.lib import distribution, tf_util, util
import pydash as ps
import tensorflow as tf


class MLPPolicy(Net):
    '''
    Policy network
    adapted from OpenAI https://github.com/openai/baselines/blob/master/baselines/ppo1/mlp_policy.py
    '''

    def __init__(self, algorithm, body, name=''):
        super(MLPPolicy, self).__init__(algorithm, body)
        util.set_attr(self, self.net_spec, [
            'hid_layers', 'hid_layers_activation'
        ])
        info_space = algorithm.agent.info_space
        scope = util.get_prepath(algorithm.agent.agent_space.spec, info_space, unit='session').split('/')[-1] + '_' + name
        with tf.variable_scope(scope):
            self.scope = tf.get_variable_scope().name
            self._init()

    def _init(self):
        self.pdtype = distribution.make_pdtype(self.body.action_space)

        self.ob = tf_util.get_global_placeholder(
            name='ob', dtype=tf.float32, shape=[None] + list(self.body.observation_space.shape))

        with tf.variable_scope('ob_filter'):
            self.ob_rms = tf_util.RunningMeanStd(shape=self.body.observation_space.shape, comm=self.algorithm.comm)

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((self.ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = obz
            for i, hid_size in enumerate(self.hid_layers):
                last_out = getattr(tf.nn, self.hid_layers_activation)(tf.layers.dense(
                    last_out, hid_size, name=f'fc_{i+1}', kernel_initializer=tf_util.normc_initializer(1.0)))
            self.v_pred = tf.layers.dense(
                last_out, 1, name='final', kernel_initializer=tf_util.normc_initializer(1.0))[:, 0]

        with tf.variable_scope('pol'):
            last_out = obz
            for i, hid_size in enumerate(self.hid_layers):
                last_out = getattr(tf.nn, self.hid_layers_activation)(tf.layers.dense(
                    last_out, hid_size, name=f'fc_{i+1}', kernel_initializer=tf_util.normc_initializer(1.0)))
            # TODO restore param gaussian_fixed_var=True
            gaussian_fixed_var = True
            # continuous action output layer
            if gaussian_fixed_var and not self.body.is_discrete:
                mean = tf.layers.dense(
                    last_out, self.pdtype.param_shape()[0] // 2, name='final', kernel_initializer=tf_util.normc_initializer(0.01))
                logstd = tf.get_variable(
                    name='logstd', shape=[1, self.pdtype.param_shape()[0] // 2], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(
                    last_out, self.pdtype.param_shape()[0], name='final', kernel_initializer=tf_util.normc_initializer(0.01))

        self.pd = self.pdtype.pdfromflat(pdparam)
        self.ac = self.pd.sample()
        self.ac_fn = tf_util.function([self.ob], [self.ac, self.v_pred])

    def act(self, state):
        actions, v_preds = self.ac_fn(state[None])
        return actions[0], v_preds[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
