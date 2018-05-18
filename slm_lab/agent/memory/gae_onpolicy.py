from collections import deque
from slm_lab.agent.memory.base import Memory
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np

logger = logger.get_logger(__name__)


class GAEOnPolicyReplay(Memory):
    '''
    On-policy replay memory that supports GAE computation
    Needs to store v_pred computed with action whenever agent acts
    and support the creation of trajectory segment used for computation and training
    adapted from OpenAI https://github.com/openai/baselines/blob/master/baselines/ppo1/pposgd_simple.py
    '''

    def __init__(self, memory_spec, algorithm, body):
        super(GAEOnPolicyReplay, self).__init__(memory_spec, algorithm, body)
        util.set_attr(self, self.memory_spec, [
            'batch_size',
        ])
        util.set_attr(self, self.agent_spec['algorithm'], ['horizon'])
        self.v_pred = np.nan
        self.state_buffer = deque(maxlen=0)  # for API consistency
        self.reset()

    def reset(self):
        self.total_t = 0
        self.done = False
        self.new = True  # is the start of epi, to set v=0

        horizon = self.horizon
        # just for shape
        sample_ob = self.body.observation_space.sample()
        ob = np.zeros(shape=sample_ob.shape, dtype=sample_ob.dtype)
        ac = self.body.action_space.sample()
        self.obs = np.array([ob for _ in range(horizon)])
        self.acs = np.array([ac for _i in range(horizon)])
        self.v_preds = np.zeros(horizon, 'float32')
        self.rews = np.zeros(horizon, 'float32')
        self.news = np.zeros(horizon, 'int8')

        self.cur_epi_ret = 0.0
        self.cur_epi_len = 0
        self.epi_rets = []
        self.epi_lens = []

    @lab_api
    def update(self, action, reward, state, done):
        self.base_update(action, reward, state, done)
        if not np.isnan(reward):  # not the start of episode
            pass  # current memory design absorbs episodic interface
        self.add_experience(self.last_state, action, reward, state, done)
        self.last_state = state

    def add_experience(self, state, action, reward, next_state, done):
        '''Interface helper method for update() to add experience to memory'''
        i = self.total_t % self.horizon
        # ignore terminal state and the followed tuple of action, v, etc. except reward
        if not done:
            self.obs[i] = state
            self.acs[i] = action
            self.v_preds[i] = self.v_pred
            self.next_v_pred = self.v_pred * (1 - self.new)
            self.news[i] = self.new
        # reward index is offset for computation and ease of saving to index
        if not self.new:  # not at new epi, i.e. previous done == true
            rew_offset_idx = i - 1
            if rew_offset_idx >= 0:
                self.rews[rew_offset_idx] = reward

        self.new = self.done = done

        self.cur_epi_ret += reward
        self.cur_epi_len += 1
        if done:
            self.epi_rets.append(self.cur_epi_ret)
            self.epi_lens.append(self.cur_epi_len)
            # episodic-reset
            self.cur_epi_ret = 0.0
            self.cur_epi_len = 0
        self.last_state = next_state
        self.total_t += 1

    @lab_api
    def sample(self):
        segment = {
            'obs': self.obs,
            'acs': self.acs,
            'v_preds': self.v_preds,
            'next_v_pred': self.next_v_pred,
            'rews': self.rews,
            'news': self.news,
            'epi_rets': self.epi_rets,
            'epi_lens': self.epi_lens,
        }
        return segment
