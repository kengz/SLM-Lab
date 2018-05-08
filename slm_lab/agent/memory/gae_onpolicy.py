from slm_lab.agent.memory.base import Memory
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import copy

logger = logger.get_logger(__name__)


class GAEOnPolicyReplay(Memory):
    '''
    On-policy replay memory that supports GAE computation
    Needs to store v_pred computed with action whenever agent acts
    and support the creation of trajectory segment used for computation and training
    '''

    def __init__(self, body):
        super(GAEOnPolicyReplay, self).__init__(body)
        # TODO properly design sub specs
        self.horizon = self.memory_spec['horizon']
        self.last_state = None
        self.done = None
        self.v_pred = None

    def reset(self):
        # self.last_state is updated from Memory.reset_last_state
        self.t = 0
        self.done = False
        self.new = True  # is the start of epi, to set v=0

        ac = self.body.env.action_space.sample()
        horizon = self.horizon
        self.obs = np.array([self.last_state for _ in range(horizon)])
        self.acs = np.array([ac for _i in range(horizon)])
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
        self.add_experience(self.last_state, action, reward, state, done)
        self.last_state = state

    def add_experience(self, state, action, reward, next_state, done):
        '''Interface helper method for update() to add experience to memory'''
        i = self.t % self.horizon
        self.obs[i] = state
        self.acs[i] = action
        self.v_preds[i] = self.v_pred
        self.next_v_pred = self.v_pred * (1 - self.new)
        self.rews[i] = self.reward
        self.news[i] = self.new
        self.new = self.done = done

        self.cur_epi_ret += reward
        self.cur_epi_len += 1
        if self.done:
            self.epi_rets.append(self.cur_epi_ret)
            self.epi_lens.append(self.cur_epi_len)
            # episodic-reset
            self.cur_epi_ret = 0.0
            self.cur_epi_len = 0
        self.last_state = next_state
        self.t += 1

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
