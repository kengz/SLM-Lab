from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.lib.decorator import lab_api
import numpy as np

from slm_lab.lib import logger
logger = logger.get_logger(__name__)


class StateActionData():
    def __init__(self, name, n_possible_next_states):
        self.n_occurences = 0
        self.hash = name  # hash(hash(state.data) + hash(action.data))
        self.next_states = [[]] * n_possible_next_states

    def add_occurence(self, reward, next_state_idx):
        self.n_occurences += 1
        self.next_states[next_state_idx].append(float(reward))

    def get_transitions_proba(self):
        assert self.n_occurences > 0
        transitions_proba = [len(next_state) / self.n_occurences for next_state in self.next_states]
        return transitions_proba

    def get_mean_reward(self):
        assert self.n_occurences > 0
        mean_reward = sum([sum(next_state) for next_state in self.next_states]) / self.n_occurences
        return mean_reward

    def __str__(self):
        return self.hash

class MeanPastTransitionsAndRewards(Algorithm):
    # TODO add documentation

    @lab_api
    def init_algorithm_params(self):
        '''Initialize other algorithm parameters'''
        assert self.agent.body.env.action_space_is_discrete
        assert self.agent.body.env.observation_space_is_discrete

        self.state_dim = self.agent.body.observation_dim
        self.action_dim = self.agent.body.action_dim
        self.n_possible_states_actions = 1
        for n in self.state_dim:
            self.n_possible_states_actions *= n
        for n in self.action_dim:
            self.n_possible_states_actions *= n

        self.state_action_history = {}


    @lab_api
    def init_nets(self, global_nets=None):
        pass

    @lab_api
    def proba_distrib_params(self, x, net=None):
        raise NotImplementedError()


    @lab_api
    def act(self, state):
        raise NotImplementedError()

    def transitions_and_reward(self, state, action):
        state_action_name = hash( hash(state.data) + hash(action.data) )
        transitions = self.state_action_history[state_action_name].get_transitions_proba()
        reward = self.state_action_history[state_action_name].get_mean_reward()
        return transitions, reward

    @lab_api
    def train(self):
        raise NotImplementedError()

    @lab_api
    def update(self):
        raise NotImplementedError()


    def memory_update(self, state, action, reward, next_state, done):
        # TODO Support vectorized env += n vectorized, done ?
        self.internal_clock.tick(unit='t')

        next_state = np.asarray(next_state)
        state_action_name = hash( hash(state.data) + hash(action.data) )

        if self.state_dim == (1,):
            assert sum(next_state) == 1
            next_state_idx = next_state.argmax()
        else:
            next_state_idx = 1
            argmax = next_state.argmax(axis=-1)
            for n in argmax:
                next_state_idx *= n

        if state_action_name not in self.state_action_history.keys():
            self.state_action_history[state_action_name] = StateActionData(state_action_name, self.n_possible_states_actions)
        self.state_action_history[state_action_name].add_occurence(reward, next_state_idx)

        return None