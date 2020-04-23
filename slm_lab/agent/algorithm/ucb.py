from slm_lab.agent.algorithm.base import Algorithm
from slm_lab.lib import logger
from slm_lab.lib.decorator import lab_api
import numpy as np

logger = logger.get_logger(__name__)


# Modified by https://github.com/bgalbraith/bandits/blob/master/bandits/policy.py
class UCB1(Algorithm):
    """
    The Upper Confidence Bound algorithm (UCB1). It applies an exploration
    factor to the expected value of each arm which can influence a greedy
    selection strategy to more intelligently explore less confident options.
    """

    def __init__(self, agent, global_nets=None, algorithm_spec=None, memory_spec=None, net_spec=None, algo_idx=0):
        super().__init__(agent, global_nets, algorithm_spec, memory_spec, net_spec, algo_idx)

        self.c = 1
        self.k_choice = algorithm_spec["k_choice"]
        self._tot_n_attempts = 0
        # TODO infer the dimension
        self._action_attempts = np.zeros(self.k_choice)
        self._value_estimates = np.zeros(self.k_choice)
        self.last_action = None

    @lab_api
    def act(self, state):
        exploration = np.log(self._tot_n_attempts + 1) / self._action_attempts
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1 / self.c)
        for idx, (v_e,expl) in enumerate(zip(self._value_estimates.tolist(), exploration.tolist())):
            print("UCB1 idx", idx)
            self.to_log[f"UCB1_{idx}_mean"] = v_e
            self.to_log[f"UCB1_{idx}_expl"] = expl

        q = self._value_estimates + exploration
        action = np.argmax(q)
        check = np.where(q == q[action])[0]
        if len(check) != 1:  # If several chooses have the save UCB1 index
            action = np.random.choice(check)

        self.last_action = action
        # return action, action_pd
        return action, None

    @lab_api
    def sample(self):
        pass

    def memory_update(self, state, action, welfare, next_state, done):
        self._tot_n_attempts += 1
        self._action_attempts[self.last_action] += 1
        # Update mean of _value_estimates
        self._value_estimates[self.last_action] += ((1 / self._action_attempts[self.last_action]) *
                                                    (welfare - self._value_estimates[self.last_action]))

        return None

    @lab_api
    def train(self):
        loss = 0.0
        return loss

    @lab_api
    def update(self):
        explore_var = 0.0
        return explore_var

    @lab_api
    def init_algorithm_params(self):
        pass

    @lab_api
    def init_nets(self, global_nets=None):
        pass
