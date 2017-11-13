'''
The agent module
Contains graduated components from experiments for building agents
and be taught, tested, evaluated on curriculum.
To be designed by human and evolution module,
based on the experiment aim (trait) and fitness metrics.
Main SLM components (refer to SLM doc for more):
- primary survival objective
- control policies
- sensors (input) for embodiment
- motors (output) for embodiment
- neural architecture
- memory (with time)
- prioritization mechanism and "emotions"
- strange loop must be created
- social aspect
- high level properties of thinking, e.g. creativity, planning.

Agent components:
- algorithms
- memory
- net
- policy
'''

import numpy as np
from abc import ABC, abstractmethod, abstractproperty
from slm_lab.lib import util


class Agent(ABC):
    # eventually, wanna init with string name of components
    # but dont wanna be too restrictive
    # start with a random agent first, with default components
    # tie into a trial of experiment (every run is a trial)
    env = None

    # @abstractproperty
    # def property_to_be_implemented(self):
    #     return 'property_to_be_implemented'

    # @abstractmethod
    # def method_to_be_implemented(self):
    #     pass

    @abstractmethod
    def __init__(self):
        pass

    def set_env(self, env):
        '''
        Make env visible to agent.
        TODO anticipate multi-environments for AEB space
        '''
        self.env = env

    def reset(self):
        return

    @abstractmethod
    def act(self):
        return

    def update(self):
        return


class Random(Agent):
    # TODO need proper space resolution for multi-env
    env = None

    def __init__(self, spec):
        # TODO also spec needs to specify AEB space and bodies
        util.set_attr(self, spec)

    def set_env(self, env):
        '''
        Make env visible to agent.
        TODO make consistent with AEB space
        '''
        self.env = env
        # TODO do body num here
        # analogously, do other dim counts as needed from env
        self.body_num = 1
        # TODO delegate a copy of variable like action_dim to agent too

    def reset(self):
        return

    def act(self, state):
        if self.env.is_discrete():
            # get environment action dim
            action = np.random.randint(
                0, self.env.get_action_dim(), size=(self.body_num))
        else:
            action = np.random.randn(
                self.body_num, self.env.get_action_dim())
        return action

    def update(self, reward, state):
        return

    def close(self):
        # TODO save or smth
        return
