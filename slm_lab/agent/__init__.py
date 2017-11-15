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
    '''
    Abstract class ancestor to all Agents,
    specifies the necessary design blueprint for agent to work in Lab.
    Mostly, implement just the abstract methods and properties.
    '''
    env = None
    memory = None
    net = None

    def __init__(self, spec):
        # TODO also spec needs to specify AEB space and bodies
        util.set_attr(self, spec)

    def set_env(self, env):
        '''Make env visible to agent.'''
        # TODO make consistent with AEB space
        self.env = env
        # TODO do body num here
        # analogously, do other dim counts as needed from env
        self.body_num = 1
        # TODO delegate a copy of variable like action_dim to agent too

    def reset(self):
        '''Do agent reset per episode, such as memory pointer'''
        # TODO implement
        return

    @abstractmethod
    def act_discrete(self, state):
        '''Implement discrete action, or throw NotImplementedError'''
        # TODO auto AEB space resolver from atomic method
        raise NotImplementedError
        return action

    @abstractmethod
    def act_continuous(self, state):
        '''Implement continuous action, or throw NotImplementedError'''
        # TODO auto AEB space resolver from atomic method
        raise NotImplementedError
        return action

    def act(self, state):
        '''Standard act method. Actions should be implemented in submethods'''
        # TODO right now, act shd be batched in the submethods and assumes all bodies are in the same env type: discrete or continuous. Things might change in the future, then act is just generic act.
        if self.env.is_discrete():
            return self.act_discrete(state)
        else:
            return self.act_continuous(state)

    def update_param(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def update(self, reward, state):
        '''
        Update per timestep after env transitions,
        e.g. memory, policy, update agent params, train net
        '''
        # TODO implement
        # self.memory.update()
        # self.policy.update()
        self.update_param()
        self.train()
        return

    def close(self):
        '''Close agent at the end of a session, e.g. save model'''
        # TODO save model
        model_for_loading_next_trial = 'not implemented'
        return model_for_loading_next_trial


class Random(Agent):
    '''
    Example Random agent that works in both discrete and continuous envs
    '''
    env = None

    def act_discrete(self, state):
        '''Random discrete action'''
        # TODO AEB space resolver, lineup index
        action = np.random.randint(
            0, self.env.get_action_dim(), size=(self.body_num))
        return action

    def act_continuous(self, state):
        '''Random continuous action'''
        # TODO AEB space resolver, lineup index
        action = np.random.randn(
            self.body_num, self.env.get_action_dim())
        return action

    def update_param(self):
        return

    def train(self):
        return
