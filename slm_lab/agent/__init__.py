'''
The agent module
Contains graduated components from experiments for building agents and be taught, tested, evaluated on curriculum.
To be designed by human and evolution module, based on the experiment aim (trait) and fitness metrics.
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
- algorithm
- memory
- net
- policy
'''
# TODO need a mechanism that compose the components together using spec
from slm_lab.agent import algorithm
from slm_lab.experiment.monitor import data_space
from slm_lab.lib import util


class Agent:
    '''
    Class for all Agents.
    Standardizes the Agent design to work in Lab.
    '''
    env = None
    algorithm = None
    memory = None
    net = None

    def __init__(self, spec):
        # TODO also spec needs to specify AEB space and bodies
        data_space.init_lab_comp_coor(self, spec)

        AlgoClass = getattr(algorithm, self.name)
        self.algorithm = AlgoClass(self)

    def set_env(self, env):
        '''Make env visible to agent.'''
        # TODO AEB space resolver pending, needs to be powerful enuf to for auto-architecture, action space, body num resolution, other dim counts from env
        self.env = env
        self.body_num = 1
        # TODO delegate a copy of variable like action_dim to agent too

    def reset(self):
        '''Do agent reset per episode, such as memory pointer'''
        # TODO implement
        return

    def act(self, state):
        '''Standard act method from algorithm.'''
        return self.algorithm.act(state)

    def update(self, reward, state):
        '''
        Update per timestep after env transitions, e.g. memory, algorithm, update agent params, train net
        '''
        # TODO implement generic method
        # self.memory.update()
        # self.net.train()
        self.algorithm.update()
        return

    def close(self):
        '''Close agent at the end of a session, e.g. save model'''
        # TODO save model
        model_for_loading_next_trial = 'not implemented'
        return model_for_loading_next_trial
