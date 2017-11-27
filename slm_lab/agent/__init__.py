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
from slm_lab.agent import algorithm
from slm_lab.experiment.monitor import data_space
from slm_lab.lib import util


class Agent:
    '''
    Class for all Agents.
    Standardizes the Agent design to work in Lab.
    '''
    # TODO ok need architecture spec for each agent: disjoint or joint, time or space multiplicity

    def __init__(self, multi_spec):
        self.coor, self.index, self.spec = data_space.init_lab_comp(
            self, multi_spec)
        util.set_attr(self, self.spec)

        AlgoClass = getattr(algorithm, self.name)
        self.algorithm = AlgoClass(self)
        # TODO tmp use Body in the space
        # TODO also resolve architecture and data input, output dims via some architecture spec
        self.memory = None
        self.net = None
        # TODO tmp
        self.env = None
        self.body_num = 1
        # TODO delegate a copy of variable like action_dim to agent too

    def set_env(self, env):
        '''Make env visible to agent.'''
        # TODO AEB space resolver pending, needs to be powerful enuf to for auto-architecture, action space, body num resolution, other dim counts from env
        self.env = env

    def reset(self):
        '''Do agent reset per episode, such as memory pointer'''
        # TODO implement
        return

    def act(self, state):
        '''Standard act method from algorithm.'''
        # TODO tmp make act across bodies, work on AEB
        return [self.algorithm.act(state)]

    def update(self, reward, state, done):
        '''
        Update per timestep after env transitions, e.g. memory, algorithm, update agent params, train net
        '''
        # TODO build and access timestep, episode, absolute number of timesteps from Dataspace
        # TODO implement generic method, work on AEB
        # self.memory.update()
        # self.net.train()
        self.algorithm.update(reward, state, done)
        return

    def close(self):
        '''Close agent at the end of a session, e.g. save model'''
        # TODO save model
        model_for_loading_next_trial = 'not implemented'
        return model_for_loading_next_trial


class AgentSpace:
    def __init__(self, spec):
        self.aeb_space = None
        self.agents = []
        for agent_spec in spec['agent']:
            agent = Agent(agent_spec)
            self.agents.append(agent)

    def set_space_ref(self, aeb_space):
        '''Make super aeb_space visible to agent_space.'''
        self.aeb_space = aeb_space
        # TODO tmp, resolve later from AEB
        env_space = aeb_space.env_space
        self.agents[0].set_env(env_space.envs[0])

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def act(self, state_space):
        action_proj = []
        for a, agent in enumerate(self.agents):
            state = state_space.get(a=a)
            action = agent.act(state)
            action_proj.append(action)
        action_space = self.aeb_space.add('action', action_proj)
        return action_space

    def update(self, reward_space, state_space, done_space):
        for a, agent in enumerate(self.agents):
            reward = reward_space.get(a=a)
            state = state_space.get(a=a)
            done = done_space.get(a=a)
            agent.update(reward, state, done)

    def close(self):
        for agent in self.agents:
            agent.close()
