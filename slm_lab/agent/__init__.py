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
    spec = None
    # TODO only reference via body
    # TODO ok need architecture spec for each agent: disjoint or joint, time or space multiplicity
    EB_space = []
    # active_eb_coor = None
    # TODO then for high level call, update active_eb_coor (switch focus body), call atomic methods
    # what if ur atomic methods wants to be batched, need to specify resolver than, will do batch method instead of atomic, and auto collect all the EB space data for this agent
    # consider EB resolver and collector with net, action
    # EB_space = {
    #     0: [b0, b1,  b2],
    # }
    env = None
    algorithm = None
    memory = None
    net = None

    def __init__(self, multi_spec):
        # TODO also spec needs to specify AEB space and bodies
        data_space.init_lab_comp_coor(self, multi_spec)
        util.set_attr(self, self.spec)

        AlgoClass = getattr(algorithm, self.name)
        self.algorithm = AlgoClass(self)
        # TODO tmp use Body in the space
        # TODO also resolve architecture and data input, output dims via some architecture spec
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
        # TODO tmp make act across bodies
        return [self.algorithm.act(state)]

    def update(self, reward, state, done):
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


class AgentSpace:
    # TODO rename method args to space
    aeb_space = None
    agents = []

    def __init__(self, spec):
        for agent_spec in spec['agent']:
            agent = Agent(agent_spec)
            self.add(agent)

    def add(self, agent):
        self.agents.append(agent)
        return self.agents

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
        # return self.agents[0].act(state)
        # TODO use DataSpace class, with np array
        action_proj = []
        for a, agent in enumerate(self.agents):
            state = state_space.get(a=a)
            action = agent.act(state)
            action_proj.append(action)
        action_space = self.aeb_space.add('action', action_proj)
        return action_space

    def update(self, reward_space, state_space, done_space):
        # resolve data_space by AEB method again, spread
        # return self.agents[0].update(reward, state, done)
        # TODO use DataSpace class, with np array
        for a, agent in enumerate(self.agents):
            reward = reward_space.get(a=a)
            state = state_space.get(a=a)
            done = done_space.get(a=a)
            agent.update(reward, state, done)

    def close(self):
        for agent in self.agents:
            agent.close()
