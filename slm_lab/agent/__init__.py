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


class AgentSpace:
    agents = []
    A_EB_space = {}

    def __init__(self, spec):
        for agent_spec in spec['agent']:
            agent = Agent(agent_spec)
            self.add(agent)

    def add(self, agent):
        self.agents.append(agent)
        return self.agents

    def set_env_space(self, env_space):
        '''Make env_space visible to agent_space.'''
        self.env_space = env_space
        self.agents[0].set_env(env_space.envs[0])

    def add_body(self, body):
        # TODO add to A_EB_space
        # TODO set reference to envs, add_env(env), or not, just use AEB
        return

    def reset(self):
        return self.agents[0].reset()

    def act(self, state):
        # TODO tmp
        return self.agents[0].act(state)

    def update(self, reward, state):
        # TODO tmp
        return self.agents[0].update(reward, state)

    def close(self):
        for agent in self.agents:
            agent.close()
