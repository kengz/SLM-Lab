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
- algorithm (with net, policy)
- memory (per body)
'''
from slm_lab.agent import algorithm, memory
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import pydash as ps

AGENT_DATA_NAMES = ['action', 'loss', 'explore_var']
logger = logger.get_logger(__name__)


class Body:
    '''
    Body is the handler with proper info to reference the single-agent-single-environment unit in this generalized multi-agent-env setting.
    '''

    def __init__(self, aeb, agent, env):
        # essential reference variables
        self.aeb = aeb
        self.agent = agent
        self.env = env
        self.a, self.e, self.b = aeb
        self.nanflat_a_idx = None
        self.nanflat_e_idx = None

        # TODO generalize and make state_space to include observables
        # the specific agent-env interface variables for a body
        self.observable_dim = self.env.get_observable_dim(self.a)
        self.state_dim = self.observable_dim['state']
        self.observation_space = self.env.get_observation_space(self.a)
        self.action_dim = self.env.get_action_dim(self.a)
        self.action_space = self.env.get_action_space(self.a)
        self.is_discrete = self.env.is_discrete(self.a)

        # every body has its own memory for ease of computation
        MemoryClass = getattr(memory, ps.get(self.agent.spec, 'memory.name'))
        self.memory = MemoryClass(self)
        self.state_buffer = []

    def __str__(self):
        return 'body: ' + util.to_json(util.get_class_attr(self))


class Agent:
    '''
    Class for all Agents.
    Standardizes the Agent design to work in Lab.
    Access Envs properties by: Agents - AgentSpace - AEBSpace - EnvSpace - Envs
    '''

    def __init__(self, spec, agent_space, a=0):
        self.spec = spec
        self.agent_space = agent_space
        self.a = a
        self.name = self.spec['name']
        self.body_a = None
        self.nanflat_body_a = None  # nanflatten version of bodies
        self.body_num = None

        AlgoClass = getattr(algorithm, ps.get(self.spec, 'algorithm.name'))
        self.algorithm = AlgoClass(self)

        # TODO uhh handle internally to memory?
        memory_name = spec['memory']['name']
        if 'NStep' in memory_name or 'Stack' in memory_name:
            self.len_state_buffer = spec['memory']['length_history']
        elif 'Atari' in memory_name:
            self.len_state_buffer = 4
        else:
            self.len_state_buffer = 0

    @lab_api
    def post_body_init(self):
        '''Run init for components that need bodies to exist first, e.g. memory or architecture.'''
        self.nanflat_body_a = util.nanflatten(self.body_a)
        for idx, body in enumerate(self.nanflat_body_a):
            body.nanflat_a_idx = idx
        self.body_num = len(self.nanflat_body_a)
        self.algorithm.post_body_init()
        logger.info(util.self_desc(self))

    @lab_api
    def reset(self, state_a):
        '''Do agent reset per session, such as memory pointer'''
        for (e, b), body in util.ndenumerate_nonan(self.body_a):
            body.memory.reset_last_state(state_a[(e, b)])

    @lab_api
    def act(self, state_a):
        '''Standard act method from algorithm.'''
        action_a = self.algorithm.act(state_a)
        return action_a

    @lab_api
    def update(self, action_a, reward_a, state_a, done_a):
        '''
        Update per timestep after env transitions, e.g. memory, algorithm, update agent params, train net
        '''
        for (e, b), body in util.ndenumerate_nonan(self.body_a):
            body.memory.update(action_a[(e, b)], reward_a[(e, b)], state_a[(e, b)], done_a[(e, b)])
            # TODO also internalize to memory
            if self.len_state_buffer > 0:
                if len(body.state_buffer) == self.len_state_buffer:
                    del body.state_buffer[0]
                body.state_buffer.append(state_a[(e, b)])
        loss_a = self.algorithm.train()
        loss_a = util.guard_data_a(self, loss_a, 'loss')
        explore_var_a = self.algorithm.update()
        explore_var_a = util.guard_data_a(self, explore_var_a, 'explore_var')
        return loss_a, explore_var_a

    @lab_api
    def close(self):
        '''Close agent at the end of a session, e.g. save model'''
        # TODO save model
        model_for_loading_next_trial = 'not implemented'
        return model_for_loading_next_trial


class AgentSpace:
    '''
    Subspace of AEBSpace, collection of all agents, with interface to Session logic; same methods as singleton agents.
    Access EnvSpace properties by: AgentSpace - AEBSpace - EnvSpace - Envs
    '''

    def __init__(self, spec, aeb_space):
        self.spec = spec
        self.aeb_space = aeb_space
        self.agent_spec = spec['agent']
        self.aeb_shape = aeb_space.aeb_shape
        aeb_space.agent_space = self
        self.agents = [Agent(agent_spec, self, a) for a, agent_spec in enumerate(self.agent_spec)]

    @lab_api
    def post_body_init(self):
        '''Run init for components that need bodies to exist first, e.g. memory or architecture.'''
        for agent in self.agents:
            agent.post_body_init()
        logger.info(util.self_desc(self))

    def get(self, a):
        return self.agents[a]

    @lab_api
    def reset(self, state_space):
        logger.debug('AgentSpace.reset')
        _action_v, _loss_v, _explore_var_v = self.aeb_space.init_data_v(AGENT_DATA_NAMES)
        for agent in self.agents:
            state_a = state_space.get(a=agent.a)
            agent.reset(state_a)
        _action_space, _loss_space, _explore_var_space = self.aeb_space.add(AGENT_DATA_NAMES, [_action_v, _loss_v, _explore_var_v])
        return _action_space

    @lab_api
    def act(self, state_space):
        data_names = ['action']
        action_v, = self.aeb_space.init_data_v(data_names)
        for agent in self.agents:
            a = agent.a
            state_a = state_space.get(a=a)
            action_a = agent.act(state_a)
            action_v[a, 0:len(action_a)] = action_a
        action_space, = self.aeb_space.add(data_names, [action_v])
        logger.debug(f'\naction_space: {action_space}')
        return action_space

    @lab_api
    def update(self, action_space, reward_space, state_space, done_space):
        data_names = ['loss', 'explore_var']
        loss_v, explore_var_v = self.aeb_space.init_data_v(data_names)
        for agent in self.agents:
            a = agent.a
            action_a = action_space.get(a=a)
            reward_a = reward_space.get(a=a)
            state_a = state_space.get(a=a)
            done_a = done_space.get(a=a)
            loss_a, explore_var_a = agent.update(action_a, reward_a, state_a, done_a)
            loss_v[a, 0:len(loss_a)] = loss_a
            explore_var_v[a, 0:len(explore_var_a)] = explore_var_a
        loss_space, explore_var_space = self.aeb_space.add(data_names, [loss_v, explore_var_v])
        logger.debug(f'\nloss_space: {loss_space}\nexplore_var_space: {explore_var_space}')
        return loss_space, explore_var_space

    @lab_api
    def close(self):
        logger.info('AgentSpace.close')
        for agent in self.agents:
            agent.close()
