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
from collections import deque
from slm_lab.agent import algorithm, memory
from slm_lab.agent.algorithm import policy_util
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

    def __init__(self, env, agent_spec, aeb=(0, 0, 0)):
        # TODO move this to under env for interface with agent. maybe like create_env_body
        # essential reference variables
        self.env = env
        self.aeb = aeb
        self.a, self.e, self.b = aeb
        # the specific agent-env interface variables for a body
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.observable_dim = self.env.observable_dim
        self.state_dim = self.observable_dim['state']
        self.action_dim = self.env.action_dim
        self.is_discrete = self.env.is_discrete
        self.action_type = util.get_action_type(self.action_space)
        self.action_pdtype = agent_spec['algorithm'].get('action_pdtype')
        if self.action_pdtype in (None, 'default'):
            self.action_pdtype = policy_util.ACTION_PDS[self.action_type][0]

        self.loss = np.nan  # training losses
        # for action policy exploration, so be set in algo during init_algorithm_params()
        self.explore_var = np.nan

        # diagnostics variables from action_policy prob. dist.
        self.entropies = []  # check exploration
        self.log_probs = []  # calculate loss
        self.kls = []  # to compare old and new nets

        # TODO move out
        # # every body has its own memory for ease of computation
        # memory_spec = self.agent.agent_spec['memory']
        # memory_name = memory_spec['name']
        # MemoryClass = getattr(memory, memory_name)
        # self.memory = MemoryClass(memory_spec, self.agent.algorithm, self)

    def epi_reset(self):
        '''
        Handles any body attribute reset at the start of an episode.
        This method is called automatically at base memory.epi_reset().
        '''
        assert self.env.clock.get('t') == 0, self.env.clock.get('t')

    def __str__(self):
        return 'body: ' + util.to_json(util.get_class_attr(self))

    def log_summary(self):
        '''Log the summary for this body when its environment is done'''
        spec = self.agent.spec
        info_space = self.agent.info_space
        clock = self.env.clock
        memory = self.memory
        msg = f'{spec["name"]} trial {info_space.get("trial")} session {info_space.get("session")} env {self.env.e}, body {self.aeb}, epi {clock.get("epi")}, t {clock.get("t")}, loss: {self.loss:.4f}, total_reward: {memory.total_reward:.4f}, last-{memory.avg_window}-epi avg: {memory.avg_total_reward:.4f}'
        logger.info(msg)


class Agent:
    '''
    Class for all Agents.
    Standardizes the Agent design to work in Lab.
    Access Envs properties by: Agents - AgentSpace - AEBSpace - EnvSpace - Envs
    '''

    # def __init__(self, agent_spec, agent_space, a=0):
    def __init__(self, spec, info_space, body, global_nets=None):
        self.spec = spec
        self.info_space = info_space
        self.body = body
        body.agent = self
        self.a = 0  # for compatibility with agent_space
        self.agent_spec = spec['agent']
        self.name = self.agent_spec['name']

        AlgorithmClass = getattr(algorithm, ps.get(self.agent_spec, 'algorithm.name'))
        self.algorithm = AlgorithmClass(self, global_nets)

        MemoryClass = getattr(memory, ps.get(self.agent_spec, 'memory.name'))
        self.body.memory = MemoryClass(self.agent_spec['memory'], self.algorithm, self.body)

        logger.info(util.self_desc(self))

    @lab_api
    def reset(self, state):
        '''Do agent reset per session, such as memory pointer'''
        logger.debug(f'Agent {self.a} reset')
        self.body.memory.epi_reset(state)

    @lab_api
    def act(self, state):
        '''Standard act method from algorithm.'''
        action = self.algorithm.act(state)
        logger.debug(f'Agent {self.a} act: {action}')
        return action

    @lab_api
    def update(self, action, reward, state, done):
        '''
        Update per timestep after env transitions, e.g. memory, algorithm, update agent params, train net
        '''
        self.body.memory.update(action, reward, state, done)
        loss = self.algorithm.train()
        self.body.loss = loss
        explore_var = self.algorithm.update()
        logger.debug(f'Agent {self.a} loss: {loss}, explore_var {explore_var}')
        return loss, explore_var

    @lab_api
    def save(self):
        '''Save agent'''
        self.algorithm.save()

    @lab_api
    def close(self):
        '''Close agent at the end of a session, e.g. save model'''
        self.save()


class SpaceBody:
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
        self.action_type = util.get_action_type(self.action_space)
        self.action_pdtype = self.agent.agent_spec['algorithm'].get('action_pdtype')
        if self.action_pdtype in (None, 'default'):
            self.action_pdtype = policy_util.ACTION_PDS[self.action_type][0]
        self.is_discrete = self.env.is_discrete(self.a)

        self.loss = np.nan  # training losses
        # for action policy exploration, so be set in algo during init_algorithm_params()
        self.explore_var = np.nan

        # diagnostics variables from action_policy prob. dist.
        self.entropies = []  # check exploration
        self.log_probs = []  # calculate loss
        self.kls = []  # to compare old and new nets

        # every body has its own memory for ease of computation
        memory_spec = self.agent.agent_spec['memory']
        memory_name = memory_spec['name']
        MemoryClass = getattr(memory, memory_name)
        self.memory = MemoryClass(memory_spec, self.agent.algorithm, self)

    def epi_reset(self):
        '''
        Agent will still produce action (and the action stat) at terminal step and feed to env.step() at t=0. Remove them at epi_reset.
        This method is called automatically at base memory.epi_reset().
        '''
        assert self.env.clock.get('t') == 0, self.env.clock.get('t')
        action_stats = [self.entropies, self.log_probs, self.kls]
        for action_stat in action_stats:
            # use pop instead of clear for cross-epi training
            if len(action_stat) > 0:
                action_stat.pop()

    def __str__(self):
        return 'body: ' + util.to_json(util.get_class_attr(self))


class SpaceAgent:
    '''
    Class for all Agents.
    Standardizes the Agent design to work in Lab.
    Access Envs properties by: Agents - AgentSpace - AEBSpace - EnvSpace - Envs
    '''

    def __init__(self, agent_spec, agent_space, a=0, global_nets=None):
        self.agent_spec = agent_spec
        self.agent_space = agent_space
        self.a = a
        self.spec = agent_space.spec
        self.info_space = agent_space.info_space
        self.name = self.agent_spec['name']
        self.body_a = None
        self.nanflat_body_a = None  # nanflatten version of bodies
        self.body_num = None

        AlgorithmClass = getattr(algorithm, ps.get(self.agent_spec, 'algorithm.name'))
        self.algorithm = AlgorithmClass(self, global_nets)

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
        logger.debug(f'Agent {self.a} reset')
        for (e, b), body in util.ndenumerate_nonan(self.body_a):
            body.memory.epi_reset(state_a[(e, b)])

    @lab_api
    def act(self, state_a):
        '''Standard act method from algorithm.'''
        action_a = self.algorithm.act(state_a)
        logger.debug(f'Agent {self.a} act: {action_a}')
        return action_a

    @lab_api
    def update(self, action_a, reward_a, state_a, done_a):
        '''
        Update per timestep after env transitions, e.g. memory, algorithm, update agent params, train net
        '''
        for (e, b), body in util.ndenumerate_nonan(self.body_a):
            body.memory.update(action_a[(e, b)], reward_a[(e, b)], state_a[(e, b)], done_a[(e, b)])
        loss_a = self.algorithm.train()
        loss_a = util.guard_data_a(self, loss_a, 'loss')
        for (e, b), body in util.ndenumerate_nonan(self.body_a):
            body.loss = loss_a[(e, b)]
        explore_var_a = self.algorithm.update()
        explore_var_a = util.guard_data_a(self, explore_var_a, 'explore_var')
        logger.debug(f'Agent {self.a} loss: {loss_a}, explore_var_a {explore_var_a}')
        return loss_a, explore_var_a

    @lab_api
    def save(self):
        '''Save agent'''
        self.algorithm.save()

    @lab_api
    def close(self):
        '''Close agent at the end of a session, e.g. save model'''
        self.save()


class AgentSpace:
    '''
    Subspace of AEBSpace, collection of all agents, with interface to Session logic; same methods as singleton agents.
    Access EnvSpace properties by: AgentSpace - AEBSpace - EnvSpace - Envs
    '''

    def __init__(self, spec, aeb_space, global_nets=None):
        self.spec = spec
        self.aeb_space = aeb_space
        aeb_space.agent_space = self
        self.agent_spec = spec['agent']
        self.info_space = aeb_space.info_space
        self.aeb_shape = aeb_space.aeb_shape
        self.agents = [Agent(agent_spec, self, a, global_nets) for a, agent_spec in enumerate(self.agent_spec)]

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
        logger.debug3('AgentSpace.reset')
        _action_v, _loss_v, _explore_var_v = self.aeb_space.init_data_v(AGENT_DATA_NAMES)
        for agent in self.agents:
            state_a = state_space.get(a=agent.a)
            agent.reset(state_a)
        _action_space, _loss_space, _explore_var_space = self.aeb_space.add(AGENT_DATA_NAMES, [_action_v, _loss_v, _explore_var_v])
        logger.debug3(f'action_space: {_action_space}')
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
        logger.debug3(f'\naction_space: {action_space}')
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
        logger.debug3(f'\nloss_space: {loss_space}\nexplore_var_space: {explore_var_space}')
        return loss_space, explore_var_space

    @lab_api
    def close(self):
        logger.info('AgentSpace.close')
        for agent in self.agents:
            agent.close()
