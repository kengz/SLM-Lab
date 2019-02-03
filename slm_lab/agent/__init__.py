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


class Agent:
    '''
    Class for all Agents.
    Standardizes the Agent design to work in Lab.
    Access Envs properties by: Agents - AgentSpace - AEBSpace - EnvSpace - Envs
    '''

    def __init__(self, spec, info_space, body, a=None, agent_space=None, global_nets=None):
        self.spec = spec
        self.info_space = info_space
        self.a = a or 0  # for compatibility with agent_space
        self.agent_spec = spec['agent'][self.a]
        self.name = self.agent_spec['name']
        assert not ps.is_list(global_nets), f'single agent global_nets must be a dict, got {global_nets}'
        if agent_space is None:  # singleton mode
            self.body = body
            body.agent = self
            MemoryClass = getattr(memory, ps.get(self.agent_spec, 'memory.name'))
            self.body.memory = MemoryClass(self.agent_spec['memory'], self.body)
            AlgorithmClass = getattr(algorithm, ps.get(self.agent_spec, 'algorithm.name'))
            self.algorithm = AlgorithmClass(self, global_nets)
        else:
            self.space_init(agent_space, body, global_nets)

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
        '''Update per timestep after env transitions, e.g. memory, algorithm, update agent params, train net'''
        self.body.action_pd_update()
        self.body.memory.update(action, reward, state, done)
        loss = self.algorithm.train()
        if not np.isnan(loss):  # set for log_summary()
            self.body.loss = loss
        explore_var = self.algorithm.update()
        logger.debug(f'Agent {self.a} loss: {loss}, explore_var {explore_var}')
        if done:
            self.body.epi_update()
        return loss, explore_var

    @lab_api
    def save(self, ckpt=None):
        '''Save agent'''
        if util.get_lab_mode() in ('enjoy', 'eval'):
            # eval does not save new models
            return
        self.algorithm.save(ckpt=ckpt)

    @lab_api
    def close(self):
        '''Close and cleanup agent at the end of a session, e.g. save model'''
        self.save()

    @lab_api
    def space_init(self, agent_space, body_a, global_nets):
        '''Post init override for space env. Note that aeb is already correct from __init__'''
        self.agent_space = agent_space
        self.body_a = body_a
        self.aeb_space = agent_space.aeb_space
        self.nanflat_body_a = util.nanflatten(self.body_a)
        for idx, body in enumerate(self.nanflat_body_a):
            if idx == 0:  # NOTE set default body
                self.body = body
            body.agent = self
            body.nanflat_a_idx = idx
            MemoryClass = getattr(memory, ps.get(self.agent_spec, 'memory.name'))
            body.memory = MemoryClass(self.agent_spec['memory'], body)
        self.body_num = len(self.nanflat_body_a)
        AlgorithmClass = getattr(algorithm, ps.get(self.agent_spec, 'algorithm.name'))
        self.algorithm = AlgorithmClass(self, global_nets)
        # after algo init, transfer any missing variables from default body
        for idx, body in enumerate(self.nanflat_body_a):
            for k, v in vars(self.body).items():
                if util.gen_isnan(getattr(body, k, None)):
                    setattr(body, k, v)

    @lab_api
    def space_reset(self, state_a):
        '''Do agent reset per session, such as memory pointer'''
        logger.debug(f'Agent {self.a} reset')
        for eb, body in util.ndenumerate_nonan(self.body_a):
            body.memory.epi_reset(state_a[eb])

    @lab_api
    def space_act(self, state_a):
        '''Standard act method from algorithm.'''
        action_a = self.algorithm.space_act(state_a)
        logger.debug(f'Agent {self.a} act: {action_a}')
        return action_a

    @lab_api
    def space_update(self, action_a, reward_a, state_a, done_a):
        '''Update per timestep after env transitions, e.g. memory, algorithm, update agent params, train net'''
        for eb, body in util.ndenumerate_nonan(self.body_a):
            body.action_pd_update()
            body.memory.update(action_a[eb], reward_a[eb], state_a[eb], done_a[eb])
        loss_a = self.algorithm.space_train()
        loss_a = util.guard_data_a(self, loss_a, 'loss')
        for eb, body in util.ndenumerate_nonan(self.body_a):
            if not np.isnan(loss_a[eb]):  # set for log_summary()
                body.loss = loss_a[eb]
        explore_var_a = self.algorithm.space_update()
        explore_var_a = util.guard_data_a(self, explore_var_a, 'explore_var')
        logger.debug(f'Agent {self.a} loss: {loss_a}, explore_var_a {explore_var_a}')
        for eb, body in util.ndenumerate_nonan(self.body_a):
            if body.env.done:
                body.epi_update()
        return loss_a, explore_var_a


class AgentSpace:
    '''
    Subspace of AEBSpace, collection of all agents, with interface to Session logic; same methods as singleton agents.
    Access EnvSpace properties by: AgentSpace - AEBSpace - EnvSpace - Envs
    '''

    def __init__(self, spec, aeb_space, global_nets=None):
        self.spec = spec
        self.aeb_space = aeb_space
        aeb_space.agent_space = self
        self.info_space = aeb_space.info_space
        self.aeb_shape = aeb_space.aeb_shape
        assert not ps.is_dict(global_nets), f'multi agent global_nets must be a list of dicts, got {global_nets}'
        assert ps.is_list(self.spec['agent'])
        self.agents = []
        for a in range(len(self.spec['agent'])):
            body_a = self.aeb_space.body_space.get(a=a)
            if global_nets is not None:
                agent_global_nets = global_nets[a]
            else:
                agent_global_nets = None
            agent = Agent(self.spec, self.info_space, body=body_a, a=a, agent_space=self, global_nets=agent_global_nets)
            self.agents.append(agent)
        logger.info(util.self_desc(self))

    def get(self, a):
        return self.agents[a]

    @lab_api
    def reset(self, state_space):
        logger.debug3('AgentSpace.reset')
        _action_v, _loss_v, _explore_var_v = self.aeb_space.init_data_v(AGENT_DATA_NAMES)
        for agent in self.agents:
            state_a = state_space.get(a=agent.a)
            agent.space_reset(state_a)
        _action_space, _loss_space, _explore_var_space = self.aeb_space.add(AGENT_DATA_NAMES, (_action_v, _loss_v, _explore_var_v))
        logger.debug3(f'action_space: {_action_space}')
        return _action_space

    @lab_api
    def act(self, state_space):
        data_names = ('action',)
        action_v, = self.aeb_space.init_data_v(data_names)
        for agent in self.agents:
            a = agent.a
            state_a = state_space.get(a=a)
            action_a = agent.space_act(state_a)
            action_v[a, 0:len(action_a)] = action_a
        action_space, = self.aeb_space.add(data_names, (action_v,))
        logger.debug3(f'\naction_space: {action_space}')
        return action_space

    @lab_api
    def update(self, action_space, reward_space, state_space, done_space):
        data_names = ('loss', 'explore_var')
        loss_v, explore_var_v = self.aeb_space.init_data_v(data_names)
        for agent in self.agents:
            a = agent.a
            action_a = action_space.get(a=a)
            reward_a = reward_space.get(a=a)
            state_a = state_space.get(a=a)
            done_a = done_space.get(a=a)
            loss_a, explore_var_a = agent.space_update(action_a, reward_a, state_a, done_a)
            loss_v[a, 0:len(loss_a)] = loss_a
            explore_var_v[a, 0:len(explore_var_a)] = explore_var_a
        loss_space, explore_var_space = self.aeb_space.add(data_names, (loss_v, explore_var_v))
        logger.debug3(f'\nloss_space: {loss_space}\nexplore_var_space: {explore_var_space}')
        return loss_space, explore_var_space

    @lab_api
    def close(self):
        logger.info('AgentSpace.close')
        for agent in self.agents:
            agent.close()
