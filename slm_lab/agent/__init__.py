# the agent module
from slm_lab.agent import algorithm, memory
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import pydash as ps
import torch

logger = logger.get_logger(__name__)


class Agent:
    '''
    Agent abstraction; implements the API to interface with Env in SLM Lab
    Contains algorithm, memory, body
    '''

    def __init__(self, spec, body, a=None, global_nets=None):
        self.spec = spec
        self.a = a or 0  # for multi-agent
        self.agent_spec = spec['agent'][self.a]
        self.name = self.agent_spec['name']
        assert not ps.is_list(global_nets), f'single agent global_nets must be a dict, got {global_nets}'
        # set components
        self.body = body
        body.agent = self
        MemoryClass = getattr(memory, ps.get(self.agent_spec, 'memory.name'))
        self.body.memory = MemoryClass(self.agent_spec['memory'], self.body)
        AlgorithmClass = getattr(algorithm, ps.get(self.agent_spec, 'algorithm.name'))
        self.algorithm = AlgorithmClass(self, global_nets)

        logger.info(util.self_desc(self))

    @lab_api
    def act(self, state):
        '''Standard act method from algorithm.'''
        with torch.no_grad():  # for efficiency, only calc grad in algorithm.train
            action = self.algorithm.act(state)
        return action

    @lab_api
    def update(self, state, action, reward, next_state, done):
        '''Update per timestep after env transitions, e.g. memory, algorithm, update agent params, train net'''
        self.body.update(state, action, reward, next_state, done)
        if util.in_eval_lab_modes():  # eval does not update agent for training
            return
        self.body.memory.update(state, action, reward, next_state, done)
        loss = self.algorithm.train()
        if not np.isnan(loss):  # set for log_summary()
            self.body.loss = loss
        explore_var = self.algorithm.update()
        return loss, explore_var

    @lab_api
    def save(self, ckpt=None):
        '''Save agent'''
        if util.in_eval_lab_modes():  # eval does not save new models
            return
        self.algorithm.save(ckpt=ckpt)

    @lab_api
    def close(self):
        '''Close and cleanup agent at the end of a session, e.g. save model'''
        self.save()
