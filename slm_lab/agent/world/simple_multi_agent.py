from slm_lab.lib import logger, util, viz
from slm_lab.lib.decorator import lab_api
import torch
from copy import deepcopy
from slm_lab.agent.agent import Agent
import numpy as np

logger = logger.get_logger(__name__)


class DefaultMultiAgentWorld:
    '''
    World abstraction; implements the API to interface with Env in SLM Lab
    Contains algorithm(s), memory(s), body(s), agent(s)

    world < env
        + welfare_function
        + agents < algorithm + memory + body

    '''

    def __init__(self, spec, env, global_nets_list=None):

        self.spec = spec
        self.agents = []
        self.best_total_rewards_ma = -np.inf
        for i, spec_agent in enumerate(deepcopy(spec['agent'])):
            a_spec = deepcopy(spec)
            a_spec.update({'agent': [spec_agent], 'name':a_spec['name'] + '-' + str(i)})
            self.agents.append(Agent(spec=a_spec , env=env, global_nets=global_nets_list[i] if global_nets_list is
                                                                                               not None else None))

    @property
    def bodies(self):
        return [a.body for a in self.agents]

    @property
    def total_rewards_ma(self):
        return sum([body.total_reward_ma for body in self.bodies])

    @property
    def algorithms(self):
        return [ a.algorithm for a in self.agents]

    @property
    def name(self):
        return str([ a.name for a in self.agents])

    @lab_api
    def act(self, state):
        '''Standard act method from algorithm.'''

        actions = []
        for agent, s in zip(self.agents, state):
            actions.append(agent.act(s))
        return actions

    @lab_api
    def update(self, state, action, reward, next_state, done):
        '''Update per timestep after env transitions, e.g. memory, algorithm, update agent params, train net'''


        # logger.info(f"state {state} {type(state)}")

        loss, explore_var = [], []
        assert len(self.agents) == len(state) == len(action) == len(reward) == len(next_state)
        for agent, s, a, r, n_s in zip(self.agents, state, action, reward, next_state):
            # logger.info(f"s {s} a {a} r {r}")
            if util.in_eval_lab_modes():
                l, e_v = agent.update(s, a, r, n_s, done)
                loss.append(l)
                # logger.info(f"loss {torch.cat(loss, dim=0).mean()}")
                explore_var.append(e_v)
            else:
                agent.update(s, a, r, n_s, done)

        if util.in_eval_lab_modes():
            loss = torch.cat(loss, dim=0).mean()
            # logger.info(f"loss {loss}")
            explore_var = torch.cat(explore_var, dim=0).mean()
            return loss, explore_var

    @lab_api
    def save(self, ckpt=None):
        '''Save agent'''
        for agent in self.agents:
            agent.save(ckpt=ckpt)
    @lab_api
    def close(self):
        '''Close and cleanup agent at the end of a session, e.g. save model'''
        for agent in self.agents:
            agent.close()

