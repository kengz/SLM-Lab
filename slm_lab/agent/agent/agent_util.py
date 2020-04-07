from slm_lab.lib import logger
# import numpy as np
# import torch
import copy
import pydash as ps

logger = logger.get_logger(__name__)

def get_from_current_agents(agent, key, default=None):
    other_agents_rewards = copy.deepcopy(ps.get(agent.world.shared_dict, key, default))
    return other_agents_rewards[agent.agent_idx]

# TODO write tests for get_from_other_agents and (especially on the use of the OrderedDict)
def get_from_other_agents(agent, key, default):
    values = []
    for k, observed_agent_dict in agent.other_ag_observations.items():
        values.append(ps.get(observed_agent_dict, key, default))
        # print("values", values)
    return values

# def remove_current_agent_idx(agent, list_value):
#     if len(list_value) > agent.agent_idx:
#         list_value.pop(agent.agent_idx)
#     return list_value


##### Welfare functions ######

def default_welfare(agent, current_agent_reward):
    welfare = current_agent_reward
    return welfare


def utilitarian_welfare(agent, current_agent_reward):
    # print("utilitarian_welfare")
    other_agents_rewards = get_from_other_agents(agent, key='reward', default=[])

    welfare = current_agent_reward + sum(other_agents_rewards)
    # print(agent.agent_idx,"current_agent_reward",current_agent_reward, "welfare", welfare,
    #       "sum(other_agents_rewards)", sum(other_agents_rewards))
    return welfare


def egalitarian_welfare(agent, current_agent_reward):
    current_agent_v_with_disagreement = get_from_current_agents(agent, 'agent_v_with_disagreement', default=0)
    other_agents_rewards = get_from_other_agents(agent, key='reward', default=[])
    other_agent_v_with_disagreement = get_from_other_agents(agent,
                          key='agent_v_with_disagreement',
                          default=[0] * len(other_agents_rewards)
                          )

    c_agent_v_over_disagreement = current_agent_reward - current_agent_v_with_disagreement
    other_agents_v_over_disagreement = [ag_v - ag_d_v for ag_v, ag_d_v in
                                        zip(other_agents_rewards,other_agent_v_with_disagreement) ]
    welfare = [c_agent_v_over_disagreement] + other_agents_v_over_disagreement
    welfare = min(welfare)

    return welfare


def nash_welfare(agent, current_agent_reward):
    current_agent_v_with_disagreement = get_from_current_agents(agent, 'agent_v_with_disagreement', default=0)
    other_agents_rewards = get_from_other_agents(agent, key='reward', default=[])
    other_agent_v_with_disagreement = get_from_other_agents(agent,
                          key='agent_v_with_disagreement',
                          default=[0] * len(other_agents_rewards)
                          )

    c_agent_v_over_disagreement = current_agent_reward - current_agent_v_with_disagreement
    other_agents_v_over_disagreement = [a_v - a_d_v for a_v, a_d_v in
                                        zip(other_agents_rewards, other_agent_v_with_disagreement)]

    welfare = c_agent_v_over_disagreement
    for v in other_agents_v_over_disagreement:
        welfare *= v

    return welfare