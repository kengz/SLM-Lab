# from slm_lab.lib import logger
# import numpy as np
# import torch
#
# logger = logger.get_logger(__name__)
#
#
# def default_welfare(current_agent_reward, other_agents_rewards, **kwargs):
#     welfare = current_agent_reward
#
#     return welfare
#
# def utilitarian_welfare(current_agent_reward, other_agents_rewards, **kwargs):
#     welfare = current_agent_reward + sum(other_agents_rewards)
#     return welfare
#
#
# def egalitarian_welfare(current_agent_reward, other_agents_rewards, **kwargs):
#     c_agent_v_over_disagreement = current_agent_reward - kwargs['current_agent_v_with_disagreement']
#     other_agents_v_over_disagreement = [ a_v - a_d_v for a_v, a_d_v in zip(other_agents_rewards,
#                                                          kwargs['other_agents_v_with_disagreement'])]
#     welfare = [c_agent_v_over_disagreement] + other_agents_v_over_disagreement
#     welfare = min(welfare)
#
#     return welfare
#
#
# def nash_welfare(current_agent_reward, other_agents_rewards, **kwargs):
#     c_agent_v_over_disagreement = current_agent_reward - kwargs['current_agent_v_with_disagreement']
#     other_agents_v_over_disagreement = [ a_v - a_d_v for a_v, a_d_v in zip(other_agents_rewards,
#                                                          kwargs['other_agents_v_with_disagreement'])]
#
#     welfare = c_agent_v_over_disagreement
#     for v in other_agents_v_over_disagreement:
#         welfare *= v
#
#     return welfare