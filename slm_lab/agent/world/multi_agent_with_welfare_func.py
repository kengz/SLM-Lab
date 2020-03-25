# from slm_lab.agent.world import DefaultMultiAgentWorld, world_util
# from slm_lab.lib import logger, util, viz
# from slm_lab.lib.decorator import lab_api
# import torch
# from copy import deepcopy
#
# logger = logger.get_logger(__name__)
#
#
# class WelfareMultiAgentWorld(DefaultMultiAgentWorld):
#     '''
#     SimpleMultiAgent abstraction; implements the API to interface with Env in SLM Lab
#     Contains algorithm(s), memory(s), body
#     '''
#
#     def __init__(self, spec, env, global_nets_list=None):
#
#         logger.info("INIT WelfareMultiAgentWorld")
#         assert len(spec['agent']) > 1
#         super().__init__(spec, env, global_nets_list)
#
#     @lab_api
#     def update(self, state, action, reward, next_state, done, additional_list_of_dict):
#         '''Update per timestep after env transitions, e.g. memory, algorithm, update agent params, train net'''
#
#         loss, explore_var = [], []
#         reward = list(reward)
#         assert len(self.agents) == len(state) == len(action) == len(reward) == len(next_state) == len(additional_list_of_dict)
#         for idx, (agent, s, a, r, n_s, additional_dict) in enumerate(zip(
#                 self.agents, state, action, reward, next_state, additional_list_of_dict)):
#             other_agent_reward = deepcopy(reward)
#             other_agent_reward.pop(idx)
#             welfare = agent.welfare_function(r, other_agent_reward, additional_dict)
#
#             if util.in_eval_lab_modes():
#                 l, e_v = agent.update(s, a, welfare, n_s, done)
#                 loss.append(l)
#                 explore_var.append(e_v)
#             else:
#                 agent.update(s, a, welfare, n_s, done)
#
#         if util.in_eval_lab_modes():
#             loss = torch.cat(loss, dim=0).mean()
#             explore_var = torch.cat(explore_var, dim=0).mean()
#             return loss, explore_var