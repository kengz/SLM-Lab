from slm_lab.lib import logger
from slm_lab.agent.world import DefaultMultiAgentWorld
from slm_lab.lib import logger, util, viz
from slm_lab.lib.decorator import lab_api

logger = logger.get_logger(__name__)


class DefaultSingleAgentWorld(DefaultMultiAgentWorld):
    '''
    SimpleMultiAgent abstraction; implements the API to interface with Env in SLM Lab
    Contains algorithm(s), memory(s), body
    '''

    def __init__(self, spec, env, global_nets_list=None, **kwargs):

        assert len(spec['agent']) == 1
        super().__init__(spec, env, global_nets_list)