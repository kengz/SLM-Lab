from collections import deque
from copy import deepcopy
from slm_lab.agent.memory.base import Memory
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import pydash as ps

logger = logger.get_logger(__name__)


class OnPolicyReplay(Memory):
    '''
    Stores agent experiences and returns them in a batch for agent training.

    An experience consists of
        - state: representation of a state
        - action: action taken
        - reward: scalar value
        - next state: representation of next state (should be same as state)
        - done: 0 / 1 representing if the current state is the last in an episode

    The memory does not have a fixed size. Instead the memory stores data from N episodes, where N is determined by the user. After N episodes, all of the examples are returned to the agent to learn from.

    When the examples are returned to the agent, the memory is cleared to prevent the agent from learning from off policy experiences. This memory is intended for on policy algorithms.

    Differences vs. Replay memory:
        - Experiences are nested into episodes. In Replay experiences are flat, and episode is not tracked
        - The entire memory constitues a batch. In Replay batches are sampled from memory.
        - The memory is cleared automatically when a batch is given to the agent.

    e.g. memory_spec
    "memory": {
        "name": "OnPolicyReplay"
    }
    '''

    def __init__(self, memory_spec, body):
        super().__init__(memory_spec, body)
        # NOTE for OnPolicy replay, frequency = episode; for other classes below frequency = frames
        # Don't want total experiences reset when memory is
        self.is_episodic = True
        self.size = 0  # total experiences stored
        self.seen_size = 0  # total experiences seen cumulatively
        # declare what data keys to store
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones']
        self.reset()

    @lab_api
    def reset(self):
        '''Resets the memory. Also used to initialize memory vars'''
        for k in self.data_keys:
            setattr(self, k, [])
        self.cur_epi_data = {k: [] for k in self.data_keys}
        self.most_recent = (None,) * len(self.data_keys)
        self.size = 0

    @lab_api
    def update(self, state, action, reward, next_state, done):
        '''Interface method to update memory'''
        self.add_experience(state, action, reward, next_state, done)

    def add_experience(self, state, action, reward, next_state, done):
        '''Interface helper method for update() to add experience to memory'''
        self.most_recent = (state, action, reward, next_state, done)
        for idx, k in enumerate(self.data_keys):
            self.cur_epi_data[k].append(self.most_recent[idx])
        # If episode ended, add to memory and clear cur_epi_data
        if util.epi_done(done):
            for k in self.data_keys:
                getattr(self, k).append(self.cur_epi_data[k])
            self.cur_epi_data = {k: [] for k in self.data_keys}
            # If agent has collected the desired number of episodes, it is ready to train
            # length is num of epis due to nested structure
            if len(self.states) == self.body.agent.algorithm.training_frequency:
                self.body.agent.algorithm.to_train = 1
        # Track memory size and num experiences
        self.size += 1
        self.seen_size += 1

    def sample(self):
        '''
        Returns all the examples from memory in a single batch. Batch is stored as a dict.
        Keys are the names of the different elements of an experience. Values are nested lists of the corresponding sampled elements. Elements are nested into episodes
        e.g.
        batch = {
            'states'     : [[s_epi1], [s_epi2], ...],
            'actions'    : [[a_epi1], [a_epi2], ...],
            'rewards'    : [[r_epi1], [r_epi2], ...],
            'next_states': [[ns_epi1], [ns_epi2], ...],
            'dones'      : [[d_epi1], [d_epi2], ...]}
        '''
        batch = {k: getattr(self, k) for k in self.data_keys}
        self.reset()
        return batch


class OnPolicyBatchReplay(OnPolicyReplay):
    '''
    Same as OnPolicyReplay Memory with the following difference.

    The memory does not have a fixed size. Instead the memory stores data from N experiences, where N is determined by the user. After N experiences or if an episode has ended, all of the examples are returned to the agent to learn from.

    In contrast, OnPolicyReplay stores entire episodes and stores them in a nested structure. OnPolicyBatchReplay stores experiences in a flat structure.

    e.g. memory_spec
    "memory": {
        "name": "OnPolicyBatchReplay"
    }
    * batch_size is training_frequency provided by algorithm_spec
    '''

    def __init__(self, memory_spec, body):
        super().__init__(memory_spec, body)
        self.is_episodic = False

    def add_experience(self, state, action, reward, next_state, done):
        '''Interface helper method for update() to add experience to memory'''
        self.most_recent = [state, action, reward, next_state, done]
        for idx, k in enumerate(self.data_keys):
            getattr(self, k).append(self.most_recent[idx])
        # Track memory size and num experiences
        self.size += 1
        self.seen_size += 1
        # Decide if agent is to train
        if len(self.states) == self.body.agent.algorithm.training_frequency:
            self.body.agent.algorithm.to_train = 1

    def sample(self):
        '''
        Returns all the examples from memory in a single batch. Batch is stored as a dict.
        Keys are the names of the different elements of an experience. Values are a list of the corresponding sampled elements
        e.g.
        batch = {
            'states'     : states,
            'actions'    : actions,
            'rewards'    : rewards,
            'next_states': next_states,
            'dones'      : dones}
        '''
        return super().sample()


class OnPolicyCrossEntropy(OnPolicyReplay):
    '''
    Same as OnPolicyReplay with the addition of the cross entropy method.

    We collect a bach of episode with respect to the training_frequency argument and then we keep only the top cross_entropy
    percent of episodes with respect to the accumulated reward.

    e.g. memory_spec
    "memory": {
        "name": "OnPolicyCrossEntropy",
        "cross_entropy" : 1.0,
    }

    See: Kroese, Dirk P., et al. "Cross-entropy method." Encyclopedia of Operations Research and Management Science (2013): 326-333.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.399.7005&rep=rep1&type=pdf (section 2)
    '''

    def __init__(self, memory_spec, body):
        # set default
        util.set_attr(self, dict(
            cross_entropy=1.0,
        ))
        util.set_attr(self, memory_spec, [
            'cross_entropy',
        ])
        super().__init__(memory_spec, body)

    def filter_episodes(self, batch, cross_entropy):
        '''Filter the episodes for the cross_entropy method'''
        accumulated_reward = [sum(rewards) for rewards in batch['rewards']]
        percentile = cross_entropy * 100
        reward_bound = np.percentile(accumulated_reward, percentile)
        # we save the batch with reward above the bound
        result = {k: [] for k in self.data_keys}
        episode_kept = 0
        for i in range(len(accumulated_reward)):
            if accumulated_reward[i] >= reward_bound:
                for k in self.data_keys:
                    result[k].append(batch[k][i])
                episode_kept += 1
        return result

    def sample(self):
        '''
        Refer to the parent methods for documentation
        If the cross entropy parameter is activated, we filter the collected episodes
        '''
        batch = {k: getattr(self, k) for k in self.data_keys}
        self.reset()
        # we remove the episodes below the cross_entropy percentage
        if self.cross_entropy < 1.0:
            batch = self.filter_episodes(batch, self.cross_entropy)
        return batch
