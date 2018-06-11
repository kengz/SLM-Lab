from collections import deque
from copy import deepcopy
from slm_lab.agent.memory.base import Memory
from slm_lab.lib import logger, util
from slm_lab.lib.decorator import lab_api
import numpy as np
import pydash as ps

logger = logger.get_logger(__name__)


class Replay(Memory):
    '''
    Stores agent experiences and samples from them for agent training

    An experience consists of
        - state: representation of a state
        - action: action taken
        - reward: scalar value
        - next state: representation of next state (should be same as state)
        - done: 0 / 1 representing if the current state is the last in an episode
        - priority (optional): scalar value, unnormalized

    The memory has a size of N. When capacity is reached, the oldest experience
    is deleted to make space for the lastest experience.
        - This is implemented as a circular buffer so that inserting experiences are O(1)
        - Each element of an experience is stored as a separate array of size N * element dim

    When a batch of experiences is requested, K experiences are sampled according to a random uniform distribution.

    All experiences have a priority of 1.
    This allows for other implementations to sample based on the experience priorities
    '''

    def __init__(self, memory_spec, algorithm, body):
        super(Replay, self).__init__(memory_spec, algorithm, body)
        util.set_attr(self, self.memory_spec, [
            'batch_size',
            'max_size',
        ])
        self.state_buffer = deque(maxlen=0)  # for API consistency
        self.batch_idxs = None
        self.total_experiences = 0  # To track total experiences encountered even with forgetting
        self.reset()
        self.print_memory_info()

    def reset(self):
        '''Initializes the memory arrays, size and head pointer'''
        states_shape = np.concatenate([[self.max_size], np.reshape(self.body.state_dim, -1)])
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones', 'priorities']
        setattr(self, 'states', np.zeros(states_shape))
        setattr(self, 'actions', np.zeros((self.max_size,), dtype=self.body.action_space.dtype))
        setattr(self, 'rewards', np.zeros((self.max_size,)))
        setattr(self, 'next_states', np.zeros(states_shape))
        setattr(self, 'dones', np.zeros((self.max_size,), dtype=np.uint8))
        setattr(self, 'priorities', np.zeros((self.max_size,)))
        self.true_size = 0
        self.head = -1  # Index of most recent experience

        self.state_buffer.clear()
        for _ in range(self.state_buffer.maxlen):
            self.state_buffer.append(np.zeros(self.body.state_dim))

    @lab_api
    def update(self, action, reward, state, done):
        '''Interface method to update memory.'''
        self.base_update(action, reward, state, done)
        if not np.isnan(reward):  # not the start of episode
            self.add_experience(self.last_state, action, reward, state, done)
        self.last_state = state

    def add_experience(self, state, action, reward, next_state, done, priority=1):
        '''Implementation for update() to add experience to memory, expanding the memory size if necessary'''
        # Move head pointer. Wrap around if necessary
        self.head = (self.head + 1) % self.max_size
        self.states[self.head] = state
        self.actions[self.head] = action
        self.rewards[self.head] = reward
        self.next_states[self.head] = next_state
        self.dones[self.head] = done
        self.priorities[self.head] = priority
        # Actually occupied size of memory
        if self.true_size < self.max_size:
            self.true_size += 1
        self.total_experiences += 1

    @lab_api
    def sample(self):
        '''
        Returns a batch of batch_size samples.
        Batch is stored as a dict.
        Keys are the names of the different elements of an experience. Values are an array of the corresponding sampled elements
        e.g.
        batch = {
            'states'     : states,
            'actions'    : actions,
            'rewards'    : rewards,
            'next_states': next_states,
            'dones'      : dones,
            'priorities' : priorities}
        '''
        self.batch_idxs = self.sample_idxs(self.batch_size)
        batch = {k: getattr(self, k)[self.batch_idxs] for k in self.data_keys}
        return batch

    def sample_idxs(self, batch_size):
        '''Batch indices a sampled random uniformly'''
        batch_idxs = np.random.choice(list(range(self.true_size)), batch_size)
        return batch_idxs

    def update_priorities(self, priorities):
        '''
        Updates the priorities from the most recent batch
        Assumes the relevant batch indices are stored in self.batch_idxs
        '''
        assert len(priorities) == self.batch_idxs.size
        self.priorities[self.batch_idxs] = priorities

    def print_memory_info(self):
        '''Prints size of all of the memory arrays'''
        for k in self.data_keys:
            d = getattr(self, k)
            logger.info(f'Memory for body {self.body.aeb}: {k} :shape: {d.shape}, dtype: {d.dtype}, size: {util.memory_size(d)}MB')


class SeqReplay(Replay):
    '''Preprocesses a state to be the stacked sequence of the last n states. Otherwise the same as Replay memory'''

    def __init__(self, memory_spec, algorithm, body):
        util.set_attr(self, memory_spec, [
            'batch_size',
            'max_size',
        ])
        self.seq_len = algorithm.net_spec['seq_len']
        super(SeqReplay, self).__init__(memory_spec, algorithm, body)
        self.state_buffer = deque(maxlen=self.seq_len)
        self.reset()

    def reset(self):
        '''Initializes the memory arrays, size and head pointer'''
        super(SeqReplay, self).reset()
        # override state shape for concat
        states_shape = np.concatenate([[self.max_size], np.reshape([self.seq_len, self.body.state_dim], -1)])
        setattr(self, 'states', np.zeros(states_shape))
        setattr(self, 'next_states', np.zeros(states_shape))

    def epi_reset(self, state):
        '''Method to reset at new episode'''
        super(SeqReplay, self).epi_reset(self.preprocess_state(state, append=False))

    def preprocess_state(self, state, append=True):
        '''Transforms the raw state into format that is fed into the network'''
        # append when state is first seen when acting in policy_util, don't append elsewhere in memory
        if append:
            self.state_buffer.append(state)
        processed_state = np.stack(self.state_buffer)
        return processed_state

    @lab_api
    def update(self, action, reward, state, done):
        '''Interface method to update memory'''
        self.base_update(action, reward, state, done)
        state = self.preprocess_state(state, append=False)  # prevent conflict with preprocess in epi_reset
        if not np.isnan(reward):  # not the start of episode
            self.add_experience(self.last_state, action, reward, state, done)
        self.last_state = state


class StackReplay(Replay):
    '''Preprocesses a state to be the stacked sequence of the last n states. Otherwise the same as Replay memory'''

    def __init__(self, memory_spec, algorithm, body):
        util.set_attr(self, memory_spec, [
            'batch_size',
            'max_size',
            'stack_len',  # num_stack_states
        ])
        body.state_dim = (body.state_dim * self.stack_len)  # flattened stacked input
        super(StackReplay, self).__init__(memory_spec, algorithm, body)
        self.state_buffer = deque(maxlen=self.stack_len)
        self.reset()

    def epi_reset(self, state):
        '''Method to reset at new episode'''
        super(StackReplay, self).epi_reset(self.preprocess_state(state, append=False))

    def preprocess_state(self, state, append=True):
        '''Transforms the raw state into format that is fed into the network'''
        # append when state is first seen when acting in policy_util, don't append elsewhere in memory
        if append:
            self.state_buffer.append(state)
        processed_state = np.concatenate(self.state_buffer)
        return processed_state


class Atari(Replay):
    '''
    Preprocesses an state to be the concatenation of the last four states, after converting the 210 x 160 x 3 image to 84 x 84 x 1 grayscale image, and clips all rewards to [-1, 1] as per "Playing Atari with Deep Reinforcement Learning", Mnih et al, 2013
    Otherwise the same as Replay memory
    '''

    def __init__(self, memory_spec, algorithm, body):
        self.atari = True  # Memory is specialized for playing Atari games
        util.set_attr(self, memory_spec, [
            'batch_size',
            'max_size',
            'stack_len',  # num_stack_states
        ])
        body.state_dim = (84, 84, self.stack_len)  # greyscale downsized, stacked
        super(Atari, self).__init__(memory_spec, algorithm, body)
        self.state_buffer = deque(maxlen=self.stack_len)
        self.reset()

    def reset(self):
        '''Initializes the memory arrays, size and head pointer'''
        super(Atari, self).reset()
        self.state_buffer.clear()
        for _ in range(self.state_buffer.maxlen):
            self.state_buffer.append(np.zeros(self.body.state_dim[:-1]))

    def epi_reset(self, state):
        '''Method to reset at new episode'''
        super(Atari, self).epi_reset(self.preprocess_state(state, append=False))
        for _ in range(self.state_buffer.maxlen):
            self.state_buffer.append(np.zeros(self.body.state_dim[:-1]))

    def preprocess_state(self, state, append=True):
        '''Transforms the raw state into format that is fed into the network'''
        state = util.transform_image(state)
        # append when state is first seen when acting in policy_util, don't append elsewhere in memory
        if append:
            self.state_buffer.append(state)
        processed_state = np.stack(self.state_buffer, axis=-1).astype(np.float16)
        assert processed_state.shape == self.body.state_dim
        return processed_state

    @lab_api
    def update(self, action, reward, state, done):
        '''Interface method to update memory'''
        self.base_update(action, reward, state, done)
        state = self.preprocess_state(state, append=False)  # prevent conflict with preprocess in epi_reset
        if not np.isnan(reward):  # not the start of episode
            logger.debug2(f'original reward: {reward}')
            if not np.isnan(reward):
                reward = max(-1, min(1, reward))
            logger.info(f'state: {state.shape}, reward: {reward}, last_state: {self.last_state.shape}')
            self.add_experience(self.last_state, action, reward, state, done)
        self.last_state = state
