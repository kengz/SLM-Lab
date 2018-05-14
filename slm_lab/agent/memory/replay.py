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
        - action: action taken.
                - One hot encoding (discrete)
                - Real numbers representing mean on action dist (continuous)
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
        self.state_dim = self.body.state_dim
        self.action_dim = self.body.action_dim
        self.batch_idxs = None
        self.total_experiences = 0  # To track total experiences encountered even with forgetting
        self.stacked = False  # Memory does not stack states
        self.atari = False  # Memory is not specialised for Atari games
        self.last_done = None
        self.reset()
        self.print_memory_info()

    def reset(self):
        '''Initializes the memory arrays, size and head pointer'''
        states_shape = np.concatenate([[self.max_size], np.reshape(self.state_dim, -1)])
        actions_shape = np.concatenate([[self.max_size], np.reshape(self.action_dim, -1)])
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones', 'priorities']
        setattr(self, 'states', np.zeros(states_shape))
        setattr(self, 'actions', np.zeros(actions_shape, dtype=np.uint16))
        setattr(self, 'rewards', np.zeros((self.max_size, 1)))
        setattr(self, 'next_states', np.zeros(states_shape))
        setattr(self, 'dones', np.zeros((self.max_size, 1), dtype=np.uint8))
        setattr(self, 'priorities', np.zeros((self.max_size, 1)))
        self.true_size = 0
        self.head = -1  # Index of most recent experience

    @lab_api
    def update(self, action, reward, state, done):
        '''Interface method to update memory.
        Guard for nan rewards and last state from previous episode'''
        if (not np.isnan(reward)) and (self.last_done != 1):
            self.add_experience(self.last_state, action, reward, state, done)
        self.last_state = state
        self.last_done = done

    def add_experience(self, state, action, reward, next_state, done, priority=1):
        '''Implementation for update() to add experience to memory, expanding the memory size if necessary'''
        # Move head pointer. Wrap around if necessary
        self.head = (self.head + 1) % self.max_size
        logger.debug2(f'state: {state.shape}')
        self.states[self.head] = state
        # make action into one_hot
        if ps.is_iterable(action):
            # non-singular action
            # self.actions[self.head] = one hot of multi-action (matrix) on a 3rd axis, to be implement
            raise NotImplementedError
        else:
            self.actions[self.head][action] = 1
        self.rewards[self.head] = reward
        self.next_states[self.head] = next_state
        self.dones[self.head] = done
        self.priorities[self.head] = priority
        # Actually occupied size of memory
        if self.true_size < self.max_size:
            self.true_size += 1
        self.total_experiences += 1

    @lab_api
    def sample(self, latest=False):
        '''
        Returns a batch of batch_size samples.
        Batch is stored as a dict.
        Keys are the names of the different elements of an experience. Values are an array of the corresponding sampled elements
        e.g.
            batch = {'states'      : states,
                     'actions'     : actions,
                     'rewards'     : rewards,
                     'next_states' : next_states,
                     'dones'       : dones,
                     'priorities'  : priorities}
        '''
        # TODO if latest, return unused. implement
        if latest:
            raise NotImplementedError
        batch_idxs = self.sample_idxs(self.batch_size)
        self.batch_idxs = batch_idxs
        batch = {k: getattr(self, k)[batch_idxs] for k in self.data_keys}
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
            logger.info(f'MEMORY: {k} :shape: {d.shape}, dtype: {d.dtype}, size: {util.memory_size(d)}MB')


class StackReplay(Replay):
    '''Preprocesses an state to be the concatenation of the last n states. Otherwise the same as Replay memory'''

    def __init__(self, memory_spec, algorithm, body):
        super(StackReplay, self).__init__(memory_spec, algorithm, body)
        self.stacked = True  # Memory stacks states
        util.set_attr(self, self.memory_spec, [
            'batch_size',
            'max_size',
            'stack_len',  # num_stack_states
        ])

    def reset_last_state(self, state):
        '''Do reset of body memory per session during agent_space.reset() to set last_state'''
        self.last_state = self.preprocess_state(state)

    def clear_buffer(self):
        '''Clears the raw state buffer'''
        self.state_buffer = []
        for _ in range(self.stack_len - 1):
            self.state_buffer.append(np.zeros((self.orig_state_dim)))

    def reset(self):
        '''Initializes the memory arrays, size and head pointer'''
        self.orig_state_dim = self.state_dim
        self.state_dim = self.state_dim * self.stack_len
        super(StackReplay, self).reset()
        self.state_buffer = []
        self.clear_buffer()

    def preprocess_state(self, state):
        '''Transforms the raw state into format that is fed into the network'''
        if len(self.state_buffer) == self.stack_len:
            del self.state_buffer[0]
        self.state_buffer.append(state)
        processed_state = np.concatenate(self.state_buffer)
        return processed_state

    @lab_api
    def update(self, action, reward, state, done):
        '''Interface method to update memory'''
        state = self.preprocess_state(state)
        logger.debug(f'state: {state.shape}, reward: {reward}, last_state: {self.last_state.shape}')
        logger.debug2(f'state buffer: {self.state_buffer}, state: {state}')
        if (not np.isnan(reward)) and (self.last_done != 1):
            self.add_experience(self.last_state, action, reward, state, done)
        self.last_state = state
        self.last_done = done
        if done:
            '''Clear buffer so there are no experiences from previous states spilling over to new episodes'''
            self.clear_buffer()
            self.body.state_buffer = []


class Atari(Replay):
    '''Preprocesses an state to be the concatenation of the last four states, after converting the 210 x 160 x 3 image to 84 x 84 x 1 grayscale image, and clips all rewards to [-1, 1] as per "Playing Atari with Deep Reinforcement Learning", Mnih et al, 2013
       Otherwise the same as Replay memory'''

    def __init__(self, memory_spec, algorithm, body):
        super(Atari, self).__init__(memory_spec, algorithm, body)
        self.atari = True  # Memory is specialized for playing Atari games

    def reset_last_state(self, state):
        '''Do reset of body memory per session during agent_space.reset() to set last_state'''
        self.last_state = self.preprocess_state(state)

    def clear_buffer(self):
        '''Clears the raw state buffer'''
        self.state_buffer = []
        for _ in range(3):
            self.state_buffer.append(np.zeros((84, 84)))

    def reset(self):
        '''Initializes the memory arrays, size and head pointer'''
        assert self.state_dim == (84, 84, 4)
        super(Atari, self).reset()
        self.state_buffer = []
        self.clear_buffer()

    def preprocess_state(self, state):
        '''Transforms the raw state into format that is fed into the network'''
        if len(self.state_buffer) == 4:
            del self.state_buffer[0]
        state = util.transform_image(state)
        self.state_buffer.append(state)
        processed_state = np.stack(self.state_buffer, axis=-1).astype(np.float16)
        return processed_state

    @lab_api
    def update(self, action, reward, state, done):
        '''Interface method to update memory'''
        logger.debug2(f'original reward: {reward}')
        state = self.preprocess_state(state)
        reward = max(-1, min(1, reward))
        logger.debug3(f'state: {state.shape}, reward: {reward}, last_state: {self.last_state.shape}')
        if (not np.isnan(reward)) and (self.last_done != 1):
            self.add_experience(self.last_state, action, reward, state, done)
        self.last_state = state
        self.last_done = done
        if done:
            '''Clear buffer so there are no experiences from previous states spilling over to new episodes'''
            self.clear_buffer()
            self.body.state_buffer = []
