from slm_lab.agent.memory.base import Memory
from slm_lab.lib import util, logger
from slm_lab.lib.decorator import lab_api
import numpy as np
import pydash as _


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

    def __init__(self, body):
        super(Replay, self).__init__(body)

        self.max_size = self.body.agent.spec['memory']['max_size']
        self.state_dim = self.body.state_dim
        self.action_dim = self.body.action_dim
        self.batch_idxs = None
        self.total_experiences = 0  # To track total experiences encountered even with forgetting
        self.stacked = False  # Memory does not stack states
        self.atari = False  # Memory is not specialised for Atari games
        self.reset()

    def reset(self):
        if type(self.state_dim) is int:
            self.states = np.zeros((self.max_size, self.state_dim))
            self.next_states = np.zeros((self.max_size, self.state_dim))
        elif type(self.state_dim) is tuple:
            self.states = np.zeros((self.max_size, *self.state_dim))
            self.next_states = np.zeros((self.max_size, *self.state_dim))
        self.actions = np.zeros((self.max_size, self.action_dim))
        self.rewards = np.zeros((self.max_size, 1))
        self.dones = np.zeros((self.max_size, 1))
        self.priorities = np.zeros((self.max_size, 1))
        self.true_size = 0
        self.head = -1  # Index of most recent experience

    @lab_api
    def update(self, action, reward, state, done):
        '''Interface method to update memory'''
        if not np.isnan(reward):
            self.add_experience(self.last_state, action, reward, state, done)
        self.last_state = state

    def add_experience(self, state, action, reward, next_state, done, priority=1):
        '''Implementation for update() to add experience to memory, expanding the memory size if necessary'''
        # Move head pointer. Wrap around if necessary
        self.head = (self.head + 1) % self.max_size
        logger.debug2(f'state: {state.shape}')
        self.states[self.head] = state
        # make action into one_hot
        if _.is_iterable(action):
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

    # @util.fn_timer
    @lab_api
    def sample(self, batch_size, latest=False):
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
        batch_idxs = self.sample_idxs(batch_size)
        self.batch_idxs = batch_idxs
        batch = {}
        batch['states'] = self.states[batch_idxs]
        batch['actions'] = self.actions[batch_idxs]
        batch['rewards'] = self.rewards[batch_idxs]
        batch['next_states'] = self.next_states[batch_idxs]
        batch['dones'] = self.dones[batch_idxs]
        batch['priorities'] = self.priorities[batch_idxs]
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
        assert len(priorites) == self.batch_idxs.size
        self.priorities[self.batch_idxs] = priorities


class StackReplay(Replay):
    '''Preprocesses an state to be the concatenation of the last n states. Otherwise the same as Replay memory'''
    def __init__(self, body):
        self.num_stacked_states = body.agent.spec['memory']['length_history']
        super(StackReplay, self).__init__(body)
        self.stacked = True  # Memory stacks states

    def reset_last_state(self, state):
        '''Do reset of body memory per session during agent_space.reset() to set last_state'''
        self.last_state = self.preprocess_state(state)

    def clear_buffer(self):
        self.state_buffer = []
        for _ in range(self.num_stacked_states - 1):
            self.state_buffer.append(np.zeros((self.state_dim)))

    def reset(self):
        super(StackReplay, self).reset()
        self.states = np.zeros((self.max_size, self.state_dim * self.num_stacked_states))
        self.next_states = np.zeros((self.max_size, self.state_dim * self.num_stacked_states))
        self.state_buffer = []
        self.clear_buffer()

    def preprocess_state(self, state):
        if len(self.state_buffer) == self.num_stacked_states:
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
        if not np.isnan(reward):
            self.add_experience(self.last_state, action, reward, state, done)
        self.last_state = state
        if done:
            '''Clear buffer so there are no experiences from previous states spilling over to new episodes'''
            self.clear_buffer()
            self.body.state_buffer = []


class Atari(Replay):
    '''Preprocesses an state to be the concatenation of the last four states, after converting the 210 x 160 x 3 image to 84 x 84 x 1 grayscale image, and clips all rewards to [-1, 1] as per "Playing Atari with Deep Reinforcement Learning", Mnih et al, 2013
       Otherwise the same as Replay memory'''
    def __init__(self, body):
        super(Atari, self).__init__(body)
        self.atari = True  # Memory is specialized for playing Atari games

    def reset_last_state(self, state):
        '''Do reset of body memory per session during agent_space.reset() to set last_state'''
        self.last_state = self.preprocess_state(state)

    def clear_buffer(self):
        self.state_buffer = []
        for _ in range(3):
            self.state_buffer.append(np.zeros((84, 84)))

    def reset(self):
        super(Atari, self).reset()
        self.states = np.zeros((self.max_size, 84, 84, 4))
        self.next_states = np.zeros((self.max_size, 84, 84, 4))
        self.state_buffer = []
        self.clear_buffer()

    def preprocess_state(self, state):
        if len(self.state_buffer) == 4:
            del self.state_buffer[0]
        state = util.transform_image(state)
        self.state_buffer.append(state)
        processed_state = np.stack(self.state_buffer, axis=-1)
        return processed_state

    @lab_api
    def update(self, action, reward, state, done):
        '''Interface method to update memory'''
        logger.debug2(f'original reward: {reward}')
        state = self.preprocess_state(state)
        reward = max(-1, min(1, reward))
        logger.debug3(f'state: {state.shape}, reward: {reward}, last_state: {self.last_state.shape}')
        if not np.isnan(reward):
            self.add_experience(self.last_state, action, reward, state, done)
        self.last_state = state
        if done:
            '''Clear buffer so there are no experiences from previous states spilling over to new episodes'''
            self.clear_buffer()
            self.body.state_buffer = []
