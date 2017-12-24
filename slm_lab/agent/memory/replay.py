from slm_lab.agent.memory.base import Memory
from slm_lab.lib import util
import numpy as np
import pydash as _


class Replay(Memory):
    '''
    Simple storage for storing agent experiences and sampling from them for agent training

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

        self.states = np.zeros((self.max_size, self.state_dim))
        self.actions = np.zeros((self.max_size, self.action_dim))
        self.rewards = np.zeros((self.max_size, 1))
        self.next_states = np.zeros((self.max_size, self.state_dim))
        self.dones = np.zeros((self.max_size, 1))
        self.priorities = np.zeros((self.max_size, 1))

        self.true_size = 0
        self.head = -1  # Index of most recent experience
        self.batch_idxs = None
        self.total_experiences = 0

    def update(self, action, reward, state, done):
        '''interface method to update memory'''
        self.add_experience(self.last_state, action, reward, state, done)
        self.last_state = state

    def add_experience(self, state, action, reward, next_state, done, priority=1):
        '''Implementation for update() to add experience to memory, expanding the memory size if necessary'''
        # Move head pointer. Wrap around if necessary
        self.head = (self.head + 1) % self.max_size
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
        batch_idxs = np.random.choice(
            list(range(self.true_size)), batch_size)
        return batch_idxs

    def update_priorities(self, priorities):
        '''
        Updates the priorities from the most recent batch
        Assumes the relevant batch indices are stored in self.batch_idxs
        '''
        assert len(priorites) == self.batch_idxs.size
        self.priorities[self.batch_idxs] = priorities
