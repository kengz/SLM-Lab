import numpy as np


class ReplayMemory:
    '''
    Simple storage for storing agent experiences and sampling from them for
    agent training

    An experience consists of
        - state: representation of a state
        - action: action taken.
                - One hot encoding (discrete)
                - Real numbers representing mean on action dist (continuous)
        - reward: scalar value
        - next state: representation of next state (should be same as state)
        - terminal: 0 / 1 representing if the current state is the last
                    in an episode
        - priority (optional): scalar value, unnormalized

    The memory has a size of N. When capacity is reached, the oldest experience
    is deleted to make space for the lastest experience.
        - This is implemented as a circular buffer so that inserting
          experiences are O(1)
        - Each element of an experience is stored as a separate array of size
          N * element dim

    When a batch of experiences is requested, K experiences are sampled
    according to a random uniform distribution.

    All experiences have a priority of 1.
    This allows for other implementations to sample based on the experience
    priorities
    '''

    def __init__(self, size, state_dim, action_dim):
        '''
        size: maximum size of the memory
        state_dim: tuple of state dims e.g [5] or [3, 84, 84]
        action_dim: tuple of action dime e.g. [4]
        '''
        super(ReplayMemory, self).__init__()
        self.max_size = size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reset_memory()

    def add_experience(self,
                       state,
                       action,
                       reward,
                       terminal,
                       next_state,
                       priority=1):
        '''Adds experience to memory, expanding the memory size if necessary'''
        # Move head pointer. Wrap around if necessary
        self.head = (self.head + 1) % self.max_size
        # Add newest experience
        self.states[self.head] = state
        self.actions[self.head] = action
        self.rewards[self.head] = reward
        self.terminals[self.head] = terminal
        self.next_states[self.head] = next_state
        self.priorities[self.head] = priority
        # Update memory size if necessary
        if self.current_size < self.max_size:
            self.current_size += 1
        self.total_experiences += 1

    def get_most_recent_experience(self):
        '''Returns the most recent experience'''
        experience = []
        experience.append(self.states[self.head])
        experience.append(self.actions[self.head])
        experience.append(self.rewards[self.head])
        experience.append(self.terminals[self.head])
        experience.append(self.next_states[self.head])
        experience.append(self.priorities[self.head])
        return experience

    def get_batch(self, batch_size):
        '''
        Returns a batch of batch_size samples.
        Batch is stored as a dict.
        Keys are the names of the different elements of
        an experience. Values are an array of the corresponding
        sampled elements
        e.g.
            batch = {'states'      : states,
                     'actions'     : actions,
                     'rewards'     : rewards,
                     'terminals'   : terminals,
                     'next_states' : next_states,
                     'priorities'  : priorities}
        '''
        self.sample_indices(batch_size)
        batch = {}
        batch['states'] = self.states[self.current_batch_indices]
        batch['actions'] = self.actions[self.current_batch_indices]
        batch['rewards'] = self.rewards[self.current_batch_indices]
        batch['terminals'] = self.terminals[self.current_batch_indices]
        batch['next_states'] = self.next_states[self.current_batch_indices]
        batch['priorities'] = self.priorities[self.current_batch_indices]
        return batch

    def sample_indices(self, batch_size):
        '''Batch indices a sampled random uniformly'''
        self.current_batch_indices = \
            np.random.choice(list(range(self.current_size)), batch_size)

    def update_priorities(self, priorities):
        '''
        Updates the priorities from the most recent batch
        Assumes the relevant batch indices are stored in
        self.current_batch_indicies
        '''
        assert len(priorites) == self.current_batch_indicies.size
        self.priorities[self.current_batch_indicies] = priorities

    def reset_memory(self):
        '''
        Initializes all of the memory parameters to a blank memory
        Can also be used to clear the memory
        '''
        self.states = np.zeros((self.max_size, *self.state_dim))
        self.actions = np.zeros((self.max_size, *self.action_dim))
        self.rewards = np.zeros((self.max_size, 1))
        self.terminals = np.zeros((self.max_size, 1))
        self.next_states = np.zeros((self.max_size, *self.state_dim))
        self.priorities = np.zeros((self.max_size, 1))
        self.current_size = 0
        self.head = -1  # Index of most recent experience
        self.current_batch_indices = None
        self.total_experiences = 0
        assert self.states is not None
        assert self.actions is not None
        assert self.rewards is not None
        assert self.terminals is not None
        assert self.next_states is not None
        assert self.priorities is not None
