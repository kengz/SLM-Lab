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
        - terminal: True / False representing if the current state is the last
                    in an episode
        - priority (optional): scalar value, unnormalized

    The memory has a size of N. When capacity is reached, the oldest experience
    is deleted to make space for the lastest experience.
        - This is implemented as a circular buffer so that inserting and
        deleting memories are O(1) once the memory is at capacity
        - Each element of an experience is stored as a separate array of size
          N * element dim

    When a batch of experiences is requested, K experiences are sampled according
    to a random uniform distribution.

    All experiences have a priority of 1.
    This allows for other implementations to sample based on the experience
    priorities
    '''

    def __init__(self, size, state_dim, action_dim):
        super(ReplayMemory, self).__init__()
        self.states = np.zeros(size, *state_dim)
        self.actions = np.zeros(size, *action_dim)
        self.rewards = np.zeros(size, 1)
        self.next_states = np.zeros(size, *state_dim)
        self.priorities = np.zeros(size, 1)
        self.max_size = size
        self.current_size = 0
        self.head = 0  # Index of most recent experience
        self.tail = -1  # Index of least recent experience
        self.current_batch_indices = None
        self.total_experiences = 0

    def add_experience(self, state, action, reward, next_state, priority=1):
        '''Adds experience to memory, expanding the memory size if necessary'''
        # Move head and tail pointers. Wrap around if necessary
        self.head = (self.head + 1) % self.max_size
        self.tail = (self.tail + 1) % self.max_size
        # Add newest experience
        self.states[self.head] = state
        self.actions[self.head] = action
        self.rewards[self.head] = reward
        self.next_states[self.head] = next_states
        self.priorities[self.head] = priority
        # Update memory size if necessary
        if self.current_size < self.max_size:
            self.current_size += 1
        self.total_experiences += 1
        self.check_lengths()

    def get_most_recent_experience(self):
        '''Returns the most recent experience'''
        experience = []
        experience.append(self.states[self.head])
        experience.append(self.actions[self.head])
        experience.append(self.rewards[self.head])
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
            batch = {'states'       : states,
                     'actions'      : actions,
                     'rewards'      : rewards,
                     'next_states'  : next_states,
                     'priorities'   : priorities}
        '''
        self.sample_indices(batch_size)
        batch = {}
        batch['states'] = self.states[self.current_batch_indices]
        batch['actions'] = self.actions[self.current_batch_indices]
        batch['rewards'] = self.rewards[self.current_batch_indices]
        batch['next_states'] = self.next_states[self.current_batch_indices]
        batch['priorities'] = self.priorities[self.current_batch_indices]
        return batch

    def sample_indices(self, batch_size):
        '''Batch indices a sampled random uniformly'''
        self.current_batch_indices = \
            np.random.choice(list(range(self.current_size)))

    def update_priorities(self, priorities):
        '''
        Updates the priorities from the most recent batch
        Assumes the relevant batch indices are stored in
        self.current_batch_indicies
        '''
        assert len(priorites) == self.current_batch_indicies.size
        self.priorities[self.current_batch_indicies] = priorities
