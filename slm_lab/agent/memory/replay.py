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

    def __init__(self, agent):
        super(Replay, self).__init__(agent)

    def post_body_init(self, bodies=None):
        '''
        Initializes the part of algorithm needing a body to exist first.
        Can also be used to clear the memory.
        '''
        # TODO update for multi bodies
        # TODO also for multi state, multi actions per body, need to be 3D
        # bodies using this shared memory, should be congruent (have same state_dim, action_dim)
        # TODO add warning that memory is env-specific now
        self.bodies = bodies or util.s_get(
            self, 'aeb_space.body_space').get(e=0)
        self.coor_list = [body.coor for body in self.bodies]
        default_body = self.bodies[0]
        self.max_size = self.agent.spec['memory']['max_size']
        self.state_dim = default_body.state_dim
        self.action_dim = default_body.action_dim

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
        # interface
        # add memory from all bodies, interleave
        # TODO proper body-based storage
        for eb_idx, body in enumerate(self.agent.bodies):
            # add only those belonging to the bodies using this memory
            if body.coor in self.coor_list:
                self.add_experience(
                    self.last_state[eb_idx], action[eb_idx], reward[eb_idx], state[eb_idx], done[eb_idx])
        self.last_state = state

    def add_experience(self,
                       state,
                       action,
                       reward,
                       next_state,
                       done,
                       priority=1):
        '''Interface helper method for update() to add experience to memory, expanding the memory size if necessary'''
        # TODO this is still single body
        # Move head pointer. Wrap around if necessary
        self.head = (self.head + 1) % self.max_size
        # spread numbers in numpy since direct list setting is impossible
        self.states[self.head, :] = state[:]
        # make action into one_hot
        if _.is_iterable(action):
            # non-singular action
            # self.actions[self.head] = one hot of multi-action (matrix) on a 3rd axis, to be implement
            raise NotImplementedError
        else:
            self.actions[self.head][action] = 1
        self.rewards[self.head] = reward
        self.next_states[self.head, :] = next_state[:]
        self.dones[self.head] = done
        self.priorities[self.head] = priority
        # Actually occupied size of memory
        if self.true_size < self.max_size:
            self.true_size += 1
        self.total_experiences += 1

    def get_most_recent_experience(self):
        '''Returns the most recent experience'''
        # TODO need to foolproof index reference error. Simple as add a dict. if not private method, data format need be consistent with batch format with keys.
        experience = []
        experience.append(self.states[self.head])
        experience.append(self.actions[self.head])
        experience.append(self.rewards[self.head])
        experience.append(self.next_states[self.head])
        experience.append(self.dones[self.head])
        experience.append(self.priorities[self.head])
        return experience

    def get_batch(self, batch_size):
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
        self.batch_idxs = self.sample_idxs(batch_size)
        batch = {}
        batch['states'] = self.states[self.batch_idxs]
        batch['actions'] = self.actions[self.batch_idxs]
        batch['rewards'] = self.rewards[self.batch_idxs]
        batch['next_states'] = self.next_states[self.batch_idxs]
        batch['dones'] = self.dones[self.batch_idxs]
        batch['priorities'] = self.priorities[self.batch_idxs]
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
