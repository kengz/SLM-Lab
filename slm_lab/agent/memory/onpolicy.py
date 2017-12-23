from collections import Iterable
from slm_lab.agent.memory.replay import Replay
from slm_lab.lib import logger, util
import numpy as np


class OnPolicyReplay(Replay):
    '''
    Stores agent experiences and returns them in a batch for agent training.

    An experience consists of
        - state: representation of a state
        - action: action taken.
                - One hot encoding (discrete)
                - Real numbers representing mean on action dist (continuous)
        - reward: scalar value
        - next state: representation of next state (should be same as state)
        - done: 0 / 1 representing if the current state is the last in an episode
        - priority (optional): scalar value, unnormalized

    The memory does not have a fixed size. Instead the memory stores data from N episodes, where N is determined by the user. After N episodes, all of the examples are returned to the agent to learn from.

    When the examples are returned to the agent, the memory is cleared to prevent the agent from learning from off policy experiences. This memory is intended for on policy algorithms.

    Differences vs. Replay memory:
        - Experiences are nested into episodes. In Replay experiences are flat, and episode is not tracked
        - The entire memory constitues a batch. In Replay batches are sampled from memory.
        - The memory is cleared automatically when a batch is given to the agent.
    '''

    def __init__(self, agent):
        super(OnPolicyReplay, self).__init__(agent)
        # Don't want total experiences reset when memory is
        self.total_experiences = 0

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
        self.state_dim = default_body.state_dim
        self.action_dim = default_body.action_dim
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.priorities = []
        self.current_episode = {'states': [],
                                'actions': [],
                                'rewards': [],
                                'next_states': [],
                                'dones': [],
                                'priorities': []}
        self.most_recent = [None, None, None, None, None, None]
        self.true_size = 0  # Size of the current memory

    def update(self, action, reward, state, done):
        super(OnPolicyReplay, self).update(action, reward, state, done)

    def add_experience(self,
                       state,
                       action,
                       reward,
                       next_state,
                       done,
                       priority=1):
        '''Interface helper method for update() to add experience to memory'''
        # TODO this is still single body
        self.current_episode['states'].append(state)
        self.current_episode['actions'].append(action)
        self.current_episode['rewards'].append(reward)
        self.current_episode['next_states'].append(next_state)
        self.current_episode['dones'].append(done)
        self.current_episode['priorities'].append(priority)
        # Set most recent
        self.most_recent[0] = state
        self.most_recent[1] = action
        self.most_recent[2] = reward
        self.most_recent[3] = next_state
        self.most_recent[4] = done
        self.most_recent[5] = priority
        # If episode ended, add to memory and clear current_episode
        if done:
            self.states.append(self.current_episode['states'])
            self.actions.append(self.current_episode['actions'])
            self.rewards.append(self.current_episode['rewards'])
            self.next_states.append(self.current_episode['next_states'])
            self.dones.append(self.current_episode['dones'])
            self.priorities.append(self.current_episode['priorities'])
            self.current_episode = {'states': [],
                                    'actions': [],
                                    'rewards': [],
                                    'next_states': [],
                                    'dones': [],
                                    'priorities': []}
            # If agent has collected the desired number of episodes, it is ready to train
            if len(self.states) == self.agent.algorithm.num_epis:
                self.agent.algorithm.to_train = 1
        # Track memory size and num experiences
        self.true_size += 1
        if self.true_size > 1000:
            logger.warn("Memory size exceeded {}".format(true_size))
        self.total_experiences += 1

    def get_most_recent_experience(self):
        '''Returns the most recent experience'''
        return self.most_recent

    def get_batch(self):
        '''
        Returns all the examples from memory in a single batch
        Batch is stored as a dict.
        Keys are the names of the different elements of an experience. Values are nested lists of the corresponding sampled elements. Elements are nested into episodes
        e.g.
            batch = {'states'      : [[s_epi1],[s_epi2],...],
                     'actions'     : [[a_epi1],[a_epi2],...],
                     'rewards'     : [[r_epi1],[r_epi2],...],
                     'next_states' : [[ns_epi1],[ns_epi2],...],
                     'dones'       : [[d_epi1],[d_epi2],...],
                     'priorities'  : [[p_epi1],[p_epi2],...]}
        '''
        batch = {}
        batch['states'] = self.states
        batch['actions'] = self.actions
        batch['rewards'] = self.rewards
        batch['next_states'] = self.next_states
        batch['dones'] = self.dones
        batch['priorities'] = self.priorities
        # Reset memory
        self.post_body_init()
        return batch

    def update_priorities(self, priorities):
        ''' Not relevant for this memory'''
        pass


class OnPolicyBatchReplay(OnPolicyReplay):
    '''
    Same as OnPolicyReplay Memory with the following difference.

    The memory does not have a fixed size. Instead the memory stores data from N experiences, where N is determined by the user. After N experiences, all of the examples are returned to the agent to learn from.

    In contrast, OnPolicyReplay stores entire episodes and stores them in a nested structure. OnPolicyBatchReplay stores experiences in a flat structure.
    '''
    def add_experience(self,
                       state,
                       action,
                       reward,
                       next_state,
                       done,
                       priority=1):
        '''Interface helper method for update() to add experience to memory'''
        # TODO this is still single body
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.priorities.append(priority)
        # Set most recent
        self.most_recent[0] = state
        self.most_recent[1] = action
        self.most_recent[2] = reward
        self.most_recent[3] = next_state
        self.most_recent[4] = done
        self.most_recent[5] = priority
        # Track memory size and num experiences
        self.true_size += 1
        if self.true_size > 1000:
            logger.warn("Memory size exceeded {}".format(true_size))
        self.total_experiences += 1
        # Decide if agent is to train
        if (len(self.states)) == self.agent.algorithm.training_frequency:
            self.agent.algorithm.to_train = 1
            # print("Memory size: {}".format(self.true_size))

    def get_batch(self):
        '''
        Returns all the examples from memory in a single batch
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
        return super(OnPolicyBatchReplay, self).get_batch()
