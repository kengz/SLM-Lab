from collections import Counter
from copy import deepcopy
from flaky import flaky
import numpy as np
import pytest


class TestMemory:
    '''
    Base class for unit testing replay memory
    Note: each test examples from test_memory consists of
          a tuple containing three elements:
          (memory, batch_size, experiences)
    '''

    def test_memory_init(self, test_memory):
        memory = test_memory[0]
        assert memory.true_size == 0
        assert memory.states.shape == (memory.max_size, memory.state_dim)
        assert memory.actions.shape == (memory.max_size, memory.action_dim)
        assert memory.rewards.shape == (memory.max_size, 1)
        assert memory.next_states.shape == (memory.max_size, memory.state_dim)
        assert memory.dones.shape == (memory.max_size, 1)
        assert memory.priorities.shape == (memory.max_size, 1)

    def test_add_experience(self, test_memory):
        '''Adds an experience to the memory.
        Checks that memory size = 1, and checks that the experience values are equal to the experience added'''
        memory = test_memory[0]
        memory.reset()
        experiences = test_memory[2]
        exp = experiences[0]
        memory.add_experience(*exp)
        assert memory.true_size == 1
        assert memory.head == 0
        # Handle states and actions with multiple dimensions
        assert np.array_equal(memory.states[memory.head], exp[0])
        assert memory.actions[memory.head][exp[1]] == exp[1]
        assert memory.rewards[memory.head] == exp[2]
        assert np.array_equal(memory.next_states[memory.head], exp[3])
        assert memory.dones[memory.head] == exp[4]
        assert memory.priorities[memory.head] == 1

    def test_wrap(self, test_memory):
        '''Tests that the memory wraps round when it is at capacity'''
        memory = test_memory[0]
        memory.reset()
        experiences = test_memory[2]
        num_added = 0
        for e in experiences:
            memory.add_experience(*e)
            num_added += 1
            assert memory.true_size == min(memory.max_size, num_added)
            assert memory.head == (num_added - 1) % memory.max_size

    def test_sample(self, test_memory):
        '''Tests that a sample of batch size is returned with the correct dimensions'''
        memory = test_memory[0]
        memory.reset()
        batch_size = test_memory[1]
        experiences = test_memory[2]
        for e in experiences:
            memory.add_experience(*e)
        batch = memory.sample(batch_size)
        assert batch['states'].shape == (batch_size, memory.state_dim)
        assert batch['actions'].shape == (batch_size, memory.action_dim)
        assert batch['rewards'].shape == (batch_size, 1)
        assert batch['next_states'].shape == (batch_size, memory.state_dim)
        assert batch['dones'].shape == (batch_size, 1)
        assert batch['priorities'].shape == (batch_size, 1)

    def test_sample_changes(self, test_memory):
        '''Tests if memory.current_batch_indices changes from sample to sample'''
        memory = test_memory[0]
        memory.reset()
        batch_size = test_memory[1]
        experiences = test_memory[2]
        for e in experiences:
            memory.add_experience(*e)
        _batch = memory.sample(batch_size)
        old_idx = deepcopy(memory.batch_idxs).tolist()
        for i in range(5):
            _batch = memory.sample(batch_size)
            new_idx = memory.batch_idxs.tolist()
            assert old_idx != new_idx
            old_idx = deepcopy(memory.batch_idxs).tolist()

    def test_reset(self, test_memory):
        '''Tests memory reset.
        Adds 2 experiences, then resets the memory and checks if all appropriate values have been zeroed'''
        memory = test_memory[0]
        memory.reset()
        experiences = test_memory[2]
        for i in range(2):
            e = experiences[i]
            memory.add_experience(*e)
        memory.reset()
        assert memory.head == -1
        assert memory.true_size == 0
        assert np.sum(memory.states) == 0
        assert np.sum(memory.actions) == 0
        assert np.sum(memory.rewards) == 0
        assert np.sum(memory.next_states) == 0
        assert np.sum(memory.dones) == 0
        assert np.sum(memory.priorities) == 0

    @pytest.mark.skip(reason="Not implemented yet")
    def test_sample_dist(self, test_memory):
        '''Samples 100 times from memory.
        Accumulates the indices sampled and checks for significant deviation from a uniform distribution'''
        # TODO test_sample_dist
        assert None is None

    @pytest.mark.skip(reason="Not implemented yet")
    def test_update_priorities(self, test_memory):
        '''Samples from memory, and updates all priorities from 1 to 2. Checks that correct experiences are updated'''
        # TODO implement test_update_priorities
        assert None is None


class TestOnPolicyBatchMemory:
    '''
    Class for unit testing OnPolicyBatchReplay memory
    Note: each test examples from test_memory consists of
          a tuple containing three elements:
          (memory, batch_size, experiences)
    '''
    def test_memory_init(self, test_on_policy_batch_memory):
        memory = test_on_policy_batch_memory[0]
        assert memory.true_size == 0
        assert memory.total_experiences == 0

    def test_add_experience(self, test_on_policy_batch_memory):
        '''Adds an experience to the memory.
        Checks that memory size = 1, and checks that the experience values are equal to the experience added'''
        memory = test_on_policy_batch_memory[0]
        memory.reset()
        experiences = test_on_policy_batch_memory[2]
        exp = experiences[0]
        memory.add_experience(*exp)
        assert memory.true_size == 1
        assert len(memory.states) == 1
        # Handle states and actions with multiple dimensions
        assert np.array_equal(memory.states[-1], exp[0])
        assert memory.rewards[-1] == exp[1]
        assert memory.actions[-1] == exp[2]
        assert np.array_equal(memory.next_states[-1], exp[3])
        assert memory.dones[-1] == exp[4]
        assert memory.priorities[-1] == 1

    def test_sample(self, test_on_policy_batch_memory):
        '''Tests that a sample of batch size is returned with the correct dimensions'''
        memory = test_on_policy_batch_memory[0]
        memory.reset()
        batch_size = test_on_policy_batch_memory[1]
        experiences = test_on_policy_batch_memory[2]
        size = len(experiences)
        for e in experiences:
            memory.add_experience(*e)
        batch = memory.sample()
        assert len(batch['states']) == size
        assert len(batch['rewards']) == size
        assert len(batch['next_states']) == size
        assert len(batch['actions']) == size
        assert len(batch['dones']) == size
        assert len(batch['priorities']) == size
        assert len(memory.states) == 0

    def test_batch_size(self, test_on_policy_batch_memory):
        '''Tests that memory sets agent training flag correctly'''
        memory = test_on_policy_batch_memory[0]
        memory.reset()
        memory.body.agent.algorithm.to_train = 0
        batch_size = test_on_policy_batch_memory[1]
        experiences = test_on_policy_batch_memory[2]
        size = len(experiences)
        for i, e in enumerate(experiences):
            if i == batch_size:
                break
            else:
                assert memory.body.agent.algorithm.to_train == 0
                memory.add_experience(*e)
        assert memory.body.agent.algorithm.to_train == 1

    def test_reset(self, test_on_policy_batch_memory):
        '''Tests memory reset.
        Adds 2 experiences, then resets the memory and checks if all appropriate values have been zeroed'''
        memory = test_on_policy_batch_memory[0]
        memory.reset()
        experiences = test_on_policy_batch_memory[2]
        for i in range(2):
            e = experiences[i]
            memory.add_experience(*e)
        memory.reset()
        assert memory.true_size == 0
        assert np.sum(memory.states) == 0
        assert np.sum(memory.actions) == 0
        assert np.sum(memory.rewards) == 0
        assert np.sum(memory.next_states) == 0
        assert np.sum(memory.dones) == 0
        assert np.sum(memory.priorities) == 0


class TestOnPolicyMemory:
    '''
    Class for unit testing OnPolicyReplay memory
    Note: each test examples from test_memory consists of
          a tuple containing three elements:
          (memory, batch_size, experiences)
    '''
    def test_memory_init(self, test_on_policy_episodic_memory):
        memory = test_on_policy_episodic_memory[0]
        assert memory.true_size == 0
        assert memory.total_experiences == 0

    def test_add_experience(self, test_on_policy_episodic_memory):
        '''Adds an experience to the memory.
        Checks that memory size = 1, and checks that the experience values are equal to the experience added'''
        memory = test_on_policy_episodic_memory[0]
        memory.reset()
        experiences = test_on_policy_episodic_memory[2]
        exp = experiences[0]
        memory.add_experience(*exp)
        assert memory.true_size == 1
        assert len(memory.states) == 0
        # Handle states and actions with multiple dimensions
        assert np.array_equal(memory.current_episode['states'][-1], exp[0])
        assert memory.current_episode['rewards'][-1] == exp[1]
        assert memory.current_episode['actions'][-1] == exp[2]
        assert np.array_equal(memory.current_episode['next_states'][-1], exp[3])
        assert memory.current_episode['dones'][-1] == exp[4]
        assert memory.current_episode['priorities'][-1] == 1

    def test_sample(self, test_on_policy_episodic_memory):
        '''Tests that a sample of batch size is returned with the correct dimensions'''
        memory = test_on_policy_episodic_memory[0]
        memory.reset()
        batch_size = test_on_policy_episodic_memory[1]
        experiences = test_on_policy_episodic_memory[2]
        size = len(experiences)
        for e in experiences:
            memory.add_experience(*e)
        batch = memory.sample()
        assert len(batch['states'][0]) == size
        assert len(batch['rewards'][0]) == size
        assert len(batch['next_states'][0]) == size
        assert len(batch['actions'][0]) == size
        assert len(batch['dones'][0]) == size
        assert len(batch['priorities'][0]) == size
        assert len(memory.states) == 0

    def test_batch_size(self, test_on_policy_episodic_memory):
        '''Tests that memory sets agent training flag correctly'''
        memory = test_on_policy_episodic_memory[0]
        memory.reset()
        memory.body.agent.algorithm.to_train = 0
        batch_size = test_on_policy_episodic_memory[1]
        experiences = test_on_policy_episodic_memory[2]
        size = len(experiences)
        for e in experiences:
            assert memory.body.agent.algorithm.to_train == 0
            memory.add_experience(*e)
        assert memory.body.agent.algorithm.to_train == 1

    def test_multiple_epis_samples(self, test_on_policy_episodic_memory):
        '''Tests that a sample of batch size is returned with the correct number of episodes'''
        memory = test_on_policy_episodic_memory[0]
        memory.reset()
        batch_size = test_on_policy_episodic_memory[1]
        experiences = test_on_policy_episodic_memory[2]
        size = len(experiences)
        for i in range(3):
            for e in experiences:
                memory.add_experience(*e)
        batch = memory.sample()
        assert len(batch['states']) == 3
        assert len(batch['rewards']) == 3
        assert len(batch['next_states']) == 3
        assert len(batch['actions']) == 3
        assert len(batch['dones']) == 3
        assert len(batch['priorities']) == 3
        assert len(batch['states'][0]) == size
        assert len(batch['states'][1]) == size
        assert len(batch['states'][2]) == size
        assert len(memory.states) == 0

    def test_reset(self, test_on_policy_episodic_memory):
        '''Tests memory reset.
        Adds 2 experiences, then resets the memory and checks if all appropriate values have been zeroed'''
        memory = test_on_policy_episodic_memory[0]
        memory.reset()
        experiences = test_on_policy_episodic_memory[2]
        for i in range(2):
            e = experiences[i]
            memory.add_experience(*e)
        memory.reset()
        assert memory.true_size == 0
        assert np.sum(memory.states) == 0
        assert np.sum(memory.actions) == 0
        assert np.sum(memory.rewards) == 0
        assert np.sum(memory.next_states) == 0
        assert np.sum(memory.dones) == 0
        assert np.sum(memory.priorities) == 0
