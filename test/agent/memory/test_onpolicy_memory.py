from collections import Counter
from flaky import flaky
import numpy as np
import pytest


def memory_init_util(memory):
    assert memory.true_size == 0
    assert memory.total_experiences == 0
    return True


def memory_reset_util(memory, experiences):
    memory.reset()
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
    return True


class TestOnPolicyBatchMemory:
    '''
    Class for unit testing OnPolicyBatchReplay memory
    Note: each test examples from test_memory consists of
          a tuple containing three elements:
          (memory, batch_size, experiences)
    '''

    def test_memory_init(self, test_on_policy_batch_memory):
        memory = test_on_policy_batch_memory[0]
        assert memory_init_util(memory)

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
                memory.add_experience(*e)
        assert memory.body.agent.algorithm.to_train == 1

    def test_reset(self, test_on_policy_batch_memory):
        '''Tests memory reset.
        Adds 2 experiences, then resets the memory and checks if all appropriate values have been zeroed'''
        memory = test_on_policy_batch_memory[0]
        experiences = test_on_policy_batch_memory[2]
        assert memory_reset_util(memory, experiences)


class TestOnPolicyMemory:
    '''
    Class for unit testing OnPolicyReplay memory
    Note: each test examples from test_memory consists of
          a tuple containing three elements:
          (memory, batch_size, experiences)
    '''

    def test_memory_init(self, test_on_policy_episodic_memory):
        memory = test_on_policy_episodic_memory[0]
        assert memory_init_util(memory)

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
        assert np.array_equal(memory.cur_epi_data['states'][-1], exp[0])
        assert memory.cur_epi_data['rewards'][-1] == exp[1]
        assert memory.cur_epi_data['actions'][-1] == exp[2]
        assert np.array_equal(memory.cur_epi_data['next_states'][-1], exp[3])
        assert memory.cur_epi_data['dones'][-1] == exp[4]
        assert memory.cur_epi_data['priorities'][-1] == 1

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
        experiences = test_on_policy_episodic_memory[2]
        assert memory_reset_util(memory, experiences)
