import pytest
import numpy as np
import copy
from collections import Counter


class TestMemory:
    '''
    Base class for unit testing replay memory
    Note: each test examples from test_memory consists of
          a tuple containing three elements:
          (memory, batch_size, experiences)
    '''

    def test_memory_init(self, test_memory):
        memory = test_memory[0]
        assert memory.current_size == 0
        assert memory.states.shape == (memory.max_size, *memory.state_dim)
        assert memory.actions.shape == (memory.max_size, *memory.action_dim)
        assert memory.next_states.shape == (memory.max_size, *memory.state_dim)
        assert memory.terminals.shape == (memory.max_size, 1)
        assert memory.rewards.shape == (memory.max_size, 1)
        assert memory.priorities.shape == (memory.max_size, 1)

    def test_add_experience(self, test_memory):
        '''
        Adds an experience to the memory.
        Checks that memory size = 1, and checks that
        the experience values are equal to the experience
        added
        '''
        memory = test_memory[0]
        memory.reset_memory()
        experiences = test_memory[2]
        exp = experiences[0]
        memory.add_experience(*exp)
        assert memory.current_size == 1
        assert memory.head == 0
        # Handle states and actions with multiple dimensions
        if memory.states[memory.head].size > 1:
            for i in range(memory.states[memory.head].size):
                assert memory.states[memory.head][i] == exp[0][i]
                assert memory.next_states[memory.head][i] == exp[4][i]
            for i in range(memory.actions[memory.head].size):
                assert memory.actions[memory.head][i] == exp[1][i]
        else:
            assert memory.states[memory.head] == exp[0]
            assert memory.actions[memory.head] == exp[1]
            assert memory.next_states[memory.head] == exp[4]
        assert memory.rewards[memory.head] == exp[2]
        assert memory.terminals[memory.head] == exp[3]
        assert memory.priorities[memory.head] == 1

    def test_get_most_recent_experience(self, test_memory):
        '''
        Adds 6 experiences to the memory and checks
        that the most recent experience is equal
        to the last experience added
        '''
        memory = test_memory[0]
        memory.reset_memory()
        experiences = test_memory[2]
        last_e = None
        for i in range(6):
            e = experiences[i]
            memory.add_experience(*e)
            last_e = copy.deepcopy(e)
        e = memory.get_most_recent_experience()
        # Handle states and actions with multiple dimensions
        for orig_e, mem_e in zip(last_e, e):
            if mem_e.size > 1:
                for i in range(mem_e.size):
                    assert orig_e[i] == mem_e[i]
            else:
                assert orig_e == mem_e

    def test_wrap(self, test_memory):
        '''Tests that the memory wraps round when it is at capacity'''
        memory = test_memory[0]
        memory.reset_memory()
        experiences = test_memory[2]
        num_added = 0
        for e in experiences:
            memory.add_experience(*e)
            num_added += 1
            assert memory.current_size == min(memory.max_size, num_added)
            assert memory.head == (num_added - 1) % memory.max_size

    @pytest.mark.skip(reason='flaky test, see https://circleci.com/gh/kengz/SLM-Lab/262')
    def test_sample(self, test_memory):
        '''
        Tests that a sample of batch size is returned
        with the correct dimensions
        '''
        memory = test_memory[0]
        memory.reset_memory()
        batch_size = test_memory[1]
        experiences = test_memory[2]
        for e in experiences:
            memory.add_experience(*e)
        batch = memory.get_batch(batch_size)
        assert batch['states'].shape == (batch_size, *memory.state_dim)
        assert batch['actions'].shape == (batch_size, *memory.action_dim)
        assert batch['rewards'].shape == (batch_size, 1)
        assert batch['terminals'].shape == (batch_size, 1)
        assert batch['next_states'].shape == (batch_size, *memory.state_dim)
        assert batch['priorities'].shape == (batch_size, 1)

    @pytest.mark.skip(reason='flaky test, see https://circleci.com/gh/kengz/SLM-Lab/296')
    def test_sample_changes(self, test_memory):
        '''
        Tests if memory.current_batch_indices changes
        from sample to sample
        '''
        memory = test_memory[0]
        memory.reset_memory()
        batch_size = test_memory[1]
        experiences = test_memory[2]
        for e in experiences:
            memory.add_experience(*e)
        _ = memory.get_batch(batch_size)
        old_idx = copy.deepcopy(memory.current_batch_indices).tolist()
        for i in range(5):
            _ = memory.get_batch(batch_size)
            new_idx = memory.current_batch_indices.tolist()
            assert old_idx != new_idx
            old_idx = copy.deepcopy(memory.current_batch_indices).tolist()

    def test_reset_memory(self, test_memory):
        '''
        Tests memory reset.
        Adds 2 experiences, then resets the memory
        and checks if all appropriate values have been
        zeroed'''
        memory = test_memory[0]
        memory.reset_memory()
        experiences = test_memory[2]
        for i in range(2):
            e = experiences[i]
            memory.add_experience(*e)
        memory.reset_memory()
        assert memory.head == -1
        assert memory.current_size == 0
        assert np.sum(memory.states) == 0
        assert np.sum(memory.actions) == 0
        assert np.sum(memory.rewards) == 0
        assert np.sum(memory.terminals) == 0
        assert np.sum(memory.next_states) == 0
        assert np.sum(memory.priorities) == 0

    def test_sample_dist(self, test_memory):
        '''
        Samples 100 times from memory.
        Accumulates the indices sampled and checks
        for significant deviation from a uniform distribution'''
        # TODO test_sample_dist
        assert None is None

    def test_update_priorities(self, test_memory):
        '''
        Samples from memory, and updates all priorities from
        1 to 2. Checks that correct experiences are updated
        '''
        # TODO implement test_update_priorities
        assert None is None
