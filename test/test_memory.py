import pytest
import numpy as np


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
        assert memory.states.shape = (memory.size, *memory.state_dim)
        assert memory.actions.shape = (memory.size, *memory.action_dim)
        assert memory.next_states.shape = (memory.size, *memory.state_dim)
        assert memory.terminals.shape = (memory.size, 1)
        assert memory.rewards.shape = (memory.size, 1)
        assert memory.priorities.shape = (memory.size, 1)

    def test_reset_memory(self, test_memory):
        '''
        Tests memory reset.
        Adds 2 experiences, then resets the memory
        and checks if all appropriate values have been
        zeroed'''
        pass

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
        experience = experiences[0]
        memory.add(exp)
        assert memory.current_size == 1
        assert memory.head = 0
        assert memory.states[memory.head] == exp[0]
        assert memory.actions[memory.head] == exp[1]
        assert memory.rewards[memory.head] == exp[2]
        assert memory.terminals[memory.head] == exp[3]
        assert memory.next_states[memory.head] == exp[4]
        assert memory.priorities[memory.head] == 1

    def test_get_most_recent_experience(self, test_memory):
        '''
        Adds 6 experiences to the memory and checks
        that the most recent experience is equal
        to the last experience added
        '''
        pass

    def test_wrap(self, test_memory):
        '''Tests that the memory wraps round when it is at capacity'''
        memory = test_memory[0]
        memory.reset_memory()
        experiences = test_memory[2]
        num_added = 0
        for e in experiences:
            memory.add(*e)
            num_added += 1
            assert memory.current_size = min(memory.max_size, i + 1)
            assert self.head = (i + 1) % memory.max_size

    def test_sample(self, test_memory):
        pass

    def test_sample_dist(self, test_memory):
        pass

    def test_sample_changes(self, test_memory):
        '''
        Tests if memory.current_batch_indices changes
        from sample to sample
        '''
        pass
