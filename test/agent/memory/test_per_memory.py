from collections import Counter
from flaky import flaky
import numpy as np
import pytest
import torch


@flaky
class TestPERMemory:
    '''
    Class for unit testing prioritized replay memory
    Note: each test examples from test_prioritized_replay_memory consists of
          a tuple containing three elements:
          (memory, batch_size, experiences)
    '''

    def test_prioritized_replay_memory_init(self, test_prioritized_replay_memory):
        memory = test_prioritized_replay_memory[0]
        assert memory.true_size == 0
        assert memory.states.shape == (memory.max_size, memory.body.state_dim)
        assert memory.actions.shape == (memory.max_size,)
        assert memory.rewards.shape == (memory.max_size,)
        assert memory.dones.shape == (memory.max_size,)
        assert memory.priorities.shape == (memory.max_size,)
        assert memory.tree.write == 0
        assert memory.tree.total() == 0
        assert memory.epsilon[0] == 0
        assert memory.alpha[0] == 1

    def test_add_experience(self, test_prioritized_replay_memory):
        '''Adds an experience to the memory. Checks that memory size = 1, and checks that the experience values are equal to the experience added'''
        memory = test_prioritized_replay_memory[0]
        memory.reset()
        experiences = test_prioritized_replay_memory[2]
        exp = experiences[0]
        memory.add_experience(*exp)
        assert memory.true_size == 1
        assert memory.head == 0
        # Handle states and actions with multiple dimensions
        assert np.array_equal(memory.states[memory.head], exp[0])
        assert memory.actions[memory.head] == exp[1]
        assert memory.rewards[memory.head] == exp[2]
        assert memory.dones[memory.head] == exp[4]
        assert memory.priorities[memory.head] == 100000

    def test_wrap(self, test_prioritized_replay_memory):
        '''Tests that the memory wraps round when it is at capacity'''
        memory = test_prioritized_replay_memory[0]
        memory.reset()
        experiences = test_prioritized_replay_memory[2]
        num_added = 0
        for e in experiences:
            memory.add_experience(*e)
            num_added += 1
            assert memory.true_size == min(memory.max_size, num_added)
            assert memory.head == (num_added - 1) % memory.max_size
            write = (num_added - 1) % memory.max_size + 1
            if write == memory.max_size:
                write = 0
            assert memory.tree.write == write

    def test_sample(self, test_prioritized_replay_memory):
        '''Tests that a sample of batch size is returned with the correct dimensions'''
        memory = test_prioritized_replay_memory[0]
        memory.reset()
        batch_size = test_prioritized_replay_memory[1]
        experiences = test_prioritized_replay_memory[2]
        for e in experiences:
            memory.add_experience(*e)
        batch = memory.sample()
        assert batch['states'].shape == (batch_size, memory.body.state_dim)
        assert batch['actions'].shape == (batch_size,)
        assert batch['rewards'].shape == (batch_size,)
        assert batch['next_states'].shape == (batch_size, memory.body.state_dim)
        assert batch['dones'].shape == (batch_size,)
        assert batch['priorities'].shape == (batch_size,)

    def test_sample_distribution(self, test_prioritized_replay_memory):
        '''Tests if batch conforms to prioritized distribution'''
        memory = test_prioritized_replay_memory[0]
        memory.reset()
        batch_size = test_prioritized_replay_memory[1]
        experiences = test_prioritized_replay_memory[2]
        for e in experiences:
            memory.add_experience(*e)
        batch = memory.sample()
        assert memory.batch_idxs[0] == 0
        assert memory.batch_idxs[1] == 1
        assert memory.batch_idxs[2] == 2
        assert memory.batch_idxs[3] == 3

    def test_reset(self, test_prioritized_replay_memory):
        '''Tests memory reset. Adds 2 experiences, then resets the memory and checks if all appropriate values have been zeroed'''
        memory = test_prioritized_replay_memory[0]
        memory.reset()
        experiences = test_prioritized_replay_memory[2]
        for i in range(2):
            e = experiences[i]
            memory.add_experience(*e)
        memory.reset()
        assert memory.head == -1
        assert memory.true_size == 0
        assert np.sum(memory.states) == 0
        assert np.sum(memory.actions) == 0
        assert np.sum(memory.rewards) == 0
        assert np.sum(memory.dones) == 0
        assert np.sum(memory.priorities) == 0
        assert memory.tree.write == 0
        assert memory.tree.total() == 0

    def test_update_priorities(self, test_prioritized_replay_memory):
        '''Samples from memory, and updates priorities twice. Each time checks that correct experiences are updated. Finally checks that a new sampled batch corresponds to the prioritized distribution'''
        memory = test_prioritized_replay_memory[0]
        memory.reset()
        batch_size = test_prioritized_replay_memory[1]
        experiences = test_prioritized_replay_memory[2]
        for e in experiences:
            memory.add_experience(*e)
        batch = memory.sample()
        # First update
        print(f'batch_size: {batch_size}, batch_idxs: {memory.batch_idxs}, tree_idxs: {memory.tree_idxs}')
        new_errors = torch.from_numpy(np.asarray([0, 10, 10, 20])).float().unsqueeze_(dim=1)
        print(f'new_errors: {new_errors}')
        memory.update_priorities(new_errors)
        memory.tree.print_tree()
        assert memory.priorities[0] == 0
        assert memory.priorities[1] == 10
        assert memory.priorities[2] == 10
        assert memory.priorities[3] == 20
        batch = memory.sample()
        print(f'batch_size: {batch_size}, batch_idxs: {memory.batch_idxs}, tree_idxs: {memory.tree_idxs}')
        assert memory.batch_idxs[0] == 1
        assert memory.batch_idxs[1] == 2
        assert memory.batch_idxs[2] == 3
        assert memory.batch_idxs[3] == 3
        assert batch['rewards'][0] == 6
        assert batch['rewards'][1] == 7
        assert batch['rewards'][2] == 8
        assert batch['rewards'][3] == 8
        # Second update
        new_errors = torch.from_numpy(np.asarray([90, 0, 30, 0])).float().unsqueeze_(dim=1)
        # Manually change tree idxs and batch idxs
        memory.batch_idxs = np.asarray([0, 1, 2, 3]).astype(int)
        memory.tree_idxs = [3, 4, 5, 6]
        print(f'new_errors: {new_errors}')
        memory.update_priorities(new_errors)
        memory.tree.print_tree()
        assert memory.priorities[0] == 90
        assert memory.priorities[1] == 0
        assert memory.priorities[2] == 30
        assert memory.priorities[3] == 0
        batch = memory.sample()
        assert memory.batch_idxs[0] == 0
        assert memory.batch_idxs[1] == 0
        assert memory.batch_idxs[2] == 0
        assert memory.batch_idxs[3] == 2
        assert batch['rewards'][0] == 5
        assert batch['rewards'][1] == 5
        assert batch['rewards'][2] == 5
        assert batch['rewards'][3] == 7
