from flaky import flaky
import numpy as np


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
        memory.reset()
        assert memory.size == 0
        assert len(memory.states) == memory.max_size
        assert len(memory.actions) == memory.max_size
        assert len(memory.rewards) == memory.max_size
        assert len(memory.dones) == memory.max_size
        assert len(memory.priorities) == memory.max_size
        assert memory.tree.write == 0
        assert memory.tree.total() == 0
        assert memory.epsilon[0] == 0.0  # Updated for identity transformation config
        assert memory.alpha[0] == 1.0    # Updated for identity transformation config

    def test_add_experience(self, test_prioritized_replay_memory):
        '''Adds an experience to the memory. Checks that memory size = 1, and checks that the experience values are equal to the experience added'''
        memory = test_prioritized_replay_memory[0]
        memory.reset()
        experiences = test_prioritized_replay_memory[2]
        exp = experiences[0]
        memory.add_experience(**exp)
        assert memory.size == 1
        assert memory.head == 0
        # Handle states and actions with multiple dimensions
        assert np.array_equal(memory.states[memory.head], exp['state'])
        assert memory.actions[memory.head] == exp['action']
        assert memory.rewards[memory.head] == exp['reward']
        assert np.array_equal(memory.ns_buffer[0], exp['next_state'])
        assert memory.dones[memory.head] == exp['done']
        assert memory.priorities[memory.head] == 1000

    def test_wrap(self, test_prioritized_replay_memory):
        '''Tests that the memory wraps round when it is at capacity'''
        memory = test_prioritized_replay_memory[0]
        memory.reset()
        experiences = test_prioritized_replay_memory[2]
        num_added = 0
        for e in experiences:
            memory.add_experience(**e)
            num_added += 1
            assert memory.size == min(memory.max_size, num_added)
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
            memory.add_experience(**e)
        batch = memory.sample()
        assert batch['states'].shape == (batch_size, memory.agent.state_dim)
        assert batch['actions'].shape == (batch_size,)
        assert batch['rewards'].shape == (batch_size,)
        assert batch['next_states'].shape == (batch_size, memory.agent.state_dim)
        assert batch['dones'].shape == (batch_size,)
        assert batch['priorities'].shape == (batch_size,)

    def test_sample_distribution(self, test_prioritized_replay_memory):
        '''Tests if batch conforms to prioritized distribution'''
        memory = test_prioritized_replay_memory[0]
        memory.reset()
        experiences = test_prioritized_replay_memory[2]
        for e in experiences:
            memory.add_experience(**e)
        memory.sample()
        # High priority indices (priority=1000): 0, 4, 7
        # Low priority indices (priority=0): 1, 2, 3, 5, 6
        # Should sample from high priority indices more often
        high_priority_indices = {0, 4, 7}
        sampled_indices = set(memory.batch_idxs)
        # At least one high priority index should be sampled
        assert len(sampled_indices & high_priority_indices) > 0

    def test_reset(self, test_prioritized_replay_memory):
        '''Tests memory reset. Adds 2 experiences, then resets the memory and checks if all appropriate values have been zeroed'''
        memory = test_prioritized_replay_memory[0]
        memory.reset()
        experiences = test_prioritized_replay_memory[2]
        for i in range(2):
            e = experiences[i]
            memory.add_experience(**e)
        memory.reset()
        assert memory.head == -1
        assert memory.size == 0
        assert memory.states[0] is None
        assert memory.actions[0] is None
        assert memory.rewards[0] is None
        assert memory.dones[0] is None
        assert memory.priorities[0] is None
        assert len(memory.ns_buffer) == 0
        assert memory.tree.write == 0
        assert memory.tree.total() == 0

    def test_update_priorities(self, test_prioritized_replay_memory):
        '''Samples from memory, and updates priorities twice. Each time checks that the priorities are updated'''
        memory = test_prioritized_replay_memory[0]
        memory.reset()
        experiences = test_prioritized_replay_memory[2]
        for e in experiences:
            memory.add_experience(**e)
        print(f'memory.priorities: {memory.priorities}')
        memory.sample()
        # First update
        # Manually change tree idxs and batch idxs
        memory.batch_idxs = np.asarray([0, 1, 2, 3]).astype(int)
        memory.tree_idxs = [3, 4, 5, 6]
        print(f'batch_size: {test_prioritized_replay_memory[1]}, batch_idxs: {memory.batch_idxs}, tree_idxs: {memory.tree_idxs}')
        new_errors = np.array([0, 10, 10, 20], dtype=np.float32)
        print(f'new_errors: {new_errors}')
        memory.update_priorities(new_errors)
        memory.tree.print_tree()
        print(f'memory.priorities: {memory.priorities}')
        assert memory.priorities[0] == 0
        assert memory.priorities[1] == 10
        assert memory.priorities[2] == 10
        assert memory.priorities[3] == 20
        # Second update
        new_errors = np.array([90, 0, 30, 0], dtype=np.float32)
        # Manually change tree idxs and batch idxs
        memory.batch_idxs = np.asarray([0, 1, 2, 3]).astype(int)
        memory.tree_idxs = [3, 4, 5, 6]
        print(f'new_errors: {new_errors}')
        memory.update_priorities(new_errors)
        memory.tree.print_tree()
        print(f'memory.priorities: {memory.priorities}')
        assert memory.priorities[0] == 90
        assert memory.priorities[1] == 0
        assert memory.priorities[2] == 30
        assert memory.priorities[3] == 0
