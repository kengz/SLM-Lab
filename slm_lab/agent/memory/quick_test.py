import numpy as np
import copy
import pprint
from base_memory import ReplayMemory

def print_memory(memory):
    print("Max size: {}, current size: {}, head: {}".format(
    memory.max_size, memory.current_size, memory.head
    ))
    print("States: \n{}\n".format(memory.states))
    print("Actions: \n{}\n".format(memory.actions))
    print("Rewards: \n{}\n".format(memory.rewards))
    print("Terminals: \n{}\n".format(memory.terminals))
    print("Next states: \n{}\n".format(memory.next_states))
    print("Priorities: \n{}\n".format(memory.priorities))

memory = ReplayMemory(5, [1], [1])
batch_size = 2
experiences = [
    [1,1,1,0,2],
    [2,2,2,0,3],
    [3,3,3,0,4],
    [4,4,4,0,5],
    [5,5,5,0,6],
    [6,6,6,0,7],
    [7,7,7,0,8],
    [8,8,8,0,9],
    [9,9,9,0,10],
    [10,10,10,0,11],
    [11,11,11,1,0]]

# Testing add experience
exp = experiences[0]
print(exp)
memory.add_experience(*exp)
assert memory.current_size == 1
assert memory.head == 0
assert memory.states[memory.head] == exp[0]
assert memory.actions[memory.head] == exp[1]
assert memory.rewards[memory.head] == exp[2]
assert memory.terminals[memory.head] == exp[3]
assert memory.next_states[memory.head] == exp[4]
assert memory.priorities[memory.head] == 1

# Testing get most recent experience
memory.reset_memory()
print("Memory as 6 experiences are added")
print_memory(memory)
for i in range(10):
    e = experiences[i]
    memory.add_experience(*e)
    print("Memory after {}th experience added".format(i))
    print_memory(memory)
    last_e = copy.deepcopy(e)
e = memory.get_most_recent_experience()
for orig_e, mem_e in zip(last_e, e):
    assert orig_e == mem_e

# Testing wrap memory
print("Testing memory wrap")
memory.reset_memory()
num_added = 0
for e in experiences:
    memory.add_experience(*e)
    num_added += 1
    print("Max size: {}, current size: {}, head: {}, num_added: {}".format(
    memory.max_size, memory.current_size, memory.head, num_added
    ))
    assert memory.current_size == min(memory.max_size, num_added)
    assert memory.head == (num_added - 1) % memory.max_size

# Testing sample
memory.reset_memory()
for e in experiences:
    memory.add_experience(*e)
batch = memory.get_batch(batch_size)
print("batch size = {}".format(batch_size))
assert batch['states'].shape == (batch_size, *memory.state_dim)
assert batch['actions'].shape == (batch_size, *memory.action_dim)
assert batch['rewards'].shape == (batch_size, 1)
assert batch['terminals'].shape == (batch_size, 1)
assert batch['next_states'].shape == (batch_size, *memory.state_dim)
assert batch['priorities'].shape == (batch_size, 1)
print("A batch looks like this")
pprint.pprint(batch)

# Testing samples are different
memory.reset_memory()
for e in experiences:
    memory.add_experience(*e)
_ = memory.get_batch(batch_size)
old_idx = copy.deepcopy(memory.current_batch_indices).tolist()
for i in range(5):
    _ = memory.get_batch(batch_size)
    new_idx = memory.current_batch_indices.tolist()
    assert old_idx != new_idx
    old_idx = copy.deepcopy(memory.current_batch_indices).tolist()

# Test reset memory
memory.reset_memory()
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
