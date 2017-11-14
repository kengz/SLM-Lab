'''
Introduction to Unity ml-agents

Refer to README for setup

if you're live-hacking stuff, use `pip install -e .`
https://github.com/Unity-Technologies/ml-agents/blob/master/docs/installation.md

ml-agents Python API doc:
https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Unity-Agents---Python-API.md
'''

import numpy as np
import time
from slm_lab.lib import util
from unityagents import UnityEnvironment

# # Multiple env classes simultaneously
# env_path = environment.get_env_path('3dball')
# env_1 = UnityEnvironment(file_name=env_path, worker_id=1)
# env_path = environment.get_env_path('gridworld')
# env_2 = UnityEnvironment(file_name=env_path, worker_id=2)
# env_1.reset(train_mode=False)
# env_2.reset(train_mode=False)

env_path = util.get_env_path('gridworld')
# use train_mode = False to debug, i.e. render env at real size, real time
train_mode = False

# UnityEnvironment interfaces python with Unity,
# and contains brains for controlling connected agents.
env = UnityEnvironment(file_name=env_path)
print(str(env))

# get the default brain
default_brain = env.brain_names[0]
brain = env.brains[default_brain]
env_info = env.reset(train_mode=train_mode)[default_brain]
'''
is_continuous = (brain.action_space_type == 'continuous')
use_observations = (brain.number_observations > 0)
use_states = (brain.state_space_size > 0)

- reset env with param, returns dict of {brain: BrainInfo}
env.reset(train_mode=train_mode)
env_info = env.reset(train_mode=train_mode)[default_brain]

- list of 4D np arrays. nth element = nth observation (pixel-wise) of the brain
env_info.observations
- 2D np array of (batch_size, state_size) for cont and discrete
env_info.states.shape

- 2D np array of (batch_size, memory_size) which corresponds to
  the memories sent at previous step
env_info.memories

- list of scalar rewards for each agent of the brain
env_info.rewards

- list of done status of each agent of the brain
env_info.local_done

- list of ids of agents of the brain
env_info.agents

env.reset(train_mode=True, config=None)
env.step(action, memory=None, value=None)
- action can be 1D array or 2D array if you have multiple agents per brains
- memory is an optional input that can be used to send a list of floats
  per agents to be retrieved at the next step.
- value is an optional input that be used to send a single float per agent
  to be displayed if and AgentMonitor.cs component is attached to the agent.
if u have more than one brain, use dict for action per brain
action = {'brain1': [1.0, 2.0], 'brain2': [3.0, 4.0]}
'''


# TODO move interface logic from openai lab
for epi in range(10):
    # env.global_done could be used to check all
    env_info = env.reset(train_mode=train_mode)[default_brain]
    done = False
    epi_rewards = 0
    while not done:
        if brain.action_space_type == 'continuous':
            action = np.random.randn(
                len(env_info.agents), brain.action_space_size)
            env_info = env.step(action)[default_brain]
        else:
            action = np.random.randint(
                0, brain.action_space_size, size=(len(env_info.agents)))
            env_info = env.step(action)[default_brain]
        epi_rewards += env_info.rewards[0]
        done = env_info.local_done[0]
    print('Total reward for this episode: {}'.format(epi_rewards))


env.close()
print('Environment is closed')
