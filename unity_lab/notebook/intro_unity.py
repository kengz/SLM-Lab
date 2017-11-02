'''
Introduction to Unity ml-agents

Install Unity ml-agents in lab Conda env
if you're live-hacking stuff, use `pip install -e .`
https://github.com/Unity-Technologies/ml-agents/blob/master/docs/installation.md

ml-agents Python API doc:
https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Unity-Agents---Python-API.md
'''

# TODO find env app path, use environment module method
# TODO unity app sizing bug lol

import numpy as np
from unity_lab.lib import util
from unityagents import UnityEnvironment

# TODO util method to infer path of environment, plus gitignore the file
env_name = f'{util.ROOT_DIR}/unity_lab/environment/3DBall'
train_mode = True

# UnityEnvironment interfaces python with Unity,
# and contains brains for controlling connected agents.
env = UnityEnvironment(file_name=env_name)
print(str(env))

# get the default brain
default_brain = env.brain_names[0]
brain = env.brains[default_brain]

is_continuous = (brain.action_space_type == 'continuous')
use_observations = (brain.number_observations > 0)
use_states = (brain.state_space_size > 0)

# reset env with param, returns dict of {brain: BrainInfo}
env.reset(train_mode=train_mode)
env_info = env.reset(train_mode=train_mode)[default_brain]

# list of 4D np arrays. nth element = nth observation (pixel-wise) of the brain
env_info.observations
# 2D np array of (batch_size, state_size) for cont and discrete
env_info.states.shape

# 2D np array of (batch_size, memory_size) which corresponds to the memories sent at previous step
env_info.memories

# list of scalar rewards for each agent of the brain
env_info.rewards

# list of done status of each agent of the brain
env_info.local_done

# list of ids of agents of the brain
env_info.agents


'''
env.reset(train_model=True, config=None)
env.step(action, memory=None, value=None)
where action can be 1D array or 2D array if you have multiple agents per brains
memory is an optional input that can be used to send a list of floats per agents to be retrieved at the next step.
value is an optional input that be used to send a single float per agent to be displayed if and AgentMonitor.cs component is attached to the agent.
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
    print('Total reward this episode: {}'.format(epi_rewards))


env.close()
