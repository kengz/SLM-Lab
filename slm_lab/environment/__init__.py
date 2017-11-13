'''
The environment module
Contains graduated components from experiments for building/using environment.
Provides the rich experience for agent embodiment,
reflects the curriculum and allows teaching (possibly allows teacher to enter).
To be designed by human and evolution module,
based on the curriculum and fitness metrics.

---

Agents-Environments-Bodies design
Proper semantics yield better understanding; below lays out the Lab's generalized structure and relations of agents, bodies and environments.

First, some semantics correction of Unity ml-agents is needed. Sine this environment module handles the interface with Untiy ml-agents, the correction will happen here.

The motivating problem:
Originally, in a single instance of environment sits the Academy, which houses multiple Brains, which can each control multiple "Agents". The Brains can be controlled externally from Unity, e.g. via DQN implementation in PyTorch.
However, in Lab, we also call DQN an Agent (different from the Agent inside Unity). Each instance of DQN (Agent) controls a Unity Brain, which can then control multiple Agents (name clash) in Unity, e.g. robot arms. Whereas the multiple arms should be seen as a DQN Agent having many arms, or having an arm in multiple incarnations across space.
Hence, we will call Unity Brain's "Agents" as "Bodies", consistent with SLM's need to have a body in environment for embodiment.

Then, the proper semantics is as follow:
- Agent: a single class/instance of the SLM entity, e.g. DQN agent. This corresponds precisely to a single Brain in Unity Academy.
- Environment: a single class/instance of the Unity space, as usual.
- Body: a single incarnation of an Agent in the Environment. A single Agent (Brain) can have multiple bodies in parallel for batch training.

Note that the parallel bodies (identical and non-interacting) of an agent in an environment is equivalent to an agent with a single body existing in multiple copies of the environment. This insight is crucial for the symmetry between Agent and Environment space, and helps generalize further later.

The base case:
- 1 agent, 1 environment, 1 body
This is the most straightforward case, directly runnable as a common session without any multiplicity resolution.

Multi-body case:
- 1 agent, 1 environment, multiple bodies
This is just the base case ran in batch, where the agent does batch-processing on input and output.
Alternatively the bodies could be distinct, such as having inverse rewards. This would be the adversarial case where a single agent self-plays.

Multi-agent case:
- multiple agents, 1 environment, multiple bodies
The next extension is having multiple agents interacting in an environment. Each agent can posses 1 body or more as per cases above.

Multi-environment case:
- 1 agent, multiple environments, multiple bodies
This is the more novel case. When an agent can have parallel incarnations, nothing restrictst the bodies to be constructed identically or be subject to the same environment. An agent can have multiple bodies in different environments.
This can be used for simultaneous multi-task training. An example is to expose an agent's legs to ground for walking, wings to air for flying, and fins for swimming. The goal would be to do generalization or transfer learning on all 3 types of limbs to multiple environments. Then perhaps it would generalize to use legs and wings for swimming too.

Full generalization, multi-agent multi-environment case:
- multiple agents, multiple environments, multiple bodies
This generalizes all the cases above and allow us to have a neat representation that corresponds to the Agent-Environment product space before.
The generalization gives us the 3D space of `Agents x Environments x Bodies`. We will call this product space `AEB space`. It will be the basis of our experiment design.
In AEB space, We have the projections:
- AgentSpace, A: each value in this space is a class of agent
- EnvSpace, E: each value in this space is a class of environment
- BodySpace, B: each value in this space is a body of an agent in an environment (indexed by coordinates (a,e) in AE space)

In a general experiment with multiple bodies, with single or multiple agents and environments, each body instance can be marked with the 3D coordinate `(a,e,b)` in `AEB` space. Each body is also associated with the body-specific data: observables, actions, rewards, done flags. We can call these the data space, i.e. observable space, action space, reward space, etc.

When controlling a session of experiment, execute the agent and environment logic as usual, but the singletons for AgentSpace and EnvSpace respectively. Internally, they shall produce the usual singleton data across all bodies at each point `(a,e,b)`. When passing the data around, simply flatten the data on the corresponding axis and spread the data. E.g. when passing new states from EnvSpace to AgentSpace, group `state(a,e,b)` for each `a` value and pass state(e,b)_a to the right agent `a`.

Hence, the experiment session loop generalizes directly from:
```
state = self.env.reset()
self.agent.reset()
# RL steps for SARS
for t in range(self.env.max_timestep):
    action = self.agent.act(state)
    reward, state, done = self.env.step(action)
    # fully observable SARS from env, memory and training internally
    self.agent.update(reward, state)
    self.monitor.update()
    if done:
        break
```

to direct substitutions for singletons with spaces:
```
state_space = self.env_space.reset()
self.agent_space.reset()
# RL steps for SARS
for t in range(self.env_space.common_refinement_max_timestep):
    action_space = self.agent_space.act(state_space)
    reward_space, state_space, done_space = self.env_space.step(action_space)
    # fully observable SARS from env_space, memory and training internally
    self.agent_space.update(reward_space, state_space)
    self.monitor.update()
    if done_space.all_done:
        break
```
'''
import os
import pydash as _
from slm_lab.lib import util
from unityagents import UnityEnvironment
from unityagents.brain import BrainParameters

# TODO should really move into somewhere in experiment module, cuz agent and environment modules really should be atomic and not worry much about multiplicities
# TODO implement base case first


class BrainExt:
    '''
    Unity Brain class extension, where self = brain
    to be absorbed into ml-agents Brain class later
    '''

    # TODO or just set properties for all these, no method
    def is_discrete(self):
        return self.action_space_type == 'discrete'

    def get_action_dim(self):
        return self.action_space_size

    def get_observable(self):
        '''What channels are observable: state, visual, sound, touch, etc.'''
        observable = {
            'state': self.state_space_size > 0,
            'visual': self.number_observations > 0,
        }
        return observable

    def get_observable_dim(self):
        '''Get observable dimensions'''
        observable_dim = {
            'state': self.state_space_size,
            'visual': 'some np array shape, as opposed to what Arthur called size',
        }
        return


def extend_unity_brain():
    '''Extend Unity BrainParameters class at runtime to add BrainExt methods'''
    ext_fn_list = util.get_fn_list(BrainExt)
    for fn in ext_fn_list:
        setattr(BrainParameters, fn, getattr(BrainExt, fn))


extend_unity_brain()


def get_env_path(env_name):
    env_path = util.smart_path(
        f'node_modules/slm-env-{env_name}/build/{env_name}')
    env_dir = os.path.dirname(env_path)
    assert os.path.exists(
        env_dir), f'Missing {env_path}. See README to install from yarn.'
    return env_path


class Env:
    '''
    Do the above
    Also standardize logic from Unity environments
    '''
    # TODO perhaps do extension like above again
    name = None
    index = None
    max_timestep = None
    train_mode = None
    u_env = None
    agent = None

    def __init__(self, name, index, train_mode):
        self.name = name
        self.index = index
        self.train_mode = train_mode
        self.u_env = UnityEnvironment(
            file_name=get_env_path(self.name),
            worker_id=self.index)
        # TODO set proper from spec
        self.max_timestep = 40
        # TODO expose brain methods properly to env
        default_brain = self.u_env.brain_names[0]
        brain = self.u_env.brains[default_brain]
        ext_fn_list = util.get_fn_list(brain)
        for fn in ext_fn_list:
            setattr(self, fn, getattr(brain, fn))

    def set_agent(self, agent):
        '''
        Make agent visible to env.
        TODO anticipate multi-agents
        '''
        self.agent = agent

    def reset(self):
        # TODO need AEB space resolver
        default_brain = self.u_env.brain_names[0]
        env_info = self.u_env.reset(train_mode=self.train_mode)[default_brain]
        return env_info

    def step(self, action):
        agent_index = 0
        default_brain = self.u_env.brain_names[agent_index]
        env_info = self.u_env.step(action)[default_brain]
        # TODO body-resolver:
        body_index = 0
        reward = env_info.rewards[body_index]
        state = env_info.states[body_index]
        done = env_info.local_done[body_index]
        return reward, state, done

    def close(self):
        self.u_env.close()


class RectifiedUnityEnv:
    '''
    Unity Environment wrapper
    '''

    def get_brain(brain_name):
        return self.u_env.brains[brain_name]

    def fn_spread_brains(self, brain_fn):
        '''Call single-brain function on all for {brain_name: info}'''
        brains_info = {
            brain_name: brain_fn(brain_name)
            for brain_name in self.u_env.brains
        }
        return brains_info

    def is_discrete(self):
        return self.fn_spread_brains('is_discrete')

    def get_observable():
        observable = self.fn_spread_brains('get_observable')
        return observable

    # TODO handle multi-brain logic
    # TODO split subclass to handle unity specific logic,
    # and the other half to handle Lab specific logic
    # TODO also make clear that unity.brain.agent is not the same as RL agent here. unity agents could be seen as multiple simultaneous incarnations of this agent
    # TODO actually shd do a single-brain wrapper instead
    # then on env level call on all brains with wrapper methods, much easier
    # Remedy:
    # 1. Env class to rectify UnityEnv
    # 2. use Env class as proper
    # Rectify steps:
