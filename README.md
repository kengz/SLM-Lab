# SLM Lab [![CircleCI](https://circleci.com/gh/kengz/SLM-Lab.svg?style=shield)](https://circleci.com/gh/kengz/SLM-Lab) [![Maintainability](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/maintainability)](https://codeclimate.com/github/kengz/SLM-Lab/maintainability) [![Test Coverage](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/test_coverage)](https://codeclimate.com/github/kengz/SLM-Lab/test_coverage)
_(Work In Progress)_ An experimentation framework for Reinforcement Learning using Unity, OpenAI Gym, and PyTorch.

## Installation

1.  Clone the Lab repo:
    ```shell
    git clone https://github.com/kengz/SLM-Lab.git
    ```

2.  Install Lab dependencies (or inspect `bin/*` before running):
    ```shell
    cd SLM-Lab/
    bin/setup
    yarn install
    source activate lab
    ```

3.  Install the bleeding-edge Unity `ml-agents` dependency:
    ```shell
    source activate lab
    cd ..
    git clone https://github.com/Unity-Technologies/ml-agents.git
    cd ml-agents/python
    pip install .
    cd ../SLM-Lab/
    ```

4.  Setup the created config files:
    -   sign up for a free [Plotly account](https://plot.ly/) and get the API key to put in the config files below.
    -   `config/default.json` for local development, used when `grunt` is ran without a production flag.
    -   `config/production.json` for production lab run when `grunt -prod` is ran with the production flag `-prod`.

## Usage

### Quickstart

Once you're all set up, run the demo of `DQN` in `CartPole-v0`:

- see `slm_lab/spec/demo.json` for example spec.
- run the following on terminal, in the repo directory:
    ```shell
    source activate lab
    yarn start
    ```
- check the output for data `slm_lab/data/dqn_cartpole/`


### Tips

If you experience package errors, try the following from the root of the SLM-Lab folder

```bash
git pull
bin/setup
```

This will update all of the packages to the latest required versions.

### High Level `yarn` commands

| Function | `command` |
| :------------- | :------------- |
| install SLM-Env Unity env and Lab dependencies | `yarn install` |
| start the Lab | `yarn start` |
| update dependency file | `yarn export-env` |
| update environment | `yarn update-env` |
| run tests | `yarn test` |
| reset lab - clear data/ and log/ | `yarn reset` |
| clear cache | `yarn clear` |
| kill stuck Unity ports/processes | `yarn kill` |

## Notebook

The Lab uses interactive programming and lit workflow:

1.  Install [Atom text editor](https://atom.io/)
2.  Install [Hydrogen for Atom](https://atom.io/packages/hydrogen) and [these other Atom packages optionally](https://gist.github.com/kengz/70c20a0cb238ba1fbb29cdfe402c6470#file-packages-json-L3)
3.  Use this [killer keymap for Hydrogen and other Atom shortcuts](https://gist.github.com/kengz/70c20a0cb238ba1fbb29cdfe402c6470#file-keymap-cson-L15-L18).
    -   Install [Sync Settings](https://atom.io/packages/sync-settings)
    -   Fork the keymap gist
    -   Then update the sync settings config
4.  Open and run the example `slm_lab/notebook/intro_hydrogen.py` on Atom using Hydrogen and those keymaps
5.  See `slm_lab/notebook/intro_unity.py` to see example of Unity environment usage
6.  Start working from `slm_lab/notebook/`

## Experiment

_To be set up_

## Unity Environment

The prebuilt Unity environments are released and versioned on [npmjs.com](https://www.npmjs.com/), under names `slm-env-{env_name}`, e.g. `slm-env-3dball`, `slm-env-gridworld`. To use, just install them via `yarn`: e.g. `yarn add slm-env-3dball`.

Check what env_name is available on the git branches of [SLM-Env](https://github.com/kengz/SLM-Env.git).

For building and releasing Unity environments, refer to [README of SLM-Env](https://github.com/kengz/SLM-Env.git).


## Agents-Environments-Bodies design

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
    logger.debug(f'action {action}')
    reward, state, done = self.env.step(action)
    logger.debug(f'reward: {reward}, state: {state}, done: {done}')
    # fully observable SARS from env, memory and training internally
    self.agent.update(reward, state)
    if done:
        break
```

to direct substitutions for singletons with spaces:
```
state_space = self.env_space.reset()
self.agent_space.reset()
# RL steps for SARS
for t in range(self.env_space.max_timestep):
    action_space = self.agent_space.act(state_space)
    logger.debug(f'action_space {action_space}')
    (reward_space, state_space,
     done_space) = self.env_space.step(action_space)
    # completes cycle of full info for agent_space
    self.agent_space.update(reward_space, state_space, done_space)
    if bool(done_space):
        break
```
