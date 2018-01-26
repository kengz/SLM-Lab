# SLM Lab
[![CircleCI](https://circleci.com/gh/kengz/SLM-Lab.svg?style=shield)](https://circleci.com/gh/kengz/SLM-Lab) [![Maintainability](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/maintainability)](https://codeclimate.com/github/kengz/SLM-Lab/maintainability) [![Test Coverage](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/test_coverage)](https://codeclimate.com/github/kengz/SLM-Lab/test_coverage)
A research framework for Deep Reinforcement Learning using Unity, OpenAI Gym, PyTorch, Tensorflow.

**[Gitbook Documentation](https://kengz.gitbooks.io/slm-lab/content/)**

>Doc will migrate to Gitbook entirely soon, except for some crucial commands below.

## Installation

1. Clone the [SLM-Lab repo](https://github.com/kengz/SLM-Lab):
    ```shell
    git clone https://github.com/kengz/SLM-Lab.git
    ```

2. Install dependencies (or inspect `bin/setup_*` first):
    ```shell
    cd SLM-Lab/
    bin/setup
    yarn install
    source activate lab
    ```

### Setup

A config file `config/default.json` will be created.

```json
{
  "data_sync_dir": "~/Dropbox/SLM-Lab/data",
  "plotly": {
    "username": "get from https://plot.ly/settings/api",
    "api_key": "generate from https://plot.ly/settings/api"
  }
}
```

- update `"data_sync_dir"` if you run lab on remote and want to sync data for easy access; it will copy `data/` there.
- for plots, sign up for a free [Plotly account](https://plot.ly/) and update `"plotly"`.


#### `screen` mode

If you're running the lab on a remote ssh server, use a screen so you can exit:

```shell
# enter the screen with the name "lab"
screen -S lab
# run lab over ssh
source activate lab
yarn start
# use Cmd+A+D to detach from screen, then Cmd+D to disconnect ssh

# to resume screen next time
screen -r lab
# use Cmd+D to terminate screen when lab ends
```

### Experiment

>This is still under development; will be improved in the coming releases.

To run an experiment:

- set up your spec in `spec/`, e.g. the `agent, env, body, meta, search` info. There's a spec checker to verify it is valid.
- all algorithm hyperparams are stored in a spec file in `slm_lab/spec` organized by algorithm. Modify parameters in the relevant spec
- set `meta.train_mode` to `False` to render env to debug; set to `True` to run faster without rendering.
- specify the name of your spec in `run_lab.py`, and if you wish to run `Experiment` instead of `Trial`. Note, you need to specify `search` in spec to run `Experiment`.
- schedule experiment in `config/experiments.json`. See more below.
- run `yarn start`
- after completion, check `data/`. You should see `session_graph.png` and `experiment_graph.png`, along with accompanying data.
- to kill stuck processes, run `yarn kill`. To reset, run `yarn reset`. More commands below.

### Scheduling Lab Experiments

Use `config/experiments.json` to specify what experiments to run in the Lab. The format is:
```json
{
	"{spec_file}": {
		"{spec_name}": "{run_mode}"
	}
}
```

E.g. the following will run 2 experiments with param search defined in the spec files.
```json
"dqn.json": {
	"dqn_cartpole": "search"
},
"reinforce.json": {
	"reinforce_cartpole": "search"
}
```

The `run_mode`s are:
- `search`: use `Ray.tune` to run experiment trials on specs over the defined search space.
- `train`: run a trial from scratch (with training) using its default values.
- `enjoy`: run a trial using a saved model (assuming model file exists).
- `benchmark`: to test a new algorithm, run `search` over a predefined set of benchmark environments.
- `dev`: development mode, with override of DEBUG flag, shorter episodes, train_mode.

### Tips

This will update all of the packages to the latest required versions.

### High Level `yarn` commands

| Function | `command` |
| :------------- | :------------- |
| start the Lab | `yarn start` |
| watch and sync the Lab data | `yarn watch` |
| reset lab - clear data/ and log/ | `yarn reset` |
| clear cache | `yarn clear` |
| kill stuck processes | `yarn kill` |
| update dependency file | `yarn export-env` |
| update environment | `yarn update-env` |
| install SLM-Env Unity env and Lab dependencies | `yarn install` |
| run tests | `yarn test` |

## Features

This lab is for general deep reinforcement learning research, built with proper software engineering:

- reusable components for fast experimentation
- established baseline algorithms (in progress). Currently implemented are
    - DQN
    - Double DQN
    - REINFORCE
        - Option to add entropy to encourage exploration
    - Actor-Critic
        - Batch or episodic training
        - Advantage calculated using n-step returns or generalized advantage estimation
        - Option to add entropy to encourage exploration
- plugs in to OpenAI gym, Unity ml-agents environments
- generalized multi-agent, multi-environment setup
- auto hyperparameter search within experiment
- standardized fitness vector to compare results
- standardized benchmarking across algorithms and environments (coming next)
- useful session and experiment graphs and data format
- more experiments coming soon, e.g. multitask architecture, mutual information, mixed model-value-based learning, correspondence principle

## Development

The Lab uses interactive programming workflow:

1.  Install [Atom text editor](https://atom.io/)
2.  Install [Hydrogen for Atom](https://atom.io/packages/hydrogen) and [these other Atom packages optionally](https://gist.github.com/kengz/70c20a0cb238ba1fbb29cdfe402c6470#file-packages-json-L3)
3.  Use this [killer keymap for Hydrogen and other Atom shortcuts](https://gist.github.com/kengz/70c20a0cb238ba1fbb29cdfe402c6470#file-keymap-cson-L15-L18).
    -   Install [Sync Settings](https://atom.io/packages/sync-settings)
    -   Fork the keymap gist
    -   Then update the sync settings config
4.  Open and run the example `slm_lab/notebook/intro_hydrogen.py` on Atom using Hydrogen and those keymaps
5.  See `slm_lab/notebook/intro_unity.py` to see example of Unity environment usage
6.  Start working from `slm_lab/notebook/`


### Unity Environment

The prebuilt Unity environments are released and versioned on [npmjs.com](https://www.npmjs.com/), under names `slm-env-{env_name}`, e.g. `slm-env-3dball`, `slm-env-gridworld`. To use, just install them via `yarn`: e.g. `yarn add slm-env-3dball`.

Check what env_name is available on the git branches of [SLM-Env](https://github.com/kengz/SLM-Env.git).

For building and releasing Unity environments, refer to [README of SLM-Env](https://github.com/kengz/SLM-Env.git).


### Agents-Environments-Bodies design

Proper semantics yield better understanding; below lays out the Lab's generalized structure and relations of agents, bodies and environments.

First, some semantics correction of Unity ml-agents is needed. Sine this environment module handles the interface with Unity ml-agents, the correction will happen here.

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

When controlling a session of experiment, execute the agent and environment logic as usual, but the singletons for AgentSpace and EnvSpace respectively. Internally, they shall produce the usual singleton data across all bodies at each point `(a,e,b)`. When passing the data around, simply flatten the data on the corresponding axis and spread the data. E.g. when passing new states from EnvSpace to AgentSpace, group `state(a,e,b)` for each `a` value and pass `state(e,b)_a` to the right agent `a`.

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

## Contributing

If you're interested in using the lab, please reach out to the authors.
