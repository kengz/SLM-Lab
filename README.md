# SLM Lab
[![CircleCI](https://circleci.com/gh/kengz/SLM-Lab.svg?style=shield)](https://circleci.com/gh/kengz/SLM-Lab) [![Maintainability](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/maintainability)](https://codeclimate.com/github/kengz/SLM-Lab/maintainability) [![Test Coverage](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/test_coverage)](https://codeclimate.com/github/kengz/SLM-Lab/test_coverage)

Modular Deep Reinforcement Learning framework in PyTorch.

**[Github Repo](https://github.com/kengz/SLM-Lab) | [Lab Documentation](https://kengz.gitbooks.io/slm-lab/content/) | [Experiment Log Book](https://lgraesser.gitbooks.io/slm-experiment-log/content/)**

## Features

SLM Lab is created for deep reinforcement learning research.

#### algorithms
- numerous canonical algorithms (list below)
- reusable modular components: algorithm, policy, network, memory
- ease and speed of building new algorithms
- clear and unified design; production-grade code

#### environments
- works with [OpenAI gym](https://github.com/openai/gym), [Unity environments](https://github.com/Unity-Technologies/ml-agents), and more soon
- supports multi-agents, multi-environments
- API for adding custom environments

#### experimentation
- scalable hyperparameter search with [ray](https://github.com/ray-project/ray)
- graphs and analytics
- fitness function for comparing experiments
- open science - [Log Book](https://lgraesser.gitbooks.io/slm-experiment-log/content/)

#### Baselines

The implemented baseline algorithms (besides research) are:
- SARSA
- DQN
- Double DQN
- REINFORCE
    - Option to add entropy to encourage exploration
- Actor-Critic
    - Batch or episodic training
    - Shared or separate actor and critic params
    - Advantage calculated using n-step returns or generalized advantage estimation
    - Option to add entropy to encourage exploration

#### Feature Demos

![dqn cartpole ball2d](https://media.giphy.com/media/l0DAIymuiMS3HyW9G/giphy.gif)
>A multitask agent solving both OpenAI Cartpole-v0 and Unity Ball2D.

![](https://kengz.gitbooks.io/slm-lab/content/assets/Reinforce_CartPole-v0_experiment_graph.png)
>Experiment graph showing fitness from hyperparameter search.

![](https://kengz.gitbooks.io/slm-lab/content/assets/Reinforce_CartPole-v0_t150_s0_session_graph.png)
>Example total reward and loss graph from a session.

Read on for tutorials, research and results.

**[Github Repo](https://github.com/kengz/SLM-Lab) | [Lab Documentation](https://kengz.gitbooks.io/slm-lab/content/) | [Experiment Log Book](https://lgraesser.gitbooks.io/slm-experiment-log/content/)**

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

> Alternatively, run the content of `bin/setup_macOS` or `bin/setup_ubuntu` on your terminal manually.

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
- for plots, sign up for a free [Plotly account](https://plot.ly/) and update the `"plotly"` JSON key.

### Update

To update SLM Lab, pull the latest git commits and run update:

```shell
git pull
yarn update
```

### Demo

Run the demo to quickly see the lab in action (and to test your installation).

![](https://kengz.gitbooks.io/slm-lab/content/assets/demo.png)

It is `ActorCritic` in `CartPole-v0`:

1. see `slm_lab/spec/demo.json` for example spec:
    ```json
    "actor_critic_cartpole": {
      "agent": [{
        "name": "ActorCritic",
        "algorithm": {
        "name": "ActorCritic",
        "action_policy": "default",
        "gamma": 0.91,
        ...
        }
      }]
    }
    ```

2. see `config/experiments.json` to schedule experiments:
    ```json
    "demo.json": {
      "actor_critic_cartpole": "train"
    }
    ```

3. launch terminal in the repo directory, run the lab:
    ```shell
    source activate lab
    yarn start
    ```

4. This demo will run a single trial using the default parameters, and render the environment. After completion, check the output for data `data/actor_critic_cartpole/`. You should see a healthy session graph.

5. Next, change the run mode from `"train"` to `"search"`  `config/experiments.json`, and rerun. This runs experiments of multiple trials with hyperparameter search. Environments will not be rendered.:
    ```json
    "demo.json": {
      "actor_critic_cartpole": "search"
    }
    ```

>If the quick start fails, consult [Debugging](https://kengz.gitbooks.io/slm-lab/content/debugging.html).

Now the lab is ready for usage.

**Read on: [Github Repo](https://github.com/kengz/SLM-Lab) | [Lab Documentation](https://kengz.gitbooks.io/slm-lab/content/) | [Experiment Log Book](https://lgraesser.gitbooks.io/slm-experiment-log/content/)**

## Citing

If you use `SLM-Lab` in your research, you can cite it as follows:

```
@misc{kenggraesser2017slmlab,
    author = {Wah Loon Keng, Laura Graesser},
    title = {SLM-Lab},
    year = {2017},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/kengz/SLM-Lab}},
}
```

## Contributing

If you're interested in using the lab for research, teaching or applications, please contact the [authors](https://twitter.com/kengzwl).
