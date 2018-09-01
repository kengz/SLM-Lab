# SLM Lab
[![CircleCI](https://circleci.com/gh/kengz/SLM-Lab.svg?style=shield)](https://circleci.com/gh/kengz/SLM-Lab) [![Maintainability](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/maintainability)](https://codeclimate.com/github/kengz/SLM-Lab/maintainability) [![Test Coverage](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/test_coverage)](https://codeclimate.com/github/kengz/SLM-Lab/test_coverage)

Modular Deep Reinforcement Learning framework in PyTorch.

**[Github](https://github.com/kengz/SLM-Lab) | [Documentation](https://kengz.gitbooks.io/slm-lab/content/) | [Experiment Log](https://lgraesser.gitbooks.io/slm-experiment-log/content/)**

![dqn cartpole ball2d](https://media.giphy.com/media/l0DAIymuiMS3HyW9G/giphy.gif)
>A multitask agent solving both OpenAI Cartpole-v0 and Unity Ball2D.

## Features

SLM Lab is created for deep reinforcement learning research.

#### Algorithms
- numerous canonical algorithms (list below)
- reusable modular components: algorithm, policy, network, memory
- ease and speed of building new algorithms
- clear and unified design; production-grade code

#### Environments
- works with [OpenAI gym](https://github.com/openai/gym), [Unity environments](https://github.com/Unity-Technologies/ml-agents), and more soon
- supports multi-agents, multi-environments
- API for adding custom environments

#### Experimentation
- scalable hyperparameter search with [ray](https://github.com/ray-project/ray)
- graphs and analytics
- fitness function for comparing experiments
- open science - [Log Book](https://lgraesser.gitbooks.io/slm-experiment-log/content/)

## Implementations

SLM Lab implements most of the recent canonical algorithms and various extensions. These are used as the base of research.

#### Algorithm

code: [slm_lab/agent/algorithm](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/agent/algorithm)

Various algorithms are in fact extensions of some simpler ones, and they are implemented as such. This makes the code very concise.

**Policy Gradient:**
- REINFORCE
- AC (Vanilla Actor-Critic)
    - shared or separate actor critic networks
    - plain TD
    - entropy term control
- A2C (Advantage Actor-Critic)
    - extension of AC with with advantage function
    - N-step returns as advantage
    - GAE (Generalized Advantage Estimate) as advantage
- PPO (Proximal Policy Optimization)
    - extension of A2C with PPO loss function
- SIL (Self-Imitation Learning)
    - extension of A2C with off-policy training on custom loss
- PPOSIL
    - SIL with PPO instead of A2C

Using the lab's unified API, **all the algorithms be ran in a distributed fashion**, with Session (DistSession in fact) taking the role of workers under a Trial. Some of the distributed algorithms have their own name:

- A3C (Asynchronous A2C / distributed A2C)
- DPPO (Distributed PPO)

**Value-based:**
- SARSA
- DQN (Deep Q Learning)
    - boltzmann or epsilon-greedy policy
- DRQN (Recurrent DQN)
- Dueling DQN
- DDQN (Double DQN)
- DDRQN
- Dueling DDQN
- Multitask DQN (multi-environment DQN)
- Hydra DQN (multi-environment DQN)

As mentioned above, **all these algorithms can be turned into distributed algorithms too**, altough we do not have special names for them.

Below are the modular building blocks for the algorithms. They are designed to be general, and are reused extensively.

#### Memory

code: [slm_lab/agent/memory](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/agent/memory)

For on-policy algorithms (policy gradient):
- OnPolicyReplay
- OnPolicySeqReplay
- OnPolicyBatchReplay
- OnPolicyBatchSeqReplay

For off-policy algorithms (value-based)
- Replay
- SeqReplay
- SILReplay (special Replay for SIL)
- SILSeqReplay (special SeqReplay for SIL)
- ConcatReplay
- AtariReplay
- PrioritizedReplay

#### Neural Network

code: [slm_lab/agent/net](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/agent/net)

These networks are usable for all algorithms.

- MLPNet (Multi Layer Perceptron, with multi-tails support)
- HydraMLPNet (multi-heads, multi-tails)
- RecurrentNet (with multi-tails support)
- ConvNet (with multi-tails support)

These networks are usable for Q-learning algorithms. For more details see [this paper](http://proceedings.mlr.press/v48/wangf16.pdf).

- DuelingMLPNet
- DuelingConvNet

#### Policy

code: [slm_lab/agent/algorithm/policy_util.py](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/agent/algorithm/policy_util.py)

- different probability distributions for sampling actions
- default policy
- Boltzmann policy
- Epsilon-greedy policy
- numerous rate decay methods

## Experimentation framework

Deep Reinforcement Learning is very empirical. The systematic building blocks above need a experimentation framework to study variables systematically. The experiment framework completes the lab.

![](https://kengz.gitbooks.io/slm-lab/content/assets/demo_experiment_graph.png)
>Experiment graph summarizing the trials in hyperparameter search.

![](https://kengz.gitbooks.io/slm-lab/content/assets/demo_trial_graph.png)
>Trial graph showing average envelope of repeated sessions.

![](https://kengz.gitbooks.io/slm-lab/content/assets/demo_session_graph.png)
>Session graph showing total rewards, exploration variable and loss for the episodes.

Read on for tutorials, research and results.

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

>Alternatively, run the content of `bin/setup_macOS` or `bin/setup_ubuntu` on your terminal manually.
>Docker image and Dockerfile with instructions are also available

### Setup

A config file `config/default.json` will be created.

```json
{
  "data_sync_dir": "~/Dropbox/SLM-Lab/data"
}
```

- update `"data_sync_dir"` if you run lab on remote and want to sync data for easy access; it will copy `data/` there.

### Update

To update SLM Lab, pull the latest git commits and run update:

```shell
git pull
yarn update
```

### Demo

Run the demo to quickly see the lab in action (and to test your installation).

![](https://kengz.gitbooks.io/slm-lab/content/assets/demo.png)

It is `DQN` in `CartPole-v0`:

1. see `slm_lab/spec/demo.json` for example spec:
    ```json
    "dqn_cartpole": {
      "agent": [{
        "name": "DQN",
        "algorithm": {
          "name": "DQN",
          "action_pdtype": "Argmax",
          "action_policy": "epsilon_greedy",
        ...
        }
      }]
    }
    ```

2. see `config/experiments.json` to schedule experiments:
    ```json
    "demo.json": {
      "dqn_cartpole": "dev"
    }
    ```
    > To run faster, change lab mode from "dev" to "train" above and rendering will be disabled.

3. launch terminal in the repo directory, run the lab:
    ```shell
    source activate lab
    yarn start
    ```

4. This demo will run a single trial using the default parameters, and render the environment. After completion, check the output for data `data/dqn_cartpole_2018_06_16_214527/` (timestamp will differ). You should see some healthy graphs.

    ![](https://kengz.gitbooks.io/slm-lab/content/assets/demo_trial_graph.png)
    >Trial graph showing average envelope of repeated sessions.

    ![](https://kengz.gitbooks.io/slm-lab/content/assets/demo_session_graph.png)
    >Session graph showing total rewards, exploration variable and loss for the episodes.

5. Enjoy mode - when a session ends, a model file will automatically save. You can find the session `prepath` that ends in its trial and session numbers. The example above is trial 1 session 0, and you can see a pyotrch model saved at `data/dqn_cartpole_2018_06_16_214527/dqn_cartpole_t1_s0_model_net.pth`. Use the prepath at `config/experiments.json` to run eval mode:
    ```json
    "demo.json": {
      "dqn_cartpole": "eval@data/dqn_cartpole_2018_06_16_214527/dqn_cartpole_t1_s0"
    }
    ```
    >enjoy mode will automatically disable learning and exploration. Graphs will still save.

6. Next, change the run mode from `"train"` to `"search"`  `config/experiments.json`, and rerun. This runs experiments of multiple trials with hyperparameter search. Environments will not be rendered.:
    ```json
    "demo.json": {
      "dqn_cartpole": "search"
    }
    ```
    When it ends, refer to `{prepath}_experiment_graph.png` and `{prepath}_experiment_df.csv` to find the best trials.

>If the demo fails, consult [Debugging](https://kengz.gitbooks.io/slm-lab/content/debugging.html).

Now the lab is ready for usage.

**Read on: [Github](https://github.com/kengz/SLM-Lab) | [Documentation](https://kengz.gitbooks.io/slm-lab/content/) | [Experiment Log](https://lgraesser.gitbooks.io/slm-experiment-log/content/)**

## Citing

If you use `SLM-Lab` in your research, please cite below:

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

SLM Lab is an MIT-licensed open source project. Contributions are very much welcome, no matter if it's a quick bug-fix or new feature addition.

If you have an idea for a new algorithm, environment support, analytics, benchmarking, or new experiment design, let us know.

If you're interested in using the lab for **research, teaching or applications**, please contact the [authors](https://twitter.com/kengzwl).
