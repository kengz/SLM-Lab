# SLM Lab
[![CircleCI](https://circleci.com/gh/kengz/SLM-Lab.svg?style=shield)](https://circleci.com/gh/kengz/SLM-Lab) [![Maintainability](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/maintainability)](https://codeclimate.com/github/kengz/SLM-Lab/maintainability) [![Test Coverage](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/test_coverage)](https://codeclimate.com/github/kengz/SLM-Lab/test_coverage)

Modular Deep Reinforcement Learning framework in PyTorch.

||||
|:---:|:---:|:---:|
| ![ddqn_beamrider](https://user-images.githubusercontent.com/8209263/49688812-b7e04200-facc-11e8-9a1a-d5c8e512f26c.gif) |  ![ddqn_breakout](https://user-images.githubusercontent.com/8209263/49688819-c29ad700-facc-11e8-842b-1dc6f6f38495.gif) |![ddqn_pong](https://user-images.githubusercontent.com/8209263/49688793-54eeab00-facc-11e8-80fe-4b76a12180a0.gif) |
| BeamRider | Breakout | Pong |
| ![ddqn_qbert](https://user-images.githubusercontent.com/8209263/49688862-6be1cd00-facd-11e8-849d-61aef598611b.gif) | ![ddqn_seaquest](https://user-images.githubusercontent.com/8209263/49688863-70a68100-facd-11e8-9303-73bea9b9987a.gif) | ![ddqn_spaceinvaders](https://user-images.githubusercontent.com/8209263/49688875-87e56e80-facd-11e8-90be-9d6be7bace03.gif) |
| Qbert | Seaquest | SpaceInvaders |


| References | |
|------------|--|
| [Installation](#installation) | How to install SLM Lab |
| [Documentation](https://kengz.gitbooks.io/slm-lab/content/) | Usage documentation |
| [Benchmark](https://github.com/kengz/SLM-Lab/blob/master/BENCHMARK.md)| Benchmark results |
| [Gitter](https://gitter.im/SLM-Lab/SLM-Lab) | SLM Lab user chatroom |


## Features

### [Algorithms](#algorithms)

SLM Lab implements a number of canonical RL [algorithms](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/agent/algorithm) with reusable **modular components** and *class-inheritance*, with commitment to code quality and performance.

The benchmark results also include complete [spec files](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark) to enable full **reproducibility** using SLM Lab.

Below shows the latest benchmark status. See [benchmark results here](https://github.com/kengz/SLM-Lab/blob/master/BENCHMARK.md).

| **Algorithm\Benchmark** | Atari | Roboschool |
|-------------------------|-------|------------|
| SARSA                   | -     |            |
| DQN, distributed-DQN    | :white_check_mark: |            |
| Double-DQN, PER-DQN     | :white_check_mark: |            |
| REINFORCE               | -     |            |
| A2C, A3C (N-step & GAE) | :white_check_mark: |            |
| PPO, distributed-PPO    | :white_check_mark: |            |
| SIL (A2C, PPO)          |       |            |

### [Environments](#environments)

SLM Lab integrates with multiple environment offerings:
  - [OpenAI gym](https://github.com/openai/gym)
  - [OpenAI Roboschool](https://github.com/openai/roboschool)
  - [VizDoom](https://github.com/mwydmuch/ViZDoom#documentation) (credit: joelouismarino)
  - [Unity environments](https://github.com/Unity-Technologies/ml-agents) with prebuilt binaries

*Contributions are welcome to integrate more environments!*

### [Metrics and Experimentation](#experimentation-framework)

To facilitate better RL development, SLM Lab also comes with prebuilt *metrics* and *experimentation framework*:
- every run generates metrics, graphs and data for analysis, as well as spec for reproducibility
- scalable hyperparameter search using [Ray tune](https://ray.readthedocs.io/en/latest/tune.html)


## Installation

1. Clone the [SLM Lab repo](https://github.com/kengz/SLM-Lab):
    ```shell
    git clone https://github.com/kengz/SLM-Lab.git
    ```

2. Install dependencies (this uses Conda for optimality):
    ```shell
    cd SLM-Lab/
    bin/setup
    ```

  >Alternatively, instead of `bin/setup`, copy-paste from [`bin/setup_macOS` or `bin/setup_ubuntu`](https://github.com/kengz/SLM-Lab/tree/master/bin) into your terminal to install manually.

  >Useful reference: [Debugging](https://kengz.gitbooks.io/slm-lab/content/installation/debugging.html)

## Quick Start

#### DQN CartPole

Everything in the lab is ran using a `spec file`, which contains all the information for the run to be reproducible. These are located in `slm_lab/spec/`.

Run a quick demo of DQN and CartPole:

```shell
conda activate lab
python run_lab.py slm_lab/spec/demo.json dqn_cartpole dev
```

This will launch a `Trial` in *development mode*, which enables verbose logging and environment rendering. An example screenshot is shown below.

![](https://kengz.gitbooks.io/slm-lab/content/assets/demo.png)

Next, run it in training mode. The `total_reward` should converge to 200 within a few minutes.

```shell
python run_lab.py slm_lab/spec/demo.json dqn_cartpole train
```

>Tip: All lab command should be ran from within a Conda environment. Run `conda activate lab` once at the beginning of a new terminal session.

This will run a new `Trial` in *training mode*. At the end of it, all the metrics and graphs will be output to the `data/` folder.

![](https://kengz.gitbooks.io/slm-lab/content/assets/demo_training.png)


#### A2C Atari

Run A2C to solve Atari Pong:

```shell
conda activate lab
python run_lab.py slm_lab/spec/experimental/a2c/a2c_pong.json a2c_pong train
```

![](https://kengz.gitbooks.io/slm-lab/content/assets/demo_atari.png)
>Atari Pong ran with `dev` mode to render the environment

This will run a `Trial` with multiple Sessions in *training mode*. In the beginning, the `total_reward` should be around -21. After about 1 million frames, it should begin to converge to around +21 (perfect score). At the end of it, all the metrics and graphs will be output to the `data/` folder.

Below shows a trial graph with multiple sessions:

![](https://kengz.gitbooks.io/slm-lab/content/assets/demo_atari_graph.png)

#### Benchmark

To run a full benchmark, simply pick a file and run it in train mode. For example, for A2C Atari benchmark, the spec file is `slm_lab/spec/benchmark/a2c/a2c_atari.json`. This file is parametrized to run on a set of environments. Run the benchmark:

```shell
python run_lab.py slm_lab/spec/benchmark/a2c/a2c_atari.json a2c_atari train
```

This will spawn multiple processes to run each environment in its separate `Trial`, and the data is saved to `data/` as usual.

#### Experimentation / Hyperparameter search

An [`Experiment`](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/experiment/control.py) is a hyperparameter search, which samples multiple `spec`s from a search space. `Experiment` spawns a `Trial` for each `spec`, and each `Trial` runs multiple duplicated `Session`s for averaging its results.

Given a spec file in `slm_lab/spec/`, if it has a `search` field defining a search space, then it can be ran as an Experiment. For example,

```shell
python run_lab.py slm_lab/spec/demo.json dqn_cartpole search
```

Deep Reinforcement Learning is highly empirical. The lab enables rapid and massive experimentations, hence it needs a way to quickly analyze data from many trials. The experiment  and analytics framework is the scientific method of the lab.

![](https://kengz.gitbooks.io/slm-lab/content/assets/demo_experiment_graph.png)
>Experiment graph summarizing the trials in hyperparameter search.

![](https://kengz.gitbooks.io/slm-lab/content/assets/demo_trial_graph.png)
>Trial graph showing average envelope of repeated sessions.

![](https://kengz.gitbooks.io/slm-lab/content/assets/demo_session_graph.png)
>Session graph showing total rewards.

This is the end of the quick start tutorial. Continue reading the full documentation to start using SLM Lab.

**Read on: [Github](https://github.com/kengz/SLM-Lab) | [Documentation](https://kengz.gitbooks.io/slm-lab/content/)**

## Design Principles

SLM Lab is created for deep reinforcement learning research and applications. The design was guided by four principles
- **modularity**
- **simplicity**
- **analytical clarity**
- **reproducibility**

#### Modularity

- makes research easier and more accessible: reuse well-tested components and only focus on the relevant work
- makes learning deep RL easier: the algorithms are complex; SLM Lab breaks them down into more manageable, digestible components
- components get reused maximally, which means less code, more tests, and fewer bugs

#### Simplicity

- the components are designed to closely correspond to the way papers or books discuss RL
- modular libraries are not necessarily simple. Simplicity balances modularity to prevent overly complex abstractions that are difficult to understand and use

#### Analytical clarity

- hyperparameter search results are automatically analyzed and presented hierarchically in increasingly granular detail
- it should take less than 1 minute to understand if an experiment yielded a successful result using the [experiment graph](https://kengz.gitbooks.io/slm-lab/content/analytics/experiment-graph.html)
- it should take less than 5 minutes to find and review the top 3 parameter settings using the [trial](https://kengz.gitbooks.io/slm-lab/content/analytics/trial-graph.html) and [session](https://kengz.gitbooks.io/slm-lab/content/analytics/session-graph.html) graphs

#### Reproducibility

- only the spec file and a git SHA are needed to fully reproduce an experiment
- all the results are recorded in [BENCHMARK.md](https://github.com/kengz/SLM-Lab/blob/master/BENCHMARK.md)
- experiment reproduction instructions are submitted to the Lab via [`result` Pull Requests](https://github.com/kengz/SLM-Lab/pulls?utf8=%E2%9C%93&q=is%3Apr+label%3Aresult+)
- the full experiment datas contributed are [public on Dropbox ](https://www.dropbox.com/sh/y738zvzj3nxthn1/AAAg1e6TxXVf3krD81TD5V0Ra?dl=0)

## Citing

If you use `SLM Lab` in your research, please cite below:

```
@misc{kenggraesser2017slmlab,
    author = {Wah Loon Keng, Laura Graesser},
    title = {SLM Lab},
    year = {2017},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/kengz/SLM-Lab}},
}
```

## Contributing

SLM Lab is an MIT-licensed open source project. Contributions are very much welcome, no matter if it's a quick bug-fix or new feature addition. Please see [CONTRIBUTING.md](https://github.com/kengz/SLM-Lab/blob/master/CONTRIBUTING.md) for more info.

If you have an idea for a new algorithm, environment support, analytics, benchmarking, or new experiment design, let us know.

If you're interested in using the lab for **research, teaching or applications**, please contact the [authors](https://twitter.com/kengzwl).
