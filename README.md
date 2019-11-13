# SLM Lab
![GitHub tag (latest SemVer)](https://img.shields.io/github/tag/kengz/slm-lab) [![CircleCI](https://circleci.com/gh/kengz/SLM-Lab.svg?style=shield)](https://circleci.com/gh/kengz/SLM-Lab) [![Maintainability](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/maintainability)](https://codeclimate.com/github/kengz/SLM-Lab/maintainability) [![Test Coverage](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/test_coverage)](https://codeclimate.com/github/kengz/SLM-Lab/test_coverage)

Modular Deep Reinforcement Learning framework in PyTorch.

|||||
|:---:|:---:|:---:|:---:|
| ![ppo beamrider](https://user-images.githubusercontent.com/8209263/63994698-689ecf00-caaa-11e9-991f-0a5e9c2f5804.gif) | ![ppo breakout](https://user-images.githubusercontent.com/8209263/63994695-650b4800-caaa-11e9-9982-2462738caa45.gif) | ![ppo kungfumaster](https://user-images.githubusercontent.com/8209263/63994690-60469400-caaa-11e9-9093-b1cd38cee5ae.gif) | ![ppo mspacman](https://user-images.githubusercontent.com/8209263/63994685-5cb30d00-caaa-11e9-8f35-78e29a7d60f5.gif) |
| BeamRider | Breakout | KungFuMaster | MsPacman |
| ![ppo pong](https://user-images.githubusercontent.com/8209263/63994680-59b81c80-caaa-11e9-9253-ed98370351cd.gif) | ![ppo qbert](https://user-images.githubusercontent.com/8209263/63994672-54f36880-caaa-11e9-9757-7780725b53af.gif) | ![ppo seaquest](https://user-images.githubusercontent.com/8209263/63994665-4dcc5a80-caaa-11e9-80bf-c21db818115b.gif) | ![ppo spaceinvaders](https://user-images.githubusercontent.com/8209263/63994624-15c51780-caaa-11e9-9c9a-854d3ce9066d.gif) |
| Pong | Qbert | Seaquest | Sp.Invaders |
| ![sac ant](https://user-images.githubusercontent.com/8209263/63994867-ff6b8b80-caaa-11e9-971e-2fac1cddcbac.gif) | ![sac halfcheetah](https://user-images.githubusercontent.com/8209263/63994869-01354f00-caab-11e9-8e11-3893d2c2419d.gif) | ![sac hopper](https://user-images.githubusercontent.com/8209263/63994871-0397a900-caab-11e9-9566-4ca23c54b2d4.gif) | ![sac humanoid](https://user-images.githubusercontent.com/8209263/63994883-0befe400-caab-11e9-9bcc-c30c885aad73.gif) |
| Ant | HalfCheetah | Hopper | Humanoid |
| ![sac doublependulum](https://user-images.githubusercontent.com/8209263/63994879-07c3c680-caab-11e9-974c-06cdd25bfd68.gif) | ![sac pendulum](https://user-images.githubusercontent.com/8209263/63994880-085c5d00-caab-11e9-850d-049401540e3b.gif) | ![sac reacher](https://user-images.githubusercontent.com/8209263/63994881-098d8a00-caab-11e9-8e19-a3b32d601b10.gif) | ![sac walker](https://user-images.githubusercontent.com/8209263/63994882-0abeb700-caab-11e9-9e19-b59dc5c43393.gif) |
| Inv.DoublePendulum | InvertedPendulum | Reacher | Walker |


| References | |
|------------|--|
| [Installation](#installation) | How to install SLM Lab |
| [Documentation](https://kengz.gitbooks.io/slm-lab/content/) | Usage documentation |
| [Benchmark](https://github.com/kengz/SLM-Lab/blob/master/BENCHMARK.md)| Benchmark results |
| [Gitter](https://gitter.im/SLM-Lab/SLM-Lab) | SLM Lab user chatroom |


## Features

### Algorithms

SLM Lab implements a number of canonical RL [algorithms](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/agent/algorithm) with reusable **modular components** and *class-inheritance*, with commitment to code quality and performance.

The benchmark results also include complete [spec files](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark) to enable full **reproducibility** using SLM Lab.

Below shows the latest benchmark status. See the full [benchmark results here](https://github.com/kengz/SLM-Lab/blob/master/BENCHMARK.md).

| **Algorithm\Benchmark** | Atari | Roboschool |
|-------------------------|-------|-------|
| SARSA                   | -     | - |
| DQN (Deep Q-Network)    | :white_check_mark: | - |
| Double-DQN, Dueling-DQN, PER | :white_check_mark: | - |
| REINFORCE               | -     | - |
| A2C with GAE & n-step (Advantage Actor-Critic) | :white_check_mark: | :white_check_mark: |
| PPO (Proximal Policy Optimization)   | :white_check_mark: | :white_check_mark: |
| SAC (Soft Actor-Critic) |       | :white_check_mark: |
| SIL (Self Imitation Learning)          |       |  |

Due to their standardized design, all the algorithms can be parallelized asynchronously using Hogwild. Hence, SLM Lab also includes A3C, distributed-DQN, distributed-PPO.


#### Discrete Benchmark

>See the full [benchmark results here](https://github.com/kengz/SLM-Lab/blob/master/BENCHMARK.md).

||||||||
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Env. \ Alg. | DQN | DDQN+PER | A2C (GAE) | A2C (n-step) | PPO | SAC |
| Breakout <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/68744935-84dee200-05aa-11ea-9086-12546f5aa606.png"></details> | 80.88 | 182 | 377 | 398 | **443** | 3.51* |
| Pong <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/68744936-85777880-05aa-11ea-8a6a-4c364a27ba81.png"></details> | 18.48 | 20.5 | 19.31 | 19.56 | **20.58** | 19.87* |
| Seaquest <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/68744937-85777880-05aa-11ea-9927-e300309d1e9c.png"></details> | 1185 | **4405** | 1070 | 1684 | 1715 | 171* |
| Qbert <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/68744939-85777880-05aa-11ea-825c-b1225a0539af.png"></details> | 5494 | 11426 | 12405 | **13590** | 13460 | 923* |
| LunarLander <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67737566-e7d85280-f9c8-11e9-8df8-39c1205c5308.png"></details> | 192 | 233 | 25.21 | 68.23 | 214 | **276** |
| UnityHallway <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/68744968-90caa400-05aa-11ea-9247-d4d81533965a.png"></details> | -0.32 | 0.27 | 0.08 | -0.96 | **0.73** | 0.01 |
| UnityPushBlock <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/68744969-90caa400-05aa-11ea-9b17-45fb852e2671.png"></details> | 4.88 | 4.93 | 4.68 | 4.93 | **4.97** | -0.70 |

>For the full Atari benchmark, see [Atari Benchmark](https://github.com/kengz/SLM-Lab/blob/benchmark/BENCHMARK.md#atari-benchmark)

#### Continuous Benchmark

||||||
|:---:|:---:|:---:|:---:|:---:|
| Env. \ Alg. | A2C (GAE) | A2C (n-step) | PPO | SAC |
| RoboschoolAnt <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67737923-1571cb80-f9ca-11e9-8f6b-b288fa19bff0.png"></details> | 787 | 1396 | 1843 | **2915** |
| RoboschoolAtlasForwardWalk <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67737924-1571cb80-f9ca-11e9-98ee-82c920dfbf44.png"></details> | 59.87 | 88.04 | 172 | **800** |
| RoboschoolHalfCheetah <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67737925-1571cb80-f9ca-11e9-9c7f-3a8294a517af.png"></details> | 712 | 439 | 1960 | **2497** |
| RoboschoolHopper <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67737926-160a6200-f9ca-11e9-8cae-9afc532e5af8.png"></details> | 710 | 285 | 2042 | **2045** |
| RoboschoolInvertedDoublePendulum <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67737927-160a6200-f9ca-11e9-8eb2-e04554e3844f.png"></details> | 996 | 4410 | 8076 | **8085** |
| RoboschoolInvertedPendulum <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67737928-160a6200-f9ca-11e9-8eae-e7a3ccbe914a.png"></details> | **995** | 978 | 986 | 941 |
| RoboschoolReacher <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67737929-160a6200-f9ca-11e9-9423-b27165def32e.png"></details> | 12.9 | 10.16 | 19.51 | **19.99** |
| RoboschoolWalker2d <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67737930-160a6200-f9ca-11e9-9a0f-edbd4f01f4e0.png"></details> | 280 | 220 | 1660 | **1894** |
| RoboschoolHumanoid <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67737931-16a2f880-f9ca-11e9-9340-fe90ab48e95f.png"></details> | 99.31 | 54.58 | 2388 | **2621*** |
| RoboschoolHumanoidFlagrun <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67737932-16a2f880-f9ca-11e9-92bb-9c896ec3991e.png"></details> | 73.57 | 178 | 2014 | **2056*** |
| RoboschoolHumanoidFlagrunHarder <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67737933-16a2f880-f9ca-11e9-98c8-7388fa9e1775.png"></details> | -429 | 253 | **680** | 280* |
| Unity3DBall <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67737934-16a2f880-f9ca-11e9-912b-37c8840d0acc.png"></details> | 33.48 | 53.46 | 78.24 | **98.44** |
| Unity3DBallHard <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67737935-16a2f880-f9ca-11e9-9275-f3b5fef22e1b.png"></details> | 62.92 | 71.92 | 91.41 | **97.06** |

### Environments

SLM Lab integrates with multiple environment offerings:
  - [OpenAI gym](https://github.com/openai/gym)
  - [OpenAI Roboschool](https://github.com/openai/roboschool)
  - [VizDoom](https://github.com/mwydmuch/ViZDoom#documentation) (credit: joelouismarino)
  - [Unity environments](https://github.com/Unity-Technologies/ml-agents) with prebuilt binaries

*Contributions are welcome to integrate more environments!*

### Metrics and Experimentation

To facilitate better RL development, SLM Lab also comes with prebuilt *metrics* and *experimentation framework*:
- every run generates metrics, TensorBoard summaries, graphs and data for analysis, as well as spec for reproducibility
- scalable hyperparameter search using [Ray tune](https://ray.readthedocs.io/en/latest/tune.html)


## Installation

1. Clone the [SLM Lab repo](https://github.com/kengz/SLM-Lab):
    ```shell
    git clone https://github.com/kengz/SLM-Lab.git
    ```

2. Install dependencies (this uses Conda for optimality):
    ```shell
    cd SLM-Lab/
    ./bin/setup
    ```

  >Alternatively, instead of running `./bin/setup`, copy-paste from [`bin/setup_macOS` or `bin/setup_ubuntu`](https://github.com/kengz/SLM-Lab/tree/master/bin) into your terminal and add `sudo` accordingly to run the installation commands.

  >Useful reference: [Debugging](https://kengz.gitbooks.io/slm-lab/content/installation/debugging.html)

#### Hardware Requirements

Non-image based environments can run on a laptop. Only image based environments such as the Atari games benefit from a GPU speedup. For these, we recommend 1 GPU and at least 4 CPUs. This can run a single Atari `Trial` consisting of 4 `Sessions`.

For desktop, a reference spec is GTX 1080 GPU, 4 CPUs above 3.0 GHz, and 32 Gb RAM.

For cloud computing, start with an affordable instance of [AWS EC2 `p2.xlarge`](https://aws.amazon.com/ec2/instance-types/p2/) with a K80 GPU and 4 CPUs. Use the Deep Learning AMI with Conda when [creating an instance](https://aws.amazon.com/getting-started/tutorials/get-started-dlami/).


## Quick Start

### DQN CartPole

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


### A2C Atari

Run A2C to solve Atari Pong:

```shell
conda activate lab
python run_lab.py slm_lab/spec/benchmark/a2c/a2c_gae_pong.json a2c_gae_pong train
```

>When running on a headless server, prepend a command with `xvfb-run -a`, for example `xvfb-run -a python run_lab.py slm_lab/spec/benchmark/a2c/a2c_gae_pong.json a2c_gae_pong train`

![](https://kengz.gitbooks.io/slm-lab/content/assets/demo_atari.png)
>Atari Pong ran with `dev` mode to render the environment

This will run a `Trial` with multiple Sessions in *training mode*. In the beginning, the `total_reward` should be around -21. After about 1 million frames, it should begin to converge to around +21 (perfect score). At the end of it, all the metrics and graphs will be output to the `data/` folder.

Below shows a trial graph with multiple sessions:

![](https://kengz.gitbooks.io/slm-lab/content/assets/a2c_gae_pong_t0_trial_graph_mean_returns_ma_vs_frames.png)

### TensorBoard

TensorBoard writer is initialized in `agent.body` along with other logging variables and methods. It will log summary variables, graph, network parameter histograms, and action histograms automatically during checkpoint logging. This allows for richer diagnosis of the network and policy, e.g. by seeing if the distributions shift over the course of learning.

Launch TensorBoard during/after a run for diagnosis.

```shell
tensorboard --log_dir=data
```

![](https://user-images.githubusercontent.com/8209263/66803221-d9bc0980-eed3-11e9-92b8-0e5cd42a6eab.png)


### Enjoy mode

Once a Trial completes with a good model saved into the `data/` folder, for example `data/a2c_gae_pong_2019_08_01_010727`, use the `enjoy` mode to show the trained agent playing the environment. Use the `enjoy@{prename}` mode to pick a saved trial-sesison, for example:

```shell
python run_lab.py data/a2c_gae_pong_2019_08_01_010727/a2c_gae_pong_spec.json a2c_gae_pong enjoy@a2c_gae_pong_t0_s0
```

### Benchmark

To run a full benchmark, simply pick a file and run it in train mode. For example, for A2C Atari benchmark, the spec file is `slm_lab/spec/benchmark/a2c/a2c_atari.json`. This file is parametrized to run on a set of environments. Run the benchmark:

```shell
python run_lab.py slm_lab/spec/benchmark/a2c/a2c_atari.json a2c_atari train
```

This will spawn multiple processes to run each environment in its separate `Trial`, and the data is saved to `data/` as usual. See the uploaded [benchmark results here](https://github.com/kengz/SLM-Lab/blob/master/BENCHMARK.md).

### Experimentation / Hyperparameter search

An [`Experiment`](https://github.com/kengz/SLM-Lab/blob/master/slm_lab/experiment/control.py) is a hyperparameter search, which samples multiple `spec`s from a search space. `Experiment` spawns a `Trial` for each `spec`, and each `Trial` runs multiple duplicated `Session`s for averaging its results.

Given a spec file in `slm_lab/spec/`, if it has a `search` field defining a search space, then it can be ran as an Experiment. For example,

```shell
python run_lab.py slm_lab/spec/experimental/ppo/ppo_lam_search.json ppo_breakout search
```

Deep Reinforcement Learning is highly empirical. The lab enables rapid and massive experimentations, hence it needs a way to quickly analyze data from many trials. The experiment  and analytics framework is the scientific method of the lab.

|||
|:---:|:---:|
| ![](https://kengz.gitbooks.io/slm-lab/content/assets/ppo_breakout_experiment_graph1.png) | ![](https://kengz.gitbooks.io/slm-lab/content/assets/ppo_breakout_experiment_graph2.png) |
| Experiment graph | Experiment graph |
>Segments of the experiment graph summarizing the trials in hyperparameter search.

|||
|:---:|:---:|
| ![](https://kengz.gitbooks.io/slm-lab/content/assets/ppo_breakout_multi_trial_graph_mean_returns_vs_frames.png) | ![](https://kengz.gitbooks.io/slm-lab/content/assets/ppo_breakout_multi_trial_graph_mean_returns_ma_vs_frames.png)|
| Multi-trial graph | with moving average |
>The multi-trial experiment graph and its moving average version comparing the trials. These graph show the effect of different GAE λ values of PPO on the Breakout environment. λ= 0.70 performs the best, while λ values closer to 0.90 do not perform as well.

|||
|:---:|:---:|
| ![](https://kengz.gitbooks.io/slm-lab/content/assets/ppo_breakout_t1_trial_graph_mean_returns_vs_frames.png) | ![](https://kengz.gitbooks.io/slm-lab/content/assets/ppo_breakout_t1_trial_graph_mean_returns_ma_vs_frames.png)|
| Trial graph | with moving average |
>A trial graph showing average from repeated sessions, and its moving average version.

|||
|:---:|:---:|
| ![](https://kengz.gitbooks.io/slm-lab/content/assets/ppo_breakout_t1_s0_session_graph_eval_mean_returns_vs_frames.png) | ![](https://kengz.gitbooks.io/slm-lab/content/assets/ppo_breakout_t1_s0_session_graph_eval_mean_returns_ma_vs_frames.png)|
| Session graph | with moving average |
>A session graph showing the total rewards and its moving average version.

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
