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

#### Atari benchmark

>See the full [benchmark results here](https://github.com/kengz/SLM-Lab/blob/master/BENCHMARK.md).

>Click on the algorithm to see the result upload Pull Request.

| Env. \ Alg. | [DQN](https://github.com/kengz/SLM-Lab/pull/396) | [DDQN+PER](https://github.com/kengz/SLM-Lab/pull/396) | [A2C (GAE)](https://github.com/kengz/SLM-Lab/pull/396) | [A2C (n-step)](https://github.com/kengz/SLM-Lab/pull/396) | [PPO](https://github.com/kengz/SLM-Lab/pull/396) |
|:---|---|---|---|---|---|
| Breakout <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62232119-554cf680-b37a-11e9-9059-3e49bbb799d2.png"><img src="https://user-images.githubusercontent.com/8209263/62232118-554cf680-b37a-11e9-9d5b-dd2ddf527305.png"></details> | **425.89** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62067085-c0b28f00-b1e7-11e9-9dd5-c52b6104878f.png"></details> | 65.04 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62100441-9ba13900-b246-11e9-9373-95c6063915ab.png"></details> | 181.72 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62230967-dd7dcc80-b377-11e9-965b-60a9f3d5a7a1.png"></details> | 389.99 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62019989-0171c000-b176-11e9-94da-017b146afe65.png"></details> | 391.32 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020340-6c6fc680-b177-11e9-8aa1-9ac5c2001783.png"></details> |
| Pong <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62232135-5b42d780-b37a-11e9-9454-ff2d109ef4f4.png"><img src="https://user-images.githubusercontent.com/8209263/62232134-5b42d780-b37a-11e9-892f-a84ea8881e78.png"></details> | 20.09 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62067100-c6a87000-b1e7-11e9-919e-ad68e4166213.png"></details> | 18.34 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62100450-9fcd5680-b246-11e9-8170-2ad4473e8294.png"></details> | **20.44** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62230975-e2428080-b377-11e9-8970-6917ae80c0b4.png"></details> | 20.04 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020247-10a53d80-b177-11e9-9f0d-1433d4d87210.png"></details> | 19.66 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020342-6f6ab700-b177-11e9-824e-75f431dc14ec.png"></details> |
| Qbert <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62232149-60078b80-b37a-11e9-99bb-cedc9fe064d5.png"><img src="https://user-images.githubusercontent.com/8209263/62232148-60078b80-b37a-11e9-9610-17ac447a479f.png"></details> | 4,787.79 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62100455-a4920a80-b246-11e9-8ca5-d4dc1ce3d76f.png"></details> | 11,673.52 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62230986-e79fcb00-b377-11e9-8861-3686954b7e1a.png"></details> | 13,328.32 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020263-261a6780-b177-11e9-8936-22a74d2405d3.png"></details> | 13,259.19 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020347-742f6b00-b177-11e9-8bfb-edfcfd44c8b7.png"></details> | **13,691.89** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62067104-cb6d2400-b1e7-11e9-9c4f-9eaac265d7d6.png"></details> |
| Seaquest <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62232168-6bf34d80-b37a-11e9-9564-fa3609dc5c75.png"><img src="https://user-images.githubusercontent.com/8209263/62232167-6bf34d80-b37a-11e9-8db3-c79a0e78292b.png"></details> | 1,118.50 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62100462-a9ef5500-b246-11e9-8699-9356ff81ff93.png"></details> | **3,751.34** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62230991-ebcbe880-b377-11e9-8de4-a01379d1d61c.png"></details> | 892.68 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020266-29adee80-b177-11e9-83c2-fafbdbb982b9.png"></details> | 1,686.08 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020350-772a5b80-b177-11e9-8917-e3c8a745cd08.png"></details> | 1,583.04 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62067113-cf994180-b1e7-11e9-870b-b9bba71f2a7e.png"></details> |


#### Roboschool Benchmark

|||||
|:---:|:---:|:---:|:---:|
| ![sac ant](https://user-images.githubusercontent.com/8209263/63994867-ff6b8b80-caaa-11e9-971e-2fac1cddcbac.gif) | ![sac halfcheetah](https://user-images.githubusercontent.com/8209263/63994869-01354f00-caab-11e9-8e11-3893d2c2419d.gif) | ![sac hopper](https://user-images.githubusercontent.com/8209263/63994871-0397a900-caab-11e9-9566-4ca23c54b2d4.gif) | ![sac humanoid](https://user-images.githubusercontent.com/8209263/63994883-0befe400-caab-11e9-9bcc-c30c885aad73.gif) |
| Ant | HalfCheetah | Hopper | Humanoid |
| ![sac doublependulum](https://user-images.githubusercontent.com/8209263/63994879-07c3c680-caab-11e9-974c-06cdd25bfd68.gif) | ![sac pendulum](https://user-images.githubusercontent.com/8209263/63994880-085c5d00-caab-11e9-850d-049401540e3b.gif) | ![sac reacher](https://user-images.githubusercontent.com/8209263/63994881-098d8a00-caab-11e9-8e19-a3b32d601b10.gif) | ![sac walker](https://user-images.githubusercontent.com/8209263/63994882-0abeb700-caab-11e9-9e19-b59dc5c43393.gif) |
| Inv.DoublePendulum | InvertedPendulum | Reacher | Walker |

[Roboschool](https://github.com/openai/roboschool) by OpenAI offers free open source robotics simulations with improved physics. Although it mirrors the environments from MuJuCo, its environments' rewards are different.

>See the full [benchmark results here](https://github.com/kengz/SLM-Lab/blob/master/BENCHMARK.md).

>Click on the algorithm to see the result upload Pull Request.

| Env. \ Alg. | [A2C (GAE)](https://github.com/kengz/SLM-Lab/pull/416) | [A2C (n-step)](https://github.com/kengz/SLM-Lab/pull/416) | [PPO](https://github.com/kengz/SLM-Lab/pull/416) | [SAC](https://github.com/kengz/SLM-Lab/pull/416) |
|:---|---|---|---|---|
| RoboschoolAnt <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/65396960-d0220480-dd60-11e9-94ed-b6be9310a3e8.png"></details> | 787 | 1396 | 1843 | **2915** |
| RoboschoolAtlasForwardWalk <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/65396961-d0220480-dd60-11e9-8c0c-bc60a2c80f80.png"></details> | 59.87 | 88.04 | 172 | **800** |
| RoboschoolHalfCheetah <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/65396962-d0ba9b00-dd60-11e9-904d-4592f81a1432.png"></details> | 712 | 439 | 1960 | **2497** |
| RoboschoolHopper <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/65396963-d0ba9b00-dd60-11e9-9e8c-87e442d61dd4.png"></details> | 710 | 285 | 2042 | **2045** |
| RoboschoolInvertedDoublePendulum <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/65396967-d0ba9b00-dd60-11e9-9f30-514aee1fb88f.png"></details> | 996 | 4410 | 8076 | **8085** |
| RoboschoolInvertedPendulum <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/65396968-d1533180-dd60-11e9-8dba-c838a58b9fd3.png"></details> | **995** | 978 | 986 | 941 |
| RoboschoolReacher <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/65396970-d1533180-dd60-11e9-9f5b-e57502b1b094.png"></details> | 12.9 | 10.16 | 19.51 | **19.99** |
| RoboschoolWalker2d <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/65396971-d1533180-dd60-11e9-9c07-b83215f19a20.png"></details> | 280 | 220 | 1660 | **1894** |

Humanoid environments are significantly harder. Note that due to the number of frames required, we could only run Async-SAC.

| Env. \ Alg. | [A2C (GAE)](https://github.com/kengz/SLM-Lab/pull/416) | [A2C (n-step)](https://github.com/kengz/SLM-Lab/pull/416) | [PPO](https://github.com/kengz/SLM-Lab/pull/416) | [Async-SAC](https://github.com/kengz/SLM-Lab/pull/416) |
|:---|---|---|---|---|
| RoboschoolHumanoid <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/65396964-d0ba9b00-dd60-11e9-9177-9bad7329ba48.png"></details> | 99.31 | 54.58 | 2388 | **2621** |
| RoboschoolHumanoidFlagrun <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/65396965-d0ba9b00-dd60-11e9-8796-cc2f0efe0818.png"></details> | 73.57 | 178 | 2014 | **2056** |
| RoboschoolHumanoidFlagrunHarder <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/65396966-d0ba9b00-dd60-11e9-8814-1d8c06fb0527.png"></details> | -429 | 253 | **680** | 280 |


### Environments

SLM Lab integrates with multiple environment offerings:
  - [OpenAI gym](https://github.com/openai/gym)
  - [OpenAI Roboschool](https://github.com/openai/roboschool)
  - [VizDoom](https://github.com/mwydmuch/ViZDoom#documentation) (credit: joelouismarino)
  - [Unity environments](https://github.com/Unity-Technologies/ml-agents) with prebuilt binaries

*Contributions are welcome to integrate more environments!*

### Metrics and Experimentation

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
