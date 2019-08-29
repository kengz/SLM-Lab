# SLM Lab
![GitHub tag (latest SemVer)](https://img.shields.io/github/tag/kengz/slm-lab) [![CircleCI](https://circleci.com/gh/kengz/SLM-Lab.svg?style=shield)](https://circleci.com/gh/kengz/SLM-Lab) [![Maintainability](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/maintainability)](https://codeclimate.com/github/kengz/SLM-Lab/maintainability) [![Test Coverage](https://api.codeclimate.com/v1/badges/20c6a124c468b4d3e967/test_coverage)](https://codeclimate.com/github/kengz/SLM-Lab/test_coverage)

Modular Deep Reinforcement Learning framework in PyTorch.

|||||
|:---:|:---:|:---:|:---:|
| ![ddqn_breakout](https://user-images.githubusercontent.com/8209263/49688819-c29ad700-facc-11e8-842b-1dc6f6f38495.gif) |![ddqn_pong](https://user-images.githubusercontent.com/8209263/49688793-54eeab00-facc-11e8-80fe-4b76a12180a0.gif) | ![ddqn_qbert](https://user-images.githubusercontent.com/8209263/49688862-6be1cd00-facd-11e8-849d-61aef598611b.gif) | ![ddqn_spaceinvaders](https://user-images.githubusercontent.com/8209263/49688875-87e56e80-facd-11e8-90be-9d6be7bace03.gif) |
| Breakout | Pong | Qbert | Sp.Invaders |


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

This benchmark table is pulled from [PR 396](https://github.com/kengz/SLM-Lab/pull/396). See the full [benchmark results here](https://github.com/kengz/SLM-Lab/blob/master/BENCHMARK.md).

| Env. \ Alg. | A2C (GAE) | A2C (n-step) | PPO | DQN | DDQN+PER |
|:---|---|---|---|---|---|
| Breakout <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62232119-554cf680-b37a-11e9-9059-3e49bbb799d2.png"><img src="https://user-images.githubusercontent.com/8209263/62232118-554cf680-b37a-11e9-9d5b-dd2ddf527305.png"></details> | 389.99 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62019989-0171c000-b176-11e9-94da-017b146afe65.png"></details> | 391.32 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020340-6c6fc680-b177-11e9-8aa1-9ac5c2001783.png"></details> | **425.89** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62067085-c0b28f00-b1e7-11e9-9dd5-c52b6104878f.png"></details> | 65.04 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62100441-9ba13900-b246-11e9-9373-95c6063915ab.png"></details> | 181.72 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62230967-dd7dcc80-b377-11e9-965b-60a9f3d5a7a1.png"></details> |
| Pong <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62232135-5b42d780-b37a-11e9-9454-ff2d109ef4f4.png"><img src="https://user-images.githubusercontent.com/8209263/62232134-5b42d780-b37a-11e9-892f-a84ea8881e78.png"></details> | 20.04 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020247-10a53d80-b177-11e9-9f0d-1433d4d87210.png"></details> | 19.66 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020342-6f6ab700-b177-11e9-824e-75f431dc14ec.png"></details> | 20.09 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62067100-c6a87000-b1e7-11e9-919e-ad68e4166213.png"></details> | 18.34 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62100450-9fcd5680-b246-11e9-8170-2ad4473e8294.png"></details> | **20.44** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62230975-e2428080-b377-11e9-8970-6917ae80c0b4.png"></details> |
| Qbert <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62232149-60078b80-b37a-11e9-99bb-cedc9fe064d5.png"><img src="https://user-images.githubusercontent.com/8209263/62232148-60078b80-b37a-11e9-9610-17ac447a479f.png"></details> | 13,328.32 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020263-261a6780-b177-11e9-8936-22a74d2405d3.png"></details> | 13,259.19 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020347-742f6b00-b177-11e9-8bfb-edfcfd44c8b7.png"></details> | **13,691.89** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62067104-cb6d2400-b1e7-11e9-9c4f-9eaac265d7d6.png"></details> | 4,787.79 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62100455-a4920a80-b246-11e9-8ca5-d4dc1ce3d76f.png"></details> | 11,673.52 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62230986-e79fcb00-b377-11e9-8861-3686954b7e1a.png"></details> |
| Seaquest <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62232168-6bf34d80-b37a-11e9-9564-fa3609dc5c75.png"><img src="https://user-images.githubusercontent.com/8209263/62232167-6bf34d80-b37a-11e9-8db3-c79a0e78292b.png"></details> | 892.68 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020266-29adee80-b177-11e9-83c2-fafbdbb982b9.png"></details> | 1,686.08 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020350-772a5b80-b177-11e9-8917-e3c8a745cd08.png"></details> | 1,583.04 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62067113-cf994180-b1e7-11e9-870b-b9bba71f2a7e.png"></details> | 1,118.50 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62100462-a9ef5500-b246-11e9-8699-9356ff81ff93.png"></details> | **3,751.34** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62230991-ebcbe880-b377-11e9-8de4-a01379d1d61c.png"></details> |

#### Roboschool Benchmark

This benchmark table is pulled from [PR 405](https://github.com/kengz/SLM-Lab/pull/405). See the full [benchmark results here](https://github.com/kengz/SLM-Lab/blob/master/BENCHMARK.md).

| Env. \ Alg. | A2C (GAE) | A2C (n-step) | PPO | SAC |
|:---|---|---|---|---|
| RoboschoolAnt <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63823815-fbecce80-c909-11e9-82be-58b5b27a1f22.png"><img src="https://user-images.githubusercontent.com/8209263/63823816-fbecce80-c909-11e9-81ff-92373420582a.png"></details> | 1029.51 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822013-d4930300-c903-11e9-91f8-13e9bda44b59.png"></details> | 1148.76 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822318-d90beb80-c904-11e9-8ddf-16c105508e5e.png"></details> | 1931.35 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822580-a57d9100-c905-11e9-8d31-6df248d62a74.png"></details> | **2903.32** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822931-dca07200-c906-11e9-9687-812ab83a3bdd.png"></details> |
| RoboschoolAtlasForwardWalk <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63823818-fbecce80-c909-11e9-8872-f40951ffb7c0.png"><img src="https://user-images.githubusercontent.com/8209263/63823819-fc856500-c909-11e9-876e-6222a21e036c.png"></details> | 68.15 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822079-16bc4480-c904-11e9-9b0b-d63267f820aa.png"></details> | 73.46 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822362-f17c0600-c904-11e9-888b-2511bc59df9a.png"></details> | 148.81 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822614-c7771380-c905-11e9-9978-5fc9a6b0f218.png"></details> | **942.39** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822963-f93caa00-c906-11e9-869f-df8f6032d469.png"></details> |
| RoboschoolHalfCheetah <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63823820-fc856500-c909-11e9-88d6-ed606cbffe5a.png"><img src="https://user-images.githubusercontent.com/8209263/63823821-fc856500-c909-11e9-8974-de3cbd03815c.png"></details> | 895.24 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822128-3a7f8a80-c904-11e9-8e48-12c3214197ec.png"></details> | 409.59 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822377-02c51280-c905-11e9-9ff3-5a96f3c7ba05.png"></details> | 1838.69 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822643-dcec3d80-c905-11e9-89df-a30f84842bb1.png"></details> | **2350.94** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822986-0fe30100-c907-11e9-8f1d-0137e1facdff.png"></details> |
| RoboschoolHopper <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63823822-fc856500-c909-11e9-8577-7a5ddf5dbea2.png"><img src="https://user-images.githubusercontent.com/8209263/63823823-fc856500-c909-11e9-9814-72fb7afcb96f.png"></details> | 286.67 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822146-45d2b600-c904-11e9-976d-e61d1c215427.png"></details> | -187.91 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822416-196b6980-c905-11e9-8410-f2ff57a63983.png"></details> | **2079.22** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822669-ed9cb380-c905-11e9-9732-9cce76683f60.png"></details> | 1826.50 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63823013-20937700-c907-11e9-98c9-8ecb3a750187.png"></details> |
| RoboschoolInvertedDoublePendulum <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63823824-fc856500-c909-11e9-8405-8753908b2009.png"><img src="https://user-images.githubusercontent.com/8209263/63823825-fd1dfb80-c909-11e9-9899-7de3f3365382.png"></details> | 1769.74 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822174-5f73fd80-c904-11e9-848f-959aaff30f11.png"></details> | 486.76 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822446-2f792a00-c905-11e9-972e-545493c136be.png"></details> | **7967.03** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822686-0016ed00-c906-11e9-8064-7eecab895646.png"></details> | 6664.53 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63823039-3c971880-c907-11e9-845a-7acd316c16f7.png"></details> |
| RoboschoolInvertedPendulum <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63823826-fd1dfb80-c909-11e9-9635-579ea9bd4982.png"><img src="https://user-images.githubusercontent.com/8209263/63823827-fd1dfb80-c909-11e9-9d91-370629edcf33.png"></details> | **1000.0** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822205-7b779f00-c904-11e9-9ac7-d686fc640a3b.png"></details> | 997.54 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822472-44ee5400-c905-11e9-8c93-2973786ba561.png"></details> | 930.29 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822719-19b83480-c906-11e9-9ec7-245f9dcf4431.png"></details> | 955.16 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63823047-4e78bb80-c907-11e9-9dbc-e8c5996aef9e.png"></details> |
| RoboschoolInvertedPendulumSwingup <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63823828-fd1dfb80-c909-11e9-9d79-e0bbc6594e33.png"><img src="https://user-images.githubusercontent.com/8209263/63823829-fd1dfb80-c909-11e9-997b-fc71a7c95955.png"></details> | **1000.0** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822227-8fbb9c00-c904-11e9-93de-5668b0790ff4.png"></details> | 997.54 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822495-58012400-c905-11e9-98bc-42d22c73dc0a.png"></details> | 930.29 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822745-32284f00-c906-11e9-9f8a-e04d8c5a9f88.png"></details> | 958.29 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63823068-5fc1c800-c907-11e9-80b1-78508edd379a.png"></details> |
| RoboschoolReacher <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63823830-fdb69200-c909-11e9-906c-933f6d72abf6.png"><img src="https://user-images.githubusercontent.com/8209263/63823831-fdb69200-c909-11e9-8903-2b301836bdd4.png"></details> | 14.57 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822262-a661f300-c904-11e9-80be-8a7cb986fa77.png"></details> | -6.18 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822533-75ce8900-c905-11e9-87a6-dbdc1c0a74cc.png"></details> | 19.18 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822853-9ea34e00-c906-11e9-8fef-93e18f17c5fc.png"></details> | **19.73** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63823084-70723e00-c907-11e9-99d2-26935eba9179.png"></details> |
| RoboschoolWalker2d <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63823832-fdb69200-c909-11e9-9ca4-d051940b84ac.png"><img src="https://user-images.githubusercontent.com/8209263/63823833-fdb69200-c909-11e9-9155-6124ecd90676.png"></details> | 413.26 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822280-b548a580-c904-11e9-943b-168bcd58ea01.png"></details> | 141.83 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822553-87b02c00-c905-11e9-99b0-87da689a3bd2.png"></details> | 1368.25 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63822867-afec5a80-c906-11e9-8b14-453bf1f381d2.png"></details> | **1982.05** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63823102-841da480-c907-11e9-9434-f87ab6a343af.png"></details> |

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
