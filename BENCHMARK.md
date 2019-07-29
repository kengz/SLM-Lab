## Benchmark

The SLM Lab provides a set of benchmark results that are periodically updated with new feature releases. All the results below link to their respective PRs with the full experiment reports. To see more:
- [the `result` PRs](https://github.com/kengz/SLM-Lab/pulls?utf8=%E2%9C%93&q=is%3Apr+label%3Aresult+).
- the full experiment datas contributed are [public on Dropbox ](https://www.dropbox.com/sh/urifraklxcvol70/AADxtt6zUNuVR6qe288JYNCNa?dl=0).

The data can be downloaded into SLM Lab's `data/` folder and [reran in enjoy mode](https://kengz.gitbooks.io/slm-lab/content/usage/lab-commands.html).

#### Terminology

- A2C (GAE): Advantage Actor-Critic with GAE as advantage estimation
- A2C (n-step): Advantage Actor-Critic with n-step return as advantage estimation
- A3C: Asynchronous Advantage Actor-Critic
- CER: Combined Experience Replay
- DDQN: Double Deep Q-Learning
- DIST: Distributed
- DQN: Deep Q-learning
- GAE: Generalized Advantage Estimation
- PER: Prioritized Experience Replay
- PPO: Proximal Policy Optimization
- SIL: Self Imitation Learning

### Atari Benchmark

[OpenAI gym](https://gym.openai.com/envs/#atari) offers a wrapper for the [Atari Learning Environment (ALE)](https://github.com/mgbellemare/Arcade-Learning-Environment).

This benchmark table shows the `Trial` level `final_return_ma` from SLM Lab. This is final value of the 100-ckpt moving average of the return (total rewards) from evaluation. Each `Trial` is ran with 4 `Session`s with different random seeds, and their `final_return_ma` are averaged on the `Trial` level.

All the results are shown below and the data folders including the metrics and models are uploaded to the [SLM Lab public Dropbox](https://www.dropbox.com/sh/urifraklxcvol70/AADxtt6zUNuVR6qe288JYNCNa?dl=0)

>The results for A2C (GAE), A2C (n-step), PPO, DQN, DDQN+PER are uploaded in [PR 396](https://github.com/kengz/SLM-Lab/pull/396).

| Env \ Algorithm | A2C (GAE) | A2C (n-step) | PPO |
|:---|---|---|---|
| BreakoutNoFrameskip-v4 | 389.99 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62019989-0171c000-b176-11e9-94da-017b146afe65.png"></details> | 391.32 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020340-6c6fc680-b177-11e9-8aa1-9ac5c2001783.png"></details> | **456.30** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020417-b48ee900-b177-11e9-92f4-9c197d056b87.png"></details> |
| PongNoFrameskip-v4 | **20.04** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020247-10a53d80-b177-11e9-9f0d-1433d4d87210.png"></details> | 19.66 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020342-6f6ab700-b177-11e9-824e-75f431dc14ec.png"></details> | 19.78 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020426-b6f14300-b177-11e9-9927-da784d9dd1e4.png"></details> |
| QbertNoFrameskip-v4 | 13328.32 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020263-261a6780-b177-11e9-8936-22a74d2405d3.png"></details> | 13259.19 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020347-742f6b00-b177-11e9-8bfb-edfcfd44c8b7.png"></details> | **13784.93** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020432-ba84ca00-b177-11e9-9f30-fde235b1d470.png"></details> |
| SeaquestNoFrameskip-v4 | 892.68 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020266-29adee80-b177-11e9-83c2-fafbdbb982b9.png"></details> | **1686.08** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020350-772a5b80-b177-11e9-8917-e3c8a745cd08.png"></details> | 1393.63 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020438-beb0e780-b177-11e9-83f1-44ed80c3ab52.png"></details> |


### Classic Benchmark

>TODO This section is to be updated

The numbers in the table are fitness scores, which is a high level metric summarizing the strength, speed, stability and consistency of a trial. An experiment generates many specs to search through; each spec is ran in a trial, and each trial runs multiple repeated sessions for reproducibility. For more, see [analytics](https://kengz.gitbooks.io/slm-lab/content/analytics/analytics.html).


| Algorithm / *Owner* | [DQN](https://arxiv.org/abs/1312.5602) | [DDQN](https://arxiv.org/abs/1509.06461) | [Dueling DQN](https://arxiv.org/abs/1511.06581) | DQN + [PER](https://arxiv.org/abs/1511.05952) | DDQN + [PER](https://arxiv.org/abs/1511.05952) | DQN + [CER](https://arxiv.org/abs/1712.01275) | DDQN + [CER](https://arxiv.org/abs/1712.01275) | [DIST DQN](https://arxiv.org/abs/1602.01783) | REINFORCE | A2C | A2C + [GAE](https://arxiv.org/abs/1506.02438) | A2C + [GAE](https://arxiv.org/abs/1506.02438) + [SIL](https://arxiv.org/abs/1806.05635) | [A3C](https://arxiv.org/abs/1602.01783) | [A3C](https://arxiv.org/abs/1602.01783) + [GAE](https://arxiv.org/abs/1506.02438) | [PPO](https://arxiv.org/abs/1707.06347) | [PPO](https://arxiv.org/abs/1707.06347) + [SIL](https://arxiv.org/abs/1806.05635) | [DPPO](https://arxiv.org/pdf/1707.02286.pdf) |
|------------|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|[CartPole-v0](https://gym.openai.com/envs/CartPole-v0/)|[3.52](https://github.com/kengz/SLM-Lab/pull/213) |[0.85](https://github.com/kengz/SLM-Lab/pull/214) | | | |[4.79](https://github.com/kengz/SLM-Lab/pull/184) |[5.65](https://github.com/kengz/SLM-Lab/pull/195) | |[1.21](https://github.com/kengz/SLM-Lab/pull/200) |[7.10](https://github.com/kengz/SLM-Lab/pull/185) | [1.20](https://github.com/kengz/SLM-Lab/pull/180) |[6.26](https://github.com/kengz/SLM-Lab/pull/201) |[0.93](https://github.com/kengz/SLM-Lab/pull/205)| [1.60](https://github.com/kengz/SLM-Lab/pull/204) |[0.88](https://github.com/kengz/SLM-Lab/pull/211) |[1.48](https://github.com/kengz/SLM-Lab/pull/212) | |
|[LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/)|[1.15](https://github.com/kengz/SLM-Lab/pull/250)|[1.39](https://github.com/kengz/SLM-Lab/pull/251)| | | | | | | [0.77](https://github.com/kengz/SLM-Lab/pull/232) | | | | | | | | |
|[MountainCar-v0](https://gym.openai.com/envs/MountainCar-v0/)|[1.04](https://github.com/kengz/SLM-Lab/pull/219)|[1.02](https://github.com/kengz/SLM-Lab/pull/220)| | | | | | | | | | | | | | | |
|[3dball](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#3dball-3d-balance-ball)| | | | | | | | | | | | | | | | | |
|[gridworld](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#gridworld)| | | | | | | | | | | | | | | | | |
|[BeamRider-v0](https://gym.openai.com/envs/BeamRider-v0/)| | | | | | | | | | | | | | | | | |
|[Pendulum-v0](https://gym.openai.com/envs/Pendulum-v0/)| n/a | n/a| n/a | n/a | n/a | n/a | n/a | n/a | | | | | | | | | |
|[Acrobot-v1](https://gym.openai.com/envs/Acrobot-v1/)| n/a | n/a| n/a | n/a | n/a | n/a | n/a | n/a | | | | | | | | | |
|[BipedalWalker-v2](https://gym.openai.com/envs/BipedalWalker-v2/)| n/a | n/a| n/a | n/a | n/a | n/a | n/a | n/a | | | | | | | | | |
|[CarRacing-v0](https://gym.openai.com/envs/CarRacing-v0/)| n/a | n/a| n/a | n/a | n/a | n/a | n/a | n/a | | | | | | | | | |
