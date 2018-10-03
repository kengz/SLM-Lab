## Benchmarks

The numbers in the table are fitness scores, which is a high level metric summarizing the strength, speed, stability and consistency of a trial. An experiment generates many specs to search through; each spec is ran in a trial, and each trial runs multiple repeated sessions for reproducibility. For more, see [analytics](https://kengz.gitbooks.io/slm-lab/content/analytics/analytics.html).

All the results below link to their respective PRs with the full experiment reports. To see more:
- [the `result` PRs](https://github.com/kengz/SLM-Lab/pulls?utf8=%E2%9C%93&q=is%3Apr+label%3Aresult+).
- the full experiment datas contributed are [public on Dropbox ](https://www.dropbox.com/sh/y738zvzj3nxthn1/AAAg1e6TxXVf3krD81TD5V0Ra?dl=0)

| Algorithm / *Owner* | [DQN](https://arxiv.org/abs/1312.5602) | [DDQN](https://arxiv.org/abs/1509.06461) | [Dueling DQN](https://arxiv.org/abs/1511.06581) | DQN + [PER](https://arxiv.org/abs/1511.05952) | DDQN + [PER](https://arxiv.org/abs/1511.05952) | DQN + [CER](https://arxiv.org/abs/1712.01275) | DDQN + [CER](https://arxiv.org/abs/1712.01275) | [DIST DQN](https://arxiv.org/abs/1602.01783) | REINFORCE | A2C | A2C + [GAE](https://arxiv.org/abs/1506.02438) | A2C + [GAE](https://arxiv.org/abs/1506.02438) + [SIL](https://arxiv.org/abs/1806.05635) | [A3C](https://arxiv.org/abs/1602.01783) | [A3C](https://arxiv.org/abs/1602.01783) + [GAE](https://arxiv.org/abs/1506.02438) | [PPO](https://arxiv.org/abs/1707.06347) | [PPO](https://arxiv.org/abs/1707.06347) + [SIL](https://arxiv.org/abs/1806.05635) | [DPPO](https://arxiv.org/pdf/1707.02286.pdf) |
|------------|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|[CartPole-v0](https://gym.openai.com/envs/CartPole-v0/)| | | | | |[4.79](https://github.com/kengz/SLM-Lab/pull/184) |[5.65](https://github.com/kengz/SLM-Lab/pull/195) | |[1.21](https://github.com/kengz/SLM-Lab/pull/200) |[7.10](https://github.com/kengz/SLM-Lab/pull/185) | [1.20](https://github.com/kengz/SLM-Lab/pull/180) |[6.26](https://github.com/kengz/SLM-Lab/pull/201) |[0.93](https://github.com/kengz/SLM-Lab/pull/205)| [1.60](https://github.com/kengz/SLM-Lab/pull/204) | | | |
|[LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/)|[1.14](https://github.com/kengz/SLM-Lab/pull/191)|[1.15](https://github.com/kengz/SLM-Lab/pull/203)| | | | | | | | | | | | | | | |
|[MountainCar-v0](https://gym.openai.com/envs/MountainCar-v0/)|[0.92](https://github.com/kengz/SLM-Lab/pull/208)| | | | | | | | | | | | | | | | |
|[3dball](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#3dball-3d-balance-ball)| | | | | | | | | | | | | | | | | |
|[gridworld](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#gridworld)| | | | | | | | | | | | | | | | | |
|[BeamRider-v0](https://gym.openai.com/envs/BeamRider-v0/)| | | | | | | | | | | | | | | | | |
|[Pendulum-v0](https://gym.openai.com/envs/Pendulum-v0/)| n/a | n/a| n/a | n/a | n/a | n/a | n/a | n/a | | | | | | | | | |
|[Acrobot-v1](https://gym.openai.com/envs/Acrobot-v1/)| n/a | n/a| n/a | n/a | n/a | n/a | n/a | n/a | | | | | | | | | |
|[BipedalWalker-v2](https://gym.openai.com/envs/BipedalWalker-v2/)| n/a | n/a| n/a | n/a | n/a | n/a | n/a | n/a | | | | | | | | | |
|[CarRacing-v0](https://gym.openai.com/envs/CarRacing-v0/)| n/a | n/a| n/a | n/a | n/a | n/a | n/a | n/a | | | | | | | | | |

### Terminology
- DQN: Deep Q-learning
- DDQN: Double Deep Q-Learning
- PER: Prioritized Experience Replay
- CER: Combined Experience Replay
- DIST: Distributed
- A2C: Advantage Actor-Critic
- A3C: Asynchronous Advantage Actor-Critic
- GAE: Generalized Advantage Estimation
- PPO: Proximal Policy Optimization
- SIL: Self Imitation Learning

### Discrete environments
- [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/)
- [3dball](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#3dball-3d-balance-ball)
- [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/)
- [gridworld](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#gridworld)
- [BeamRider-v0](https://gym.openai.com/envs/BeamRider-v0/)
- *more coming soon*

### Continuous environments
- [Pendulum-v0](https://gym.openai.com/envs/Pendulum-v0/)
- [Acrobot-v1](https://gym.openai.com/envs/Acrobot-v1/)
- [BipedalWalker-v2](https://gym.openai.com/envs/BipedalWalker-v2/)
- [CarRacing-v0](https://gym.openai.com/envs/CarRacing-v0/)
- *more coming soon*
