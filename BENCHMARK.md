## Benchmarks

| Algorithm / *Owner* | [DQN](https://arxiv.org/abs/1312.5602) / *[Keng](https://github.com/kengz)* | DQN + RNN | [DDQN](https://arxiv.org/abs/1509.06461) | DDQN + RNN | [Dueling DQN](https://arxiv.org/abs/1511.06581) | DQN + [PER](https://arxiv.org/abs/1511.05952) | DDQN + [PER](https://arxiv.org/abs/1511.05952) | DQN + [CER](https://arxiv.org/abs/1712.01275) | DDQN + [CER](https://arxiv.org/abs/1712.01275) | [DIST DQN](https://arxiv.org/abs/1602.01783) | REINFORCE | A2C | A2C + [GAE](https://arxiv.org/abs/1506.02438) / *[Laura](https://github.com/lgraesser)* | A2C + [GAE](https://arxiv.org/abs/1506.02438) + [SIL](https://arxiv.org/abs/1806.05635) | [A3C](https://arxiv.org/abs/1602.01783) | [A3C](https://arxiv.org/abs/1602.01783) + [GAE](https://arxiv.org/abs/1506.02438) | [PPO](https://arxiv.org/abs/1707.06347) | [PPO](https://arxiv.org/abs/1707.06347) + [SIL](https://arxiv.org/abs/1806.05635) | [DPPO](https://arxiv.org/pdf/1707.02286.pdf) |
|------------|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|[CartPole-v0](https://gym.openai.com/envs/CartPole-v0/)| | | | | | | | | | | | | [1.20](https://github.com/kengz/SLM-Lab/pull/180) | | | | | | |
|[3dball](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#3dball-3d-balance-ball)| | | | | | | | | | | | | | | | | | | |
|[LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/)| | | | | | | | | | | | | | | | | | | |
|[gridworld](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#gridworld)| | | | | | | | | | | | | | | | | | | |
|[BeamRider-v0](https://gym.openai.com/envs/BeamRider-v0/)| | | | | | | | | | | | | | | | | | | |
|[Pendulum-v0](https://gym.openai.com/envs/Pendulum-v0/)| n/a | n/a | n/a | n/a| n/a | n/a | n/a | n/a | n/a | n/a | | | | | | | | | |
|[Acrobot-v1](https://gym.openai.com/envs/Acrobot-v1/)| n/a | n/a | n/a | n/a| n/a | n/a | n/a | n/a | n/a | n/a | | | | | | | | | |
|[BipedalWalker-v2](https://gym.openai.com/envs/BipedalWalker-v2/)| n/a | n/a | n/a | n/a| n/a | n/a | n/a | n/a | n/a | n/a | | | | | | | | | |
|[CarRacing-v0](https://gym.openai.com/envs/CarRacing-v0/)| n/a | n/a | n/a | n/a| n/a | n/a | n/a | n/a | n/a | n/a | | | | | | | | | |

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
