## Benchmarks

| Algorithm: *Owner* | [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/) | [3dball](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#3dball-3d-balance-ball) | [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/) | [gridworld](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#gridworld) | [BeamRider-v0](https://gym.openai.com/envs/BeamRider-v0/)| [Pendulum-v0](https://gym.openai.com/envs/Pendulum-v0/) | [Acrobot-v1](https://gym.openai.com/envs/Acrobot-v1/) | [BipedalWalker-v2](https://gym.openai.com/envs/BipedalWalker-v2/) | [CarRacing-v0](https://gym.openai.com/envs/CarRacing-v0/) |
|------------|--|--|--|--|--|--|--|--|--|
| [DQN](https://arxiv.org/abs/1312.5602): *[Keng](https://github.com/kengz)* | | | | | | n/a | n/a | n/a | n/a |
| DQN + RNN | | | | | | n/a | n/a | n/a | n/a |
| [DDQN](https://arxiv.org/abs/1509.06461) | | | | | | n/a | n/a | n/a | n/a |
| DDQN + RNN | | | | | | n/a | n/a | n/a | n/a |
| [Dueling DQN](https://arxiv.org/abs/1511.06581) | | | | | | n/a | n/a | n/a | n/a |
| DQN + [PER](https://arxiv.org/abs/1511.05952) | | | | | | n/a | n/a | n/a | n/a |
| DQN + [PER](https://arxiv.org/abs/1511.05952) + [CER](https://arxiv.org/abs/1712.01275) | | | | | | n/a | n/a | n/a | n/a |
| DDQN + [PER](https://arxiv.org/abs/1511.05952) | | | | | | n/a | n/a | n/a | n/a |
| DDQN + [PER](https://arxiv.org/abs/1511.05952) + [CER](https://arxiv.org/abs/1712.01275) | | | | | | n/a | n/a | n/a | n/a |
| [DST DQN](https://arxiv.org/abs/1602.01783) | | | | | | n/a | n/a | n/a | n/a |
| Reinforce | | | | | | | | | |
| A2C | | | | | | | | | |
| A2C + [GAE](https://arxiv.org/abs/1506.02438): *[Laura](https://github.com/lgraesser)* | [1.20](https://github.com/lgraesser) | | | | | | | | |
| [A3C](https://arxiv.org/abs/1602.01783) | | | | | | | | | |
| [A3C](https://arxiv.org/abs/1602.01783) + [GAE](https://arxiv.org/abs/1506.02438) | | | | | | | | | |
| [PPO](https://arxiv.org/abs/1707.06347) | | | | | | | | | |
| DST [PPO](https://arxiv.org/abs/1707.06347) | | | | | | | | | |
| [PPO](https://arxiv.org/abs/1707.06347) + [SIL](https://arxiv.org/abs/1806.05635) | | | | | | | | | ||

### Terminology
- DQN: Deep Q-learning
- DDQN: Double Deep Q-Learning
- PER: Prioritized experience replay
- CER: Combined experience replay
- DST: Distributed
- A2C: Advantage Actor-Critic
- A3C: Asynchronous advantage Actor-Critic
- GAE: Generalized advantage estimation
- PPO: Proximal policy optimization
- SIL: Self imitation learning

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
