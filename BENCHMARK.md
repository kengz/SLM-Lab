## Benchmarks

| Algorithm: *Owner*  | [CartPole](https://gym.openai.com/envs/CartPole-v0/) | 3d Ball | [Lunar Lander](https://gym.openai.com/envs/LunarLander-v2/) | Grid World | [Beamrider](https://gym.openai.com/envs/BeamRider-v0/)| [Pendulum](https://gym.openai.com/envs/Pendulum-v0/) | [Acrobot](https://gym.openai.com/envs/Acrobot-v1/) | [Bipedal Walker](https://gym.openai.com/envs/BipedalWalker-v2/) | [Car Racing](https://gym.openai.com/envs/CarRacing-v0/) |
|------------|--|--|--|--|--|--|--|--|--|
| [DQN](https://arxiv.org/abs/1312.5602): *[WLK](https://github.com/kengz)* | | | | | | n/a | n/a | n/a | n/a |
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
| A2C: *[LG](https://github.com/lgraesser)* | | | | | | | | | |
| A2C + [GAE](https://arxiv.org/abs/1506.02438) | | | | | | | | | |
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
- Cartpole
- 3d ball
- Lunar Lander
- Grid World
- Beamrider (Atari)

### Continuous environments
- Pendulum
- Acrobot
- Bipedal Walker
- Car Racing
