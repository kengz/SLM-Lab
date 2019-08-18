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
- DDQN: Double Deep Q-Network
- DIST: Distributed
- DQN: Deep Q-Network
- GAE: Generalized Advantage Estimation
- PER: Prioritized Experience Replay
- PPO: Proximal Policy Optimization
- SAC: Soft Actor-Critic
- SIL: Self Imitation Learning

### Atari Benchmark

[OpenAI gym](https://gym.openai.com/envs/#atari) offers a wrapper for the [Atari Learning Environment (ALE)](https://github.com/mgbellemare/Arcade-Learning-Environment).

This benchmark table shows the `Trial` level `final_return_ma` from SLM Lab. This is final value of the 100-ckpt moving average of the return (total rewards) from evaluation. Each `Trial` is ran with 4 `Session`s with different random seeds, and their `final_return_ma` are averaged on the `Trial` level.

The specs for these are contained in the [`slm_lab/spec/benchmark`](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark) folder. All the results are shown below and the data folders including the metrics and models are uploaded to the [SLM Lab public Dropbox](https://www.dropbox.com/sh/urifraklxcvol70/AADxtt6zUNuVR6qe288JYNCNa?dl=0)

>The results for A2C (GAE), A2C (n-step), PPO, DQN, DDQN+PER were uploaded in [PR 396](https://github.com/kengz/SLM-Lab/pull/396).

| Env. \ Alg. | A2C (GAE) | A2C (n-step) | PPO | DQN | DDQN+PER |
|:---|---|---|---|---|---|
| Breakout <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62232119-554cf680-b37a-11e9-9059-3e49bbb799d2.png"><img src="https://user-images.githubusercontent.com/8209263/62232118-554cf680-b37a-11e9-9d5b-dd2ddf527305.png"></details> | 389.99 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62019989-0171c000-b176-11e9-94da-017b146afe65.png"></details> | 391.32 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020340-6c6fc680-b177-11e9-8aa1-9ac5c2001783.png"></details> | **425.89** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62067085-c0b28f00-b1e7-11e9-9dd5-c52b6104878f.png"></details> | 65.04 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62100441-9ba13900-b246-11e9-9373-95c6063915ab.png"></details> | 181.72 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62230967-dd7dcc80-b377-11e9-965b-60a9f3d5a7a1.png"></details> |
| Pong <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62232135-5b42d780-b37a-11e9-9454-ff2d109ef4f4.png"><img src="https://user-images.githubusercontent.com/8209263/62232134-5b42d780-b37a-11e9-892f-a84ea8881e78.png"></details> | 20.04 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020247-10a53d80-b177-11e9-9f0d-1433d4d87210.png"></details> | 19.66 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020342-6f6ab700-b177-11e9-824e-75f431dc14ec.png"></details> | 20.09 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62067100-c6a87000-b1e7-11e9-919e-ad68e4166213.png"></details> | 18.34 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62100450-9fcd5680-b246-11e9-8170-2ad4473e8294.png"></details> | **20.44** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62230975-e2428080-b377-11e9-8970-6917ae80c0b4.png"></details> |
| Qbert <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62232149-60078b80-b37a-11e9-99bb-cedc9fe064d5.png"><img src="https://user-images.githubusercontent.com/8209263/62232148-60078b80-b37a-11e9-9610-17ac447a479f.png"></details> | 13,328.32 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020263-261a6780-b177-11e9-8936-22a74d2405d3.png"></details> | 13,259.19 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020347-742f6b00-b177-11e9-8bfb-edfcfd44c8b7.png"></details> | **13,691.89** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62067104-cb6d2400-b1e7-11e9-9c4f-9eaac265d7d6.png"></details> | 4,787.79 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62100455-a4920a80-b246-11e9-8ca5-d4dc1ce3d76f.png"></details> | 11,673.52 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62230986-e79fcb00-b377-11e9-8861-3686954b7e1a.png"></details> |
| Seaquest <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62232168-6bf34d80-b37a-11e9-9564-fa3609dc5c75.png"><img src="https://user-images.githubusercontent.com/8209263/62232167-6bf34d80-b37a-11e9-8db3-c79a0e78292b.png"></details> | 892.68 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020266-29adee80-b177-11e9-83c2-fafbdbb982b9.png"></details> | 1,686.08 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020350-772a5b80-b177-11e9-8917-e3c8a745cd08.png"></details> | 1,583.04 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62067113-cf994180-b1e7-11e9-870b-b9bba71f2a7e.png"></details> | 1,118.50 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62100462-a9ef5500-b246-11e9-8699-9356ff81ff93.png"></details> | **3,751.34** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62230991-ebcbe880-b377-11e9-8de4-a01379d1d61c.png"></details> |


### Roboschool Benchmark

[Roboschool](https://github.com/openai/roboschool) by OpenAI offers free open source robotics simulations with improved physics. Although it mirrors the environments from MuJuCo, its environments' rewards are different.

>The results for A2C (GAE), A2C (n-step), PPO, SAC were uploaded in [PR 402](https://github.com/kengz/SLM-Lab/pull/402).

| Env. \ Alg. | A2C (GAE) | A2C (n-step) | PPO | SAC |
|:---|---|---|---|---|
| RoboschoolAnt <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63227722-75641e80-c19e-11e9-8d8c-eda59309b6b8.png"><img src="https://user-images.githubusercontent.com/8209263/63227723-75641e80-c19e-11e9-9191-107204eefceb.png"></details> | 752.68 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63217723-629b0c80-c100-11e9-8220-c8df373af50f.png"></details> | 1152.45 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63217729-76df0980-c100-11e9-8dee-7a0a5ae226f6.png"></details> | 1094.02 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63217739-8d856080-c100-11e9-931b-0b7052073f2c.png"></details> | **2657.06** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63227680-fbcc3080-c19d-11e9-8a32-f54bfbad01f8.png"></details> |
| RoboschoolHalfCheetah <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63227725-7b59ff80-c19e-11e9-9198-2b3c0f4739cc.png"><img src="https://user-images.githubusercontent.com/8209263/63227726-7b59ff80-c19e-11e9-9c09-09c8868ed825.png"></details> | 707.09 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63217724-675fc080-c100-11e9-98a0-09ea6aa15e56.png"></details> | 840.26 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63217730-7a729080-c100-11e9-81be-e75996ddb5b0.png"></details> | 1781.18 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63217743-924a1480-c100-11e9-8f6f-18b698bcd031.png"></details> | **2331.95** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63227682-fff84e00-c19d-11e9-858b-14524062a01a.png"></details> |
| RoboschoolHopper <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63227728-7eed8680-c19e-11e9-9837-d51d2b5e67b1.png"><img src="https://user-images.githubusercontent.com/8209263/63227729-7f861d00-c19e-11e9-92b9-7ddda05921f8.png"></details> | 733.37 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63217725-6dee3800-c100-11e9-9c39-b89ebbad8657.png"></details> | 506.39 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63217733-7e061780-c100-11e9-8ff1-5f0a981d8b19.png"></details> | 1754.80 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63217744-95450500-c100-11e9-8583-51d6430331c3.png"></details> | **2234.18** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63227683-05559880-c19e-11e9-97d7-635f87fcd792.png"></details> |
| RoboschoolWalker2d <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63227730-8319a400-c19e-11e9-8554-5142393ee3ab.png"><img src="https://user-images.githubusercontent.com/8209263/63227731-8319a400-c19e-11e9-8c3a-96bb66e8fdd9.png"></details> | 263.61 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63217727-7181bf00-c100-11e9-8fce-ac9f941a905b.png"></details> | 149.38 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63217734-81010800-c100-11e9-9553-070761c6f274.png"></details> | 1341.25 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63217746-99712280-c100-11e9-8ff9-5a31aa547092.png"></details> | **2052.53** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63227685-0edf0080-c19e-11e9-8708-f33a05eb7c37.png"></details> |


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
