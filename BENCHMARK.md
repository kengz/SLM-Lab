## Benchmark

The SLM Lab provides a set of benchmark results that are periodically updated with new feature releases. All the results below link to their respective PRs with the full experiment reports. To see more:
- [the `result` PRs](https://github.com/kengz/SLM-Lab/pulls?utf8=%E2%9C%93&q=is%3Apr+label%3Aresult+).
- the full experiment datas contributed are [public on Dropbox ](https://www.dropbox.com/sh/urifraklxcvol70/AADxtt6zUNuVR6qe288JYNCNa?dl=0).

The data can be downloaded into SLM Lab's `data/` folder and [reran in enjoy mode](https://kengz.gitbooks.io/slm-lab/content/usage/lab-commands.html).

#### Hardware

For reference, the image based environment benchmarks are run on AWS GPU box `p2.16xlarge`, and the non-image based environments are run on AWS CPU box `m5a.24xlarge`.

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

|||||
|:---:|:---:|:---:|:---:|
| ![ppo beamrider](https://user-images.githubusercontent.com/8209263/63994698-689ecf00-caaa-11e9-991f-0a5e9c2f5804.gif) | ![ppo breakout](https://user-images.githubusercontent.com/8209263/63994695-650b4800-caaa-11e9-9982-2462738caa45.gif) | ![ppo kungfumaster](https://user-images.githubusercontent.com/8209263/63994690-60469400-caaa-11e9-9093-b1cd38cee5ae.gif) | ![ppo mspacman](https://user-images.githubusercontent.com/8209263/63994685-5cb30d00-caaa-11e9-8f35-78e29a7d60f5.gif) |
| BeamRider | Breakout | KungFuMaster | MsPacman |
| ![ppo pong](https://user-images.githubusercontent.com/8209263/63994680-59b81c80-caaa-11e9-9253-ed98370351cd.gif) | ![ppo qbert](https://user-images.githubusercontent.com/8209263/63994672-54f36880-caaa-11e9-9757-7780725b53af.gif) | ![ppo seaquest](https://user-images.githubusercontent.com/8209263/63994665-4dcc5a80-caaa-11e9-80bf-c21db818115b.gif) | ![ppo spaceinvaders](https://user-images.githubusercontent.com/8209263/63994624-15c51780-caaa-11e9-9c9a-854d3ce9066d.gif) |
| Pong | Qbert | Seaquest | Sp.Invaders |

[OpenAI gym](https://gym.openai.com/envs/#atari) offers a wrapper for the [Atari Learning Environment (ALE)](https://github.com/mgbellemare/Arcade-Learning-Environment).

This benchmark table shows the `Trial` level `final_return_ma` from SLM Lab. This is final value of the 100-ckpt moving average of the return (total rewards) from evaluation. Each `Trial` is ran with 4 `Session`s with different random seeds, and their `final_return_ma` are averaged on the `Trial` level.

The specs for these are contained in the [`slm_lab/spec/benchmark`](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark) folder. All the results are shown below and the data folders including the metrics and models are uploaded to the [SLM Lab public Dropbox](https://www.dropbox.com/sh/urifraklxcvol70/AADxtt6zUNuVR6qe288JYNCNa?dl=0)


### Discrete Benchmark

||||||||
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Env. \ Alg. | DQN | DDQN+PER | A2C (GAE) | A2C (n-step) | PPO | SAC |
| Breakout | 80.88 | 182 | 377 | 398 | **443** |  - |
| Pong | 18.48 | 20.5 | 19.31 | 19.56 | **20.58** | 19.87* |
| Seaquest | 1185 | **4405** | 1070 | 1684 | 1715 |  - |
| Qbert | 5494 | 11426 | 12405 | **13590** | 13460 | 214* |
| LunarLander | 192 | 233 | 25.21 | 68.23 | 214 | **276** |
| UnityHallway | -0.32 | 0.27 | 0.08 | -0.96 | **0.73** | - |
| UnityPushBlock | 4.88 | 4.93 | 4.68 | 4.93 | **4.97** | - |

>Episode score at the end of training attained by SLM Lab implementations on discrete-action control problems. Reported episode scores are the average over the last 100 checkpoints, and then averaged over 4 Sessions. Results marked with `*` were trained using the hybrid synchronous/asynchronous version of SAC to parallelize and speed up training time.

### Continuous Benchmark

||||||
|:---:|:---:|:---:|:---:|:---:|
| Env. \ Alg. | A2C (GAE) | A2C (n-step) | PPO | SAC |
| RoboschoolAnt | 787 | 1396 | 1843 | **2915** |
| RoboschoolAtlasForwardWalk | 59.87 | 88.04 | 172 | **800** |
| RoboschoolHalfCheetah | 712 | 439 | 1960 | **2497** |
| RoboschoolHopper | 710 | 285 | 2042 | **2045** |
| RoboschoolInvertedDoublePendulum | 996 | 4410 | 8076 | **8085** |
| RoboschoolInvertedPendulum | **995** | 978 | 986 | 941 |
| RoboschoolReacher | 12.9 | 10.16 | 19.51 | **19.99** |
| RoboschoolWalker2d | 280 | 220 | 1660 | **1894** |
| RoboschoolHumanoid | 99.31 | 54.58 | 2388 | **2621*** |
| RoboschoolHumanoidFlagrun | 73.57 | 178 | 2014 | **2056*** |
| RoboschoolHumanoidFlagrunHarder | -429 | 253 | **680** | 280* |
| Unity3DBall | 33.48 | 53.46 | 78.24 | **98.44** |
| Unity3DBallHard | 62.92 | 71.92 | 91.41 | **97.06** |

>Episode score at the end of training attained by SLM Lab implementations on continuous control problems. Reported episode scores are the average over the last 100 checkpoints, and then averaged over 4 Sessions. Results marked with `*` require 50M-100M frames, so we use the hybrid synchronous/asynchronous version of SAC to parallelize and speed up training time.

### Atari Benchmark

|||||||
|:---:|:---:|:---:|:---:|:---:|:---:|
| Env. \ Alg. | DQN | DDQN+PER | A2C (GAE) | A2C (n-step) | PPO |
| Adventure | -0.94 | -0.92 | -0.77 | -0.85 | **-0.3** |
| AirRaid | 1876 | 3974 | **4202** | 3557 | 4028 |
| Alien | 822 | 1574 | 1519 | **1627** | 1413 |
| Amidar | 90.95 | 431 | 577 | 418 | **795** |
| Assault | 1392 | 2567 | 3366 | 3312 | **3619** |
| Asterix | 1253 | **6866** | 5559 | 5223 | 6132 |
| Asteroids | 439 | 426 | **2951** | 2147 | 2186 |
| Atlantis | 68679 | 644810 | **2747371** | 2259733 | 2148077 |
| BankHeist | 131 | 623 | 855 | 1170 | **1183** |
| BattleZone | 6564 | 6395 | 4336 | 4533 | **13649** |
| BeamRider | 2799 | **5870** | 2659 | 4139 | 4299 |
| Berzerk | 319 | 401 | **1073** | 763 | 860 |
| Bowling | 30.29 | **39.5** | 24.51 | 23.75 | 31.64 |
| Boxing | 72.11 | 90.98 | 1.57 | 1.26 | **96.53** |
| Breakout | 80.88 | 182 | 377 | 398 | **443** |
| Carnival | 4280 | **4773** | 2473 | 1827 | 4566 |
| Centipede | 1899 | 2153 | 3909 | 4202 | **5003** |
| ChopperCommand | 1083 | **4020** | 3043 | 1280 | 3357 |
| CrazyClimber | 46984 | 88814 | 106256 | 109998 | **116820** |
| Defender | 281999 | 313018 | **665609** | 657823 | 534639 |
| DemonAttack | 1705 | 19856 | 23779 | 19615 | **121172** |
| DoubleDunk | -21.44 | -22.38 | **-5.15** | -13.3 | -6.01 |
| ElevatorAction | 32.62 | 17.91 | **9966** | 8818 | 6471 |
| Enduro | 437 | 959 | 787 | 0.0 | **1926** |
| FishingDerby | -88.14 | -1.7 | 16.54 | 1.65 | **36.03** |
| Freeway | 24.46 | 30.49 | 30.97 | 0.0 | **32.11** |
| Frostbite | 98.8 | **2497** | 277 | 261 | 1062 |
| Gopher | 1095 | **7562** | 929 | 1545 | 2933 |
| Gravitar | 87.34 | 258 | 313 | **433** | 223 |
| Hero | 1051 | 12579 | 16502 | **19322** | 17412 |
| IceHockey | -14.96 | -14.24 | **-5.79** | -6.06 | -6.43 |
| Jamesbond | 44.87 | **702** | 521 | 453 | 561 |
| JourneyEscape | -4818 | -2003 | **-921** | -2032 | -1094 |
| Kangaroo | 1965 | **8897** | 67.62 | 554 | 4989 |
| Krull | 5522 | 6650 | 7785 | 6642 | **8477** |
| KungFuMaster | 2288 | 16547 | 31199 | 25554 | **34523** |
| MontezumaRevenge | 0.0 | 0.02 | 0.08 | 0.19 | **1.08** |
| MsPacman | 1175 | 2215 | 1965 | 2158 | **2350** |
| NameThisGame | 3915 | 4474 | 5178 | 5795 | **6386** |
| Phoenix | 2909 | 8179 | 16345 | 13586 | **30504** |
| Pitfall | -68.83 | -73.65 | -101 | **-31.13** | -35.93 |
| Pong | 18.48 | 20.5 | 19.31 | 19.56 | **20.58** |
| Pooyan | 1958 | 2741 | 2862 | 2531 | **6799** |
| PrivateEye | **784** | 303 | 93.22 | 78.07 | 50.12 |
| Qbert | 5494 | 11426 | 12405 | **13590** | 13460 |
| Riverraid | 953 | **10492** | 8308 | 7565 | 9636 |
| RoadRunner | 15237 | 29047 | 30152 | 31030 | **32956** |
| Robotank | 3.43 | **9.05** | 2.98 | 2.27 | 2.27 |
| Seaquest | 1185 | **4405** | 1070 | 1684 | 1715 |
| Skiing | -14094 | **-12883** | -19481 | -14234 | -24713 |
| Solaris | 612 | 1396 | 2115 | **2236** | 1892 |
| SpaceInvaders | 451 | 670 | 733 | 750 | **797** |
| StarGunner | 3565 | 38238 | 44816 | 48410 | **60579** |
| Tennis | -23.78 | **-10.33** | -22.42 | -19.06 | -11.52 |
| TimePilot | 2819 | 1884 | 3331 | 3440 | **4398** |
| Tutankham | 35.03 | 159 | 161 | 175 | **211** |
| UpNDown | 2043 | 11632 | 89769 | 18878 | **262208** |
| Venture | 4.56 | 9.61 | 0.0 | 0.0 | **11.84** |
| VideoPinball | 8056 | **79730** | 35371 | 40423 | 58096 |
| WizardOfWor | 869 | 328 | 1516 | 1247 | **4283** |
| YarsRevenge | 5816 | 15698 | **27097** | 11742 | 10114 |
| Zaxxon | 442 | 54.28 | 64.72 | 24.7 | **641** |

>The table below presents results for 62 Atari games. All agents were trained for 10M frames (40M including skipped frames). Reported results are the episode score at the end of training, averaged over the previous 100 evaluation checkpoints with each checkpoint averaged over 4 Sessions. Agents were checkpointed every 10k training frames.



| Env. \ Alg. | [DQN](https://github.com/kengz/SLM-Lab/pull/396) | [DDQN+PER](https://github.com/kengz/SLM-Lab/pull/396) | [A2C (GAE)](https://github.com/kengz/SLM-Lab/pull/396) | [A2C (n-step)](https://github.com/kengz/SLM-Lab/pull/396) | [PPO](https://github.com/kengz/SLM-Lab/pull/396) |
|:---|---|---|---|---|---|
| Breakout <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62232119-554cf680-b37a-11e9-9059-3e49bbb799d2.png"><img src="https://user-images.githubusercontent.com/8209263/62232118-554cf680-b37a-11e9-9d5b-dd2ddf527305.png"></details> | **425.89** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62067085-c0b28f00-b1e7-11e9-9dd5-c52b6104878f.png"></details> | 65.04 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62100441-9ba13900-b246-11e9-9373-95c6063915ab.png"></details> | 181.72 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62230967-dd7dcc80-b377-11e9-965b-60a9f3d5a7a1.png"></details> | 389.99 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62019989-0171c000-b176-11e9-94da-017b146afe65.png"></details> | 391.32 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020340-6c6fc680-b177-11e9-8aa1-9ac5c2001783.png"></details> |
| Pong <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62232135-5b42d780-b37a-11e9-9454-ff2d109ef4f4.png"><img src="https://user-images.githubusercontent.com/8209263/62232134-5b42d780-b37a-11e9-892f-a84ea8881e78.png"></details> | 20.09 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62067100-c6a87000-b1e7-11e9-919e-ad68e4166213.png"></details> | 18.34 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62100450-9fcd5680-b246-11e9-8170-2ad4473e8294.png"></details> | **20.44** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62230975-e2428080-b377-11e9-8970-6917ae80c0b4.png"></details> | 20.04 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020247-10a53d80-b177-11e9-9f0d-1433d4d87210.png"></details> | 19.66 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020342-6f6ab700-b177-11e9-824e-75f431dc14ec.png"></details> |
| Qbert <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62232149-60078b80-b37a-11e9-99bb-cedc9fe064d5.png"><img src="https://user-images.githubusercontent.com/8209263/62232148-60078b80-b37a-11e9-9610-17ac447a479f.png"></details> | 4,787.79 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62100455-a4920a80-b246-11e9-8ca5-d4dc1ce3d76f.png"></details> | 11,673.52 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62230986-e79fcb00-b377-11e9-8861-3686954b7e1a.png"></details> | 13,328.32 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020263-261a6780-b177-11e9-8936-22a74d2405d3.png"></details> | 13,259.19 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020347-742f6b00-b177-11e9-8bfb-edfcfd44c8b7.png"></details> | **13,691.89** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62067104-cb6d2400-b1e7-11e9-9c4f-9eaac265d7d6.png"></details> |
| Seaquest <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62232168-6bf34d80-b37a-11e9-9564-fa3609dc5c75.png"><img src="https://user-images.githubusercontent.com/8209263/62232167-6bf34d80-b37a-11e9-8db3-c79a0e78292b.png"></details> | 1,118.50 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62100462-a9ef5500-b246-11e9-8699-9356ff81ff93.png"></details> | **3,751.34** <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62230991-ebcbe880-b377-11e9-8de4-a01379d1d61c.png"></details> | 892.68 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020266-29adee80-b177-11e9-83c2-fafbdbb982b9.png"></details> | 1,686.08 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62020350-772a5b80-b177-11e9-8917-e3c8a745cd08.png"></details> | 1,583.04 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/62067113-cf994180-b1e7-11e9-870b-b9bba71f2a7e.png"></details> |


### Roboschool Benchmark

|||||
|:---:|:---:|:---:|:---:|
| ![sac ant](https://user-images.githubusercontent.com/8209263/63994867-ff6b8b80-caaa-11e9-971e-2fac1cddcbac.gif) | ![sac halfcheetah](https://user-images.githubusercontent.com/8209263/63994869-01354f00-caab-11e9-8e11-3893d2c2419d.gif) | ![sac hopper](https://user-images.githubusercontent.com/8209263/63994871-0397a900-caab-11e9-9566-4ca23c54b2d4.gif) | ![sac humanoid](https://user-images.githubusercontent.com/8209263/63994883-0befe400-caab-11e9-9bcc-c30c885aad73.gif) |
| Ant | HalfCheetah | Hopper | Humanoid |
| ![sac doublependulum](https://user-images.githubusercontent.com/8209263/63994879-07c3c680-caab-11e9-974c-06cdd25bfd68.gif) | ![sac pendulum](https://user-images.githubusercontent.com/8209263/63994880-085c5d00-caab-11e9-850d-049401540e3b.gif) | ![sac reacher](https://user-images.githubusercontent.com/8209263/63994881-098d8a00-caab-11e9-8e19-a3b32d601b10.gif) | ![sac walker](https://user-images.githubusercontent.com/8209263/63994882-0abeb700-caab-11e9-9e19-b59dc5c43393.gif) |
| Inv.DoublePendulum | InvertedPendulum | Reacher | Walker |

[Roboschool](https://github.com/openai/roboschool) by OpenAI offers free open source robotics simulations with improved physics. Although it mirrors the environments from MuJuCo, its environments' rewards are different.

>Click on the algorithm to see the result upload Pull Request.

**Legend:**
<img width="200" alt="legend" src="https://user-images.githubusercontent.com/8209263/65397799-df588080-dd67-11e9-8eb2-339c4d1491b7.png">

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


#### Asynchronous Algorithms Benchmark

>Click on the algorithm to see the result upload Pull Request.

The frames in the graphs are per worker, and graphs are averaged across workers. To get the total frames, simply multiply the x-axis with the number of sessions (workers).

| Env. \ Alg. | A3C (GAE) | A3C (n-step) | Async PPO | [Async SAC](https://github.com/kengz/SLM-Lab/pull/404) |
|:---|---|---|---|---|
| RoboschoolAnt |  |  |  | 2525.08 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63918070-d25aa280-c9f0-11e9-8672-2fa18987d936.png"></details> |
| RoboschoolAtlasForwardWalk |  |  |  | 1849.50 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63918273-3c734780-c9f1-11e9-993c-d22cb0740223.png"></details> |
| RoboschoolHalfCheetah |  |  |  | 2278.03 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63918328-56148f00-c9f1-11e9-9c77-21dfe22b581e.png"></details> |
| RoboschoolHopper |  |  |  | 2376.96 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63918357-662c6e80-c9f1-11e9-91dd-9668fd96aa5b.png"></details> |
| RoboschoolInvertedDoublePendulum |  |  |  | 8030.81 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63918390-75132100-c9f1-11e9-925a-c3d8e229ff78.png"></details> |
| RoboschoolInvertedPendulum |  |  |  | 966.41 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63918442-91af5900-c9f1-11e9-8aab-9e2165fd9ecd.png"></details> |
| RoboschoolInvertedPendulumSwingup |  |  |  | 847.06 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63918480-a1c73880-c9f1-11e9-8c19-b69c875c3143.png"></details> |
| RoboschoolReacher |  |  |  | 19.73 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63918535-b60b3580-c9f1-11e9-8e26-4cbdd3f3c42f.png"></details> |
| RoboschoolWalker2d |  |  |  | 1386.15 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63918584-ccb18c80-c9f1-11e9-9126-0c9e9d8ac04b.png"></details> |
| RoboschoolHumanoid |  |  |  |  2458.23 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63919458-a260ce80-c9f3-11e9-81ee-fdee30d293a7.png"></details> |
| RoboschoolHumanoidFlagrun |  |  |  |  2056.06 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63919472-a8ef4600-c9f3-11e9-9394-b150fd71e4a4.png"></details> |
| RoboschoolHumanoidFlagrunHarder |  |  |  |  267.36 <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/63919477-ac82cd00-c9f3-11e9-9038-29df678edc60.png"></details> |
