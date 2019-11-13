## Benchmark

|||||
|:---:|:---:|:---:|:---:|
| ![ppo beamrider](https://user-images.githubusercontent.com/8209263/63994698-689ecf00-caaa-11e9-991f-0a5e9c2f5804.gif) | ![ppo breakout](https://user-images.githubusercontent.com/8209263/63994695-650b4800-caaa-11e9-9982-2462738caa45.gif) | ![ppo kungfumaster](https://user-images.githubusercontent.com/8209263/63994690-60469400-caaa-11e9-9093-b1cd38cee5ae.gif) | ![ppo mspacman](https://user-images.githubusercontent.com/8209263/63994685-5cb30d00-caaa-11e9-8f35-78e29a7d60f5.gif) |
| BeamRider | Breakout | KungFuMaster | MsPacman |
| ![ppo pong](https://user-images.githubusercontent.com/8209263/63994680-59b81c80-caaa-11e9-9253-ed98370351cd.gif) | ![ppo qbert](https://user-images.githubusercontent.com/8209263/63994672-54f36880-caaa-11e9-9757-7780725b53af.gif) | ![ppo seaquest](https://user-images.githubusercontent.com/8209263/63994665-4dcc5a80-caaa-11e9-80bf-c21db818115b.gif) | ![ppo spaceinvaders](https://user-images.githubusercontent.com/8209263/63994624-15c51780-caaa-11e9-9c9a-854d3ce9066d.gif) |
| Pong | Qbert | Seaquest | Sp.Invaders |

SLM Lab provides a set of benchmark results that are periodically updated with new feature releases. All the results below link to their respective PRs with the full experiment reports. To see more:
- [the `result` PRs](https://github.com/kengz/SLM-Lab/pulls?utf8=%E2%9C%93&q=is%3Apr+label%3Aresult+).
- the full experiment datas contributed are [public on Dropbox ](https://www.dropbox.com/sh/urifraklxcvol70/AADxtt6zUNuVR6qe288JYNCNa?dl=0).

The data can be downloaded into SLM Lab's `data/` folder and [reran in enjoy mode](https://kengz.gitbooks.io/slm-lab/content/usage/lab-commands.html).

#### Hardware

For reference, the image based environment benchmarks are run on AWS GPU box `p2.16xlarge`, and the non-image based environments are run on AWS CPU box `m5a.24xlarge`.

#### Reproducibility

The benchmark tables in this page show the `Trial` level `final_return_ma` from SLM Lab. This is final value of the 100-ckpt moving average of the return (total rewards) from evaluation. Each `Trial` is ran with 4 `Session`s with different random seeds, and their `final_return_ma` are averaged on the `Trial` level.

The specs for these are contained in the [`slm_lab/spec/benchmark`](https://github.com/kengz/SLM-Lab/tree/master/slm_lab/spec/benchmark) folder, descriptively named `{algorithm}_{environment}.json`.

All the results are shown below and the data folders including the metrics and models are uploaded to the [SLM Lab public Dropbox](https://www.dropbox.com/sh/urifraklxcvol70/AADxtt6zUNuVR6qe288JYNCNa?dl=0)

## Benchmark Results

#### Environments

SLM Lab's benchmark includes environments from the following offerings:

- [OpenAI gym default environments](https://github.com/openai/gym)
- [OpenAI gym Atari environments](https://gym.openai.com/envs/#atari) offers a wrapper for the [Atari Learning Environment (ALE)](https://github.com/mgbellemare/Arcade-Learning-Environment)
- [OpenAI Roboschool](https://github.com/openai/roboschool)
- [Unity ML Agents](https://github.com/Unity-Technologies/ml-agents)

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

#### Plot Legend

><img width="400" alt="legend" src="https://user-images.githubusercontent.com/8209263/67737544-d727dc80-f9c8-11e9-904a-319b9aafd41b.png">


### Discrete Benchmark

- [Upload PR #427](https://github.com/kengz/SLM-Lab/pull/427)
- [Dropbox data](https://www.dropbox.com/s/az4vncwwktyotol/benchmark_discrete_2019_09.zip?dl=0)

||||||||
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Env. \ Alg. | DQN | DDQN+PER | A2C (GAE) | A2C (n-step) | PPO | SAC |
| Breakout <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67737546-dabb6380-f9c8-11e9-901e-b96cc28f1fdf.png"></details> | 80.88 | 182 | 377 | 398 | **443** | 3.51* |
| Pong <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67737554-e018ae00-f9c8-11e9-92b5-3bd8d213b1e0.png"></details> | 18.48 | 20.5 | 19.31 | 19.56 | **20.58** | 19.87* |
| Seaquest <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67737557-e3139e80-f9c8-11e9-9446-119593ca956b.png"></details> | 1185 | **4405** | 1070 | 1684 | 1715 | 171* |
| Qbert <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67737559-e575f880-f9c8-11e9-8c98-f14c82041a45.png"></details> | 5494 | 11426 | 12405 | **13590** | 13460 | 923* |
| LunarLander <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67737566-e7d85280-f9c8-11e9-8df8-39c1205c5308.png"></details> | 192 | 233 | 25.21 | 68.23 | 214 | **276** |
| UnityHallway <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67737569-ead34300-f9c8-11e9-9e26-61fe1d779989.png"></details> | -0.32 | 0.27 | 0.08 | -0.96 | **0.73** | 0.01 |
| UnityPushBlock <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67737577-eeff6080-f9c8-11e9-931c-843ba697779c.png"></details> | 4.88 | 4.93 | 4.68 | 4.93 | **4.97** | -0.70 |

>Episode score at the end of training attained by SLM Lab implementations on discrete-action control problems. Reported episode scores are the average over the last 100 checkpoints, and then averaged over 4 Sessions. A Random baseline with score averaged over 100 episodes is included. Results marked with `*` were trained using the hybrid synchronous/asynchronous version of SAC to parallelize and speed up training time. For SAC, Breakout, Pong and Seaquest were trained for 2M frames instead of 10M frames.

>For the full Atari benchmark, see [Atari Benchmark](https://github.com/kengz/SLM-Lab/blob/benchmark/BENCHMARK.md#atari-benchmark)

### Continuous Benchmark

- [Upload PR #427](https://github.com/kengz/SLM-Lab/pull/427)
- [Dropbox data](https://www.dropbox.com/s/xaxybertpwt4s9j/benchmark_continuous_2019_09.zip?dl=0)

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

>Episode score at the end of training attained by SLM Lab implementations on continuous control problems. Reported episode scores are the average over the last 100 checkpoints, and then averaged over 4 Sessions. Results marked with `*` require 50M-100M frames, so we use the hybrid synchronous/asynchronous version of SAC to parallelize and speed up training time.

### Atari Benchmark

- [Upload PR #427](https://github.com/kengz/SLM-Lab/pull/427)
- [Dropbox data: DQN](https://www.dropbox.com/s/5hg78znvmi41ys5/benchmark_dqn_atari_2019_09.zip?dl=0)
- [Dropbox data: DDQN+PER](https://www.dropbox.com/s/s8pgset1ewi0da1/benchmark_ddqn_per_atari_2019_09.zip?dl=0)
- [Dropbox data: A2C (GAE)](https://www.dropbox.com/s/kbqw9a5f0e55y0y/benchmark_a2c_gae_atari_2019_09.zip?dl=0)
- [Dropbox data: A2C (n-step)](https://www.dropbox.com/s/jc1xzd4uru6tksd/benchmark_a2c_nstep_atari_2019_09.zip?dl=0)
- [Dropbox data: PPO](https://www.dropbox.com/s/o42fsnfoef5y9zl/benchmark_ppo_atari_2019_09.zip?dl=0)
- [Dropbox data: all Atari graphs](https://www.dropbox.com/s/odxxr2cquw4bcfj/benchmark_atari_graphs_2019_09.zip?dl=0)

|||||||
|:---:|:---:|:---:|:---:|:---:|:---:|
| Env. \ Alg. | DQN | DDQN+PER | A2C (GAE) | A2C (n-step) | PPO |
| Adventure <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738131-d6904580-f9ca-11e9-8818-0d027b668a97.png"></details> | -0.94 | -0.92 | -0.77 | -0.85 | **-0.3** |
| AirRaid <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738132-d6904580-f9ca-11e9-9585-41f69fd8bb33.png"></details> | 1876 | 3974 | **4202** | 3557 | 4028 |
| Alien <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738133-d6904580-f9ca-11e9-8375-4c134255cfe1.png"></details> | 822 | 1574 | 1519 | **1627** | 1413 |
| Amidar <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738134-d6904580-f9ca-11e9-865c-eb41f4e712f9.png"></details> | 90.95 | 431 | 577 | 418 | **795** |
| Assault <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738135-d6904580-f9ca-11e9-8f8d-61732ecc3ce4.png"></details> | 1392 | 2567 | 3366 | 3312 | **3619** |
| Asterix <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738138-d6904580-f9ca-11e9-86c0-3589622a311c.png"></details> | 1253 | **6866** | 5559 | 5223 | 6132 |
| Asteroids <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738139-d728dc00-f9ca-11e9-8741-e9a59883197e.png"></details> | 439 | 426 | **2951** | 2147 | 2186 |
| Atlantis <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738140-d728dc00-f9ca-11e9-9649-ecc4b2db782f.png"></details> | 68679 | 644810 | **2747371** | 2259733 | 2148077 |
| BankHeist <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738141-d728dc00-f9ca-11e9-924a-a02be1639ee6.png"></details> | 131 | 623 | 855 | 1170 | **1183** |
| BattleZone <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738142-d728dc00-f9ca-11e9-82b0-382bbb0bcc6c.png"></details> | 6564 | 6395 | 4336 | 4533 | **13649** |
| BeamRider <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738143-d728dc00-f9ca-11e9-84eb-2ec8988ff545.png"></details> | 2799 | **5870** | 2659 | 4139 | 4299 |
| Berzerk <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738144-d728dc00-f9ca-11e9-83c6-2e50a69b4ed3.png"></details> | 319 | 401 | **1073** | 763 | 860 |
| Bowling <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738145-d7c17280-f9ca-11e9-9a2e-bc179e3186f4.png"></details> | 30.29 | **39.5** | 24.51 | 23.75 | 31.64 |
| Boxing <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738146-d7c17280-f9ca-11e9-95ac-008f35834ed1.png"></details> | 72.11 | 90.98 | 1.57 | 1.26 | **96.53** |
| Breakout <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738147-d7c17280-f9ca-11e9-890e-319a21e036e0.png"></details> | 80.88 | 182 | 377 | 398 | **443** |
| Carnival <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738148-d7c17280-f9ca-11e9-95e9-58309efb8ee4.png"></details> | 4280 | **4773** | 2473 | 1827 | 4566 |
| Centipede <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738150-d7c17280-f9ca-11e9-8a27-3cc7160c1e60.png"></details> | 1899 | 2153 | 3909 | 4202 | **5003** |
| ChopperCommand <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738151-d7c17280-f9ca-11e9-8316-90cf4e944e97.png"></details> | 1083 | **4020** | 3043 | 1280 | 3357 |
| CrazyClimber <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738152-d85a0900-f9ca-11e9-8b48-1a988dc31627.png"></details> | 46984 | 88814 | 106256 | 109998 | **116820** |
| Defender <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738153-d85a0900-f9ca-11e9-8b30-750fc49b25dd.png"></details> | 281999 | 313018 | **665609** | 657823 | 534639 |
| DemonAttack <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738154-d85a0900-f9ca-11e9-8e5e-e99b336e6fbb.png"></details> | 1705 | 19856 | 23779 | 19615 | **121172** |
| DoubleDunk <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738155-d85a0900-f9ca-11e9-8fd4-e94d1be4a6ee.png"></details> | -21.44 | -22.38 | **-5.15** | -13.3 | -6.01 |
| ElevatorAction <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738156-d85a0900-f9ca-11e9-9006-903a9c823230.png"></details> | 32.62 | 17.91 | **9966** | 8818 | 6471 |
| Enduro <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738158-d85a0900-f9ca-11e9-8167-ebc713c59fdc.png"></details> | 437 | 959 | 787 | 0.0 | **1926** |
| FishingDerby <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738159-d8f29f80-f9ca-11e9-9166-ebe3ea5339ab.png"></details> | -88.14 | -1.7 | 16.54 | 1.65 | **36.03** |
| Freeway <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738161-d8f29f80-f9ca-11e9-9727-2584ac850507.png"></details> | 24.46 | 30.49 | 30.97 | 0.0 | **32.11** |
| Frostbite <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738163-d8f29f80-f9ca-11e9-9d36-1cb7985360ac.png"></details> | 98.8 | **2497** | 277 | 261 | 1062 |
| Gopher <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738164-d8f29f80-f9ca-11e9-8ba3-fb1d75ef81f1.png"></details> | 1095 | **7562** | 929 | 1545 | 2933 |
| Gravitar <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738166-d8f29f80-f9ca-11e9-9d57-c02118eba7c1.png"></details> | 87.34 | 258 | 313 | **433** | 223 |
| Hero <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738167-d8f29f80-f9ca-11e9-9faf-2c30048c8621.png"></details> | 1051 | 12579 | 16502 | **19322** | 17412 |
| IceHockey <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738168-d98b3600-f9ca-11e9-8695-8014fd177416.png"></details> | -14.96 | -14.24 | **-5.79** | -6.06 | -6.43 |
| Jamesbond <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738170-d98b3600-f9ca-11e9-9f4a-25929639efc1.png"></details> | 44.87 | **702** | 521 | 453 | 561 |
| JourneyEscape <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738171-d98b3600-f9ca-11e9-9679-15a1586719dd.png"></details> | -4818 | -2003 | **-921** | -2032 | -1094 |
| Kangaroo <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738172-d98b3600-f9ca-11e9-9770-3d63043a716b.png"></details> | 1965 | **8897** | 67.62 | 554 | 4989 |
| Krull <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738173-d98b3600-f9ca-11e9-9244-0933adbfedd8.png"></details> | 5522 | 6650 | 7785 | 6642 | **8477** |
| KungFuMaster <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738174-d98b3600-f9ca-11e9-95e3-33621db77541.png"></details> | 2288 | 16547 | 31199 | 25554 | **34523** |
| MontezumaRevenge <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738175-da23cc80-f9ca-11e9-81cf-58e16e210b5e.png"></details> | 0.0 | 0.02 | 0.08 | 0.19 | **1.08** |
| MsPacman <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738176-da23cc80-f9ca-11e9-8906-d54475705442.png"></details> | 1175 | 2215 | 1965 | 2158 | **2350** |
| NameThisGame <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738177-da23cc80-f9ca-11e9-9093-0a0e2456fb4c.png"></details> | 3915 | 4474 | 5178 | 5795 | **6386** |
| Phoenix <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738178-da23cc80-f9ca-11e9-93a1-188c75b888f6.png"></details> | 2909 | 8179 | 16345 | 13586 | **30504** |
| Pitfall <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738179-da23cc80-f9ca-11e9-8c76-0d339ac0034a.png"></details> | -68.83 | -73.65 | -101 | **-31.13** | -35.93 |
| Pong <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738180-dabc6300-f9ca-11e9-826b-3d72cd0b13a0.png"></details> | 18.48 | 20.5 | 19.31 | 19.56 | **20.58** |
| Pooyan <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738181-dabc6300-f9ca-11e9-922e-0b13b973a4d9.png"></details> | 1958 | 2741 | 2862 | 2531 | **6799** |
| PrivateEye <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738182-dabc6300-f9ca-11e9-87b3-072ce2637405.png"></details> | **784** | 303 | 93.22 | 78.07 | 50.12 |
| Qbert <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738183-dabc6300-f9ca-11e9-8ab1-d66c6b12cd2f.png"></details> | 5494 | 11426 | 12405 | **13590** | 13460 |
| Riverraid <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738184-dabc6300-f9ca-11e9-82fb-d6b7f7f0d696.png"></details> | 953 | **10492** | 8308 | 7565 | 9636 |
| RoadRunner <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738185-dabc6300-f9ca-11e9-9291-1303718c9a50.png"></details> | 15237 | 29047 | 30152 | 31030 | **32956** |
| Robotank <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738186-db54f980-f9ca-11e9-8aef-41c9a3250d8c.png"></details> | 3.43 | **9.05** | 2.98 | 2.27 | 2.27 |
| Seaquest <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738187-db54f980-f9ca-11e9-9764-da60d54e1406.png"></details> | 1185 | **4405** | 1070 | 1684 | 1715 |
| Skiing <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738188-db54f980-f9ca-11e9-9966-1f22f57a96e0.png"></details> | -14094 | **-12883** | -19481 | -14234 | -24713 |
| Solaris <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738190-db54f980-f9ca-11e9-84c6-8bc1313e1e96.png"></details> | 612 | 1396 | 2115 | **2236** | 1892 |
| SpaceInvaders <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738191-dbed9000-f9ca-11e9-84e9-ec324d7b2544.png"></details> | 451 | 670 | 733 | 750 | **797** |
| StarGunner <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738193-dbed9000-f9ca-11e9-9d01-42865df8ca1e.png"></details> | 3565 | 38238 | 44816 | 48410 | **60579** |
| Tennis <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738194-dbed9000-f9ca-11e9-84c4-aaf8c59371a2.png"></details> | -23.78 | **-10.33** | -22.42 | -19.06 | -11.52 |
| TimePilot <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738195-dbed9000-f9ca-11e9-8bea-33ed2428afe2.png"></details> | 2819 | 1884 | 3331 | 3440 | **4398** |
| Tutankham <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738196-dc862680-f9ca-11e9-8beb-144e4fb4b36d.png"></details> | 35.03 | 159 | 161 | 175 | **211** |
| UpNDown <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738197-dc862680-f9ca-11e9-9903-d1eb924f56e2.png"></details> | 2043 | 11632 | 89769 | 18878 | **262208** |
| Venture <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738198-dc862680-f9ca-11e9-8c37-04e057822a20.png"></details> | 4.56 | 9.61 | 0.0 | 0.0 | **11.84** |
| VideoPinball <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738199-dc862680-f9ca-11e9-9ab3-50064bd5112c.png"></details> | 8056 | **79730** | 35371 | 40423 | 58096 |
| WizardOfWor <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738200-dc862680-f9ca-11e9-8722-67a664dbbf10.png"></details> | 869 | 328 | 1516 | 1247 | **4283** |
| YarsRevenge <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738201-dd1ebd00-f9ca-11e9-9c27-3a8dd8c13953.png"></details> | 5816 | 15698 | **27097** | 11742 | 10114 |
| Zaxxon <details><summary><i>graph</i></summary><img src="https://user-images.githubusercontent.com/8209263/67738202-dd1ebd00-f9ca-11e9-98bd-f737a02107f9.png"></details> | 442 | 54.28 | 64.72 | 24.7 | **641** |

>The table above presents results for 62 Atari games. All agents were trained for 10M frames (40M including skipped frames). Reported results are the episode score at the end of training, averaged over the previous 100 evaluation checkpoints with each checkpoint averaged over 4 Sessions. Agents were checkpointed every 10k training frames.
