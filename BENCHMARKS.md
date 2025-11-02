# SLM-Lab Official Benchmarking

**Purpose**: Systematic algorithm validation across Gymnasium environments
**Status**: 🔄 Phase 1 In Progress
**Last Updated**: 2025-10-12

---

## Quick Start

1. **Current Phase**: Check [Phase Tracker](#phase-tracker)
2. **Run Experiment**: Use commands in phase sections below
3. **Record Results**: Fill in tables for each environment
4. **Next Steps**: Update phase tracker and continue

---

## Phase Tracker

| Phase | Category        | Environments | Status | Progress                   |
| ----- | --------------- | ------------ | ------ | -------------------------- |
| **1** | Classic Control | 5 envs       | 🔄     | 1/5 complete (CartPole ✅) |
| **2** | Box2D           | 3 envs       | ⏸️     | 0/3 complete               |
| **3** | MuJoCo          | 9 envs       | ⏸️     | 0/9 complete               |
| **4** | Atari           | 6+ envs      | ⏸️     | 0/6 complete               |

**Current Focus**: Phase 1.2 (Acrobot-v1)
**Started**: 2025-10-10

---

## Core Algorithms

**Mainstream Algorithms** (primary benchmarking):

- **REINFORCE**: Vanilla policy gradient (CartPole only - educational baseline)
- **A2C**: Advantage Actor-Critic with GAE (CartPole only - educational baseline)
- **PPO**: Proximal Policy Optimization (all environments - strongest general-purpose algorithm)
- **SARSA**: On-policy TD learning (CartPole only - educational baseline)
- **DQN**: Deep Q-Network (discrete only, use DDQN+PER for best results)
- **DDQN+PER**: Double DQN with Prioritized Experience Replay (strongest DQN variant)
- **SAC**: Soft Actor-Critic (discrete and continuous - rivals PPO for continuous control)

**Excluded**:

- **SIL**: Not mainstream, incompatible with dense rewards
- **A3C/Async SAC/DPPO**: Distributed variants (specialized use)

**Algorithm Selection by Environment**:

**CartPole only** (educational baselines):

- REINFORCE, A2C (GAE), SARSA, DQN, PPO

**Discrete environments** (Acrobot, MountainCar, Lunar, Atari):

- **DQN**: Baseline value-based
- **DDQN+PER**: Enhanced DQN (Double DQN + Prioritized Experience Replay)
- **A2C (GAE)**: GAE advantage estimation
- **A2C (n-step)**: n-step returns (alternative)
- **PPO**: Primary policy gradient
- **SAC**: Discrete action variant (experimental - underperforms on simple tasks like CartPole)

**Continuous environments** (Pendulum, BipedalWalker, MuJoCo):

- **A2C (GAE)**: GAE advantage estimation
- **A2C (n-step)**: n-step returns (alternative)
- **PPO**: Primary on-policy algorithm
- **SAC**: Primary off-policy algorithm (rivals PPO)

---

## Phase 1: Classic Control

**Goal**: Validate algorithm correctness on simple physics tasks
**Environments**: CartPole, Acrobot, MountainCar (discrete & continuous), Pendulum
**Doc**: https://gymnasium.farama.org/environments/classic_control/

### Phase 1.1: CartPole-v1 ✅ COMPLETE

- **Environment**: https://gymnasium.farama.org/environments/classic_control/cart_pole/
- **Action Space**: Discrete(2) - push left/right
- **State Space**: Box(4) - position, velocity, angle, angular velocity
- **max_frames**: 200k
- **log_frequency**: 500
- **Target total_reward_ma**: > 400

| Algorithm     | MA    | FPS  | Spec File                                                                           | Spec Name          | Status | Notes                                   |
| ------------- | ----- | ---- | ----------------------------------------------------------------------------------- | ------------------ | ------ | --------------------------------------- |
| **PPO**       | 499.7 | 315  | [ppo_cartpole.json](slm_lab/spec/benchmark/ppo/ppo_cartpole.json)                   | `ppo_cartpole`     | ✅     | 124.9% of target                        |
| **A2C**       | 488.7 | 3.5k | [a2c_gae_cartpole.json](slm_lab/spec/benchmark/a2c/a2c_gae_cartpole.json)           | `a2c_gae_cartpole` | ✅     | 122.2% of target                        |
| **DQN**       | 437.8 | 1k   | [dqn_cartpole.json](slm_lab/spec/benchmark/dqn/dqn_cartpole.json)                   | `dqn_boltzmann_cartpole` | ✅     | 109.5% of target, 3-stage ASHA          |
| **REINFORCE** | 427.2 | 14k  | [reinforce_cartpole.json](slm_lab/spec/benchmark/reinforce/reinforce_cartpole.json) | `reinforce_cartpole` | ✅     | 106.8% of target                        |
| **SAC**       | 431.1 | <100 | [sac_cartpole.json](slm_lab/spec/benchmark/sac/sac_cartpole.json)                   | `sac_cartpole`     | 🔄     | 107.8% of target, ASHA speed search running |
| **SARSA**     | 393.2 | ~7k  | [sarsa_cartpole.json](slm_lab/spec/benchmark/sarsa/sarsa_cartpole.json)             | `sarsa_epsilon_greedy_cartpole` | ✅     | 98.3% of target, 3-stage ASHA           |

---

### Phase 1.2: Acrobot-v1

- **Environment**: https://gymnasium.farama.org/environments/classic_control/acrobot/
- **Action Space**: Discrete(3) - apply torque (-1, 0, +1)
- **State Space**: Box(6) - link positions and angular velocities
- **max_frames**: 300k
- **log_frequency**: 500
- **Target total_reward_ma**: > -100

| Algorithm           | MA     | FPS  | Spec File                                                                 | Spec Name                    | Status | Notes                            |
| ------------------- | ------ | ---- | ------------------------------------------------------------------------- | ---------------------------- | ------ | -------------------------------- |
| **DQN (Boltzmann)** | -96.2  | ~600 | [dqn_acrobot.json](slm_lab/spec/benchmark/dqn/dqn_acrobot.json)           | `dqn_boltzmann_acrobot`      | ✅     | Best - 3.8% better than target   |
| **PPO**             | -80.8  | 777  | [ppo_acrobot.json](slm_lab/spec/benchmark/ppo/ppo_acrobot.json)           | `ppo_acrobot`                | ✅     | Solves target                    |
| **DDQN+PER**        | -83.0  | ~700 | [ddqn_per_acrobot.json](slm_lab/spec/benchmark/dqn/ddqn_per_acrobot.json) | `ddqn_per_acrobot`           | ✅     | Solves target                    |
| **A2C**             | -84.2  | 3.4k | [a2c_gae_acrobot.json](slm_lab/spec/benchmark/a2c/a2c_gae_acrobot.json)   | `a2c_gae_acrobot`            | ✅     | Solves target                    |
| **DQN (ε-greedy)**  | -104.0 | ~720 | [dqn_acrobot.json](slm_lab/spec/benchmark/dqn/dqn_acrobot.json)           | `dqn_epsilon_greedy_acrobot` | ✅     | Misses target (4% below)         |
| **SAC**             | -      | -    | [sac_acrobot.json](slm_lab/spec/benchmark/sac/sac_acrobot.json)           | `sac_acrobot`                | ⏸️     | Discrete action variant          |

---

### Phase 1.3: MountainCar-v0

- **Environment**: https://gymnasium.farama.org/environments/classic_control/mountain_car/
- **Action Space**: Discrete(3) - accelerate left/none/right
- **State Space**: Box(2) - position, velocity
- **max_frames**: TBD
- **log_frequency**: 500
- **Target total_reward_ma**: > -110

| Algorithm    | MA  | FPS | Spec File                                                                         | Spec Name              | Status | Notes                   |
| ------------ | --- | --- | --------------------------------------------------------------------------------- | ---------------------- | ------ | ----------------------- |
| **PPO**      | -   | -   | [ppo_mountaincar.json](slm_lab/spec/benchmark/ppo/ppo_mountaincar.json)           | `ppo_mountaincar`      | ⏸️     | Primary                 |
| **A2C**      | -   | -   | [a2c_mountaincar.json](slm_lab/spec/benchmark/a2c/a2c_mountaincar.json)           | `a2c_mountaincar`      | ⏸️     | Secondary               |
| **DQN**      | -   | -   | [dqn_mountaincar.json](slm_lab/spec/benchmark/dqn/dqn_mountaincar.json)           | `dqn_mountaincar`      | ⏸️     | Baseline                |
| **DDQN+PER** | -   | -   | [ddqn_per_mountaincar.json](slm_lab/spec/benchmark/dqn/ddqn_per_mountaincar.json) | `ddqn_per_mountaincar` | ⏸️     | Enhanced DQN            |
| **SAC**      | -   | -   | [sac_mountaincar.json](slm_lab/spec/benchmark/sac/sac_mountaincar.json)           | `sac_mountaincar`      | ⏸️     | Discrete action variant |

---

### Phase 1.4: MountainCarContinuous-v0

- **Environment**: https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/
- **Action Space**: Box(1) - continuous force [-1, 1]
- **State Space**: Box(2) - position, velocity
- **max_frames**: TBD
- **log_frequency**: 500
- **Target total_reward_ma**: > 90

| Algorithm | MA  | FPS | Spec File                                                 | Spec Name                    | Status | Notes              |
| --------- | --- | --- | --------------------------------------------------------- | ---------------------------- | ------ | ------------------ |
| **PPO**   | -   | -   | [ppo_cont.json](slm_lab/spec/benchmark/ppo/ppo_cont.json) | `ppo_mountaincar_continuous` | ⏸️     | Primary            |
| **SAC**   | -   | -   | [sac_cont.json](slm_lab/spec/benchmark/sac/sac_cont.json) | `sac_mountaincar_continuous` | ⏸️     | Continuous control |

---

### Phase 1.5: Pendulum-v1

- **Environment**: https://gymnasium.farama.org/environments/classic_control/pendulum/
- **Action Space**: Box(1) - torque [-2, 2]
- **State Space**: Box(3) - cos(theta), sin(theta), angular velocity
- **max_frames**: TBD
- **log_frequency**: 500
- **Target total_reward_ma**: > -200

| Algorithm | MA  | FPS | Spec File                                                 | Spec Name      | Status | Notes                        |
| --------- | --- | --- | --------------------------------------------------------- | -------------- | ------ | ---------------------------- |
| **SAC**   | -   | -   | [sac_cont.json](slm_lab/spec/benchmark/sac/sac_cont.json) | `sac_pendulum` | ⏸️     | Recommended                  |
| **PPO**   | -   | -   | [ppo_cont.json](slm_lab/spec/benchmark/ppo/ppo_cont.json) | `ppo_pendulum` | ⏸️     | Test only (expected to fail) |

---

## Phase 2: Box2D

**Goal**: Physics-based control with Box2D simulator
**Environments**: LunarLander, BipedalWalker, CarRacing
**Doc**: https://gymnasium.farama.org/environments/box2d/
**Prerequisites**: `pip install swig && pip install gymnasium[box2d]`

### Phase 2.1: LunarLander-v3 (Discrete)

- **Environment**: https://gymnasium.farama.org/environments/box2d/lunar_lander/
- **Action Space**: Discrete(4) - no-op, fire left/main/right engine
- **State Space**: Box(8) - position, velocity, angle, angular velocity, leg contact
- **max_frames**: TBD
- **log_frequency**: 1000 (episodes ~400-500 steps)
- **Target total_reward_ma**: > 200

| Algorithm     | MA  | FPS | Spec File                                                             | Spec Name               | Status | Notes                   |
| ------------- | --- | --- | --------------------------------------------------------------------- | ----------------------- | ------ | ----------------------- |
| **PPO**       | -   | -   | [ppo_lunar.json](slm_lab/spec/benchmark/ppo/ppo_lunar.json)           | `ppo_lunar`             | ⏸️     | Primary                 |
| **DDQN+PER**  | -   | -   | [ddqn_per_lunar.json](slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json) | `ddqn_per_concat_lunar` | ⏸️     | Enhanced DQN            |
| **A2C (GAE)** | -   | -   | [a2c_gae_lunar.json](slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json)   | `a2c_gae_lunar`         | ⏸️     | Secondary               |
| **DQN**       | -   | -   | [dqn_lunar.json](slm_lab/spec/benchmark/dqn/dqn_lunar.json)           | `dqn_lunar`             | ⏸️     | Baseline                |
| **SAC**       | -   | -   | [sac_lunar.json](slm_lab/spec/benchmark/sac/sac_lunar.json)           | `sac_lunar`             | ⏸️     | Discrete action variant |

---

### Phase 2.1b: LunarLanderContinuous-v3

- **Environment**: https://gymnasium.farama.org/environments/box2d/lunar_lander/
- **Action Space**: Box(2) - main engine [-1, 1], side engines [-1, 1]
- **State Space**: Box(8) - position, velocity, angle, angular velocity, leg contact
- **max_frames**: TBD
- **log_frequency**: 1000
- **Target total_reward_ma**: > 200

| Algorithm     | MA  | FPS | Spec File                                                 | Spec Name                  | Status | Notes      |
| ------------- | --- | --- | --------------------------------------------------------- | -------------------------- | ------ | ---------- |
| **PPO**       | -   | -   | [ppo_cont.json](slm_lab/spec/benchmark/ppo/ppo_cont.json) | `ppo_lunar_continuous`     | ⏸️     | Primary    |
| **SAC**       | -   | -   | [sac_cont.json](slm_lab/spec/benchmark/sac/sac_cont.json) | `sac_lunar_continuous`     | ⏸️     | Off-policy |
| **A2C (GAE)** | -   | -   | [a2c_cont.json](slm_lab/spec/benchmark/a2c/a2c_cont.json) | `a2c_gae_lunar_continuous` | ⏸️     | Secondary  |

---

### Phase 2.2: BipedalWalker-v3

- **Environment**: https://gymnasium.farama.org/environments/box2d/bipedal_walker/
- **Action Space**: Box(4) - motor speeds for 4 joints [-1, 1]
- **State Space**: Box(24) - hull state, joint positions, velocities, lidar
- **max_frames**: TBD
- **log_frequency**: 1600 (episodes max 1600 steps)
- **Target total_reward_ma**: > 300

| Algorithm | MA  | FPS | Spec File                                                 | Spec Name           | Status | Notes       |
| --------- | --- | --- | --------------------------------------------------------- | ------------------- | ------ | ----------- |
| **PPO**   | -   | -   | [ppo_cont.json](slm_lab/spec/benchmark/ppo/ppo_cont.json) | `ppo_bipedalwalker` | ⏸️     | Primary     |
| **SAC**   | -   | -   | [sac_cont.json](slm_lab/spec/benchmark/sac/sac_cont.json) | `sac_bipedalwalker` | ⏸️     | Alternative |

---

### Phase 2.3: CarRacing-v3

- **Environment**: https://gymnasium.farama.org/environments/box2d/car_racing/
- **Action Space**: Box(3) or Discrete(5) - steering, gas, brake
- **State Space**: Box(96, 96, 3) - RGB image top-down view
- **max_frames**: TBD
- **log_frequency**: 1000 (episodes ~1000 steps)
- **Target total_reward_ma**: > 900

| Algorithm | MA  | FPS | Spec File                                                           | Spec Name       | Status | Notes    |
| --------- | --- | --- | ------------------------------------------------------------------- | --------------- | ------ | -------- |
| **PPO**   | -   | -   | [ppo_carracing.json](slm_lab/spec/benchmark/ppo/ppo_carracing.json) | `ppo_carracing` | ⏸️     | Optional |

---

## Phase 3: MuJoCo

**Goal**: Standard continuous control benchmarks with MuJoCo physics
**Doc**: https://gymnasium.farama.org/environments/mujoco/
**Prerequisites**: `pip install gymnasium[mujoco]`
**Hardware**: GPU recommended (use dstack for cloud)

### Core Environments

| Environment            | Action Dim | State Dim | Algorithms | Existing Specs | Priority |
| ---------------------- | ---------- | --------- | ---------- | -------------- | -------- |
| **HalfCheetah-v5**     | 6          | 17        | PPO, SAC   | ✓              | High     |
| **Ant-v5**             | 8          | 111       | PPO, SAC   | ✓              | High     |
| **Hopper-v5**          | 3          | 11        | PPO, SAC   | ✓              | High     |
| **Walker2d-v5**        | 6          | 17        | PPO, SAC   | ✓              | High     |
| **Humanoid-v5**        | 17         | 376       | PPO, SAC   | ✓              | Medium   |
| **HumanoidStandup-v5** | 17         | 376       | PPO, SAC   | -              | Low      |
| **Swimmer-v5**         | 2          | 8         | PPO, SAC   | -              | Low      |
| **Reacher-v4**         | 2          | 11        | PPO, SAC   | -              | Low      |
| **Pusher-v4**          | 7          | 23        | PPO, SAC   | -              | Low      |

**Environment URLs**:

- HalfCheetah: https://gymnasium.farama.org/environments/mujoco/half_cheetah/
- Ant: https://gymnasium.farama.org/environments/mujoco/ant/
- Hopper: https://gymnasium.farama.org/environments/mujoco/hopper/
- Walker2d: https://gymnasium.farama.org/environments/mujoco/walker2d/
- Humanoid: https://gymnasium.farama.org/environments/mujoco/humanoid/

### Phase 3.1-3.4: Core Suite (HalfCheetah, Ant, Hopper, Walker2d)

**Target**: Baseline performance established
**Max Frames**: TBD
**log_frequency**: 1000 (episodes ~1000 steps)
**Hardware**: GPU via dstack

| Environment        | PPO MA | SAC MA | Status | Notes      |
| ------------------ | ------ | ------ | ------ | ---------- |
| **HalfCheetah-v5** | -      | -      | ⏸️     | Locomotion |
| **Ant-v5**         | -      | -      | ⏸️     | Quadruped  |
| **Hopper-v5**      | -      | -      | ⏸️     | Monoped    |
| **Walker2d-v5**    | -      | -      | ⏸️     | Biped      |

---

### Phase 3.5: Humanoid

- **Environment**: https://gymnasium.farama.org/environments/mujoco/humanoid/
- **Action Space**: Box(17)
- **State Space**: Box(376)
- **max_frames**: TBD
- **log_frequency**: 1000
- **Target total_reward_ma**: > 6000

| Algorithm | MA  | FPS | Spec File                                                     | Spec Name    | Status | Notes       |
| --------- | --- | --- | ------------------------------------------------------------- | ------------ | ------ | ----------- |
| **PPO**   | -   | -   | [ppo_mujoco.json](slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | `ppo_mujoco` | ⏸️     | Complex     |
| **SAC**   | -   | -   | [sac_mujoco.json](slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` | ⏸️     | Alternative |

---

## Phase 4: Atari

**Goal**: Visual processing and high-dimensional state spaces
**Doc**: https://gymnasium.farama.org/environments/atari/
**Prerequisites**: `pip install gymnasium[atari]` or `pip install gymnasium[accept-rom-license]`
**Hardware**: GPU required (use dstack for cloud)

### Standard Benchmarks

| Environment      | Action Space | Algorithms | Existing Specs | Priority |
| ---------------- | ------------ | ---------- | -------------- | -------- |
| **Pong-v5**      | Discrete(6)  | PPO, DQN   | ✓              | High     |
| **Qbert-v5**     | Discrete(6)  | PPO, DQN   | ✓              | High     |
| **Breakout-v5**  | Discrete(4)  | PPO, DQN   | ✓              | High     |
| **Seaquest-v5**  | Discrete(18) | PPO, DQN   | -              | Medium   |
| **Enduro-v5**    | Discrete(9)  | PPO, DQN   | -              | Low      |
| **BeamRider-v5** | Discrete(9)  | PPO, DQN   | -              | Low      |

**Note**: All Atari environments use `ALE/[GameName]-v5` format (e.g., `ALE/Pong-v5`)

### Phase 4.1: Pong-v5

- **Environment**: https://gymnasium.farama.org/environments/atari/pong/
- **max_frames**: TBD
- **log_frequency**: 10000
- **Target total_reward_ma**: > 18

| Algorithm    | MA  | FPS | Spec File                                                           | Spec Name       | Status | Notes                   |
| ------------ | --- | --- | ------------------------------------------------------------------- | --------------- | ------ | ----------------------- |
| **PPO**      | -   | -   | [ppo_pong.json](slm_lab/spec/benchmark/ppo/ppo_pong.json)           | `ppo_pong`      | ⏸️     | Primary                 |
| **DQN**      | -   | -   | [dqn_pong.json](slm_lab/spec/benchmark/dqn/dqn_pong.json)           | `dqn_pong`      | ⏸️     | Value-based             |
| **DDQN+PER** | -   | -   | [ddqn_per_pong.json](slm_lab/spec/benchmark/dqn/ddqn_per_pong.json) | `ddqn_per_pong` | ⏸️     | Enhanced                |
| **SAC**      | -   | -   | [sac_pong.json](slm_lab/spec/benchmark/sac/sac_pong.json)           | `sac_pong`      | ⏸️     | Discrete action variant |

---

### Phase 4.2: Qbert-v5

- **Environment**: https://gymnasium.farama.org/environments/atari/qbert/
- **max_frames**: TBD
- **log_frequency**: 10000
- **Target total_reward_ma**: > 15000

| Algorithm    | MA  | FPS | Spec File                                                             | Spec Name        | Status | Notes                   |
| ------------ | --- | --- | --------------------------------------------------------------------- | ---------------- | ------ | ----------------------- |
| **PPO**      | -   | -   | [ppo_qbert.json](slm_lab/spec/benchmark/ppo/ppo_qbert.json)           | `ppo_qbert`      | ⏸️     | Primary                 |
| **DQN**      | -   | -   | [dqn_qbert.json](slm_lab/spec/benchmark/dqn/dqn_qbert.json)           | `dqn_qbert`      | ⏸️     | Value-based             |
| **DDQN+PER** | -   | -   | [ddqn_per_qbert.json](slm_lab/spec/benchmark/dqn/ddqn_per_qbert.json) | `ddqn_per_qbert` | ⏸️     | Enhanced                |
| **SAC**      | -   | -   | [sac_qbert.json](slm_lab/spec/benchmark/sac/sac_qbert.json)           | `sac_qbert`      | ⏸️     | Discrete action variant |

---

### Phase 4.3: Breakout-v5

- **Environment**: https://gymnasium.farama.org/environments/atari/breakout/
- **max_frames**: TBD
- **log_frequency**: 10000
- **Target total_reward_ma**: > 400

| Algorithm | MA  | FPS | Spec File                                                   | Spec Name   | Status | Notes                   |
| --------- | --- | --- | ----------------------------------------------------------- | ----------- | ------ | ----------------------- |
| **PPO**   | -   | -   | [ppo_atari.json](slm_lab/spec/benchmark/ppo/ppo_atari.json) | `ppo_atari` | ⏸️     | Primary                 |
| **DQN**   | -   | -   | [dqn_atari.json](slm_lab/spec/benchmark/dqn/dqn_atari.json) | `dqn_atari` | ⏸️     | Value-based             |
| **SAC**   | -   | -   | [sac_atari.json](slm_lab/spec/benchmark/sac/sac_atari.json) | `sac_atari` | ⏸️     | Discrete action variant |

---

### Phase 4.4+: Additional Atari (Optional)

**Environments**: Seaquest, Enduro, BeamRider, etc.
**Priority**: Low (after core benchmarks complete)

---

## Three-Stage ASHA Methodology

**When to use**: Algorithm fails to reach target on first baseline run

### Stage 1: Manual Iteration (Fast Validation)

**Goal**: Quick validation with sensible defaults
**Time**: 1-2 iterations, <30 min total

**Process**:

1. Start with paper defaults or proven library configs (SB3, CleanRL)
2. Run quick validation trial (`max_session=1-4`)
3. Identify critical failures (e.g., divergence, no learning)
4. Adjust obvious issues (learning rate, training frequency)

---

### Stage 2: ASHA Wide Exploration

**Goal**: Find promising hyperparameter ranges
**Config**:

- `max_session=1` (REQUIRED for ASHA)
- `max_trial=20-30`
- `search_scheduler` enabled with early termination
- Wide search spaces (uniform, loguniform distributions)

**Example**:

```json
{
  "meta": {
    "max_session": 1,
    "max_trial": 30,
    "search_scheduler": {
      "grace_period": 30000,
      "reduction_factor": 3
    }
  },
  "search": {
    "agent.algorithm.gamma__uniform": [0.95, 0.999],
    "agent.net.optim_spec.lr__loguniform": [1e-5, 5e-3],
    "agent.net.hid_layers__choice": [[64], [128], [256], [64, 64], [128, 128]]
  }
}
```

**Analysis**:

1. Review `data/experiment_df.csv`
2. Sort by `total_reward_ma` descending
3. Identify top 3-5 trials
4. Note patterns in successful hyperparameters

---

### Stage 3: Multi-Session Refinement

**Goal**: Validate best ranges with robust statistics
**Config**:

- `max_session=4` (multi-session averaging)
- `max_trial=8-12`
- **NO** `search_scheduler` (incompatible with multi-session)
- Narrowed search spaces around Stage 2 winners

**Example**:

```json
{
  "meta": {
    "max_session": 4,
    "max_trial": 8
  },
  "search": {
    "agent.algorithm.gamma__choice": [0.97, 0.98, 0.99],
    "agent.net.optim_spec.lr__choice": [0.0001, 0.00015, 0.0003],
    "agent.net.hid_layers__choice": [[128, 128], [256]]
  }
}
```

**Final Step**:

1. Select best trial from Stage 3 (highest MA with low variance)
2. Update base spec with winning hyperparameters
3. Validate with single training run
4. Commit to repository

**Key Insight**: ASHA and multi-session are **mutually exclusive** - ASHA requires single session for early termination, multi-session requires full runs for robust averaging.

---

## Known Issues & Limitations

### DQN Compute Inefficiency ⚠️ RESOLVED ✅

- **Issue**: DQN/DDQN were 84x slower than A2C and 9x slower than PPO in wall-clock time
- **Cause**: Excessive gradient updates per environment step
  - Original: `training_batch_iter: 8`, `training_iter: 4` → 32 updates per 4 steps = **10 updates/step**
  - Standard DQN: ~1 update per step
- **Evidence** (Acrobot-v1, 300k frames):
  - A2C: 34 seconds, 4.7k optimizer steps
  - PPO: 318 seconds, 93k optimizer steps
  - DQN: 2,850 seconds, 2.4M optimizer steps (50-500x more than necessary)
- **Fix Validated**:
  - CartPole-v1 (200k frames): MA 429.5 @ 198s (1,010 FPS) vs original MA 450 @ 3,000s (66 FPS)
  - Acrobot-v1 (300k frames):
    - DQN: MA -104.0 @ ~780s (720 FPS) vs original -105.8 @ 2,850s (95 FPS)
    - DDQN+PER: MA -83.0 @ ~780s (700 FPS) vs original -95.0 @ 2,850s (95 FPS)
  - **Result**: 3.7-15x speedup with equivalent or better learning
- **Status**: ✅ Applied to all DQN/DDQN specs

### SIL (Self-Imitation Learning)

- **Status**: ❌ Removed from benchmarking
- **Cause**: Incompatible with dense reward environments (CartPole, etc.)
- **Reason**: Requires sparse rewards where occasional good experiences can be self-imitated
- **Action**: Use PPO/A2C/DQN for dense rewards

### SAC on Discrete Sparse Rewards

- **Issue**: Poor performance (Lunar MA ~20)
- **Cause**: GumbelSoftmax discrete adaptation suboptimal
- **Action**: Use PPO/DQN for discrete tasks

### A2C on Complex Sparse Rewards

- **Issue**: High variance, fails to solve Lunar despite extensive search
- **Cause**: Algorithm limitation on sparse rewards
- **Action**: Use PPO/DDQN instead for moderate/complex environments

### PPO on Pendulum

- **Issue**: Known to be unsuitable
- **Cause**: Algorithmic mismatch
- **Action**: Use SAC for Pendulum

### REINFORCE, A2C & SARSA Scalability

- **Limitation**: Suitable for CartPole only (educational baselines)
- **Reason**: High variance, sample inefficiency (REINFORCE/A2C), limited value (SARSA)
- **Action**: Exclude from all environments beyond CartPole

---

## Utility Commands

### Check Results

```bash
# View experiment summary
cat data/[experiment_name]/experiment_df.csv | column -t -s,

# List trials by performance
cat data/[experiment_name]/experiment_df.csv | sort -t, -k3 -nr | head -10

# View specific trial metrics
cat data/[experiment_name]/[trial_name]/session_metrics.csv
```

### Clean Up

```bash
# Stop Ray search
slm-lab --stop-ray

# Remove old experiment data
rm -rf data/[old_experiment_name]/

# Clean all old data (careful!)
find data/ -type d -mtime +7 -exec rm -rf {} +
```

### Git Workflow

```bash
# After successful benchmark
git add slm_lab/spec/benchmark/[algo]/
git commit -m "feat: [algo] [env] achieves MA [value]"

# Update this document
git add BENCHMARKS.md
git commit -m "docs: complete Phase [X] - [environment]"
```

---

## Summary Statistics

**Last Updated**: 2025-10-12

### Overall Progress

| Metric                   | Value     |
| ------------------------ | --------- |
| Phases Complete          | 0/4       |
| Environments Complete    | 1/23+     |
| Classic Control Progress | 20% (1/5) |
| Box2D Progress           | 0% (0/3)  |
| MuJoCo Progress          | 0% (0/9)  |
| Atari Progress           | 0% (0/6+) |

### Phase Completion

| Phase       | Environments | Complete | Percentage |
| ----------- | ------------ | -------- | ---------- |
| **Phase 1** | 5            | 1        | 20%        |
| **Phase 2** | 3            | 0        | 0%         |
| **Phase 3** | 9            | 0        | 0%         |
| **Phase 4** | 6+           | 0        | 0%         |

### Algorithm Performance (Phase 1 Only)

| Algorithm | Envs Tested | Envs Solved | Success Rate |
| --------- | ----------- | ----------- | ------------ |
| PPO       | 1           | 1           | 100%         |
| A2C       | 1           | 1           | 100%         |
| DQN       | 1           | 1           | 100%         |
| REINFORCE | 1           | 1           | 100%         |
| SARSA     | 1           | 1           | 100%         |
| SAC       | 0           | 0           | -            |

---

## Next Actions

**Immediate** (Phase 1.2):

- [ ] Create Acrobot spec files (PPO, DQN, A2C, SARSA)
- [ ] Run baseline Acrobot experiments
- [ ] Record results and update tables

**This Week** (Phase 1):

- [ ] Complete all Classic Control environments
- [ ] Identify algorithms requiring ASHA refinement
- [ ] Update specs with validated hyperparameters

**Upcoming** (Phase 2):

- [ ] Begin LunarLander validation
- [ ] BipedalWalker baseline
- [ ] Phase 1 summary analysis

**Future** (Phases 3-4):

- [ ] MuJoCo suite on dstack
- [ ] Atari suite on dstack
- [ ] Complete benchmarking documentation

---

_This is the single source of truth for SLM-Lab benchmarking. Update as experiments complete._
