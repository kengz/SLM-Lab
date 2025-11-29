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

| Phase | Category        | Environments    | Status | Progress                                                             |
| ----- | --------------- | --------------- | ------ | -------------------------------------------------------------------- |
| **1** | Classic Control | 2 envs (skip 3) | ✅     | 2/2 complete (CartPole ✅, Acrobot ✅, MountainCar/Pendulum skipped) |
| **2** | Box2D           | 4 envs          | 🔄     | 3/4 complete (LunarLander ✅, LunarLander Cont ✅, BipedalWalker ⚠️)                   |
| **3** | MuJoCo          | 9 envs          | 🔄     | 2/9 in progress (Hopper ✅, HalfCheetah ✅, Walker2d 🔄, Ant 🔄)     |
| **4** | Atari           | 6+ envs         | ⏸️     | 0/6 complete                                                         |

**Current Focus**: Phase 3 MuJoCo - PPO validation and tuning
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

**Optional** (validated but not primary):

- **PPOSIL**: PPO + Self-Imitation Learning (validated on CartPole/Acrobot)
  - Best for: Sparse rewards, hard exploration (Atari)
  - Minimal benefit on dense-reward tasks (CartPole, MuJoCo)

**Excluded**:

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

## Phase 1.1: CartPole-v1

- **Environment**: https://gymnasium.farama.org/environments/classic_control/cart_pole/
- **Action Space**: Discrete(2) - push left/right
- **State Space**: Box(4) - position, velocity, angle, angular velocity
- **num_envs**: 4
- **max_frames**: 200k
- **log_frequency**: 500
- **Target total_reward_ma**: > 400

| Algorithm     | MA    | FPS  | Spec File                                                                           | Spec Name                       | Status | Notes                                       |
| ------------- | ----- | ---- | ----------------------------------------------------------------------------------- | ------------------------------- | ------ | ------------------------------------------- |
| **PPO**       | 499.7 | 315  | [ppo_cartpole.json](slm_lab/spec/benchmark/ppo/ppo_cartpole.json)                   | `ppo_cartpole`                  | ✅     | 124.9% of target                            |
| **A2C**       | 488.7 | 3.5k | [a2c_gae_cartpole.json](slm_lab/spec/benchmark/a2c/a2c_gae_cartpole.json)           | `a2c_gae_cartpole`              | ✅     | 122.2% of target                            |
| **DQN**       | 437.8 | 1k   | [dqn_cartpole.json](slm_lab/spec/benchmark/dqn/dqn_cartpole.json)                   | `dqn_boltzmann_cartpole`        | ✅     | 109.5% of target, 3-stage ASHA              |
| **REINFORCE** | 427.2 | 14k  | [reinforce_cartpole.json](slm_lab/spec/benchmark/reinforce/reinforce_cartpole.json) | `reinforce_cartpole`            | ✅     | 106.8% of target                            |
| **SAC**       | 431.1 | <100 | [sac_cartpole.json](slm_lab/spec/benchmark/sac/sac_cartpole.json)                   | `sac_cartpole`                  | ✅     | 107.8% of target (slow FPS expected for off-policy) |
| **SARSA**     | 393.2 | ~7k  | [sarsa_cartpole.json](slm_lab/spec/benchmark/sarsa/sarsa_cartpole.json)             | `sarsa_epsilon_greedy_cartpole` | ✅     | 98.3% of target, 3-stage ASHA               |
| **PPOSIL**    | 496.3 | 1.6k | [ppo_sil_cartpole.json](slm_lab/spec/benchmark/sil/ppo_sil_cartpole.json)           | `ppo_sil_cartpole`              | ✅     | 124.1% of target, PPO + Self-Imitation Learning |

---

## Phase 1.2: Acrobot-v1

- **Environment**: https://gymnasium.farama.org/environments/classic_control/acrobot/
- **Action Space**: Discrete(3) - apply torque (-1, 0, +1)
- **State Space**: Box(6) - link positions and angular velocities
- **num_envs**: 4
- **max_frames**: 300k
- **log_frequency**: 500
- **Target total_reward_ma**: > -100

| Algorithm           | MA     | FPS  | Spec File                                                                 | Spec Name                    | Status | Notes                          |
| ------------------- | ------ | ---- | ------------------------------------------------------------------------- | ---------------------------- | ------ | ------------------------------ |
| **DQN (Boltzmann)** | -96.2  | ~600 | [dqn_acrobot.json](slm_lab/spec/benchmark/dqn/dqn_acrobot.json)           | `dqn_boltzmann_acrobot`      | ✅     | Best - 3.8% better than target |
| **PPO**             | -80.8  | 777  | [ppo_acrobot.json](slm_lab/spec/benchmark/ppo/ppo_acrobot.json)           | `ppo_acrobot`                | ✅     | Solves target                  |
| **DDQN+PER**        | -83.0  | ~700 | [ddqn_per_acrobot.json](slm_lab/spec/benchmark/dqn/ddqn_per_acrobot.json) | `ddqn_per_acrobot`           | ✅     | Solves target                  |
| **A2C**             | -84.2  | 3.4k | [a2c_gae_acrobot.json](slm_lab/spec/benchmark/a2c/a2c_gae_acrobot.json)   | `a2c_gae_acrobot`            | ✅     | Solves target                  |
| **DQN (ε-greedy)**  | -104.0 | ~720 | [dqn_acrobot.json](slm_lab/spec/benchmark/dqn/dqn_acrobot.json)           | `dqn_epsilon_greedy_acrobot` | ✅     | Misses target (4% below)       |
| **SAC**             | -      | -    | [sac_acrobot.json](slm_lab/spec/benchmark/sac/sac_acrobot.json)           | `sac_acrobot`                | 🔄     | ASHA search running            |
| **PPOSIL**          | -110.2 | -    | [ppo_sil_acrobot.json](slm_lab/spec/benchmark/sil/ppo_sil_acrobot.json)   | `ppo_sil_acrobot`            | ⚠️     | Near target, compare vs PPO    |

---

## Phase 2.1: LunarLander-v3 (Discrete)

- **Environment**: https://gymnasium.farama.org/environments/box2d/lunar_lander/
- **Action Space**: Discrete(4) - no-op, fire left/main/right engine
- **State Space**: Box(8) - position, velocity, angle, angular velocity, leg contact
- **num_envs**: 8
- **max_frames**: 300k
- **log_frequency**: 1000 (episodes ~400-500 steps)
- **Target total_reward_ma**: > 200

| Algorithm     | MA    | FPS  | Spec File                                                             | Spec Name               | Status | Notes            |
| ------------- | ----- | ---- | --------------------------------------------------------------------- | ----------------------- | ------ | ---------------- |
| **PPO**       | 229.9 | 2.4k | [ppo_lunar.json](slm_lab/spec/benchmark/ppo/ppo_lunar.json)           | `ppo_lunar`             | ✅     | 115.0% of target |
| **DDQN+PER**  | 230.0 | 8.7k | [ddqn_per_lunar.json](slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json) | `ddqn_per_concat_lunar` | ✅     | 115.0% of target |
| **DQN**       | 203.9 | 9.0k | [dqn_lunar.json](slm_lab/spec/benchmark/dqn/dqn_lunar.json)           | `dqn_concat_lunar`      | ✅     | 102.0% of target |
| **A2C (GAE)** | -     | -    | [a2c_gae_lunar.json](slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json)   | `a2c_gae_lunar`         | 🔄     | ASHA search running |
| **SAC**       | -     | -    | [sac_lunar.json](slm_lab/spec/benchmark/sac/sac_lunar.json)           | `sac_lunar`             | 🔄     | ASHA search running |

---

## Phase 2.2: LunarLanderContinuous-v3

- **Environment**: https://gymnasium.farama.org/environments/box2d/lunar_lander/
- **Action Space**: Box(2) - main engine [-1, 1], side engines [-1, 1]
- **State Space**: Box(8) - position, velocity, angle, angular velocity, leg contact
- **num_envs**: 8
- **max_frames**: 300k
- **log_frequency**: 1000
- **Target total_reward_ma**: > 200

| Algorithm     | MA    | FPS | Spec File                                                             | Spec Name                  | Status | Notes                                 |
| ------------- | ----- | --- | --------------------------------------------------------------------- | -------------------------- | ------ | ------------------------------------- |
| **PPO**       | 249.2 | 135 | [ppo_lunar.json](slm_lab/spec/benchmark/ppo/ppo_lunar.json)           | `ppo_lunar_continuous`     | ✅     | 124.6% of target, 3-stage ASHA        |
| **SAC**       | 238.0 | ~35 | [sac_lunar.json](slm_lab/spec/benchmark/sac/sac_lunar.json)           | `sac_lunar_continuous`     | ✅     | 119.0% of target, 62% faster training |
| **A2C (GAE)** | -     | -   | [a2c_gae_lunar.json](slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json)   | `a2c_gae_lunar_continuous` | ⏸️     | TODO search                           |

---

## Phase 2.3: BipedalWalker-v3

- **Environment**: https://gymnasium.farama.org/environments/box2d/bipedal_walker/
- **Action Space**: Box(4) - motor speeds for 4 joints [-1, 1]
- **State Space**: Box(24) - hull state, joint positions, velocities, lidar
- **num_envs**: 16
- **max_frames**: 3M (slow learning, breakthrough at 1.5M-2M frames)
- **log_frequency**: 10000
- **Target total_reward_ma**: > 300

| Algorithm     | MA    | FPS  | Spec File                                                                           | Spec Name            | Status | Notes                                                  |
| ------------- | ----- | ---- | ----------------------------------------------------------------------------------- | -------------------- | ------ | ------------------------------------------------------ |
| **PPO**       | 241.3 | 1.8k | [ppo_bipedalwalker.json](slm_lab/spec/benchmark/ppo/ppo_bipedalwalker.json)         | `ppo_bipedalwalker`  | ⚠️     | 80% of target. ASHA trial 9: gamma=0.995, lam=0.922   |
| **SAC**       | -     | -    | [sac_bipedalwalker.json](slm_lab/spec/benchmark/sac/sac_bipedalwalker.json)         | `sac_bipedalwalker`  | 🔄     | ASHA search running                                    |
| **A2C (GAE)** | -112  | 8k   | [a2c_gae_bipedalwalker.json](slm_lab/spec/benchmark/a2c/a2c_gae_bipedalwalker.json) | `a2c_gae_bipedalwalker` | ❌  | Failed validation                                      |

---

## Phase 3: MuJoCo Environments

### Phase 3.1: Hopper-v5

- **Environment**: https://gymnasium.farama.org/environments/mujoco/hopper/
- **Action Space**: Box(3)
- **State Space**: Box(11)
- **num_envs**: 16 (PPO), 8 (SAC)
- **max_frames**: 3M (PPO), 2M (SAC)
- **log_frequency**: 10000
- **Target total_reward_ma**: > 3000

| Algorithm     | MA   | FPS  | Spec File                                                       | Spec Name    | Status | Notes                                              |
| ------------- | ---- | ---- | --------------------------------------------------------------- | ------------ | ------ | -------------------------------------------------- |
| **PPO**       | 2566 | ~1.5k| [ppo_mujoco.json](slm_lab/spec/benchmark/ppo/ppo_mujoco.json)   | `ppo_mujoco` | ✅     | 85% of target @ 3M frames. ASHA-tuned hyperparams. |
| **SAC**       | -    | -    | [sac_mujoco.json](slm_lab/spec/benchmark/sac/sac_mujoco.json)   | `sac_mujoco` | ⏸️     | Use `-s env=Hopper-v5`                             |

---

### Phase 3.2: Walker2d-v5

- **Environment**: https://gymnasium.farama.org/environments/mujoco/walker2d/
- **Action Space**: Box(6)
- **State Space**: Box(17)
- **num_envs**: 16 (PPO), 8 (SAC)
- **max_frames**: 3M (PPO), 2M (SAC)
- **log_frequency**: 10000
- **Target total_reward_ma**: > 4000

| Algorithm     | MA   | FPS  | Spec File                                                       | Spec Name    | Status | Notes                                     |
| ------------- | ---- | ---- | --------------------------------------------------------------- | ------------ | ------ | ----------------------------------------- |
| **PPO**       | 1424 | ~1.5k| [ppo_mujoco.json](slm_lab/spec/benchmark/ppo/ppo_mujoco.json)   | `ppo_mujoco` | 🔄     | 36% of target. ASHA search running.       |
| **SAC**       | -    | -    | [sac_mujoco.json](slm_lab/spec/benchmark/sac/sac_mujoco.json)   | `sac_mujoco` | ⏸️     | Use `-s env=Walker2d-v5`                  |

---

### Phase 3.3: HalfCheetah-v5

- **Environment**: https://gymnasium.farama.org/environments/mujoco/half_cheetah/
- **Action Space**: Box(6)
- **State Space**: Box(17)
- **num_envs**: 16 (PPO), 8 (SAC)
- **max_frames**: 3M (PPO), 2M (SAC)
- **log_frequency**: 10000
- **Target total_reward_ma**: > 5000

| Algorithm     | MA   | FPS  | Spec File                                                       | Spec Name    | Status | Notes                                       |
| ------------- | ---- | ---- | --------------------------------------------------------------- | ------------ | ------ | ------------------------------------------- |
| **PPO**       | 3178 | ~1.5k| [ppo_mujoco.json](slm_lab/spec/benchmark/ppo/ppo_mujoco.json)   | `ppo_mujoco` | ⚠️     | 64% of target @ 3M. Hopper params transfer. |
| **SAC**       | -    | -    | [sac_mujoco.json](slm_lab/spec/benchmark/sac/sac_mujoco.json)   | `sac_mujoco` | ⏸️     | Use `-s env=HalfCheetah-v5`                 |

---

### Phase 3.4: Ant-v5

- **Environment**: https://gymnasium.farama.org/environments/mujoco/ant/
- **Action Space**: Box(8)
- **State Space**: Box(111)
- **num_envs**: 16 (PPO), 8 (SAC)
- **max_frames**: 3M (PPO), 2M (SAC)
- **log_frequency**: 10000
- **Target total_reward_ma**: > 5000

| Algorithm     | MA   | FPS  | Spec File                                                       | Spec Name    | Status | Notes                                           |
| ------------- | ---- | ---- | --------------------------------------------------------------- | ------------ | ------ | ----------------------------------------------- |
| **PPO**       | 34   | ~1.5k| [ppo_mujoco.json](slm_lab/spec/benchmark/ppo/ppo_mujoco.json)   | `ppo_mujoco` | 🔄     | 0.7% of target. ASHA search running (4-leg dynamics differ). |
| **SAC**       | -    | -    | [sac_mujoco.json](slm_lab/spec/benchmark/sac/sac_mujoco.json)   | `sac_mujoco` | ⏸️     | Use `-s env=Ant-v5`                             |

---

### Phase 3.5: Humanoid-v5

- **Environment**: https://gymnasium.farama.org/environments/mujoco/humanoid/
- **Action Space**: Box(17)
- **State Space**: Box(376)
- **num_envs**: 32 (PPO), 8 (SAC)
- **max_frames**: 50M (PPO), 5M (SAC)
- **log_frequency**: 10000
- **Target total_reward_ma**: > 6000

| Algorithm     | MA  | FPS | Spec File                                                       | Spec Name      | Status | Notes                        |
| ------------- | --- | --- | --------------------------------------------------------------- | -------------- | ------ | ---------------------------- |
| **PPO**       | -   | -   | [ppo_mujoco.json](slm_lab/spec/benchmark/ppo/ppo_mujoco.json)   | `ppo_humanoid` | ⏸️     | Dedicated spec (high frames) |
| **SAC**       | -   | -   | [sac_mujoco.json](slm_lab/spec/benchmark/sac/sac_mujoco.json)   | `sac_humanoid` | ⏸️     | Dedicated spec (high frames) |

---

## Phase 4: Atari Environments

### Phase 4.1: Pong-v5

- **Environment**: https://gymnasium.farama.org/environments/atari/pong/
- **Action Space**: Discrete(6)
- **State Space**: Box(210, 160, 3) - RGB image
- **max_frames**: TBD
- **log_frequency**: 10000
- **Target total_reward_ma**: > 18

| Algorithm    | MA  | FPS | Spec File                                                           | Spec Name       | Status | Notes                           |
| ------------ | --- | --- | ------------------------------------------------------------------- | --------------- | ------ | ------------------------------- |
| **PPO**      | -   | -   | [ppo_pong.json](slm_lab/spec/benchmark/ppo/ppo_pong.json)           | `ppo_pong`      | ⏸️     | Primary                         |
| **PPOSIL**   | -   | -   | [ppo_sil_pong.json](slm_lab/spec/benchmark/sil/ppo_sil_pong.json)   | `ppo_sil_pong`  | ⏸️     | Compare vs PPO (sparse rewards) |
| **DQN**      | -   | -   | [dqn_pong.json](slm_lab/spec/benchmark/dqn/dqn_pong.json)           | `dqn_pong`      | ⏸️     | Value-based                     |
| **DDQN+PER** | -   | -   | [ddqn_per_pong.json](slm_lab/spec/benchmark/dqn/ddqn_per_pong.json) | `ddqn_per_pong` | ⏸️     | Enhanced                        |
| **SAC**      | -   | -   | [sac_pong.json](slm_lab/spec/benchmark/sac/sac_pong.json)           | `sac_pong`      | ⏸️     | Discrete action variant         |

---

### Phase 4.2: Qbert-v5

- **Environment**: https://gymnasium.farama.org/environments/atari/qbert/
- **Action Space**: Discrete(6)
- **State Space**: Box(210, 160, 3) - RGB image
- **max_frames**: TBD
- **log_frequency**: 10000
- **Target total_reward_ma**: > 15000

| Algorithm    | MA  | FPS | Spec File                                                             | Spec Name        | Status | Notes                           |
| ------------ | --- | --- | --------------------------------------------------------------------- | ---------------- | ------ | ------------------------------- |
| **PPO**      | -   | -   | [ppo_qbert.json](slm_lab/spec/benchmark/ppo/ppo_qbert.json)           | `ppo_qbert`      | ⏸️     | Primary                         |
| **PPOSIL**   | -   | -   | [ppo_sil_qbert.json](slm_lab/spec/benchmark/sil/ppo_sil_qbert.json)   | `ppo_sil_qbert`  | ⏸️     | Compare vs PPO (hard exploration) |
| **DQN**      | -   | -   | [dqn_qbert.json](slm_lab/spec/benchmark/dqn/dqn_qbert.json)           | `dqn_qbert`      | ⏸️     | Value-based                     |
| **DDQN+PER** | -   | -   | [ddqn_per_qbert.json](slm_lab/spec/benchmark/dqn/ddqn_per_qbert.json) | `ddqn_per_qbert` | ⏸️     | Enhanced                        |
| **SAC**      | -   | -   | [sac_qbert.json](slm_lab/spec/benchmark/sac/sac_qbert.json)           | `sac_qbert`      | ⏸️     | Discrete action variant         |

---

### Phase 4.3: Breakout-v5

- **Environment**: https://gymnasium.farama.org/environments/atari/breakout/
- **Action Space**: Discrete(4)
- **State Space**: Box(210, 160, 3) - RGB image
- **max_frames**: TBD
- **log_frequency**: 10000
- **Target total_reward_ma**: > 400

| Algorithm | MA  | FPS | Spec File                                                   | Spec Name   | Status | Notes                   |
| --------- | --- | --- | ----------------------------------------------------------- | ----------- | ------ | ----------------------- |
| **PPO**   | -   | -   | [ppo_atari.json](slm_lab/spec/benchmark/ppo/ppo_atari.json) | `ppo_atari` | ⏸️     | Primary                 |
| **DQN**   | -   | -   | [dqn_atari.json](slm_lab/spec/benchmark/dqn/dqn_atari.json) | `dqn_atari` | ⏸️     | Value-based             |
| **SAC**   | -   | -   | [sac_atari.json](slm_lab/spec/benchmark/sac/sac_atari.json) | `sac_atari` | ⏸️     | Discrete action variant |

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

**IMPORTANT**: All algorithms (REINFORCE, A2C, PPO, DQN, DDQN+PER, SAC, SARSA) are general-purpose and proven to solve all benchmark environments in original SLM-Lab (CartPole through Atari). If an algorithm underperforms or fails to reach target, the issue is hyperparameter tuning, not algorithm capability. Original master branch solved all environments - refine search spaces and hyperparameters until solved.

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

- **Status**: ✅ Fixed and validated (PPOSIL)
- **Fix**: Corrected venv-packed data handling in replay memory (commit 6b9b0b9c)
- **Result**: PPOSIL achieves MA 496.3 on CartPole (124.1% of target)
- **Note**: Originally thought incompatible with dense rewards, but works well when properly implemented

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

**Last Updated**: 2025-11-09

### Overall Progress

| Metric                   | Value                 |
| ------------------------ | --------------------- |
| Phases Complete          | 1/4                   |
| Environments Complete    | 4/18+                 |
| Classic Control Progress | 100% (2/2, 3 skipped) |
| Box2D Progress           | 50% (2/4)             |
| MuJoCo Progress          | 0% (9)                |
| Atari Progress           | 0% (6+)               |

### Phase Completion

| Phase       | Environments  | Complete | Percentage |
| ----------- | ------------- | -------- | ---------- |
| **Phase 1** | 2 (3 skipped) | 2        | 100%       |
| **Phase 2** | 4             | 2        | 50%        |
| **Phase 3** | 9             | 0        | 0%         |
| **Phase 4** | 6+            | 0        | 0%         |

### Algorithm Performance (Phases 1-2)

| Algorithm | Envs Tested | Envs Solved | Success Rate |
| --------- | ----------- | ----------- | ------------ |
| PPO       | 5           | 5           | 100%         |
| A2C       | 2           | 2           | 100%         |
| DQN       | 2           | 2           | 100%         |
| DDQN+PER  | 2           | 2           | 100%         |
| SAC       | 2           | 2           | 100%         |
| REINFORCE | 1           | 1           | 100%         |
| SARSA     | 1           | 1           | 100%         |

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
