# SLM-Lab Benchmarks

Systematic algorithm validation across Gymnasium environments.

**Status**: Phase 3 MuJoCo in progress | **Started**: 2025-10-10 | **Updated**: 2025-12-09

---

## Active Runs

Track dstack runs for continuity. Use `dstack ps` to check status.

**Budget**: 12 parallel runs

### Current Runs

| Run Name | Command | Status | Notes |
|----------|---------|--------|-------|
| ppo-pendulum | `ppo_pendulum search` | 🔄 | Phase 1.3 |
| sac-pendulum | `sac_pendulum search` | 🔄 | Phase 1.3 |
| sac-lunar-discrete | `sac_lunar search` | 🔄 | Phase 2.1 |
| sac-lunar-cont | `sac_lunar_continuous search` | 🔄 | Phase 2.2 |
| ppo-hopper | `ppo_hopper search` | 🔄 | Phase 3 |
| ppo-halfcheetah | `ppo_halfcheetah search` | 🔄 | Phase 3 |
| ppo-walker2d | `ppo_walker2d search` | 🔄 | Phase 3 |
| ppo-ant | `ppo_ant search` | 🔄 | Phase 3 |
| ppo-swimmer | `ppo_swimmer search` | 🔄 | Phase 3 |

### Completed Runs

| Run Name | Result | Notes |
|----------|--------|-------|
| ppo-lunar-cont | ✅ MA 245.7 | Phase 2.2, target 200 |

### Quick Reference

```bash
# Monitor
dstack ps                    # List runs
dstack logs <run-name>       # View logs
dstack stop <run-name> -y    # Stop run

# Pull results
source .env && uv run slm-lab pull SPEC_NAME

# Dispatch template
source .env && uv run slm-lab run-remote --gpu SPEC_FILE SPEC_NAME search -n NAME
```

### Key Findings

- **Normalization helps MuJoCo**: A/B test shows obs+reward normalization is 2-4x better for locomotion
- **Action scaling fixed**: Automatic RescaleAction wrapper for envs with bounds != [-1, 1] (commit 8741370f)
- **Bug fixed**: TrackReward now wraps base env BEFORE NormalizeReward - reports raw rewards

---

## Progress

**Completion = all algorithms reaching target (100% of target reward).**

| Phase | Category | Envs | PPO | DQN | A2C | SAC | Overall |
|-------|----------|------|-----|-----|-----|-----|---------|
| 1.1-1.2 | Classic Control (Discrete) | 2 | ✅ | ✅ | ✅ | ✅ | ✅ 100% |
| 1.3 | Classic Control (Continuous) | 1 | 🔄 | N/A | N/A | 🔄 | 🔄 0% |
| 2 | Box2D | 2 | ✅ | ✅ | ✅ | ✅ | ✅ 100% |
| 3 | MuJoCo | 11 | 🔄 | N/A | ⏸️ | ⏸️ | 🔄 25% |
| 4 | Atari | 6+ | ⏸️ | ⏸️ | N/A | N/A | ⏸️ 0% |

**Legend**: ✅ All envs solved | 🔄 In progress | ❌ Failing | ⏸️ Not started | N/A Not applicable

## Benchmark Algorithms

**Discrete** (Classic Control, Box2D, Atari):
- DQN, DDQN+PER
- A2C (GAE), A2C (n-step)
- PPO

**Continuous** (MuJoCo):
- PPO
- SAC (future)

*Other algorithms (REINFORCE, SARSA, SIL, etc.) are included for completeness in Phase 1 but not benchmarked beyond Classic Control.*

---

## Running Experiments

### Local

```bash
uv run slm-lab SPEC_FILE SPEC_NAME train
uv run slm-lab SPEC_FILE SPEC_NAME search
```

### Remote (dstack)

```bash
# Always source .env for HF upload credentials
source .env && uv run slm-lab run-remote --gpu SPEC_FILE SPEC_NAME train -n NAME
source .env && uv run slm-lab run-remote --gpu SPEC_FILE SPEC_NAME search -n NAME

# With variable substitution for templated specs
source .env && uv run slm-lab run-remote --gpu -s env=Hopper-v5 SPEC_FILE SPEC_NAME search -n NAME
```

### Result Tables

Tables below use columns that map to command arguments:
- **Spec File**: `SPEC_FILE` argument
- **Spec Name**: `SPEC_NAME` argument
- Use `train` mode for validation runs, `search` mode for hyperparameter tuning

---

## Hyperparameter Search

**When to use**: Algorithm fails to reach target on first run.

### Search Budget Guidelines

**Search space vs trial count**: Balance is critical for efficient exploration.
- **3 parameters** (e.g., gamma, lam, lr): Use 16 trials minimum
- **4-5 parameters**: Use 20-30 trials
- **>5 parameters**: Reduce search space first - too many dims won't explore well

**Keep search space small and tractable** - don't search obvious params that won't matter. Focus only on salient hyperparameters.

### Stage 1: ASHA Search

ASHA (Asynchronous Successive Halving) with early termination for wide exploration.

**Config**: `max_session=1`, `max_trial=16` (minimum for 3-param search), `search_scheduler` enabled

```json
{
  "meta": {
    "max_session": 1,
    "max_trial": 16,
    "search_resources": {"cpu": 1, "gpu": 0.125},
    "search_scheduler": {"grace_period": 1e5, "reduction_factor": 3}
  },
  "search": {
    "agent.algorithm.gamma__uniform": [0.98, 0.999],
    "agent.algorithm.lam__uniform": [0.9, 0.98],
    "agent.net.optim_spec.lr__loguniform": [1e-4, 1e-3]
  }
}
```

**IMPORTANT**: Always use scientific notation for large numbers (e.g., `1e5`, `3e6`, `50e6`).

### Stage 2: Multi-Session Refinement (Optional)

Narrow search with robust statistics. Skip if Stage 1 results are good.

**Config**: `max_session=4`, `max_trial=8`, **NO** `search_scheduler`

### Stage 3: Finalize

1. Update spec defaults with winning hyperparams (keep `search` block for future tuning)
2. For templated specs (`${env}`): create dedicated spec file
3. Commit spec changes
4. Run validation with `train` mode
5. Pull & verify: `uv run slm-lab pull SPEC_NAME`
6. Update Active Runs and env table in this doc, push to public HF if good

**Note**: ASHA requires `max_session=1`. Multi-session requires no scheduler. They are mutually exclusive.

---

## Spec Standardization

**Keep similar specs standardized** for consistency and reproducibility.

### MuJoCo PPO Specs

All 11 MuJoCo PPO specs use uniform config:
- **Network**: `[256, 256]` + tanh + orthogonal init
- **Optimizer**: Adam 3e-4 (single optimizer with `use_same_optim=true`)
- **Search**: 3 params (gamma, lam, lr) × 16 trials
- **Normalization**: `normalize_obs=true`, `normalize_reward=true`, `normalize_v_targets=true`

### Atari Specs

All Atari specs use uniform config:
- **Network**: ConvNet with shared actor-critic
- **Frame stacking**: 4 frames
- **Preprocessing**: Standard Atari wrappers

### Environment-Specific Requirements

**Always follow these requirements when creating/updating specs:**

| Env Category | num_envs | max_frame | log_frequency | grace_period |
|--------------|----------|-----------|---------------|--------------|
| Classic Control | 4 | 2e5-3e5 | 500 | 1e4 |
| Box2D (easy) | 8 | 3e5 | 1000 | 5e4 |
| Box2D (hard) | 16 | 3e6 | 1e4 | 2e5 |
| MuJoCo (easy) | 16 | 1e6 | 1e4 | 1e5 |
| MuJoCo (medium) | 16 | 3e6 | 1e4 | 2e5 |
| MuJoCo (hard) | 32 | 50e6 | 5e4 | 1e6 |
| Atari | 16 | 10e6 | 1e4 | 5e5 |

---

## Workflow Checklist

**When starting new runs:**
1. ☐ Update "Active Runs" section at top of this doc
2. ☐ Update env table with run status (🔄)
3. ☐ Verify spec file follows standardization guidelines
4. ☐ Commit spec changes if any

**When runs complete:**
1. ☐ Pull results: `source .env && uv run slm-lab pull SPEC_NAME`
2. ☐ Analyze `experiment_df.csv` for best hyperparams
3. ☐ Update spec file with best hyperparams as defaults
4. ☐ Run validation with `train` mode to confirm
5. ☐ Update env table with results (MA, FPS, status)
6. ☐ Move run to "Completed Runs" in Active Runs section
7. ☐ Commit all changes together

---

## Phase 1: Classic Control

### 1.1 CartPole-v1

[Environment docs](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

| Property | Value |
|----------|-------|
| Action | Discrete(2) - push left/right |
| State | Box(4) - position, velocity, angle, angular velocity |
| Target | **MA > 400** |
| num_envs | 4 |
| max_frame | 2e5 |
| log_frequency | 500 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ✅ | 499.7 | 315 | [slm_lab/spec/benchmark/ppo/ppo_cartpole.json](../slm_lab/spec/benchmark/ppo/ppo_cartpole.json) | `ppo_cartpole` |
| A2C | ✅ | 488.7 | 3.5k | [slm_lab/spec/benchmark/a2c/a2c_gae_cartpole.json](../slm_lab/spec/benchmark/a2c/a2c_gae_cartpole.json) | `a2c_gae_cartpole` |
| DQN | ✅ | 437.8 | 1k | [slm_lab/spec/benchmark/dqn/dqn_cartpole.json](../slm_lab/spec/benchmark/dqn/dqn_cartpole.json) | `dqn_boltzmann_cartpole` |
| DDQN+PER | ✅ | 430.4 | 8k | [slm_lab/spec/benchmark/dqn/dqn_cartpole.json](../slm_lab/spec/benchmark/dqn/dqn_cartpole.json) | `ddqn_per_boltzmann_cartpole` |
| SAC | ✅ | 431.1 | <100 | [slm_lab/spec/benchmark/sac/sac_cartpole.json](../slm_lab/spec/benchmark/sac/sac_cartpole.json) | `sac_cartpole` |
| PPOSIL | ✅ | 496.3 | 1.6k | [slm_lab/spec/benchmark/sil/ppo_sil_cartpole.json](../slm_lab/spec/benchmark/sil/ppo_sil_cartpole.json) | `ppo_sil_cartpole` |
| REINFORCE | ✅ | 427.2 | 14k | [slm_lab/spec/benchmark/reinforce/reinforce_cartpole.json](../slm_lab/spec/benchmark/reinforce/reinforce_cartpole.json) | `reinforce_cartpole` |
| SARSA | ✅ | 393.2 | 7k | [slm_lab/spec/benchmark/sarsa/sarsa_cartpole.json](../slm_lab/spec/benchmark/sarsa/sarsa_cartpole.json) | `sarsa_epsilon_greedy_cartpole` |

### 1.2 Acrobot-v1

[Environment docs](https://gymnasium.farama.org/environments/classic_control/acrobot/)

| Property | Value |
|----------|-------|
| Action | Discrete(3) - torque (-1, 0, +1) |
| State | Box(6) - link positions and angular velocities |
| Target | **MA > -100** |
| num_envs | 4 |
| max_frame | 3e5 |
| log_frequency | 500 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ✅ | -80.8 | 777 | [slm_lab/spec/benchmark/ppo/ppo_acrobot.json](../slm_lab/spec/benchmark/ppo/ppo_acrobot.json) | `ppo_acrobot` |
| DQN (Boltzmann) | ✅ | -96.2 | 600 | [slm_lab/spec/benchmark/dqn/dqn_acrobot.json](../slm_lab/spec/benchmark/dqn/dqn_acrobot.json) | `dqn_boltzmann_acrobot` |
| DDQN+PER | ✅ | -83.0 | 700 | [slm_lab/spec/benchmark/dqn/ddqn_per_acrobot.json](../slm_lab/spec/benchmark/dqn/ddqn_per_acrobot.json) | `ddqn_per_acrobot` |
| A2C | ✅ | -84.2 | 3.4k | [slm_lab/spec/benchmark/a2c/a2c_gae_acrobot.json](../slm_lab/spec/benchmark/a2c/a2c_gae_acrobot.json) | `a2c_gae_acrobot` |
| SAC | ✅ | -92 | 60 | [slm_lab/spec/benchmark/sac/sac_acrobot.json](../slm_lab/spec/benchmark/sac/sac_acrobot.json) | `sac_acrobot` |
| DQN (ε-greedy) | ✅ | -79.5 | 720 | [slm_lab/spec/benchmark/dqn/dqn_acrobot.json](../slm_lab/spec/benchmark/dqn/dqn_acrobot.json) | `dqn_epsilon_greedy_acrobot` |
| PPOSIL | ✅ | -83.1 | - | [slm_lab/spec/benchmark/sil/ppo_sil_acrobot.json](../slm_lab/spec/benchmark/sil/ppo_sil_acrobot.json) | `ppo_sil_acrobot` |

### 1.3 Pendulum-v1 (Continuous)

[Environment docs](https://gymnasium.farama.org/environments/classic_control/pendulum/)

| Property | Value |
|----------|-------|
| Action | Box(1) - torque [-2, 2] |
| State | Box(3) - cos(θ), sin(θ), angular velocity |
| Target | **MA > -200** |
| num_envs | 4 |
| max_frame | 3e5 |
| log_frequency | 500 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | 🔄 | - | - | [slm_lab/spec/benchmark/ppo/ppo_pendulum.json](../slm_lab/spec/benchmark/ppo/ppo_pendulum.json) | `ppo_pendulum` |
| SAC | 🔄 | - | - | [slm_lab/spec/benchmark/sac/sac_pendulum.json](../slm_lab/spec/benchmark/sac/sac_pendulum.json) | `sac_pendulum` |

**Note**: Classic Control continuous benchmark (not MuJoCo). Action bounds [-2, 2] use automatic RescaleAction wrapper. Target **MA > -200** (best possible ~0).

---

## Phase 2: Box2D

### 2.1 LunarLander-v3 (Discrete)

[Environment docs](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

| Property | Value |
|----------|-------|
| Action | Discrete(4) - no-op, fire left/main/right engine |
| State | Box(8) - position, velocity, angle, angular velocity, leg contact |
| Target | **MA > 200** |
| num_envs | 8 |
| max_frame | 3e5 |
| log_frequency | 1000 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| DDQN+PER | ✅ | 230.0 | 8.7k | [slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json](../slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json) | `ddqn_per_concat_lunar` |
| PPO | ✅ | 229.9 | 2.4k | [slm_lab/spec/benchmark/ppo/ppo_lunar.json](../slm_lab/spec/benchmark/ppo/ppo_lunar.json) | `ppo_lunar` |
| DQN | ✅ | 203.9 | 9.0k | [slm_lab/spec/benchmark/dqn/dqn_lunar.json](../slm_lab/spec/benchmark/dqn/dqn_lunar.json) | `dqn_concat_lunar` |
| A2C | ✅ | 304 | 3k | [slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json](../slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json) | `a2c_gae_lunar` |
| SAC | 🔄 | - | - | [slm_lab/spec/benchmark/sac/sac_lunar.json](../slm_lab/spec/benchmark/sac/sac_lunar.json) | `sac_lunar` |

### 2.2 LunarLander-v3 (Continuous)

[Environment docs](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

| Property | Value |
|----------|-------|
| Action | Box(2) - main engine [-1,1], side engines [-1,1] |
| State | Box(8) - position, velocity, angle, angular velocity, leg contact |
| Target | **MA > 200** |
| num_envs | 8 |
| max_frame | 3e5 |
| log_frequency | 1000 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ✅ | 245.7 | 135 | [slm_lab/spec/benchmark/ppo/ppo_lunar.json](../slm_lab/spec/benchmark/ppo/ppo_lunar.json) | `ppo_lunar_continuous` |
| SAC | ✅ | 238.0 | 35 | [slm_lab/spec/benchmark/sac/sac_lunar.json](../slm_lab/spec/benchmark/sac/sac_lunar.json) | `sac_lunar_continuous` |

---

## Phase 3: MuJoCo

All MuJoCo environments use **continuous** action spaces.

**Template specs**: Use `-s env=EnvName-v5` for variable substitution. Successful envs get dedicated specs.

### 3.1 Hopper-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/hopper/)

| Property | Value |
|----------|-------|
| Action | Box(3) |
| State | Box(11) |
| Target | **MA > 3000** |
| num_envs | 16 |
| max_frame | 3e6 |
| log_frequency | 1e4 |
| grace_period | 2e5 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⚠️ 78% | 2517 | 1.5k | [slm_lab/spec/benchmark/ppo/ppo_hopper.json](../slm_lab/spec/benchmark/ppo/ppo_hopper.json) | `ppo_hopper` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=Hopper-v5` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=Hopper-v5` |

### 3.2 HalfCheetah-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/half_cheetah/)

| Property | Value |
|----------|-------|
| Action | Box(6) |
| State | Box(17) |
| Target | **MA > 5000** |
| num_envs | 16 |
| max_frame | 3e6 |
| log_frequency | 1e4 |
| grace_period | 2e5 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | 🔄 51% | 2615 | 1.5k | [slm_lab/spec/benchmark/ppo/ppo_halfcheetah.json](../slm_lab/spec/benchmark/ppo/ppo_halfcheetah.json) | `ppo_halfcheetah` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=HalfCheetah-v5` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=HalfCheetah-v5` |

### 3.3 Walker2d-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/walker2d/)

| Property | Value |
|----------|-------|
| Action | Box(6) |
| State | Box(17) |
| Target | **MA > 5000** |
| num_envs | 16 |
| max_frame | 3e6 |
| log_frequency | 1e4 |
| grace_period | 2e5 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⚠️ 45% | 1997 | 1.5k | [slm_lab/spec/benchmark/ppo/ppo_walker2d.json](../slm_lab/spec/benchmark/ppo/ppo_walker2d.json) | `ppo_walker2d` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=Walker2d-v5` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=Walker2d-v5` |

### 3.4 Ant-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/ant/)

| Property | Value |
|----------|-------|
| Action | Box(8) |
| State | Box(111) |
| Target | **MA > 5000** |
| num_envs | 16 |
| max_frame | 3e6 |
| log_frequency | 1e4 |
| grace_period | 2e5 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ❌ | -105 | 1.5k | [slm_lab/spec/benchmark/ppo/ppo_ant.json](../slm_lab/spec/benchmark/ppo/ppo_ant.json) | `ppo_ant` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=Ant-v5` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=Ant-v5` |

### 3.5 Swimmer-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/swimmer/)

| Property | Value |
|----------|-------|
| Action | Box(2) |
| State | Box(8) |
| Target | **MA > 130** |
| num_envs | 16 |
| max_frame | 1e6 |
| log_frequency | 1e4 |
| grace_period | 1e5 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | 🔄 | 43.3 | 1.5k | [slm_lab/spec/benchmark/ppo/ppo_swimmer.json](../slm_lab/spec/benchmark/ppo/ppo_swimmer.json) | `ppo_swimmer` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=Swimmer-v5` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=Swimmer-v5` |

### 3.6 Reacher-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/reacher/)

| Property | Value |
|----------|-------|
| Action | Box(2) |
| State | Box(11) |
| Target | **MA > -5** |
| num_envs | 16 |
| max_frame | 1e6 |
| log_frequency | 1e4 |
| grace_period | 1e5 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ✅ | -0.003 | 1.5k | [slm_lab/spec/benchmark/ppo/ppo_reacher.json](../slm_lab/spec/benchmark/ppo/ppo_reacher.json) | `ppo_reacher` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=Reacher-v5` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=Reacher-v5` |

### 3.7 Pusher-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/pusher/)

| Property | Value |
|----------|-------|
| Action | Box(7) |
| State | Box(23) |
| Target | **MA > -20** |
| num_envs | 16 |
| max_frame | 1e6 |
| log_frequency | 1e4 |
| grace_period | 1e5 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ✅ | -0.75 | - | [slm_lab/spec/benchmark/ppo/ppo_pusher.json](../slm_lab/spec/benchmark/ppo/ppo_pusher.json) | `ppo_pusher` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=Pusher-v5` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=Pusher-v5` |

### 3.8 InvertedPendulum-v5 (PARKED)

[Environment docs](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/)

| Property | Value |
|----------|-------|
| Action | Box(1) |
| State | Box(4) |
| Target | **MA > 1000** |
| num_envs | 16 |
| max_frame | 1e6 |
| log_frequency | 1e4 |
| grace_period | 1e5 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⏸️ | 19.2 | 3k | [slm_lab/spec/benchmark/ppo/ppo_inverted_pendulum.json](../slm_lab/spec/benchmark/ppo/ppo_inverted_pendulum.json) | `ppo_inverted_pendulum` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=InvertedPendulum-v5` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=InvertedPendulum-v5` |

**PARKED**: Consistently failing despite many attempted fixes (log_std, entropy, architecture). Needs deeper investigation.

### 3.9 InvertedDoublePendulum-v5 (PARKED)

[Environment docs](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/)

| Property | Value |
|----------|-------|
| Action | Box(1) |
| State | Box(11) |
| Target | **MA > 9100** |
| num_envs | 16 |
| max_frame | 1e6 |
| log_frequency | 1e4 |
| grace_period | 1e5 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⏸️ | 73.6 | - | [slm_lab/spec/benchmark/ppo/ppo_inverted_double_pendulum.json](../slm_lab/spec/benchmark/ppo/ppo_inverted_double_pendulum.json) | `ppo_inverted_double_pendulum` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=InvertedDoublePendulum-v5` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=InvertedDoublePendulum-v5` |

**PARKED**: Consistently failing despite many attempted fixes (log_std, entropy, architecture). Needs deeper investigation.

### 3.10 Humanoid-v5 (PARKED)

[Environment docs](https://gymnasium.farama.org/environments/mujoco/humanoid/)

| Property | Value |
|----------|-------|
| Action | Box(17) |
| State | Box(376) |
| Target | **MA > 6000** |
| num_envs | 32 |
| max_frame | 50e6 |
| log_frequency | 5e4 |
| grace_period | 1e6 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⏸️ | - | - | [slm_lab/spec/benchmark/ppo/ppo_humanoid.json](../slm_lab/spec/benchmark/ppo/ppo_humanoid.json) | `ppo_humanoid` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_humanoid.json](../slm_lab/spec/benchmark/sac/sac_humanoid.json) | `sac_humanoid` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=Humanoid-v5` |

**PARKED**: Requires 50M frames - will hit 4h dstack limit (~20-25M frames typically). Deprioritized until medium envs solved.

### 3.11 HumanoidStandup-v5 (PARKED)

[Environment docs](https://gymnasium.farama.org/environments/mujoco/humanoid_standup/)

| Property | Value |
|----------|-------|
| Action | Box(17) |
| State | Box(376) |
| Target | **MA > 100000** |
| num_envs | 32 |
| max_frame | 50e6 |
| log_frequency | 5e4 |
| grace_period | 1e6 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⏸️ | - | - | [slm_lab/spec/benchmark/ppo/ppo_humanoid_standup.json](../slm_lab/spec/benchmark/ppo/ppo_humanoid_standup.json) | `ppo_humanoid_standup` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=HumanoidStandup-v5` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=HumanoidStandup-v5` |

**PARKED**: Very high reward scale environment. Requires 50M frames. Deprioritized until medium envs solved.

---

## Phase 4: Atari

All Atari environments use **Discrete** action spaces and **Box(210,160,3)** RGB image observations.

### 4.1 Pong-v5

[Environment docs](https://gymnasium.farama.org/environments/atari/pong/)

| Property | Value |
|----------|-------|
| Action | Discrete(6) |
| State | Box(210,160,3) RGB |
| Target | **MA > 18** |
| num_envs | 16 |
| max_frame | 10e6 |
| log_frequency | 1e4 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⏸️ | - | - | [slm_lab/spec/benchmark/ppo/ppo_pong.json](../slm_lab/spec/benchmark/ppo/ppo_pong.json) | `ppo_pong` |
| DQN | ⏸️ | - | - | [slm_lab/spec/benchmark/dqn/dqn_pong.json](../slm_lab/spec/benchmark/dqn/dqn_pong.json) | `dqn_pong` |

### 4.2 Qbert-v5

[Environment docs](https://gymnasium.farama.org/environments/atari/qbert/)

| Property | Value |
|----------|-------|
| Action | Discrete(6) |
| State | Box(210,160,3) RGB |
| Target | **MA > 15000** |
| num_envs | 16 |
| max_frame | 10e6 |
| log_frequency | 1e4 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⏸️ | - | - | [slm_lab/spec/benchmark/ppo/ppo_qbert.json](../slm_lab/spec/benchmark/ppo/ppo_qbert.json) | `ppo_qbert` |
| DQN | ⏸️ | - | - | [slm_lab/spec/benchmark/dqn/dqn_qbert.json](../slm_lab/spec/benchmark/dqn/dqn_qbert.json) | `dqn_qbert` |

### 4.3 Breakout-v5

[Environment docs](https://gymnasium.farama.org/environments/atari/breakout/)

| Property | Value |
|----------|-------|
| Action | Discrete(4) |
| State | Box(210,160,3) RGB |
| Target | **MA > 400** |
| num_envs | 16 |
| max_frame | 10e6 |
| log_frequency | 1e4 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⏸️ | - | - | [slm_lab/spec/benchmark/ppo/ppo_atari.json](../slm_lab/spec/benchmark/ppo/ppo_atari.json) | `ppo_atari` `-s env=ALE/Breakout-v5` |
| DQN | ⏸️ | - | - | [slm_lab/spec/benchmark/dqn/dqn_atari.json](../slm_lab/spec/benchmark/dqn/dqn_atari.json) | `dqn_atari` `-s env=ALE/Breakout-v5` |

---

## Known Issues

**DQN Compute Inefficiency** ✅ RESOLVED
- Was 84x slower than A2C due to excessive gradient updates (10 updates/step vs standard 1)
- Fixed by adjusting `training_batch_iter` and `training_iter`
- Result: 3.7-15x speedup with equivalent learning

**SIL (Self-Imitation Learning)** ✅ RESOLVED
- Fixed venv-packed data handling in replay memory
- PPOSIL now achieves 124% of target on CartPole

---

