# SLM-Lab Benchmarks

Systematic algorithm validation across Gymnasium environments.

**Status**: Phase 3 MuJoCo in progress | **Started**: 2025-10-10 | **Updated**: 2025-11-29

## Progress

| Phase | Category | Envs | Status | Notes |
|-------|----------|------|--------|-------|
| 1 | Classic Control | 2 | ✅ 100% | CartPole, Acrobot done |
| 2 | Box2D | 4 | 🔄 75% | BipedalWalker needs tuning |
| 3 | MuJoCo | 9 | 🔄 33% | Hopper, HalfCheetah, Reacher done |
| 4 | Atari | 6+ | ⏸️ 0% | Pending |

## Benchmark Algorithms

**Discrete** (Classic Control, Box2D discrete, Atari):
- DQN, DDQN+PER
- A2C (GAE), A2C (n-step)
- PPO
- SAC

**Continuous** (Box2D continuous, MuJoCo):
- A2C (GAE), A2C (n-step)
- PPO
- SAC

*Other algorithms (REINFORCE, SARSA, etc.) are included for completeness in Phase 1 but not benchmarked beyond Classic Control.*

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

### Stage 1: ASHA Search

ASHA (Asynchronous Successive Halving) with early termination for wide exploration.

**Config**: `max_session=1`, `max_trial=8-16`, `search_scheduler` enabled

```json
{
  "meta": {
    "max_session": 1,
    "max_trial": 8,
    "search_resources": {"cpu": 1, "gpu": 0.125},
    "search_scheduler": {"grace_period": 100000, "reduction_factor": 3}
  },
  "search": {
    "agent.algorithm.gamma__uniform": [0.993, 0.999],
    "agent.algorithm.lam__uniform": [0.90, 0.95],
    "agent.net.optim_spec.lr__loguniform": [1e-4, 1e-3]
  }
}
```

### Stage 2: Multi-Session Refinement (Optional)

Narrow search with robust statistics. Skip if Stage 1 results are good.

**Config**: `max_session=4`, `max_trial=8`, **NO** `search_scheduler`

### Stage 3: Finalize

1. Update spec defaults with winning hyperparams (keep `search` block for existing specs)
2. For templated specs (`${env}`): create dedicated spec file without search block
3. Commit spec
4. Validation run with `train` mode
5. Pull & verify: `uv run slm-lab pull SPEC_NAME`
6. Update this doc, push to public HF if good

**Note**: ASHA requires `max_session=1`. Multi-session requires no scheduler. They are mutually exclusive.

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
| max_frames | 200k |
| log_frequency | 500 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ✅ | 499.7 | 315 | [slm_lab/spec/benchmark/ppo/ppo_cartpole.json](../slm_lab/spec/benchmark/ppo/ppo_cartpole.json) | `ppo_cartpole` |
| A2C | ✅ | 488.7 | 3.5k | [slm_lab/spec/benchmark/a2c/a2c_gae_cartpole.json](../slm_lab/spec/benchmark/a2c/a2c_gae_cartpole.json) | `a2c_gae_cartpole` |
| PPOSIL | ✅ | 496.3 | 1.6k | [slm_lab/spec/benchmark/sil/ppo_sil_cartpole.json](../slm_lab/spec/benchmark/sil/ppo_sil_cartpole.json) | `ppo_sil_cartpole` |
| DQN | ✅ | 437.8 | 1k | [slm_lab/spec/benchmark/dqn/dqn_cartpole.json](../slm_lab/spec/benchmark/dqn/dqn_cartpole.json) | `dqn_boltzmann_cartpole` |
| SAC | ✅ | 431.1 | <100 | [slm_lab/spec/benchmark/sac/sac_cartpole.json](../slm_lab/spec/benchmark/sac/sac_cartpole.json) | `sac_cartpole` |
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
| max_frames | 300k |
| log_frequency | 500 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ✅ | -80.8 | 777 | [slm_lab/spec/benchmark/ppo/ppo_acrobot.json](../slm_lab/spec/benchmark/ppo/ppo_acrobot.json) | `ppo_acrobot` |
| DQN (Boltzmann) | ✅ | -96.2 | 600 | [slm_lab/spec/benchmark/dqn/dqn_acrobot.json](../slm_lab/spec/benchmark/dqn/dqn_acrobot.json) | `dqn_boltzmann_acrobot` |
| DDQN+PER | ✅ | -83.0 | 700 | [slm_lab/spec/benchmark/dqn/ddqn_per_acrobot.json](../slm_lab/spec/benchmark/dqn/ddqn_per_acrobot.json) | `ddqn_per_acrobot` |
| A2C | ✅ | -84.2 | 3.4k | [slm_lab/spec/benchmark/a2c/a2c_gae_acrobot.json](../slm_lab/spec/benchmark/a2c/a2c_gae_acrobot.json) | `a2c_gae_acrobot` |
| SAC | ✅ | -92 | 60 | [slm_lab/spec/benchmark/sac/sac_acrobot.json](../slm_lab/spec/benchmark/sac/sac_acrobot.json) | `sac_acrobot` |
| DQN (ε-greedy) | ⚠️ | -104.0 | 720 | [slm_lab/spec/benchmark/dqn/dqn_acrobot.json](../slm_lab/spec/benchmark/dqn/dqn_acrobot.json) | `dqn_epsilon_greedy_acrobot` |
| PPOSIL | ⚠️ | -110.2 | - | [slm_lab/spec/benchmark/sil/ppo_sil_acrobot.json](../slm_lab/spec/benchmark/sil/ppo_sil_acrobot.json) | `ppo_sil_acrobot` |

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
| max_frames | 300k |
| log_frequency | 1000 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| DDQN+PER | ✅ | 230.0 | 8.7k | [slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json](../slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json) | `ddqn_per_concat_lunar` |
| PPO | ✅ | 229.9 | 2.4k | [slm_lab/spec/benchmark/ppo/ppo_lunar.json](../slm_lab/spec/benchmark/ppo/ppo_lunar.json) | `ppo_lunar` |
| DQN | ✅ | 203.9 | 9.0k | [slm_lab/spec/benchmark/dqn/dqn_lunar.json](../slm_lab/spec/benchmark/dqn/dqn_lunar.json) | `dqn_concat_lunar` |
| A2C | ❌ | 5.5 | 3k | [slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json](../slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json) | `a2c_gae_lunar` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_lunar.json](../slm_lab/spec/benchmark/sac/sac_lunar.json) | `sac_lunar` |

### 2.2 LunarLander-v3 (Continuous)

[Environment docs](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

| Property | Value |
|----------|-------|
| Action | Box(2) - main engine [-1,1], side engines [-1,1] |
| State | Box(8) - position, velocity, angle, angular velocity, leg contact |
| Target | **MA > 200** |
| num_envs | 8 |
| max_frames | 300k |
| log_frequency | 1000 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ✅ | 249.2 | 135 | [slm_lab/spec/benchmark/ppo/ppo_lunar.json](../slm_lab/spec/benchmark/ppo/ppo_lunar.json) | `ppo_lunar_continuous` |
| SAC | ✅ | 238.0 | 35 | [slm_lab/spec/benchmark/sac/sac_lunar.json](../slm_lab/spec/benchmark/sac/sac_lunar.json) | `sac_lunar_continuous` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json](../slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json) | `a2c_gae_lunar_continuous` |

### 2.3 BipedalWalker-v3

[Environment docs](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)

| Property | Value |
|----------|-------|
| Action | Box(4) - motor speeds for 4 joints [-1,1] |
| State | Box(24) - hull state, joint positions, velocities, lidar |
| Target | **MA > 300** |
| num_envs | 16 |
| max_frames | 3M |
| log_frequency | 10000 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⚠️ 80% | 241.3 | 1.8k | [slm_lab/spec/benchmark/ppo/ppo_bipedalwalker.json](../slm_lab/spec/benchmark/ppo/ppo_bipedalwalker.json) | `ppo_bipedalwalker` |
| SAC | 🔄 | - | - | [slm_lab/spec/benchmark/sac/sac_bipedalwalker.json](../slm_lab/spec/benchmark/sac/sac_bipedalwalker.json) | `sac_bipedalwalker` |
| A2C | ❌ | -112 | 8k | [slm_lab/spec/benchmark/a2c/a2c_gae_bipedalwalker.json](../slm_lab/spec/benchmark/a2c/a2c_gae_bipedalwalker.json) | `a2c_gae_bipedalwalker` |

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
| max_frames | 3M |
| log_frequency | 10000 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ✅ 94% | 2816 | 1.5k | [slm_lab/spec/benchmark/ppo/ppo_hopper.json](../slm_lab/spec/benchmark/ppo/ppo_hopper.json) | `ppo_hopper` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=Hopper-v5` |

### 3.2 HalfCheetah-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/half_cheetah/)

| Property | Value |
|----------|-------|
| Action | Box(6) |
| State | Box(17) |
| Target | **MA > 5000** |
| num_envs | 16 |
| max_frames | 3M |
| log_frequency | 10000 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ✅ 81% | 4042 | 1.5k | [slm_lab/spec/benchmark/ppo/ppo_halfcheetah.json](../slm_lab/spec/benchmark/ppo/ppo_halfcheetah.json) | `ppo_halfcheetah` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=HalfCheetah-v5` |

### 3.3 Walker2d-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/walker2d/)

| Property | Value |
|----------|-------|
| Action | Box(6) |
| State | Box(17) |
| Target | **MA > 4000** |
| num_envs | 16 |
| max_frames | 3M |
| log_frequency | 10000 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⚠️ 64% | 2573 | 1.5k | [slm_lab/spec/benchmark/ppo/ppo_mujoco.json](../slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | `ppo_mujoco` `-s env=Walker2d-v5` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=Walker2d-v5` |

### 3.4 Ant-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/ant/)

| Property | Value |
|----------|-------|
| Action | Box(8) |
| State | Box(111) |
| Target | **MA > 5000** |
| num_envs | 16 |
| max_frames | 3M |
| log_frequency | 10000 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ❌ 0.7% | 36 | 1.5k | [slm_lab/spec/benchmark/ppo/ppo_mujoco.json](../slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | `ppo_mujoco` `-s env=Ant-v5` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=Ant-v5` |

### 3.5 Swimmer-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/swimmer/)

| Property | Value |
|----------|-------|
| Action | Box(2) |
| State | Box(8) |
| Target | **MA > 100** |
| num_envs | 16 |
| max_frames | 3M |
| log_frequency | 10000 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⚠️ 44% | 44 | 1.5k | [slm_lab/spec/benchmark/ppo/ppo_mujoco.json](../slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | `ppo_mujoco` `-s env=Swimmer-v5` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=Swimmer-v5` |

### 3.6 Reacher-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/reacher/)

| Property | Value |
|----------|-------|
| Action | Box(2) |
| State | Box(11) |
| Target | **MA > -5** |
| num_envs | 16 |
| max_frames | 3M |
| log_frequency | 10000 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | 🔄 near | -6.2 | 1.5k | [slm_lab/spec/benchmark/ppo/ppo_mujoco.json](../slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | `ppo_mujoco` `-s env=Reacher-v5` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=Reacher-v5` |

### 3.7 InvertedPendulum-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/)

| Property | Value |
|----------|-------|
| Action | Box(1) |
| State | Box(4) |
| Target | **MA > 1000** |
| num_envs | 16 |
| max_frames | 1M |
| log_frequency | 10000 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⏸️ | - | - | [slm_lab/spec/benchmark/ppo/ppo_mujoco.json](../slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | `ppo_mujoco` `-s env=InvertedPendulum-v5` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=InvertedPendulum-v5` |

### 3.8 InvertedDoublePendulum-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/)

| Property | Value |
|----------|-------|
| Action | Box(1) |
| State | Box(11) |
| Target | **MA > 9000** |
| num_envs | 16 |
| max_frames | 3M |
| log_frequency | 10000 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⏸️ | - | - | [slm_lab/spec/benchmark/ppo/ppo_mujoco.json](../slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | `ppo_mujoco` `-s env=InvertedDoublePendulum-v5` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=InvertedDoublePendulum-v5` |

### 3.9 Humanoid-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/humanoid/)

| Property | Value |
|----------|-------|
| Action | Box(17) |
| State | Box(376) |
| Target | **MA > 6000** |
| num_envs | 32 |
| max_frames | 50M |
| log_frequency | 10000 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⏸️ | - | - | [slm_lab/spec/benchmark/ppo/ppo_mujoco.json](../slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | `ppo_mujoco` `-s env=Humanoid-v5` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_humanoid.json](../slm_lab/spec/benchmark/sac/sac_humanoid.json) | `sac_humanoid` |

**Key finding**: ASHA found gamma~0.998, lam~0.905 works well for Hopper/HalfCheetah. Ant/Swimmer need different tuning.

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
| max_frames | 10M |
| log_frequency | 10000 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⏸️ | - | - | [slm_lab/spec/benchmark/ppo/ppo_pong.json](../slm_lab/spec/benchmark/ppo/ppo_pong.json) | `ppo_pong` |
| DQN | ⏸️ | - | - | [slm_lab/spec/benchmark/dqn/dqn_pong.json](../slm_lab/spec/benchmark/dqn/dqn_pong.json) | `dqn_pong` |
| PPOSIL | ⏸️ | - | - | [slm_lab/spec/benchmark/sil/ppo_sil_pong.json](../slm_lab/spec/benchmark/sil/ppo_sil_pong.json) | `ppo_sil_pong` |

### 4.2 Qbert-v5

[Environment docs](https://gymnasium.farama.org/environments/atari/qbert/)

| Property | Value |
|----------|-------|
| Action | Discrete(6) |
| State | Box(210,160,3) RGB |
| Target | **MA > 15000** |
| num_envs | 16 |
| max_frames | 10M |
| log_frequency | 10000 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⏸️ | - | - | [slm_lab/spec/benchmark/ppo/ppo_qbert.json](../slm_lab/spec/benchmark/ppo/ppo_qbert.json) | `ppo_qbert` |
| DQN | ⏸️ | - | - | [slm_lab/spec/benchmark/dqn/dqn_qbert.json](../slm_lab/spec/benchmark/dqn/dqn_qbert.json) | `dqn_qbert` |
| PPOSIL | ⏸️ | - | - | [slm_lab/spec/benchmark/sil/ppo_sil_qbert.json](../slm_lab/spec/benchmark/sil/ppo_sil_qbert.json) | `ppo_sil_qbert` |

### 4.3 Breakout-v5

[Environment docs](https://gymnasium.farama.org/environments/atari/breakout/)

| Property | Value |
|----------|-------|
| Action | Discrete(4) |
| State | Box(210,160,3) RGB |
| Target | **MA > 400** |
| num_envs | 16 |
| max_frames | 10M |
| log_frequency | 10000 |

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

## Next Steps

**Immediate** (failing envs):
- [ ] A2C LunarLander: investigate (2.8% of target)
- [ ] PPO Ant: dedicated tuning (0.7% of target)
- [ ] PPO Walker2d/Swimmer: env-specific search

**Queue**:
- [ ] SAC on failing PPO envs (Ant, Swimmer may benefit)
- [ ] Remaining MuJoCo (InvertedPendulum, Humanoid)
- [ ] Phase 4 Atari
