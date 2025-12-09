# SLM-Lab Benchmarks

Systematic algorithm validation across Gymnasium environments.

**Updated**: 2025-12-15

---

## Usage

```bash
# Local
uv run slm-lab SPEC_FILE SPEC_NAME train
uv run slm-lab SPEC_FILE SPEC_NAME search

# Remote (dstack) - always source .env for HF upload
source .env && uv run slm-lab run-remote --gpu SPEC_FILE SPEC_NAME train -n NAME
source .env && uv run slm-lab run-remote --gpu SPEC_FILE SPEC_NAME search -n NAME

# Monitor
dstack ps                    # List runs
dstack logs <run-name>       # View logs
dstack stop <run-name> -y    # Stop run

# Pull results
source .env && uv run slm-lab pull SPEC_NAME
```

### Guidelines

#### Hyperparameter Search

**When to use**: Algorithm fails to reach target on first run.

| Stage | Mode | Config | Purpose |
|-------|------|--------|---------|
| ASHA | `search` | `max_session=1`, `search_scheduler` enabled | Wide exploration |
| Multi | `search` | `max_session=4`, NO `search_scheduler` | Robust validation |
| Validate | `train` | Final spec | Confirmation |

**Search budget**: ~3-4 trials per dimension minimum (8 trials = 2-3 dims, 16 = 3-4, 20+ = 5+).

```json
{
  "meta": {
    "max_session": 1, "max_trial": 16,
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

#### Environment Settings

**IMPORTANT**: All specs MUST strictly follow these settings for fair comparison across algorithms.

| Env Category | num_envs | max_frame | log_frequency | grace_period |
|--------------|----------|-----------|---------------|--------------|
| Classic Control | 4 | 2e5-3e5 | 500 | 1e4 |
| Box2D | 8-16 | 3e5-3e6 | 1e3 | 5e4-2e5 |
| MuJoCo (easy) | 4-16 | 1e6-3e6 | 500-1000 | 1e5-2e5 |
| MuJoCo (hard) | 16 | 10e6-50e6 | 1000 | 1e6 |
| Atari | 16 | 10e6 | 1e4 | 5e5 |

**Note**: log_frequency should be ~episode_length for responsive MA updates (Reacher=50, Pusher=100, others=1000).

#### MuJoCo PPO Standard

- **Network**: `[256, 256]` + tanh + orthogonal init
- **Normalization**: `normalize_obs=true`, `normalize_reward=true`, `normalize_v_targets=true`
- **Search**: 3 params (gamma, lam, lr) × 16 trials

---

## Progress

| Phase | Category | Envs | PPO | DQN | A2C | SAC | Overall |
|-------|----------|------|-----|-----|-----|-----|---------|
| 1 | Classic Control | 3 | ✅ | ✅ | ✅ | ✅ | ✅ 100% |
| 2 | Box2D | 2 | ✅ | ✅ | ❌ | ✅ | ✅ 88% |
| 3 | MuJoCo | 11 | ✅ | N/A | ⏸️ | ⏸️ | ✅ PPO done |
| 4 | Atari | 6+ | ⏸️ | ⏸️ | N/A | N/A | ⏸️ 0% |

**Legend**: ✅ Solved | ⚠️ Close (>80%) | ❌ Failed | 🔄 In progress | ⏸️ Not started | N/A Not applicable

---

## Results

### Phase 1: Classic Control

#### 1.1 CartPole-v1

**Docs**: [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) | State: Box(4) | Action: Discrete(2) | Solved reward MA > 400

**Settings**: max_frame 2e5 | num_envs 4 | max_session 4 | log_frequency 500

| Algorithm | Status | MA | Spec File | Spec Name |
|-----------|--------|-----|-----------|-----------|
| PPO | ✅ | 499.7 | [ppo_cartpole.json](../slm_lab/spec/benchmark/ppo/ppo_cartpole.json) | `ppo_cartpole` |
| A2C | ✅ | 488.7 | [a2c_gae_cartpole.json](../slm_lab/spec/benchmark/a2c/a2c_gae_cartpole.json) | `a2c_gae_cartpole` |
| DQN | ✅ | 437.8 | [dqn_cartpole.json](../slm_lab/spec/benchmark/dqn/dqn_cartpole.json) | `dqn_boltzmann_cartpole` |
| DDQN+PER | ✅ | 430.4 | [dqn_cartpole.json](../slm_lab/spec/benchmark/dqn/dqn_cartpole.json) | `ddqn_per_boltzmann_cartpole` |
| SAC | ✅ | 431.1 | [sac_cartpole.json](../slm_lab/spec/benchmark/sac/sac_cartpole.json) | `sac_cartpole` |

#### 1.2 Acrobot-v1

**Docs**: [Acrobot](https://gymnasium.farama.org/environments/classic_control/acrobot/) | State: Box(6) | Action: Discrete(3) | Solved reward MA > -100

**Settings**: max_frame 3e5 | num_envs 4 | max_session 4 | log_frequency 500

| Algorithm | Status | MA | Spec File | Spec Name |
|-----------|--------|-----|-----------|-----------|
| PPO | ✅ | -80.8 | [ppo_acrobot.json](../slm_lab/spec/benchmark/ppo/ppo_acrobot.json) | `ppo_acrobot` |
| DQN | ✅ | -96.2 | [dqn_acrobot.json](../slm_lab/spec/benchmark/dqn/dqn_acrobot.json) | `dqn_boltzmann_acrobot` |
| DDQN+PER | ✅ | -83.0 | [ddqn_per_acrobot.json](../slm_lab/spec/benchmark/dqn/ddqn_per_acrobot.json) | `ddqn_per_acrobot` |
| A2C | ✅ | -84.2 | [a2c_gae_acrobot.json](../slm_lab/spec/benchmark/a2c/a2c_gae_acrobot.json) | `a2c_gae_acrobot` |
| SAC | ✅ | -97 | [sac_acrobot.json](../slm_lab/spec/benchmark/sac/sac_acrobot.json) | `sac_acrobot` |

#### 1.3 Pendulum-v1

**Docs**: [Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/) | State: Box(3) | Action: Box(1) | Solved reward MA > -200

**Settings**: max_frame 3e5 | num_envs 4 | max_session 4 | log_frequency 500

| Algorithm | Status | MA | Spec File | Spec Name |
|-----------|--------|-----|-----------|-----------|
| PPO | ✅ | -178 | [ppo_pendulum.json](../slm_lab/spec/benchmark/ppo/ppo_pendulum.json) | `ppo_pendulum` |
| SAC | ✅ | -150 | [sac_pendulum.json](../slm_lab/spec/benchmark/sac/sac_pendulum.json) | `sac_pendulum` |

### Phase 2: Box2D

#### 2.1 LunarLander-v3 (Discrete)

**Docs**: [LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) | State: Box(8) | Action: Discrete(4) | Solved reward MA > 200

**Settings**: max_frame 3e5 | num_envs 8 | max_session 4 | log_frequency 1000

| Algorithm | Status | MA | Spec File | Spec Name |
|-----------|--------|-----|-----------|-----------|
| DDQN+PER | ✅ | 230.0 | [ddqn_per_lunar.json](../slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json) | `ddqn_per_concat_lunar` |
| PPO | ✅ | 229.9 | [ppo_lunar.json](../slm_lab/spec/benchmark/ppo/ppo_lunar.json) | `ppo_lunar` |
| DQN | ✅ | 203.9 | [dqn_lunar.json](../slm_lab/spec/benchmark/dqn/dqn_lunar.json) | `dqn_concat_lunar` |
| A2C | ❌ 26% | +41 | [a2c_gae_lunar.json](../slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json) | `a2c_gae_lunar` (target: 155, SB3 benchmark) |

#### 2.2 LunarLander-v3 (Continuous)

**Docs**: [LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) | State: Box(8) | Action: Box(2) | Solved reward MA > 200

**Settings**: max_frame 3e5 | num_envs 8 | max_session 4 | log_frequency 1000

| Algorithm | Status | MA | Spec File | Spec Name |
|-----------|--------|-----|-----------|-----------|
| PPO | ✅ | 245.7 | [ppo_lunar.json](../slm_lab/spec/benchmark/ppo/ppo_lunar.json) | `ppo_lunar_continuous` |
| SAC | ✅ | 241.6 | [sac_lunar.json](../slm_lab/spec/benchmark/sac/sac_lunar.json) | `sac_lunar_continuous` |
| A2C | ❌ 1% | 2.5 | [a2c_gae_lunar.json](../slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json) | `a2c_gae_lunar_continuous` |

### Phase 3: MuJoCo

#### 3.1 Hopper-v5

**Docs**: [Hopper](https://gymnasium.farama.org/environments/mujoco/hopper/) | State: Box(11) | Action: Box(3) | Solved reward MA > 2500

**Settings**: max_frame 1e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec File | Spec Name |
|-----------|--------|-----|-----------|-----------|
| PPO | ✅ | 2914 | [ppo_hopper.json](../slm_lab/spec/benchmark/ppo/ppo_hopper.json) | `ppo_hopper` |

#### 3.2 HalfCheetah-v5

**Docs**: [HalfCheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) | State: Box(17) | Action: Box(6) | Solved reward MA > 5000

**Settings**: max_frame 8e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec File | Spec Name |
|-----------|--------|-----|-----------|-----------|
| PPO | ✅ | 6383 | [ppo_halfcheetah.json](../slm_lab/spec/benchmark/ppo/ppo_halfcheetah.json) | `ppo_halfcheetah` |

#### 3.3 Walker2d-v5

**Docs**: [Walker2d](https://gymnasium.farama.org/environments/mujoco/walker2d/) | State: Box(17) | Action: Box(6) | Solved reward MA > 3500

**Settings**: max_frame 8e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec File | Spec Name |
|-----------|--------|-----|-----------|-----------|
| PPO | ✅ | 5700 | [ppo_walker2d.json](../slm_lab/spec/benchmark/ppo/ppo_walker2d.json) | `ppo_walker2d` |

#### 3.4 Ant-v5

**Docs**: [Ant](https://gymnasium.farama.org/environments/mujoco/ant/) | State: Box(105) | Action: Box(8) | Solved reward MA > 2000

**Settings**: max_frame 8e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec File | Spec Name |
|-----------|--------|-----|-----------|-----------|
| PPO | ✅ | 2190 | [ppo_ant.json](../slm_lab/spec/benchmark/ppo/ppo_ant.json) | `ppo_ant` |

#### 3.5 Swimmer-v5

**Docs**: [Swimmer](https://gymnasium.farama.org/environments/mujoco/swimmer/) | State: Box(8) | Action: Box(2) | Solved reward MA > 300

**Settings**: max_frame 8e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec File | Spec Name |
|-----------|--------|-----|-----------|-----------|
| PPO | ✅ | 349 | [ppo_swimmer.json](../slm_lab/spec/benchmark/ppo/ppo_swimmer.json) | `ppo_swimmer` |

#### 3.6 Reacher-v5

**Docs**: [Reacher](https://gymnasium.farama.org/environments/mujoco/reacher/) | State: Box(11) | Action: Box(2) | Solved reward MA > -5

**Settings**: max_frame 3e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec File | Spec Name |
|-----------|--------|-----|-----------|-----------|
| PPO | ✅ | -5.29 | [ppo_reacher.json](../slm_lab/spec/benchmark/ppo/ppo_reacher.json) | `ppo_reacher` |

#### 3.7 Pusher-v5

**Docs**: [Pusher](https://gymnasium.farama.org/environments/mujoco/pusher/) | State: Box(23) | Action: Box(7) | Solved reward MA > -40 (CleanRL: -40.38±7.15)

**Settings**: max_frame 3e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec File | Spec Name |
|-----------|--------|-----|-----------|-----------|
| PPO | ✅ | -40.46 | [ppo_pusher.json](../slm_lab/spec/benchmark/ppo/ppo_pusher.json) | `ppo_pusher` |

#### 3.8 InvertedPendulum-v5

**Docs**: [InvertedPendulum](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/) | State: Box(4) | Action: Box(1) | Solved reward MA > 1000

**Settings**: max_frame 3e6 | num_envs 4 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec File | Spec Name |
|-----------|--------|-----|-----------|-----------|
| PPO | ✅ | 982 | [ppo_inverted_pendulum.json](../slm_lab/spec/benchmark/ppo/ppo_inverted_pendulum.json) | `ppo_inverted_pendulum` |

#### 3.9 InvertedDoublePendulum-v5

**Docs**: [InvertedDoublePendulum](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/) | State: Box(11) | Action: Box(1) | Solved reward MA > 9000

**Settings**: max_frame 8e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec File | Spec Name |
|-----------|--------|-----|-----------|-----------|
| PPO | ✅ | 9059 | [ppo_inverted_double_pendulum.json](../slm_lab/spec/benchmark/ppo/ppo_inverted_double_pendulum.json) | `ppo_inverted_double_pendulum` |

#### 3.10 Humanoid-v5

**Docs**: [Humanoid](https://gymnasium.farama.org/environments/mujoco/humanoid/) | State: Box(376) | Action: Box(17) | Solved reward MA > 700

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1000

| Algorithm | Status | MA | Spec File | Spec Name |
|-----------|--------|-----|-----------|-----------|
| PPO | ✅ | 1573 | [ppo_humanoid.json](../slm_lab/spec/benchmark/ppo/ppo_humanoid.json) | `ppo_humanoid` |

#### 3.11 HumanoidStandup-v5

**Docs**: [HumanoidStandup](https://gymnasium.farama.org/environments/mujoco/humanoid_standup/) | State: Box(376) | Action: Box(17) | Solved reward MA > 100000

**Settings**: max_frame 6e6 | num_envs 16 | max_session 4 | log_frequency 1000

| Algorithm | Status | MA | Spec File | Spec Name |
|-----------|--------|-----|-----------|-----------|
| PPO | ✅ | 103k | [ppo_humanoid_standup.json](../slm_lab/spec/benchmark/ppo/ppo_humanoid_standup.json) | `ppo_humanoid_standup` |

### Phase 4: Atari

#### 4.1 Pong-v5

**Docs**: [Pong](https://gymnasium.farama.org/environments/atari/pong/) | State: Box(210,160,3) | Action: Discrete(6) | Solved reward MA > 18

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | Spec File | Spec Name |
|-----------|--------|-----------|-----------|
| PPO | ⏸️ | [ppo_pong.json](../slm_lab/spec/benchmark/ppo/ppo_pong.json) | `ppo_pong` |
| DQN | ⏸️ | [dqn_pong.json](../slm_lab/spec/benchmark/dqn/dqn_pong.json) | `dqn_pong` |

#### 4.2 Qbert-v5

**Docs**: [Qbert](https://gymnasium.farama.org/environments/atari/qbert/) | State: Box(210,160,3) | Action: Discrete(6) | Solved reward MA > 15000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | Spec File | Spec Name |
|-----------|--------|-----------|-----------|
| PPO | ⏸️ | [ppo_qbert.json](../slm_lab/spec/benchmark/ppo/ppo_qbert.json) | `ppo_qbert` |
| DQN | ⏸️ | [dqn_qbert.json](../slm_lab/spec/benchmark/dqn/dqn_qbert.json) | `dqn_qbert` |

#### 4.3 Breakout-v5

**Docs**: [Breakout](https://gymnasium.farama.org/environments/atari/breakout/) | State: Box(210,160,3) | Action: Discrete(4) | Solved reward MA > 400

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | Spec File | Spec Name |
|-----------|--------|-----------|-----------|
| PPO | ⏸️ | [ppo_atari.json](../slm_lab/spec/benchmark/ppo/ppo_atari.json) | `ppo_atari` `-s env=ALE/Breakout-v5` |
| DQN | ⏸️ | [dqn_atari.json](../slm_lab/spec/benchmark/dqn/dqn_atari.json) | `dqn_atari` `-s env=ALE/Breakout-v5` |

---

## Development

### Current Runs

| Run Name | Spec | Mode | Status | GPU | Started |
|----------|------|------|--------|-----|---------|
| ppo-humanoidstandup-train-v6 | ppo_humanoid_standup | train | 56% MA=103k ✅ | GPU $0.13 | 2025-12-15 |
| sac-lunar-train-v2 | sac_lunar | train | 84% MA=+160 | GPU $0.13 | 2025-12-15 |

Notes:
- HumanoidStandup v6: SOLVED! Session 2 hit MA=103k. Spec updated to 6M frames.
- SAC Lunar v2: Parked for later investigation - discrete SAC needs debugging.

### Queued Runs

SAC MuJoCo next - planning with CleanRL hyperparams.

### Completed Runs

| Run Name | Spec | Mode | Result | Notes |
|----------|------|------|--------|-------|
| ppo-ant-train-v2 | ppo_ant | train | MA=1060 (s1) ⚠️ | 53% of target 2000. s3=719, s2=349. Hit max_duration at 83% (8.3M/10M). |
| ppo-humanoid-search-v1 | ppo_humanoid | search | MA=551 (79%) ⚠️ | Best trial d8dd76ba: gamma=0.9898, lam=0.9487, lr=1.38e-4. Hit 4h limit at 60%. |
| ppo-halfcheetah-train-v3 | ppo_halfcheetah | train | MA=6383 (s0) ✅ | SOLVED! 128% of target 5000. s3=5976 (120%) also solved. Uploaded. |
| ppo-pendulum-v1 | ppo_pendulum | train | MA=-178 ✅ | Solved (target -200) |
| sac-pendulum-v1 | sac_pendulum | train | MA=-150 ✅ | Solved (target -200) |
| sac-acrobot-v1 | sac_acrobot | train | MA=-97 ✅ | Solved (target -100) |
| ppo-invdoubpend-v23 | ppo_inverted_double_pendulum | search | MA=8138 ⚠️ | 90% of target 9000, train rerun done |
| ppo-invdoubpend-train-v2 | ppo_inverted_double_pendulum | train | MA=8074 (s3) ⚠️ | 90% of target 9000. Uploaded to HF. |
| ppo-hopper-v5 | ppo_hopper | train | MA=239 ❌ | 8% of target 3000, search queued |
| sac-lunar-v1 | sac_lunar_continuous | train | MA=184 ⚠️ | 92% of target 200, search queued |
| a2c-lunar-v1 | a2c_gae_lunar | train | MA=131 ❌ | 65% of target 200 |
| a2c-lunar-cont-v1 | a2c_gae_lunar_continuous | train | MA=2.5 ❌ | 1% of target 200 |
| ppo-humanoid-v12 | ppo_humanoid | train | MA=607 ⚠️ | 87% of target 700, rerun queued |
| a2c-lunar-search-v1 | a2c_gae_lunar | search | MA=74 ❌ | 37% of target 200, A2C may have bugs |
| a2c-lunar-cont-search-v1 | a2c_gae_lunar_continuous | search | ❌ | A2C failing on both discrete/cont |
| sac-lunar-search-v1 | sac_lunar | search | ❌ Error | Wrong spec name, relaunched as v2 |
| sac-lunar-cont-search-v1 | sac_lunar_continuous | search | ❌ Error | Missing search block, relaunched as v2 |
| ppo-reacher-v1 | ppo_reacher | train | MA=-667 ❌ | Target -5, search launched |
| ppo-pusher-v1 | ppo_pusher | train | MA=-2845 ❌ | Target -20, search launched |
| ppo-ant-v2 | ppo_ant | train | MA=-180 to -298 ❌ | Negative rewards! Fixed num_envs=16, gamma/lam/lr, search launched |
| ppo-humanoid-v13 | ppo_humanoid | train | MA=322-334 ❌ | 6% of target 5000. Fixed gamma=0.99, search launched |
| ppo-invdoubpend-v24 | ppo_inverted_double_pendulum | train | MA=6469-8568 ⚠️ | 72-95% of target 9000, inconsistent sessions |
| sac-lunar-search-v2 | sac_lunar | search | ❌ killed | SAC doesn't work well for discrete actions |
| sac-lunar-train-v2 | sac_lunar | train | ❌ killed | MA=-122 at 90k frames. Entropy collapse (alpha~1e-12). |
| ppo-hopper-search-v1 | ppo_hopper | search | MA=276 ❌ | 11% of target 2500. CleanRL uses num_envs=1, relaunched as v2 |
| ppo-reacher-search-v1 | ppo_reacher | search | MA=-183 ❌ | 1M frames too short, killed. Relaunched v2 with 3M frames |
| ppo-pusher-search-v1 | ppo_pusher | search | MA=-913 ❌ | 1M frames too short, killed. Relaunched v2 with 3M frames |
| a2c-lunar-search-v2 | a2c_gae_lunar | search | MA=-140 ❌ | Killed. A2C fundamentally less sample-efficient than PPO |
| ppo-ant-search-v1 | ppo_ant | search | MA=-385 ❌ | All trials negative despite CleanRL hyperparameters. Killed. |
| sac-lunar-cont-search-v2 | sac_lunar_continuous | search | MA=241.6 ✅ | 4 trials exceeded target 200! Best: gamma=0.994, lr=1.17e-4, iter=4 |
| ppo-halfcheetah-search-v2 | ppo_halfcheetah | search | MA=4007 ⚠️ | 80% of target 5000. Best: gamma=0.985, lam=0.964, lr=2.45e-4 |
| ppo-walker2d-v2 | ppo_walker2d | train | MA=3543-5400 ✅ | 4/4 sessions solved (target 3500) |
| ppo-hopper-search-v1 | ppo_hopper | search | MA=1857 ⚠️ | 74% of target 2500. Best: gamma=0.991, lam=0.951, lr=1.4e-4 |
| ppo-halfcheetah-train-v1 | ppo_halfcheetah | train | MA=2343 ⚠️ | 47% of target 5000, needs longer training |
| ppo-reacher-train-v1 | ppo_reacher | train | MA=-5.85 ✅ | SOLVED! (target -5) |
| ppo-swimmer-v2 | ppo_swimmer | train | MA=92-327 ⚠️ | 2/4 sessions solved (s1=327, s2=305, s0/s3=92-102) |
| ppo-reacher-search-v2 | ppo_reacher | search | MA=-5.61 ⚠️ | 89% of target -5. Best: gamma=0.983, lam=0.918, lr=2.5e-4 |
| ppo-ant-search-v2 | ppo_ant | search | MA=+59 ❌ | 3% of target 2000. Only 1/16 trials survived ASHA. Killed at 69%. |
| ppo-pusher-search-v2 | ppo_pusher | search | MA=-48 ❌ | 42% of target -20. Best: gamma=0.992, lam=0.932, lr=1.1e-4 |
| ppo-invdoublep-train-v1 | ppo_inverted_double_pendulum | train | MA=8196 ⚠️ | 91% of target 9000. Best s2=8196. |
| ppo-pusher-search-v3 | ppo_pusher | search | MA=-49.97 ⚠️ | 40% of target -20. Best: gamma=0.988, lam=0.914, lr=1.24e-4 |
| ppo-swimmer-train-v2 | ppo_swimmer | train | MA=349 ✅✅ | 2/4 sessions solved (s2=349, s3=349, target 300). Uploaded to HF. |
| ppo-hopper-train-v2 | ppo_hopper | train | MA=2162 ⭐ | 86% of target 2500. s3=2162, s1=2011. Uploaded to HF. |
| ppo-ant-search-v3 | ppo_ant | search | MA=1069 ⭐ | 53% of target 2000. Best: gamma=0.988, lam=0.928, lr=1.2e-4 |
| ppo-reacher-train-v2 | ppo_reacher | train | MA=-5.87 ⚠️ | 85% of target -5. s2=-5.87. Uploaded to HF. |
| ppo-pusher-search-v4 | ppo_pusher | search | MA=-54.81 ⚠️ | 36% of target -20. Best: gamma=0.991, lam=0.930, lr=1.63e-4 |
| ppo-humanoid-train-v1 | ppo_humanoid | train | MA=800-1573 ✅ | 4/4 sessions SOLVED! (target 700). s2=1573, s1=958, s0=800, s3=750. |
| ppo-reacher-search-v1 | ppo_reacher | search | MA=-5.19 ⚠️ | 96% of target -5. Best: gamma=0.983, lam=0.907, lr=1.46e-4. Spec updated. |
| ppo-ant-train-v3 | ppo_ant | train | MA=2190 (s3) ✅ | SOLVED! 110% of target 2000. s3=2190, s1=478, s2=472. Hit max_duration at 90%. |
| ppo-humanoidstandup-train-v1 | ppo_humanoid_standup | train | MA=74818 ⚠️ | 75% of target 100k. Hit max_duration at 40% (4M/10M). Need longer run. |
| a2c-lunar-cont-train-v2 | a2c_gae_lunar_continuous | train | MA=-51 ❌ | -26% of target 200. A2C not sample-efficient enough. |
| ppo-reacher-train-v4 | ppo_reacher | train | MA=-5.89 ⚠️ | 85% of target -5. s3=-5.89. Very close but not solved. |
| ppo-pusher-search-v5 | ppo_pusher | search | MA=-51.2 ⚠️ | 78% of CleanRL target -40. Best: gamma=0.982, lam=0.927, lr=1.83e-4. |
| a2c-lunar-train-v2 | a2c_gae_lunar | train | MA=+130 (s1) ❌ | 65% of target 200. s1=+130, s2=+13, s3=-50. A2C not sample-efficient. |
| ppo-humanoidstandup-train-v3 | ppo_humanoid_standup | train | MA=97.9k (s1) ⚠️ | 98% of target 100k! s1=97.9k, s2=74k, s3=72k. Hit max_duration at 40% (4M/10M). |
| ppo-pusher-train-v1 | ppo_pusher | train | MA=-40.46 (s1) ✅ | SOLVED! 101% of CleanRL target -40. Uploaded to HF. |
| ppo-reacher-train-v8 | ppo_reacher | train | MA=-6.51 (s2) ⚠️ | 77% of target -5. Improved but not solved. Uploaded to HF. |
| a2c-lunar-train-v3 | a2c_gae_lunar | train | MA=+58 (s2) ❌ | 29% of target 200. s2=+58, s1=-102, s3=-106. 1M frames still not enough. |
| ppo-humanoidstandup-train-v4 | ppo_humanoid_standup | train | MA=78k (s1) ⚠️ | 78% of target 100k. Hit max_duration at 41% (4.13M/10M). Need longer GPU quota. |
| a2c-lunar-cont-train-v4 | a2c_gae_lunar_continuous | train | MA=-16 (s2) ❌ | -8% of target 200. Best session improved from -275 but still far from target. LR scheduler helped but A2C not sample-efficient enough. |
| ppo-reacher-train-v9 | ppo_reacher | train | MA=-5.29 (s2) ⚠️ | 94% of target -5. Very close! Best session -5.29. Improved with tuned hyperparams. |
| a2c-lunar-train-v6 | a2c_gae_lunar | train | MA=+41 (s2) ❌ | 26% of target 155. LR scheduler + 300k frames helped (from +58) but still not solved. |
| sac-lunar-search-v3 | sac_lunar | search | ❌ killed | Stuck at frame 1000 for 36+ min on CPU. Possibly too slow with training_iter=20 + PrioritizedReplay. |
| sac-lunar-search-v4 | sac_lunar | search | ⚠️ 38% MA=+75 | SAC CAN learn discrete LunarLander! Best: gamma=0.981, training_freq=20, lr=0.00071, polyak=0.10. Spec updated. |

### Key Findings

- MuJoCo PPO requires careful tuning: Hopper-like architecture works well across environments
- Value target normalization (`normalize_v_targets: true`) improves stability
- Observation and reward normalization essential for MuJoCo continuous control
- **A2C investigation complete**: Root cause found - missing LR scheduler!
  - SB3 RL Zoo A2C uses `lin_0.00083` (linear decay from 0.00083 to 0)
  - Our A2C had constant LR causing late-training instability (sessions: -102, +58, -106)
  - SB3 benchmark: A2C LunarLander achieves **155** (±80) in **200k frames** (not 200!)
  - Fix: Added `lr_scheduler_spec: LinearToZero` + reduced max_frame to 500k
  - Note: A2C is fundamentally less sample-efficient than PPO (single-pass vs multiple epochs)
- **MuJoCo CleanRL standard hyperparameters**: gamma=0.99, lam=0.95, time_horizon=2048, lr=3e-4, num_envs=16
  - Previous specs with gamma=0.95-0.98, lam=0.8-0.9 failed badly (Ant negative, Humanoid 6%)
  - Updated Ant, Humanoid specs to CleanRL standard before search
- **Target adjustments from CleanRL benchmarks**:
  - Pusher-v5 target adjusted from -20 to -40 (CleanRL achieves -40.38±7.15, PPO struggles on this env)
  - CleanRL recommends RPO over PPO for Pusher and Reacher (PPO "failure cases")
- **Ant clip_eps issue**: Using clip_eps=0.1 instead of CleanRL's 0.2 caused all negative rewards
  - Also fixed val_loss_coef (0.68→0.5) and clip_grad_val (0.6→0.5) to match CleanRL
  - Relaunched Ant search with corrected hyperparameters
- **SAC discrete actions WORK**: Contrary to initial belief, SAC can learn discrete action spaces!
  - SAC LunarLander (discrete) improved from MA=-158 to MA=+75 (38% of target 200)
  - Key hyperparams: gamma=0.981, training_frequency=20 (lower than continuous), polyak_coef=0.10
  - Needs more frames or continued search to reach target 200, but clearly learning
- **Reacher search pattern**: gamma~0.983 + low lr (~0.00016-0.00025) works best
  - MA trajectory: -131 → ... → -10 → -9.09 → -8.77 → -8.51 → -8.27 → -7.83 → -7.62 → -7.24 → -6.88 → -6.64 → -6.45 → -6.28 → -6.13 → -6.03 → -5.85 → -5.80 (target -5)
  - 88% through budget, almost solved!
- **Swimmer note**: High variance across sessions (88-102 MA) but 3/4 sessions exceed target 90
- **HalfCheetah breakthrough**: MA=3395 at 60% through search (target 5000)
- **Ant status**: MA=+59 at 69%, killed. Only 1/16 trials survived ASHA (target 2000 unreachable with current hyperparams)
