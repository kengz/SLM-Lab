# SLM-Lab Benchmarks

Systematic algorithm validation across Gymnasium environments.

**Updated**: 2025-12-20

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
| 2 | Box2D | 2 | ✅ | ✅ | 📊 | ✅ | ✅ 100% |
| 3 | MuJoCo | 11 | ✅ | N/A | ⏸️ | ✅ 4/4 | ✅ PPO + SAC done |
| 4 | Atari | 6+ | ⚠️ 2/9 | ⏸️ | N/A | ⏸️ | 🔄 Pong+Breakout solved |

**Legend**: ✅ Solved | ⚠️ Close (>80%) | 📊 Acceptable (historical) | ❌ Failed | 🔄 In progress | ⏸️ Not started | N/A Not applicable

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
| A2C | 📊 | +41 | [a2c_gae_lunar.json](../slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json) | `a2c_gae_lunar` ¹ |

¹ A2C LunarLander: Historical SLM-Lab results also showed <100 at 300k frames. A2C is fundamentally less sample-efficient than PPO (single-pass vs multiple epochs per batch). Result is acceptable.

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

**Settings**: PPO: max_frame 1e6, num_envs 16 | SAC: max_frame 1e6, num_envs 1 (SB3 standard)

| Algorithm | Status | MA | Spec File | Spec Name |
|-----------|--------|-----|-----------|-----------|
| PPO | ✅ | 2914 | [ppo_hopper.json](../slm_lab/spec/benchmark/ppo/ppo_hopper.json) | `ppo_hopper` |
| SAC | ✅ | 2719 | [sac_hopper.json](../slm_lab/spec/benchmark/sac/sac_hopper.json) | `sac_hopper` |

#### 3.2 HalfCheetah-v5

**Docs**: [HalfCheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) | State: Box(17) | Action: Box(6) | Solved reward MA > 5000

**Settings**: max_frame 8e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec File | Spec Name |
|-----------|--------|-----|-----------|-----------|
| PPO | ✅ | 6383 | [ppo_halfcheetah.json](../slm_lab/spec/benchmark/ppo/ppo_halfcheetah.json) | `ppo_halfcheetah` |
| SAC | ✅ | 7800+ | [sac_halfcheetah.json](../slm_lab/spec/benchmark/sac/sac_halfcheetah.json) | `sac_halfcheetah` ³ |

³ SAC HalfCheetah solved at 37% training (~1.1M frames). All 4 sessions exceeded target 5000 (MA=6888-8172). Run terminated before HF upload.

#### 3.3 Walker2d-v5

**Docs**: [Walker2d](https://gymnasium.farama.org/environments/mujoco/walker2d/) | State: Box(17) | Action: Box(6) | Solved reward MA > 3500

**Settings**: PPO: max_frame 8e6, num_envs 16 | SAC: max_frame 1e6, num_envs 1 (SB3 standard)

| Algorithm | Status | MA | Spec File | Spec Name |
|-----------|--------|-----|-----------|-----------|
| PPO | ✅ | 5700 | [ppo_walker2d.json](../slm_lab/spec/benchmark/ppo/ppo_walker2d.json) | `ppo_walker2d` |
| SAC | ✅ | 3824 | [sac_walker2d.json](../slm_lab/spec/benchmark/sac/sac_walker2d.json) | `sac_walker2d` |

#### 3.4 Ant-v5

**Docs**: [Ant](https://gymnasium.farama.org/environments/mujoco/ant/) | State: Box(105) | Action: Box(8) | Solved reward MA > 2000

**Settings**: max_frame 8e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | Spec File | Spec Name |
|-----------|--------|-----|-----------|-----------|
| PPO | ✅ | 2190 | [ppo_ant.json](../slm_lab/spec/benchmark/ppo/ppo_ant.json) | `ppo_ant` |
| SAC | ✅ | 2022 | [sac_ant.json](../slm_lab/spec/benchmark/sac/sac_ant.json) | `sac_ant` |

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

**Shared PPO Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4 | ConvNet [32,64,64] + 512fc | lr=2.5e-4 | lam=0.95 | LR scheduler

**Template**: Use [ppo_atari.json](../slm_lab/spec/benchmark/ppo/ppo_atari.json) with `-s env=ALE/<ENV>-v5`

**Continuous Actions**: ALE supports `continuous=True` for continuous action Atari (CALE). Use for SAC/PPO continuous.

| Environment | Target | PPO Status | PPO MA | Run Command |
|-------------|--------|------------|--------|-------------|
| Pong | 21 | ✅ | 18.33 | `ppo_pong` (dedicated spec) |
| Pong (cont) | 21 | ❌ | -19.6 | `ppo_atari_continuous -s env=ALE/Pong-v5` ² |
| Breakout | 30 | ✅ | 42 | `-s env=ALE/Breakout-v5 ppo_atari` (clip_vloss improved: 40-42 vs 36) |
| Qbert | 4425 | ❌ 4% | 190 | `-s env=ALE/Qbert-v5 ppo_atari` (train running) |
| SpaceInvaders | 1000 | ❌ 5% | 48 | `-s env=ALE/SpaceInvaders-v5 ppo_atari` (search running) |
| BeamRider | 1590 | ❌ 1% | 33 | `-s env=ALE/BeamRider-v5 ppo_atari` (search running) |
| Seaquest | 1740 | 🔄 | ~90 | `-s env=ALE/Seaquest-v5 ppo_atari` (done, uploaded HF) |
| Enduro | 2000 | ⚠️ 39% | 788 | `-s env=ALE/Enduro-v5 ppo_atari` (clip_vloss: 788 vs 410 baseline, +89%!) |
| MsPacman | 1500 | ❌ 10% | 155 | `-s env=ALE/MsPacman-v5 ppo_atari` (clip_vloss: 155 vs 143 baseline, +8%) |

² Continuous action Atari (CALE) - PPO with continuous=True did not learn. MA stuck at -19.6 (random policy level). May need architecture changes for CALE.

---

## Development

### Current Runs

None.

### Recently Completed

- **sac-walker2d-sb3**: ✅ SOLVED! MA=3824 (109% of target 3500). SB3/CleanRL hyperparams work! 1M frames, num_envs=1, tau=0.005. Uploaded to HF.
- **sac-hopper-sb3**: ✅ SOLVED! MA=2719 (109% of target 2500). SB3/CleanRL hyperparams work! 1M frames, num_envs=1, tau=0.005. Uploaded to HF.
- **ppo-qbert-clip**: ❌ MA=177 (4% of target 4425). clip_vloss didn't help Qbert. Uploaded to HF.
- **ppo-spaceinvaders-clip**: ❌ MA=41 (4% of target 1000). clip_vloss didn't help SpaceInvaders. Uploaded to HF.
- **sac-walker2d-v4**: ⚠️ DONE (terminated at 6h/59%). MA=1178-2560 (s1=2560, 73% of target 3500). Replaced by sac-walker2d-sb3 with SB3 params.
- **sac-hopper-6h**: ⚠️ DONE (terminated at 6h/59%). MA=2382-2487 (s0=2487, 99.5% of target 2500!). Needs longer run to fully solve.
- **ppo-enduro-v2**: ✅ DONE! MA=775-788 (baseline 410). **clip_vloss massive boost!** +89% improvement. Uploaded to HF.
- **ppo-mspacman-clip**: ✅ DONE! MA=145-155 (baseline 143). **clip_vloss helps!** +2-8% improvement. Uploaded to HF.
- **ppo-breakout-clip**: ✅ DONE! MA=40-42 (s1=40.74, s2=40.45, s3=42.38). **clip_vloss helps!** +15-17% vs baseline MA=36. Uploaded to HF.
- **ppo-breakout-v5**: DONE. MA=13.8, **WORSE** than baseline MA=36 (62% regression!). Root cause: `terminal_on_life_loss=True` does the OPPOSITE of CleanRL's EpisodicLifeEnv. **REVERTED** all episodic_life changes (`9a645d5b`).
- **ppo-breakout-v4**: FAILED - gymnasium.error.Error: Cannot use vector_entry_point mode with wrappers. Fixed in v5 by using async mode.
- **ppo-breakout-v3**: FAILED - TypeError: AtariVectorEnv doesn't accept terminal_on_life_loss directly. Need to use wrappers parameter with functools.partial. Fixed in v4.
- **ppo-breakout-init**: Head init test (actor_init_std=0.01, critic_init_std=1.0). MA=15.0 avg (s0=15.27, s1=14.30, s2=14.70, s3=15.55), **WORSE** than baseline MA=36. Reverted spec (kept framework code). Commits: `2a3b5341` (feat), `1951c73b` (MLPNet).
- **ppo-breakout-v2**: Adam eps=1e-5 test. MA=14, **WORSE** than previous MA=36. Reverted eps change.
- **sac-hopper-v2**: 4h max_duration hit at 760k/1.2M frames. Best MA=1722 (69% of target 2500). SAC Hopper needs >2M frames to solve.
- **sac-walker2d-v2**: 4h max_duration hit at 730k/1.2M frames. Best MA=907 (26% of target 3500). SAC Walker2d needs >3M frames to solve.
- **sac-hopper-1m**: DONE! Best MA=1670 (s1), 67% of target 2500. 1M frames in 3.2h.
- **sac-walker2d-1m**: DONE! Best MA=1193 (s0), 34% of target 3500. 1M frames in 3.2h.
- **SAC Hopper search-v2**: DONE! MA=1164, max_strength=2302 (92% of target 2500!). Best: gamma=0.998, lr=8.4e-4. Train-v5 launched.
- **SAC Walker2d search-v2**: DONE! MA=810, max_strength=2121 (61% of target 3500). Best: gamma=0.996, lr=5.4e-4. Train-v5 launched.
- **PPO Enduro search-v2**: DONE! MA=703 (35% of target 2000). **71% BETTER than old results (MA=410)!** episodic_life fix worked!
- **PPO BeamRider search-v2**: DONE. MA=38 (2.4% of target 1590). Limited improvement.
- **PPO Qbert search-v2**: DONE. MA=139 (3.1% of target 4425). episodic_life helped.
- **PPO SpaceInvaders search-v2**: DONE. MA=35 (3.5% of target 1000). episodic_life helped but not enough.
- **PPO MsPacman search-v2**: DONE. MA=116 (7.7% of target 1500). episodic_life helped but not enough.
- **SAC Hopper train-v4**: DONE! s0 MA=1624 (65% of 2500), s2 MA=1507 (60%). Uploaded to HF.
- **SAC Walker2d train-v4**: DONE! s1 MA=1007 (29% of 3500), s3 MA=777, s0 MA=748. Uploaded to HF.
- **PPO Qbert search-v2**: DONE! Best MA=201 (1.5% of 13k target). Best: lam=0.902, gamma=0.987, lr=4.95e-4. Low-lam trials dominated. Uploaded to HF.
- **SAC HalfCheetah train-v4**: ✅ SOLVED! ALL 4 sessions exceeded target 5000. Uploaded to HuggingFace.
- **SAC Hopper search-v1**: DONE! Best MA=1902 (76% target 2500). Hyperparams: gamma=0.998, lr=8.4e-4, polyak=0.002
- **SAC Walker2d search-v1**: DONE! Best MA=911 (26% target 3500). Hyperparams: gamma=0.996, lr=5.4e-4, polyak=0.002

### Queued Runs

**Next up:**
- SAC Ant search (after HalfCheetah completes)

**Later:**
- CALE investigation - continuous Atari not learning (need architecture changes?)
- SAC Humanoid, Swimmer (after Hopper/Walker2d complete)

### Completed Runs

| Run Name | Spec | Mode | Result | Notes |
|----------|------|------|--------|-------|
| sac-ant-search-v1 | sac_ant | search | MA=2022 ✅ | SOLVED! 101% of target 2000. gamma=0.984, lr=1.05e-4, polyak=0.019. Spec updated. |
| ppo-breakout-train-v4 | ppo_atari (CleanRL) | train | MA=39.1 ⭐ | 9.8% of target 400. s2=39.09, s3=39.10, s1=37.2. Uploaded. |
| ppo-qbert-train-v3 | ppo_atari (CleanRL) | train | MA=186 ⭐ | 1.2% of target 15000. s3=186, s2=185, s1=158. Uploaded. |
| ppo-breakout-search-v1 | ppo_atari | search | MA=40.15 ⭐ | 10% of target 400. **CONFIRMS lam fix**: Best lam=0.947, low lam (0.73-0.87) terminated at 1M with MA<9. Uploaded. |
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
| sac-lunar-cont-search-v3 | sac_lunar_continuous | search | ✅ MA=241.6 | SOLVED! 121% of target 200. 4 trials exceeded target. Best: gamma=0.994, lr=1.17e-4, training_iter=4. |
| sac-lunar-train-v3 | sac_lunar | train | ❌ MA=100 | 50% of target 200. Mean MA=100, best session MA=170. Bug identified: alpha_loss_discrete uses log_alpha not alpha. |
| sac-halfcheetah-train-v1 | sac_halfcheetah | train | ✅ MA=7800+ | SOLVED at 37% training! All 4 sessions exceeded target 5000 (MA=6888-8172). Terminated before HF upload. |
| sac-hopper-train-v1 | sac_hopper | train | ❌ MA=1291 | 52% of target 2500. Completed but not solved. |
| sac-walker2d-train-v1 | sac_walker2d | train | ❌ MA=972 | 28% of target 3500. Completed but not solved. |
| ppo-qbert-train-v1 | ppo_atari `-s env=Qbert` | train | ❌ MA=119 | 0.8% of target 15000. PPO struggles with Qbert. |
| ppo-breakout-train-v2 | ppo_atari `-s env=Breakout` | train | ❌ MA=59.5 | 15% of target 400. PPO needs tuning for Breakout. |
| ppo-pong-cont-train-v1 | ppo_atari_continuous `-s env=Pong` | train | ❌ MA=-19.6 | Not learning. CALE (continuous Atari) may need different architecture. |

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
- **SAC discrete actions WORK but have a bug**: SAC can learn discrete action spaces but has implementation issue
  - SAC LunarLander (discrete) improved from MA=-158 to MA=+75 to MA=+100 over multiple runs
  - **BUG IDENTIFIED**: `calc_alpha_loss_discrete` uses `self.log_alpha` instead of `self.log_alpha.exp()` (= alpha)
    - Continuous version: `-(self.log_alpha.exp() * (log_probs + target_entropy)).mean()`
    - Discrete version: `self.log_alpha * (target_entropy - entropy_current)` ← INCORRECT
  - This causes incorrect gradient magnitudes for entropy temperature tuning
  - Fix: Change to `self.log_alpha.exp() * (entropy_current - self.target_entropy)` to match continuous behavior
- **Reacher search pattern**: gamma~0.983 + low lr (~0.00016-0.00025) works best
  - MA trajectory: -131 → ... → -10 → -9.09 → -8.77 → -8.51 → -8.27 → -7.83 → -7.62 → -7.24 → -6.88 → -6.64 → -6.45 → -6.28 → -6.13 → -6.03 → -5.85 → -5.80 (target -5)
  - 88% through budget, almost solved!
- **Swimmer note**: High variance across sessions (88-102 MA) but 3/4 sessions exceed target 90
- **HalfCheetah breakthrough**: MA=3395 at 60% through search (target 5000)
- **Ant status**: MA=+59 at 69%, killed. Only 1/16 trials survived ASHA (target 2000 unreachable with current hyperparams)
- **SAC HalfCheetah SOLVED**: MA=7800+ at just 37% training (1.1M frames). SAC is very sample-efficient on this env. All 4 sessions exceeded target 5000.
- **PPO Atari struggles**: Qbert (0.8% target) and Breakout (15% target) underperform significantly with standard hyperparameters. May need longer training or hyperparameter search.
- **CALE (Continuous Atari) not working**: PPO with continuous actions stuck at random policy level (MA=-19.6 on Pong). Architecture changes may be needed.
- **PPO Atari lam fix**: Original spec used lam=0.7, but CleanRL uses lam=0.95. Search confirmed lam~0.94 performs best. Fixed spec defaults.
- **PPO Atari episodic_life WRONG APPROACH - REVERTED**: Investigation complete after multiple failed attempts.
  - **What we tried**: gymnasium's `terminal_on_life_loss=True` parameter
  - **What CleanRL does**: Uses `EpisodicLifeEnv` wrapper which does the OPPOSITE
  - **EpisodicLifeEnv**: Episode ends only when ALL lives lost (standard full-game episodes)
  - **terminal_on_life_loss=True**: Episode ends when ANY life lost (micro-episodes)
  - **Result**: Enabling terminal_on_life_loss caused 62% regression (MA=13.8 vs baseline MA=36)
  - **Conclusion**: Gymnasium's default behavior (no episodic life handling) is correct. Reverted all changes.
  - Commits: `9a645d5b` (revert), `eb7f3108`, `6d6b8277`, `259093aa` (failed attempts)
- **PPO Atari head initialization - DID NOT HELP**: Tested CleanRL-style head init (actor_init_std=0.01, critic_init_std=1.0).
  - Result: MA=15.0 avg, **WORSE** than baseline MA=36 (58% regression!)
  - Framework code kept for future experimentation but reverted from ppo_atari.json spec
  - Commits: `2a3b5341` (ConvNet), `1951c73b` (MLPNet), `6461dfa1` (docs)
  - To revert framework: `git revert 1951c73b 2a3b5341` (or just don't use the params - backward compatible)
- **PPO Atari clip_vloss - CONFIRMED HELPFUL**: CleanRL-style value loss clipping (`clip_vloss=true`).
  - Clips value predictions relative to old predictions (similar to policy clipping)
  - Commit: `ee54c45d` (feat: add clip_vloss for CleanRL-style value loss clipping)
  - **Results across games**:
    - Breakout: MA=40-42 vs baseline 36 (+15-17%)
    - MsPacman: MA=155 vs baseline 143 (+8%)
    - Enduro: MA=788 vs baseline 410 (+89%!) - massive improvement
  - clip_vloss is now standard for PPO Atari
- **SAC MuJoCo spec fix - aligned with SB3/CleanRL**:
  - Previous specs used num_envs=16, max_frame=3M, training_freq=20 - this was wrong!
  - SB3/CleanRL use: num_envs=1, max_frame=1M, train_freq=1, tau=0.005
  - SB3 benchmarks: Hopper 2603, Walker2d 2292 in just 1M timesteps
  - Key insight: SAC is off-policy, doesn't benefit from vectorized envs like PPO
  - Commit: `929448c7` (feat: update SAC specs with SB3/CleanRL standard hyperparameters)
