# TorchArcNet Benchmark Results

Reproduction of all SLM-Lab benchmarks using TorchArcNet (declarative YAML specs via [torcharc](https://github.com/kengz/torcharc)) instead of hand-coded net classes. Specs are in `slm_lab/spec/benchmark_arc/`.

---

## Usage

After [installation](../README.md#quick-start), copy `SPEC_FILE` and `SPEC_NAME` from result tables below (Atari uses one shared spec file - see [Phase 4](#phase-4-atari)).

### Running Benchmarks

**Local** - runs on your machine (Classic Control: minutes):
```bash
slm-lab run SPEC_FILE SPEC_NAME train
```

**Remote** - cloud GPU via [dstack](https://dstack.ai), auto-syncs to HuggingFace:
```bash
source .env && slm-lab run-remote --gpu SPEC_FILE SPEC_NAME train -n NAME
```

Remote setup: `cp .env.example .env` then set `HF_TOKEN`. See [README](../README.md#cloud-training-dstack) for dstack config.

### Atari

All games share one spec file (54 tested, 5 hard exploration skipped). Use `-s env=ENV` to substitute. Runs take ~2-3 hours on GPU.

```bash
source .env && slm-lab run-remote --gpu -s env=ALE/Pong-v5 slm_lab/spec/benchmark_arc/ppo/ppo_atari_arc.yaml ppo_atari_arc train -n pong
```

### Download Results

Trained models and metrics sync to [HuggingFace](https://huggingface.co/datasets/SLM-Lab/benchmark). Pull locally:
```bash
source .env && slm-lab pull SPEC_NAME
slm-lab list  # see available experiments
```

### Environment Settings

Standardized settings for fair comparison. The **Settings** line in each result table shows these values.

| Env Category | num_envs | max_frame | log_frequency | grace_period |
|--------------|----------|-----------|---------------|--------------|
| Classic Control | 4 | 2e5-3e5 | 500 | 1e4 |
| Box2D | 8 | 3e5 | 1000 | 5e4 |
| MuJoCo | 16 | 1e6-10e6 | 1e4 | 1e5-1e6 |
| Atari | 16 | 10e6 | 10000 | 5e5 |

## Progress

| Phase | Category | Envs | REINFORCE | SARSA | DQN | DDQN+PER | A2C | PPO | SAC | Overall |
|-------|----------|------|-----------|-------|-----|----------|-----|-----|-----|---------|
| 1 | Classic Control | 3 | - | - | - | - | - | - | - | Pending |
| 2 | Box2D | 2 | N/A | N/A | - | - | - | - | - | Pending |
| 3 | MuJoCo | 11 | N/A | N/A | N/A | N/A | - | - | - | Pending |
| 4 | Atari | 59 | N/A | N/A | N/A | Skip | - | - | N/A | Pending |

**Legend**: ✅ Solved | ⚠️ Close (>80%) | 📊 Acceptable | ❌ Failed | - Pending | Skip Not started | N/A Not applicable

---

## Results

### Phase 1: Classic Control

#### 1.1 CartPole-v1

**Docs**: [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) | State: Box(4) | Action: Discrete(2) | Target reward MA > 400

**Settings**: max_frame 2e5 | num_envs 4 | max_session 4 | log_frequency 500

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME |
|-----------|--------|-----|-----------|-----------|
| REINFORCE | - | - | slm_lab/spec/benchmark_arc/reinforce/reinforce_arc.yaml | reinforce_cartpole_arc |
| SARSA | - | - | slm_lab/spec/benchmark_arc/sarsa/sarsa_arc.yaml | sarsa_boltzmann_cartpole_arc |
| DQN | - | - | slm_lab/spec/benchmark_arc/dqn/dqn_classic_arc.yaml | dqn_boltzmann_cartpole_arc |
| DDQN+PER | - | - | slm_lab/spec/benchmark_arc/dqn/dqn_classic_arc.yaml | ddqn_per_boltzmann_cartpole_arc |
| A2C | - | - | slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml | a2c_gae_cartpole_arc |
| PPO | - | - | slm_lab/spec/benchmark_arc/ppo/ppo_classic_arc.yaml | ppo_cartpole_arc |
| SAC | - | - | slm_lab/spec/benchmark_arc/sac/sac_classic_arc.yaml | sac_cartpole_arc |

#### 1.2 Acrobot-v1

**Docs**: [Acrobot](https://gymnasium.farama.org/environments/classic_control/acrobot/) | State: Box(6) | Action: Discrete(3) | Target reward MA > -100

**Settings**: max_frame 3e5 | num_envs 4 | max_session 4 | log_frequency 500

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME |
|-----------|--------|-----|-----------|-----------|
| DQN | - | - | slm_lab/spec/benchmark_arc/dqn/dqn_classic_arc.yaml | dqn_boltzmann_acrobot_arc |
| DDQN+PER | - | - | slm_lab/spec/benchmark_arc/dqn/dqn_classic_arc.yaml | ddqn_per_acrobot_arc |
| A2C | - | - | slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml | a2c_gae_acrobot_arc |
| PPO | - | - | slm_lab/spec/benchmark_arc/ppo/ppo_classic_arc.yaml | ppo_acrobot_arc |
| SAC | - | - | slm_lab/spec/benchmark_arc/sac/sac_classic_arc.yaml | sac_acrobot_arc |

#### 1.3 Pendulum-v1

**Docs**: [Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/) | State: Box(3) | Action: Box(1) | Target reward MA > -200

**Settings**: max_frame 3e5 | num_envs 4 | max_session 4 | log_frequency 500

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME |
|-----------|--------|-----|-----------|-----------|
| A2C | - | - | slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml | a2c_gae_pendulum_arc |
| PPO | - | - | slm_lab/spec/benchmark_arc/ppo/ppo_classic_arc.yaml | ppo_pendulum_arc |
| SAC | - | - | slm_lab/spec/benchmark_arc/sac/sac_classic_arc.yaml | sac_pendulum_arc |

### Phase 2: Box2D

#### 2.1 LunarLander-v3 (Discrete)

**Docs**: [LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) | State: Box(8) | Action: Discrete(4) | Target reward MA > 200

**Settings**: max_frame 3e5 | num_envs 8 | max_session 4 | log_frequency 1000

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME |
|-----------|--------|-----|-----------|-----------|
| DQN | - | - | slm_lab/spec/benchmark_arc/dqn/dqn_box2d_arc.yaml | dqn_concat_lunar_arc |
| DDQN+PER | - | - | slm_lab/spec/benchmark_arc/dqn/dqn_box2d_arc.yaml | ddqn_per_concat_lunar_arc |
| A2C | - | - | slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml | a2c_gae_lunar_arc |
| PPO | - | - | slm_lab/spec/benchmark_arc/ppo/ppo_box2d_arc.yaml | ppo_lunar_arc |
| SAC | - | - | slm_lab/spec/benchmark_arc/sac/sac_box2d_arc.yaml | sac_lunar_arc |

#### 2.2 LunarLander-v3 (Continuous)

**Docs**: [LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) | State: Box(8) | Action: Box(2) | Target reward MA > 200

**Settings**: max_frame 3e5 | num_envs 8 | max_session 4 | log_frequency 1000

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME |
|-----------|--------|-----|-----------|-----------|
| A2C | - | - | slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml | a2c_gae_lunar_continuous_arc |
| PPO | - | - | slm_lab/spec/benchmark_arc/ppo/ppo_box2d_arc.yaml | ppo_lunar_continuous_arc |
| SAC | - | - | slm_lab/spec/benchmark_arc/sac/sac_box2d_arc.yaml | sac_lunar_continuous_arc |

### Phase 3: MuJoCo

**Docs**: [MuJoCo environments](https://gymnasium.farama.org/environments/mujoco/) | State/Action: Continuous | Target: Practical baselines (no official "solved" threshold)

**Settings**: max_frame 4e6-10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

**Algorithms**: PPO and SAC. Network: MLP [256,256], orthogonal init. PPO uses tanh activation; SAC uses relu.

**Note on SAC frame budgets**: SAC uses higher update-to-data ratios (more gradient updates per step), making it more sample-efficient but slower per frame than PPO. SAC benchmarks use 1-2M frames (vs PPO's 4-10M) to fit within practical GPU wall-time limits (~6h). Scores may still be improving at cutoff.

**Spec Variants**: All PPO MuJoCo specs in one file: [ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml). YAML anchors share the base config; per-env specs override only what differs.

| SPEC_NAME | Envs | Key Config |
|-----------|------|------------|
| ppo_mujoco_arc | HalfCheetah, Walker, Humanoid, HumanoidStandup | gamma=0.99, lam=0.95 |
| ppo_mujoco_longhorizon_arc | Reacher, Pusher | gamma=0.997, lam=0.97 |
| ppo_ant_arc, ppo_hopper_arc, etc. | Individual envs | Per-env tuned hyperparams |

**Reproduce**: Copy `ENV`, `SPEC_NAME` from table. All use the same spec file, with `-s env=` and `-s max_frame=`:
```bash
source .env && slm-lab run-remote --gpu -s env=ENV -s max_frame=MAX_FRAME \
  slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml SPEC_NAME train -n NAME
```

| ENV | MAX_FRAME | SPEC_NAME |
|-----|-----------|-----------|
| Ant-v5 | 10e6 | ppo_ant_arc |
| HalfCheetah-v5 | 10e6 | ppo_mujoco_arc |
| Hopper-v5 | 4e6 | ppo_hopper_arc |
| Humanoid-v5 | 10e6 | ppo_mujoco_arc |
| HumanoidStandup-v5 | 4e6 | ppo_mujoco_arc |
| InvertedDoublePendulum-v5 | 10e6 | ppo_inverted_double_pendulum_arc |
| InvertedPendulum-v5 | 4e6 | ppo_inverted_pendulum_arc |
| Pusher-v5 | 4e6 | ppo_mujoco_longhorizon_arc |
| Reacher-v5 | 4e6 | ppo_mujoco_longhorizon_arc |
| Swimmer-v5 | 4e6 | ppo_swimmer_arc |
| Walker2d-v5 | 10e6 | ppo_mujoco_arc |

**SAC Reproduce**: All SAC MuJoCo specs in one file: [sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml). Env and max_frame are hardcoded per spec.
```bash
source .env && slm-lab run-remote --gpu \
  slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml SPEC_NAME train -n NAME

# Example: reproduce Hopper SAC
source .env && slm-lab run-remote --gpu \
  slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml sac_hopper_arc train -n sac-hopper
```

| ENV | MAX_FRAME | SPEC_NAME |
|-----|-----------|-----------|
| Ant-v5 | 2e6 | sac_ant_arc |
| HalfCheetah-v5 | 2e6 | sac_halfcheetah_arc |
| Hopper-v5 | 2e6 | sac_hopper_arc |
| Humanoid-v5 | 1e6 | sac_humanoid_arc |
| HumanoidStandup-v5 | 1e6 | sac_humanoid_standup_arc |
| InvertedDoublePendulum-v5 | 2e6 | sac_inverted_double_pendulum_arc |
| InvertedPendulum-v5 | 2e6 | sac_inverted_pendulum_arc |
| Pusher-v5 | 1e6 | sac_pusher_arc |
| Reacher-v5 | 1e6 | sac_reacher_arc |
| Swimmer-v5 | 2e6 | sac_swimmer_arc |
| Walker2d-v5 | 2e6 | sac_walker2d_arc |

#### 3.1 Ant-v5

**Docs**: [Ant](https://gymnasium.farama.org/environments/mujoco/ant/) | State: Box(105) | Action: Box(8) | Target reward MA > 2000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME |
|-----------|--------|-----|-----------|-----------|
| PPO | - | - | slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml | ppo_ant_arc |
| SAC | - | - | slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml | sac_ant_arc |

#### 3.2 HalfCheetah-v5

**Docs**: [HalfCheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) | State: Box(17) | Action: Box(6) | Target reward MA > 5000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME |
|-----------|--------|-----|-----------|-----------|
| PPO | - | - | slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml | ppo_mujoco_arc |
| SAC | - | - | slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml | sac_halfcheetah_arc |

#### 3.3 Hopper-v5

**Docs**: [Hopper](https://gymnasium.farama.org/environments/mujoco/hopper/) | State: Box(11) | Action: Box(3) | Target reward MA ~ 2000

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME |
|-----------|--------|-----|-----------|-----------|
| PPO | - | - | slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml | ppo_hopper_arc |
| SAC | - | - | slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml | sac_hopper_arc |

#### 3.4 Humanoid-v5

**Docs**: [Humanoid](https://gymnasium.farama.org/environments/mujoco/humanoid/) | State: Box(348) | Action: Box(17) | Target reward MA > 1000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME |
|-----------|--------|-----|-----------|-----------|
| PPO | - | - | slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml | ppo_mujoco_arc |
| SAC | - | - | slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml | sac_humanoid_arc |

#### 3.5 HumanoidStandup-v5

**Docs**: [HumanoidStandup](https://gymnasium.farama.org/environments/mujoco/humanoid_standup/) | State: Box(348) | Action: Box(17) | Target reward MA > 100000

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME |
|-----------|--------|-----|-----------|-----------|
| PPO | - | - | slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml | ppo_mujoco_arc |
| SAC | - | - | slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml | sac_humanoid_standup_arc |

#### 3.6 InvertedDoublePendulum-v5

**Docs**: [InvertedDoublePendulum](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/) | State: Box(9) | Action: Box(1) | Target reward MA ~8000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME |
|-----------|--------|-----|-----------|-----------|
| PPO | - | - | slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml | ppo_inverted_double_pendulum_arc |
| SAC | - | - | slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml | sac_inverted_double_pendulum_arc |

#### 3.7 InvertedPendulum-v5

**Docs**: [InvertedPendulum](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/) | State: Box(4) | Action: Box(1) | Target reward MA ~1000

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME |
|-----------|--------|-----|-----------|-----------|
| PPO | - | - | slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml | ppo_inverted_pendulum_arc |
| SAC | - | - | slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml | sac_inverted_pendulum_arc |

#### 3.8 Pusher-v5

**Docs**: [Pusher](https://gymnasium.farama.org/environments/mujoco/pusher/) | State: Box(23) | Action: Box(7) | Target reward MA > -50

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME |
|-----------|--------|-----|-----------|-----------|
| PPO | - | - | slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml | ppo_mujoco_longhorizon_arc |
| SAC | - | - | slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml | sac_pusher_arc |

#### 3.9 Reacher-v5

**Docs**: [Reacher](https://gymnasium.farama.org/environments/mujoco/reacher/) | State: Box(10) | Action: Box(2) | Target reward MA > -10

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME |
|-----------|--------|-----|-----------|-----------|
| PPO | - | - | slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml | ppo_mujoco_longhorizon_arc |
| SAC | - | - | slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml | sac_reacher_arc |

#### 3.10 Swimmer-v5

**Docs**: [Swimmer](https://gymnasium.farama.org/environments/mujoco/swimmer/) | State: Box(8) | Action: Box(2) | Target reward MA > 200

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME |
|-----------|--------|-----|-----------|-----------|
| PPO | - | - | slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml | ppo_swimmer_arc |
| SAC | - | - | slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml | sac_swimmer_arc |

#### 3.11 Walker2d-v5

**Docs**: [Walker2d](https://gymnasium.farama.org/environments/mujoco/walker2d/) | State: Box(17) | Action: Box(6) | Target reward MA > 3500

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME |
|-----------|--------|-----|-----------|-----------|
| PPO | - | - | slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml | ppo_mujoco_arc |
| SAC | - | - | slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml | sac_walker2d_arc |

### Phase 4: Atari

**Docs**: [Atari environments](https://ale.farama.org/environments/) | State: Box(84,84,4 after preprocessing) | Action: Discrete(4-18, game-dependent)

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 10000

**Environment**:
- Gymnasium ALE v5 with `life_loss_info=true`
- v5 uses sticky actions (`repeat_action_probability=0.25`) per [Machado et al. (2018)](https://arxiv.org/abs/1709.06009) best practices

**Algorithm Specs** (all use Nature CNN [32,64,64] + 512fc):
- **DDQN+PER**: Skipped - off-policy variants ~6x slower (~230 fps vs ~1500 fps), not cost effective at 10M frames
- **A2C**: [a2c_atari_arc.yaml](../slm_lab/spec/benchmark_arc/a2c/a2c_atari_arc.yaml) - RMSprop (lr=7e-4), training_frequency=32
- **PPO**: [ppo_atari_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_atari_arc.yaml) - AdamW (lr=2.5e-4), minibatch=256, horizon=128, epochs=4

**PPO Lambda Variants** (table shows best result per game):

| SPEC_NAME | Lambda | Best for |
|-----------|--------|----------|
| ppo_atari_arc | 0.95 | Strategic games (default) |
| ppo_atari_lam85_arc | 0.85 | Mixed games |
| ppo_atari_lam70_arc | 0.70 | Action games |

**Reproduce**:
```bash
# A2C
source .env && slm-lab run-remote --gpu -s env=ENV -s max_frame=1e7 \
  slm_lab/spec/benchmark_arc/a2c/a2c_atari_arc.yaml a2c_gae_atari_arc train -n NAME

# PPO
source .env && slm-lab run-remote --gpu -s env=ENV -s max_frame=1e7 \
  slm_lab/spec/benchmark_arc/ppo/ppo_atari_arc.yaml SPEC_NAME train -n NAME
```

| ENV | A2C Score | PPO Score | PPO SPEC_NAME |
|-----|-----------|-----------|---------------|
| ALE/AirRaid-v5 | - | - | ppo_atari_arc |
| ALE/Alien-v5 | - | - | ppo_atari_arc |
| ALE/Amidar-v5 | - | - | ppo_atari_lam85_arc |
| ALE/Assault-v5 | - | - | ppo_atari_lam85_arc |
| ALE/Asterix-v5 | - | - | ppo_atari_lam85_arc |
| ALE/Asteroids-v5 | - | - | ppo_atari_lam85_arc |
| ALE/Atlantis-v5 | - | - | ppo_atari_arc |
| ALE/BankHeist-v5 | - | - | ppo_atari_arc |
| ALE/BattleZone-v5 | - | - | ppo_atari_lam85_arc |
| ALE/BeamRider-v5 | - | - | ppo_atari_arc |
| ALE/Berzerk-v5 | - | - | ppo_atari_arc |
| ALE/Bowling-v5 | - | - | ppo_atari_arc |
| ALE/Boxing-v5 | - | - | ppo_atari_arc |
| ALE/Breakout-v5 | - | - | ppo_atari_lam70_arc |
| ALE/Carnival-v5 | - | - | ppo_atari_lam70_arc |
| ALE/Centipede-v5 | - | - | ppo_atari_lam70_arc |
| ALE/ChopperCommand-v5 | - | - | ppo_atari_arc |
| ALE/CrazyClimber-v5 | - | - | ppo_atari_lam85_arc |
| ALE/Defender-v5 | - | - | ppo_atari_lam70_arc |
| ALE/DemonAttack-v5 | - | - | ppo_atari_lam70_arc |
| ALE/DoubleDunk-v5 | - | - | ppo_atari_arc |
| ALE/ElevatorAction-v5 | - | - | ppo_atari_arc |
| ALE/Enduro-v5 | - | - | ppo_atari_lam85_arc |
| ALE/FishingDerby-v5 | - | - | ppo_atari_lam85_arc |
| ALE/Freeway-v5 | - | - | ppo_atari_arc |
| ALE/Frostbite-v5 | - | - | ppo_atari_arc |
| ALE/Gopher-v5 | - | - | ppo_atari_lam70_arc |
| ALE/Gravitar-v5 | - | - | ppo_atari_arc |
| ALE/Hero-v5 | - | - | ppo_atari_lam85_arc |
| ALE/IceHockey-v5 | - | - | ppo_atari_arc |
| ALE/Jamesbond-v5 | - | - | ppo_atari_arc |
| ALE/JourneyEscape-v5 | - | - | ppo_atari_lam85_arc |
| ALE/Kangaroo-v5 | - | - | ppo_atari_lam85_arc |
| ALE/Krull-v5 | - | - | ppo_atari_arc |
| ALE/KungFuMaster-v5 | - | - | ppo_atari_lam70_arc |
| ALE/MsPacman-v5 | - | - | ppo_atari_lam85_arc |
| ALE/NameThisGame-v5 | - | - | ppo_atari_arc |
| ALE/Phoenix-v5 | - | - | ppo_atari_lam70_arc |
| ALE/Pong-v5 | - | - | ppo_atari_lam85_arc |
| ALE/Pooyan-v5 | - | - | ppo_atari_lam70_arc |
| ALE/Qbert-v5 | - | - | ppo_atari_arc |
| ALE/Riverraid-v5 | - | - | ppo_atari_lam85_arc |
| ALE/RoadRunner-v5 | - | - | ppo_atari_lam85_arc |
| ALE/Robotank-v5 | - | - | ppo_atari_arc |
| ALE/Seaquest-v5 | - | - | ppo_atari_arc |
| ALE/Skiing-v5 | - | - | ppo_atari_arc |
| ALE/Solaris-v5 | - | - | ppo_atari_arc |
| ALE/SpaceInvaders-v5 | - | - | ppo_atari_arc |
| ALE/StarGunner-v5 | - | - | ppo_atari_lam70_arc |
| ALE/Surround-v5 | - | - | ppo_atari_arc |
| ALE/Tennis-v5 | - | - | ppo_atari_lam85_arc |
| ALE/TimePilot-v5 | - | - | ppo_atari_arc |
| ALE/Tutankham-v5 | - | - | ppo_atari_lam85_arc |
| ALE/UpNDown-v5 | - | - | ppo_atari_arc |
| ALE/VideoPinball-v5 | - | - | ppo_atari_lam70_arc |
| ALE/WizardOfWor-v5 | - | - | ppo_atari_arc |
| ALE/YarsRevenge-v5 | - | - | ppo_atari_arc |
| ALE/Zaxxon-v5 | - | - | ppo_atari_arc |

**Skipped** (hard exploration): Adventure, MontezumaRevenge, Pitfall, PrivateEye, Venture

---

## Run Plan

### Phase 1: Classic Control
Quick validation (~minutes each). All algorithms.

### Phase 2: Box2D
Medium runs (~30min each). PPO, DQN, SAC.

### Phase 3: MuJoCo
Long runs (~hours). PPO and SAC, all envs.

### Phase 4: Atari (54 games)
Longest runs (~2h each x 4 sessions). Start with representative games:
- Easy: Breakout, Pong, SpaceInvaders
- Hard: MontezumaRevenge, Solaris, Pitfall
- Medium: Seaquest, Qbert, BeamRider

Validate these 9 games first, then launch remaining 45.
