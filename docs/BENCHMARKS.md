# SLM-Lab Benchmarks

Reproducible deep RL algorithm validation across Gymnasium environments (Classic Control, Box2D, MuJoCo, Atari).

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

All games share one spec file (57 tested, 6 hard exploration skipped). Use `-s env=ENV` to substitute. Runs take ~2-3 hours on GPU.

```bash
source .env && slm-lab run-remote --gpu -s env=ALE/Pong-v5 slm_lab/spec/benchmark_arc/ppo/ppo_atari_arc.yaml ppo_atari_arc train -n pong
```

### Download Results

Trained models and metrics sync to [HuggingFace](https://huggingface.co/datasets/SLM-Lab/benchmark). Pull locally:
```bash
source .env && slm-lab pull SPEC_NAME
slm-lab list  # see available experiments
```

### Benchmark Contribution
To ensure benchmark integrity, follow these steps when adding or updating results:

#### 1. Audit Spec Settings
*   **Before Running**: Ensure `spec.yaml` matches the **Settings** line defined in each benchmark table.
*   **Example**: `max_frame 3e5 | num_envs 4 | max_session 4 | log_frequency 500`
*   **After Pulling**: Verify the downloaded `spec.yaml` matches these rules before using the data.

#### 2. Run Benchmark & Commit Specs
*   **Run**: Execute the benchmark locally or remotely using the commands in [Usage](#usage).
*   **Commit Specs**: Always commit the `spec.yaml` file used for the run to the repo.
*   **Table Entry**: Ensure `BENCHMARKS.md` has an entry with the correct `SPEC_FILE` and `SPEC_NAME`.

#### 3. Record Scores & Plots
*   **Score**: At run completion, extract `total_reward_ma` from logs (`trial_metrics`).
*   **Link**: Add HuggingFace folder link: `[FOLDER](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/FOLDER)`
*   **Pull data**: `source .env && uv run hf download SLM-Lab/benchmark --include "data/FOLDER/*" --local-dir hf_data --repo-type dataset`
*   **Plot**: Generate with folders from table:
    ```bash
    slm-lab plot -t "CartPole-v1" -f ppo_cartpole_2026...,dqn_cartpole_2026...
    ```

### Environment Settings

Standardized settings for fair comparison. The **Settings** line in each result table shows these values.

| Env Category | num_envs | max_frame | log_frequency | grace_period |
|--------------|----------|-----------|---------------|--------------|
| Classic Control | 4 | 2e5-3e5 | 500 | 1e4 |
| Box2D | 8 | 3e5 | 1000 | 5e4 |
| MuJoCo | 16 | 1e6-10e6 | 1e4 | 1e5-1e6 |
| Atari | 16 | 10e6 | 10000 | 5e5 |

### Hyperparameter Search

When algorithm fails to reach target, run search instead of train:

```bash
slm-lab run SPEC_FILE SPEC_NAME search                                        # local
source .env && slm-lab run-remote --gpu SPEC_FILE SPEC_NAME search -n NAME    # remote
```

| Stage | Mode | Config | Purpose |
|-------|------|--------|---------|
| ASHA | `search` | `max_session=1`, `search_scheduler` enabled | Wide exploration with early stopping |
| Multi | `search` | `max_session=4`, NO `search_scheduler` | Robust validation with averaging |
| Validate | `train` | Final spec | Confirmation run |

> Do not use search result in benchmark results - use final validation run with committed spec.

Search budget: ~3-4 trials per dimension (8 trials = 2-3 dims, 16 = 3-4 dims, 20+ = 5+ dims).

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

## Progress

| Phase | Category | Envs | REINFORCE | SARSA | DQN | DDQN+PER | A2C | PPO | SAC | Overall |
|-------|----------|------|-----------|-------|-----|----------|-----|-----|-----|---------|
| 1 | Classic Control | 3 | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ | Done |
| 2 | Box2D | 2 | N/A | N/A | ⚠️ | ✅ | ❌ | ⚠️ | ⚠️ | Done |
| 3 | MuJoCo | 11 | N/A | N/A | N/A | N/A | N/A | ⚠️ | ⚠️ | Done |
| 4 | Atari | 57 | N/A | N/A | N/A | Skip | Done | Done | Done | Done |

**Legend**: ✅ Solved | ⚠️ Close (>80%) | 📊 Acceptable | ❌ Failed | 🔄 In progress/Pending | Skip Not started | N/A Not applicable

---

## Results

### Phase 1: Classic Control

#### 1.1 CartPole-v1

**Docs**: [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) | State: Box(4) | Action: Discrete(2) | Target reward MA > 400

**Settings**: max_frame 2e5 | num_envs 4 | max_session 4 | log_frequency 500

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| REINFORCE | ✅ | 483.31 | [slm_lab/spec/benchmark_arc/reinforce/reinforce_arc.yaml](../slm_lab/spec/benchmark_arc/reinforce/reinforce_arc.yaml) | reinforce_cartpole_arc | [reinforce_cartpole_arc_2026_02_11_135616](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/reinforce_cartpole_arc_2026_02_11_135616) |
| SARSA | ✅ | 430.95 | [slm_lab/spec/benchmark_arc/sarsa/sarsa_arc.yaml](../slm_lab/spec/benchmark_arc/sarsa/sarsa_arc.yaml) | sarsa_boltzmann_cartpole_arc | [sarsa_boltzmann_cartpole_arc_2026_02_11_135616](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sarsa_boltzmann_cartpole_arc_2026_02_11_135616) |
| DQN | ⚠️ | 239.94 | [slm_lab/spec/benchmark_arc/dqn/dqn_classic_arc.yaml](../slm_lab/spec/benchmark_arc/dqn/dqn_classic_arc.yaml) | dqn_boltzmann_cartpole_arc | [dqn_boltzmann_cartpole_arc_2026_02_11_135648](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/dqn_boltzmann_cartpole_arc_2026_02_11_135648) |
| DDQN+PER | ✅ | 451.51 | [slm_lab/spec/benchmark_arc/dqn/dqn_classic_arc.yaml](../slm_lab/spec/benchmark_arc/dqn/dqn_classic_arc.yaml) | ddqn_per_boltzmann_cartpole_arc | [ddqn_per_boltzmann_cartpole_arc_2026_02_11_140518](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ddqn_per_boltzmann_cartpole_arc_2026_02_11_140518) |
| A2C | ✅ | 496.68 | [slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml](../slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml) | a2c_gae_cartpole_arc | [a2c_gae_cartpole_arc_2026_02_11_142531](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_cartpole_arc_2026_02_11_142531) |
| PPO | ✅ | 498.94 | [slm_lab/spec/benchmark_arc/ppo/ppo_classic_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_classic_arc.yaml) | ppo_cartpole_arc | [ppo_cartpole_arc_2026_02_11_144029](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_cartpole_arc_2026_02_11_144029) |
| SAC | ✅ | 406.09 | [slm_lab/spec/benchmark_arc/sac/sac_classic_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_classic_arc.yaml) | sac_cartpole_arc | [sac_cartpole_arc_2026_02_11_144155](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_cartpole_arc_2026_02_11_144155) |

![CartPole-v1](plots/CartPole-v1_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 1.2 Acrobot-v1

**Docs**: [Acrobot](https://gymnasium.farama.org/environments/classic_control/acrobot/) | State: Box(6) | Action: Discrete(3) | Target reward MA > -100

**Settings**: max_frame 3e5 | num_envs 4 | max_session 4 | log_frequency 500

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| DQN | ✅ | -94.17 | [slm_lab/spec/benchmark_arc/dqn/dqn_classic_arc.yaml](../slm_lab/spec/benchmark_arc/dqn/dqn_classic_arc.yaml) | dqn_boltzmann_acrobot_arc | [dqn_boltzmann_acrobot_arc_2026_02_11_144342](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/dqn_boltzmann_acrobot_arc_2026_02_11_144342) |
| DDQN+PER | ✅ | -83.92 | [slm_lab/spec/benchmark_arc/dqn/dqn_classic_arc.yaml](../slm_lab/spec/benchmark_arc/dqn/dqn_classic_arc.yaml) | ddqn_per_acrobot_arc | [ddqn_per_acrobot_arc_2026_02_11_153725](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ddqn_per_acrobot_arc_2026_02_11_153725) |
| A2C | ✅ | -83.99 | [slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml](../slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml) | a2c_gae_acrobot_arc | [a2c_gae_acrobot_arc_2026_02_11_153806](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_acrobot_arc_2026_02_11_153806) |
| PPO | ✅ | -81.28 | [slm_lab/spec/benchmark_arc/ppo/ppo_classic_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_classic_arc.yaml) | ppo_acrobot_arc | [ppo_acrobot_arc_2026_02_11_153758](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_acrobot_arc_2026_02_11_153758) |
| SAC | ✅ | -92.60 | [slm_lab/spec/benchmark_arc/sac/sac_classic_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_classic_arc.yaml) | sac_acrobot_arc | [sac_acrobot_arc_2026_02_11_162211](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_acrobot_arc_2026_02_11_162211) |

![Acrobot-v1](plots/Acrobot-v1_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 1.3 Pendulum-v1

**Docs**: [Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/) | State: Box(3) | Action: Box(1) | Target reward MA > -200

**Settings**: max_frame 3e5 | num_envs 4 | max_session 4 | log_frequency 500

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| A2C | ❌ | -820.74 | [slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml](../slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml) | a2c_gae_pendulum_arc | [a2c_gae_pendulum_arc_2026_02_11_162217](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_pendulum_arc_2026_02_11_162217) |
| PPO | ✅ | -174.87 | [slm_lab/spec/benchmark_arc/ppo/ppo_classic_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_classic_arc.yaml) | ppo_pendulum_arc | [ppo_pendulum_arc_2026_02_11_162156](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_pendulum_arc_2026_02_11_162156) |
| SAC | ✅ | -150.97 | [slm_lab/spec/benchmark_arc/sac/sac_classic_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_classic_arc.yaml) | sac_pendulum_arc | [sac_pendulum_arc_2026_02_11_162240](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_pendulum_arc_2026_02_11_162240) |

![Pendulum-v1](plots/Pendulum-v1_multi_trial_graph_mean_returns_ma_vs_frames.png)

### Phase 2: Box2D

#### 2.1 LunarLander-v3 (Discrete)

**Docs**: [LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) | State: Box(8) | Action: Discrete(4) | Target reward MA > 200

**Settings**: max_frame 3e5 | num_envs 8 | max_session 4 | log_frequency 1000

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| DQN | ⚠️ | 195.21 | [slm_lab/spec/benchmark_arc/dqn/dqn_box2d_arc.yaml](../slm_lab/spec/benchmark_arc/dqn/dqn_box2d_arc.yaml) | dqn_concat_lunar_arc | [dqn_concat_lunar_arc_2026_02_11_201407](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/dqn_concat_lunar_arc_2026_02_11_201407) |
| DDQN+PER | ✅ | 265.90 | [slm_lab/spec/benchmark_arc/dqn/dqn_box2d_arc.yaml](../slm_lab/spec/benchmark_arc/dqn/dqn_box2d_arc.yaml) | ddqn_per_concat_lunar_arc | [ddqn_per_concat_lunar_arc_2026_02_13_105115](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ddqn_per_concat_lunar_arc_2026_02_13_105115) |
| A2C | ❌ | 27.38 | [slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml](../slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml) | a2c_gae_lunar_arc | [a2c_gae_lunar_arc_2026_02_11_224304](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_lunar_arc_2026_02_11_224304) |
| PPO | ⚠️ | 183.30 | [slm_lab/spec/benchmark_arc/ppo/ppo_box2d_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_box2d_arc.yaml) | ppo_lunar_arc | [ppo_lunar_arc_2026_02_11_201303](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_lunar_arc_2026_02_11_201303) |
| SAC | ⚠️ | 106.17 | [slm_lab/spec/benchmark_arc/sac/sac_box2d_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_box2d_arc.yaml) | sac_lunar_arc | [sac_lunar_arc_2026_02_11_201417](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_lunar_arc_2026_02_11_201417) |

![LunarLander-v3 Discrete](plots/LunarLander-v3_Discrete_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 2.2 LunarLander-v3 (Continuous)

**Docs**: [LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) | State: Box(8) | Action: Box(2) | Target reward MA > 200

**Settings**: max_frame 3e5 | num_envs 8 | max_session 4 | log_frequency 1000

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| A2C | ❌ | -76.81 | [slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml](../slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml) | a2c_gae_lunar_continuous_arc | [a2c_gae_lunar_continuous_arc_2026_02_11_224301](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_lunar_continuous_arc_2026_02_11_224301) |
| PPO | ⚠️ | 132.58 | [slm_lab/spec/benchmark_arc/ppo/ppo_box2d_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_box2d_arc.yaml) | ppo_lunar_continuous_arc | [ppo_lunar_continuous_arc_2026_02_11_224229](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_lunar_continuous_arc_2026_02_11_224229) |
| SAC | ⚠️ | 125.00 | [slm_lab/spec/benchmark_arc/sac/sac_box2d_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_box2d_arc.yaml) | sac_lunar_continuous_arc | [sac_lunar_continuous_arc_2026_02_12_222203](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_lunar_continuous_arc_2026_02_12_222203) |

![LunarLander-v3 Continuous](plots/LunarLander-v3_Continuous_multi_trial_graph_mean_returns_ma_vs_frames.png)

### Phase 3: MuJoCo

**Docs**: [MuJoCo environments](https://gymnasium.farama.org/environments/mujoco/) | State/Action: Continuous | Target: Practical baselines (no official "solved" threshold)

**Settings**: max_frame 4e6-10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

**Algorithms**: PPO and SAC. Network: MLP [256,256], orthogonal init. PPO uses tanh activation; SAC uses relu.

**Note on SAC frame budgets**: SAC uses higher update-to-data ratios (more gradient updates per step), making it more sample-efficient but slower per frame than PPO. SAC benchmarks use 1-4M frames (vs PPO's 4-10M) to fit within practical GPU wall-time limits (~6h). Scores may still be improving at cutoff.

**Spec Files** (one file per algorithm, all envs via YAML anchors):
- **PPO**: [ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml)
- **SAC**: [sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml)

**Spec Variants**: Each file has a base config (shared via YAML anchors) with per-env overrides:

| SPEC_NAME | Envs | Key Config |
|-----------|------|------------|
| ppo_mujoco_arc | HalfCheetah, Walker, Humanoid, HumanoidStandup | Base: gamma=0.99, lam=0.95, lr=3e-4 |
| ppo_mujoco_longhorizon_arc | Reacher, Pusher | gamma=0.997, lam=0.97, lr=2e-4, entropy=0.001 |
| ppo_{env}_arc | Ant, Hopper, Swimmer, IP, IDP | Per-env tuned (gamma, lam, lr) |
| sac_mujoco_arc | (generic, use with -s flags) | Base: gamma=0.99, iter=4, lr=3e-4, [256,256] |
| sac_{env}_arc | All 11 envs | Per-env tuned (iter, gamma, lr, net size) |

**Reproduce**: Copy `SPEC_NAME` and `MAX_FRAME` from the table below.

```bash
# PPO: env and max_frame are parameterized via -s flags
source .env && slm-lab run-remote --gpu -s env=ENV -s max_frame=MAX_FRAME \
  slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml SPEC_NAME train -n NAME

# SAC: env and max_frame are hardcoded per spec — no -s flags needed
source .env && slm-lab run-remote --gpu \
  slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml SPEC_NAME train -n NAME
```

| ENV | SPEC_NAME | MAX_FRAME |
|-----|-----------|-----------|
| Ant-v5 | ppo_ant_arc | 10e6 |
| | sac_ant_arc | 2e6 |
| HalfCheetah-v5 | ppo_mujoco_arc | 10e6 |
| | sac_halfcheetah_arc | 4e6 |
| Hopper-v5 | ppo_hopper_arc | 4e6 |
| | sac_hopper_arc | 3e6 |
| Humanoid-v5 | ppo_mujoco_arc | 10e6 |
| | sac_humanoid_arc | 1e6 |
| HumanoidStandup-v5 | ppo_mujoco_arc | 4e6 |
| | sac_humanoid_standup_arc | 1e6 |
| InvertedDoublePendulum-v5 | ppo_inverted_double_pendulum_arc | 10e6 |
| | sac_inverted_double_pendulum_arc | 2e6 |
| InvertedPendulum-v5 | ppo_inverted_pendulum_arc | 4e6 |
| | sac_inverted_pendulum_arc | 2e6 |
| Pusher-v5 | ppo_mujoco_longhorizon_arc | 4e6 |
| | sac_pusher_arc | 1e6 |
| Reacher-v5 | ppo_mujoco_longhorizon_arc | 4e6 |
| | sac_reacher_arc | 1e6 |
| Swimmer-v5 | ppo_swimmer_arc | 4e6 |
| | sac_swimmer_arc | 2e6 |
| Walker2d-v5 | ppo_mujoco_arc | 10e6 |
| | sac_walker2d_arc | 3e6 |

#### 3.1 Ant-v5

**Docs**: [Ant](https://gymnasium.farama.org/environments/mujoco/ant/) | State: Box(105) | Action: Box(8) | Target reward MA > 2000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 2138.28 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_ant_arc | [ppo_ant_arc_ant_2026_02_12_190644](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_ant_arc_ant_2026_02_12_190644) |
| SAC | ✅ | 4942.91 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_ant_arc | [sac_ant_arc_2026_02_11_225529](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_ant_arc_2026_02_11_225529) |

![Ant-v5](plots/Ant-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.2 HalfCheetah-v5

**Docs**: [HalfCheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) | State: Box(17) | Action: Box(6) | Target reward MA > 5000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 6240.68 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_mujoco_arc | [ppo_mujoco_arc_halfcheetah_2026_02_12_195553](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_arc_halfcheetah_2026_02_12_195553) |
| SAC | ✅ | 9815.16 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_halfcheetah_arc | [sac_halfcheetah_4m_i2_arc_2026_02_14_185522](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_halfcheetah_4m_i2_arc_2026_02_14_185522) |

![HalfCheetah-v5](plots/HalfCheetah-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.3 Hopper-v5

**Docs**: [Hopper](https://gymnasium.farama.org/environments/mujoco/hopper/) | State: Box(11) | Action: Box(3) | Target reward MA ~ 2000

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ⚠️ | 1653.74 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_hopper_arc | [ppo_hopper_arc_hopper_2026_02_12_222206](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_hopper_arc_hopper_2026_02_12_222206) |
| SAC | ⚠️ | 1416.52 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_hopper_arc | [sac_hopper_3m_i4_arc_2026_02_14_185434](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_hopper_3m_i4_arc_2026_02_14_185434) |

![Hopper-v5](plots/Hopper-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.4 Humanoid-v5

**Docs**: [Humanoid](https://gymnasium.farama.org/environments/mujoco/humanoid/) | State: Box(348) | Action: Box(17) | Target reward MA > 1000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 2661.26 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_mujoco_arc | [ppo_mujoco_arc_humanoid_2026_02_12_185439](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_arc_humanoid_2026_02_12_185439) |
| SAC | ✅ | 1989.65 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_humanoid_arc | [sac_humanoid_arc_2026_02_12_020016](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_humanoid_arc_2026_02_12_020016) |

![Humanoid-v5](plots/Humanoid-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.5 HumanoidStandup-v5

**Docs**: [HumanoidStandup](https://gymnasium.farama.org/environments/mujoco/humanoid_standup/) | State: Box(348) | Action: Box(17) | Target reward MA > 100000

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 150104.59 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_mujoco_arc | [ppo_mujoco_arc_humanoidstandup_2026_02_12_115050](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_arc_humanoidstandup_2026_02_12_115050) |
| SAC | ✅ | 137357.00 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_humanoid_standup_arc | [sac_humanoid_standup_arc_2026_02_12_225150](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_humanoid_standup_arc_2026_02_12_225150) |

![HumanoidStandup-v5](plots/HumanoidStandup-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.6 InvertedDoublePendulum-v5

**Docs**: [InvertedDoublePendulum](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/) | State: Box(9) | Action: Box(1) | Target reward MA ~8000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 8383.76 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_inverted_double_pendulum_arc | [ppo_inverted_double_pendulum_arc_inverteddoublependulum_2026_02_12_225231](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_inverted_double_pendulum_arc_inverteddoublependulum_2026_02_12_225231) |
| SAC | ✅ | 9032.67 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_inverted_double_pendulum_arc | [sac_inverted_double_pendulum_arc_2026_02_12_025206](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_inverted_double_pendulum_arc_2026_02_12_025206) |

![InvertedDoublePendulum-v5](plots/InvertedDoublePendulum-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.7 InvertedPendulum-v5

**Docs**: [InvertedPendulum](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/) | State: Box(4) | Action: Box(1) | Target reward MA ~1000

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 949.94 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_inverted_pendulum_arc | [ppo_inverted_pendulum_arc_invertedpendulum_2026_02_12_062037](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_inverted_pendulum_arc_invertedpendulum_2026_02_12_062037) |
| SAC | ✅ | 928.43 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_inverted_pendulum_arc | [sac_inverted_pendulum_arc_2026_02_12_225503](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_inverted_pendulum_arc_2026_02_12_225503) |

![InvertedPendulum-v5](plots/InvertedPendulum-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.8 Pusher-v5

**Docs**: [Pusher](https://gymnasium.farama.org/environments/mujoco/pusher/) | State: Box(23) | Action: Box(7) | Target reward MA > -50

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | -49.59 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_mujoco_longhorizon_arc | [ppo_mujoco_longhorizon_arc_pusher_2026_02_12_222228](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_longhorizon_arc_pusher_2026_02_12_222228) |
| SAC | ✅ | -43.00 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_pusher_arc | [sac_pusher_arc_2026_02_12_053603](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_pusher_arc_2026_02_12_053603) |

![Pusher-v5](plots/Pusher-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.9 Reacher-v5

**Docs**: [Reacher](https://gymnasium.farama.org/environments/mujoco/reacher/) | State: Box(10) | Action: Box(2) | Target reward MA > -10

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | -5.03 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_mujoco_longhorizon_arc | [ppo_mujoco_longhorizon_arc_reacher_2026_02_12_115033](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_longhorizon_arc_reacher_2026_02_12_115033) |
| SAC | ✅ | -6.31 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_reacher_arc | [sac_reacher_arc_2026_02_12_055200](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_reacher_arc_2026_02_12_055200) |

![Reacher-v5](plots/Reacher-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.10 Swimmer-v5

**Docs**: [Swimmer](https://gymnasium.farama.org/environments/mujoco/swimmer/) | State: Box(8) | Action: Box(2) | Target reward MA > 200

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 282.44 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_swimmer_arc | [ppo_swimmer_arc_swimmer_2026_02_12_100445](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_swimmer_arc_swimmer_2026_02_12_100445) |
| SAC | ✅ | 301.34 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_swimmer_arc | [sac_swimmer_arc_2026_02_12_054349](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_swimmer_arc_2026_02_12_054349) |

![Swimmer-v5](plots/Swimmer-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.11 Walker2d-v5

**Docs**: [Walker2d](https://gymnasium.farama.org/environments/mujoco/walker2d/) | State: Box(17) | Action: Box(6) | Target reward MA > 3500

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 4378.62 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_mujoco_arc | [ppo_mujoco_arc_walker2d_2026_02_12_190312](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_arc_walker2d_2026_02_12_190312) |
| SAC | ⚠️ | 3123.66 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_walker2d_arc | [sac_walker2d_3m_i4_arc_2026_02_14_185550](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_walker2d_3m_i4_arc_2026_02_14_185550) |

![Walker2d-v5](plots/Walker2d-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

### Phase 4: Atari

**Docs**: [Atari environments](https://ale.farama.org/environments/) | State: Box(84,84,4 after preprocessing) | Action: Discrete(4-18, game-dependent)

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 10000

**Environment**:
- Gymnasium ALE v5 with `life_loss_info=true`
- v5 uses sticky actions (`repeat_action_probability=0.25`) per [Machado et al. (2018)](https://arxiv.org/abs/1709.06009) best practices

**Algorithm Specs** (all use Nature CNN [32,64,64] + 512fc):
- **DDQN+PER**: Skipped - off-policy variants ~6x slower (~230 fps vs ~1500 fps), not cost effective at 10M frames
- **A2C**: [a2c_atari_arc.yaml](../slm_lab/spec/benchmark_arc/a2c/a2c_atari_arc.yaml) - RMSprop (lr=7e-4), training_frequency=32
- **PPO**: [ppo_atari_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_atari_arc.yaml) - AdamW (lr=2.5e-4), minibatch=256, horizon=128, epochs=4, max_frame=10e6
- **SAC**: [sac_atari_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_atari_arc.yaml) - Categorical SAC, AdamW (lr=3e-4), training_iter=3, training_frequency=4, max_frame=2e6

**PPO Lambda Variants** (table shows best result per game):

| SPEC_NAME | Lambda | Best for |
|-----------|--------|----------|
| ppo_atari_arc | 0.95 | Strategic games (default) |
| ppo_atari_lam85_arc | 0.85 | Mixed games |
| ppo_atari_lam70_arc | 0.70 | Action games |

**Reproduce**:
```bash
# A2C (10M frames)
source .env && slm-lab run-remote --gpu -s env=ENV -s max_frame=1e7 \
  slm_lab/spec/benchmark_arc/a2c/a2c_atari_arc.yaml a2c_gae_atari_arc train -n NAME

# PPO (10M frames)
source .env && slm-lab run-remote --gpu -s env=ENV -s max_frame=1e7 \
  slm_lab/spec/benchmark_arc/ppo/ppo_atari_arc.yaml SPEC_NAME train -n NAME

# SAC (2M frames - off-policy, more sample-efficient but slower per frame)
source .env && slm-lab run-remote --gpu -s env=ENV \
  slm_lab/spec/benchmark_arc/sac/sac_atari_arc.yaml sac_atari_arc train -n NAME
```

> **Note**: HF Data links marked "-" indicate runs completed but not yet uploaded to HuggingFace. Scores are extracted from local trial_metrics.

| ENV | Score | SPEC_NAME | HF Data |
|-----|-------|-----------|---------|
| ALE/AirRaid-v5 | 7042.84 | ppo_atari_arc | [ppo_atari_arc_airraid_2026_02_13_124015](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_airraid_2026_02_13_124015) |
| | 1832.54 | sac_atari_arc | [sac_atari_arc_airraid_2026_02_17_104002](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_airraid_2026_02_17_104002) |
| | 5067 | a2c_gae_atari_arc | [a2c_gae_atari_airraid_2026_02_01_082446](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_airraid_2026_02_01_082446) |
| ALE/Alien-v5 | 1789.26 | ppo_atari_arc | [ppo_atari_arc_alien_2026_02_13_124017](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_alien_2026_02_13_124017) |
| | 833.53 | sac_atari_arc | [sac_atari_arc_alien_2026_02_15_200940](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_alien_2026_02_15_200940) |
| | 1488 | a2c_gae_atari_arc | [a2c_gae_atari_alien_2026_02_01_000858](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_alien_2026_02_01_000858) |
| ALE/Amidar-v5 | 584.28 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_amidar_2026_02_13_124155](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_amidar_2026_02_13_124155) |
| | 185.45 | sac_atari_arc | [sac_atari_arc_amidar_2026_02_16_042529](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_amidar_2026_02_16_042529) |
| | 330 | a2c_gae_atari_arc | [a2c_gae_atari_amidar_2026_02_01_082251](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_amidar_2026_02_01_082251) |
| ALE/Assault-v5 | 4448.16 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_assault_2026_02_13_124219](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_assault_2026_02_13_124219) |
| | 1009.42 | sac_atari_arc | [sac_atari_arc_assault_2026_02_16_042532](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_assault_2026_02_16_042532) |
| | 1646 | a2c_gae_atari_arc | [a2c_gae_atari_assault_2026_02_01_082252](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_assault_2026_02_01_082252) |
| ALE/Asterix-v5 | 3235.46 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_asterix_2026_02_13_124329](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_asterix_2026_02_13_124329) |
| | 1504.44 | sac_atari_arc | [sac_atari_arc_asterix_2026_02_16_064430](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_asterix_2026_02_16_064430) |
| | 2712 | a2c_gae_atari_arc | [a2c_gae_atari_asterix_2026_02_01_082315](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_asterix_2026_02_01_082315) |
| ALE/Asteroids-v5 | 1577.92 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_asteroids_2026_02_13_171445](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_asteroids_2026_02_13_171445) |
| | 1203.52 | sac_atari_arc | [sac_atari_arc_asteroids_2026_02_16_051747](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_asteroids_2026_02_16_051747) |
| | 2106 | a2c_gae_atari_arc | [a2c_gae_atari_asteroids_2026_02_01_082328](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_asteroids_2026_02_01_082328) |
| ALE/Atlantis-v5 | 848087.19 | ppo_atari_arc | [ppo_atari_arc_atlantis_2026_02_13_171349](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_atlantis_2026_02_13_171349) |
| | 56787.32 | sac_atari_arc | [sac_atari_arc_atlantis_2026_02_17_105837](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_atlantis_2026_02_17_105837) |
| | 873365 | a2c_gae_atari_arc | [a2c_gae_atari_atlantis_2026_02_01_082330](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_atlantis_2026_02_01_082330) |
| ALE/BankHeist-v5 | 1058.25 | ppo_atari_arc | [ppo_atari_arc_bankheist_2026_02_13_230416](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_bankheist_2026_02_13_230416) |
| | 138.43 | sac_atari_arc | [sac_atari_arc_bankheist_2026_02_17_105306](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_bankheist_2026_02_17_105306) |
| | 1099 | a2c_gae_atari_arc | [a2c_gae_atari_bankheist_2026_02_01_082403](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_bankheist_2026_02_01_082403) |
| ALE/BattleZone-v5 | 27176.78 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_battlezone_2026_02_13_171436](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_battlezone_2026_02_13_171436) |
| | 6906.47 | sac_atari_arc | [sac_atari_arc_battlezone_2026_02_17_112313](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_battlezone_2026_02_17_112313) |
| | 2437 | a2c_gae_atari_arc | [a2c_gae_atari_battlezone_2026_02_01_082425](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_battlezone_2026_02_01_082425) |
| ALE/BeamRider-v5 | 2761.75 | ppo_atari_arc | [ppo_atari_arc_beamrider_2026_02_13_171450](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_beamrider_2026_02_13_171450) |
| | 4061.05 | sac_atari_arc | [sac_atari_arc_beamrider_2026_02_17_110505](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_beamrider_2026_02_17_110505) |
| | 2767 | a2c_gae_atari_arc | [a2c_gae_atari_beamrider_2026_02_01_000921](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_beamrider_2026_02_01_000921) |
| ALE/Berzerk-v5 | 835.46 | ppo_atari_arc | [ppo_atari_arc_berzerk_2026_02_13_171449](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_berzerk_2026_02_13_171449) |
| | 313.87 | sac_atari_arc | [sac_atari_arc_berzerk_2026_02_17_105608](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_berzerk_2026_02_17_105608) |
| | 439 | a2c_gae_atari_arc | [a2c_gae_atari_berzerk_2026_02_01_082540](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_berzerk_2026_02_01_082540) |
| ALE/Bowling-v5 | 45.02 | ppo_atari_arc | [ppo_atari_arc_bowling_2026_02_13_230507](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_bowling_2026_02_13_230507) |
| | 26.55 | sac_atari_arc | [sac_atari_arc_bowling_2026_02_18_101223](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_bowling_2026_02_18_101223) |
| | 23.96 | a2c_gae_atari_arc | [a2c_gae_atari_bowling_2026_02_01_082529](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_bowling_2026_02_01_082529) |
| ALE/Boxing-v5 | 92.18 | ppo_atari_arc | [ppo_atari_arc_boxing_2026_02_13_230504](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_boxing_2026_02_13_230504) |
| | 44.03 | sac_atari_arc | [sac_atari_arc_boxing_2026_02_15_201228](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_boxing_2026_02_15_201228) |
| | 1.80 | a2c_gae_atari_arc | [a2c_gae_atari_boxing_2026_02_01_082539](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_boxing_2026_02_01_082539) |
| ALE/Breakout-v5 | 326.47 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_breakout_2026_02_13_230455](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_breakout_2026_02_13_230455) |
| | 20.23 | sac_atari_arc | [sac_atari_arc_breakout_2026_02_15_201235](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_breakout_2026_02_15_201235) |
| | 273 | a2c_gae_atari_arc | [a2c_gae_atari_breakout_2026_01_31_213610](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_breakout_2026_01_31_213610) |
| ALE/Carnival-v5 | 3912.59 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_carnival_2026_02_13_230438](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_carnival_2026_02_13_230438) |
| | 3501.37 | sac_atari_arc | [sac_atari_arc_carnival_2026_02_17_105834](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_carnival_2026_02_17_105834) |
| | 2170 | a2c_gae_atari_arc | [a2c_gae_atari_carnival_2026_02_01_082726](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_carnival_2026_02_01_082726) |
| ALE/Centipede-v5 | 4780.75 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_centipede_2026_02_13_230434](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_centipede_2026_02_13_230434) |
| | 2255.45 | sac_atari_arc | [sac_atari_arc_centipede_2026_02_18_101425](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_centipede_2026_02_18_101425) |
| | 1382 | a2c_gae_atari_arc | [a2c_gae_atari_centipede_2026_02_01_082643](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_centipede_2026_02_01_082643) |
| ALE/ChopperCommand-v5 | 5391.30 | ppo_atari_arc | [ppo_atari_arc_choppercommand_2026_02_13_230448](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_choppercommand_2026_02_13_230448) |
| | 1036.91 | sac_atari_arc | [sac_atari_arc_choppercommand_2026_02_17_110523](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_choppercommand_2026_02_17_110523) |
| | 2446 | a2c_gae_atari_arc | [a2c_gae_atari_choppercommand_2026_02_01_082626](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_choppercommand_2026_02_01_082626) |
| ALE/CrazyClimber-v5 | 112094.03 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_crazyclimber_2026_02_13_230445](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_crazyclimber_2026_02_13_230445) |
| | 75712.12 | sac_atari_arc | [sac_atari_arc_crazyclimber_2026_02_15_201349](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_crazyclimber_2026_02_15_201349) |
| | 96943 | a2c_gae_atari_arc | [a2c_gae_atari_crazyclimber_2026_02_01_082625](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_crazyclimber_2026_02_01_082625) |
| ALE/Defender-v5 | 47894.69 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_defender_2026_02_14_023317](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_defender_2026_02_14_023317) |
| | 4386.79 | sac_atari_arc | [sac_atari_arc_defender_2026_02_18_101518](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_defender_2026_02_18_101518) |
| | 33149 | a2c_gae_atari_arc | [a2c_gae_atari_defender_2026_02_01_082658](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_defender_2026_02_01_082658) |
| ALE/DemonAttack-v5 | 19370.38 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_demonattack_2026_02_14_023650](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_demonattack_2026_02_14_023650) |
| | 4555.58 | sac_atari_arc | [sac_atari_arc_demonattack_2026_02_18_101610](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_demonattack_2026_02_18_101610) |
| | 2962 | a2c_gae_atari_arc | [a2c_gae_atari_demonattack_2026_02_01_082717](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_demonattack_2026_02_01_082717) |
| ALE/DoubleDunk-v5 | -3.03 | ppo_atari_arc | [ppo_atari_arc_doubledunk_2026_02_14_043639](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_doubledunk_2026_02_14_043639) |
| | -18.65 | sac_atari_arc | [sac_atari_arc_doubledunk_2026_02_17_160707](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_doubledunk_2026_02_17_160707) |
| | -1.69 | a2c_gae_atari_arc | [a2c_gae_atari_doubledunk_2026_02_01_082901](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_doubledunk_2026_02_01_082901) |
| ALE/Enduro-v5 | 986.46 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_enduro_2026_02_11_101739](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_enduro_2026_02_11_101739) |
| | 45.80 | sac_atari_arc | [sac_atari_arc_enduro_2026_02_17_160716](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_enduro_2026_02_17_160716) |
| | 681 | a2c_gae_atari_arc | [a2c_gae_atari_enduro_2026_02_01_001123](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_enduro_2026_02_01_001123) |
| ALE/FishingDerby-v5 | 25.71 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_fishingderby_2026_02_14_024158](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_fishingderby_2026_02_14_024158) |
| | -75.82 | sac_atari_arc | [sac_atari_arc_fishingderby_2026_02_17_160848](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_fishingderby_2026_02_17_160848) |
| | -16.38 | a2c_gae_atari_arc | [a2c_gae_atari_fishingderby_2026_02_01_082906](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_fishingderby_2026_02_01_082906) |
| ALE/Freeway-v5 | 32.42 | ppo_atari_arc | [ppo_atari_arc_freeway_2026_02_14_023359](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_freeway_2026_02_14_023359) |
| | 0.00 | sac_atari_arc | [sac_atari_arc_freeway_2026_02_17_161324](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_freeway_2026_02_17_161324) |
| | 23.13 | a2c_gae_atari_arc | [a2c_gae_atari_freeway_2026_02_01_082931](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_freeway_2026_02_01_082931) |
| ALE/Frostbite-v5 | 284.07 | ppo_atari_arc | [ppo_atari_arc_frostbite_2026_02_14_024247](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_frostbite_2026_02_14_024247) |
| | 355.80 | sac_atari_arc | [sac_atari_arc_frostbite_2026_02_17_160759](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_frostbite_2026_02_17_160759) |
| | 266 | a2c_gae_atari_arc | [a2c_gae_atari_frostbite_2026_02_01_082915](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_frostbite_2026_02_01_082915) |
| ALE/Gopher-v5 | 6500.38 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_gopher_2026_02_14_024237](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_gopher_2026_02_14_024237) |
| | 1608.59 | sac_atari_arc | [sac_atari_arc_gopher_2026_02_17_161047](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_gopher_2026_02_17_161047) |
| | 984 | a2c_gae_atari_arc | [a2c_gae_atari_gopher_2026_02_01_133323](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_gopher_2026_02_01_133323) |
| ALE/Gravitar-v5 | 602.58 | ppo_atari_arc | [ppo_atari_arc_gravitar_2026_02_14_075743](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_gravitar_2026_02_14_075743) |
| | 233.02 | sac_atari_arc | [sac_atari_arc_gravitar_2026_02_17_160858](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_gravitar_2026_02_17_160858) |
| | 270 | a2c_gae_atari_arc | [a2c_gae_atari_gravitar_2026_02_01_133244](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_gravitar_2026_02_01_133244) |
| ALE/Hero-v5 | 22477.89 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_hero_2026_02_15_232615](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_hero_2026_02_15_232615) |
| | 4873.09 | sac_atari_arc | [sac_atari_arc_hero_2026_02_17_161420](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_hero_2026_02_17_161420) |
| | 18680 | a2c_gae_atari_arc | [a2c_gae_atari_hero_2026_02_01_175903](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_hero_2026_02_01_175903) |
| ALE/IceHockey-v5 | -4.05 | ppo_atari_arc | [ppo_atari_arc_icehockey_2026_02_14_231829](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_icehockey_2026_02_14_231829) |
| | -19.78 | sac_atari_arc | [sac_atari_arc_icehockey_2026_02_18_101834](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_icehockey_2026_02_18_101834) |
| | -5.92 | a2c_gae_atari_arc | [a2c_gae_atari_icehockey_2026_02_01_175745](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_icehockey_2026_02_01_175745) |
| ALE/Jamesbond-v5 | 710.98 | ppo_atari_arc | [ppo_atari_arc_jamesbond_2026_02_14_080649](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_jamesbond_2026_02_14_080649) |
| | 328.27 | sac_atari_arc | [sac_atari_arc_jamesbond_2026_02_17_220305](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_jamesbond_2026_02_17_220305) |
| | 460 | a2c_gae_atari_arc | [a2c_gae_atari_jamesbond_2026_02_01_175945](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_jamesbond_2026_02_01_175945) |
| ALE/JourneyEscape-v5 | -1248.98 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_journeyescape_2026_02_14_080656](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_journeyescape_2026_02_14_080656) |
| | -3268.80 | sac_atari_arc | [sac_atari_arc_journeyescape_2026_02_17_215843](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_journeyescape_2026_02_17_215843) |
| | -965 | a2c_gae_atari_arc | [a2c_gae_atari_journeyescape_2026_02_01_084415](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_journeyescape_2026_02_01_084415) |
| ALE/Kangaroo-v5 | 10660.35 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_kangaroo_2026_02_16_030656](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_kangaroo_2026_02_16_030656) |
| | 2990.74 | sac_atari_arc | [sac_atari_arc_kangaroo_2026_02_17_220652](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_kangaroo_2026_02_17_220652) |
| | 322 | a2c_gae_atari_arc | [a2c_gae_atari_kangaroo_2026_02_01_084415](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_kangaroo_2026_02_01_084415) |
| ALE/Krull-v5 | 7874.33 | ppo_atari_arc | [ppo_atari_arc_krull_2026_02_14_080657](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_krull_2026_02_14_080657) |
| | 6630.02 | sac_atari_arc | [sac_atari_arc_krull_2026_02_17_221656](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_krull_2026_02_17_221656) |
| | 7519 | a2c_gae_atari_arc | [a2c_gae_atari_krull_2026_02_01_084420](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_krull_2026_02_01_084420) |
| ALE/KungFuMaster-v5 | 28128.04 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_kungfumaster_2026_02_14_080730](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_kungfumaster_2026_02_14_080730) |
| | 9932.72 | sac_atari_arc | [sac_atari_arc_kungfumaster_2026_02_17_221024](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_kungfumaster_2026_02_17_221024) |
| | 23006 | a2c_gae_atari_arc | [a2c_gae_atari_kungfumaster_2026_02_01_085101](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_kungfumaster_2026_02_01_085101) |
| ALE/MsPacman-v5 | 2330.74 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_mspacman_2026_02_14_102435](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_mspacman_2026_02_14_102435) |
| | 1336.96 | sac_atari_arc | [sac_atari_arc_mspacman_2026_02_17_221523](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_mspacman_2026_02_17_221523) |
| | 2110 | a2c_gae_atari_arc | [a2c_gae_atari_mspacman_2026_02_01_001100](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_mspacman_2026_02_01_001100) |
| ALE/NameThisGame-v5 | 6879.23 | ppo_atari_arc | [ppo_atari_arc_namethisgame_2026_02_14_103319](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_namethisgame_2026_02_14_103319) |
| | 3992.71 | sac_atari_arc | [sac_atari_arc_namethisgame_2026_02_17_220905](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_namethisgame_2026_02_17_220905) |
| | 5412 | a2c_gae_atari_arc | [a2c_gae_atari_namethisgame_2026_02_01_132733](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_namethisgame_2026_02_01_132733) |
| ALE/Phoenix-v5 | 13923.26 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_phoenix_2026_02_14_102636](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_phoenix_2026_02_14_102636) |
| | 3958.46 | sac_atari_arc | [sac_atari_arc_phoenix_2026_02_17_222102](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_phoenix_2026_02_17_222102) |
| | 5635 | a2c_gae_atari_arc | [a2c_gae_atari_phoenix_2026_02_01_085101](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_phoenix_2026_02_01_085101) |
| ALE/Pong-v5 | 16.69 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_pong_2026_02_14_103722](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_pong_2026_02_14_103722) |
| | 10.89 | sac_atari_arc | [sac_atari_arc_pong_2026_02_17_160429](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_pong_2026_02_17_160429) |
| | 10.17 | a2c_gae_atari_arc | [a2c_gae_atari_pong_2026_01_31_213635](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_pong_2026_01_31_213635) |
| ALE/Pooyan-v5 | 5308.66 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_pooyan_2026_02_14_114730](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_pooyan_2026_02_14_114730) |
| | 2530.78 | sac_atari_arc | [sac_atari_arc_pooyan_2026_02_17_220346](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_pooyan_2026_02_17_220346) |
| | 2997 | a2c_gae_atari_arc | [a2c_gae_atari_pooyan_2026_02_01_132748](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_pooyan_2026_02_01_132748) |
| ALE/Qbert-v5 | 15460.48 | ppo_atari_arc | [ppo_atari_arc_qbert_2026_02_14_120409](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_qbert_2026_02_14_120409) |
| | 3331.98 | sac_atari_arc | [sac_atari_arc_qbert_2026_02_17_223117](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_qbert_2026_02_17_223117) |
| | 12619 | a2c_gae_atari_arc | [a2c_gae_atari_qbert_2026_01_31_213720](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_qbert_2026_01_31_213720) |
| ALE/Riverraid-v5 | 9599.75 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_riverraid_2026_02_14_124700](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_riverraid_2026_02_14_124700) |
| | 4744.95 | sac_atari_arc | [sac_atari_arc_riverraid_2026_02_18_014310](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_riverraid_2026_02_18_014310) |
| | 6558 | a2c_gae_atari_arc | [a2c_gae_atari_riverraid_2026_02_01_132507](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_riverraid_2026_02_01_132507) |
| ALE/RoadRunner-v5 | 37980.95 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_roadrunner_2026_02_14_124844](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_roadrunner_2026_02_14_124844) |
| | 25975.39 | sac_atari_arc | [sac_atari_arc_roadrunner_2026_02_18_015052](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_roadrunner_2026_02_18_015052) |
| | 29810 | a2c_gae_atari_arc | [a2c_gae_atari_roadrunner_2026_02_01_132509](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_roadrunner_2026_02_01_132509) |
| ALE/Robotank-v5 | 21.04 | ppo_atari_arc | [ppo_atari_arc_robotank_2026_02_14_124751](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_robotank_2026_02_14_124751) |
| | 9.01 | sac_atari_arc | [sac_atari_arc_robotank_2026_02_18_032313](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_robotank_2026_02_18_032313) |
| | 2.80 | a2c_gae_atari_arc | [a2c_gae_atari_robotank_2026_02_01_132434](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_robotank_2026_02_01_132434) |
| ALE/Seaquest-v5 | 1775.14 | ppo_atari_arc | [ppo_atari_arc_seaquest_2026_02_11_095444](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_seaquest_2026_02_11_095444) |
| | 1565.44 | sac_atari_arc | [sac_atari_arc_seaquest_2026_02_18_020822](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_seaquest_2026_02_18_020822) |
| | 850 | a2c_gae_atari_arc | [a2c_gae_atari_seaquest_2026_02_01_001001](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_seaquest_2026_02_01_001001) |
| ALE/Skiing-v5 | -28217.28 | ppo_atari_arc | [ppo_atari_arc_skiing_2026_02_14_174807](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_skiing_2026_02_14_174807) |
| | -17464.22 | sac_atari_arc | [sac_atari_arc_skiing_2026_02_18_024444](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_skiing_2026_02_18_024444) |
| | -14235 | a2c_gae_atari_arc | [a2c_gae_atari_skiing_2026_02_01_132451](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_skiing_2026_02_01_132451) |
| ALE/Solaris-v5 | 2212.78 | ppo_atari_arc | [ppo_atari_arc_solaris_2026_02_14_124751](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_solaris_2026_02_14_124751) |
| | 1803.74 | sac_atari_arc | [sac_atari_arc_solaris_2026_02_18_031943](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_solaris_2026_02_18_031943) |
| | 2224 | a2c_gae_atari_arc | [a2c_gae_atari_solaris_2026_02_01_212137](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_solaris_2026_02_01_212137) |
| ALE/SpaceInvaders-v5 | 892.49 | ppo_atari_arc | [ppo_atari_arc_spaceinvaders_2026_02_14_131114](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_spaceinvaders_2026_02_14_131114) |
| | 507.33 | sac_atari_arc | [sac_atari_arc_spaceinvaders_2026_02_18_033139](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_spaceinvaders_2026_02_18_033139) |
| | 784 | a2c_gae_atari_arc | [a2c_gae_atari_spaceinvaders_2026_02_01_000950](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_spaceinvaders_2026_02_01_000950) |
| ALE/StarGunner-v5 | 49328.73 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_stargunner_2026_02_14_131149](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_stargunner_2026_02_14_131149) |
| | 4295.97 | sac_atari_arc | [sac_atari_arc_stargunner_2026_02_18_033151](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_stargunner_2026_02_18_033151) |
| | 8665 | a2c_gae_atari_arc | [a2c_gae_atari_stargunner_2026_02_01_132406](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_stargunner_2026_02_01_132406) |
| ALE/Surround-v5 | -4.47 | ppo_atari_arc | [ppo_atari_arc_surround_2026_02_14_132941](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_surround_2026_02_14_132941) |
| | -9.87 | sac_atari_arc | [sac_atari_arc_surround_2026_02_18_034423](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_surround_2026_02_18_034423) |
| | -9.72 | a2c_gae_atari_arc | [a2c_gae_atari_surround_2026_02_01_132215](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_surround_2026_02_01_132215) |
| ALE/Tennis-v5 | -12.27 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_tennis_2026_02_14_173639](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_tennis_2026_02_14_173639) |
| | -397.44 | sac_atari_arc | [sac_atari_arc_tennis_2026_02_18_032540](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_tennis_2026_02_18_032540) |
| | -2873 | a2c_gae_atari_arc | [a2c_gae_atari_tennis_2026_02_01_175829](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_tennis_2026_02_01_175829) |
| ALE/TimePilot-v5 | 4432.73 | ppo_atari_arc | [ppo_atari_arc_timepilot_2026_02_14_173642](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_timepilot_2026_02_14_173642) |
| | 3164.97 | sac_atari_arc | [sac_atari_arc_timepilot_2026_02_18_102038](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_timepilot_2026_02_18_102038) |
| | 3376 | a2c_gae_atari_arc | [a2c_gae_atari_timepilot_2026_02_01_175930](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_timepilot_2026_02_01_175930) |
| ALE/Tutankham-v5 | 210.87 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_tutankham_2026_02_14_173722](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_tutankham_2026_02_14_173722) |
| | 147.25 | sac_atari_arc | [sac_atari_arc_tutankham_2026_02_18_102729](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_tutankham_2026_02_18_102729) |
| | 167 | a2c_gae_atari_arc | [a2c_gae_atari_tutankham_2026_02_01_132347](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_tutankham_2026_02_01_132347) |
| ALE/UpNDown-v5 | 147168.80 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_upndown_2026_02_15_232448](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_upndown_2026_02_15_232448) |
| | 3351.89 | sac_atari_arc | [sac_atari_arc_upndown_2026_02_18_135442](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_upndown_2026_02_18_135442) |
| | 57099 | a2c_gae_atari_arc | [a2c_gae_atari_upndown_2026_02_01_132435](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_upndown_2026_02_01_132435) |
| ALE/VideoPinball-v5 | 38370.30 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_videopinball_2026_02_14_173728](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_videopinball_2026_02_14_173728) |
| | 21088.68 | sac_atari_arc | [sac_atari_arc_videopinball_2026_02_18_141245](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_videopinball_2026_02_18_141245) |
| | 25310 | a2c_gae_atari_arc | [a2c_gae_atari_videopinball_2026_02_01_083457](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_videopinball_2026_02_01_083457) |
| ALE/WizardOfWor-v5 | 6100.42 | ppo_atari_arc | [ppo_atari_arc_wizardofwor_2026_02_14_173945](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_wizardofwor_2026_02_14_173945) |
| | 1241.92 | sac_atari_arc | [sac_atari_arc_wizardofwor_2026_02_18_140750](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_wizardofwor_2026_02_18_140750) |
| | 2682 | a2c_gae_atari_arc | [a2c_gae_atari_wizardofwor_2026_02_01_132449](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_wizardofwor_2026_02_01_132449) |
| ALE/YarsRevenge-v5 | 12873.91 | ppo_atari_arc | [ppo_atari_arc_yarsrevenge_2026_02_14_174019](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_yarsrevenge_2026_02_14_174019) |
| | 13710.18 | sac_atari_arc | [sac_atari_arc_yarsrevenge_2026_02_18_134921](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_yarsrevenge_2026_02_18_134921) |
| | 24371 | a2c_gae_atari_arc | [a2c_gae_atari_yarsrevenge_2026_02_01_132224](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_yarsrevenge_2026_02_01_132224) |
| ALE/Zaxxon-v5 | 9523.49 | ppo_atari_arc | [ppo_atari_arc_zaxxon_2026_02_14_174806](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_zaxxon_2026_02_14_174806) |
| | 3205.98 | sac_atari_arc | [sac_atari_arc_zaxxon_2026_02_18_135502](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_zaxxon_2026_02_18_135502) |
| | 29.46 | a2c_gae_atari_arc | [a2c_gae_atari_zaxxon_2026_02_01_131758](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_zaxxon_2026_02_01_131758) |


**Training Curves** (A2C vs PPO vs SAC):

| | | |
|:---:|:---:|:---:|
| ![AirRaid](plots/AirRaid-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Alien](plots/Alien-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Amidar](plots/Amidar-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Assault](plots/Assault-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Asterix](plots/Asterix-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Asteroids](plots/Asteroids-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Atlantis](plots/Atlantis-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![BankHeist](plots/BankHeist-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![BattleZone](plots/BattleZone-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![BeamRider](plots/BeamRider-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Berzerk](plots/Berzerk-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Bowling](plots/Bowling-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Boxing](plots/Boxing-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Breakout](plots/Breakout-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Carnival](plots/Carnival-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Centipede](plots/Centipede-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![ChopperCommand](plots/ChopperCommand-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![CrazyClimber](plots/CrazyClimber-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Defender](plots/Defender-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![DemonAttack](plots/DemonAttack-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![DoubleDunk](plots/DoubleDunk-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Enduro](plots/Enduro-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![FishingDerby](plots/FishingDerby-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Freeway](plots/Freeway-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Frostbite](plots/Frostbite-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Gopher](plots/Gopher-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Gravitar](plots/Gravitar-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Hero](plots/Hero-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![IceHockey](plots/IceHockey-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Jamesbond](plots/Jamesbond-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![JourneyEscape](plots/JourneyEscape-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Kangaroo](plots/Kangaroo-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Krull](plots/Krull-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![KungFuMaster](plots/KungFuMaster-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![MsPacman](plots/MsPacman-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![NameThisGame](plots/NameThisGame-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Phoenix](plots/Phoenix-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Pong](plots/Pong-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Pooyan](plots/Pooyan-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Qbert](plots/Qbert-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Riverraid](plots/Riverraid-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![RoadRunner](plots/RoadRunner-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Robotank](plots/Robotank-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Seaquest](plots/Seaquest-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Skiing](plots/Skiing-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Solaris](plots/Solaris-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![SpaceInvaders](plots/SpaceInvaders-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![StarGunner](plots/StarGunner-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Surround](plots/Surround-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Tennis](plots/Tennis-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![TimePilot](plots/TimePilot-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Tutankham](plots/Tutankham-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![UpNDown](plots/UpNDown-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![VideoPinball](plots/VideoPinball-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![WizardOfWor](plots/WizardOfWor-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![YarsRevenge](plots/YarsRevenge-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Zaxxon](plots/Zaxxon-v5_multi_trial_graph_mean_returns_ma_vs_frames.png) |

**Skipped**: Adventure, MontezumaRevenge, Pitfall, PrivateEye, Venture (hard exploration), ElevatorAction (deprecated env)

<details>
<summary><b>PPO Lambda Comparison</b> (click to expand)</summary>

| ENV | ppo_atari_arc | ppo_atari_lam85_arc | ppo_atari_lam70_arc |
|-----|---------------|---------------------|---------------------|
| ALE/AirRaid-v5 | **7042.84** | - | - |
| ALE/Alien-v5 | **1789.26** | - | - |
| ALE/Amidar-v5 | - | **584.28** | - |
| ALE/Assault-v5 | - | **4448.16** | - |
| ALE/Asterix-v5 | - | **3235.46** | - |
| ALE/Asteroids-v5 | - | **1577.92** | - |
| ALE/Atlantis-v5 | **848087.19** | - | - |
| ALE/BankHeist-v5 | **1058.25** | - | - |
| ALE/BattleZone-v5 | - | **27176.78** | - |
| ALE/BeamRider-v5 | **2761.75** | - | - |
| ALE/Berzerk-v5 | **835.46** | - | - |
| ALE/Bowling-v5 | **45.02** | - | - |
| ALE/Boxing-v5 | **92.18** | - | - |
| ALE/Breakout-v5 | - | - | **326.47** |
| ALE/Carnival-v5 | - | - | **3912.59** |
| ALE/Centipede-v5 | - | - | **4780.75** |
| ALE/ChopperCommand-v5 | **5391.30** | - | - |
| ALE/CrazyClimber-v5 | - | **112094.03** | - |
| ALE/Defender-v5 | - | - | **47894.69** |
| ALE/DemonAttack-v5 | - | - | **19370.38** |
| ALE/DoubleDunk-v5 | **-3.03** | - | - |
| ALE/Enduro-v5 | - | **986.46** | - |
| ALE/FishingDerby-v5 | - | **25.71** | - |
| ALE/Freeway-v5 | **32.42** | - | - |
| ALE/Frostbite-v5 | **284.07** | - | - |
| ALE/Gopher-v5 | - | - | **6500.38** |
| ALE/Gravitar-v5 | **602.58** | - | - |
| ALE/Hero-v5 | - | **22477.89** | - |
| ALE/IceHockey-v5 | **-4.05** | - | - |
| ALE/Jamesbond-v5 | **710.98** | - | - |
| ALE/JourneyEscape-v5 | - | **-1248.98** | - |
| ALE/Kangaroo-v5 | - | - | **10660.35** |
| ALE/Krull-v5 | **7874.33** | - | - |
| ALE/KungFuMaster-v5 | - | - | **28128.04** |
| ALE/MsPacman-v5 | - | **2330.74** | - |
| ALE/NameThisGame-v5 | **6879.23** | - | - |
| ALE/Phoenix-v5 | - | - | **13923.26** |
| ALE/Pong-v5 | - | **16.69** | - |
| ALE/Pooyan-v5 | - | - | **5308.66** |
| ALE/Qbert-v5 | **15460.48** | - | - |
| ALE/Riverraid-v5 | - | **9599.75** | - |
| ALE/RoadRunner-v5 | - | **37980.95** | - |
| ALE/Robotank-v5 | **21.04** | - | - |
| ALE/Seaquest-v5 | **1775.14** | - | - |
| ALE/Skiing-v5 | **-28217.28** | - | - |
| ALE/Solaris-v5 | **2212.78** | - | - |
| ALE/SpaceInvaders-v5 | **892.49** | - | - |
| ALE/StarGunner-v5 | - | - | **49328.73** |
| ALE/Surround-v5 | **-4.47** | - | - |
| ALE/Tennis-v5 | - | **-12.27** | - |
| ALE/TimePilot-v5 | **4432.73** | - | - |
| ALE/Tutankham-v5 | - | **210.87** | - |
| ALE/UpNDown-v5 | - | **147168.80** | - |
| ALE/VideoPinball-v5 | - | - | **38370.30** |
| ALE/WizardOfWor-v5 | **6100.42** | - | - |
| ALE/YarsRevenge-v5 | **12873.91** | - | - |
| ALE/Zaxxon-v5 | **9523.49** | - | - |

**Legend**: **Bold** = Best score | - = Not tested

</details>

