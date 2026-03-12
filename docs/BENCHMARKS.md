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

| Phase | Category | Envs | REINFORCE | SARSA | DQN | DDQN+PER | A2C | PPO | SAC | CrossQ | Overall |
|-------|----------|------|-----------|-------|-----|----------|-----|-----|-----|--------|---------|
| 1 | Classic Control | 3 | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ | ⚠️ | Done |
| 2 | Box2D | 2 | N/A | N/A | ⚠️ | ✅ | ❌ | ⚠️ | ⚠️ | ⚠️ | Done |
| 3 | MuJoCo | 11 | N/A | N/A | N/A | N/A | N/A | ⚠️ | ⚠️ | ⚠️ | Done |
| 4 | Atari | 57 | N/A | N/A | N/A | Skip | Done | Done | Done | ❌ | Done |
| 5 | Playground | 54 | N/A | N/A | N/A | N/A | N/A | 🔄 | 🔄 | N/A | In progress |

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
| CrossQ | ⚠️ | 334.59 | [slm_lab/spec/benchmark/crossq/crossq_classic.yaml](../slm_lab/spec/benchmark/crossq/crossq_classic.yaml) | crossq_cartpole | [crossq_cartpole_2026_03_02_100434](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_cartpole_2026_03_02_100434) |

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
| CrossQ | ✅ | -103.13 | [slm_lab/spec/benchmark/crossq/crossq_classic.yaml](../slm_lab/spec/benchmark/crossq/crossq_classic.yaml) | crossq_acrobot | [crossq_acrobot_2026_02_23_153622](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_acrobot_2026_02_23_153622) |

![Acrobot-v1](plots/Acrobot-v1_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 1.3 Pendulum-v1

**Docs**: [Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/) | State: Box(3) | Action: Box(1) | Target reward MA > -200

**Settings**: max_frame 3e5 | num_envs 4 | max_session 4 | log_frequency 500

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| A2C | ❌ | -820.74 | [slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml](../slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml) | a2c_gae_pendulum_arc | [a2c_gae_pendulum_arc_2026_02_11_162217](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_pendulum_arc_2026_02_11_162217) |
| PPO | ✅ | -174.87 | [slm_lab/spec/benchmark_arc/ppo/ppo_classic_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_classic_arc.yaml) | ppo_pendulum_arc | [ppo_pendulum_arc_2026_02_11_162156](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_pendulum_arc_2026_02_11_162156) |
| SAC | ✅ | -150.97 | [slm_lab/spec/benchmark_arc/sac/sac_classic_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_classic_arc.yaml) | sac_pendulum_arc | [sac_pendulum_arc_2026_02_11_162240](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_pendulum_arc_2026_02_11_162240) |
| CrossQ | ✅ | -145.66 | [slm_lab/spec/benchmark/crossq/crossq_classic.yaml](../slm_lab/spec/benchmark/crossq/crossq_classic.yaml) | crossq_pendulum | [crossq_pendulum_2026_02_28_130648](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_pendulum_2026_02_28_130648) |

![Pendulum-v1](plots/Pendulum-v1_multi_trial_graph_mean_returns_ma_vs_frames.png)

### Phase 2: Box2D

#### 2.1 LunarLander-v3

**Docs**: [LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) | State: Box(8) | Action: Discrete(4) | Target reward MA > 200

**Settings**: max_frame 3e5 | num_envs 8 | max_session 4 | log_frequency 1000

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| DQN | ⚠️ | 195.21 | [slm_lab/spec/benchmark_arc/dqn/dqn_box2d_arc.yaml](../slm_lab/spec/benchmark_arc/dqn/dqn_box2d_arc.yaml) | dqn_concat_lunar_arc | [dqn_concat_lunar_arc_2026_02_11_201407](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/dqn_concat_lunar_arc_2026_02_11_201407) |
| DDQN+PER | ✅ | 265.90 | [slm_lab/spec/benchmark_arc/dqn/dqn_box2d_arc.yaml](../slm_lab/spec/benchmark_arc/dqn/dqn_box2d_arc.yaml) | ddqn_per_concat_lunar_arc | [ddqn_per_concat_lunar_arc_2026_02_13_105115](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ddqn_per_concat_lunar_arc_2026_02_13_105115) |
| A2C | ❌ | 27.38 | [slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml](../slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml) | a2c_gae_lunar_arc | [a2c_gae_lunar_arc_2026_02_11_224304](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_lunar_arc_2026_02_11_224304) |
| PPO | ⚠️ | 183.30 | [slm_lab/spec/benchmark_arc/ppo/ppo_box2d_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_box2d_arc.yaml) | ppo_lunar_arc | [ppo_lunar_arc_2026_02_11_201303](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_lunar_arc_2026_02_11_201303) |
| SAC | ⚠️ | 106.17 | [slm_lab/spec/benchmark_arc/sac/sac_box2d_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_box2d_arc.yaml) | sac_lunar_arc | [sac_lunar_arc_2026_02_11_201417](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_lunar_arc_2026_02_11_201417) |
| CrossQ | ❌ | 139.21 | [slm_lab/spec/benchmark/crossq/crossq_box2d.yaml](../slm_lab/spec/benchmark/crossq/crossq_box2d.yaml) | crossq_lunar | [crossq_lunar_2026_02_28_130733](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_lunar_2026_02_28_130733) |

![LunarLander-v3](plots/LunarLander-v3_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 2.2 LunarLanderContinuous-v3

**Docs**: [LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) | State: Box(8) | Action: Box(2) | Target reward MA > 200

**Settings**: max_frame 3e5 | num_envs 8 | max_session 4 | log_frequency 1000

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| A2C | ❌ | -76.81 | [slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml](../slm_lab/spec/benchmark_arc/a2c/a2c_classic_arc.yaml) | a2c_gae_lunar_continuous_arc | [a2c_gae_lunar_continuous_arc_2026_02_11_224301](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_lunar_continuous_arc_2026_02_11_224301) |
| PPO | ⚠️ | 132.58 | [slm_lab/spec/benchmark_arc/ppo/ppo_box2d_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_box2d_arc.yaml) | ppo_lunar_continuous_arc | [ppo_lunar_continuous_arc_2026_02_11_224229](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_lunar_continuous_arc_2026_02_11_224229) |
| SAC | ⚠️ | 125.00 | [slm_lab/spec/benchmark_arc/sac/sac_box2d_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_box2d_arc.yaml) | sac_lunar_continuous_arc | [sac_lunar_continuous_arc_2026_02_12_222203](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_lunar_continuous_arc_2026_02_12_222203) |
| CrossQ | ✅ | 268.91 | [slm_lab/spec/benchmark/crossq/crossq_box2d.yaml](../slm_lab/spec/benchmark/crossq/crossq_box2d.yaml) | crossq_lunar_continuous | [crossq_lunar_continuous_2026_03_01_140517](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_lunar_continuous_2026_03_01_140517) |

![LunarLanderContinuous-v3](plots/LunarLanderContinuous-v3_multi_trial_graph_mean_returns_ma_vs_frames.png)

### Phase 3: MuJoCo

**Docs**: [MuJoCo environments](https://gymnasium.farama.org/environments/mujoco/) | State/Action: Continuous | Target: Practical baselines (no official "solved" threshold)

**Settings**: max_frame 4e6-10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

**Algorithms**: PPO, SAC, and CrossQ. Network: MLP [256,256], orthogonal init. PPO uses tanh activation; SAC and CrossQ use relu. CrossQ uses Batch Renormalization in critics (no target networks).

**Note on SAC/CrossQ frame budgets**: SAC uses higher update-to-data ratios (more gradient updates per step), making it more sample-efficient but slower per frame than PPO. SAC benchmarks use 1-4M frames (vs PPO's 4-10M) to fit within practical GPU wall-time limits (~6h). CrossQ uses UTD=1 (like PPO) but eliminates target network overhead, achieving ~700 fps — its frame budgets (3-7.5M) reflect this speed advantage. Scores may still be improving at cutoff.

**Spec Files** (one file per algorithm, all envs via YAML anchors):
- **PPO**: [ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml)
- **SAC**: [sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml)
- **CrossQ**: [crossq_mujoco.yaml](../slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml)

**Spec Variants**: Each file has a base config (shared via YAML anchors) with per-env overrides:

| SPEC_NAME | Envs | Key Config |
|-----------|------|------------|
| ppo_mujoco_arc | HalfCheetah, Walker, Humanoid, HumanoidStandup | Base: gamma=0.99, lam=0.95, lr=3e-4 |
| ppo_mujoco_longhorizon_arc | Reacher, Pusher | gamma=0.997, lam=0.97, lr=2e-4, entropy=0.001 |
| ppo_{env}_arc | Ant, Hopper, Swimmer, IP, IDP | Per-env tuned (gamma, lam, lr) |
| sac_mujoco_arc | (generic, use with -s flags) | Base: gamma=0.99, iter=4, lr=3e-4, [256,256] |
| sac_{env}_arc | All 11 envs | Per-env tuned (iter, gamma, lr, net size) |
| crossq_mujoco | (generic base) | Base: gamma=0.99, iter=1, lr=1e-3, policy_delay=3 |
| crossq_{env} | All 11 envs | Per-env tuned (critic width, actor LN, iter) |

**Reproduce**: Copy `SPEC_NAME` and `MAX_FRAME` from the table below.

```bash
# PPO: env and max_frame are parameterized via -s flags
source .env && slm-lab run-remote --gpu -s env=ENV -s max_frame=MAX_FRAME \
  slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml SPEC_NAME train -n NAME

# SAC: env and max_frame are hardcoded per spec — no -s flags needed
source .env && slm-lab run-remote --gpu \
  slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml SPEC_NAME train -n NAME

# CrossQ: env and max_frame are hardcoded per spec — no -s flags needed
source .env && slm-lab run-remote --gpu \
  slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml SPEC_NAME train -n NAME
```

| ENV | SPEC_NAME | MAX_FRAME |
|-----|-----------|-----------|
| Ant-v5 | ppo_ant_arc | 10e6 |
| | sac_ant_arc | 2e6 |
| | crossq_ant | 3e6 |
| HalfCheetah-v5 | ppo_mujoco_arc | 10e6 |
| | sac_halfcheetah_arc | 4e6 |
| | crossq_halfcheetah | 4e6 |
| Hopper-v5 | ppo_hopper_arc | 4e6 |
| | sac_hopper_arc | 3e6 |
| | crossq_hopper | 3e6 |
| Humanoid-v5 | ppo_mujoco_arc | 10e6 |
| | sac_humanoid_arc | 1e6 |
| | crossq_humanoid | 2e6 |
| HumanoidStandup-v5 | ppo_mujoco_arc | 4e6 |
| | sac_humanoid_standup_arc | 1e6 |
| | crossq_humanoid_standup | 2e6 |
| InvertedDoublePendulum-v5 | ppo_inverted_double_pendulum_arc | 10e6 |
| | sac_inverted_double_pendulum_arc | 2e6 |
| | crossq_inverted_double_pendulum | 2e6 |
| InvertedPendulum-v5 | ppo_inverted_pendulum_arc | 4e6 |
| | sac_inverted_pendulum_arc | 2e6 |
| | crossq_inverted_pendulum | 7e6 |
| Pusher-v5 | ppo_mujoco_longhorizon_arc | 4e6 |
| | sac_pusher_arc | 1e6 |
| | crossq_pusher | 2e6 |
| Reacher-v5 | ppo_mujoco_longhorizon_arc | 4e6 |
| | sac_reacher_arc | 1e6 |
| | crossq_reacher | 2e6 |
| Swimmer-v5 | ppo_swimmer_arc | 4e6 |
| | sac_swimmer_arc | 2e6 |
| | crossq_swimmer | 3e6 |
| Walker2d-v5 | ppo_mujoco_arc | 10e6 |
| | sac_walker2d_arc | 3e6 |
| | crossq_walker2d | 7e6 |

#### 3.1 Ant-v5

**Docs**: [Ant](https://gymnasium.farama.org/environments/mujoco/ant/) | State: Box(105) | Action: Box(8) | Target reward MA > 2000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 2138.28 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_ant_arc | [ppo_ant_arc_ant_2026_02_12_190644](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_ant_arc_ant_2026_02_12_190644) |
| SAC | ✅ | 4942.91 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_ant_arc | [sac_ant_arc_2026_02_11_225529](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_ant_arc_2026_02_11_225529) |
| CrossQ | ✅ | 4517.00 | [slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml](../slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml) | crossq_ant | [crossq_ant_2026_03_01_102428](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_ant_2026_03_01_102428) |

![Ant-v5](plots/Ant-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.2 HalfCheetah-v5

**Docs**: [HalfCheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) | State: Box(17) | Action: Box(6) | Target reward MA > 5000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 6240.68 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_mujoco_arc | [ppo_mujoco_arc_halfcheetah_2026_02_12_195553](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_arc_halfcheetah_2026_02_12_195553) |
| SAC | ✅ | 9815.16 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_halfcheetah_arc | [sac_halfcheetah_4m_i2_arc_2026_02_14_185522](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_halfcheetah_4m_i2_arc_2026_02_14_185522) |
| CrossQ | ✅ | 8616.52 | [slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml](../slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml) | crossq_halfcheetah | [crossq_halfcheetah_2026_03_01_101317](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_halfcheetah_2026_03_01_101317) |

![HalfCheetah-v5](plots/HalfCheetah-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.3 Hopper-v5

**Docs**: [Hopper](https://gymnasium.farama.org/environments/mujoco/hopper/) | State: Box(11) | Action: Box(3) | Target reward MA ~ 2000

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ⚠️ | 1653.74 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_hopper_arc | [ppo_hopper_arc_hopper_2026_02_12_222206](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_hopper_arc_hopper_2026_02_12_222206) |
| SAC | ⚠️ | 1416.52 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_hopper_arc | [sac_hopper_3m_i4_arc_2026_02_14_185434](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_hopper_3m_i4_arc_2026_02_14_185434) |
| CrossQ | ⚠️ | 1168.53 | [slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml](../slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml) | crossq_hopper | [crossq_hopper_2026_02_21_101148](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_hopper_2026_02_21_101148) |

![Hopper-v5](plots/Hopper-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.4 Humanoid-v5

**Docs**: [Humanoid](https://gymnasium.farama.org/environments/mujoco/humanoid/) | State: Box(348) | Action: Box(17) | Target reward MA > 1000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 2661.26 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_mujoco_arc | [ppo_mujoco_arc_humanoid_2026_02_12_185439](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_arc_humanoid_2026_02_12_185439) |
| SAC | ✅ | 1989.65 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_humanoid_arc | [sac_humanoid_arc_2026_02_12_020016](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_humanoid_arc_2026_02_12_020016) |
| CrossQ | ✅ | 1755.29 | [slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml](../slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml) | crossq_humanoid | [crossq_humanoid_2026_03_01_165208](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_humanoid_2026_03_01_165208) |

![Humanoid-v5](plots/Humanoid-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.5 HumanoidStandup-v5

**Docs**: [HumanoidStandup](https://gymnasium.farama.org/environments/mujoco/humanoid_standup/) | State: Box(348) | Action: Box(17) | Target reward MA > 100000

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 150104.59 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_mujoco_arc | [ppo_mujoco_arc_humanoidstandup_2026_02_12_115050](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_arc_humanoidstandup_2026_02_12_115050) |
| SAC | ✅ | 137357.00 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_humanoid_standup_arc | [sac_humanoid_standup_arc_2026_02_12_225150](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_humanoid_standup_arc_2026_02_12_225150) |
| CrossQ | ✅ | 150912.66 | [slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml](../slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml) | crossq_humanoid_standup | [crossq_humanoid_standup_2026_02_28_184305](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_humanoid_standup_2026_02_28_184305) |

![HumanoidStandup-v5](plots/HumanoidStandup-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.6 InvertedDoublePendulum-v5

**Docs**: [InvertedDoublePendulum](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/) | State: Box(9) | Action: Box(1) | Target reward MA ~8000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 8383.76 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_inverted_double_pendulum_arc | [ppo_inverted_double_pendulum_arc_inverteddoublependulum_2026_02_12_225231](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_inverted_double_pendulum_arc_inverteddoublependulum_2026_02_12_225231) |
| SAC | ✅ | 9032.67 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_inverted_double_pendulum_arc | [sac_inverted_double_pendulum_arc_2026_02_12_025206](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_inverted_double_pendulum_arc_2026_02_12_025206) |
| CrossQ | ✅ | 8027.38 | [slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml](../slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml) | crossq_inverted_double_pendulum | [crossq_inverted_double_pendulum_2026_03_01_101354](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_inverted_double_pendulum_2026_03_01_101354) |

![InvertedDoublePendulum-v5](plots/InvertedDoublePendulum-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.7 InvertedPendulum-v5

**Docs**: [InvertedPendulum](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/) | State: Box(4) | Action: Box(1) | Target reward MA ~1000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 949.94 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_inverted_pendulum_arc | [ppo_inverted_pendulum_arc_invertedpendulum_2026_02_12_062037](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_inverted_pendulum_arc_invertedpendulum_2026_02_12_062037) |
| SAC | ✅ | 928.43 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_inverted_pendulum_arc | [sac_inverted_pendulum_arc_2026_02_12_225503](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_inverted_pendulum_arc_2026_02_12_225503) |
| CrossQ | ⚠️ | 877.83 | [slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml](../slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml) | crossq_inverted_pendulum | [crossq_inverted_pendulum_2026_02_28_184348](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_inverted_pendulum_2026_02_28_184348) |

![InvertedPendulum-v5](plots/InvertedPendulum-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.8 Pusher-v5

**Docs**: [Pusher](https://gymnasium.farama.org/environments/mujoco/pusher/) | State: Box(23) | Action: Box(7) | Target reward MA > -50

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | -49.59 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_mujoco_longhorizon_arc | [ppo_mujoco_longhorizon_arc_pusher_2026_02_12_222228](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_longhorizon_arc_pusher_2026_02_12_222228) |
| SAC | ✅ | -43.00 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_pusher_arc | [sac_pusher_arc_2026_02_12_053603](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_pusher_arc_2026_02_12_053603) |
| CrossQ | ✅ | -37.08 | [slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml](../slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml) | crossq_pusher | [crossq_pusher_2026_02_21_134637](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_pusher_2026_02_21_134637) |

![Pusher-v5](plots/Pusher-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.9 Reacher-v5

**Docs**: [Reacher](https://gymnasium.farama.org/environments/mujoco/reacher/) | State: Box(10) | Action: Box(2) | Target reward MA > -10

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | -5.03 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_mujoco_longhorizon_arc | [ppo_mujoco_longhorizon_arc_reacher_2026_02_12_115033](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_longhorizon_arc_reacher_2026_02_12_115033) |
| SAC | ✅ | -6.31 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_reacher_arc | [sac_reacher_arc_2026_02_12_055200](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_reacher_arc_2026_02_12_055200) |
| CrossQ | ✅ | -5.65 | [slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml](../slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml) | crossq_reacher | [crossq_reacher_2026_02_28_184304](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_reacher_2026_02_28_184304) |

![Reacher-v5](plots/Reacher-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.10 Swimmer-v5

**Docs**: [Swimmer](https://gymnasium.farama.org/environments/mujoco/swimmer/) | State: Box(8) | Action: Box(2) | Target reward MA > 200

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 282.44 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_swimmer_arc | [ppo_swimmer_arc_swimmer_2026_02_12_100445](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_swimmer_arc_swimmer_2026_02_12_100445) |
| SAC | ✅ | 301.34 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_swimmer_arc | [sac_swimmer_arc_2026_02_12_054349](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_swimmer_arc_2026_02_12_054349) |
| CrossQ | ✅ | 221.12 | [slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml](../slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml) | crossq_swimmer | [crossq_swimmer_2026_02_21_184204](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_swimmer_2026_02_21_184204) |

![Swimmer-v5](plots/Swimmer-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.11 Walker2d-v5

**Docs**: [Walker2d](https://gymnasium.farama.org/environments/mujoco/walker2d/) | State: Box(17) | Action: Box(6) | Target reward MA > 3500

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Data |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 4378.62 | [slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_mujoco_arc.yaml) | ppo_mujoco_arc | [ppo_mujoco_arc_walker2d_2026_02_12_190312](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_arc_walker2d_2026_02_12_190312) |
| SAC | ⚠️ | 3123.66 | [slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml](../slm_lab/spec/benchmark_arc/sac/sac_mujoco_arc.yaml) | sac_walker2d_arc | [sac_walker2d_3m_i4_arc_2026_02_14_185550](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_walker2d_3m_i4_arc_2026_02_14_185550) |
| CrossQ | ✅ | 4389.62 | [slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml](../slm_lab/spec/benchmark/crossq/crossq_mujoco.yaml) | crossq_walker2d | [crossq_walker2d_2026_02_28_184343](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_walker2d_2026_02_28_184343) |

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
- **CrossQ**: [crossq_atari.yaml](../slm_lab/spec/benchmark/crossq/crossq_atari.yaml) - Categorical CrossQ, AdamW (lr=1e-3), training_iter=3, training_frequency=4, max_frame=2e6 (experimental — limited results on 6 games)

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

# CrossQ (2M frames - experimental, limited games tested)
source .env && slm-lab run-remote --gpu -s env=ENV \
  slm_lab/spec/benchmark/crossq/crossq_atari.yaml crossq_atari train -n NAME
```

> **Note**: HF Data links marked "-" indicate runs completed but not yet uploaded to HuggingFace. Scores are extracted from local trial_metrics.

| ENV | MA | SPEC_NAME | HF Data |
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
| | ❌ 4.40 | crossq_atari | [crossq_atari_breakout_2026_02_25_030241](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_atari_breakout_2026_02_25_030241) |
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
| | ❌ 327.79 | crossq_atari | [crossq_atari_mspacman_2026_02_23_171317](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_atari_mspacman_2026_02_23_171317) |
| ALE/NameThisGame-v5 | 6879.23 | ppo_atari_arc | [ppo_atari_arc_namethisgame_2026_02_14_103319](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_namethisgame_2026_02_14_103319) |
| | 3992.71 | sac_atari_arc | [sac_atari_arc_namethisgame_2026_02_17_220905](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_namethisgame_2026_02_17_220905) |
| | 5412 | a2c_gae_atari_arc | [a2c_gae_atari_namethisgame_2026_02_01_132733](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_namethisgame_2026_02_01_132733) |
| ALE/Phoenix-v5 | 13923.26 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_phoenix_2026_02_14_102636](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_phoenix_2026_02_14_102636) |
| | 3958.46 | sac_atari_arc | [sac_atari_arc_phoenix_2026_02_17_222102](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_phoenix_2026_02_17_222102) |
| | 5635 | a2c_gae_atari_arc | [a2c_gae_atari_phoenix_2026_02_01_085101](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_phoenix_2026_02_01_085101) |
| ALE/Pong-v5 | 16.69 | ppo_atari_lam85_arc | [ppo_atari_lam85_arc_pong_2026_02_14_103722](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_arc_pong_2026_02_14_103722) |
| | 10.89 | sac_atari_arc | [sac_atari_arc_pong_2026_02_17_160429](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_pong_2026_02_17_160429) |
| | 10.17 | a2c_gae_atari_arc | [a2c_gae_atari_pong_2026_01_31_213635](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_pong_2026_01_31_213635) |
| | ❌ -20.59 | crossq_atari | [crossq_atari_pong_2026_02_23_171158](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_atari_pong_2026_02_23_171158) |
| ALE/Pooyan-v5 | 5308.66 | ppo_atari_lam70_arc | [ppo_atari_lam70_arc_pooyan_2026_02_14_114730](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_arc_pooyan_2026_02_14_114730) |
| | 2530.78 | sac_atari_arc | [sac_atari_arc_pooyan_2026_02_17_220346](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_pooyan_2026_02_17_220346) |
| | 2997 | a2c_gae_atari_arc | [a2c_gae_atari_pooyan_2026_02_01_132748](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_pooyan_2026_02_01_132748) |
| ALE/Qbert-v5 | 15460.48 | ppo_atari_arc | [ppo_atari_arc_qbert_2026_02_14_120409](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_qbert_2026_02_14_120409) |
| | 3331.98 | sac_atari_arc | [sac_atari_arc_qbert_2026_02_17_223117](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_qbert_2026_02_17_223117) |
| | 12619 | a2c_gae_atari_arc | [a2c_gae_atari_qbert_2026_01_31_213720](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_qbert_2026_01_31_213720) |
| | ❌ 3189.73 | crossq_atari | [crossq_atari_qbert_2026_02_25_030458](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_atari_qbert_2026_02_25_030458) |
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
| | ❌ 234.63 | crossq_atari | [crossq_atari_seaquest_2026_02_25_030441](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_atari_seaquest_2026_02_25_030441) |
| ALE/Skiing-v5 | -28217.28 | ppo_atari_arc | [ppo_atari_arc_skiing_2026_02_14_174807](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_skiing_2026_02_14_174807) |
| | -17464.22 | sac_atari_arc | [sac_atari_arc_skiing_2026_02_18_024444](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_skiing_2026_02_18_024444) |
| | -14235 | a2c_gae_atari_arc | [a2c_gae_atari_skiing_2026_02_01_132451](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_skiing_2026_02_01_132451) |
| ALE/Solaris-v5 | 2212.78 | ppo_atari_arc | [ppo_atari_arc_solaris_2026_02_14_124751](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_solaris_2026_02_14_124751) |
| | 1803.74 | sac_atari_arc | [sac_atari_arc_solaris_2026_02_18_031943](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_solaris_2026_02_18_031943) |
| | 2224 | a2c_gae_atari_arc | [a2c_gae_atari_solaris_2026_02_01_212137](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_solaris_2026_02_01_212137) |
| ALE/SpaceInvaders-v5 | 892.49 | ppo_atari_arc | [ppo_atari_arc_spaceinvaders_2026_02_14_131114](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_arc_spaceinvaders_2026_02_14_131114) |
| | 507.33 | sac_atari_arc | [sac_atari_arc_spaceinvaders_2026_02_18_033139](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_arc_spaceinvaders_2026_02_18_033139) |
| | 784 | a2c_gae_atari_arc | [a2c_gae_atari_spaceinvaders_2026_02_01_000950](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_spaceinvaders_2026_02_01_000950) |
| | ❌ 404.50 | crossq_atari | [crossq_atari_spaceinvaders_2026_02_25_030410](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/crossq_atari_spaceinvaders_2026_02_25_030410) |
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

---

### Phase 5: MuJoCo Playground (JAX/MJX GPU-Accelerated)

**Docs**: [MuJoCo Playground](https://google-deepmind.github.io/mujoco_playground/) | State/Action: Continuous | Target: Research-grade baselines (no official solved threshold)

**Settings**: max_session 4 | log_frequency 100000 — see sub-phase sections for max_frame and num_envs

**Hardware**: RunPod RTX A4500 (20GB) / A5000 (24GB) — MJWarp (Warp CUDA kernels) + DLPack zero-copy to PyTorch

**Install**: `uv sync --group playground` (includes JAX + warp-lang + jax[cuda12])

**Backend**: MJWarp (`impl='warp'`) — JAX dispatches Warp CUDA kernels for physics, DLPack zero-copy transfers to PyTorch.

**Algorithms**: PPO, SAC, and CrossQ.

**Spec Files** (one file per algorithm, all envs via `-s env=` flag):
- **PPO**: [ppo_playground.yaml](../slm_lab/spec/benchmark_arc/ppo/ppo_playground.yaml)
- **SAC**: [sac_playground.yaml](../slm_lab/spec/benchmark_arc/sac/sac_playground.yaml)
- **CrossQ**: [crossq_playground.yaml](../slm_lab/spec/benchmark_arc/crossq/crossq_playground.yaml)

**Spec Variants**:

**PPO** (ppo_playground.yaml):

| SPEC_NAME | num_envs | time_horizon | batch_size | Notes |
|-----------|----------|--------------|------------|-------|
| ppo_playground | 1024 | 128 | 131K | DM Control (gamma=0.995, 16 epochs) |
| ppo_playground_loco | 512 | 256 | 131K | Locomotion/Manipulation (gamma=0.97, 4 epochs) |

**SAC** (sac_playground.yaml):

| SPEC_NAME | num_envs | UTD | Notes |
|-----------|----------|-----|-------|
| sac_playground | 256 | 0.016 | Standard — most envs |
| sac_playground_hard | 16 | 1.0 | High UTD — HopperHop, Acrobot*, PendulumSwingup |

**CrossQ** (crossq_playground.yaml):

| SPEC_NAME | num_envs | critics | Notes |
|-----------|----------|---------|-------|
| crossq_playground | 16 | [512,512]+BRN | Standard — most envs |
| crossq_playground_vhard | 16 | [1024,1024]+BRN | Heavy envs — Humanoid*, CheetahRun |

#### Frame Budget Math

MJWarp GPU throughput scales roughly linearly with `num_envs` (GPU parallelism). All physics environments run in parallel on CUDA via Warp kernels; more envs = more parallel work = higher throughput until GPU saturates.

**Formula**: `max_frame = observed_fps × 5.5h × 3600` (5.5h budget; dstack kills at 6h with **zero data** — always leave 30min margin). Always check fps after 5-10 min on first run of any env.

**Confirmed throughput** (A5000 24GB, 1024 envs PPO): ~10K–15K fps for DM Control — 100M frames in ~2–3h.

**Per-category defaults** (conservative, verify on first run):

| Category | Spec | num_envs | Default max_frame | Observed FPS (A5000) |
|----------|------|----------|-------------------|----------------------|
| DM Control (PPO) | ppo_playground | 1024 | 100M | ~10K–15K fps → 100M in ~2–3h |
| Locomotion (PPO) | ppo_playground_loco | 512 | 100M | ~5K–8K fps → 100M in ~3.5–5.5h |
| Manipulation (PPO) | ppo_playground_loco | 512 | 100M | ~3K–5K fps → 100M in ~5–9h; verify fps first |
| SAC standard | sac_playground | 256 | 20M | ~1500fps → 20M in ~3.7h |
| SAC hard / CrossQ | sac_playground_hard / crossq_playground | 16 | 2M | ~60–500fps; gradient-bound |
| Rough terrain loco | ppo_playground_loco | 512 | 10M | ~500–1500fps; lower due to complex physics |

**Reference throughput** (MuJoCo Playground paper, PPO on A100 at 2048–8192 envs): Cartpole ~720K sps | Cheetah ~435K sps | Walker ~140K sps | Humanoid ~92K sps. SLM-Lab at 1024 envs on A5000 achieves ~10K–15K fps for DM Control (confirmed), which is ~2–5% of reference steps/sec but sufficient to reach 100M frames in 2–3h.

#### Autonomous Benchmark Guidelines

**Frame target**: 100M frames per env for PPO (all phases). SAC standard: 20M. SAC hard / CrossQ at 16 envs are gradient-bound — 2M is acceptable.

**Wall-time budget**: 5.5h budget per run (dstack kills at 6h with **zero data** — no trial_metrics, no HF upload). Always calculate: `max_frame = observed_fps × 5.5h × 3600` and stop any run projected to exceed this.

**FPS calibration**: On first run of any new env, check fps after 5-10 min (`dstack logs NAME --since 10m | grep trial_metrics`). If projected wall clock exceeds 5.5h, stop immediately and relaunch with reduced max_frame.

**Spec selection**: DM Control envs use `ppo_playground` (1024 envs). Locomotion and Manipulation envs use `ppo_playground_loco` (512 envs).

**normalize_obs warning**: DM Control envs have bounded observations — `normalize_obs=true` (the playground spec default) may cause NaN rewards. If NaN is observed in training logs, override with `-s normalize_obs=false`.

**Target (ref) scores**: From official mujoco_playground training plots (2048 envs DM Control / 8192 envs Loco+Manip, 100M steps). Our runs use 1024/512 envs — use as directional targets, not hard thresholds.

**Run order**: Submit fastest algorithms first — PPO (high num_envs, ~2000+ fps) finishes in minutes, then SAC standard (256 envs), then SAC hard / CrossQ (16 envs, gradient-bound, slowest).

**Iteration**: If score is far below Target (ref), consider: (1) increasing max_frame if fps allows, (2) trying a different spec variant (e.g., `ppo_playground_loco` instead of `ppo_playground`), (3) hyperparameter search if the algorithm fundamentally stalls.

**Reproduce** (`-s env=ENV -s max_frame=N`):

```bash
# PPO DM Control — 1024 envs, 100M frames
source .env && uv run slm-lab run-remote --gpu \
  slm_lab/spec/benchmark_arc/ppo/ppo_playground.yaml ppo_playground train \
  -s env=playground/CartpoleBalance -s max_frame=100000000 -n NAME

# PPO Locomotion/Manipulation — 512 envs, 100M frames
source .env && uv run slm-lab run-remote --gpu \
  slm_lab/spec/benchmark_arc/ppo/ppo_playground.yaml ppo_playground_loco train \
  -s env=playground/Go1Getup -s max_frame=100000000 -n NAME

# SAC standard — 256 envs, 20M frames
source .env && uv run slm-lab run-remote --gpu \
  slm_lab/spec/benchmark_arc/sac/sac_playground.yaml sac_playground train \
  -s env=playground/CheetahRun -s max_frame=20000000 -n NAME

# SAC hard — 16 envs, 2M frames (gradient-bound)
source .env && uv run slm-lab run-remote --gpu \
  slm_lab/spec/benchmark_arc/sac/sac_playground.yaml sac_playground_hard train \
  -s env=playground/HopperHop -s max_frame=2000000 -n NAME

# CrossQ — 16 envs, 2M frames (gradient-bound)
source .env && uv run slm-lab run-remote --gpu \
  slm_lab/spec/benchmark_arc/crossq/crossq_playground.yaml crossq_playground train \
  -s env=playground/WalkerRun -s max_frame=2000000 -n NAME
```

#### Phase 5.1: DM Control Suite (25 envs)

**Settings**: max_frame 100M | num_envs 1024 | max_session 4 | log_frequency 100000

**Target (ref)**: scores from mujoco_playground official runs (2048 envs, 100M steps) — use as directional targets.

| ENV | Algorithm | Status | MA | SPEC_NAME | HF Data | Target (ref) | FPS | Frames | Wall Clock |
|-----|-----------|--------|-----|-----------|---------|--------------|-----|--------|------------|
| playground/AcrobotSwingup | PPO | ❌ | 20.29 | ppo_playground | [ppo_playground_acrobotswingup_2026_03_11_142836](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_acrobotswingup_2026_03_11_142836) | 220 | 12730 | 100M | 2h 11m |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| | SAC | 🔄 | - | sac_playground_hard | - | | - | - | - |
| playground/AcrobotSwingupSparse | PPO | ❌ | 0.35 | ppo_playground | [ppo_playground_acrobotswingupsparse_2026_03_11_160346](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_acrobotswingupsparse_2026_03_11_160346) | 15 | 15610 | 100M | 1h 47m |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| | SAC | 🔄 | - | sac_playground_hard | - | | - | - | - |
| playground/BallInCup | PPO | ✅ | 718.66 | ppo_playground | [ppo_playground_ballincup_2026_03_11_124251](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_ballincup_2026_03_11_124251) | ~0 (ref fails) | 15140 | 100M | 1h 50m |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| | SAC | 🔄 | - | sac_playground | - | | - | - | - |
| playground/CartpoleBalance | PPO | ⚠️ | 876.30 | ppo_playground | [ppo_playground_cartpolebalance_2026_03_11_095201](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_cartpolebalance_2026_03_11_095201) | 950 | 12130 | 100M | 2h 17m |
| | SAC | 🔄 | - | sac_playground | - | | - | - | - |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| playground/CartpoleBalanceSparse | PPO | ✅ | 835.74 | ppo_playground | [ppo_playground_cartpolebalancesparse_2026_03_11_142826](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_cartpolebalancesparse_2026_03_11_142826) | 700 | 12040 | 100M | 2h 18m |
| | SAC | 🔄 | - | sac_playground | - | | - | - | - |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| playground/CartpoleSwingup | PPO | ⚠️ | 636.71 | ppo_playground | [ppo_playground_cartpoleswingup_2026_03_11_100907](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_cartpoleswingup_2026_03_11_100907) | 800 | 15050 | 100M | 1h 50m |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| | SAC | 🔄 | - | sac_playground | - | | - | - | - |
| playground/CartpoleSwingupSparse | PPO | ❌ | 27.54 | ppo_playground | [ppo_playground_cartpoleswingupsparse_2026_03_11_142924](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_cartpoleswingupsparse_2026_03_11_142924) | 425 | 11200 | 100M | 2h 29m |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| | SAC | 🔄 | - | sac_playground_hard | - | | - | - | - |
| playground/CheetahRun | PPO | ⚠️ | 662.97 | ppo_playground | [ppo_playground_v2_cheetahrun_2026_03_11_182514](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_v2_cheetahrun_2026_03_11_182514) | 850 | 13990 | 100M | 1h 59m |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| | SAC | 🔄 | - | sac_playground | - | | - | - | - |
| playground/FingerSpin | PPO | ⚠️ | 500.95 | ppo_playground | [ppo_playground_v2_fingerspin_2026_03_11_182614](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_v2_fingerspin_2026_03_11_182614) | 600 | 15920 | 100M | 1h 45m |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| | SAC | 🔄 | - | sac_playground | - | | - | - | - |
| playground/FingerTurnEasy | PPO | ❌ | 598.75 | ppo_playground | [ppo_playground_fingerturneasy_2026_03_11_210311](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_fingerturneasy_2026_03_11_210311) | 950 | 12090 | 100M | 2h 18m |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| | SAC | 🔄 | - | sac_playground | - | | - | - | - |
| playground/FingerTurnHard | PPO | ❌ | 553.35 | ppo_playground | [ppo_playground_fingerturnhard_2026_03_11_213004](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_fingerturnhard_2026_03_11_213004) | 950 | 12320 | 100M | 2h 15m |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| | SAC | 🔄 | - | sac_playground | - | | - | - | - |
| playground/FishSwim | PPO | ❌ | 550.04 | ppo_playground | [ppo_playground_fishswim_2026_03_11_205020](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_fishswim_2026_03_11_205020) | 650 | 9650 | 100M | 2h 53m |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| | SAC | 🔄 | - | sac_playground | - | | - | - | - |
| playground/HopperHop | PPO | 🔄 | - | ppo_playground | - | ~2 | - | - | - |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| | SAC | 🔄 | - | sac_playground_hard | - | | - | - | - |
| playground/HopperStand | PPO | ✅ | 202.54 | ppo_playground_loco | [ppo_playground_loco_hopperstand_2026_03_11_213050](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_loco_hopperstand_2026_03_11_213050) | ~70 | 21000 | 100M | 1h 19m |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| | SAC | 🔄 | - | sac_playground | - | | - | - | - |
| playground/HumanoidRun | PPO | ❌ | 7 | ppo_playground | [ppo_playground_humanoidrun_2026_03_11_210531](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_humanoidrun_2026_03_11_210531) | 130 | 9300 | 100M | 2h 59m |
| | CrossQ | 🔄 | - | crossq_playground_vhard | - | | - | - | - |
| | SAC | 🔄 | - | sac_playground | - | | - | - | - |
| playground/HumanoidStand | PPO | ❌ | 30.04 | ppo_playground_loco | [ppo_playground_loco_humanoidstand_2026_03_11_223955](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_loco_humanoidstand_2026_03_11_223955) | 700 | 14960 | 100M | 1h 55m |
| | CrossQ | 🔄 | - | crossq_playground_vhard | - | | - | - | - |
| | SAC | 🔄 | - | sac_playground | - | | - | - | - |
| playground/HumanoidWalk | PPO | ❌ | 29.89 | ppo_playground | [ppo_playground_humanoidwalk_2026_03_11_205019](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_humanoidwalk_2026_03_11_205019) | 500 | 9400 | 100M | 2h 57m |
| | CrossQ | 🔄 | - | crossq_playground_vhard | - | | - | - | - |
| | SAC | 🔄 | - | sac_playground | - | | - | - | - |
| playground/PendulumSwingup | PPO | ✅ | 395.31 | ppo_playground | [ppo_playground_pendulumswingup_2026_03_11_142901](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_pendulumswingup_2026_03_11_142901) | ~70 | 12080 | 100M | 2h 18m |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| | SAC | 🔄 | - | sac_playground | - | | - | - | - |
| playground/PointMass | PPO | ⚠️ | 801.57 | ppo_playground | [ppo_playground_pointmass_2026_03_11_100801](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_pointmass_2026_03_11_100801) | 900 | 13920 | 100M | 1h 59m |
| | SAC | 🔄 | - | sac_playground | - | | - | - | - |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| playground/ReacherEasy | PPO | ✅ | 964.20 | ppo_playground | [ppo_playground_reachereasy_2026_03_11_121032](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_reachereasy_2026_03_11_121032) | 950 | 14020 | 100M | 1h 59m |
| | SAC | 🔄 | - | sac_playground | - | | - | - | - |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| playground/ReacherHard | PPO | ✅ | 956.05 | ppo_playground | [ppo_playground_reacherhard_2026_03_11_142826](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_reacherhard_2026_03_11_142826) | 950 | 11520 | 100M | 2h 25m |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| | SAC | 🔄 | - | sac_playground | - | | - | - | - |
| playground/SwimmerSwimmer6 | PPO | ❌ | 161.07 | ppo_playground | [ppo_playground_swimmerswimmer6_2026_03_11_152737](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_swimmerswimmer6_2026_03_11_152737) | 560 | 11310 | 50M | 1h 14m |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| | SAC | 🔄 | - | sac_playground | - | | - | - | - |
| playground/WalkerRun | PPO | ⚠️ | 482.17 | ppo_playground | [ppo_playground_v2_walkerrun_2026_03_11_182608](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_v2_walkerrun_2026_03_11_182608) | 560 | 11720 | 100M | 2h 22m |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| | SAC | 🔄 | - | sac_playground | - | | - | - | - |
| playground/WalkerStand | PPO | ⚠️ | 781.26 | ppo_playground | [ppo_playground_walkerstand_2026_03_11_100507](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_walkerstand_2026_03_11_100507) | 1000 | 10170 | 100M | 2h 44m |
| | PPO | 🔄 | - | ppo_playground_loco | - | | - | - | - |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| | SAC | 🔄 | - | sac_playground | - | | - | - | - |
| playground/WalkerWalk | PPO | ❌ | 613.47 | ppo_playground | [ppo_playground_walkerwalk_2026_03_11_095325](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/ppo_playground_walkerwalk_2026_03_11_095325) | 960 | 10730 | 100M | 2h 35m |
| | CrossQ | 🔄 | - | crossq_playground | - | | - | - | - |
| | SAC | 🔄 | - | sac_playground | - | | - | - | - |

| | | |
|---|---|---|
| ![AcrobotSwingup](plots/AcrobotSwingup_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![AcrobotSwingupSparse](plots/AcrobotSwingupSparse_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![BallInCup](plots/BallInCup_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![CartpoleBalance](plots/CartpoleBalance_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![CartpoleBalanceSparse](plots/CartpoleBalanceSparse_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![CartpoleSwingup](plots/CartpoleSwingup_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![CartpoleSwingupSparse](plots/CartpoleSwingupSparse_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![CheetahRun](plots/CheetahRun_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![FingerSpin](plots/FingerSpin_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![FingerTurnEasy](plots/FingerTurnEasy_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![FingerTurnHard](plots/FingerTurnHard_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![FishSwim](plots/FishSwim_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![HopperHop](plots/HopperHop_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![HopperStand](plots/HopperStand_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![HumanoidRun](plots/HumanoidRun_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![HumanoidStand](plots/HumanoidStand_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![HumanoidWalk](plots/HumanoidWalk_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![PendulumSwingup](plots/PendulumSwingup_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![PointMass](plots/PointMass_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![ReacherEasy](plots/ReacherEasy_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![ReacherHard](plots/ReacherHard_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![SwimmerSwimmer6](plots/SwimmerSwimmer6_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![WalkerRun](plots/WalkerRun_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![WalkerStand](plots/WalkerStand_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![WalkerWalk](plots/WalkerWalk_multi_trial_graph_mean_returns_ma_vs_frames.png) | | |

#### Phase 5.2: Locomotion Robots (19 envs)

**Settings**: max_frame 100M | num_envs 512 | max_session 4 | log_frequency 100000

**Target (ref)**: scores from mujoco_playground official runs (8192 envs, 100M steps) — use as directional targets.

| ENV | Algorithm | Status | MA | SPEC_NAME | HF Data | Target (ref) | FPS | Frames | Wall Clock |
|-----|-----------|--------|-----|-----------|---------|--------------|-----|--------|------------|
| playground/ApolloJoystickFlatTerrain | PPO | 🔄 | - | ppo_playground_loco | - | 15 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/BarkourJoystick | PPO | 🔄 | - | ppo_playground_loco | - | 35 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/BerkeleyHumanoidJoystickFlatTerrain | PPO | 🔄 | - | ppo_playground_loco | - | 20 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/BerkeleyHumanoidJoystickRoughTerrain | PPO | 🔄 | - | ppo_playground_loco | - | 15 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/G1JoystickFlatTerrain | PPO | 🔄 | - | ppo_playground_loco | - | 10 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/G1JoystickRoughTerrain | PPO | 🔄 | - | ppo_playground_loco | - | 5 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/Go1Footstand | PPO | 🔄 | - | ppo_playground_loco | - | 15 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/Go1Getup | PPO | 🔄 | - | ppo_playground_loco | - | 5 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/Go1Handstand | PPO | 🔄 | - | ppo_playground_loco | - | 15 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/Go1JoystickFlatTerrain | PPO | 🔄 | - | ppo_playground_loco | - | 25 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/Go1JoystickRoughTerrain | PPO | 🔄 | - | ppo_playground_loco | - | 20 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/H1InplaceGaitTracking | PPO | 🔄 | - | ppo_playground_loco | - | 10 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/H1JoystickGaitTracking | PPO | 🔄 | - | ppo_playground_loco | - | 30 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/Op3Joystick | PPO | 🔄 | - | ppo_playground_loco | - | 20 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/SpotFlatTerrainJoystick | PPO | 🔄 | - | ppo_playground_loco | - | 30 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/SpotGetup | PPO | 🔄 | - | ppo_playground_loco | - | 20 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/SpotJoystickGaitTracking | PPO | 🔄 | - | ppo_playground_loco | - | 35 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/T1JoystickFlatTerrain | PPO | 🔄 | - | ppo_playground_loco | - | 25 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/T1JoystickRoughTerrain | PPO | 🔄 | - | ppo_playground_loco | - | 10 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |

| | | |
|---|---|---|
| ![ApolloJoystickFlatTerrain](plots/ApolloJoystickFlatTerrain_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![BarkourJoystick](plots/BarkourJoystick_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![BerkeleyHumanoidJoystickFlatTerrain](plots/BerkeleyHumanoidJoystickFlatTerrain_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![G1JoystickFlatTerrain](plots/G1JoystickFlatTerrain_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Go1Footstand](plots/Go1Footstand_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Go1Handstand](plots/Go1Handstand_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![H1InplaceGaitTracking](plots/H1InplaceGaitTracking_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![H1JoystickGaitTracking](plots/H1JoystickGaitTracking_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Op3Joystick](plots/Op3Joystick_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![SpotFlatTerrainJoystick](plots/SpotFlatTerrainJoystick_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![SpotGetup](plots/SpotGetup_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![SpotJoystickGaitTracking](plots/SpotJoystickGaitTracking_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Go1JoystickFlatTerrain](plots/Go1JoystickFlatTerrain_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Go1JoystickRoughTerrain](plots/Go1JoystickRoughTerrain_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![T1JoystickFlatTerrain](plots/T1JoystickFlatTerrain_multi_trial_graph_mean_returns_ma_vs_frames.png) |

#### Phase 5.3: Manipulation (10 envs)

**Settings**: max_frame 100M | num_envs 512 | max_session 4 | log_frequency 100000

**Target (ref)**: scores from mujoco_playground official runs (8192 envs, 100M steps) — use as directional targets.

| ENV | Algorithm | Status | MA | SPEC_NAME | HF Data | Target (ref) | FPS | Frames | Wall Clock |
|-----|-----------|--------|-----|-----------|---------|--------------|-----|--------|------------|
| playground/AeroCubeRotateZAxis | PPO | 🔄 | - | ppo_playground_loco | - | — | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/AlohaHandOver | PPO | 🔄 | - | ppo_playground_loco | - | 5 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/AlohaSinglePegInsertion | PPO | 🔄 | - | ppo_playground_loco | - | 300 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/LeapCubeReorient | PPO | 🔄 | - | ppo_playground_loco | - | 200 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/LeapCubeRotateZAxis | PPO | 🔄 | - | ppo_playground_loco | - | 15 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/PandaOpenCabinet | PPO | 🔄 | - | ppo_playground_loco | - | 1250 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/PandaPickCube | PPO | 🔄 | - | ppo_playground_loco | - | 1300 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/PandaPickCubeCartesian | PPO | 🔄 | - | ppo_playground_loco | - | 10 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/PandaPickCubeOrientation | PPO | 🔄 | - | ppo_playground_loco | - | 1100 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| playground/PandaRobotiqPushCube | PPO | 🔄 | - | ppo_playground_loco | - | 20 | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |
| | - | 🔄 | - | - | - | | - | - | - |

| | | |
|---|---|---|
| ![AeroCubeRotateZAxis](plots/AeroCubeRotateZAxis_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![AlohaHandOver](plots/AlohaHandOver_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![AlohaSinglePegInsertion](plots/AlohaSinglePegInsertion_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![LeapCubeReorient](plots/LeapCubeReorient_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![LeapCubeRotateZAxis](plots/LeapCubeRotateZAxis_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![PandaOpenCabinet](plots/PandaOpenCabinet_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![PandaPickCube](plots/PandaPickCube_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![PandaPickCubeOrientation](plots/PandaPickCubeOrientation_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![PandaRobotiqPushCube](plots/PandaRobotiqPushCube_multi_trial_graph_mean_returns_ma_vs_frames.png) |

