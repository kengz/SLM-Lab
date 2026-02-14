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

All games share one spec file (54 tested, 5 hard exploration skipped). Use `-s env=ENV` to substitute. Runs take ~2-3 hours on GPU.

```bash
source .env && slm-lab run-remote --gpu -s env=ALE/Pong-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train -n pong
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
*   **Before Running**: Ensure `spec.json` matches the **Settings** line defined in each benchmark table.
*   **Example**: `max_frame 3e5 | num_envs 4 | max_session 4 | log_frequency 500`
*   **After Pulling**: Verify the downloaded `spec.json` matches these rules before using the data.

#### 2. Run Benchmark & Commit Specs
*   **Run**: Execute the benchmark locally or remotely using the commands in [Usage](#usage).
*   **Commit Specs**: Always commit the `spec.json` file used for the run to the repo.
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
| 1 | Classic Control | 3 | 🔄 | 🔄 | 🔄 | 🔄 | 🔄 | 🔄 | 🔄 | Rerun pending |
| 2 | Box2D | 2 | N/A | N/A | 🔄 | 🔄 | 🔄 | 🔄 | 🔄 | Rerun pending |
| 3 | MuJoCo | 11 | N/A | N/A | N/A | N/A | 🔄 | 🔄 | 🔄 | Rerun pending |
| 4 | Atari | 59 | N/A | N/A | N/A | Skip | 🔄 | ✅ | N/A | **54 games** |

**Legend**: ✅ Solved | ⚠️ Close (>80%) | 📊 Acceptable | ❌ Failed | 🔄 In progress/Pending | Skip Not started | N/A Not applicable

---

## Results

### Phase 1: Classic Control

#### 1.1 CartPole-v1

**Docs**: [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) | State: Box(4) | Action: Discrete(2) | Target reward MA > 400

**Settings**: max_frame 2e5 | num_envs 4 | max_session 4 | log_frequency 500

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Repo |
|-----------|--------|-----|-----------|-----------|---------|
| REINFORCE | ✅ | 469.68 | [slm_lab/spec/benchmark/reinforce/reinforce_cartpole.json](../slm_lab/spec/benchmark/reinforce/reinforce_cartpole.json) | reinforce_cartpole | [reinforce_cartpole_2026_01_30_215510](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/reinforce_cartpole_2026_01_30_215510) |
| SARSA | ✅ | 421.58 | [slm_lab/spec/benchmark/sarsa/sarsa_cartpole.json](../slm_lab/spec/benchmark/sarsa/sarsa_cartpole.json) | sarsa_boltzmann_cartpole | [sarsa_boltzmann_cartpole_2026_01_30_215508](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sarsa_boltzmann_cartpole_2026_01_30_215508) |
| DQN | ⚠️ | 188.07 | [slm_lab/spec/benchmark/dqn/dqn_cartpole.json](../slm_lab/spec/benchmark/dqn/dqn_cartpole.json) | dqn_boltzmann_cartpole | [dqn_boltzmann_cartpole_2026_01_30_215213](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/dqn_boltzmann_cartpole_2026_01_30_215213) |
| DDQN+PER | ✅ | 432.88 | [slm_lab/spec/benchmark/dqn/dqn_cartpole.json](../slm_lab/spec/benchmark/dqn/dqn_cartpole.json) | ddqn_per_boltzmann_cartpole | [ddqn_per_boltzmann_cartpole_2026_01_30_215454](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ddqn_per_boltzmann_cartpole_2026_01_30_215454) |
| A2C | ✅ | 499.73 | [slm_lab/spec/benchmark/a2c/a2c_gae_cartpole.json](../slm_lab/spec/benchmark/a2c/a2c_gae_cartpole.json) | a2c_gae_cartpole | [a2c_gae_cartpole_2026_01_30_215337](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_cartpole_2026_01_30_215337) |
| PPO | ✅ | 495.62 | [slm_lab/spec/benchmark/ppo/ppo_cartpole.json](../slm_lab/spec/benchmark/ppo/ppo_cartpole.json) | ppo_cartpole | [ppo_cartpole_2026_02_08_230219](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_cartpole_2026_02_08_230219) |
| SAC | ✅ | 414.97 | [slm_lab/spec/benchmark/sac/sac_cartpole.json](../slm_lab/spec/benchmark/sac/sac_cartpole.json) | sac_cartpole | [sac_cartpole_2026_02_08_141601](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_cartpole_2026_02_08_141601) |

![CartPole-v1 Multi-Trial Graph](./plots/CartPole-v1_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 1.2 Acrobot-v1

**Docs**: [Acrobot](https://gymnasium.farama.org/environments/classic_control/acrobot/) | State: Box(6) | Action: Discrete(3) | Target reward MA > -100

**Settings**: max_frame 3e5 | num_envs 4 | max_session 4 | log_frequency 500

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Repo |
|-----------|--------|-----|-----------|-----------|---------|
| DQN | ✅ | -94.81 | [slm_lab/spec/benchmark/dqn/dqn_acrobot.json](../slm_lab/spec/benchmark/dqn/dqn_acrobot.json) | dqn_boltzmann_acrobot | [dqn_boltzmann_acrobot_2026_01_30_215429](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/dqn_boltzmann_acrobot_2026_01_30_215429) |
| DDQN+PER | ✅ | -85.17 | [slm_lab/spec/benchmark/dqn/ddqn_per_acrobot.json](../slm_lab/spec/benchmark/dqn/ddqn_per_acrobot.json) | ddqn_per_acrobot | [ddqn_per_acrobot_2026_01_30_215436](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ddqn_per_acrobot_2026_01_30_215436) |
| A2C | ✅ | -83.75 | [slm_lab/spec/benchmark/a2c/a2c_gae_acrobot.json](../slm_lab/spec/benchmark/a2c/a2c_gae_acrobot.json) | a2c_gae_acrobot | [a2c_gae_acrobot_2026_01_30_215413](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_acrobot_2026_01_30_215413) |
| PPO | ✅ | -81.43 | [slm_lab/spec/benchmark/ppo/ppo_acrobot.json](../slm_lab/spec/benchmark/ppo/ppo_acrobot.json) | ppo_acrobot | [ppo_acrobot_2026_01_30_215352](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_acrobot_2026_01_30_215352) |
| SAC | ✅ | -90.30 | [slm_lab/spec/benchmark/sac/sac_acrobot.json](../slm_lab/spec/benchmark/sac/sac_acrobot.json) | sac_acrobot | [sac_acrobot_2026_02_08_142215](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_acrobot_2026_02_08_142215) |

![Acrobot-v1 Multi-Trial Graph](./plots/Acrobot-v1_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 1.3 Pendulum-v1

**Docs**: [Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/) | State: Box(3) | Action: Box(1) | Target reward MA > -200

**Settings**: max_frame 3e5 | num_envs 4 | max_session 4 | log_frequency 500

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Repo |
|-----------|--------|-----|-----------|-----------|---------|
| A2C | ❌ | -553.00 | [slm_lab/spec/benchmark/a2c/a2c_gae_pendulum.json](../slm_lab/spec/benchmark/a2c/a2c_gae_pendulum.json) | a2c_gae_pendulum | [a2c_gae_pendulum_2026_01_30_215421](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_pendulum_2026_01_30_215421) |
| PPO | ✅ | -168.26 | [slm_lab/spec/benchmark/ppo/ppo_pendulum.json](../slm_lab/spec/benchmark/ppo/ppo_pendulum.json) | ppo_pendulum | [ppo_pendulum_2026_01_30_215944](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_pendulum_2026_01_30_215944) |
| SAC | ✅ | -148.67 | [slm_lab/spec/benchmark/sac/sac_pendulum.json](../slm_lab/spec/benchmark/sac/sac_pendulum.json) | sac_pendulum | [sac_pendulum_2026_02_08_141615](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_pendulum_2026_02_08_141615) |

![Pendulum-v1 Multi-Trial Graph](./plots/Pendulum-v1_multi_trial_graph_mean_returns_ma_vs_frames.png)

### Phase 2: Box2D

#### 2.1 LunarLander-v3 (Discrete)

**Docs**: [LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) | State: Box(8) | Action: Discrete(4) | Target reward MA > 200

**Settings**: max_frame 3e5 | num_envs 8 | max_session 4 | log_frequency 1000

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Repo |
|-----------|--------|-----|-----------|-----------|---------|
| DQN | ⚠️ | 183.64 | [slm_lab/spec/benchmark/dqn/dqn_lunar.json](../slm_lab/spec/benchmark/dqn/dqn_lunar.json) | dqn_concat_lunar | [dqn_concat_lunar_2026_01_30_215529](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/dqn_concat_lunar_2026_01_30_215529) |
| DDQN+PER | ✅ | 261.49 | [slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json](../slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json) | ddqn_per_concat_lunar | [ddqn_per_concat_lunar_2026_01_30_215532](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ddqn_per_concat_lunar_2026_01_30_215532) |
| A2C | ❌ | 9.53 | [slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json](../slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json) | a2c_gae_lunar | [a2c_gae_lunar_2026_01_30_215529](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_lunar_2026_01_30_215529) |
| PPO | ⚠️ | 159.02 | [slm_lab/spec/benchmark/ppo/ppo_lunar.json](../slm_lab/spec/benchmark/ppo/ppo_lunar.json) | ppo_lunar | [ppo_lunar_2026_01_30_215550](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_lunar_2026_01_30_215550) |
| SAC | ⚠️ | 134.53 | [slm_lab/spec/benchmark/sac/sac_lunar.json](../slm_lab/spec/benchmark/sac/sac_lunar.json) | sac_lunar | [sac_lunar_2026_02_08_141654](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_lunar_2026_02_08_141654) |

![LunarLander-v3 (Discrete) Multi-Trial Graph](./plots/LunarLander-v3_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 2.2 LunarLander-v3 (Continuous)

**Docs**: [LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) | State: Box(8) | Action: Box(2) | Target reward MA > 200

**Settings**: max_frame 3e5 | num_envs 8 | max_session 4 | log_frequency 1000

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Repo |
|-----------|--------|-----|-----------|-----------|---------|
| A2C | ❌ | -38.18 | [slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json](../slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json) | a2c_gae_lunar_continuous | [a2c_gae_lunar_continuous_2026_01_30_215630](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_lunar_continuous_2026_01_30_215630) |
| PPO | ⚠️ | 165.48 | [slm_lab/spec/benchmark/ppo/ppo_lunar.json](../slm_lab/spec/benchmark/ppo/ppo_lunar.json) | ppo_lunar_continuous | [ppo_lunar_continuous_2026_01_31_104549](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_lunar_continuous_2026_01_31_104549) |
| SAC | ⚠️ | 179.40 | [slm_lab/spec/benchmark/sac/sac_lunar.json](../slm_lab/spec/benchmark/sac/sac_lunar.json) | sac_lunar_continuous | [sac_lunar_continuous_2026_02_08_141813](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_lunar_continuous_2026_02_08_141813) |

![LunarLander-v3 (Continuous) Multi-Trial Graph](./plots/LunarLander-v3_Continuous_multi_trial_graph_mean_returns_ma_vs_frames.png)

### Phase 3: MuJoCo

**Docs**: [MuJoCo environments](https://gymnasium.farama.org/environments/mujoco/) | State/Action: Continuous | Target: Practical baselines (no official "solved" threshold)

**Settings**: max_frame 4e6-10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

**Algorithms**: PPO and SAC. Network: MLP [256,256], orthogonal init. PPO uses tanh activation; SAC uses relu.

**Note on SAC frame budgets**: SAC uses higher update-to-data ratios (more gradient updates per step), making it more sample-efficient but slower per frame than PPO. SAC benchmarks use 1-2M frames (vs PPO's 4-10M) to fit within practical GPU wall-time limits (~6h). Scores may still be improving at cutoff.

**Spec Variants**: Two unified specs in [ppo_mujoco.json](../slm_lab/spec/benchmark/ppo/ppo_mujoco.json), plus individual specs for edge cases.

| SPEC_NAME | Envs | Key Config |
|-----------|------|------------|
| ppo_mujoco | HalfCheetah, Walker, Humanoid, HumanoidStandup | gamma=0.99, lam=0.95 |
| ppo_mujoco_longhorizon | Reacher, Pusher | gamma=0.997, lam=0.97 |
| Individual specs | Hopper, Swimmer, Ant, IP, IDP | See spec files for tuned hyperparams |

**Reproduce**: Copy `ENV`, `SPEC_FILE`, `SPEC_NAME` from table. Use `-s max_frame=` for all specs, add `-s env=` for unified specs:
```bash
# Unified specs (ppo_mujoco.json)
source .env && slm-lab run-remote --gpu -s env=ENV -s max_frame=MAX_FRAME \
  slm_lab/spec/benchmark/ppo/ppo_mujoco.json SPEC_NAME train -n NAME

# Individual specs (env hardcoded)
source .env && slm-lab run-remote --gpu -s max_frame=MAX_FRAME \
  slm_lab/spec/benchmark/ppo/SPEC_FILE SPEC_NAME train -n NAME
```

| ENV | MAX_FRAME | SPEC_FILE | SPEC_NAME |
|-----|-----------|-----------|-----------|
| Ant-v5 | 10e6 | ppo_ant.json | ppo_ant |
| HalfCheetah-v5 | 10e6 | ppo_mujoco.json | ppo_mujoco |
| Hopper-v5 | 4e6 | ppo_hopper.json | ppo_hopper |
| Humanoid-v5 | 10e6 | ppo_mujoco.json | ppo_mujoco |
| HumanoidStandup-v5 | 4e6 | ppo_mujoco.json | ppo_mujoco |
| InvertedDoublePendulum-v5 | 10e6 | ppo_inverted_double_pendulum.json | ppo_inverted_double_pendulum |
| InvertedPendulum-v5 | 4e6 | ppo_inverted_pendulum.json | ppo_inverted_pendulum |
| Pusher-v5 | 4e6 | ppo_mujoco.json | ppo_mujoco_longhorizon |
| Reacher-v5 | 4e6 | ppo_mujoco.json | ppo_mujoco_longhorizon |
| Swimmer-v5 | 4e6 | ppo_swimmer.json | ppo_swimmer |
| Walker2d-v5 | 10e6 | ppo_mujoco.json | ppo_mujoco |

**SAC Reproduce**: Use individual specs (recommended — all hyperparams pre-configured):
```bash
# Individual specs: env, max_frame, and all hyperparams are hardcoded in the spec
source .env && slm-lab run-remote --gpu \
  slm_lab/spec/benchmark/sac/SPEC_FILE SPEC_NAME train -n NAME

# Example: reproduce Hopper SAC
source .env && slm-lab run-remote --gpu \
  slm_lab/spec/benchmark/sac/sac_hopper.json sac_hopper train -n sac-hopper
```

| ENV | MAX_FRAME | SPEC_FILE | SPEC_NAME |
|-----|-----------|-----------|-----------|
| Ant-v5 | 2e6 | sac_ant.json | sac_ant |
| HalfCheetah-v5 | 2e6 | sac_halfcheetah.json | sac_halfcheetah |
| Hopper-v5 | 2e6 | sac_hopper.json | sac_hopper |
| Humanoid-v5 | 1e6 | sac_humanoid.json | sac_humanoid |
| HumanoidStandup-v5 | 1e6 | sac_humanoid_standup.json | sac_humanoid_standup |
| InvertedDoublePendulum-v5 | 2e6 | sac_inverted_double_pendulum.json | sac_inverted_double_pendulum |
| InvertedPendulum-v5 | 2e6 | sac_inverted_pendulum.json | sac_inverted_pendulum |
| Pusher-v5 | 1e6 | sac_pusher.json | sac_pusher |
| Reacher-v5 | 1e6 | sac_reacher.json | sac_reacher |
| Swimmer-v5 | 2e6 | sac_swimmer.json | sac_swimmer |
| Walker2d-v5 | 2e6 | sac_walker2d.json | sac_walker2d |

#### 3.1 Ant-v5

**Docs**: [Ant](https://gymnasium.farama.org/environments/mujoco/ant/) | State: Box(105) | Action: Box(8) | Target reward MA > 2000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Repo |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 2514.64 | [slm_lab/spec/benchmark/ppo/ppo_ant.json](../slm_lab/spec/benchmark/ppo/ppo_ant.json) | ppo_ant | [ppo_ant_2026_01_31_042006](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_ant_2026_01_31_042006) |
| SAC | ✅ | 4844.20 | [slm_lab/spec/benchmark/sac/sac_ant.json](../slm_lab/spec/benchmark/sac/sac_ant.json) | sac_ant | [sac_ant_2026_02_09_093821](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_ant_2026_02_09_093821) |

![Ant-v5 Multi-Trial Graph](./plots/Ant-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.2 HalfCheetah-v5

**Docs**: [HalfCheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) | State: Box(17) | Action: Box(6) | Target reward MA > 5000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Repo |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 5851.70 | [slm_lab/spec/benchmark/ppo/ppo_mujoco.json](../slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | ppo_mujoco | [ppo_mujoco_halfcheetah_2026_01_30_230302](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_halfcheetah_2026_01_30_230302) |
| SAC | ✅ | 7255.37 | [slm_lab/spec/benchmark/sac/sac_halfcheetah.json](../slm_lab/spec/benchmark/sac/sac_halfcheetah.json) | sac_halfcheetah | [sac_halfcheetah_2026_02_08_115456](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_halfcheetah_2026_02_08_115456) |

![HalfCheetah-v5 Multi-Trial Graph](./plots/HalfCheetah-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.3 Hopper-v5

**Docs**: [Hopper](https://gymnasium.farama.org/environments/mujoco/hopper/) | State: Box(11) | Action: Box(3) | Target reward MA ~ 2000

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Repo |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 1972.38 | [slm_lab/spec/benchmark/ppo/ppo_hopper.json](../slm_lab/spec/benchmark/ppo/ppo_hopper.json) | ppo_hopper | [ppo_hopper_2026_01_31_105438](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_hopper_2026_01_31_105438) |
| SAC | ⚠️ | 1511.00 | [slm_lab/spec/benchmark/sac/sac_hopper.json](../slm_lab/spec/benchmark/sac/sac_hopper.json) | sac_hopper | [sac_hopper_2026_02_10_031009](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_hopper_2026_02_10_031009) |

![Hopper-v5 Multi-Trial Graph](./plots/Hopper-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.4 Humanoid-v5

**Docs**: [Humanoid](https://gymnasium.farama.org/environments/mujoco/humanoid/) | State: Box(348) | Action: Box(17) | Target reward MA > 1000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Repo |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 3774.08 | [slm_lab/spec/benchmark/ppo/ppo_mujoco.json](../slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | ppo_mujoco | [ppo_mujoco_humanoid_2026_01_30_222339](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_humanoid_2026_01_30_222339) |
| SAC | ✅ | 2601.03 | [slm_lab/spec/benchmark/sac/sac_humanoid.json](../slm_lab/spec/benchmark/sac/sac_humanoid.json) | sac_humanoid | [sac_humanoid_2026_02_09_223653](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_humanoid_2026_02_09_223653) |

![Humanoid-v5 Multi-Trial Graph](./plots/Humanoid-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.5 HumanoidStandup-v5

**Docs**: [HumanoidStandup](https://gymnasium.farama.org/environments/mujoco/humanoid_standup/) | State: Box(348) | Action: Box(17) | Target reward MA > 100000

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Repo |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 165841.17 | [slm_lab/spec/benchmark/ppo/ppo_mujoco.json](../slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | ppo_mujoco | [ppo_mujoco_humanoidstandup_2026_01_30_215802](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_humanoidstandup_2026_01_30_215802) |
| SAC | ✅ | 138221.92 | [slm_lab/spec/benchmark/sac/sac_humanoid_standup.json](../slm_lab/spec/benchmark/sac/sac_humanoid_standup.json) | sac_humanoid_standup | [sac_humanoid_standup_2026_02_09_213409](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_humanoid_standup_2026_02_09_213409) |

![HumanoidStandup-v5 Multi-Trial Graph](./plots/HumanoidStandup-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.6 InvertedDoublePendulum-v5

**Docs**: [InvertedDoublePendulum](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/) | State: Box(9) | Action: Box(1) | Target reward MA ~8000

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Repo |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 7622.00 | [slm_lab/spec/benchmark/ppo/ppo_inverted_double_pendulum.json](../slm_lab/spec/benchmark/ppo/ppo_inverted_double_pendulum.json) | ppo_inverted_double_pendulum | [ppo_inverted_double_pendulum_2026_01_30_220651](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_inverted_double_pendulum_2026_01_30_220651) |
| SAC | ✅ | 9000.39 | [slm_lab/spec/benchmark/sac/sac_inverted_double_pendulum.json](../slm_lab/spec/benchmark/sac/sac_inverted_double_pendulum.json) | sac_inverted_double_pendulum | [sac_inverted_double_pendulum_2026_02_08_115548](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_inverted_double_pendulum_2026_02_08_115548) |

![InvertedDoublePendulum-v5 Multi-Trial Graph](./plots/InvertedDoublePendulum-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.7 InvertedPendulum-v5

**Docs**: [InvertedPendulum](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/) | State: Box(4) | Action: Box(1) | Target reward MA ~1000

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Repo |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 944.87 | [slm_lab/spec/benchmark/ppo/ppo_inverted_pendulum.json](../slm_lab/spec/benchmark/ppo/ppo_inverted_pendulum.json) | ppo_inverted_pendulum | [ppo_inverted_pendulum_2026_01_30_230211](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_inverted_pendulum_2026_01_30_230211) |
| SAC | ✅ | 927.58 | [slm_lab/spec/benchmark/sac/sac_inverted_pendulum.json](../slm_lab/spec/benchmark/sac/sac_inverted_pendulum.json) | sac_inverted_pendulum | [sac_inverted_pendulum_2026_02_08_132433](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_inverted_pendulum_2026_02_08_132433) |

![InvertedPendulum-v5 Multi-Trial Graph](./plots/InvertedPendulum-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.8 Pusher-v5

**Docs**: [Pusher](https://gymnasium.farama.org/environments/mujoco/pusher/) | State: Box(23) | Action: Box(7) | Target reward MA > -50

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Repo |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | -49.09 | [slm_lab/spec/benchmark/ppo/ppo_mujoco.json](../slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | ppo_mujoco_longhorizon | [ppo_mujoco_longhorizon_pusher_2026_01_30_215824](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_longhorizon_pusher_2026_01_30_215824) |
| SAC | ✅ | -42.61 | [slm_lab/spec/benchmark/sac/sac_pusher.json](../slm_lab/spec/benchmark/sac/sac_pusher.json) | sac_pusher | [sac_pusher_2026_02_08_115643](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_pusher_2026_02_08_115643) |

![Pusher-v5 Multi-Trial Graph](./plots/Pusher-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.9 Reacher-v5

**Docs**: [Reacher](https://gymnasium.farama.org/environments/mujoco/reacher/) | State: Box(10) | Action: Box(2) | Target reward MA > -10

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Repo |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | -5.08 | [slm_lab/spec/benchmark/ppo/ppo_mujoco.json](../slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | ppo_mujoco_longhorizon | [ppo_mujoco_longhorizon_reacher_2026_01_30_215805](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_longhorizon_reacher_2026_01_30_215805) |
| SAC | ✅ | -6.36 | [slm_lab/spec/benchmark/sac/sac_reacher.json](../slm_lab/spec/benchmark/sac/sac_reacher.json) | sac_reacher | [sac_reacher_2026_02_08_115637](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_reacher_2026_02_08_115637) |

![Reacher-v5 Multi-Trial Graph](./plots/Reacher-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.10 Swimmer-v5

**Docs**: [Swimmer](https://gymnasium.farama.org/environments/mujoco/swimmer/) | State: Box(8) | Action: Box(2) | Target reward MA > 200

**Settings**: max_frame 4e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Repo |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 229.31 | [slm_lab/spec/benchmark/ppo/ppo_swimmer.json](../slm_lab/spec/benchmark/ppo/ppo_swimmer.json) | ppo_swimmer | [ppo_swimmer_2026_01_30_215922](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_swimmer_2026_01_30_215922) |
| SAC | ✅ | 264.98 | [slm_lab/spec/benchmark/sac/sac_swimmer.json](../slm_lab/spec/benchmark/sac/sac_swimmer.json) | sac_swimmer | [sac_swimmer_2026_02_08_115455](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_swimmer_2026_02_08_115455) |

![Swimmer-v5 Multi-Trial Graph](./plots/Swimmer-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

#### 3.11 Walker2d-v5

**Docs**: [Walker2d](https://gymnasium.farama.org/environments/mujoco/walker2d/) | State: Box(17) | Action: Box(6) | Target reward MA > 3500

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4

| Algorithm | Status | MA | SPEC_FILE | SPEC_NAME | HF Repo |
|-----------|--------|-----|-----------|-----------|---------|
| PPO | ✅ | 4042.07 | [slm_lab/spec/benchmark/ppo/ppo_mujoco.json](../slm_lab/spec/benchmark/ppo/ppo_mujoco.json) | ppo_mujoco | [ppo_mujoco_walker2d_2026_01_30_222124](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_mujoco_walker2d_2026_01_30_222124) |
| SAC | ⚠️ | 2288.03 | [slm_lab/spec/benchmark/sac/sac_walker2d.json](../slm_lab/spec/benchmark/sac/sac_walker2d.json) | sac_walker2d | [sac_walker2d_2026_02_08_221549](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_walker2d_2026_02_08_221549) |

![Walker2d-v5 Multi-Trial Graph](./plots/Walker2d-v5_multi_trial_graph_mean_returns_ma_vs_frames.png)

### Phase 4: Atari

**Docs**: [Atari environments](https://ale.farama.org/environments/) | State: Box(84,84,4 after preprocessing) | Action: Discrete(4-18, game-dependent)

**Settings**: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 10000

**Environment**:
- Gymnasium ALE v5 with `life_loss_info=true`
- v5 uses sticky actions (`repeat_action_probability=0.25`) per [Machado et al. (2018)](https://arxiv.org/abs/1709.06009) best practices

**Algorithm Specs** (all use Nature CNN [32,64,64] + 512fc):
- **DDQN+PER**: Skipped - off-policy variants ~6x slower (~230 fps vs ~1500 fps), not cost effective at 10M frames
- **A2C**: [a2c_gae_atari.json](../slm_lab/spec/benchmark/a2c/a2c_gae_atari.json) - RMSprop (lr=7e-4), training_frequency=32
- **PPO**: [ppo_atari.json](../slm_lab/spec/benchmark/ppo/ppo_atari.json) - AdamW (lr=2.5e-4), minibatch=256, horizon=128, epochs=4
- **SAC**: [sac_atari.json](../slm_lab/spec/benchmark/sac/sac_atari.json) - Discrete SAC (Categorical), AdamW (lr=3e-4), batch=256, buffer=200K, max_frame=2e6

**PPO Lambda Variants** (table shows best result per game):

| SPEC_NAME | Lambda | Best for |
|-----------|--------|----------|
| ppo_atari | 0.95 | Strategic games (default) |
| ppo_atari_lam85 | 0.85 | Mixed games |
| ppo_atari_lam70 | 0.70 | Action games |

**Reproduce**:
```bash
# A2C
source .env && slm-lab run-remote --gpu -s env=ENV -s max_frame=1e7 \
  slm_lab/spec/benchmark/a2c/a2c_gae_atari.json a2c_gae_atari train -n NAME

# PPO
source .env && slm-lab run-remote --gpu -s env=ENV -s max_frame=1e7 \
  slm_lab/spec/benchmark/ppo/ppo_atari.json SPEC_NAME train -n NAME

# SAC (2M frames - off-policy, more sample-efficient but slower per frame)
source .env && slm-lab run-remote --gpu -s env=ENV \
  slm_lab/spec/benchmark/sac/sac_atari.json sac_atari train -n NAME
```

| ENV | Score | SPEC_NAME | HF Repo |
|-----|-------|-----------|---------|
| ALE/AirRaid-v5 | 5067 | a2c_gae_atari | [a2c_gae_atari_airraid_2026_02_01_082446](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_airraid_2026_02_01_082446) |
| | 8245 | ppo_atari | [ppo_atari_airraid_2026_01_06_113119](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_airraid_2026_01_06_113119) |
| | 2061 | sac_atari | [sac_atari_airraid_2026_02_11_203802](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_airraid_2026_02_11_203802) |
| ALE/Alien-v5 | 1488 | a2c_gae_atari | [a2c_gae_atari_alien_2026_02_01_000858](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_alien_2026_02_01_000858) |
| | 1453 | ppo_atari | [ppo_atari_alien_2026_01_06_112514](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_alien_2026_01_06_112514) |
| | 960 | sac_atari | [sac_atari_alien_2026_02_11_201443](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_alien_2026_02_11_201443) |
| ALE/Amidar-v5 | 330 | a2c_gae_atari | [a2c_gae_atari_amidar_2026_02_01_082251](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_amidar_2026_02_01_082251) |
| | 580 | ppo_atari_lam85 | [ppo_atari_lam85_amidar_2026_01_07_223416](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_amidar_2026_01_07_223416) |
| | 187 | sac_atari | [sac_atari_amidar_2026_02_11_202456](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_amidar_2026_02_11_202456) |
| ALE/Assault-v5 | 1646 | a2c_gae_atari | [a2c_gae_atari_assault_2026_02_01_082252](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_assault_2026_02_01_082252) |
| | 4293 | ppo_atari_lam85 | [ppo_atari_lam85_assault_2026_01_08_130044](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_assault_2026_01_08_130044) |
| | 1037 | sac_atari | [sac_atari_assault_2026_02_11_201441](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_assault_2026_02_11_201441) |
| ALE/Asterix-v5 | 2712 | a2c_gae_atari | [a2c_gae_atari_asterix_2026_02_01_082315](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_asterix_2026_02_01_082315) |
| | 3482 | ppo_atari_lam85 | [ppo_atari_lam85_asterix_2026_01_07_223445](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_asterix_2026_01_07_223445) |
| | 1450 | sac_atari | [sac_atari_asterix_2026_02_11_203629](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_asterix_2026_02_11_203629) |
| ALE/Asteroids-v5 | 2106 | a2c_gae_atari | [a2c_gae_atari_asteroids_2026_02_01_082328](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_asteroids_2026_02_01_082328) |
| | 1554 | ppo_atari_lam85 | [ppo_atari_lam85_asteroids_2026_01_07_224245](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_asteroids_2026_01_07_224245) |
| | 1216 | sac_atari | [sac_atari_asteroids_2026_02_11_201524](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_asteroids_2026_02_11_201524) |
| ALE/Atlantis-v5 | 873365 | a2c_gae_atari | [a2c_gae_atari_atlantis_2026_02_01_082330](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_atlantis_2026_02_01_082330) |
| | 792886 | ppo_atari | [ppo_atari_atlantis_2026_01_06_120440](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_atlantis_2026_01_06_120440) |
| | 64097 | sac_atari | [sac_atari_atlantis_2026_02_11_204715](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_atlantis_2026_02_11_204715) |
| ALE/BankHeist-v5 | 1099 | a2c_gae_atari | [a2c_gae_atari_bankheist_2026_02_01_082403](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_bankheist_2026_02_01_082403) |
| | 1045 | ppo_atari | [ppo_atari_bankheist_2026_01_06_121042](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_bankheist_2026_01_06_121042) |
| | 132 | sac_atari | [sac_atari_bankheist_2026_02_11_201643](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_bankheist_2026_02_11_201643) |
| ALE/BattleZone-v5 | 2437 | a2c_gae_atari | [a2c_gae_atari_battlezone_2026_02_01_082425](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_battlezone_2026_02_01_082425) |
| | 26383 | ppo_atari_lam85 | [ppo_atari_lam85_battlezone_2026_01_08_094729](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_battlezone_2026_01_08_094729) |
| | 6951 | sac_atari | [sac_atari_battlezone_2026_02_12_003638](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_battlezone_2026_02_12_003638) |
| ALE/BeamRider-v5 | 2767 | a2c_gae_atari | [a2c_gae_atari_beamrider_2026_02_01_000921](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_beamrider_2026_02_01_000921) |
| | 2765 | ppo_atari | [ppo_atari_beamrider_2026_01_06_112533](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_beamrider_2026_01_06_112533) |
| | 4048 | sac_atari | [sac_atari_beamrider_2026_02_12_003619](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_beamrider_2026_02_12_003619) |
| ALE/Berzerk-v5 | 439 | a2c_gae_atari | [a2c_gae_atari_berzerk_2026_02_01_082540](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_berzerk_2026_02_01_082540) |
| | 1072 | ppo_atari | [ppo_atari_berzerk_2026_01_06_112515](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_berzerk_2026_01_06_112515) |
| | 314 | sac_atari | [sac_atari_berzerk_2026_02_12_004654](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_berzerk_2026_02_12_004654) |
| ALE/Bowling-v5 | 23.96 | a2c_gae_atari | [a2c_gae_atari_bowling_2026_02_01_082529](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_bowling_2026_02_01_082529) |
| | 46.45 | ppo_atari | [ppo_atari_bowling_2026_01_06_113148](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_bowling_2026_01_06_113148) |
| | 27.95 | sac_atari | [sac_atari_bowling_2026_02_12_004809](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_bowling_2026_02_12_004809) |
| ALE/Boxing-v5 | 1.80 | a2c_gae_atari | [a2c_gae_atari_boxing_2026_02_01_082539](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_boxing_2026_02_01_082539) |
| | 91.17 | ppo_atari | [ppo_atari_boxing_2026_01_06_112531](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_boxing_2026_01_06_112531) |
| | 40.15 | sac_atari | [sac_atari_boxing_2026_02_12_004826](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_boxing_2026_02_12_004826) |
| ALE/Breakout-v5 | 273 | a2c_gae_atari | [a2c_gae_atari_breakout_2026_01_31_213610](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_breakout_2026_01_31_213610) |
| | 327 | ppo_atari_lam70 | [ppo_atari_lam70_breakout_2026_01_07_110559](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_breakout_2026_01_07_110559) |
| | 16.45 | sac_atari | [sac_atari_breakout_2026_02_12_005931](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_breakout_2026_02_12_005931) |
| ALE/Carnival-v5 | 2170 | a2c_gae_atari | [a2c_gae_atari_carnival_2026_02_01_082726](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_carnival_2026_02_01_082726) |
| | 3967 | ppo_atari_lam70 | [ppo_atari_lam70_carnival_2026_01_07_144738](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_carnival_2026_01_07_144738) |
| | 4025 | sac_atari | [sac_atari_carnival_2026_02_12_013737](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_carnival_2026_02_12_013737) |
| ALE/Centipede-v5 | 1382 | a2c_gae_atari | [a2c_gae_atari_centipede_2026_02_01_082643](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_centipede_2026_02_01_082643) |
| | 4915 | ppo_atari_lam70 | [ppo_atari_lam70_centipede_2026_01_07_223557](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_centipede_2026_01_07_223557) |
| | 2286 | sac_atari | [sac_atari_centipede_2026_02_12_013445](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_centipede_2026_02_12_013445) |
| ALE/ChopperCommand-v5 | 2446 | a2c_gae_atari | [a2c_gae_atari_choppercommand_2026_02_01_082626](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_choppercommand_2026_02_01_082626) |
| | 5355 | ppo_atari | [ppo_atari_choppercommand_2026_01_07_110539](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_choppercommand_2026_01_07_110539) |
| | 1068 | sac_atari | [sac_atari_choppercommand_2026_02_12_060732](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_choppercommand_2026_02_12_060732) |
| ALE/CrazyClimber-v5 | 96943 | a2c_gae_atari | [a2c_gae_atari_crazyclimber_2026_02_01_082625](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_crazyclimber_2026_02_01_082625) |
| | 107370 | ppo_atari_lam85 | [ppo_atari_lam85_crazyclimber_2026_01_07_223609](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_crazyclimber_2026_01_07_223609) |
| | 81839 | sac_atari | [sac_atari_crazyclimber_2026_02_12_053919](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_crazyclimber_2026_02_12_053919) |
| ALE/Defender-v5 | 33149 | a2c_gae_atari | [a2c_gae_atari_defender_2026_02_01_082658](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_defender_2026_02_01_082658) |
| | 51439 | ppo_atari_lam70 | [ppo_atari_lam70_defender_2026_01_07_205238](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_defender_2026_01_07_205238) |
| | 3832 | sac_atari | [sac_atari_defender_2026_02_12_054055](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_defender_2026_02_12_054055) |
| ALE/DemonAttack-v5 | 2962 | a2c_gae_atari | [a2c_gae_atari_demonattack_2026_02_01_082717](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_demonattack_2026_02_01_082717) |
| | 16558 | ppo_atari_lam70 | [ppo_atari_lam70_demonattack_2026_01_07_111315](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_demonattack_2026_01_07_111315) |
| | 4330 | sac_atari | [sac_atari_demonattack_2026_02_12_054035](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_demonattack_2026_02_12_054035) |
| ALE/DoubleDunk-v5 | -1.69 | a2c_gae_atari | [a2c_gae_atari_doubledunk_2026_02_01_082901](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_doubledunk_2026_02_01_082901) |
| | -2.38 | ppo_atari | [ppo_atari_doubledunk_2026_01_07_110802](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_doubledunk_2026_01_07_110802) |
| | -43.51 | sac_atari | [sac_atari_doubledunk_2026_02_12_054050](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_doubledunk_2026_02_12_054050) |
| ALE/ElevatorAction-v5 | 731 | a2c_gae_atari | [a2c_gae_atari_elevatoraction_2026_02_01_082908](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_elevatoraction_2026_02_01_082908) |
| | 5446 | ppo_atari | [ppo_atari_elevatoraction_2026_01_06_113129](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_elevatoraction_2026_01_06_113129) |
| | 4374 | sac_atari | [sac_atari_elevatoraction_2026_02_12_061339](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_elevatoraction_2026_02_12_061339) |
| ALE/Enduro-v5 | 681 | a2c_gae_atari | [a2c_gae_atari_enduro_2026_02_01_001123](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_enduro_2026_02_01_001123) |
| | 898 | ppo_atari_lam85 | [ppo_atari_lam85_enduro_2026_01_08_095448](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_enduro_2026_01_08_095448) |
| | 0 | sac_atari | [sac_atari_enduro_2026_02_12_190545](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_enduro_2026_02_12_190545) |
| ALE/FishingDerby-v5 | -16.38 | a2c_gae_atari | [a2c_gae_atari_fishingderby_2026_02_01_082906](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_fishingderby_2026_02_01_082906) |
| | 27.10 | ppo_atari_lam85 | [ppo_atari_lam85_fishingderby_2026_01_08_094158](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_fishingderby_2026_01_08_094158) |
| | -76.88 | sac_atari | [sac_atari_fishingderby_2026_02_12_061350](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_fishingderby_2026_02_12_061350) |
| ALE/Freeway-v5 | 23.13 | a2c_gae_atari | [a2c_gae_atari_freeway_2026_02_01_082931](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_freeway_2026_02_01_082931) |
| | 31.30 | ppo_atari | [ppo_atari_freeway_2026_01_06_182318](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_freeway_2026_01_06_182318) |
| | 0 | sac_atari | [sac_atari_freeway_2026_02_12_101448](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_freeway_2026_02_12_101448) |
| ALE/Frostbite-v5 | 266 | a2c_gae_atari | [a2c_gae_atari_frostbite_2026_02_01_082915](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_frostbite_2026_02_01_082915) |
| | 301 | ppo_atari | [ppo_atari_frostbite_2026_01_06_112556](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_frostbite_2026_01_06_112556) |
| | 400 | sac_atari | [sac_atari_frostbite_2026_02_12_183946](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_frostbite_2026_02_12_183946) |
| ALE/Gopher-v5 | 984 | a2c_gae_atari | [a2c_gae_atari_gopher_2026_02_01_133323](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_gopher_2026_02_01_133323) |
| | 6508 | ppo_atari_lam70 | [ppo_atari_lam70_gopher_2026_01_07_170451](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_gopher_2026_01_07_170451) |
| | 1895 | sac_atari | [sac_atari_gopher_2026_02_12_121106](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_gopher_2026_02_12_121106) |
| ALE/Gravitar-v5 | 270 | a2c_gae_atari | [a2c_gae_atari_gravitar_2026_02_01_133244](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_gravitar_2026_02_01_133244) |
| | 599 | ppo_atari | [ppo_atari_gravitar_2026_01_06_112548](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_gravitar_2026_01_06_112548) |
| | 234 | sac_atari | [sac_atari_gravitar_2026_02_12_131258](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_gravitar_2026_02_12_131258) |
| ALE/Hero-v5 | 18680 | a2c_gae_atari | [a2c_gae_atari_hero_2026_02_01_175903](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_hero_2026_02_01_175903) |
| | 28238 | ppo_atari_lam85 | [ppo_atari_lam85_hero_2026_01_07_223619](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_hero_2026_01_07_223619) |
| | 4526 | sac_atari | [sac_atari_hero_2026_02_12_115556](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_hero_2026_02_12_115556) |
| ALE/IceHockey-v5 | -5.92 | a2c_gae_atari | [a2c_gae_atari_icehockey_2026_02_01_175745](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_icehockey_2026_02_01_175745) |
| | -3.93 | ppo_atari | [ppo_atari_icehockey_2026_01_06_183721](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_icehockey_2026_01_06_183721) |
| | -19.52 | sac_atari | [sac_atari_icehockey_2026_02_12_183953](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_icehockey_2026_02_12_183953) |
| ALE/Jamesbond-v5 | 460 | a2c_gae_atari | [a2c_gae_atari_jamesbond_2026_02_01_175945](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_jamesbond_2026_02_01_175945) |
| | 662 | ppo_atari | [ppo_atari_jamesbond_2026_01_06_183717](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_jamesbond_2026_01_06_183717) |
| | 265 | sac_atari | [sac_atari_jamesbond_2026_02_12_115613](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_jamesbond_2026_02_12_115613) |
| ALE/JourneyEscape-v5 | -965 | a2c_gae_atari | [a2c_gae_atari_journeyescape_2026_02_01_084415](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_journeyescape_2026_02_01_084415) |
| | -1252 | ppo_atari_lam85 | [ppo_atari_lam85_journeyescape_2026_01_08_094842](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_journeyescape_2026_01_08_094842) |
| | -3392 | sac_atari | [sac_atari_journeyescape_2026_02_12_183939](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_journeyescape_2026_02_12_183939) |
| ALE/Kangaroo-v5 | 322 | a2c_gae_atari | [a2c_gae_atari_kangaroo_2026_02_01_084415](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_kangaroo_2026_02_01_084415) |
| | 9912 | ppo_atari_lam85 | [ppo_atari_lam85_kangaroo_2026_01_07_110838](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_kangaroo_2026_01_07_110838) |
| | 3317 | sac_atari | [sac_atari_kangaroo_2026_02_12_184007](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_kangaroo_2026_02_12_184007) |
| ALE/Krull-v5 | 7519 | a2c_gae_atari | [a2c_gae_atari_krull_2026_02_01_084420](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_krull_2026_02_01_084420) |
| | 7841 | ppo_atari | [ppo_atari_krull_2026_01_07_110747](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_krull_2026_01_07_110747) |
| | 6824 | sac_atari | [sac_atari_krull_2026_02_12_184007](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_krull_2026_02_12_184007) |
| ALE/KungFuMaster-v5 | 23006 | a2c_gae_atari | [a2c_gae_atari_kungfumaster_2026_02_01_085101](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_kungfumaster_2026_02_01_085101) |
| | 29068 | ppo_atari_lam70 | [ppo_atari_lam70_kungfumaster_2026_01_07_111317](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_kungfumaster_2026_01_07_111317) |
| | 8511 | sac_atari | [sac_atari_kungfumaster_2026_02_12_184021](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_kungfumaster_2026_02_12_184021) |
| ALE/MsPacman-v5 | 2110 | a2c_gae_atari | [a2c_gae_atari_mspacman_2026_02_01_001100](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_mspacman_2026_02_01_001100) |
| | 2372 | ppo_atari_lam85 | [ppo_atari_lam85_mspacman_2026_01_07_223522](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_mspacman_2026_01_07_223522) |
| | 1396 | sac_atari | [sac_atari_mspacman_2026_02_12_184018](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_mspacman_2026_02_12_184018) |
| ALE/NameThisGame-v5 | 5412 | a2c_gae_atari | [a2c_gae_atari_namethisgame_2026_02_01_132733](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_namethisgame_2026_02_01_132733) |
| | 5993 | ppo_atari | [ppo_atari_namethisgame_2026_01_06_182952](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_namethisgame_2026_01_06_182952) |
| | 4034 | sac_atari | [sac_atari_namethisgame_2026_02_12_230011](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_namethisgame_2026_02_12_230011) |
| ALE/Phoenix-v5 | 5635 | a2c_gae_atari | [a2c_gae_atari_phoenix_2026_02_01_085101](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_phoenix_2026_02_01_085101) |
| | 15659 | ppo_atari_lam70 | [ppo_atari_lam70_phoenix_2026_01_07_110832](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_phoenix_2026_01_07_110832) |
| | 3909 | sac_atari | [sac_atari_phoenix_2026_02_12_231134](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_phoenix_2026_02_12_231134) |
| ALE/Pong-v5 | 10.17 | a2c_gae_atari | [a2c_gae_atari_pong_2026_01_31_213635](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_pong_2026_01_31_213635) |
| | 16.91 | ppo_atari_lam85 | [ppo_atari_lam85_pong_2026_01_08_094454](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_pong_2026_01_08_094454) |
| | 12.14 | sac_atari | [sac_atari_pong_2026_02_12_231240](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_pong_2026_02_12_231240) |
| ALE/Pooyan-v5 | 2997 | a2c_gae_atari | [a2c_gae_atari_pooyan_2026_02_01_132748](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_pooyan_2026_02_01_132748) |
| | 5716 | ppo_atari_lam70 | [ppo_atari_lam70_pooyan_2026_01_07_224346](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_pooyan_2026_01_07_224346) |
| | 2625 | sac_atari | [sac_atari_pooyan_2026_02_12_233303](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_pooyan_2026_02_12_233303) |
| ALE/Qbert-v5 | 12619 | a2c_gae_atari | [a2c_gae_atari_qbert_2026_01_31_213720](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_qbert_2026_01_31_213720) |
| | 15094 | ppo_atari | [ppo_atari_qbert_2026_01_06_111801](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_qbert_2026_01_06_111801) |
| | 3610 | sac_atari | [sac_atari_qbert_2026_02_12_233409](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_qbert_2026_02_12_233409) |
| ALE/Riverraid-v5 | 6558 | a2c_gae_atari | [a2c_gae_atari_riverraid_2026_02_01_132507](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_riverraid_2026_02_01_132507) |
| | 9428 | ppo_atari_lam85 | [ppo_atari_lam85_riverraid_2026_01_07_204356](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_riverraid_2026_01_07_204356) |
| | 5125 | sac_atari | [sac_atari_riverraid_2026_02_12_233410](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_riverraid_2026_02_12_233410) |
| ALE/RoadRunner-v5 | 29810 | a2c_gae_atari | [a2c_gae_atari_roadrunner_2026_02_01_132509](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_roadrunner_2026_02_01_132509) |
| | 37015 | ppo_atari_lam85 | [ppo_atari_lam85_roadrunner_2026_01_07_145913](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_roadrunner_2026_01_07_145913) |
| | 23899 | sac_atari | [sac_atari_roadrunner_2026_02_12_233449](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_roadrunner_2026_02_12_233449) |
| ALE/Robotank-v5 | 2.80 | a2c_gae_atari | [a2c_gae_atari_robotank_2026_02_01_132434](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_robotank_2026_02_01_132434) |
| | 20.07 | ppo_atari | [ppo_atari_robotank_2026_01_06_183413](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_robotank_2026_01_06_183413) |
| | 9.06 | sac_atari | [sac_atari_robotank_2026_02_12_233446](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_robotank_2026_02_12_233446) |
| ALE/Seaquest-v5 | 850 | a2c_gae_atari | [a2c_gae_atari_seaquest_2026_02_01_001001](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_seaquest_2026_02_01_001001) |
| | 1796 | ppo_atari | [ppo_atari_seaquest_2026_01_06_183440](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_seaquest_2026_01_06_183440) |
| | 2010 | sac_atari | [sac_atari_seaquest_2026_02_13_111005](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_seaquest_2026_02_13_111005) |
| ALE/Skiing-v5 | -14235 | a2c_gae_atari | [a2c_gae_atari_skiing_2026_02_01_132451](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_skiing_2026_02_01_132451) |
| | -19340 | ppo_atari | [ppo_atari_skiing_2026_01_06_183424](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_skiing_2026_01_06_183424) |
| | -16710 | sac_atari | [sac_atari_skiing_2026_02_13_034039](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_skiing_2026_02_13_034039) |
| ALE/Solaris-v5 | 2224 | a2c_gae_atari | [a2c_gae_atari_solaris_2026_02_01_212137](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_solaris_2026_02_01_212137) |
| | 2094 | ppo_atari | [ppo_atari_solaris_2026_01_06_192643](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_solaris_2026_01_06_192643) |
| | 1803 | sac_atari | [sac_atari_solaris_2026_02_13_105520](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_solaris_2026_02_13_105520) |
| ALE/SpaceInvaders-v5 | 784 | a2c_gae_atari | [a2c_gae_atari_spaceinvaders_2026_02_01_000950](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_spaceinvaders_2026_02_01_000950) |
| | 726 | ppo_atari | [ppo_atari_spaceinvaders_2026_01_07_102346](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_spaceinvaders_2026_01_07_102346) |
| | 517 | sac_atari | [sac_atari_spaceinvaders_2026_02_11_080424](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_spaceinvaders_2026_02_11_080424) |
| ALE/StarGunner-v5 | 8665 | a2c_gae_atari | [a2c_gae_atari_stargunner_2026_02_01_132406](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_stargunner_2026_02_01_132406) |
| | 47495 | ppo_atari_lam70 | [ppo_atari_lam70_stargunner_2026_01_07_111404](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_stargunner_2026_01_07_111404) |
| | 3809 | sac_atari | [sac_atari_stargunner_2026_02_13_040158](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_stargunner_2026_02_13_040158) |
| ALE/Surround-v5 | -9.72 | a2c_gae_atari | [a2c_gae_atari_surround_2026_02_01_132215](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_surround_2026_02_01_132215) |
| | -2.52 | ppo_atari | [ppo_atari_surround_2026_01_07_102404](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_surround_2026_01_07_102404) |
| | -9.86 | sac_atari | [sac_atari_surround_2026_02_13_042319](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_surround_2026_02_13_042319) |
| ALE/Tennis-v5 | -2873 | a2c_gae_atari | [a2c_gae_atari_tennis_2026_02_01_175829](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_tennis_2026_02_01_175829) |
| | -4.41 | ppo_atari_lam85 | [ppo_atari_lam85_tennis_2026_01_07_223532](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_tennis_2026_01_07_223532) |
| | -374 | sac_atari | [sac_atari_tennis_2026_02_13_105531](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_tennis_2026_02_13_105531) |
| ALE/TimePilot-v5 | 3376 | a2c_gae_atari | [a2c_gae_atari_timepilot_2026_02_01_175930](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_timepilot_2026_02_01_175930) |
| | 4668 | ppo_atari | [ppo_atari_timepilot_2026_01_07_101010](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_timepilot_2026_01_07_101010) |
| | 3003 | sac_atari | [sac_atari_timepilot_2026_02_13_110656](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_timepilot_2026_02_13_110656) |
| ALE/Tutankham-v5 | 167 | a2c_gae_atari | [a2c_gae_atari_tutankham_2026_02_01_132347](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_tutankham_2026_02_01_132347) |
| | 217 | ppo_atari_lam85 | [ppo_atari_lam85_tutankham_2026_01_08_095251](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam85_tutankham_2026_01_08_095251) |
| | 126 | sac_atari | [sac_atari_tutankham_2026_02_13_105535](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_tutankham_2026_02_13_105535) |
| ALE/UpNDown-v5 | 57099 | a2c_gae_atari | [a2c_gae_atari_upndown_2026_02_01_132435](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_upndown_2026_02_01_132435) |
| | 182472 | ppo_atari | [ppo_atari_upndown_2026_01_07_105708](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_upndown_2026_01_07_105708) |
| | 3450 | sac_atari | [sac_atari_upndown_2026_02_13_104624](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_upndown_2026_02_13_104624) |
| ALE/VideoPinball-v5 | 25310 | a2c_gae_atari | [a2c_gae_atari_videopinball_2026_02_01_083457](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_videopinball_2026_02_01_083457) |
| | 56746 | ppo_atari_lam70 | [ppo_atari_lam70_videopinball_2026_01_07_224359](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_lam70_videopinball_2026_01_07_224359) |
| | 22541 | sac_atari | [sac_atari_videopinball_2026_02_13_222427](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_videopinball_2026_02_13_222427) |
| ALE/WizardOfWor-v5 | 2682 | a2c_gae_atari | [a2c_gae_atari_wizardofwor_2026_02_01_132449](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_wizardofwor_2026_02_01_132449) |
| | 5814 | ppo_atari | [ppo_atari_wizardofwor_2026_01_06_221154](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_wizardofwor_2026_01_06_221154) |
| | 1160 | sac_atari | [sac_atari_wizardofwor_2026_02_13_111635](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_wizardofwor_2026_02_13_111635) |
| ALE/YarsRevenge-v5 | 24371 | a2c_gae_atari | [a2c_gae_atari_yarsrevenge_2026_02_01_132224](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_yarsrevenge_2026_02_01_132224) |
| | 17120 | ppo_atari | [ppo_atari_yarsrevenge_2026_01_06_221154](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_yarsrevenge_2026_01_06_221154) |
| | 13429 | sac_atari | [sac_atari_yarsrevenge_2026_02_13_223033](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_yarsrevenge_2026_02_13_223033) |
| ALE/Zaxxon-v5 | 29.46 | a2c_gae_atari | [a2c_gae_atari_zaxxon_2026_02_01_131758](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/a2c_gae_atari_zaxxon_2026_02_01_131758) |
| | 10756 | ppo_atari | [ppo_atari_zaxxon_2026_01_06_221154](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/ppo_atari_zaxxon_2026_01_06_221154) |
| | 3453 | sac_atari | [sac_atari_zaxxon_2026_02_13_221310](https://huggingface.co/datasets/SLM-Lab/benchmark/tree/main/data/sac_atari_zaxxon_2026_02_13_221310) |

**Training Curves** (A2C vs PPO vs SAC):

| | | |
|:---:|:---:|:---:|
| ![AirRaid](./plots/AirRaid_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Alien](./plots/Alien_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Amidar](./plots/Amidar_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Assault](./plots/Assault_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Asterix](./plots/Asterix_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Asteroids](./plots/Asteroids_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Atlantis](./plots/Atlantis_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![BankHeist](./plots/BankHeist_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![BattleZone](./plots/BattleZone_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![BeamRider](./plots/BeamRider_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Berzerk](./plots/Berzerk_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Bowling](./plots/Bowling_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Boxing](./plots/Boxing_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Breakout](./plots/Breakout_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Carnival](./plots/Carnival_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Centipede](./plots/Centipede_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![ChopperCommand](./plots/ChopperCommand_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![CrazyClimber](./plots/CrazyClimber_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Defender](./plots/Defender_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![DemonAttack](./plots/DemonAttack_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![DoubleDunk](./plots/DoubleDunk_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![ElevatorAction](./plots/ElevatorAction_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Enduro](./plots/Enduro_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![FishingDerby](./plots/FishingDerby_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Freeway](./plots/Freeway_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Frostbite](./plots/Frostbite_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Gopher](./plots/Gopher_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Gravitar](./plots/Gravitar_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Hero](./plots/Hero_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![IceHockey](./plots/IceHockey_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Jamesbond](./plots/Jamesbond_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![JourneyEscape](./plots/JourneyEscape_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Kangaroo](./plots/Kangaroo_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Krull](./plots/Krull_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![KungFuMaster](./plots/KungFuMaster_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![MsPacman](./plots/MsPacman_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![NameThisGame](./plots/NameThisGame_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Phoenix](./plots/Phoenix_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Pong](./plots/Pong_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Pooyan](./plots/Pooyan_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Qbert](./plots/Qbert_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Riverraid](./plots/Riverraid_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![RoadRunner](./plots/RoadRunner_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Robotank](./plots/Robotank_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Seaquest](./plots/Seaquest_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Skiing](./plots/Skiing_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Solaris](./plots/Solaris_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![SpaceInvaders](./plots/SpaceInvaders_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![StarGunner](./plots/StarGunner_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Surround](./plots/Surround_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Tennis](./plots/Tennis_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![TimePilot](./plots/TimePilot_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![Tutankham](./plots/Tutankham_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![UpNDown](./plots/UpNDown_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![VideoPinball](./plots/VideoPinball_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![WizardOfWor](./plots/WizardOfWor_multi_trial_graph_mean_returns_ma_vs_frames.png) | ![YarsRevenge](./plots/YarsRevenge_multi_trial_graph_mean_returns_ma_vs_frames.png) |
| ![Zaxxon](./plots/Zaxxon_multi_trial_graph_mean_returns_ma_vs_frames.png) | | |

**Skipped** (hard exploration): Adventure, MontezumaRevenge, Pitfall, PrivateEye, Venture

<details>
<summary><b>PPO Lambda Comparison</b> (click to expand)</summary>

| ENV | ppo_atari | ppo_atari_lam85 | ppo_atari_lam70 |
|-----|-----------|-----------------|-----------------|
| ALE/AirRaid-v5 | **8245** | - | - |
| ALE/Alien-v5 | **1453** | 1353 | 1274 |
| ALE/Amidar-v5 | 574 | **580** | - |
| ALE/Assault-v5 | 4059 | **4293** | 3314 |
| ALE/Asterix-v5 | 2967 | **3482** | - |
| ALE/Asteroids-v5 | 1497 | **1554** | - |
| ALE/Atlantis-v5 | **792886** | 754k | 710k |
| ALE/BankHeist-v5 | **1045** | 1045 | - |
| ALE/BattleZone-v5 | 21270 | **26383** | 13857 |
| ALE/BeamRider-v5 | **2765** | - | - |
| ALE/Berzerk-v5 | **1072** | - | - |
| ALE/Bowling-v5 | **46.45** | - | - |
| ALE/Boxing-v5 | **91.17** | - | - |
| ALE/Breakout-v5 | 191 | 292 | **327** |
| ALE/Carnival-v5 | 3071 | 3013 | **3967** |
| ALE/Centipede-v5 | 3917 | - | **4915** |
| ALE/ChopperCommand-v5 | **5355** | - | - |
| ALE/CrazyClimber-v5 | 107183 | **107370** | - |
| ALE/Defender-v5 | 37162 | - | **51439** |
| ALE/DemonAttack-v5 | 7755 | - | **16558** |
| ALE/DoubleDunk-v5 | **-2.38** | - | - |
| ALE/ElevatorAction-v5 | **5446** | 363 | 3933 |
| ALE/Enduro-v5 | 414 | **898** | 872 |
| ALE/FishingDerby-v5 | 22.80 | **27.10** | - |
| ALE/Freeway-v5 | **31.30** | - | - |
| ALE/Frostbite-v5 | **301** | 275 | 267 |
| ALE/Gopher-v5 | 4172 | - | **6508** |
| ALE/Gravitar-v5 | **599** | 253 | 145 |
| ALE/Hero-v5 | 21052 | **28238** | - |
| ALE/IceHockey-v5 | **-3.93** | -5.58 | -7.36 |
| ALE/Jamesbond-v5 | **662** | - | - |
| ALE/JourneyEscape-v5 | -1582 | **-1252** | -1547 |
| ALE/Kangaroo-v5 | 2623 | **9912** | - |
| ALE/Krull-v5 | **7841** | - | - |
| ALE/KungFuMaster-v5 | 18973 | 28334 | **29068** |
| ALE/MsPacman-v5 | 2308 | **2372** | 2297 |
| ALE/NameThisGame-v5 | **5993** | - | - |
| ALE/Phoenix-v5 | 7940 | - | **15659** |
| ALE/Pong-v5 | 15.01 | **16.91** | 12.85 |
| ALE/Pooyan-v5 | 4704 | - | **5716** |
| ALE/Qbert-v5 | **15094** | - | - |
| ALE/Riverraid-v5 | 7319 | **9428** | - |
| ALE/RoadRunner-v5 | 24204 | **37015** | - |
| ALE/Robotank-v5 | **20.07** | 8.24 | 2.59 |
| ALE/Seaquest-v5 | 1796 | - | **2010** |
| ALE/Skiing-v5 | **-19340** | -22980 | -29975 |
| ALE/Solaris-v5 | **2094** | 1803 | - |
| ALE/SpaceInvaders-v5 | **726** | - | - |
| ALE/StarGunner-v5 | 31862 | - | **47495** |
| ALE/Surround-v5 | **-2.52** | - | -6.79 |
| ALE/Tennis-v5 | -7.66 | **-4.41** | -374 |
| ALE/TimePilot-v5 | **4668** | 3003 | - |
| ALE/Tutankham-v5 | 203 | **217** | 126 |
| ALE/UpNDown-v5 | **182472** | 3450 | - |
| ALE/VideoPinball-v5 | 31385 | 22541 | **56746** |
| ALE/WizardOfWor-v5 | **5814** | 1160 | 4740 |
| ALE/YarsRevenge-v5 | **17120** | 13429 | - |
| ALE/Zaxxon-v5 | **10756** | 3453 | - |

**Legend**: **Bold** = Best score | - = Not tested

</details>

---