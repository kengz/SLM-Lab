# SLM-Lab: Modular Deep Reinforcement Learning Framework

## Project Overview

Modular deep reinforcement learning framework in PyTorch. Originally designed for comprehensive RL experimentation with flexible algorithm implementations and environment support. Currently being migrated to modern dependencies (gymnasium, latest PyTorch, etc.).

## Development Environment

### Cloud Compute

- **Use dstack** for GPU-intensive training and development
- Setup: Follow [dstack documentation](https://dstack.ai/docs/)
- Run: `slm-lab spec.json spec_name train --dstack run-name`
- **Customize hardware**: Edit `.dstack/run.yml` to change GPU type, CPU count, or backends

## Code Standards

- **Package Management**: Always use `uv` instead of pip/python (`uv add package-name`, `uv run script.py`), rely on pyproject.toml
- **Style**: DRY & KISS principles - code should be concise and simple to read and understand; avoid deep indents, avoid in-method imports, avoid defensive coding
- **Naming**: Short and obvious names, globally consistent for easy search
- **Type Hints**: Native Python types (`list[str]`, `dict[str, float]`, `str | None`)
- **Docstrings**: Concise and informative - rely on clear naming and type hints
- **Refactoring**: Maintain obsessive cleanliness - refactor immediately, remove dead code aggressively
- **Commits**: Angular convention (`feat:`, `fix:`, `docs:`, etc.)
- **Versioning**: Semantic versioning (SemVer)

## Notes for Claude Code Assistant

When working on this project:

1. **Use TODO section in instruction** to plan and do work autonomously
2. **Stage changes frequently** - commit related work as logical units
3. **Never hard reset or delete work** - preserve changes even during corruption/errors
4. **On task completion**: cleanup code, test, update docs, then commit
5. **Document major changes** in `MIGRATION_CHANGELOG.md`
6. **Use Serena MCP** for efficient work

## Framework Design Patterns

### SLM-Lab Architecture

SLM-Lab follows a modular design pattern with these core components:

1. **Agent** (`slm_lab/agent/`) - Algorithm implementations (A2C, PPO, SAC, etc.)
2. **Environment** (`slm_lab/env/`) - Environment wrappers and utilities
3. **Networks** (`slm_lab/agent/net/`) - Neural network architectures
4. **Memory** (`slm_lab/agent/memory/`) - Experience replay and storage
5. **Experiment** (`slm_lab/experiment/`) - Training loop and search utilities
6. **Spec System** (`slm_lab/spec/`) - JSON configuration for reproducible experiments

### Key Components

- **Environment wrappers**: Support for OpenAI/gymnasium, Unity, VizDoom
- **Algorithm diversity**: DQN, A2C, PPO, SAC, and variants
- **Network types**: MLP, ConvNet, RNN with flexible architectures
- **Memory systems**: Experience replay, prioritized replay
- **Experiment management**: Hyperparameter search, distributed training

## How to Run SLM-Lab

```bash
# Basic usage
uv tool install --editable .          # Install first
slm-lab --help                        # help menu
slm-lab                               # CartPole demo
slm-lab --render                      # with rendering
slm-lab spec.json spec_name dev       # custom experiment
slm-lab --job job.json                # batch experiments

# ✅ Validated algorithms (confirmed working)
# DQN CartPole
uv run slm-lab slm_lab/spec/demo.json dqn_cartpole train
# REINFORCE
uv run slm-lab slm_lab/spec/benchmark/reinforce/reinforce_cartpole.json reinforce_cartpole train
# DDQN PER
uv run slm-lab slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json ddqn_per_concat_lunar train
# PPO CartPole
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_shared_cartpole train
# PPO Lunar
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_lunar.json ppo_lunar train
# PPO Continuous
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_cont.json ppo_bipedalwalker train
# PPO Atari
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_pong.json ppo_pong train

# Variable substitution for specs with ${var} placeholders
slm-lab --set env=ALE/Breakout-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari dev
slm-lab -s env=HalfCheetah-v4 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco dev
```

## ASHA Hyperparameter Search

SLM-Lab now uses **ASHA (Async Successive Halving Algorithm)** by default for efficient deep RL hyperparameter search:

### Features

- **Early termination** of poor-performing trials
- **10x more configurations** explored with same compute budget
- **Noise-robust** successive halving for volatile RL returns
- **Timestep-based** progress tracking (works with vec envs)
- **Zero configuration** - works automatically with any search spec

### Usage

**IMPORTANT**: ASHA scheduler and multi-session trials are **mutually exclusive**:

- **ASHA scheduler** (`search_scheduler` specified): Requires `max_session=1` for periodic metric reporting and early termination
- **Multi-session trials** (`max_session>1`): Must omit `search_scheduler` - trials run to completion for robust statistics

Basic search without early termination:

```json
{
  "meta": {
    "max_session": 1 // or omit search_scheduler
  },
  "search": {
    "agent": {
      "algorithm": {
        "gamma__choice": [0.9, 0.95, 0.99]
      }
    }
  }
}
```

### ASHA Early Termination (Recommended for Fast Exploration)

Enable ASHA scheduler by adding `search_scheduler` to meta:

```json
{
  "meta": {
    "max_session": 1, // REQUIRED: single-session for periodic reporting
    "search_scheduler": {
      "grace_period": 1000, // min frames before termination
      "reduction_factor": 3 // trial elimination rate
    }
  },
  "search": {
    "agent": {
      "algorithm": {
        "gamma__choice": [0.9, 0.95, 0.99]
      }
    }
  }
}
```

### Multi-Session Search (Robust Evaluation)

For high-variance environments, use multi-session WITHOUT scheduler:

```json
{
  "meta": {
    "max_session": 3 // REQUIRED: must omit search_scheduler
    // "search_scheduler": null  // must not be specified
  },
  "search": {
    "agent": {
      "algorithm": {
        "gamma__choice": [0.9, 0.95, 0.99]
      }
    }
  }
}
```

### Examples

- `slm_lab/spec/experimental/asha_search_test.json` - Simple DQN CartPole
- `slm_lab/spec/experimental/ppo_asha_example.json` - PPO CartPole with extensive search

### Run ASHA Search

```bash
# Any search spec now uses ASHA automatically
uv run slm-lab slm_lab/spec/experimental/asha_search_test.json asha_search_test search
uv run slm-lab slm_lab/spec/experimental/ppo_asha_example.json ppo_asha_cartpole search
```

### Multi-Session Behavior

**Single-session trials** (`max_session=1`):

- ✅ Support early termination via ASHA
- ✅ Periodic metric reporting at `log_frequency`
- ✅ Fast exploration of hyperparameter space
- Best for: Initial search, exploration phase

**Multi-session trials** (`max_session>1`):

- ❌ No early termination (always run to `max_frame`)
- ✅ Robust statistics across multiple runs
- ✅ Reduced variance in metric evaluation
- Best for: Final evaluation, production configs

**Recommendation**: Use `max_session=1` for search to leverage ASHA early termination, then run best configs with `max_session>1` for robust final evaluation.

## Two-Stage Hyperparameter Search Methodology

Use this proven two-stage approach for finding robust hyperparameters:

### Stage 1: Wide ASHA Exploration (Fast, High Variance)

**Goal**: Efficiently explore large search space to identify promising hyperparameter ranges

**Configuration**:
- `max_session=1` (single session, high variance but fast)
- `search_scheduler` enabled with ASHA early termination
- Wide search space with many trials (e.g., 30 trials)

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
    "agent.algorithm.lr__loguniform": [1e-5, 5e-3]
  }
}
```

**Workflow**:
1. Run wide ASHA search: `slm-lab spec.json spec_name search`
2. Analyze `experiment_df.csv` to identify top-performing trials
3. Look for patterns in successful hyperparameter combinations
4. Define narrower search ranges around promising values

### Stage 2: Narrow Multi-Session Validation (Robust, Low Variance)

**Goal**: Validate best hyperparameter ranges with reliable averaged results

**Configuration**:
- `max_session=4` (multi-session averaging, low variance, reliable)
- **NO** `search_scheduler` (must be omitted or null)
- Narrow search space focused on promising ranges (e.g., 8-12 trials)

**Example**:
```json
{
  "meta": {
    "max_session": 4,
    "max_trial": 8
    // search_scheduler MUST be omitted
  },
  "search": {
    "agent.algorithm.gamma__choice": [0.97, 0.98, 0.99],
    "agent.algorithm.lr__choice": [0.0001, 0.0003]
  }
}
```

**Workflow**:
1. Create narrowed spec based on Stage 1 patterns
2. Run multi-session search: `slm-lab spec_narrow.json spec_name search`
3. Select best config from averaged `total_reward_ma` across 4 sessions
4. Update spec with winning hyperparameters as defaults
5. Keep `search` section in spec for reference/documentation

**Key Principle**: Stage 1 identifies promising ranges quickly (single session, ASHA). Stage 2 validates with reliable statistics (multi-session, no early termination). Never use ASHA with multi-session.

## TODO

- [ ] Ray Tune graph has no plots. also experiment df tries to report the now-nonstandard columns like efficiency etc - which arent calculated? check

- [ ] **Two-Stage Hyperparameter Search**:

  - [ ] **PPO CartPole**: Comprehensive optimization
    - Stage 1 (ASHA): `slm-lab slm_lab/spec/benchmark/ppo/ppo_cartpole_search1.json ppo_cartpole_search1 search`
      - 200 trials, max_frame=200k, ASHA early termination
      - Search: gamma, lam, clip_eps, entropy_coef, val_loss_coef, time_horizon, minibatch_size, training_epoch, loss_spec, actor_lr, critic_lr
      - After completion: analyze `experiment_df.csv` to identify top hyperparameter ranges
    - Stage 2 (Multi-session): Create `ppo_cartpole_search2.json` based on Stage 1 results
      - ~12 trials, 4 sessions, narrowed ranges
      - Validates best hyperparameters with robust averaged statistics

  - [ ] **PPO Continuous Control (dstack)**:

    ```bash
    # MuJoCo requires more compute - use dstack
    slm-lab slm_lab/spec/benchmark/ppo/ppo_cont.json ppo_bipedalwalker search --dstack ppo-bipedal-search
    slm-lab slm_lab/spec/benchmark/ppo/ppo_cont.json ppo_pendulum search --dstack ppo-pendulum-search
    slm-lab --set env=HalfCheetah-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search --dstack ppo-halfcheetah-search
    slm-lab --set env=Ant-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search --dstack ppo-ant-search
    slm-lab --set env=Hopper-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search --dstack ppo-hopper-search
    slm-lab --set env=Walker2d-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search --dstack ppo-walker-search
    slm-lab --set env=Humanoid-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search --dstack ppo-humanoid-search
    ```

  - [ ] **PPO Atari (dstack)**:

    ```bash
    # GPU-accelerated Atari training
    slm-lab --set env=ALE/Pong-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari search --dstack ppo-pong-search
    slm-lab --set env=ALE/Breakout-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari search --dstack ppo-breakout-search
    slm-lab --set env=ALE/Qbert-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari search --dstack ppo-qbert-search
    slm-lab --set env=ALE/Seaquest-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari search --dstack ppo-seaquest-search
    ```

  - [ ] **DQN Atari (dstack)**:

    ```bash
    # DQN variants for discrete action Atari
    slm-lab slm_lab/spec/benchmark/dqn/dqn_breakout.json dqn_breakout search --dstack dqn-breakout-search
    slm-lab slm_lab/spec/benchmark/dqn/dqn_pong.json dqn_pong search --dstack dqn-pong-search
    ```

  - [ ] **SAC Continuous Control (dstack)**:
    ```bash
    # SAC for continuous control benchmarks
    slm-lab --set env=HalfCheetah-v5 slm_lab/spec/benchmark/sac/sac_mujoco.json sac_mujoco search --dstack sac-halfcheetah-search
    slm-lab --set env=Ant-v5 slm_lab/spec/benchmark/sac/sac_mujoco.json sac_mujoco search --dstack sac-ant-search
    ```

- [ ] **Extended Gymnasium Support**: Explore new gymnasium environments
- [ ] **Documentation Updates**: Update gitbook with new performance optimizations
