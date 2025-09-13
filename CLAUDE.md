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
# PPO Continuous
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_cont.json ppo_bipedalwalker train
# PPO Atari
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_pong.json ppo_pong train

# Variable substitution for specs with ${var} placeholders
slm-lab --set env=ALE/Breakout-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari dev
slm-lab -s env=HalfCheetah-v4 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco dev
```

## TODO

- [ ] test mujoco first to have final clear
- [ ] write to Huggingface
- [ ] **Start comprehensive benchmark**: Classic, Box2D, and MuJoCo envs with PPO, DQN, SAC
- [ ] **Extended Gymnasium Support**: Explore new gymnasium environments
- [ ] **Documentation Updates**: Update gitbook with new performance optimizations
