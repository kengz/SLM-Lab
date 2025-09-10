# SLM-Lab: Modular Deep Reinforcement Learning Framework

## Project Overview

Modular deep reinforcement learning framework in PyTorch. Originally designed for comprehensive RL experimentation with flexible algorithm implementations and environment support. Currently being migrated to modern dependencies (gymnasium, latest PyTorch, etc.).

## Development Environment

### Cloud Compute

- **Use dstack** for GPU-intensive training and development
- Setup: Follow [dstack documentation](https://dstack.ai/docs/)
- Run: `dstack apply -f .dstack/workflows/<file>.yml`

## Code Standards

- **Package Management**: Always use `uv` instead of pip/python (`uv add package-name`, `uv run script.py`)
- **Naming**: Clear, searchable names with consistent patterns (`*_config`, `*_wrapper`, `*_callback`)
- **Type Hints**: Native Python types (`list[str]`, `dict[str, float]`, `str | None`)
- **Docstrings**: Brief and informative - rely on type hints and clear naming
- **Style**: Concise functions, no deep indents, always import at module top, no defensive coding
- **Refactoring**: Maintain obsessive cleanliness - refactor immediately, remove dead code aggressively
- **Commits**: Angular convention (`feat:`, `fix:`, `docs:`, etc.)
- **Versioning**: Semantic versioning (SemVer)

## Notes for Claude Code Assistant

When working on this project:

1. **Use `uv`** instead of base python/pip
2. **Use dstack** for compute-intensive operations
3. **Follow Angular commit convention** and semantic versioning
4. **Use TODO section in CLAUDE.md** to organize and track work
5. **On task completion**: cleanup code, test, update docs, then commit
6. **Stage changes frequently** - commit related work as logical units
7. **Keep suggestions flexible** - project structure will evolve
8. **Never hard reset or delete work** - preserve changes even during corruption/errors

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

### Migration Status

Currently migrating from:

- `gym` → `gymnasium` (new API with `terminated`/`truncated`)
- `roboschool` → removed (deprecated)
- Atari environments → `ale-py` under gymnasium
- PyTorch updates → modern optimizers and schedulers

## How to Run SLM-Lab

```bash
# Basic usage
uv tool install --editable .          # Install first
slm-lab                               # CartPole demo
slm-lab --render                      # with rendering
slm-lab spec.json spec_name dev       # custom experiment
slm-lab --job job.json                # batch experiments

# Modes: dev (debug), train (fast), train@latest (resume), enjoy@session (replay)

# Performance profiling (for GPU bottleneck analysis)
slm-lab --profile=true spec.json spec_name train
tensorboard --logdir=data/profiler_logs  # View results
```

## Migration Status

✅ **Framework Migration Complete** - See `MIGRATION_CHANGELOG.md` for detailed progress tracking.

**Key achievements:**

- Complete gymnasium migration with 1600-2000 FPS performance
- Universal action shape compatibility (8 environment type combinations)
- Centralized performance optimizations with 18% CPU improvement
- Modern toolchain: uv, dstack GPU, PyTorch 2.8.0, ALE-py 0.11.2
- Professional git history suitable for production deployment

## Current Performance Features

### Automatic Optimizations (`--optimize-perf=true` by default)

- **CPU Threading**: Uses all cores (up to 32) with intelligent platform detection
- **lightning thunder**: Auto-enabled on compatible GPUs (Ampere+ compute 8.0+)
- **GPU Optimizations**: TF32 acceleration, cuDNN benchmark, memory management
- **Universal Support**: Apple Silicon M1/M2, Intel, AMD, ARM64, x86_64

### Profiling & Debugging

```bash
# Profile performance bottlenecks
slm-lab --profile=true spec.json spec_name train
tensorboard --logdir=data/profiler_logs

# Disable optimizations for debugging
slm-lab --optimize-perf=false spec.json spec_name dev
```

## TODO

### Active TODO Items

**PRIORITY 1: Memory & Batch Optimization** (Target: 15-25% FPS improvement)

- [ ] **Implement tensor buffer pooling** to reduce memory allocations
- [ ] **Optimize batch processing** with larger effective batch sizes
- [ ] **Vectorize memory sampling operations** across environments

**PRIORITY 2: Environment & Integration Issues**

- [ ] **Fix ALE convergence issue**: PPO on Pong not converging; try A2C/other algorithms
- [ ] **Implement adaptive training frequency** based on environment complexity

**PRIORITY 3: Performance & Quality**

- [ ] **Clean up data/ output**: Reduce file sizes and checkpoint frequency
- [ ] **Start comprehensive benchmark**: Classic, Box2D, and MuJoCo envs with PPO, DQN, SAC
- [ ] **Extended Gymnasium Support**: Explore new gymnasium environments
- [ ] **RNN Sequence Input Optimization**: Enhance RecurrentNet for proper batch×seq×input handling
- [ ] **Documentation Updates**: Update gitbook with new performance optimizations

### Command to Test Current State

```bash
# ✅ Validated algorithms (confirmed working)
uv run slm-lab slm_lab/spec/demo.json dqn_cartpole train                                    # DQN CartPole
uv run slm-lab slm_lab/spec/benchmark/reinforce/reinforce_cartpole.json reinforce_cartpole train  # REINFORCE
uv run slm-lab slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json ddqn_per_concat_lunar train   # DDQN PER
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_shared_cartpole train      # PPO CartPole
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_cont.json ppo_bipedalwalker train            # PPO Continuous
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_pong.json ppo_pong train                     # PPO Atari

# Development/debugging (faster runs)
uv run slm-lab slm_lab/spec/demo.json dqn_cartpole dev
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_shared_cartpole dev
```
