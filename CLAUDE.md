# SLM-Lab: Modular Deep Reinforcement Learning Framework

## Project Overview

Modular deep reinforcement learning framework in PyTorch. Originally designed for comprehensive RL experimentation with flexible algorithm implementations and environment support. Currently being migrated to modern dependencies (gymnasium, latest PyTorch, etc.).

## Development Environment

### Package Management

- **Always use `uv`** instead of pip/python
- Installation: `uv add package-name`
- Running scripts: `uv run script.py`

### Cloud Compute

- **Use dstack** for GPU-intensive training and development
- Setup: Follow [dstack documentation](https://dstack.ai/docs/)
- Run: `dstack apply -f .dstack/workflows/<file>.yml`

## Code Standards

### Style Guide

- **Naming**: Concise and globally consistent for searchability
  - Use clear, searchable names: `train_ppo()` not `tp()`
  - Consistent patterns: `*_config`, `*_wrapper`, `*_callback`
- **Type Hints**: Native Python types preferred, keep simple
  - `list[str]` over `List[str]`
  - `dict[str, float]` over `Dict[str, float]`
  - Simple unions: `str | None` over `Optional[str]`
- **Code Complexity**: Prioritize readability and brevity
  - Short functions (max ~20 lines)
  - Avoid nested complexity - extract to helper functions
  - Clear variable names over comments when possible
- **Refactoring**: Be obsessively clean
  - Refactor immediately when code feels complex
  - Remove dead code aggressively
  - Simplify whenever possible - shorter is usually better

### SLM-Lab Best Practices

- Use gymnasium (not gym) for all environments
- Follow new gymnasium API: `reset()` returns `(obs, info)`, `step()` returns `(obs, reward, terminated, truncated, info)`
- Handle both discrete and continuous action spaces
- Use vectorized environments for parallel training
- Maintain modular algorithm structure with separate agent/network/memory components

## Version Management

- **Versioning**: Follow semantic versioning (SemVer)
- **Commits**: Use Angular commit convention (`feat:`, `fix:`, `docs:`, etc.)

## Notes for Claude Code Assistant

When working on this project:

1. **Always suggest `uv` commands** instead of pip/python
2. **Use dstack** for compute-intensive suggestions
3. **Expect project structure to evolve** - keep suggestions flexible
4. **Follow Angular commit convention** for all commits
5. **Use semantic versioning** for releases
6. **Use the TODO below** to organize work, check off when done
7. **When task is done, do quick tests, update TODO, update MIGRATION_CHANGELOG.md, then cleanup and commit**
8. **Refactor frequently, keep code DRY and simple**

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
- **torch.compile**: Auto-enabled on compatible GPUs (Ampere+ compute 8.0+)
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

1. generalize logging, e.g. gradnorm is nan outside of debug and should not show. algo specific like clip_eps for ppo should also log. use a general/dynamic one, e.g. algo_vars collection = [clip_eps, entropy_coef]
2. just retune ppo for pong. or try a2c to see of solved then it is a PPO only problem. try breakout too.
3. run with profiler to debug bottleneck. now GPU util still low and with frequent drops
4. bottleneck - check where is util slowing down.. is it training, or inference (check for loops), or loss calculations, or env stepping?
5. check data/ file output still a lot of things and might be too big. cleanup too

- [ ] **Atari Production Testing**: Full Pong training run with dstack GPU infrastructure
- [ ] **Extended Gymnasium Support**: Explore new gymnasium environments (https://farama.org/projects)
- [ ] **RNN Sequence Input Optimization**: Enhance RecurrentNet for proper batch_size×seq_len×input_dim handling
- [ ] **Comprehensive Benchmarking**: Measure actual speedup gains from torch.compile and vectorization
- [ ] **Ray/Optuna Integration**: Modern hyperparameter search with Optuna backend
- [ ] **Documentation Updates**: Update gitbook documentation reflecting new API and performance

### Command to Test Current State

```bash
# Basic functionality tests
uv run slm-lab slm_lab/spec/demo.json dqn_cartpole dev
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_shared_cartpole dev
uv run slm-lab slm_lab/spec/benchmark/a2c/a2c_gae_cartpole.json a2c_gae_cartpole dev

# Retrospective analysis (the only custom CLI we kept)
uv run slm-retro data/experiment_dir

# Performance optimizations (automatic on GPU, manual on CPU)
uv run slm-lab --torch-compile=true slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_shared_cartpole train
```
