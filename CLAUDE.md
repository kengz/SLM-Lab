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
- **Structure**: Functions under ~20 lines, imports at module top, avoid deep nesting
- **Refactoring**: Maintain obsessive cleanliness - refactor immediately, remove dead code aggressively
- **Commits**: Angular convention (`feat:`, `fix:`, `docs:`, etc.)
- **Versioning**: Semantic versioning (SemVer)

## Notes for Claude Code Assistant

When working on this project:

1. **Use `uv` commands** instead of pip/python for package management
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

### ✅ COMPLETED: Performance Bottleneck Investigation

**Status**: COMPLETED - Comprehensive analysis identified major bottlenecks

**Key Findings**:

- **PPO**: ~5,200 FPS | **DQN**: ~600-700 FPS (**8.5x slower**)
- **Root Cause**: Fundamental training frequency difference (PPO: every 128 steps, DQN: every step)

**Critical Bottlenecks Identified**:

1. **🔴 MAJOR: Training Architecture Mismatch**

   - Location: Algorithm training frequency settings
   - Issue: DQN trains 128x more frequently than PPO
   - Impact: Prevents resource utilization optimization

2. **🟡 HIGH: Memory Sampling Inefficiency**

   - Location: `slm_lab/lib/util.py:batch_get()`
   - Issue: `operator.itemgetter(*idxs)(arr)` for list operations
   - Impact: O(n) memory access vs vectorized operations

3. **🟡 MEDIUM: GPU-CPU Transfers**
   - Location: `slm_lab/agent/algorithm/dqn.py:108,212`
   - Issue: `.detach().abs().cpu().numpy()` for PER
   - Impact: Unnecessary device transfers

**Profiling Infrastructure Created**:

- ✅ Real-time resource monitoring (CPU, memory, GPU)
- ✅ Timing breakdown (forward/backward/env/memory operations)
- ✅ Plotly visualization suite with interactive dashboards
- ✅ Automatic bottleneck detection and analysis
- ✅ Integrated in training loop with `PROFILE=true` flag

**Next Immediate Actions**:

- [ ] **PRIORITY 1**: Optimize `batch_get()` to use vectorized numpy indexing
- [ ] **PRIORITY 2**: Implement mini-batch training for DQN to reduce training frequency
- [ ] **PRIORITY 3**: Test performance improvements with profiler validation

### Current TODO Items

1. with @profile stable, start adding it to the identified bottlenecks above to learn more - but also add them generically instead of ad hoc at algo-specific places. fix identified bottleneck as documented above. verify before and after with @profile.

2. clean up control loop, esp run_rl in control.py - a lot of things seems extraneous or can be absorbed better into logic. run_rl should be conceptually very simple and close to the concept of an RL loop

3. just retune ppo for pong. or try a2c to see of solved then it is a PPO only problem. try breakout too.
4. check data/ file output still a lot of things and might be too big. cleanup too

- [ ] **Start benchmark on classic, box2d, and mujoco envs** - with core algos - PPO, DQN, SAC
- [ ] **Fix ALE convergence issue**: Start with PPO on Pong. it's not converging; learning is stuck
- [ ] **Extended Gymnasium Support**: Explore new gymnasium environments (https://farama.org/projects)
- [ ] **Add Huggingface support to upload benchmark data** - but first reduce each data/ output, e.g. how many model checkpoints we save
- [ ] **Run full algos-envs benchmark**
- [ ] **RNN Sequence Input Optimization**: Enhance RecurrentNet for proper batch_size×seq_len×input_dim handling
- [ ] **Ray/Optuna Integration**: Fix hyperparam search, use Optuna for better search than grid/random
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
