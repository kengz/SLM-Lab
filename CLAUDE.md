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

### Option 1: Installed Tool (Recommended)
```bash
# Install as uv tool for shorter commands
uv tool install --editable .

# Command patterns:
slm-lab                              # CartPole demo (dev mode, no rendering)
slm-lab spec.json spec_name mode     # Single experiment
slm-lab --job job/experiments.json   # Batch experiments

# Examples:
slm-lab                                                # CartPole demo
slm-lab --render                                       # CartPole demo with rendering  
slm-lab slm_lab/spec/demo.json dqn_cartpole dev       # Custom experiment
slm-lab slm_lab/spec/demo.json dqn_cartpole train     # Training mode
slm-lab --job job/experiments.json                    # Batch mode
```

### Option 2: Direct Execution (Development)
```bash
# Equivalent to slm-lab commands above
python run_lab.py                                        # CartPole demo
python run_lab.py --render                               # CartPole demo with rendering
python run_lab.py slm_lab/spec/demo.json dqn_cartpole dev   # Custom experiment  
python run_lab.py --job job/experiments.json             # Batch mode

# Or with uv run
uv run python run_lab.py slm_lab/spec/demo.json dqn_cartpole dev
```

### Execution Modes:
- `dev` - Development mode with verbose logging and debugging features
- `train` - Training mode for fastest performance (production)
- `train@{predir}` - Resume training from directory (e.g. train@latest)
- `enjoy@{session_spec_file}` - Replay trained model from session
- `search` - Hyperparameter search (uses Ray - avoid until stable)

### CLI Flags (ordered by relevance):
```bash
# Most common
--render                          # Enable environment rendering (explicit)
--job job_file.json              # Run batch experiments  

# Configuration  
--log-level=INFO|DEBUG|WARNING|ERROR  # Logging verbosity
--torch-compile=auto|true|false   # Smart torch.compile (auto=modern GPUs only)
--cuda-offset=0                   # GPU device offset

# Advanced debugging
--profile=false|true|gpu|cpu      # Performance profiling
```

### Environment Variables:
All flags have corresponding environment variables:
```bash
RENDER=true slm-lab                    # Same as --render
LOG_LEVEL=DEBUG slm-lab               # Same as --log-level=DEBUG
TORCH_COMPILE=false slm-lab           # Same as --torch-compile=false
```

# Spec files are located in slm_lab/spec/ with structure:
# {
#   "spec_name": {
#     "agent": [{...}],
#     "env": [{...}],
#     "search": [{...}]
#   }
# }

# Representative Environment Tests (run after env-related changes):
# NOTE run them with timeout of 30s if you are testing instead of waiting for completion that can take a long time.

# 1. Single discrete:
uv run slm-lab slm_lab/spec/benchmark/reinforce/reinforce_cartpole.json reinforce_cartpole train

# 2. Vector discrete:
uv run slm-lab slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json ddqn_per_concat_lunar train

# 3. Single continuous:
uv run slm-lab slm_lab/spec/cont_test.json a2c_gae_pendulum_single train
# action dim = 4
uv run slm-lab slm_lab/spec/cont_test.json a2c_gae_bipedalwalker_single train

# 4. Vector continuous:
uv run slm-lab slm_lab/spec/cont_test.json a2c_gae_pendulum train
# action dim = 4
uv run slm-lab slm_lab/spec/cont_test.json a2c_gae_bipedalwalker train

# 5. Atari check:
uv run slm-lab slm_lab/spec/benchmark/a2c/a2c_gae_pong.json a2c_gae_pong train

# 6. MuJoCo environments check:
uv run slm-lab slm_lab/spec/benchmark/sac/sac_mujoco.json sac_mujoco train

# Representative Benchmark Commands (for full training runs):

# Discrete Environments:
uv run slm-lab slm_lab/spec/benchmark/dqn/dqn_lunar.json dqn_concat_lunar train
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_lunar.json ppo_lunar train
uv run slm-lab slm_lab/spec/benchmark/sac/sac_lunar.json sac_lunar train
uv run slm-lab slm_lab/spec/benchmark/dqn/dqn_pong.json dqn_pong train
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_pong.json ppo_pong train

# Continuous Environments (MuJoCo):
uv run slm-lab slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json a2c_gae_mujoco train
uv run slm-lab slm_lab/spec/benchmark/a2c/a2c_nstep_mujoco.json a2c_nstep_mujoco train
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco train
uv run slm-lab slm_lab/spec/benchmark/sac/sac_mujoco.json sac_mujoco train
uv run slm-lab slm_lab/spec/benchmark/async_sac/async_sac_mujoco.json async_sac_mujoco train

# Humanoid (Longer Training):
uv run slm-lab slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json a2c_gae_humanoid train
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_humanoid train
uv run slm-lab slm_lab/spec/benchmark/sac/sac_mujoco.json sac_humanoid train
uv run slm-lab slm_lab/spec/benchmark/async_sac/async_sac_mujoco.json async_sac_humanoid train
```

## TODO - Migration Progress

### ✅ Migration Complete - All Major Components

**Successfully consolidated 51 development commits into 7 production-ready commits covering comprehensive framework modernization:**

- [x] **Core Framework Migration** - Modern toolchain (gymnasium, uv, dstack GPU, PyTorch 2.8.0)
- [x] **Environment System Cleanup** - Complete gymnasium migration, removed 4000+ lines of wrapper code
- [x] **Universal Action Shape Compatibility** - All 8 environment type combinations (single/vector × discrete/continuous)
- [x] **SAC Algorithm Optimization** - Target entropy fixes, action processing, GumbelSoftmax improvements
- [x] **Smart Vectorization & MuJoCo Migration** - Intelligent sync/async selection, roboschool → MuJoCo v5
- [x] **Performance & Infrastructure** - RNN restoration, loguru logging, memory optimizations
- [x] **Testing & Validation** - Atari compatibility, algorithm verification, performance measurement

**Key Achievements:**

- Removed Unity ML-Agents, VizDoom, roboschool legacy support (use external gymnasium packages)
- Achieved 1600-2000 fps with optimized sync vectorization (≤8 envs) vs async (>8 envs)
- Universal `to_action()` method handles all environment types with 15 lines vs previous 31
- Complete ALE-py 0.11.2 integration with proper ConvNet preprocessing
- Professional git history suitable for production deployment

### Current

- btw render_mode=render_mode is not a flag for gym.make_vec
1. optimization - currently the FPS is still low. when running I don't see ful GPU utilization - GPU spikes and CPU spikes - so this indicates bottleneck of back and forth waiting.
2. Also, enabling Torch compile actually runs slower overall - I ran on T4 GPU which is older so perhaps? or it's just the models are too small? but nevertheless it should not be slower right?
3. So, let's do a proper optimization - start by solving the bottleneck/wait cycles. Then look at standard speedup tricks - GPU optim. device transfer, pin_memory, compile (the slowdown issue), or use torch.profiler.profile().

#### Benchmark

`uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_pong.json ppo_pong train`

Set session to 1 to focus on benchmarking.

- 16 envs
  - disabled compile: 1200 fps
  - enabled compile: 1000 fps hmm slower
- change to 32 envs
  - disabled compile: 1300 fps
  - enabled compile: 1100 fps
- double minibatch
  - still 1300 fps
- possible performance block - the log file (not stdout) is writing too much shit - all the debug stuff
  - info only, 32 envs, no compile: 1300 fps

### 📝 Future Enhancements

- [ ] also cleanup some of the preprocessing methods? no longer used since it's in gymnasium now. but check if book needs it
- [ ] **TrackTime Environment Wrapper**: Implement timing wrapper for comprehensive performance analysis
- [ ] **Atari Production Testing**: Full Pong training run with dstack GPU infrastructure
- [ ] **Extended Gymnasium Support**: Explore new gymnasium environments (https://farama.org/projects)
- [ ] **RNN Sequence Input Optimization**: Enhance RecurrentNet for proper batch_size×seq_len×input_dim handling
- [ ] **Comprehensive Benchmarking**: Measure actual speedup gains from torch.compile and vectorization
- [ ] **Higher Parallelization**: Test performance with more vector environments (>32)
- [ ] **Numba Integration**: Explore for remaining CPU-bound numpy bottlenecks
- [ ] **Unit Test Suite**: Execute full test suite for comprehensive validation
- [ ] **Ray/Optuna Integration**: Modern hyperparameter search with Optuna backend
- [ ] **Documentation Updates**: Update gitbook documentation reflecting new API and performance
- [ ] **Production Validation**: Ensure migrated algorithms achieve expected benchmark performance

### 🚫 Deprecated & Removed

**Successfully eliminated legacy dependencies and custom implementations:**

- [x] **roboschool** → gymnasium MuJoCo v5 (8 environments migrated)
- [x] **Unity ML-Agents** → External gymnasium packages
- [x] **VizDoom** → External gymnasium VizDoom package
- [x] **pybullet_envs** → gymnasium equivalents
- [x] **Custom wrapper system** → gymnasium's optimized C++ implementations (4000+ lines removed)

### Command to Test Current State

```bash
# Basic functionality tests
uv run slm-lab slm_lab/spec/demo.json dqn_cartpole dev
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_shared_cartpole dev
uv run slm-lab slm_lab/spec/benchmark/a2c/a2c_gae_cartpole.json a2c_gae_cartpole dev

# Retrospective analysis (the only custom CLI we kept)
uv run slm-retro data/experiment_dir

# Performance optimizations (automatic on GPU, manual on CPU)
TORCH_COMPILE=true uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_shared_cartpole train
```
