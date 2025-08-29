# Performance Comparison: Before vs After Optimizations

## Methodology

This document compares performance (measured in FPS after training has plateaued) between the SLM-Lab framework before and after comprehensive optimizations.

### Commits Compared

- **Before**: `cae945a2` - Last stable commit before gymnasium migration and performance optimizations
- **After**: `c22d03e9` - Current commit with all optimizations complete

### Test Environment

- Machine: MacBook Pro (M1 Pro, 16GB RAM)
- Python: 3.12.8 (uv managed)
- PyTorch: 2.8.0.dev20241013+cpu (from pyproject.toml)
- Test duration: Run until FPS plateaus (typically 30-90 seconds)

### Algorithms Tested

1. **Random** - Baseline to measure pure environment performance
2. **REINFORCE** - Simple policy gradient
3. **DQN** - Deep Q-Network
4. **A2C** - Advantage Actor-Critic
5. **PPO** - Proximal Policy Optimization
6. **SAC** - Soft Actor-Critic

### Environments Tested

1. **CartPole-v1** - Discrete, simple (2D state)
2. **LunarLander-v2** - Discrete, complex (8D state)
3. **Pendulum-v1** - Continuous, simple (3D state)
4. **BipedalWalker-v3** - Multi-continuous, complex (24D state)
5. **PongNoFrameskip-v4** - Discrete, image (84x84x1 after preprocessing)

### Environment Configurations

- **Single Environment**: 1 environment
- **Small Vector**: 8 environments 
- **Large Vector**: 16 environments

## Results

### Test Settings

- **Hardware**: MacBook Air M2 (2022), 8 CPUs, CPU only (no GPU)
- **Vectorization**: Smart environment-aware selection (see below)
- **Before Commit**: `cae945a2` (baseline before optimizations)
- **After Commit**: `14b7373c` (gymnasium migration + PyTorch optimizations + smart vectorization)

### Vectorization Logic (After Commit)

The framework now uses intelligent vectorization mode selection based on environment complexity:

- **Simple Environments** (classic_control, box2d): Always sync vectorization
- **Complex Environments** (MuJoCo, Atari): async for 8+ envs, sync otherwise
- **Detection Method**: Uses gymnasium's `entry_point` metadata for robust classification
- **Manual Override**: Can be set via `vectorization_mode` parameter in env specs

## Benchmark Instructions

### Quick Reference Commands

#### 1. Update num_envs in random.json
```bash
# Set to 1 environment
sed -i '' 's/"num_envs": [0-9]\+/"num_envs": 1/g' slm_lab/spec/experimental/misc/random.json

# Set to 8 environments  
sed -i '' 's/"num_envs": [0-9]\+/"num_envs": 8/g' slm_lab/spec/experimental/misc/random.json

# Set to 16 environments
sed -i '' 's/"num_envs": [0-9]\+/"num_envs": 16/g' slm_lab/spec/experimental/misc/random.json
```

#### 2. Run Each Environment
```bash
# CartPole-v1
uv run slm-lab slm_lab/spec/experimental/misc/random.json random_cartpole train

# LunarLander-v3
uv run slm-lab slm_lab/spec/experimental/misc/random.json random_lunarlander train

# Pendulum-v1
uv run slm-lab slm_lab/spec/experimental/misc/random.json random_pendulum train

# BipedalWalker-v3
uv run slm-lab slm_lab/spec/experimental/misc/random.json random_bipedalwalker train

# PongNoFrameskip-v4
uv run slm-lab slm_lab/spec/experimental/misc/random.json random_pong train
```

#### 3. Update Performance Table
Look for final stable FPS in last few log lines:
```
fps:30000            total_reward:...
```

Update table below with measured FPS values.

#### Systematic Process
1. Set num_envs to 1
2. Run all 5 environments, record FPS
3. Set num_envs to 8  
4. Run all 5 environments, record FPS
5. Set num_envs to 16
6. Run all 5 environments, record FPS

Each test takes ~30 seconds to 2 minutes to complete.

## Performance Benchmark Results

### Random Algorithm

**Note**: Previous measurements used old vectorization logic (sync ≤8, async >8). 
Re-running with new smart vectorization for comparison.

| Environment | num_envs | FPS Before | FPS After | Status |
|-------------|----------|------------|-----------|---------|
| CartPole-v1 | 1 | 60000 | 65000 | ✅ Complete |
| CartPole-v1 | 8 | 120000 | 130000 | ✅ Complete |
| CartPole-v1 | 16 | 44000 | 170000 | ✅ Complete |
| LunarLander-v3 | 1 | 28000 | 30000 | ✅ Complete |
| LunarLander-v3 | 8 | 37000 | 40000 | ✅ Complete |
| LunarLander-v3 | 16 | 33000 | 40000 | ✅ Complete |
| Pendulum-v1 | 1 | 28000 | 30000 | ✅ Complete |
| Pendulum-v1 | 8 | 42000 | 45000 | ✅ Complete |
| Pendulum-v1 | 16 | 25000 | 48000 | ✅ Complete |
| BipedalWalker-v3 | 1 | 8000 | 8500 | ✅ Complete |
| BipedalWalker-v3 | 8 | 8500 | 9000 | ✅ Complete |
| BipedalWalker-v3 | 16 | 7000 | 9500 | ✅ Complete |
| PongNoFrameskip-v4 | 1 | 12800 | 12800 | ✅ Complete |
| PongNoFrameskip-v4 | 8 | 12800 | 18500 | ✅ Complete |
| PongNoFrameskip-v4 | 16 | 17900 | 16700 | ✅ Complete |

**New Vectorization Logic**:
- **Simple Environments** (classic_control, box2d): Always sync vectorization for optimal performance
- **Complex Environments** (Atari, MuJoCo): Async for 8+ envs, sync otherwise

**Key Improvements**: The new logic provides significant performance gains for simple environments:
- **CartPole-v1 (16 envs)**: 44k → 170k FPS (+286% improvement - now uses sync)
- **Pendulum-v1 (16 envs)**: 25k → 48k FPS (+92% improvement - now uses sync)  
- **LunarLander-v3 (16 envs)**: 33k → 40k FPS (+21% improvement - now uses sync)
- **BipedalWalker-v3 (16 envs)**: 7k → 9.5k FPS (+36% improvement - now uses sync)
- **PongNoFrameskip-v4 (8 envs)**: 12.8k → 18.5k FPS (+44% improvement with async vectorization)
- **Overall**: Simple environments avoid async overhead, complex environments use async when beneficial

## Strategic Algorithm Performance Comparison

**Testing Matrix**: 3 challenging environments × 4 algorithms = 11 measurements
- **LunarLander-v3**: Complex discrete (8D state, 4 actions)
- **BipedalWalker-v3**: Complex continuous (24D state, 4D continuous actions)  
- **ALE/Pong-v5**: Atari/image processing (requires GPU via dstack)

### DQN Algorithm (Value-based)

| Environment | num_envs | FPS Before | FPS After | Platform | Status |
|-------------|----------|------------|-----------|----------|---------|
| LunarLander-v3 | 8 | TBD | TBD | CPU | ⏳ Needs measurement |
| ALE/Pong-v5 | 8 | TBD | TBD | GPU (dstack) | ⏳ Needs measurement |

### A2C Algorithm (On-policy policy gradient)

| Environment | num_envs | FPS Before | FPS After | Platform | Status |
|-------------|----------|------------|-----------|----------|---------|
| LunarLander-v3 | 8 | TBD | TBD | CPU | ⏳ Needs measurement |
| BipedalWalker-v3 | 8 | TBD | TBD | CPU | ⏳ Needs measurement |
| ALE/Pong-v5 | 8 | TBD | TBD | GPU (dstack) | ⏳ Needs measurement |

### PPO Algorithm (Advanced policy gradient)

| Environment | num_envs | FPS Before | FPS After | Platform | Status |
|-------------|----------|------------|-----------|----------|---------|
| LunarLander-v3 | 8 | TBD | TBD | CPU | ⏳ Needs measurement |
| BipedalWalker-v3 | 8 | TBD | TBD | CPU | ⏳ Needs measurement |
| ALE/Pong-v5 | 8 | TBD | TBD | GPU (dstack) | ⏳ Needs measurement |

### SAC Algorithm (Off-policy continuous)

| Environment | num_envs | FPS Before | FPS After | Platform | Status |
|-------------|----------|------------|-----------|----------|---------|
| BipedalWalker-v3 | 8 | TBD | TBD | CPU | ⏳ Needs measurement |

## Test Commands for Strategic Algorithm Comparison

### CPU Tests (Local)
```bash
# DQN - LunarLander
uv run slm-lab slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json ddqn_per_concat_lunar train

# A2C - LunarLander  
uv run slm-lab slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json a2c_gae_lunar train

# A2C - BipedalWalker
uv run slm-lab slm_lab/spec/benchmark/a2c/a2c_gae_cont.json a2c_gae_bipedalwalker train

# PPO - LunarLander
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_lunar.json ppo_lunar train

# PPO - BipedalWalker  
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_cont.json ppo_bipedalwalker train

# SAC - BipedalWalker
uv run slm-lab slm_lab/spec/benchmark/sac/sac_cont.json sac_bipedalwalker train
```

### GPU Tests (dstack) - Pong Atari
```bash
# DQN - Pong
uv run slm-lab slm_lab/spec/benchmark/dqn/dqn_pong.json dqn_pong train

# A2C - Pong  
uv run slm-lab slm_lab/spec/benchmark/a2c/a2c_gae_pong.json a2c_gae_pong train

# PPO - Pong
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_pong.json ppo_pong train
```

**Note**: Pong environments require GPU for efficient image processing. Use dstack for these tests.

