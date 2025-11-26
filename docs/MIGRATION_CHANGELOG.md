# SLM-Lab Migration Changelog

Complete modernization of SLM-Lab with gymnasium migration, modern toolchain, and significant performance improvements. **Contains breaking changes**.

## 🚨 Breaking Changes

### Package Management: conda → uv
```bash
# Old (no longer works)
conda activate lab && python run_lab.py [args]

# New (required)
uv run slm-lab [args]
```

### Environment Dependencies Removed
Unity ML-Agents, VizDoom, and roboschool support removed.
- **Unity/VizDoom**: Use external [gymnasium packages](https://gymnasium.farama.org/environments/third_party_environments/)
- **roboschool**: Automatically migrated to gymnasium MuJoCo (see mapping below)

### Logging System Changed
```bash
export LOG_METRICS=true  # Required for training metrics
```

## 🔄 Environment Updates

### Atari: New ALE Integration
- **Environment names**: `PongNoFrameskip-v4` → `ALE/Pong-v5`
- **Preprocessing**: Automatic (frame skip, resize, grayscale, stacking, NoOp reset)
- Migration is automatic - no spec file changes needed

### Environment Kwargs Pass-through
Environment specs pass through arbitrary kwargs to `gym.make()`:
```json
{
  "env": {
    "name": "LunarLander-v3",
    "continuous": true,
    "num_envs": 8
  }
}
```

### Roboschool → MuJoCo Migration
| Old | New |
|-----|-----|
| RoboschoolAnt-v1 | Ant-v5 |
| RoboschoolHalfCheetah-v1 | HalfCheetah-v5 |
| RoboschoolHopper-v1 | Hopper-v5 |
| RoboschoolWalker2d-v1 | Walker2d-v5 |
| RoboschoolHumanoid-v1 | Humanoid-v5 |

### Environment Version Updates
- **CartPole-v0 → v1**: Episodes 200→500 steps
- **LunarLander-v2 → v3**: Bug fixes
- **Pendulum-v0 → v1**: Standardized API

## 📦 Updated Dependencies

- **PyTorch**: 2.8.0 with CUDA 12.8
- **Gymnasium**: Replaces gym (`terminated`/`truncated` flags)
- **ALE-py 0.11.2**: Official Atari environment
- **gymnasium[mujoco]**: MuJoCo physics simulation
- **loguru**: Modern logging system

## ⚡ Key Optimizations

- **Vector Environments**: Sync ≤8 envs, async >8 envs (up to 4x speedup)
- **SAC**: Fixed target entropy calculation (-log(action_dim))
- **Native PyTorch Optimizers**: Replaced custom Lookahead/RAdam with AdamW
- **ASHA Scheduler**: Efficient hyperparameter search with early termination

## 🏗️ Architecture Changes

### Body → MetricsTracker Refactoring
- **Class Rename**: `Body` → `MetricsTracker`, `body` → `mt`
- **Memory**: `Memory(memory_spec, body)` → `Memory(memory_spec, agent)`
- Direct agent access instead of body indirection

## 🚀 New Features

### Remote Training (dstack + HuggingFace)
```bash
slm-lab run-remote spec.json spec_name train  # launch on dstack, auto-upload to HF
slm-lab list                                  # list remote experiments
slm-lab pull spec_name                        # sync to local
```

### Enhanced CLI
```bash
# Variable substitution
slm-lab -s env=CartPole-v1 spec.json spec_name dev

# HuggingFace upload
source .env  # Set HF_REPO and HF_TOKEN
slm-lab run spec.json spec_name train --upload-hf
```

## 🧪 Algorithm Status

### Verified Working (CartPole MA >400)
- REINFORCE, A2C, PPO, SARSA, DQN/DDQN, SAC

### Known Limitations
- **SIL**: Does not solve dense reward environments (designed for sparse rewards)

## 🎯 ASHA Hyperparameter Search

**Critical**: ASHA requires `max_session=1`. Multi-session (`max_session>1`) must not use `search_scheduler`.

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

See `CLAUDE.md` for detailed benchmarking methodology.
