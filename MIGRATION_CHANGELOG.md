# SLM-Lab Migration Changelog

## Summary

Complete modernization of SLM-Lab with gymnasium migration, modern toolchain, and significant performance improvements. **Contains breaking changes** - see migration instructions below.

## 🚨 **Breaking Changes**

### **Package Management: conda → uv**
**BREAKING**: All setup and execution now uses `uv` instead of conda.

```bash
# Old (no longer works)
conda activate lab
python run_lab.py [args]

# New (required)
uv run slm-lab [args]
bin/setup  # Now uses uv automatically
```

### **Environment Dependencies Removed**
**BREAKING**: Unity ML-Agents, VizDoom, and roboschool support removed.

- **Unity/VizDoom**: Use external [gymnasium packages](https://gymnasium.farama.org/environments/third_party_environments/)
- **roboschool**: Automatically migrated to gymnasium MuJoCo (see mapping below)

### **Logging System Changed**
**BREAKING**: New loguru-based logging requires environment variable.

```bash
export LOG_METRICS=true  # Required for training metrics
```

## 🔄 **Environment Updates**

### **Atari: New ALE Integration**
All Atari environments now use gymnasium's ALE integration with optimized preprocessing.

- **Environment names**: `PongNoFrameskip-v4` → `ALE/Pong-v5`
- **Preprocessing**: Automatic (frame skip, resize, grayscale, stacking, NoOp reset, etc.)
- **Performance**: Native C++ implementation vs custom Python wrappers
- **Migration**: Automatic - no spec file changes needed

### **Roboschool → MuJoCo Migration**
Automatic mapping to gymnasium MuJoCo v5 environments:

| Old | New |
|-----|-----|
| RoboschoolAnt-v1 | Ant-v5 |
| RoboschoolHalfCheetah-v1 | HalfCheetah-v5 |
| RoboschoolHopper-v1 | Hopper-v5 |
| RoboschoolWalker2d-v1 | Walker2d-v5 |
| RoboschoolHumanoid-v1 | Humanoid-v5 |

### **Environment Version Updates**
Automatic updates to latest stable releases:

- **CartPole-v0 → v1**: Episodes 200→500 steps, reward threshold 195→475
- **LunarLander-v2 → v3**: Bug fixes and improved stability
- **Pendulum-v0 → v1**: Standardized API

## ⚡ **Performance Improvements**

### **Vector Environment Optimization**
Intelligent vectorization mode selection:
- **≤8 environments**: Sync mode (avoids subprocess overhead)
- **>8 environments**: Async mode (leverages parallelization)
- **Result**: Up to 4x performance improvement, 1600-2000 fps on CartPole

### **Algorithm Optimizations**
- **SAC**: Fixed target entropy calculation, eliminated numpy↔torch conversions
- **All algorithms**: Universal action shape compatibility, reduced memory usage
- **Atari**: Native C++ preprocessing replaces 4000+ lines of custom Python wrappers

## 🔧 **New Features**

### **GPU Training Infrastructure**
```bash
# GPU training with dstack
dstack apply -f .dstack/train.yml

# torch.compile optimization (20-30% speedup)
TORCH_COMPILE=true uv run slm-lab [args]
```

### **Modern Development Tooling**
- **pyproject.toml**: Replaces setup.py and package.json
- **uv.lock**: Faster, deterministic dependency resolution
- **Native CLI**: Simplified command structure

## 📦 **Updated Dependencies**

- **PyTorch**: Updated to 2.8.0 with CUDA 12.8
- **Gymnasium**: Replaces gym with modern API (`terminated`/`truncated` flags)
- **ALE-py 0.11.2**: Official Atari environment with C++ preprocessing
- **gymnasium[mujoco]**: MuJoCo physics simulation
- **loguru**: Modern logging system

## 🧪 **Algorithm Status**

### **✅ Verified Working**
- DQN, DoubleDQN, Dueling DQN with Prioritized Experience Replay
- PPO (Proximal Policy Optimization) - single and vector environments
- A2C (Advantage Actor-Critic) 
- SAC (Soft Actor-Critic) - improved target entropy calculation
- REINFORCE (Policy Gradient)

### **🔧 Testing Needed**
- Some MuJoCo environments may need validation
- Ray hyperparameter search disabled until stable

## 🧪 **Testing Commands**

```bash
# Basic functionality
uv run slm-lab slm_lab/spec/demo.json dqn_cartpole dev
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_shared_cartpole dev

# Advanced algorithms
uv run slm-lab slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json ddqn_per_concat_lunar dev
uv run slm-lab slm_lab/spec/benchmark/sac/sac_lunar.json sac_lunar train
```

## 🔮 **Future Releases**

- Ray hyperparameter search with Optuna backend
- TrackTime environment wrapper
- Extended gymnasium environment support
- Comprehensive benchmark validation

**SLM-Lab is now fully modernized with gymnasium, modern toolchain, and optimized performance.**