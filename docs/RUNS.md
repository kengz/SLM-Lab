# Active Remote Runs

Track dstack runs for continuity. Use `dstack ps` to check status.

## Current Runs (2025-11-29)

### Active ASHA GPU Searches

| Run Name | Command | Hardware | Price/hr | Status |
|----------|---------|----------|----------|--------|
| `ppo-hopper-gpu` | `slm-lab run-remote --gpu -s env=Hopper-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search -n ppo-hopper-gpu` | runpod L4 | $0.39 | running |
| `ppo-walker-gpu` | `slm-lab run-remote --gpu -s env=Walker2d-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search -n ppo-walker-gpu` | runpod L4 | $0.39 | running |
| `ppo-cheetah-gpu` | `slm-lab run-remote --gpu -s env=HalfCheetah-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search -n ppo-cheetah-gpu` | runpod L4 | $0.39 | running |
| `ppo-ant-gpu` | `slm-lab run-remote --gpu -s env=Ant-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search -n ppo-ant-gpu` | runpod L4 | $0.39 | running |
| `ppo-swimmer-gpu` | `slm-lab run-remote --gpu -s env=Swimmer-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search -n ppo-swimmer-gpu` | runpod L4 | $0.39 | running |

**Total**: 5 runs @ $1.95/hr

**Estimated Duration**: 2-4 hours (ASHA early termination)

**Config**: `search_resources: {"cpu": 1, "gpu": 0.125}` enables 8 parallel trials sharing 1 GPU

## Commands

```bash
# Check status
dstack ps
dstack logs <run-name>

# Pull results when complete
slm-lab pull ppo_mujoco

# Run examples
slm-lab run-remote SPEC_FILE SPEC_NAME MODE -n RUN_NAME  # CPU (default)
slm-lab run-remote --gpu SPEC_FILE SPEC_NAME MODE -n RUN_NAME  # GPU
```

## Completed Runs (2025-11-29)

| Date | Run | Command | Result | Notes |
|------|-----|---------|--------|-------|
| 2025-11-29 | ppo-hopper-v4 | `slm-lab run-remote -s env=Hopper-v5 ... ppo_mujoco train` | MA=2566 @ 3M | 85% of target (3000) |
| 2025-11-29 | ppo-walker-v3 | `slm-lab run-remote -s env=Walker2d-v5 ... ppo_mujoco train` | MA=1424 @ 3M | 36% of target (4000) |
| 2025-11-29 | ppo-cheetah-v3 | `slm-lab run-remote -s env=HalfCheetah-v5 ... ppo_mujoco train` | MA=3178 @ 3M | 64% of target (5000) |
| 2025-11-29 | ppo-ant-v3 | `slm-lab run-remote -s env=Ant-v5 ... ppo_mujoco train` | MA=34 @ 3M | 0.7% of target (5000) |

## MuJoCo PPO Status Summary

| Environment | MA @ 3M | Target | % Target | Status | Next Step |
|-------------|---------|--------|----------|--------|-----------|
| Hopper-v5 | 2566 | 3000 | 85% | ✅ | Done (close enough) |
| HalfCheetah-v5 | 3178 | 5000 | 64% | ⚠️ | May need longer training |
| Walker2d-v5 | 1424 | 4000 | 36% | 🔄 | ASHA search running |
| Ant-v5 | 34 | 5000 | 0.7% | 🔄 | ASHA search running |

**Key Finding**: Hopper-tuned hyperparameters (gamma=0.995, lam=0.92, entropy=0.002) transfer reasonably to HalfCheetah but fail badly on Walker2d and Ant. These envs have different dynamics and need env-specific tuning.
