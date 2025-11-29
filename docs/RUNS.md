# Active Remote Runs

Track dstack runs for continuity. Use `dstack ps` to check status.

## Current Runs (2025-11-29)

### Active PPO MuJoCo GPU Searches

| Run Name | Command | Hardware | Price/hr | Status |
|----------|---------|----------|----------|--------|
| `ppo-hopper-gpu` | `slm-lab run-remote --gpu -s env=Hopper-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search -n ppo-hopper-gpu` | runpod L4 | $0.39 | running |
| `ppo-walker-gpu` | `slm-lab run-remote --gpu -s env=Walker2d-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search -n ppo-walker-gpu` | runpod L4 | $0.39 | running |
| `ppo-cheetah-gpu` | `slm-lab run-remote --gpu -s env=HalfCheetah-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search -n ppo-cheetah-gpu` | runpod L4 | $0.39 | running |
| `ppo-ant-gpu` | `slm-lab run-remote --gpu -s env=Ant-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search -n ppo-ant-gpu` | runpod L4 | $0.39 | running |
| `ppo-swimmer-gpu` | `slm-lab run-remote --gpu -s env=Swimmer-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search -n ppo-swimmer-gpu` | runpod L4 | $0.39 | running |
| `ppo-reacher-gpu` | `slm-lab run-remote --gpu -s env=Reacher-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search -n ppo-reacher-gpu` | runpod L4 | $0.39 | running (test) |

**Total**: 6 runs @ $2.34/hr

**Estimated Duration**: 2-4 hours (ASHA early termination)

**Config**: `search_resources: {"cpu": 1, "gpu": 0.125}` enables 8 parallel trials sharing 1 GPU

## Planned Runs (Phase 1-2 Catchup)

All specs use `search_resources: {"cpu": 1, "gpu": 0.125}` for 8 parallel trials sharing 1 GPU.
**Always use `--gpu`** - cheaper ($0.39/hr L4 vs $0.54/hr 16-CPU) and faster.

### Phase 1 (Classic Control)

| Priority | Algo | Env | Notes |
|----------|------|-----|-------|
| 1 | SAC | Acrobot | 8 trials, grace 50k |
| 2 | PPOSIL | Acrobot | 8 trials, grace 50k |

```bash
# Phase 1 commands (copy-paste ready)
source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/sac/sac_acrobot.json sac_acrobot search -n sac-acrobot
source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/sil/ppo_sil_acrobot.json ppo_sil_acrobot search -n pposil-acrobot
```

### Phase 2 (Box2D)

| Priority | Algo | Env | Notes |
|----------|------|-----|-------|
| 3 | A2C | LunarLander | 8 trials, grace 50k |
| 4 | SAC | LunarLander | 8 trials, grace 50k |
| 5 | A2C | LunarLander-Cont | 8 trials, grace 50k |
| 6 | SAC | BipedalWalker | 8 trials, 1M frames, grace 100k |
| 7 | A2C | BipedalWalker | 8 trials, 3M frames, grace 300k |

```bash
# Phase 2 commands (copy-paste ready)
source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json a2c_gae_lunar search -n a2c-lunar
source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/sac/sac_lunar.json sac_lunar search -n sac-lunar
source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json a2c_gae_lunar_continuous search -n a2c-lunar-cont
source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/sac/sac_bipedalwalker.json sac_bipedalwalker search -n sac-bipedal
source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/a2c/a2c_gae_bipedalwalker.json a2c_gae_bipedalwalker search -n a2c-bipedal
```

## Commands

```bash
# IMPORTANT: Always source .env first for HF upload credentials
source .env

# Run remote experiments (always use uv run)
uv run slm-lab run-remote SPEC_FILE SPEC_NAME MODE -n RUN_NAME  # CPU (default)
uv run slm-lab run-remote --gpu SPEC_FILE SPEC_NAME MODE -n RUN_NAME  # GPU

# Check status
dstack ps
dstack logs <run-name>

# Stop runs
dstack stop <run-name> -y

# Pull results when complete
uv run slm-lab pull SPEC_NAME
```

**Environment Setup**: The `.env` file contains `HF_TOKEN` and `HF_REPO` for uploading results to HuggingFace. Always `source .env` before running remote experiments to ensure results are saved.

**Max Duration**: All dstack runs have a 4h safeguard (`max_duration: 4h`) to prevent runaway costs.

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

## Phase 1-2 Algorithm Status

| Phase | Env | Algo | Status | Notes |
|-------|-----|------|--------|-------|
| 1.2 | Acrobot | SAC | 🔄 ready | Spec fixed with search_resources |
| 1.2 | Acrobot | PPOSIL | 🔄 ready | Spec fixed with search_resources |
| 2.1 | LunarLander | A2C | 🔄 ready | Spec fixed with search_resources |
| 2.1 | LunarLander | SAC | 🔄 ready | Spec fixed with search_resources |
| 2.2 | LunarLander-Cont | A2C | 🔄 ready | Spec fixed with search_resources |
| 2.3 | BipedalWalker | SAC | 🔄 ready | Spec fixed with search_resources |
| 2.3 | BipedalWalker | A2C | 🔄 ready | Spec consolidated, variants removed |
