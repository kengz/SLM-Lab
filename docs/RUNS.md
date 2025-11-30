# Active Remote Runs

Track dstack runs for continuity. Use `dstack ps` to check status.

## Current Runs (2025-11-29)

### Active Searches

| Run Name | Command | Hardware | Price/hr | Status |
|----------|---------|----------|----------|--------|
| `sac-acrobot` | Phase 1.2 SAC Acrobot | runpod L4 | $0.39 | finishing (1 trial left) |
| `ppo-reacher-gpu` | PPO Reacher-v5 search | runpod L4 | $0.39 | finishing (2 trials left) |

**Total**: 2 runs @ ~$0.78/hr

**Config**: `search_resources: {"cpu": 1, "gpu": 0.125}` enables 8 parallel trials sharing 1 GPU

## Planned Work Queue (Priority Order)

All specs use `search_resources: {"cpu": 1, "gpu": 0.125}` for 8 parallel trials sharing 1 GPU.
**Always use `--gpu`** - cheaper ($0.39/hr L4 vs $0.54/hr 16-CPU) and faster.

### Phase 1 Remaining (Classic Control)

| Priority | Algo | Env | Spec | Run Name | Status |
|----------|------|-----|------|----------|--------|
| 1 | SAC | Acrobot | `sac_acrobot` | `sac-acrobot` | 🔄 running |
| 2 | PPOSIL | Acrobot | `ppo_sil_acrobot` | `pposil-acrobot` | ⏸️ ready |

```bash
# Phase 1 commands (copy-paste ready)
source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/sil/ppo_sil_acrobot.json ppo_sil_acrobot search -n pposil-acrobot
```

### Phase 2 Remaining (Box2D)

| Priority | Algo | Env | Spec | Run Name | Status |
|----------|------|-----|------|----------|--------|
| 3 | A2C | LunarLander | `a2c_gae_lunar` | `a2c-lunar` | 🔄 running |
| 4 | SAC | LunarLander | `sac_lunar` | `sac-lunar` | ⏸️ ready |
| 5 | A2C | LunarLander-Cont | `a2c_gae_lunar_continuous` | `a2c-lunar-cont` | ⏸️ ready |
| 6 | SAC | BipedalWalker | `sac_bipedalwalker` | `sac-bipedal` | ⏸️ ready |
| 7 | A2C | BipedalWalker | `a2c_gae_bipedalwalker` | `a2c-bipedal` | ⏸️ ready |

```bash
# Phase 2 commands (copy-paste ready)
source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/sac/sac_lunar.json sac_lunar search -n sac-lunar
source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json a2c_gae_lunar_continuous search -n a2c-lunar-cont
source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/sac/sac_bipedalwalker.json sac_bipedalwalker search -n sac-bipedal
source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/a2c/a2c_gae_bipedalwalker.json a2c_gae_bipedalwalker search -n a2c-bipedal
```

### Phase 3 - PPO MuJoCo (Priority 1)

| Priority | Algo | Env | Spec | Env Var | Run Name | Status |
|----------|------|-----|------|---------|----------|--------|
| 8 | PPO | Hopper-v5 | `ppo_mujoco` | `-s env=Hopper-v5` | `ppo-hopper-gpu` | 🔄 running |
| 9 | PPO | Walker2d-v5 | `ppo_mujoco` | `-s env=Walker2d-v5` | `ppo-walker-gpu` | 🔄 running |
| 10 | PPO | HalfCheetah-v5 | `ppo_mujoco` | `-s env=HalfCheetah-v5` | `ppo-cheetah-gpu` | 🔄 running |
| 11 | PPO | Ant-v5 | `ppo_mujoco` | `-s env=Ant-v5` | `ppo-ant-gpu` | 🔄 running |
| 12 | PPO | Swimmer-v5 | `ppo_mujoco` | `-s env=Swimmer-v5` | `ppo-swimmer-gpu` | 🔄 running |
| 13 | PPO | Reacher-v5 | `ppo_mujoco` | `-s env=Reacher-v5` | `ppo-reacher-gpu` | 🔄 running |
| 14 | PPO | InvertedPendulum-v5 | `ppo_mujoco` | `-s env=InvertedPendulum-v5` | `ppo-invpend-gpu` | ⏸️ ready |
| 15 | PPO | InvertedDoublePendulum-v5 | `ppo_mujoco` | `-s env=InvertedDoublePendulum-v5` | `ppo-invdoublepend-gpu` | ⏸️ ready |
| 16 | PPO | Humanoid-v5 | `ppo_mujoco` | `-s env=Humanoid-v5` | `ppo-humanoid-gpu` | ⏸️ ready |

```bash
# Phase 3 PPO commands (copy-paste ready)
source .env && uv run slm-lab run-remote --gpu -s env=InvertedPendulum-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search -n ppo-invpend-gpu
source .env && uv run slm-lab run-remote --gpu -s env=InvertedDoublePendulum-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search -n ppo-invdoublepend-gpu
source .env && uv run slm-lab run-remote --gpu -s env=Humanoid-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search -n ppo-humanoid-gpu
```

### Phase 3 - SAC MuJoCo (Priority 2)

| Priority | Algo | Env | Spec | Env Var | Run Name | Status |
|----------|------|-----|------|---------|----------|--------|
| 17 | SAC | Hopper-v5 | `sac_mujoco` | `-s env=Hopper-v5` | `sac-hopper-gpu` | ⏸️ ready |
| 18 | SAC | Walker2d-v5 | `sac_mujoco` | `-s env=Walker2d-v5` | `sac-walker-gpu` | ⏸️ ready |
| 19 | SAC | HalfCheetah-v5 | `sac_mujoco` | `-s env=HalfCheetah-v5` | `sac-cheetah-gpu` | ⏸️ ready |
| 20 | SAC | Ant-v5 | `sac_mujoco` | `-s env=Ant-v5` | `sac-ant-gpu` | ⏸️ ready |
| 21 | SAC | Swimmer-v5 | `sac_mujoco` | `-s env=Swimmer-v5` | `sac-swimmer-gpu` | ⏸️ ready |
| 22 | SAC | Reacher-v5 | `sac_mujoco` | `-s env=Reacher-v5` | `sac-reacher-gpu` | ⏸️ ready |
| 23 | SAC | InvertedPendulum-v5 | `sac_mujoco` | `-s env=InvertedPendulum-v5` | `sac-invpend-gpu` | ⏸️ ready |
| 24 | SAC | InvertedDoublePendulum-v5 | `sac_mujoco` | `-s env=InvertedDoublePendulum-v5` | `sac-invdoublepend-gpu` | ⏸️ ready |
| 25 | SAC | Humanoid-v5 | `sac_humanoid` | (dedicated spec) | `sac-humanoid-gpu` | ⏸️ ready |

```bash
# Phase 3 SAC commands (copy-paste ready)
source .env && uv run slm-lab run-remote --gpu -s env=Hopper-v5 slm_lab/spec/benchmark/sac/sac_mujoco.json sac_mujoco search -n sac-hopper-gpu
source .env && uv run slm-lab run-remote --gpu -s env=Walker2d-v5 slm_lab/spec/benchmark/sac/sac_mujoco.json sac_mujoco search -n sac-walker-gpu
source .env && uv run slm-lab run-remote --gpu -s env=HalfCheetah-v5 slm_lab/spec/benchmark/sac/sac_mujoco.json sac_mujoco search -n sac-cheetah-gpu
source .env && uv run slm-lab run-remote --gpu -s env=Ant-v5 slm_lab/spec/benchmark/sac/sac_mujoco.json sac_mujoco search -n sac-ant-gpu
source .env && uv run slm-lab run-remote --gpu -s env=Swimmer-v5 slm_lab/spec/benchmark/sac/sac_mujoco.json sac_mujoco search -n sac-swimmer-gpu
source .env && uv run slm-lab run-remote --gpu -s env=Reacher-v5 slm_lab/spec/benchmark/sac/sac_mujoco.json sac_mujoco search -n sac-reacher-gpu
source .env && uv run slm-lab run-remote --gpu -s env=InvertedPendulum-v5 slm_lab/spec/benchmark/sac/sac_mujoco.json sac_mujoco search -n sac-invpend-gpu
source .env && uv run slm-lab run-remote --gpu -s env=InvertedDoublePendulum-v5 slm_lab/spec/benchmark/sac/sac_mujoco.json sac_mujoco search -n sac-invdoublepend-gpu
source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/sac/sac_mujoco.json sac_humanoid search -n sac-humanoid-gpu
```

### Phase 3 - A2C MuJoCo (Priority 3)

| Priority | Algo | Env | Spec | Env Var | Run Name | Status |
|----------|------|-----|------|---------|----------|--------|
| 26 | A2C | Hopper-v5 | `a2c_gae_mujoco` | `-s env=Hopper-v5` | `a2c-hopper-gpu` | ⏸️ needs spec |
| 27 | A2C | Walker2d-v5 | `a2c_gae_mujoco` | `-s env=Walker2d-v5` | `a2c-walker-gpu` | ⏸️ needs spec |
| 28 | A2C | HalfCheetah-v5 | `a2c_gae_mujoco` | `-s env=HalfCheetah-v5` | `a2c-cheetah-gpu` | ⏸️ needs spec |
| 29 | A2C | Ant-v5 | `a2c_gae_mujoco` | `-s env=Ant-v5` | `a2c-ant-gpu` | ⏸️ needs spec |

```bash
# Phase 3 A2C commands (copy-paste ready) - NOTE: need to update spec with search_resources first
source .env && uv run slm-lab run-remote --gpu -s env=Hopper-v5 slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json a2c_gae_mujoco search -n a2c-hopper-gpu
source .env && uv run slm-lab run-remote --gpu -s env=Walker2d-v5 slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json a2c_gae_mujoco search -n a2c-walker-gpu
source .env && uv run slm-lab run-remote --gpu -s env=HalfCheetah-v5 slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json a2c_gae_mujoco search -n a2c-cheetah-gpu
source .env && uv run slm-lab run-remote --gpu -s env=Ant-v5 slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json a2c_gae_mujoco search -n a2c-ant-gpu
```

## Summary Table

| Phase | Category | Total Runs | PPO | SAC | A2C | Other |
|-------|----------|------------|-----|-----|-----|-------|
| 1 | Classic Control | 2 | - | 1 | - | 1 (PPOSIL) |
| 2 | Box2D | 5 | - | 2 | 3 | - |
| 3 | MuJoCo | 22 | 9 | 9 | 4 | - |
| **Total** | | **29** | **9** | **12** | **7** | **1** |

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

| Date | Run | Result | Notes |
|------|-----|--------|-------|
| 2025-11-29 | ppo-hopper-gpu | MA=2816 @ 3M | ✅ 94% of target (3000) - ASHA search |
| 2025-11-29 | ppo-walker-gpu | MA=2573 @ 3M | ⚠️ 64% of target (4000) - needs tuning |
| 2025-11-29 | ppo-cheetah-gpu | MA=4042 @ 3M | ✅ 81% of target (5000) - ASHA search |
| 2025-11-29 | ppo-ant-gpu | MA=36 @ 3M | ❌ 0.7% of target (5000) - needs dedicated tuning |
| 2025-11-29 | ppo-swimmer-gpu | MA=44 @ 3M | ⚠️ 44% of target (100) - needs tuning |
| 2025-11-29 | a2c-lunar | MA=5.5 @ 300k | ❌ 2.8% of target (200) - needs tuning |

## MuJoCo PPO Status Summary

| Environment | MA @ 3M | Target | % Target | Status | Next Step |
|-------------|---------|--------|----------|--------|-----------|
| Hopper-v5 | 2816 | 3000 | 94% | ✅ | Done - spec updated |
| HalfCheetah-v5 | 4042 | 5000 | 81% | ✅ | Done - spec updated |
| Walker2d-v5 | 2573 | 4000 | 64% | ⚠️ | Needs env-specific search |
| Ant-v5 | 36 | 5000 | 0.7% | ❌ | 4-leg dynamics need dedicated tuning |
| Swimmer-v5 | 44 | 100 | 44% | ⚠️ | Needs tuning |
| Reacher-v5 | -6.2 | -5 | ~close | 🔄 | Search finishing |
| InvertedPendulum-v5 | - | 1000 | - | ⏸️ | Queue |
| InvertedDoublePendulum-v5 | - | 9000 | - | ⏸️ | Queue |
| Humanoid-v5 | - | 6000 | - | ⏸️ | Queue |

**Key Finding**: ASHA search found gamma~0.998, lam~0.905 works well for Hopper and HalfCheetah. Ant and Walker2d need different dynamics-specific tuning. Swimmer is hard for PPO.

## Phase 1-2 Algorithm Status

| Phase | Env | Algo | Status | Notes |
|-------|-----|------|--------|-------|
| 1.2 | Acrobot | SAC | ✅ done | MA=-92, spec updated with winning params |
| 1.2 | Acrobot | PPOSIL | ⏸️ ready | Spec fixed with search_resources |
| 2.1 | LunarLander | A2C | ❌ failed | MA=5.5 (2.8% of target) - needs investigation |
| 2.1 | LunarLander | SAC | ⏸️ ready | Spec fixed with search_resources |
| 2.2 | LunarLander-Cont | A2C | ⏸️ ready | Spec fixed with search_resources |
| 2.3 | BipedalWalker | SAC | ⏸️ ready | Spec fixed with search_resources |
| 2.3 | BipedalWalker | A2C | ⏸️ ready | Spec consolidated, variants removed |
