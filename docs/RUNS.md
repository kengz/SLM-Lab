# Active Remote Runs

Track dstack runs for continuity. Use `dstack ps` to check status.

**Reference**: See [BENCHMARKS.md](BENCHMARKS.md) for environment details, targets, and methodology.

## Current Runs

No active runs.

---

## Work Line 1: Phase 1-2 Completion

Fix unsolved entries in Classic Control and Box2D phases.

### 1.2 Acrobot

| Algo | Stage | Status | MA | Target | Command |
|------|-------|--------|-----|--------|---------|
| DQN (ε) | ASHA | ⚠️ | -104 | -100 | `source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/dqn/dqn_acrobot.json dqn_epsilon_greedy_acrobot search -n dqn-acrobot` |
| PPOSIL | ASHA | ⚠️ | -110 | -100 | `source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/sil/ppo_sil_acrobot.json ppo_sil_acrobot search -n pposil-acrobot` |

### 2.1 LunarLander (Discrete)

| Algo | Stage | Status | MA | Target | Command |
|------|-------|--------|-----|--------|---------|
| A2C | ASHA | ❌ | 5.5 | 200 | `source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json a2c_gae_lunar search -n a2c-lunar` |
| SAC | ASHA | ⏸️ | - | 200 | `source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/sac/sac_lunar.json sac_lunar search -n sac-lunar` |

### 2.2 LunarLander (Continuous)

| Algo | Stage | Status | MA | Target | Command |
|------|-------|--------|-----|--------|---------|
| A2C | ASHA | ⏸️ | - | 200 | `source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json a2c_gae_lunar_continuous search -n a2c-lunar-cont` |

### 2.3 BipedalWalker

| Algo | Stage | Status | MA | Target | Command |
|------|-------|--------|-----|--------|---------|
| PPO | Multi | ⚠️ | 241 | 300 | `source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/ppo/ppo_bipedalwalker.json ppo_bipedalwalker search -n ppo-bipedal` |
| SAC | ASHA | 🔄 | - | 300 | `source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/sac/sac_bipedalwalker.json sac_bipedalwalker search -n sac-bipedal` |
| A2C | ASHA | ❌ | -112 | 300 | `source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/a2c/a2c_gae_bipedalwalker.json a2c_gae_bipedalwalker search -n a2c-bipedal` |

---

## Work Line 2: Phase 3 MuJoCo Refinement

Continue from first ASHA round. Successful envs (✅) need validation runs; others need more tuning.

### 3.1-3.2 Hopper, HalfCheetah (PPO ✅)

ASHA complete, specs updated. Ready for validation or done.

| Env | Stage | Status | MA | Target | Command |
|-----|-------|--------|-----|--------|---------|
| Hopper-v5 | Validate | ✅ | 2816 | 3000 | `source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/ppo/ppo_hopper.json ppo_hopper train -n ppo-hopper-val` |
| HalfCheetah-v5 | Validate | ✅ | 4042 | 5000 | `source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/ppo/ppo_halfcheetah.json ppo_halfcheetah train -n ppo-cheetah-val` |

### 3.3-3.5 Walker2d, Ant, Swimmer (PPO needs tuning)

ASHA round 1 complete but below target. Need env-specific dedicated specs.

| Env | Stage | Status | MA | Target | Command |
|-----|-------|--------|-----|--------|---------|
| Walker2d-v5 | Multi | ⚠️ 64% | 2573 | 4000 | `source .env && uv run slm-lab run-remote --gpu -s env=Walker2d-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search -n ppo-walker` |
| Ant-v5 | ASHA | ❌ 0.7% | 36 | 5000 | `source .env && uv run slm-lab run-remote --gpu -s env=Ant-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search -n ppo-ant` |
| Swimmer-v5 | ASHA | ⚠️ 44% | 44 | 100 | `source .env && uv run slm-lab run-remote --gpu -s env=Swimmer-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search -n ppo-swimmer` |

### 3.6-3.9 Reacher, Pendulums, Humanoid (PPO queue)

| Env | Stage | Status | MA | Target | Command |
|-----|-------|--------|-----|--------|---------|
| Reacher-v5 | Multi | 🔄 | -6.2 | -5 | `source .env && uv run slm-lab run-remote --gpu -s env=Reacher-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search -n ppo-reacher` |
| InvertedPendulum-v5 | ASHA | ⏸️ | - | 1000 | `source .env && uv run slm-lab run-remote --gpu -s env=InvertedPendulum-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search -n ppo-invpend` |
| InvertedDoublePendulum-v5 | ASHA | ⏸️ | - | 9000 | `source .env && uv run slm-lab run-remote --gpu -s env=InvertedDoublePendulum-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search -n ppo-invdoubpend` |
| Humanoid-v5 | ASHA | ⏸️ | - | 6000 | `source .env && uv run slm-lab run-remote --gpu -s env=Humanoid-v5 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco search -n ppo-humanoid` |

### SAC MuJoCo (all queue)

SAC may outperform PPO on Ant/Swimmer. Run after PPO baseline established.

| Env | Stage | Status | Command |
|-----|-------|--------|---------|
| Hopper-v5 | ASHA | ⏸️ | `source .env && uv run slm-lab run-remote --gpu -s env=Hopper-v5 slm_lab/spec/benchmark/sac/sac_mujoco.json sac_mujoco search -n sac-hopper` |
| HalfCheetah-v5 | ASHA | ⏸️ | `source .env && uv run slm-lab run-remote --gpu -s env=HalfCheetah-v5 slm_lab/spec/benchmark/sac/sac_mujoco.json sac_mujoco search -n sac-cheetah` |
| Walker2d-v5 | ASHA | ⏸️ | `source .env && uv run slm-lab run-remote --gpu -s env=Walker2d-v5 slm_lab/spec/benchmark/sac/sac_mujoco.json sac_mujoco search -n sac-walker` |
| Ant-v5 | ASHA | ⏸️ | `source .env && uv run slm-lab run-remote --gpu -s env=Ant-v5 slm_lab/spec/benchmark/sac/sac_mujoco.json sac_mujoco search -n sac-ant` |
| Swimmer-v5 | ASHA | ⏸️ | `source .env && uv run slm-lab run-remote --gpu -s env=Swimmer-v5 slm_lab/spec/benchmark/sac/sac_mujoco.json sac_mujoco search -n sac-swimmer` |
| Humanoid-v5 | ASHA | ⏸️ | `source .env && uv run slm-lab run-remote --gpu slm_lab/spec/benchmark/sac/sac_humanoid.json sac_humanoid search -n sac-humanoid` |

---

## Commands Reference

```bash
# ASHA search (Stage 1)
source .env && uv run slm-lab run-remote --gpu SPEC_FILE SPEC_NAME search -n NAME

# Multi-session refinement (Stage 2) - same command, spec has no search_scheduler
source .env && uv run slm-lab run-remote --gpu SPEC_FILE SPEC_NAME search -n NAME

# Validation train (Stage 3)
source .env && uv run slm-lab run-remote --gpu SPEC_FILE SPEC_NAME train -n NAME

# Monitor
dstack ps
dstack logs <run-name>
dstack stop <run-name> -y

# Pull results
uv run slm-lab pull SPEC_NAME
```

**Config**: `search_resources: {"cpu": 1, "gpu": 0.125}` enables 8 parallel trials sharing 1 GPU.
**Max Duration**: 4h safeguard on all runs.

---

## Completed Runs

| Date | Run | Result | Stage | Notes |
|------|-----|--------|-------|-------|
| 2025-11-29 | ppo-hopper-gpu | MA=2816 | ASHA | ✅ 94% - spec updated |
| 2025-11-29 | ppo-cheetah-gpu | MA=4042 | ASHA | ✅ 81% - spec updated |
| 2025-11-29 | ppo-walker-gpu | MA=2573 | ASHA | ⚠️ 64% - needs tuning |
| 2025-11-29 | ppo-ant-gpu | MA=36 | ASHA | ❌ 0.7% - needs tuning |
| 2025-11-29 | ppo-swimmer-gpu | MA=44 | ASHA | ⚠️ 44% - needs tuning |
| 2025-11-29 | a2c-lunar | MA=5.5 | ASHA | ❌ 2.8% - needs tuning |

---

## Key Findings

- **MuJoCo PPO**: gamma~0.998, lam~0.905 works for Hopper/HalfCheetah. Ant/Walker2d/Swimmer need different tuning.
- **Ant**: 4-leg dynamics fundamentally different from bipeds - may need SAC or dedicated architecture.
- **Swimmer**: Hard for PPO - SAC likely better choice.
