# Active Remote Runs

Track dstack runs for continuity. Use `dstack ps` to check status.

## Current Runs (2025-11-27)

### Active ASHA Searches (CPU-only, gpu:0, faster configs)

| Run Name | Spec | Env | Mode | Hardware | Price/hr | Status | Notes |
|----------|------|-----|------|----------|----------|--------|-------|
| `ppo-bipedal2` | ppo_bipedalwalker | BipedalWalker-v3 | search | GCP e2-standard-16 | $0.0634 | running | 1.5M frames, 15 trials |
| `sac-bipedal2` | sac_bipedalwalker | BipedalWalker-v3 | search | GCP e2-standard-16 | $0.0634 | running | 1.5M frames, 15 trials |
| `ppo-hopper2` | ppo_mujoco | Hopper-v5 | search | GCP e2-standard-16 | $0.0634 | running | 1M frames, 15 trials |
| `sac-hopper2` | sac_mujoco | Hopper-v5 | search | GCP e2-standard-16 | $0.0634 | running | 1M frames, 15 trials |
| `ppo-walker2` | ppo_mujoco | Walker2d-v5 | search | GCP e2-standard-16 | $0.0634 | running | 1M frames, 15 trials |
| `sac-walker2` | sac_mujoco | Walker2d-v5 | search | GCP e2-standard-16 | $0.0634 | running | 1M frames, 15 trials |
| `ppo-cheetah2` | ppo_mujoco | HalfCheetah-v5 | search | GCP e2-standard-16 | $0.0634 | running | 1M frames, 15 trials |
| `sac-cheetah2` | sac_mujoco | HalfCheetah-v5 | search | GCP e2-standard-16 | $0.0634 | running | 1M frames, 15 trials |
| `ppo-ant2` | ppo_mujoco | Ant-v5 | search | GCP e2-standard-16 | $0.0634 | running | 1M frames, 15 trials |
| `sac-ant2` | sac_mujoco | Ant-v5 | search | GCP e2-standard-16 | $0.0634 | running | 1M frames, 15 trials |

**Total**: 10 runs @ $0.0634/hr = ~$0.63/hr

**Estimated Duration**: 3-4 hours (reduced from 12+ hours with faster configs)

**Config Changes Made**:
- `max_frame`: 1M (MuJoCo), 1.5M (BipedalWalker) - using `1e6` notation
- `max_trial`: 15 (from 20-30)
- `grace_period`: 100k (MuJoCo), 150k (BipedalWalker)
- `reduction_factor`: 4 (more aggressive early termination)

## Commands

```bash
# Check status
dstack ps
dstack logs <run-name>

# Pull results when complete
uv run slm-lab pull ppo_bipedalwalker
uv run slm-lab pull sac_bipedalwalker
uv run slm-lab pull ppo_mujoco
uv run slm-lab pull sac_mujoco
```

## Interrupted Runs (2025-11-26)

Previous runs were interrupted after 12+ hours (spot instance preemption):
- All MuJoCo runs (ppo-hopper, sac-hopper, ppo-walker, sac-walker, ppo-cheetah, sac-cheetah, ppo-ant, sac-ant)
- Relaunched with faster configs on 2025-11-27

## Completed Runs

| Date | Run | Spec | Result | Notes |
|------|-----|------|--------|-------|
| 2025-11-25 | a2c-lunar-asha | a2c_gae_lunar | Best MA=93 @ 300k | ASHA search complete |
| 2025-11-25 | a2c-lunar-val-cpu | a2c_gae_lunar | MA=-72 @ 1M | High variance - ASHA result doesn't reproduce |

## Cost Estimate

Current runs: 10 instances @ $0.0634/hr = ~$0.63/hr total
Expected duration: 3-4 hours
Estimated total cost: ~$2-3
