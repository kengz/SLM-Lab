# Active Remote Runs

Track dstack runs for continuity. Use `dstack ps` to check status.

## Current Runs (2025-11-26)

### Active ASHA Searches (CPU-only, gpu:0)

| Run Name | Spec | Env | Mode | Hardware | Status | Notes |
|----------|------|-----|------|----------|--------|-------|
| `ppo-bipedal-asha2` | ppo_bipedalwalker_asha | BipedalWalker-v3 | search | GCP e2-standard-8 | running | 3M frames, 20 trials |
| `sac-hopper-asha2` | sac_mujoco | Hopper-v5 | search | GCP e2-standard-8 | running | 2M frames, 30 trials |
| `ppo-cheetah-asha2` | ppo_mujoco | HalfCheetah-v5 | search | GCP e2-standard-8 | running | 3M frames, 50 trials |

## Commands

```bash
# Check status
dstack ps
dstack logs <run-name>

# Pull results when complete
slm-lab pull ppo_bipedalwalker
slm-lab pull sac_mujoco
slm-lab pull ppo_mujoco
```

## Completed Runs

| Date | Run | Spec | Result | Notes |
|------|-----|------|--------|-------|
| 2025-11-25 | a2c-lunar-asha | a2c_gae_lunar | Best MA=93 @ 300k | ASHA search complete |
| 2025-11-25 | a2c-lunar-val-cpu | a2c_gae_lunar | MA=-72 @ 1M | High variance - ASHA result doesn't reproduce |

## Previous Session (interrupted by credits)

The following runs were interrupted when dstack credits ran out:
- ppo-bipedal-asha-cpu, sac-lunar-asha-cpu (Phase 2)
- sac-acrobot-asha-cpu (Phase 1.2)
- ppo-hopper-asha-cpu, ppo-cheetah-asha-cpu, ppo-walker-asha-cpu (Phase 3 PPO)
- sac-hopper-asha-cpu3, sac-cheetah-asha-cpu3, sac-walker-asha-cpu3 (Phase 3 SAC)

## Cost Estimate

Current runs: 3 instances @ $0.0317/hr = ~$0.10/hr total
Expected duration: 4-12 hours depending on env complexity
