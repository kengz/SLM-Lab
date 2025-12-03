# Active Remote Runs

Track dstack runs for continuity. Use `dstack ps` to check status.

**Budget**: 8 parallel runs | **Reference**: [BENCHMARKS.md](BENCHMARKS.md)

## Current Runs (12 active)

| Run | Algo | Env | Target | Progress | Status |
|-----|------|-----|--------|----------|--------|
| ppo-invpend8 | PPO | InvertedPendulum | 1000 | - | 🔄 ASHA search (RL Zoo hparams) |
| ppo-cheetah3 | PPO | HalfCheetah | 5000 | - | 🔄 ASHA search (RL Zoo hparams) |
| a2c-bipedal3 | A2C | BipedalWalker | 300 | - | 🔄 ASHA search (retrying) |
| a2c-lunar-cont2 | A2C | LunarLander-Cont | 200 | - | 🔄 ASHA search (retrying) |
| sac-lunar | SAC | LunarLander | 200 | - | 🔄 ASHA search |
| ppo-bipedal6 | PPO | BipedalWalker | 300 | - | 🔄 ASHA search (SB3 hparams) |
| sac-bipedal5 | SAC | BipedalWalker | 300 | - | 🔄 ASHA search (SB3 hparams) |
| ppo-invdoubpend6 | PPO | InvertedDoublePend | 9000 | - | 🔄 ASHA search |
| sac-cheetah2 | SAC | HalfCheetah | 5000 | - | 🔄 train run |
| ppo-ant2 | PPO | Ant | 5000 | - | 🔄 ASHA search |
| ppo-walker3 | PPO | Walker2d | 4000 | - | 🔄 ASHA search |

---

## RL Zoo Spec Updates (2025-12-03)

Updated specs with proven RL Zoo hyperparameters:

| Spec | Key Changes |
|------|-------------|
| ppo_inverted_pendulum | gamma=0.999, lam=0.9, clip=0.4, vf_coef=0.2, time_horizon=512, epochs=5, normalize=true, clip_grad=0.3, lr=0.000222 |
| ppo_inverted_double_pendulum | gamma=0.98, lam=0.8, clip=0.4, vf_coef=0.7, time_horizon=2048, batch=512, epochs=10, lr=0.000155 |
| ppo_halfcheetah | gamma=0.98, lam=0.92, clip=0.1, ent=0.0004, vf_coef=0.58, time_horizon=512, epochs=20, clip_grad=0.8, lr=2.06e-5 |
| ppo_walker2d | gamma=0.99, lam=0.95, clip=0.1, ent=0.0006, vf_coef=0.87, time_horizon=512, batch=32, epochs=20, clip_grad=1.0, lr=5.05e-5 |
| ppo_ant | gamma=0.99, lam=0.95, clip=0.2, vf_coef=0.5, [64,64]+tanh, normalize=true, lr=0.0003 |

All specs use: [64,64] + tanh network, normalize=true, orthogonal init

---

## Next Actions (Priority Order)

When runs complete, pick from this queue:

### HIGH PRIORITY - Remaining Gaps

| Priority | Env | Algo | Last Result | Target | Command |
|----------|-----|------|-------------|--------|---------|
| 1 | InvertedPendulum | PPO | ❌ ~25 | 1000 | **LAUNCHED** ppo-invpend8 with RL Zoo hparams |
| 2 | HalfCheetah | PPO | ⚠️ 2513 (50%) | 5000 | **LAUNCHED** ppo-cheetah3 with RL Zoo hparams |

### LOW PRIORITY - Blocked/Long

| Priority | Env | Algo | Notes |
|----------|-----|------|-------|
| - | Humanoid | PPO | Needs 50M frames (too long for 4h limit) |

---

## Quick Reference

```bash
# Monitor
dstack ps                    # List runs
dstack logs <run-name>       # View logs
dstack stop <run-name> -y    # Stop run

# Pull results
uv run slm-lab pull SPEC_NAME

# Dispatch template
source .env && uv run slm-lab run-remote --gpu SPEC_FILE SPEC_NAME <train|search> -n NAME
```

---

## Completed Runs (Recent)

| Date | Run | Result | Notes |
|------|-----|--------|-------|
| 2025-12-03 | ppo-invpend7 | ❌ MA=23.3 | Target 1000 - FAILED (2.3%), still not learning |
| 2025-12-03 | a2c-bipedal2 | ❌ MA=148.8 | Target 300 - FAILED (49.6%), best trial only |
| 2025-12-02 | ddqn-per-cartpole2 | ✅ MA=430.4 | Target 400 - PASSED! Phase 1 complete! |
| 2025-12-02 | sac-bipedal3 | ❌ MA=-106 | Target 300 - FAILED, all trials negative (SAC needs more frames/tuning) |
| 2025-12-02 | dqn-acrobot2 | ✅ MA=-79.5 | Target -100 - PASSED! Tuned spec: gamma=0.987, end_val=0.027, end_step=20000, lr=0.00108 |
| 2025-12-02 | ppo-lunar-cont-val | ✅ MA=233 | Target 200 - PASSED! Validated tuned spec |
| 2025-12-02 | ppo-bipedal3 | ⚠️ MA=215 | Target 300 (72%) - needs more tuning |
| 2025-12-02 | ppo-cheetah2 | ⚠️ MA=2513 | Target 5000 (50%) - train run completed, needs search |
| 2025-12-02 | ppo-lunar-cont3 | ✅ MA=230 | Target 200 - PASSED! Tuned spec: gamma=0.987, lam=0.963, lr=0.000175/0.000758 |
| 2025-12-02 | ppo-lunar-cont2 | ⚠️ MA=177, max=455 | Target 200 - CLOSE (88.5%), best: gamma=0.98, [128,64], relu |
| 2025-12-02 | sac-hopper | ⚠️ MA=543 @ 460k | Target 3000 - stopped to prioritize PPO (18% progress) |
| 2025-12-02 | sac-cheetah | ⚠️ MA=284 @ 350k | Target 5000 - stopped to prioritize PPO (6% progress) |
| 2025-12-02 | ppo-lunar-cont | ⚠️ MA=174, max=411 | Target 200 - CLOSE (87%), needs search |
| 2025-12-02 | sac-bipedal | ❌ MA=-106, max=38 | Target 300 - FAILED, 8 trials all negative |
| 2025-12-02 | ppo-lunar2 | ✅ MA=198.5, max=449 | Target 200 - PASSED with entropy=0.00001 |
| 2025-12-02 | a2c-lunar-cont | ❌ MA=-15, max=349 | Target 200 - FAILED, needs different approach |
| 2025-12-02 | ppo-walker2d | ❌ ~198 @ 900k | Stopped - wrong spec (ppo_mujoco), restarted as ppo-walker2d2 |
| 2025-12-02 | a2c-lunar | ❌ max=300, final=108 | Target 200 - unstable; relaunched as a2c-lunar2 with entropy 0.00001 |
| 2025-12-02 | a2c-lunar2 | ✅ max=430, final=304 | Target 200 - PASSED with fixed entropy! |
| 2025-12-02 | ppo-invpend5 | ⚠️ stuck ~12 @ 455k | Target 1000 - wrong network size; relaunched as ppo-invpend6 |
| 2025-12-02 | ppo-invpend6 | ❌ max=25 | Target 1000 - still failing, needs num_envs=1 investigation |
| 2025-12-02 | pposil-acrobot2 | ✅ -83 | Target -100 - PASSED |
| 2025-12-02 | ppo-ant | ❌ -528 | Stopped - old spec |
| 2025-12-02 | ppo-bipedal2 | ❌ -108 | Stopped - not converging |
| 2025-12-01 | ppo-hopper-val4 | ✅ 3073 | Target 3000 - PASSED |
| 2025-12-01 | ppo-reacher-val3 | ✅ -0.30 | Target -5 - PASSED |
| 2025-12-01 | ppo-swimmer-asha | ✅ 51 | Target 100 - PASSED |
| 2025-12-01 | ppo-bipedal-asha | ⚠️ 227.6 | Target 300 - close |
| 2025-12-01 | ppo-cheetah-asha5 | ⚠️ 2189 | Target 5000 - needs tuning |

---

## Key Findings

- **RL Zoo hyperparams**: Using proven hyperparameters from rl-baselines3-zoo for MuJoCo envs
- **Low entropy (0.00001)**: Critical for both A2C and PPO stability - validated on LunarLander, now testing on MuJoCo
- **normalize_v_targets=true**: Critical for MuJoCo training stability
- **CleanRL defaults**: ent_coef=0.0, gamma=0.99, lam=0.95, [64,64]+tanh
- **Dedicated specs**: ppo_walker2d.json, ppo_ant.json created with proven hyperparameters
- **Trial cleanup**: Only top 3 trials keep models (verified)
- **SAC BipedalWalker**: FAILED badly (-106 MA), all 8 trials stayed negative - needs longer training or different approach
- **Search space sizing**: ~3-4 trials per dimension minimum; narrowed Phase 1-2 specs (dqn_acrobot, ppo_bipedalwalker)
- **DDQN+PER CartPole**: Fixed spec using DQN boltzmann params - local validation MA=471 (target 400) ✅
- **SAC LunarLander (discrete)**: Added back - SAC can solve discrete envs via Categorical policy
- **A2C LunarLander-Cont**: Added back - needs investigation
- **BipedalWalker**: Solvable by SAC (SB3 gets 300.53) and PPO - our hyperparams need tuning
- **A2C BipedalWalker**: Latest run reached ~180 (60% of target 300) - solvable with tuning
- **PPO InvertedPendulum**: Still failing (MA=23 vs target 1000) - now testing RL Zoo hparams

---

## Phase Progress Summary

| Phase | Status | Passed | Remaining |
|-------|--------|--------|-----------|
| 1: Classic Control | ✅ 100% | CartPole✅ Acrobot✅ DQN(ε)✅ DDQN+PER✅ | COMPLETE! |
| 2: Box2D | 🔄 60% | LunarLander✅ A2C LunarLander✅ | BipedalWalker (PPO 72%, SAC ❌) |
| 3: MuJoCo PPO | 33% | Hopper✅ Reacher✅ Swimmer✅ | HalfCheetah, Walker2d, Ant, InvPend, InvDoubPend |
| 3: MuJoCo SAC/A2C | 0% | - | Waiting for PPO baseline |
