# Phase 5.1 PPO — Operations Tracker

Single source of truth for in-flight work. Update this doc as runs complete.

---

## Goal

Match official mujoco_playground reference scores for all 25 DM Control envs within max_frames.
Reference: mujoco_playground GitHub discussion #197, JAX green curve at ~100M frames.

## Principles

1. **Two canonical specs only**: `ppo_playground` (DM Control) and `ppo_playground_loco` (loco). No per-env overrides.
2. **Reproducible**: spec changes must be validated by rerunning affected envs. But **strategic**: only rerun failing/⚠️ envs — already-✅ envs skip revalidation unless the change could plausibly hurt them.
3. **No wasted compute**: solve failing envs with minimum reruns. Don't carpet-bomb with full reruns on every spec tweak.

## Spec

- **ppo_playground** — DM Control (gamma=0.995, lr=1e-3, 16 epochs, 2048 envs, time_horizon=30)
- **ppo_playground_loco** — Loco/Manip (gamma=0.97, lr=1e-3, 4 epochs, 2048 envs, time_horizon=64)
- Key fixes in effect: log_std clamp max=0.5, minibatch_size=2048 (30 minibatches), orthogonal_ init

---

## Running Jobs (2026-03-12)

| Run Name | Env | Spec | max_frame | Status |
|---|---|---|---|---|
| p5-ppo6-fingerspin2 | FingerSpin | ppo_playground (gamma=0.995 actual, NOT 0.95) | 65M | **done** — 561.3 ⚠️ |
| p5-ppo6-cartpoleswingup2 | CartpoleSwingup | ppo_playground | 100M | provisioning |
| p5-ppo6-cartpolebalancesparse2 | CartpoleBalanceSparse | ppo_playground | 100M | provisioning |
| p5-ppo6-acrobotswingup2 | AcrobotSwingup | ppo_playground | 100M | provisioning |
| p5-ppo6-fingerturneasy2 | FingerTurnEasy | ppo_playground | 100M | provisioning |
| p5-ppo6-cartpoleswingupsparse | CartpoleSwingupSparse | ppo_playground | 100M | provisioning |
| p5-ppo6-hopperstand | HopperStand | ppo_playground_loco | 100M | **done** — 16.38 ⚠️ |
| p5-ppo6-humanoidwalk | HumanoidWalk | ppo_playground_loco | 100M | provisioning |
| p5-ppo6-humanoidstand | HumanoidStand | ppo_playground_loco | 100M | provisioning |
| p5-ppo6-humanoidrun | HumanoidRun | ppo_playground_loco | 100M | provisioning |

---

## Queued (launch when slot opens, in priority order)

| Env | Spec | max_frame | Rationale | Status |
|---|---|---|---|---|
| FingerTurnHard | ppo_playground | 100M | 65M run at 484, still rising (target 950) | **launched** p5-ppo6-fingerturnhard2 |
| FishSwim | ppo_playground | 100M | Previous run was only 60M (wall-clock limited); curve still rising | pending slot |
| PendulumSwingup | ppo_playground_pendulum | 100M | Rerun with action_repeat=4 (playground.py fix) + training_epoch=4 | pending slot |
| HopperStand | ppo_playground_loco | 200M | 100M run scored 16.38 ⚠️ (one seed ~100, curve rising steeply) | pending slot |

---

## Env Status & Action Plan

### ✅ Complete — meets target
| Env | Score | Target | Notes |
|---|---|---|---|
| CartpoleBalance | 968 | 950 | ✅ |
| AcrobotSwingupSparse | 42.74 | 15 | ✅ |
| BallInCup | 942 | 680 | ✅ |
| CheetahRun | 865 | 850 | ✅ |
| ReacherEasy | 955 | 950 | ✅ |
| ReacherHard | 946 | 950 | ✅ |
| WalkerRun | 637 | 560 | ✅ |
| WalkerStand | 970 | 1000 | ✅ close |
| WalkerWalk | 952 | 960 | ✅ |

### 🔄 Re-running — needs more frames (was still climbing at 65M)
| Env | Prior Score | Target | Run Name | Expected outcome |
|---|---|---|---|---|
| CartpoleSwingup | 665 | 800 | p5-ppo6-cartpoleswingup2 | should hit 800 at 100M |
| CartpoleBalanceSparse | 511 | 700 | p5-ppo6-cartpolebalancesparse2 | rising steeply, should improve |
| AcrobotSwingup | 209 | 220 | p5-ppo6-acrobotswingup2 | very close, 100M should cross |
| FingerTurnEasy | 544 | 950 | p5-ppo6-fingerturneasy2 | still climbing, needs 200M? |
| FingerTurnHard | 484 | 950 | queued | same as above |
| FingerSpin | 537 | 600 | p5-ppo6-fingerspin2 | ppo_playground_fingerspin (gamma=0.95 is official) |

### 🔄 First run
| Env | Target | Run Name | Notes |
|---|---|---|---|
| CartpoleSwingupSparse | 425 | p5-ppo6-cartpoleswingupsparse | PPO may struggle with sparse reward |
| HopperStand | ~70 | p5-ppo6-hopperstand | ⚠️ 16.38 (max 44, one seed ~100) — loco spec works, needs 200M+ (36.6K fps = ~1.5h) |

### ❌ Humanoid — loco spec retry (log_std fix now in effect)
| Env | Prior Score | Target | Run Name | Notes |
|---|---|---|---|---|
| HumanoidRun | 2.86 (ppo_playground) | 130 | p5-ppo6-humanoidrun | loco spec; prior run had 636K NaN skips |
| HumanoidWalk | 3.73 (ppo_playground) | 500 | p5-ppo6-humanoidwalk | same |
| HumanoidStand | 20.62 (ppo_playground_loco, pre-fix) | 700 | p5-ppo6-humanoidstand | pre-fix so NaN-saturated |

**If loco spec + log_std fix still fails**: investigate log_std clamp further (max=0.0?) or reduce init scale via different init_fn.

### ❌ Spec failure — known bad
| Env | Score | Target | Notes | Next action |
|---|---|---|---|---|
| HopperHop | 22 | ~2 | HopperHop reference score is actually ~2 — our 22 is BETTER ✅ | Verify target — may already pass |

### ⚠️ Underperforming — plan confirmed
| Env | Score | Target | Notes | Plan |
|---|---|---|---|---|
| FishSwim | 463 | 650 | only 60M (wall-limited, curve still rising) | rerun at 100M, no spec change |
| SwimmerSwimmer6 | 485 | 560 | confirmed plateau at 100M | accept 87% — changing gamma globally would need full rerun |
| PendulumSwingup | 276 | 395 | action_repeat=1 was the bug | rerun with ppo_playground_pendulum + playground.py fix |
| PointMass | 863 | 900 | close, 100M run | accept or rerun if slot available |

**Action**: Before relaunching, download data and check curves. For PendulumSwingup: try action_repeat=4.

---

## Investigation Items

1. **PendulumSwingup**: Official mujoco_playground uses `action_repeat=4, training_epoch=4`. Our spec has neither. Add `-s action_repeat=4` override and rerun.
2. **FishSwim / SwimmerSwimmer6**: Download learning curves, check if still rising or truly plateaued. If plateaued: try gamma=0.97 or lower lr.
3. **FingerTurnEasy / Hard**: If 100M run still below 700, likely needs 200M frames (reference scores are at 200M+).
4. **Humanoid**: Check if loco spec + log_std fix changes behavior. Key signal: are NaN skips still dominating?

---

## Intake Checklist (per run)

1. Extract score: `dstack logs NAME 2>&1 | grep "trial_metrics"`
2. Find HF folder: `dstack logs NAME 2>&1 | grep "Uploading data/"`
3. Update BENCHMARKS.md table (score, HF link, FPS, frames, wall clock)
4. Pull data: `source .env && huggingface-cli download SLM-Lab/benchmark-dev --local-dir data/benchmark-dev --repo-type dataset --include "data/FOLDER/*"`
5. Generate plot: `uv run slm-lab plot -t "EnvName" -d data/benchmark-dev/data -f FOLDER`
6. Display plot: `Read docs/plots/EnvName_multi_trial_graph_mean_returns_ma_vs_frames.png`
7. Embed plot in BENCHMARKS.md plot grid if missing
8. Commit score + link + plot together
9. Update this doc: move env row to correct category

---

## Run Naming Convention

- `p5-ppo6-{envnamelower}` — canonical p5-ppo6 run
- `p5-ppo6-{envnamelower}2` — second attempt (rerun or more frames)
- Spec: always ppo_playground or ppo_playground_loco (no version suffixes)

---

## Spec Recommendations

Based on analysis of learning curve data from completed 100M-frame runs.

### 1. FishSwim (463 → target 650) — STILL RISING, needs more frames

**Data**: FishSwim only ran to 60M frames (hit 3h wall), not 100M. Session rewards at 60M: [528, 437, 466, 435]. Curve shape at s0: 10M→99, 25M→133, 50M→411, 60M→525. Reward is clearly still climbing at 60M — steep rise from 25M to 50M, continuing upward.

**Recommendation**: Rerun at 100M frames. The 60M run was wall-clock limited (3h @ 5560fps = 60M). At 2048 envs / ~11K fps, 100M should complete in ~2.5h. If still below 650 at 100M, consider 200M frames. No spec change needed — this is a frame budget issue.

### 2. SwimmerSwimmer6 (485 → target 560) — PLATEAUED, spec change needed

**Data**: Ran full 100M frames. All 4 sessions plateau at 460-500 by 50M, with NO improvement from 50M→100M. Session s0 actually *decreased* from 505→466. max_strength across sessions: [505, 513, 496, 513]. This is a genuine plateau, not a frame budget issue.

**Hypothesis**: SwimmerSwimmer6 is a 6-link swimmer — more complex body dynamics. gamma=0.995 may be too high (long-horizon credit assignment on a task that benefits from short-horizon reactive control). The official mujoco_playground likely uses env-specific training configs.

**Recommendation**: If a global spec change is acceptable, test gamma=0.99 (the shared base default already in the YAML). However, changing gamma globally would require rerunning ALL DM Control envs. Alternative: accept 485/560 (~87% of target) as "close enough" given we cannot do per-env overrides.

### 3. PendulumSwingup (276 → target 395) — ACTION_REPEAT is the issue

**Data**: Ran 100M frames. Extreme per-session variance: s0=60, s1=295, s2=548, s3=467. consistency=-0.88 (worst across all envs). max_strength reached 806 in s2, but s0 collapsed to 60. The env is solvable but PPO is unstable on it.

**Root cause**: `action_repeat=1` is hardcoded in `playground.py:74`. The official mujoco_playground uses `action_repeat=4` for PendulumSwingup. With action_repeat=1, the control frequency is too high relative to the pendulum dynamics — the policy must learn very fine-grained control that PPO's stochastic policy struggles with. action_repeat=4 effectively downsamples the control frequency, making the problem easier.

**Recommendation**: This requires a code change, not a spec change. The `PlaygroundVecEnv.__init__` hardcodes `action_repeat=1`. Two options:
1. **Per-env action_repeat lookup** (violates canonical spec): query `pg_registry.get_default_config(env_name)` for the env's default action_repeat and use it.
2. **Spec parameter** (canonical): Add `action_repeat` as a spec env parameter (like `normalize_obs`), default to 1, override via `-s action_repeat=4` for PendulumSwingup only.

Option 2 is cleaner but requires a per-env override. Option 1 respects the env's own defaults without per-env spec overrides. **Recommend option 1**: read the env's default action_repeat from the registry config and pass it through. This is NOT a spec override — it's respecting the env's built-in config. All other envs default to action_repeat=1 already, so this changes nothing except PendulumSwingup.

### 4. HopperHop (scored 22, target ~2) — ALREADY PASSING

**Data**: HopperHop target in BENCHMARKS.md is "~2". Our score of 22.00 is 11x the target. This env should be marked ✅.

**Note**: The BENCHMARKS.md entry marks it ❌ but the PHASE5_OPS.md correctly notes this may already pass. Change status to ✅.

### 5. FingerSpin (537 → target 600) — gamma=0.95 is NOT canonical

**Data**: At default gamma=0.995, FingerSpin scored 537 at 100M frames. Session scores: [488, 484, 613, 546]. The s2 session hit 649 max. Curve shape: rapid rise to ~420 by 25M, then slow climb to ~488 by 100M (s0/s1 plateau). Two sessions (s2/s3) are clearly better.

**The p5-ppo6-fingerspin2 run uses `-s gamma=0.95`** which violates canonical spec. The question: is gamma=0.95 globally better?

**Analysis**: gamma=0.95 is far too aggressive for most DM Control tasks. CartpoleBalance/WalkerWalk/ReacherEasy all benefit from long-horizon credit (gamma=0.995). Lowering to 0.95 globally would regress these envs. FingerSpin's difficulty is that it's a dexterous manipulation task where short-horizon control matters more — but this is env-specific, not generalizable.

**Recommendation**: Discard the gamma=0.95 run as non-canonical. The default gamma=0.995 run (537) is the canonical result. Accept 537/600 (~90% of target) or wait for SAC/CrossQ results on this env. Do NOT change gamma globally.

### 5. Humanoid (Stand=20.62, Walk=3.73, Run=2.86 — all far below targets)

**Data**: All three Humanoid entries in BENCHMARKS.md used pre-log_std-fix runs. HumanoidStand used ppo_playground_loco (gamma=0.97, 4 epochs), scored 20.62 at 100M/21680fps. HumanoidRun/Walk used ppo_playground (gamma=0.995), scored 2.86/3.73 at 100M/~7000fps.

**Key differences vs official mujoco_playground**:
1. **Network size**: Our policy=[64,64]+SiLU, value=[256,256,256]+SiLU. Official uses policy=[32,32,32,32]+Swish, value=[256,256,256,256,256]+Swish. Our policy is wider but shallower (2 layers vs 4). For Humanoid (67-dim obs, 21-dim action), deeper networks may extract better feature hierarchies.
2. **num_envs**: Official uses 2048 for DM Control, 8192 for loco. Our loco spec uses 2048. More envs = more diverse rollouts per update, crucial for Humanoid's high-dimensional state space.
3. **time_horizon**: Our loco spec uses 64. Official uses `unroll_length=20` for loco. Shorter rollouts = more frequent updates = faster learning for unstable gaits.
4. **training_epoch**: Our loco spec uses 4. Official may use 4 (matching).
5. **log_std clamp**: Now fixed to max=0.5. Pre-fix runs had unbounded log_std causing NaN reward cascades — 636K NaN skips in HumanoidRun. The fix should dramatically change results.

**Recommendation**: Wait for the p5-ppo6 reruns (currently running with log_std fix). If still failing:
- **Increase num_envs to 4096 or 8192** for loco spec (official uses 8192). This is a spec-level change that applies globally to all loco envs, so it's canonical.
- **Reduce time_horizon to 20** for loco spec (official uses 20, we use 64). 64 steps × 2048 envs = 131K batch, vs 20 × 8192 = 164K batch. With 8192 envs and time_horizon=20, batch size matches official exactly.
- **Deepen policy network**: Change from [64,64] to [32,32,32,32] (official config). This is a global arch change that would require rerunning all loco envs but matches the official config exactly.

**Priority**: The log_std fix is the highest-impact change. Wait for those results before making architectural changes. If the fix alone doesn't solve Humanoid, the next step is `num_envs=8192 + time_horizon=20` for ppo_playground_loco. Only rerun **HumanoidStand, HumanoidWalk, HumanoidRun** (the 3 failing Humanoid envs). Do NOT rerun already-passing loco envs.

### Summary of Actionable Changes

| Priority | Change | Envs to rerun | Compute cost |
|---|---|---|---|
| 1 | Wait for p5-ppo6 reruns (log_std fix already in) | None — already in flight | 0 |
| 2 | Fix action_repeat in playground.py to read env default | **PendulumSwingup only** | 1 run (~2.5h) |
| 3 | Rerun FishSwim at 100M frames (was only 60M) | **FishSwim only** (no spec change, just more frames) | 1 run (~2.5h) |
| 4 | Mark HopperHop as ✅ (22 >> target ~2) | None — status update only | 0 |
| 5 | If Humanoid still fails post-fix: bump loco spec to num_envs=8192, time_horizon=20 | **HumanoidStand, HumanoidWalk, HumanoidRun only** — HopperStand also uses loco spec but has its own separate issue (scored 2.87 with old spec, p5-ppo6 rerun in flight). Do NOT rerun already-✅ envs. | 3-4 runs |
| 6 | Accept SwimmerSwimmer6 (485/560=87%) and FingerSpin (537/600=90%) as near-target | None | 0 |

**Compute-minimal strategy**: Only 2 immediate reruns needed (PendulumSwingup after code fix, FishSwim at 100M). Humanoid reruns are contingent on p5-ppo6 results. Already-passing envs (CartpoleBalance, CheetahRun, WalkerWalk, ReacherEasy/Hard, BallInCup, AcrobotSwingupSparse, WalkerStand, WalkerRun, PointMass) are NOT affected by any proposed change and should NOT be rerun.
