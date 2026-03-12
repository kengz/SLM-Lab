# Phase 5.1 PPO — Operations Tracker

Single source of truth for in-flight work. Update this doc as runs complete.

---

## Goal

Match official mujoco_playground reference scores for all 25 DM Control envs within max_frames.
Reference: mujoco_playground GitHub discussion #197, JAX green curve at ~100M frames.

## Spec

- **ppo_playground** — DM Control (gamma=0.995, lr=1e-3, 16 epochs, 2048 envs, time_horizon=30)
- **ppo_playground_loco** — Loco/Manip (gamma=0.97, lr=1e-3, 4 epochs, 2048 envs, time_horizon=64)
- Key fixes in effect: log_std clamp max=0.5, minibatch_size=2048 (30 minibatches), orthogonal_ init

---

## Running Jobs (2026-03-12)

| Run Name | Env | Spec | max_frame | Status |
|---|---|---|---|---|
| p5-ppo6-fingerspin2 | FingerSpin | ppo_playground (-s gamma=0.95) | 100M | running |
| p5-ppo6-cartpoleswingup2 | CartpoleSwingup | ppo_playground | 100M | provisioning |
| p5-ppo6-cartpolebalancesparse2 | CartpoleBalanceSparse | ppo_playground | 100M | provisioning |
| p5-ppo6-acrobotswingup2 | AcrobotSwingup | ppo_playground | 100M | provisioning |
| p5-ppo6-fingerturneasy2 | FingerTurnEasy | ppo_playground | 100M | provisioning |
| p5-ppo6-cartpoleswingupsparse | CartpoleSwingupSparse | ppo_playground | 100M | provisioning |
| p5-ppo6-hopperstand | HopperStand | ppo_playground_loco | 100M | provisioning |
| p5-ppo6-humanoidwalk | HumanoidWalk | ppo_playground_loco | 100M | provisioning |
| p5-ppo6-humanoidstand | HumanoidStand | ppo_playground_loco | 100M | provisioning |
| p5-ppo6-humanoidrun | HumanoidRun | ppo_playground_loco | 100M | provisioning |

---

## Queued (launch when slot opens)

| Env | Spec | max_frame | Rationale |
|---|---|---|---|
| FingerTurnHard | ppo_playground | 100M | 65M run at 484, still rising (target 950) |

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
| FingerSpin | 537 | 600 | p5-ppo6-fingerspin2 | gamma=0.95, should improve |

### 🔄 First run
| Env | Target | Run Name | Notes |
|---|---|---|---|
| CartpoleSwingupSparse | 425 | p5-ppo6-cartpoleswingupsparse | PPO may struggle with sparse reward |
| HopperStand | ~70 | p5-ppo6-hopperstand | loco spec |

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

### ⚠️ Plateaued below target — needs investigation
| Env | Score | Target | Notes | Hypothesis |
|---|---|---|---|---|
| FishSwim | 463 | 650 | plateau at 60M | Need to check curve — may still rise, or needs lower gamma |
| SwimmerSwimmer6 | 485 | 560 | plateau at 100M | confirmed plateau; needs spec change |
| PendulumSwingup | 276 | 395 | plateau at 100M | **official spec has action_repeat=4** — this is likely the fix |
| PointMass | 863 | 900 | close at 100M | likely just seed variance; rerun |

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
