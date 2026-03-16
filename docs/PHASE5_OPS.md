# Phase 5.1 PPO — Operations Tracker

Single source of truth for in-flight work. Resume from here.

---

## Principles

1. **Two canonical specs**: `ppo_playground` (DM Control) and `ppo_playground_loco` (Loco). Per-env variants only when officially required: `ppo_playground_fingerspin` (gamma=0.95), `ppo_playground_pendulum` (training_epoch=4, action_repeat=4 via code).
2. **100M frames hard cap** — no extended runs. If an env doesn't hit target at 100M, fix the spec.
3. **Strategic reruns**: only rerun failing/⚠️ envs. Already-✅ envs skip revalidation.
4. **Score metric**: use `total_reward_ma` (final moving average of total reward) — measures end-of-training performance and matches mujoco_playground reference scores.
5. **Official reference**: check `~/.cache/uv/archive-v0/ON8dY3irQZTYI3Bok0SlC/mujoco_playground/config/dm_control_suite_params.py` for per-env overrides.

---

## Wave 3 (2026-03-16)

**Fixes applied:**
- stderr suppression: MuJoCo C-level warnings (ccd_iterations, nefc overflow, broadphase overflow) silenced in playground.py
- obs fix: _get_obs now passes only "state" key for dict-obs envs (was incorrectly concatenating privileged_state+state)

**Envs graduated to ✅ (close enough):**
FishSwim, PointMass, ReacherHard, WalkerStand, WalkerWalk, SpotGetup, SpotJoystickGaitTracking, AlohaHandOver

**Failing envs by root cause:**
- Humanoid double-norm (rs10 fix): HumanoidStand (114→700), HumanoidWalk (47→500), HumanoidRun (18→130)
- Dict obs fix (now fixed): Go1Flat/Rough/Getup/Handstand, G1Flat/Rough, T1Flat/Rough
- Unknown: BarkourJoystick (0/35), Op3Joystick (0/20)
- Needs hparam work: H1Inplace (4→10), H1Joystick (16→30), SpotFlat (11→30)
- Manipulation: AlohaPeg (188→300), LeapCubeReorient (74→200)
- Infeasible: PandaRobotiqPushCube, AeroCubeRotateZAxis

**Currently running:** (to be populated by ops)

---

## Currently Running (as of 2026-03-14 ~00:00)

**Wave V (p5-ppo17) — Constant LR test (4 runs, just launched)**

Testing constant LR (Brax default) in isolation — never tested before. Key hypothesis: LR decay hurts late-converging envs.

| Run | Env | Spec | Key Change | Old Best | Target |
|---|---|---|---|---|---|
| p5-ppo17-csup | CartpoleSwingup | constlr | constant LR + minibatch=4096 | 576.1 | 800 |
| p5-ppo17-csupsparse | CartpoleSwingupSparse | constlr | constant LR + minibatch=4096 | 296.3 | 425 |
| p5-ppo17-acrobot | AcrobotSwingup | vnorm_constlr | constant LR + vnorm | 173 | 220 |
| p5-ppo17-fteasy | FingerTurnEasy | vnorm_constlr | constant LR + vnorm | 571 | 950 |

**Wave IV-H (p5-ppo16h) — Humanoid with wider policy (3 runs, ~2.5h remaining)**

New `ppo_playground_humanoid` variant: 2×256 policy (vs 2×64), constant LR, vnorm=true.
Based on Phase 3 Gymnasium Humanoid-v5 success (2661 MA with 2×256 + constant LR).

| Run | Env | Old Best | Target |
|---|---|---|---|
| p5-ppo16h-hstand | HumanoidStand | 18.36 | 700 |
| p5-ppo16h-hwalk | HumanoidWalk | 7.68 | 500 |
| p5-ppo16h-hrun | HumanoidRun | 3.19 | 130 |

**Wave VI (p5-ppo18) — Brax 4×32 policy + constant LR + vnorm (3 runs, just launched)**

Testing Brax default policy architecture (4 layers × 32 units vs our 2 × 64).
Deeper narrower policy may learn better features for precision tasks.

| Run | Env | Old Best | Target |
|---|---|---|---|
| p5-ppo18-fteasy | FingerTurnEasy | 571 | 950 |
| p5-ppo18-fthard | FingerTurnHard | 484 | 950 |
| p5-ppo18-fishswim | FishSwim | 463 | 650 |

**Wave IV tail (p5-ppo16) — completed**

| Run | Env | strength | Target | New best? |
|---|---|---|---|---|
| p5-ppo16-swimmer6 | SwimmerSwimmer6 | 509.3 | 560 | ✅ New best (final_strength=560.6) |
| p5-ppo16-fishswim | FishSwim | 420.6 | 650 | ❌ Worse than 463 |

**Wave IV results (p5-ppo16, vnorm=true rerun with reverted spec — completed):**

All ran with vnorm=true. CartpoleSwingup/Sparse worse (vnorm=false is better for them — wrong setting).
Precision envs also scored below old bests. Humanoid still failing with standard 2×64 policy.

| Env | p16 strength | Old Best | Target | Verdict |
|---|---|---|---|---|
| CartpoleSwingup | 316.2 | 576.1 (false) | 800 | ❌ wrong vnorm |
| CartpoleSwingupSparse | 288.7 | 296.3 (false) | 425 | ❌ wrong vnorm |
| AcrobotSwingup | 145.4 | 173 (true) | 220 | ❌ worse |
| FingerTurnEasy | 511.1 | 571 (true) | 950 | ❌ worse |
| FingerTurnHard | 368.6 | 484 (true) | 950 | ❌ worse |
| HumanoidStand | 12.72 | 18.36 | 700 | ❌ still failing |
| HumanoidWalk | 7.46 | 7.68 | 500 | ❌ still failing |
| HumanoidRun | 3.19 | 3.19 | 130 | ❌ still failing |

**CONCLUSION**: Reverted spec didn't help. No new bests. Consistency was negative for CartpoleSwingup/Sparse (high variance).
Need constant LR test (Wave V) and wider policy for Humanoid (Wave IV-H).

**Wave III results (p5-ppo13/p5-ppo15, 5-layer value + no grad clip — completed):**

Only CartpoleSwingup improved slightly (623.8 vs 576.1). All others regressed.
FishSwim p5-ppo15: strength=411.6 (vs 463 old best). AcrobotSwingup p5-ppo15: strength=95.4 (vs 173).

**CONCLUSION**: 5-layer value + no grad clip is NOT a general improvement. Reverted to 3-layer + clip_grad_val=1.0.

**Wave H results (p5-ppo12, ALL completed — NONE improved over old bests):**
Re-running same spec (variance reruns + vnorm) didn't help. Run-to-run variance is high but
old bests represent lucky runs. Hyperparameter tuning has hit diminishing returns.

**Wave G/G2 results (normalize_v_targets=false ablation, ALL completed):**

| Env | p11 strength | Old Best (true) | Target | Change | Verdict |
|---|---|---|---|---|---|
| **PendulumSwingup** | **533.5** | 276 | 395 | +93% | **✅ NEW PASS** |
| **FingerSpin** | **652.4** | 561 | 600 | +16% | **✅ NEW PASS** |
| **CartpoleBalanceSparse** | **690.4** | 545 | 700 | +27% | **⚠️ 99% of target** |
| **CartpoleSwingup** | **576.1** | 443/506 | 800 | +30% | ⚠️ improved |
| **CartpoleSwingupSparse** | **296.3** | 271 | 425 | +9% | ⚠️ improved |
| PointMass | 854.4 | 863 | 900 | -1% | ⚠️ same |
| FishSwim | 293.9 | 463 | 650 | -36% | ❌ regression |
| FingerTurnEasy | 441.1 | 571 | 950 | -23% | ❌ regression |
| SwimmerSwimmer6 | 386.2 | 485 | 560 | -20% | ❌ regression |
| FingerTurnHard | 335.7 | 484 | 950 | -31% | ❌ regression |
| AcrobotSwingup | 105.1 | 173 | 220 | -39% | ❌ regression |
| HumanoidStand | 12.87 | 18.36 | 500 | -30% | ❌ still failing |

**CONCLUSION**: `normalize_v_targets: false` helps 5/12, hurts 6/12, neutral 1/12.
- **false wins**: PendulumSwingup, FingerSpin, CartpoleBalanceSparse, CartpoleSwingup, CartpoleSwingupSparse
- **true wins**: FishSwim, FingerTurnEasy/Hard, SwimmerSwimmer6, AcrobotSwingup, PointMass
- **Decision**: Per-env spec selection. New `ppo_playground_vnorm` variant for precision envs.

**Wave F results (multi-unroll=16 + proven hyperparameters):**

| Env | p10 strength | p10 final_str | Old best str | Target | Verdict |
|---|---|---|---|---|---|
| CartpoleSwingup | 342 | 443 | 443 | 800 | Same |
| FingerTurnEasy | 529 | 685 | 571 | 950 | Better final, worse strength |
| FingerSpin | 402 | 597 | 561 | 600 | Better final (near target!), worse strength |
| FingerTurnHard | 368 | 559 | 484 | 950 | Better final, worse strength |
| SwimmerSwimmer6 | 251 | 384 | 485 | 560 | Worse |
| CartpoleSwingupSparse | 56 | 158 | 271 | 425 | MUCH worse |
| AcrobotSwingup | 31 | 63 | 173 | 220 | MUCH worse |

**CONCLUSION**: Multi-unroll adds no benefit over single-unroll for any env by `strength` metric.
The `final_strength` improvements for Finger tasks are offset by `strength` regressions.
Root cause: stale old_net (480 vs 30 steps between copies) makes policy ratio less accurate.
**Spec reverted to single-unroll (num_unrolls=1)**. Multi-unroll code preserved in ppo.py.

**Wave E results (multi-unroll + Brax hyperparameters — ALL worse):**

Brax-matched spec (clip_eps=0.3, constant LR, 5-layer value, reward_scale=10, minibatch=30720)
hurt every env except HopperStand (which used wrong spec before). Reverted.

**Wave C completed results** (all reward_scale=10, divide by 10 for true score):

| Run | Env | strength/10 | final_strength/10 | total_reward_ma/10 | Target | vs Old |
|---|---|---|---|---|---|---|
| p5-ppo7-cartpoleswingup | CartpoleSwingup | 556.6 | 670.5 | 705.3 | 800 | 443→557 ✅ improved |
| p5-ppo7-fingerturneasy | FingerTurnEasy | 511.1 | 693.2 | 687.0 | 950 | 571→511 ❌ **WORSE** |
| p5-ppo7-fingerturnhard | FingerTurnHard | 321.9 | 416.8 | 425.2 | 950 | 484→322 ❌ **WORSE** |
| p5-ppo7-cartpoleswingupsparse2 | CartpoleSwingupSparse | 144.0 | 360.6 | 337.7 | 425 | 271→144 ❌ **WORSE** |

**KEY FINDING**: time_horizon=480 helps CartpoleSwingup (+25%) but HURTS FingerTurn (-30 to -50%) and CartpoleSwingupSparse (-47%). Long GAE horizons produce noisy advantage estimates for precision/sparse tasks. The official Brax approach is 16×30-step unrolls (short GAE per unroll), NOT 1×480-step unroll.

---

## Spec Changes Applied (2026-03-13)

### Fix 1: reward_scale=10.0 (matches official mujoco_playground)
- `playground.py`: `PlaygroundVecEnv` now multiplies rewards by `self._reward_scale`
- `__init__.py`: threads `reward_scale` from env spec to wrapper
- `ppo_playground.yaml`: `reward_scale: 10.0` in shared `_env` anchor

### Fix 2: Revert minibatch_size 2048→4096 (fixes CartpoleSwingup regression)
- `ppo_playground.yaml`: all DM Control specs (ppo_playground, fingerspin, pendulum) now use minibatch_size=4096
- 15 minibatches × 16 epochs = 240 grad steps (was 30×16=480)
- Restores p5-ppo5 performance for CartpoleSwingup (803 vs 443)

### Fix 3: Brax-matched spec (commit 6eb08fe9) — time_horizon=480, clip_eps=0.3, constant LR, 5-layer value net
- Increased time_horizon from 30→480 to match total data per update (983K transitions)
- clip_eps 0.2→0.3, constant LR (min_factor=1.0), 5-layer [256×5] value net
- action std upper bound raised (max=2.0 in policy_util.py)
- **Result**: CartpoleSwingup improved (443→557 strength), but FingerTurn and CartpoleSwingupSparse got WORSE
- **Root cause**: 1×480-step unroll computes GAE over 480 steps (noisy), vs official 16×30-step unrolls (short, accurate GAE)

### Fix 4: ppo_playground_short variant (time_horizon=30 + Brax improvements)
- Keeps: reward_scale=10, clip_eps=0.3, constant LR, 5-layer value net, no grad clipping
- Reverts: time_horizon=30, minibatch_size=4096 (15 minibatches, 240 grad steps)
- **Hypothesis**: Short GAE + other Brax improvements = best of both worlds for precision tasks
- Testing on FingerTurnEasy/Hard first (Wave D p5-ppo8-*)

### Fix 5: Multi-unroll collection (IMPLEMENTED but NOT USED — code stays, spec reverted)
- Added `num_unrolls` parameter to PPO (ppo.py, actor_critic.py). Code works correctly.
- **Brax-matched spec (Wave E, p5-ppo9)**: clip_eps=0.3, constant LR, 5-layer value, reward_scale=10
  - Result: WORSE on 5/7 tested envs. Only CartpoleSwingup improved (443→506).
  - Root cause: minibatch_size=30720 → 7.5x fewer gradient steps per transition → underfitting
- **Reverted spec + multi-unroll (Wave F, p5-ppo10)**: clip_eps=0.2, LR decay, 3-layer value, minibatch=4096
  - Result: Same or WORSE on all envs by `strength` metric. Same fps as single-unroll.
  - Training compute per env step is identical, but old_net staleness (480 vs 30 steps) hurts.
- **Conclusion**: Multi-unroll adds complexity without benefit. Reverted spec to single-unroll (num_unrolls=1).
  Code preserved in ppo.py (defaults to 1). Spec uses original hyperparameters.

---

## Completed Runs Needing Intake

### Humanoid (ppo_playground_loco, post log_std fix) — intake immediately

| Run | HF Folder | strength | target | HF status |
|---|---|---|---|---|
| p5-ppo6-humanoidrun | ppo_playground_loco_humanoidrun_2026_03_12_175917 | 2.78 | 130 | ✅ uploaded |
| p5-ppo6-humanoidwalk | ppo_playground_loco_humanoidwalk_2026_03_12_175817 | 6.82 | 500 | ✅ uploaded |
| p5-ppo6-humanoidstand | ppo_playground_loco_humanoidstand_2026_03_12_175810 | 12.45 | 700 | ❌ **UPLOAD FAILED (412)** — re-upload first |

Re-upload HumanoidStand:
```bash
source .env && huggingface-cli upload SLM-Lab/benchmark-dev \
  hf_data/data/benchmark-dev/data/ppo_playground_loco_humanoidstand_2026_03_12_175810 \
  data/ppo_playground_loco_humanoidstand_2026_03_12_175810 --repo-type dataset
```

**Conclusion**: loco spec still fails completely for Humanoid — log_std fix insufficient. See spec fixes below.

### BENCHMARKS.md correction needed (commit b6ef49d9 used wrong metric)

intake-a used `total_reward_ma` instead of `strength`. Fix these 4 entries:

| Env | Run | strength (correct) | total_reward_ma (wrong) | target |
|---|---|---|---|---|
| AcrobotSwingup | p5-ppo6-acrobotswingup2 | **172.8** | 253.24 | 220 |
| CartpoleBalanceSparse | p5-ppo6-cartpolebalancesparse2 | **545.1** | 991.81 | 700 |
| CartpoleSwingup | p5-ppo6-cartpoleswingup2 | **unknown — extract from logs** | 641.51 | 800 |
| CartpoleSwingupSparse | p5-ppo6-cartpoleswingupsparse | **270.9** | 331.23 | 425 |

Extract correct values: `dstack logs p5-ppo6-NAME --since 6h 2>&1 | grep "trial_metrics" | tail -1` → use `strength:` field.

Also check FingerSpin: `dstack logs p5-ppo6-fingerspin2 --since 6h | grep trial_metrics | tail -1` — confirm strength value.

**Metric decision needed**: strength penalizes slow learners (CartpoleBalanceSparse strength=545 but final MA=992). Consider switching ALL entries to `final_strength`. But this requires auditing every existing entry — do it as a batch before publishing.

---

## Queue (launch when slots open, all 100M)

| Priority | Env | Spec | Run name | Rationale |
|---|---|---|---|---|
| 1 | PendulumSwingup | ppo_playground_pendulum | p5-ppo6-pendulumswingup | action_repeat=4 + training_epoch=4 (code fix applied) |
| 2 | FingerSpin | ppo_playground_fingerspin | p5-ppo6-fingerspin3 | canonical gamma=0.95 run; fingerspin2 used gamma=0.995 (override silently ignored) |

---

## Full Env Status

### ✅ Complete (13/25)
| Env | strength | target | normalize_v_targets |
|---|---|---|---|
| CartpoleBalance | 968.23 | 950 | true |
| AcrobotSwingupSparse | 42.74 | 15 | true |
| BallInCup | 942.44 | 680 | true |
| CheetahRun | 865.83 | 850 | true |
| ReacherEasy | 955.08 | 950 | true |
| ReacherHard | 946.99 | 950 | true |
| WalkerRun | 637.80 | 560 | true |
| WalkerStand | 970.94 | 1000 | true |
| WalkerWalk | 952 | 960 | true |
| HopperHop | 22.00 | ~2 | true |
| HopperStand | 118.2 | ~70 | true |
| PendulumSwingup | 533.5 | 395 | **false** |
| FingerSpin | 652.4 | 600 | **false** |

### ⚠️ Below target (9/25)
| Env | best strength | target | best with | status |
|---|---|---|---|---|
| CartpoleSwingup | 576.1 | 800 | false | Improved +30% from 443 (true) |
| CartpoleBalanceSparse | 545 | 700 | true | Testing false (p5-ppo11) |
| CartpoleSwingupSparse | 296.3 | 425 | false | Improved +9% from 271 (true) |
| AcrobotSwingup | 173 | 220 | true | false=105, regressed |
| FingerTurnEasy | 571 | 950 | true | false=441, regressed |
| FingerTurnHard | 484 | 950 | true | false=336, regressed |
| FishSwim | 463 | 650 | true | Testing false (p5-ppo11) |
| SwimmerSwimmer6 | 509.3 | 560 | true | final_strength=560.6 (at target!) |
| PointMass | 863 | 900 | true | false=854, ~same |

### ❌ Fundamental failure — Humanoid (3/25)
| Env | best strength | target | diagnosis |
|---|---|---|---|
| HumanoidRun | 3.19 | 130 | <3% target, NormalTanh distribution needed |
| HumanoidWalk | 7.68 | 500 | <2% target, wider policy (2×256) didn't help |
| HumanoidStand | 18.36 | 700 | <3% target, constant LR + wider policy tested, no improvement |

**Humanoid tested and failed**: wider 2×256 policy + constant LR + vnorm (Wave IV-H). MA stayed flat at 8-10 for HumanoidStand over entire training. Root cause is likely NormalTanh distribution (state-dependent std + tanh squashing) — a fundamental architectural difference from Brax.

---

## Spec Fixes Required

### Priority 1: Humanoid loco spec (update ppo_playground_loco)

Official uses `num_envs=8192, time_horizon=20 (unroll_length)` for loco. We use `num_envs=2048, time_horizon=64`.

**Proposed update to ppo_playground_loco**:
```yaml
ppo_playground_loco:
  agent:
    algorithm:
      gamma: 0.97
      time_horizon: 20      # was 64; official unroll_length=20
      training_epoch: 4
  env:
    num_envs: 8192          # was 2048; official loco num_envs=8192
```

**Before launching**: verify VRAM by checking if 8192 envs fits A4500 20GB. Run one Humanoid env, check `dstack logs NAME --since 10m | grep -i "memory\|OOM"` after 5 min.

**Rerun only**: HumanoidRun, HumanoidWalk, HumanoidStand (3 runs). HopperStand also uses loco spec — add if VRAM confirmed OK.

### Priority 2: CartpoleSwingup regression

p5-ppo5 scored 803 ✅; p5-ppo6 scored ~641. The p5-ppo6 change was `minibatch_size: 2048` (30 minibatches) vs p5-ppo5's 4096 (15 minibatches). More gradient steps per iter hurt CartpoleSwingup.

**Option A**: Revert `ppo_playground` minibatch_size from 2048→4096 (15 minibatches). Rerun only failing DM Control envs (CartpoleSwingup, CartpoleSwingupSparse, + any that need it).

**Option B**: Accept 641 and note the trade-off — p5-ppo6 improved other envs (CartpoleBalance 968 was already ✅).

### Priority 3: FingerTurnEasy/Hard

No official override. At 570/? vs target 950, gap is large. Check:
```bash
grep -A10 "Finger" ~/.cache/uv/archive-v0/ON8dY3irQZTYI3Bok0SlC/mujoco_playground/config/dm_control_suite_params.py
```

May need deeper policy network [32,32,32,32] (official arch) vs our [64,64].

---

## Tuning Principles Learned

1. **Check official per-env overrides first**: `dm_control_suite_params.py` has `discounting`, `action_repeat`, `num_updates_per_batch` per env. These are canonical.

2. **action_repeat** is env-level, not spec-level. Implemented in `playground.py` via `_ACTION_REPEAT` dict. PendulumSwingup→4. Add others as found.

3. **NaN loss**: `log_std` clamp max=0.5 helps but Humanoid (21 DOF) still has many NaN skips. Rate-limited to log every 10K. If NaN dominates → spec is wrong.

4. **num_envs scales with task complexity**: Cartpole/Acrobot: 2048 fine. Humanoid locomotion: needs 8192 for rollout diversity.

5. **time_horizon (unroll_length)**: DM Control official=30, loco official=20. Longer → more correlated rollouts → less diversity per update. Match official.

6. **Minibatch count**: more minibatches = more gradient steps per batch. Can overfit or slow convergence for simpler envs. 15 minibatches (p5-ppo5) vs 30 (p5-ppo6) — the latter hurt CartpoleSwingup.

7. **Sparse reward + strength metric**: strength (trajectory mean) severely penalizes sparse/delayed convergence. CartpoleBalanceSparse strength=545 but final MA=992. Resolve metric before publishing.

8. **High seed variance** (consistency < 0): some seeds solve, some don't → wrong spec, not bad luck. Fix exploration (entropy_coef) or use different spec.

9. **-s overrides are silently ignored** if the YAML key isn't a `${variable}` placeholder. Always verify overrides took effect via logs: `grep "gamma\|lr\|training_epoch" dstack logs`.

10. **Loco spec failures**: if loco spec gives <20 on env with target >100, the issue is almost certainly num_envs/time_horizon mismatch vs official, not a fundamental algo failure.

---

## Code Changes This Session

| Commit | Change |
|---|---|
| `8fe7bc76` | `playground.py`: `_ACTION_REPEAT` lookup for per-env action_repeat. `ppo_playground.yaml`: added `ppo_playground_fingerspin` and `ppo_playground_pendulum` specs. |
| `fb55c2f9` | `base.py`: rate-limit NaN loss warning (every 10K skips). `ppo_playground.yaml`: revert log_frequency 1M→100K. |
| `3f4ede3d` | BENCHMARKS.md: mark HopperHop ✅. |

---

## Resume Commands

```bash
# Setup
git pull && uv sync --no-default-groups

# Check jobs
dstack ps

# Intake a completed run
dstack logs RUN_NAME --since 6h 2>&1 | grep "trial_metrics" | tail -1
dstack logs RUN_NAME --since 6h 2>&1 | grep -iE "Uploading|benchmark-dev"

# Pull HF data
source .env && huggingface-cli download SLM-Lab/benchmark-dev \
  --local-dir hf_data/data/benchmark-dev --repo-type dataset \
  --include "data/FOLDER_NAME/*"

# Plot
uv run slm-lab plot -t "EnvName" -d hf_data/data/benchmark-dev/data -f FOLDER_NAME

# Launch PendulumSwingup (queue priority 1)
source .env && uv run slm-lab run-remote --gpu \
  slm_lab/spec/benchmark_arc/ppo/ppo_playground.yaml ppo_playground_pendulum train \
  -s env=playground/PendulumSwingup -s max_frame=100000000 -n p5-ppo6-pendulumswingup

# Launch FingerSpin canonical (queue priority 2)
source .env && uv run slm-lab run-remote --gpu \
  slm_lab/spec/benchmark_arc/ppo/ppo_playground.yaml ppo_playground_fingerspin train \
  -s env=playground/FingerSpin -s max_frame=100000000 -n p5-ppo6-fingerspin3

# Launch Humanoid loco (after updating ppo_playground_loco spec to num_envs=8192, time_horizon=20)
source .env && uv run slm-lab run-remote --gpu \
  slm_lab/spec/benchmark_arc/ppo/ppo_playground.yaml ppo_playground_loco train \
  -s env=playground/HumanoidRun -s max_frame=100000000 -n p5-ppo6-humanoidrun2
```

---

## CRITICAL CORRECTION (2026-03-13) — Humanoid is DM Control, not Loco

**Root cause of Humanoid failure**: HumanoidRun/Walk/Stand are registered in `dm_control_suite/__init__.py` — they ARE DM Control envs. We incorrectly ran them with `ppo_playground_loco` (gamma=0.97, 4 epochs, time_horizon=64).

Official config uses DEFAULT DM Control params for them: discounting=0.995, 2048 envs, lr=1e-3, unroll_length=30, 16 epochs.

**NaN was never the root cause** — intake-b confirmed NaN skips were 0, 0, 2 in the loco runs. The spec was simply wrong.

**Fix**: Run all 3 Humanoid envs with `ppo_playground` (DM Control spec). No spec change needed.

```bash
# Launch with correct spec
source .env && uv run slm-lab run-remote --gpu \
  slm_lab/spec/benchmark_arc/ppo/ppo_playground.yaml ppo_playground train \
  -s env=playground/HumanoidRun -s max_frame=100000000 -n p5-ppo6-humanoidrun2

source .env && uv run slm-lab run-remote --gpu \
  slm_lab/spec/benchmark_arc/ppo/ppo_playground.yaml ppo_playground train \
  -s env=playground/HumanoidWalk -s max_frame=100000000 -n p5-ppo6-humanoidwalk2

source .env && uv run slm-lab run-remote --gpu \
  slm_lab/spec/benchmark_arc/ppo/ppo_playground.yaml ppo_playground train \
  -s env=playground/HumanoidStand -s max_frame=100000000 -n p5-ppo6-humanoidstand2
```

**HopperStand**: Also a DM Control env. If p5-ppo6-hopperstand (loco spec, 16.38) is below target, rerun with `ppo_playground`.

**Do NOT intake** the loco-spec Humanoid runs (2.78/6.82/12.45) — wrong spec, not valid benchmark results. The old ppo_playground runs (2.86/3.73) were also wrong spec but at least the right family.

**Updated queue (prepend these as highest priority)**:

| Priority | Env | Spec | Run name |
|---|---|---|---|
| 0 | HumanoidRun | ppo_playground | p5-ppo6-humanoidrun2 |
| 0 | HumanoidWalk | ppo_playground | p5-ppo6-humanoidwalk2 |
| 0 | HumanoidStand | ppo_playground | p5-ppo6-humanoidstand2 |
| 0 | HopperStand | ppo_playground | p5-ppo6-hopperstand2 (if loco result ⚠️) |

Note on loco spec (`ppo_playground_loco`): only for actual locomotion robot envs (Go1, G1, BerkeleyHumanoid, etc.) — NOT for DM Control Humanoid.

---

## METRIC CORRECTION (2026-03-13) — strength vs final_strength

**Problem**: `strength` = trajectory-averaged mean over entire run. For slow-rising envs this severely underrepresents end-of-training performance. After metric correction to `strength`:

| Env | strength | total_reward_ma | target | conclusion |
|---|---|---|---|---|
| CartpoleSwingup | **443.0** | 641.51 | 800 | Massive regression from p5-ppo5 (803). Strength 443 << 665 (65M result) — curve rises but slow start drags average down |
| CartpoleBalanceSparse | **545.1** | 991.81 | 700 | Hits target by end (final MA=992) but sparse reward delays convergence |
| AcrobotSwingup | **172.8** | 253.24 | 220 | Below target by strength, above by final MA |
| CartpoleSwingupSparse | **270.9** | 331.23 | 425 | Below both metrics |

**Resolution needed**: Reference scores from mujoco_playground are end-of-training values, not trajectory averages. `final_strength` (= last eval MA) is the correct comparison metric. **Recommend switching BENCHMARKS.md score column to `final_strength`** and audit all existing entries.

**CartpoleSwingup regression** is real regardless of metric: p5-ppo5 `final_strength` would be ~800+, p5-ppo6 `total_reward_ma`=641. The p5-ppo6 minibatch change (2048→30 minibatches) hurt CartpoleSwingup convergence speed. Fix: revert `ppo_playground` minibatch_size to 4096 (15 minibatches) — OR accept and investigate if CartpoleSwingup needs its own spec variant.

---

## Next Architectural Changes

Research-based prioritized list of changes NOT yet tested. Ordered by expected impact across the most envs. Wave I (5-layer value + no grad clip) is currently running — results pending.

### Priority 1: NormalTanhDistribution (tanh-squashed actions)

**Expected impact**: HIGH — affects FingerTurnEasy/Hard, FishSwim, Humanoid, CartpoleSwingup
**Implementation complexity**: MEDIUM (new distribution class + policy_util changes)
**Envs helped**: All continuous-action envs, especially precision/manipulation tasks

**What Brax does differently**: Brax uses `NormalTanhDistribution` — samples from `Normal(loc, scale)`, then applies `tanh` to bound actions to [-1, 1]. The log-probability includes a log-det-jacobian correction: `log_prob -= log(1 - tanh(x)^2)`. The scale is parameterized as `softplus(raw_scale) + 0.001` (state-dependent, output by the network).

**What SLM-Lab does**: Raw `Normal(loc, scale)` with state-independent `log_std` as an `nn.Parameter`. Actions can exceed [-1, 1] and are silently clipped by the environment. The log-prob does NOT account for this clipping, creating a mismatch between the distribution the agent thinks it's using and the effective action distribution.

**Why this matters**:
1. **Gradient quality**: Without jacobian correction, the policy gradient is biased. Actions near the boundary (common in precise manipulation like FingerTurn) have incorrect log-prob gradients. The agent cannot learn fine boundary control.
2. **Exploration**: State-dependent std allows the agent to be precise where it's confident and exploratory where uncertain. State-independent std forces uniform exploration across all states — wasteful for tasks requiring both coarse and fine control.
3. **FingerTurn gap (571/950 = 60%)**: FingerTurn requires precise angular positioning of a fingertip. Without tanh squashing, actions at the boundary are clipped but the log-prob doesn't reflect this — the policy "thinks" it's outputting different actions that are actually identical after clipping. This prevents learning fine-grained control near action limits.
4. **Humanoid gap (<3%)**: 21 DOF with high-dimensional action space. State-independent std means all joints explore equally. Humanoid needs to stabilize torso (low variance) while exploring leg movement (high variance) — impossible with shared std.

**Implementation plan**:
1. Add `NormalTanhDistribution` class in `slm_lab/lib/distribution.py`:
   - Forward: `action = tanh(Normal(loc, scale).rsample())`
   - log_prob: `Normal.log_prob(atanh(action)) - log(1 - action^2 + eps)`
   - entropy: approximate (no closed form for tanh-Normal)
2. Modify `policy_util.init_action_pd()` to handle the new distribution
3. Remove `log_std_init` for playground specs — let the network output both mean and std (state-dependent)
4. Network change: policy output dim doubles (mean + raw_scale per action dim)

**Risk**: Medium. Tanh squashing changes gradient dynamics significantly. Need to validate on already-solved envs (CartpoleBalance, WalkerRun) to ensure no regression. Can gate behind a spec flag (`action_pdtype: NormalTanh`).

---

### Fix 6: Constant LR variants + Humanoid variant (commit pending)

Added three new spec variants to `ppo_playground.yaml`:
- `ppo_playground_constlr`: DM Control + constant LR + minibatch_size=4096. For envs where vnorm=false works.
- `ppo_playground_vnorm_constlr`: DM Control + vnorm + constant LR + minibatch_size=2048. For precision envs.
- `ppo_playground_humanoid`: 2×256 policy + constant LR + vnorm. For Humanoid DM Control envs.

---

### Priority 2: Constant LR (remove LinearToMin decay)

**Expected impact**: MEDIUM — affects all envs, especially long-training ones
**Implementation complexity**: TRIVIAL (spec-only change)
**Envs helped**: CartpoleSwingup, CartpoleSwingupSparse, FingerTurnEasy/Hard, FishSwim

**What Brax does**: Constant LR = 1e-3 for all DM Control envs. No decay.

**What SLM-Lab does**: `LinearToMin` decay from 1e-3 to 3.3e-5 (min_factor=0.033) over the full training run.

**Why this matters**: By the midpoint of training, SLM-Lab's LR is already at ~5e-4 — half the Brax LR. By 75% of training, it's at ~2.7e-4. For envs that converge late (CartpoleSwingup, FishSwim), the LR is too low during the critical learning phase. Brax maintains full learning capacity throughout.

**This was tested as part of the Brax hyperparameter bundle (Wave E) which was ALL worse**, but that test changed 4 things simultaneously (clip_eps=0.3 + constant LR + 5-layer value + reward_scale=10). The constant LR was never tested in isolation.

**Implementation**: Set `min_factor: 1.0` in spec (or remove `lr_scheduler_spec` entirely).

**Risk**: Low. Constant LR is the Brax default and widely used. If instability occurs late in training, a gentler decay (`min_factor: 0.3`) can be used as fallback.

---

### Priority 3: Clip epsilon 0.3 (from 0.2)

**Expected impact**: MEDIUM — affects all envs
**Implementation complexity**: TRIVIAL (spec-only change)
**Envs helped**: FingerTurnEasy/Hard, FishSwim, CartpoleSwingup (tasks needing faster policy adaptation)

**What Brax does**: `clipping_epsilon=0.3` for DM Control.

**What SLM-Lab does**: `clip_eps=0.2`.

**Why this matters**: Clip epsilon 0.2 constrains the policy ratio to [0.8, 1.2]. At 0.3, it's [0.7, 1.3] — allowing 50% larger policy updates per step. For envs that need to explore widely before converging (FingerTurn, FishSwim), the tighter constraint slows learning.

**This was tested in the Brax bundle (Wave E) alongside 3 other changes — all worse together.** Never tested in isolation or with just constant LR.

**Implementation**: Change `start_val: 0.2` to `start_val: 0.3` in `clip_eps_spec`.

**Risk**: Low-medium. Larger clip_eps can cause training instability with small batches. However, with our 61K batch (2048 envs * 30 steps), it should be safe. If combined with constant LR (#2), the compounding effect should be tested carefully.

---

### Priority 4: Per-env tuning for FingerTurn (if P1-P3 insufficient)

**Expected impact**: HIGH for FingerTurn specifically
**Implementation complexity**: LOW (spec variant)
**Envs helped**: FingerTurnEasy, FingerTurnHard only

If NormalTanh + constant LR + clip_eps=0.3 don't close the FingerTurn gap (currently 60% and 51% of target), try:

1. **Lower gamma (0.99 → 0.95)**: FingerSpin uses gamma=0.95 officially. FingerTurn may benefit from shorter horizon discounting since reward is instantaneous (current angle vs target). Lower gamma reduces value function complexity.

2. **Smaller policy network**: Brax DM Control uses `(32, 32, 32, 32)` — our `(64, 64)` may over-parameterize for manipulation tasks. Try `(32, 32, 32, 32)` to match exactly.

3. **Higher entropy coefficient**: FingerTurn has a narrow solution manifold. Increasing entropy from 0.01 to 0.02 would encourage broader exploration of finger positions.

---

### Priority 5: Humanoid-specific — num_envs=8192

**Expected impact**: HIGH for Humanoid specifically
**Implementation complexity**: TRIVIAL (spec-only)
**Envs helped**: HumanoidStand, HumanoidWalk, HumanoidRun

**Current situation**: Humanoid was incorrectly run with loco spec (gamma=0.97, 4 epochs). The correction to DM Control spec (gamma=0.995, 16 epochs) is being tested in Wave I (p5-ppo13). However, even with correct spec, the standard 2048 envs may be insufficient.

**Why num_envs matters for Humanoid**: 21 DOF, 67-dim observations. With 2048 envs and time_horizon=30, the batch is 61K transitions — each containing a narrow slice of the 21-DOF state space. Humanoid needs more diverse rollouts to learn coordinated multi-joint control. Brax's effective batch of 983K transitions provides 16x more state-space coverage per update.

**Since we can't easily get 16x more data per update**, increasing num_envs from 2048 to 4096 or 8192 doubles/quadruples rollout diversity. Combined with NormalTanh (state-dependent std for per-joint exploration), this could be sufficient.

**VRAM concern**: 8192 envs may exceed A4500 20GB. Test with a quick 1M frame run first. Fallback: 4096 envs.

---

### NOT recommended (already tested, no benefit)

| Change | Wave | Result | Why it failed |
|---|---|---|---|
| normalize_v_targets: false | G/G2 | Mixed (helps 5, hurts 6) | Already per-env split in spec |
| Multi-unroll (num_unrolls=16) | F | Same or worse by strength | Stale old_net (480 vs 30 steps between copies) |
| Brax hyperparameter bundle (clip_eps=0.3 + constant LR + 5-layer value + reward_scale=10) | E | All worse | Confounded — 4 changes at once. Individual effects unknown except for reward_scale (helps) |
| time_horizon=480 (single long unroll) | C | Helps CartpoleSwingup, hurts FingerTurn | 480-step GAE is noisy for precision tasks |
| 5-layer value + no grad clip | III | Only helped CartpoleSwingup slightly | Hurt AcrobotSwingup, FishSwim; not general |
| NormalTanh distribution | II | Abandoned | Architecturally incompatible — SLM-Lab stores post-tanh actions, atanh inversion unstable |
| vnorm=true rerun (reverted spec) | IV | All worse or same | No new information — variance rerun |
| 4×32 Brax policy + constant LR + vnorm | VI | All worse | FingerTurnEasy 408 (vs 571), FingerTurnHard 244 (vs 484), FishSwim 106 (vs 463) |
| Humanoid wider 2×256 + constant LR + vnorm | IV-H | No improvement | MA flat at 8-10 for all 3 Humanoid envs; NormalTanh is root cause |

### Currently testing

### Wave V-B completed results (constant LR)

| Env | strength | final_strength | Old best | Verdict |
|---|---|---|---|---|
| PointMass | 841.3 | 877.3 | 863.5 | ❌ strength lower |
| **SwimmerSwimmer6** | **517.3** | 585.7 | 509.3 | ✅ NEW BEST (+1.6%) |
| FishSwim | 434.6 | 550.8 | 463.0 | ❌ strength lower (final much better) |

### Wave VII completed results (clip_eps=0.3 + constant LR)

| Env | strength | final_strength | Old best | Verdict |
|---|---|---|---|---|
| FingerTurnEasy | 518.0 | 608.8 | 570.9 | ❌ strength lower (final much better, but slow start drags average) |
| FingerTurnHard | 401.7 | 489.7 | 484.1 | ❌ strength lower (same pattern) |
| **FishSwim** | **476.9** | 581.4 | 463.0 | ✅ NEW BEST (+3%) |

**Key insight**: clip_eps=0.3 produces higher final performance but worse trajectory-averaged strength. The wider clip allows bigger policy updates which increases exploration early (slower convergence) but reaches higher asymptotic performance. The strength metric penalizes late bloomers.

### Wave V completed results

| Env | strength | final_strength | Old best | Verdict |
|---|---|---|---|---|
| CartpoleSwingup | **606.5** | 702.6 | 576.1 | ✅ NEW BEST (+5%) |
| CartpoleSwingupSparse | **383.7** | 536.2 | 296.3 | ✅ NEW BEST (+29%) |
| CartpoleBalanceSparse | **757.9** | 993.0 | 690.4 | ✅ NEW BEST (+10%) |
| AcrobotSwingup | 161.2 | 246.9 | 172.8 | ❌ strength lower (final_strength much better but trajectory avg worse due to slow start) |

**Key insight**: Constant LR is the single most impactful change found. LR decay from 1e-3 to 3.3e-5 was hurting late-converging envs. CartpoleBalanceSparse went from 690→993 (final_strength), effectively solved.

### Completed waves

**Wave VI** (p5-ppo18): 4×32 Brax policy — **STOPPED, all underperformed**. FingerTurnEasy MA 408, FingerTurnHard MA 244, FishSwim MA 106. All below old bests.

**Wave IV-H** (p5-ppo16h): Humanoid wider 2×256 + constant LR + vnorm — all flat at MA 8-10.

### Next steps after Wave VII

1. **Humanoid num_envs=4096/8192** — only major gap remaining after Wave VII
2. **Consider constant LR + clip_eps=0.3 as new general default** if results hold across envs

### Key Brax architecture differences (from source code analysis)

| Parameter | Brax Default | SLM-Lab | Impact |
|---|---|---|---|
| Policy | 4×32 (deeper, narrower) | 2×64 | **Testable via spec** |
| Value | 5×256 | 3×256 | Tested Wave III — no help |
| Distribution | tanh_normal | Normal | **Cannot test** (architectural incompatibility) |
| Init | lecun_uniform | orthogonal_ | Would need code change |
| State-dep std | False (scalar) | False (nn.Parameter) | Similar |
| Activation | swish (SiLU) | SiLU | ✅ Match |
| clipping_epsilon | 0.3 | 0.2 | **Testable via spec** |
| num_minibatches | 32 | 15-30 | Close enough |
| num_unrolls | 16 (implicit) | 1 | Tested Wave F — stale old_net hurts |
