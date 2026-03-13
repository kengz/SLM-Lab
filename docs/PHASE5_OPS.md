# Phase 5.1 PPO — Operations Tracker

Single source of truth for in-flight work. Resume from here.

---

## Principles

1. **Two canonical specs**: `ppo_playground` (DM Control) and `ppo_playground_loco` (Loco). Per-env variants only when officially required: `ppo_playground_fingerspin` (gamma=0.95), `ppo_playground_pendulum` (training_epoch=4, action_repeat=4 via code).
2. **100M frames hard cap** — no extended runs. If an env doesn't hit target at 100M, fix the spec.
3. **Strategic reruns**: only rerun failing/⚠️ envs. Already-✅ envs skip revalidation.
4. **Score metric**: use `strength` (trajectory-averaged mean) for consistency with all existing BENCHMARKS.md entries. Note: `final_strength` / `total_reward_ma` may be more meaningful for sparse/delayed-reward envs — this inconsistency must be resolved before publishing.
5. **Official reference**: check `~/.cache/uv/archive-v0/ON8dY3irQZTYI3Bok0SlC/mujoco_playground/config/dm_control_suite_params.py` for per-env overrides.

---

## Still Running (as of handover 2026-03-12 ~23:30)

| Run Name | Env | Spec | Est. completion |
|---|---|---|---|
| p5-ppo6-fishswim2 | FishSwim | ppo_playground | ~2h remaining |
| p5-ppo6-fingerturnhard2 | FingerTurnHard | ppo_playground | ~1h remaining |

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

### ✅ Complete
| Env | strength | target |
|---|---|---|
| CartpoleBalance | 968.23 | 950 |
| AcrobotSwingupSparse | 42.74 | 15 |
| BallInCup | 942.44 | 680 |
| CheetahRun | 865.83 | 850 |
| ReacherEasy | 955.08 | 950 |
| ReacherHard | 946.99 | 950 |
| WalkerRun | 637.80 | 560 |
| WalkerStand | 970.94 | 1000 |
| WalkerWalk | 952 | 960 |
| HopperHop | 22.00 | ~2 |

### ⚠️ Below target — spec fix needed
| Env | strength | target | diagnosis |
|---|---|---|---|
| CartpoleSwingup | ~641 (MA) | 800 | **REGRESSION** from p5-ppo5 (803) — p5-ppo6 minibatch change hurt convergence speed |
| CartpoleBalanceSparse | 545 | 700 | Strength misleads (final MA=992); metric decision pending |
| CartpoleSwingupSparse | 271 | 425 | Sparse; consistency=-0.68 (chaotic seeds); hard case |
| AcrobotSwingup | 173 (strength) | 220 | ✅ by final MA (253); metric decision pending |
| HopperStand | 16.38 | ~70 | Loco spec insufficient — same fix as Humanoid |
| FingerSpin | ~561 | 600 | Need canonical ppo_playground_fingerspin (gamma=0.95) run |
| FingerTurnEasy | 571 | 950 | No official override found; may need arch or num_envs change |
| FingerTurnHard | running | 950 | Results pending |
| FishSwim | running | 650 | Was 60M before (curve rising); 100M run in progress |
| PendulumSwingup | 276 | 395 | Queued with action_repeat=4 fix |
| SwimmerSwimmer6 | 485 | 560 | Confirmed plateau; accept or spec investigation |
| PointMass | 863 | 900 | Close; accept or rerun |

### ❌ Fundamental failure — new spec needed
| Env | strength | target | diagnosis |
|---|---|---|---|
| HumanoidRun | 2.78 | 130 | ppo_playground_loco insufficient; needs num_envs=8192 + time_horizon=20 |
| HumanoidWalk | 6.82 | 500 | same |
| HumanoidStand | 12.45 | 700 | same |

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
