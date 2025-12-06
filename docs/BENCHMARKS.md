# SLM-Lab Benchmarks

Systematic algorithm validation across Gymnasium environments.

**Status**: Phase 3 MuJoCo in progress | **Started**: 2025-10-10 | **Updated**: 2025-12-08

---

## Active Runs

Track dstack runs for continuity. Use `dstack ps` to check status.

**Budget**: 12 parallel runs

### Current Runs (1 active)

| Run Name | Command | Status | Notes |
|----------|---------|--------|-------|
| ppo-cheetah-val | `run-remote --gpu ppo_halfcheetah.json train` | running | Validation with best search params (gamma=0.995, lam=0.969) |

**Latest status (2025-12-09 00:16):**
- **ppo-cheetah-val**: Just launched - validation run with best hyperparams from search
- **Parked envs**: InvertedPendulum, InvertedDoublePendulum, Humanoid, HumanoidStandup (need investigation)

### Notes

- **Bug fixed (commit 16d4404e)**: TrackReward now wraps base env BEFORE NormalizeReward
  - Reported metrics show raw interpretable rewards
  - Agent still sees normalized rewards for training

### Key Findings: Normalization A/B Test (2025-12-06)

| Env | Raw MA | Norm MA | Improvement | Target | Status |
|-----|--------|---------|-------------|--------|--------|
| LunarLander | 254 | 235 | -7% | 200 | ✅ Both pass |
| Hopper | 284 | 1014 | **+257%** | 3000 | ❌ Both fail (norm 3.5x better) |
| HalfCheetah | 539 | 1300 | **+141%** | 5000 | ❌ Both fail (norm 2.4x better) |

**Conclusion**: Normalization significantly helps MuJoCo locomotion but requires longer training or better hyperparams to reach targets.

### Quick Reference

```bash
# Monitor
dstack ps                    # List runs
dstack logs <run-name>       # View logs
dstack stop <run-name> -y    # Stop run

# Pull results
source .env && uv run slm-lab pull SPEC_NAME

# Dispatch template
source .env && uv run slm-lab run-remote --gpu SPEC_FILE SPEC_NAME search -n NAME
```

### Completed Runs (Recent)

| Date | Run | Result | Notes |
|------|-----|--------|-------|
| 2025-12-09 | ppo-cheetah-search2 | ⚠️ **MA=2615** | Best: gamma=0.995, lam=0.969, actor_lr=3.56e-4, critic_lr=4.24e-4 (52% target) |
| 2025-12-08 | ppo-hopper-val | ⚠️ **MA=2517** | Validation run 78% of target (3232), entropy=5.19 |
| 2025-12-08 | ppo-walker-search2 | ⚠️ **MA=1997** | Best: gamma=0.994, lam=0.955, actor_lr=3.9e-4, critic_lr=3.5e-4 (45% target) |
| 2025-12-08 | ppo-ant-search2 | ❌ **MA=-105** | Best trial still negative, needs more investigation |
| 2025-12-08 | ppo-cheetah-val2 | ✅ **MA=1468** | HalfCheetah validation passed after clip_vloss=False fix |
| 2025-12-08 | ppo-hopper-search | ⏹️ Interrupted | Best MA=2025 at 1.37M (gamma=0.994, lam=0.967) |
| 2025-12-08 | ppo-walker-search | ⏹️ Interrupted | Best MA=79 at 380k - needs more training |
| 2025-12-08 | ppo-ant-search | ⏹️ Interrupted | Best MA=-161 at 2M - still negative |
| 2025-12-08 | ppo-cheetah-logstd | ❌ MA=-581 | **log_std A/B FAIL**: baseline 5259, log_std catastrophically worse |
| 2025-12-08 | ppo-hopper-logstd | ❌ MA=299 | **log_std A/B FAIL**: search best 356 without log_std |
| 2025-12-08 | ppo-invpend-fix | ❌ MA=7.12 | log_std clamp fix but **network still wrong** (out_features=2) - architecture bug |
| 2025-12-08 | ppo-invdoublepend-fix | ❌ MA=36.35 | log_std clamp fix but network still wrong - architecture bug |
| 2025-12-08 | ppo-hopper-fix | ⚠️ MA=913 | 30% of target (3000). Needs more training or hyperparameter tuning |
| 2025-12-07 | ppo-invpend-ent003 | ❌ MA=2.3 | ent_coef=0.003 + num_envs=16, entropy stable (0.92) but no learning |
| 2025-12-07 | ppo-invdoublepend-ent003 | ❌ MA=25.7 | ent_coef=0.003 + num_envs=16, entropy 0.28 (low), no learning |
| 2025-12-07 | ppo-invpend-norm | ❌ MA=2.18 | ent_coef=0.001 too small, entropy collapsed to -0.15 |
| 2025-12-07 | ppo-invdoublepend-norm | ❌ Network | Network timeout during uv install (infra failure) |
| 2025-12-07 | ppo-invpend-logstd | ❌ MA=2.95 | log_std_init=0.0 + ent=0.0, entropy collapsed to 0.52 |
| 2025-12-07 | ppo-invdoublepend-logstd | ❌ MA=36.4 | log_std_init=0.0 + ent=0.0, entropy stayed high (3.42) |
| 2025-12-07 | ppo-invpend | ❌ MA=5.84 | Original spec entropy collapsed (-0.4), loss=5.92e+05 |
| 2025-12-07 | ppo-invdoublepend | ❌ MA=21.28 | Original spec entropy collapsed (-10.9), loss=5.38e+07 |
| 2025-12-07 | ppo-invpend-clipnorm | ❌ MA=6.83 | clip_vloss=True + normalize_obs/reward (incorrect hyperparams, missing normalize_v_targets) |
| 2025-12-07 | ppo-invdoublepend-clipnorm | ❌ MA=28.18 | clip_vloss=True + normalize_obs/reward (incorrect hyperparams, missing normalize_v_targets) |
| 2025-12-07 | ppo-invpend-raw | ❌ MA=2.69 | Raw config (no env norm): loss=2.8e+05, entropy=0.91 |
| 2025-12-07 | ppo-invdoublepend-raw | ❌ MA=29.76 | Raw config (no env norm): loss=2.59e+04, entropy=1.53 |
| 2025-12-07 | ppo-invpend-netout | ❌ MA=6.5 | Network-output std with num_envs=8: loss exploding (1.19e+03), entropy stable (1.46) |
| 2025-12-07 | ppo-invdoublepend-netout | ❌ MA=35 | Network-output std with num_envs=8: loss exploding (4.53e+04), entropy stable (1.25) |
| 2025-12-07 | ppo-invpend-cleanrl | ❌ MA=4 | CleanRL-style (log_std=0, ent_coef=0): entropy stable (0.27) but MA declining |
| 2025-12-07 | ppo-invdoublepend-cleanrl | ❌ MA=24 | CleanRL-style: entropy collapsed (-0.49) at 180k frames despite no tight clamp |
| 2025-12-07 | ppo-invpend-final | ❌ MA=5 | clamp [-1.5,2] + ent_coef=0.05: entropy stuck at -0.08, no learning |
| 2025-12-07 | ppo-invdoublepend-final | ❌ MA=38 | clamp [-1.5,2] + ent_coef=0.05: entropy stuck at -0.08, no progress |
| 2025-12-07 | ppo-invdoublepend-tight | ❌ MA=26 | clamp [-2,2] entropy stable (-0.58) but no learning - std too tight |
| 2025-12-07 | ppo-invpend-tight | ❌ MA=13 | clamp [-2,2] entropy stable (-0.58) but no learning - std too tight |
| 2025-12-07 | ppo-invdoublepend-clamp | ❌ MA=42 | clamp [-5,2] too loose - entropy collapsed (-2.7), loss exploded (6.7e+07) |
| 2025-12-07 | ppo-invpend-clamp | ❌ MA=20 | clamp [-5,2] too loose - entropy collapsed (-2.7), loss exploded (10^7) |
| 2025-12-07 | ppo-invdoublepend-sb3 | ❌ MA=29 | log_std=-2.0 + ent_coef=0.02, entropy still collapsed (-1.1), loss exploded (9k) |
| 2025-12-07 | ppo-invpend-sb3 | ❌ MA=33 | log_std=-2.0 + ent_coef=0.02, entropy stable (-0.95) but no learning (MA stuck) |
| 2025-12-07 | ppo-invdoublepend-combo | ❌ MA=28 | log_std=-0.5 + ent_coef=0.01, entropy collapsed (-2.05), loss exploded (10^7) |
| 2025-12-07 | ppo-invpend-highent | ❌ MA=3 | log_std=-0.5 + ent_coef=0.05, entropy EXPLODED (5.83), ent_coef too strong |
| 2025-12-07 | ppo-invpend-combo | ❌ MA=8 | log_std=-0.5 + ent_coef=0.01, entropy collapsed (-0.9), needs higher ent_coef |
| 2025-12-07 | ppo-invpend-lowstd | ❌ MA=6 | log_std=-0.5 only, entropy drifted up (1.08), no learning |
| 2025-12-07 | ppo-invdoublepend-lowstd | ❌ MA=25.8 | log_std=-0.5 only, entropy still collapsed (-0.37) |
| 2025-12-07 | ppo-invpend-cleanrl | ❌ MA=2.26 | CleanRL params, entropy too high (2.9), no learning |
| 2025-12-07 | ppo-invdoublepend-cleanrl | ❌ MA=27.3 | CleanRL params, entropy collapse (-1.6), loss explosion (10^7) |
| 2025-12-07 | ppo-pendulum-fix1 | ❌ MA=-1214 | target -200, needs more training or better params |
| 2025-12-07 | ppo-invdoublepend-fix3 | ❌ MA=30.5 | entropy collapsed, only 0.3% of target |
| 2025-12-07 | ppo-invpend-fix3 | ❌ MA=13.5 | entropy collapsed to -2, stopped early |
| 2025-12-07 | ppo-ant-fix2 | ⚠️ **MA=2301** | 46% target at 5M frames - huge improvement from -12! |
| 2025-12-07 | ppo-hopper-search4 | ⚠️ MA=1414 | 47% target, best: gamma=0.993, lam=0.949, lr=4.1e-4 |
| 2025-12-07 | ppo-invpend-logstd | ❌ MA=13.4 | log_std_init search failed, best trial only 1.3% of target |
| 2025-12-07 | ppo-invdoublepend-logstd | ❌ MA=49.7 | log_std_init search failed, best trial only 0.5% of target |
| 2025-12-07 | ppo-cheetah-val4 | ✅ **MA=5259** | **SOLVED** (105% of target)! 5M frames validation run |
| 2025-12-07 | ppo-walker-val3 | ⚠️ MA=1765 | 35% of target at 5M frames, needs more tuning |
| 2025-12-07 | ppo-invpend-logstd1 | ❌ MA=4.51 | log_std_init=0 train, entropy still collapsed (-2.94) |
| 2025-12-07 | ppo-invpend-fix2 | ❌ MA=4.38 | entropy collapsed (-4.3) |
| 2025-12-07 | ppo-invdoublepend-fix2 | ❌ MA=29.28 | entropy stable (1.15) but not solving |
| 2025-12-07 | ppo-cheetah-search3 | ⚠️ **MA=4751** | **95%** of target! Best: gamma=0.981, lam=0.945, lr=1.5e-4 |
| 2025-12-07 | ppo-walker-search3 | ⚠️ MA=3428 | 69% of target. Best: gamma=0.986, lam=0.924, lr=3.7e-4 |
| 2025-12-07 | ppo-bipedal-search1 | ⚠️ MA=203 | 68% of target. Best: gamma=0.993, lam=0.951, lr=1.0e-4 |
| 2025-12-07 | ppo-hopper-search3 | ⚠️ MA=925 | 31%, regressed from 90%. Different params needed |
| 2025-12-07 | sac-lunar-val2 | ❌ MA=-867 | SAC Continuous failing badly |
| 2025-12-07 | ppo-invpend-search1 | ❌ MA=14.7 | Architecture issue (target 1000) |
| 2025-12-07 | ppo-invdoublepend-search1 | ❌ MA=65.6 | Architecture issue (target 9100) |
| 2025-12-07 | ppo-humanoid-search1 | ⏹️ terminated | 4h limit reached |
| 2025-12-07 | ppo-ant-search1 | ⏹️ terminated | 4h limit reached |
| 2025-12-06 | sac-lunar3 | ⚠️ **MA=247** | **ABOVE TARGET** (200), interrupted. Best: freq=40, iter=20, lr=8.8e-4 |
| 2025-12-06 | ppo-walker-norm2 | ⚠️ MA=2807 | 56% at 4.6M (interrupted). Best: gamma=0.984, lam=0.928, lr=8.5e-4 |
| 2025-12-06 | ppo-hopper-norm2 | ⚠️ MA=2710 | 90% at 3M (4h limit), best: gamma=0.999, lam=0.905, lr=5.8e-4 |
| 2025-12-06 | ppo-cheetah-norm2 | ⚠️ MA=2712 | max=3825 at 3M (4h limit), best: gamma=0.987, lam=0.957, lr=1.7e-4 |
| 2025-12-06 | ppo-swimmer2 | ✅ MA=266 | **SOLVED** with RL Zoo params (gamma=0.9998, lam=0.965) |
| 2025-12-06 | ppo-bipedal-ext | ⚠️ MA=225 | Target 300, 75% |
| 2025-12-06 | ppo-bipedal-norm | ⚠️ MA=184 | Target 300, 61% |
| 2025-12-06 | ppo-invpend2 | ❌ MA=7.6 | RL Zoo params worse (target 1000) |
| 2025-12-06 | ppo-invdoublepend2 | ❌ MA=50.1 | RL Zoo params similar (target 9100) |
| 2025-12-06 | ppo-swimmer-norm | ❌ MA=40.3 | Old params, superseded by ppo-swimmer2 |
| 2025-12-06 | sac-lunar2 | ❌ MA=-69.5 | Fixed config still failing |
| 2025-12-06 | ppo-invpend-norm | ❌ MA=23.6 | Target 1000, far off |
| 2025-12-06 | ppo-invdoublepend-norm | ❌ MA=57.2 | Target 9100, far off |
| 2025-12-06 | ppo-walker-norm | ⚠️ MA=577 | Target 5000, extended to 5e6 |
| 2025-12-06 | ppo-ant-norm | ❌ MA=-12.5 | Target 5000, needs investigation |
| 2025-12-06 | ppo-lunar-raw | ✅ MA=254 | Validation passed (target 200) |
| 2025-12-06 | ppo-lunar-norm | ✅ MA=235 | Norm A/B test passed (target 200) |
| 2025-12-06 | ppo-hopper-raw | ❌ MA=284 | Raw baseline (target 3000) |
| 2025-12-06 | ppo-hopper-norm | ⚠️ MA=1014 | Norm 3.5x better, extended to 5e6 |
| 2025-12-06 | ppo-cheetah-raw | ❌ MA=539 | Raw baseline (target 5000) |
| 2025-12-06 | ppo-cheetah-norm | ⚠️ MA=1300 | Norm 2.4x better, extended to 5e6 |
| 2025-12-05 | ppo-hopper11 | ❌ KeyError | Spec had no search block |
| 2025-12-05 | sac-lunar1 | ❌ MA=22.7 | Failed (target 200) |
| 2025-12-04 | ppo-pusher1 | ✅ MA=-0.75 | Pusher-v5 solved |
| 2025-12-04 | ppo-reacher2 | ✅ MA=-0.003 | Reacher-v5 solved |

### 11 PPO MuJoCo Environments Summary

| Env | max_frame | grace_period | Difficulty | Best (norm) | Target | Notes |
|-----|-----------|--------------|------------|-------------|--------|-------|
| InvertedPendulum-v5 | 1e6 | 1e5 | Easy | ⏸️ 14.7 | 1000 | **PARKED** - needs investigation |
| InvertedDoublePendulum-v5 | 1e6 | 1e5 | Easy | ⏸️ 65.6 | 9100 | **PARKED** - needs investigation |
| Swimmer-v5 | 1e6 | 1e5 | Easy | ✅ 266 | 130 | **SOLVED** (RL Zoo params) |
| Reacher-v5 | 1e6 | 1e5 | Easy | ✅ -0.003 | -5 | Solved |
| Pusher-v5 | 1e6 | 1e5 | Easy | ✅ -0.75 | -20 | Solved |
| Hopper-v5 | 3e6 | 2e5 | Medium | ⚠️ 2517 | 3232 | 78%, validation run |
| HalfCheetah-v5 | 3e6 | 2e5 | Medium | 🔄 2615 | 5086 | 51%, validation running |
| Walker2d-v5 | 3e6 | 2e5 | Medium | ⚠️ 1997 | 4405 | 45%, needs more tuning |
| Ant-v5 | 3e6 | 2e5 | Medium | ❌ -105 | 5086 | Still negative, needs investigation |
| Humanoid-v5 | 50e6 | 1e6 | Hard | ⏸️ - | 6000 | **PARKED** - needs 50M frames |
| HumanoidStandup-v5 | 50e6 | 1e6 | Hard | ⏸️ - | 100000 | **PARKED** - needs 50M frames |

### Key Notes

- **Normalization helps**: A/B test shows norm is 2-4x better for locomotion envs
- **Search space**: 3 params only (gamma, lam, lr) for efficient 16-trial coverage
- **4h dstack limit**: Runs terminate at ~3-4M frames; need train mode for full runs
- **Bug fixed (2025-12-06)**: TrackReward now before NormalizeReward - reports raw rewards

### Known Issues & Investigation Queue

| Issue | Envs Affected | Root Cause | Fix Status |
|-------|---------------|------------|------------|
| Value loss explosion | InvPend, InvDoublePend | _norm specs had wrong hyperparams (missing normalize_v_targets, wrong gamma/lam/clip_eps) | 🔄 Testing: original tuned specs with RL Zoo-like hyperparams (normalize_v_targets=true, gamma=0.999, lam=0.9, clip_eps=0.4) |
| Action scaling | Pendulum, InvPend | Bounds not [-1,1] but no auto-scaling | ✅ Fixed: automatic RescaleAction wrapper |
| Wrong hyperparams | Ant | Mismatched RL Zoo params | ✅ Fixed: RL Zoo tuned params (gamma=0.98, lam=0.8, etc) |
| SAC alpha collapse | LunarLander-v3 | Alpha goes to ~0 | ❌ Deprioritized (focus on MuJoCo) |
| Hopper regression | Hopper | Hyperparams inconsistent | 🔄 Searching |

### Architecture Investigation (2025-12-06 → 2025-12-08)

**Why InvertedPendulum/DoublePendulum fail despite being "easy":**

CleanRL (gets 963 on InvertedPendulum) vs SLM-Lab (gets 7-24):
1. **log_std parameterization**: CleanRL uses separate learnable parameter (init=0), SLM-Lab learns through network head
2. **Actor output init**: CleanRL uses std=0.01 for actor output layer, SLM-Lab uses same orthogonal gain as hidden layers (~1.67)
3. **Network size**: CleanRL uses [64, 64], SLM-Lab uses [256, 256]

These differences may cause initial exploration instability for simple envs that are highly sensitive to action scale.

**Fixes implemented:**
1. ✅ Added `actor_out_init_std` option to MLPNet - initializes actor output layer with small std (0.01) (commit 2d08d3a1)
2. ✅ Updated InvPend/InvDoublePend specs with smaller network [64, 64] (like CleanRL)
3. ✅ Separate `log_std_init` parameter implemented (commit 1017dd98) - learnable log_std like CleanRL
   - Works for PPO, SAC, ActorCritic, Reinforce
   - Also fixed SAC 1D action tensor bug that prevented SAC from working on Pendulum envs
4. ✅ **Automatic action rescaling** (commit 63e6f501) - RescaleAction wrapper for envs with bounds != [-1, 1]
   - Pendulum ([-2, 2]) and InvertedPendulum ([-3, 3]) now automatically rescaled to [-1, 1]
   - Policy can output in standard range, wrapper scales to actual env bounds
5. ✅ **Entropy coefficient** added to pendulum specs (entropy_coef=0.01) to prevent entropy collapse
   - Without entropy bonus, log_std keeps decreasing → deterministic policy
   - Local test shows entropy stays positive (0.77 vs -1.54 before)
6. ✅ **Network output dimension bug** (commit fdc1916b) - CRITICAL FIX
   - **Bug**: When using `log_std_init`, network still output 2 values (loc+scale) for 1D continuous actions
   - **Impact**: Scale output was ignored (since we use separate log_std), wasting 50% of network capacity
   - **Symptom**: Network showed `out_features=2` when it should be `out_features=1`; loss:nan, entropy:nan
   - **Fix**: Added `use_log_std` parameter to `get_policy_out_dim()` to output just `action_dim` (loc only)
   - **Files changed**: `net_util.py`, `actor_critic.py`, `policy_util.py`, `sac.py`

---

## Progress

**Completion = all algorithms reaching target (100% of target reward).**

| Phase | Category | Envs | PPO | DQN | A2C | SAC | Overall |
|-------|----------|------|-----|-----|-----|-----|---------|
| 1.1-1.2 | Classic Control (Discrete) | 2 | ✅ | ✅ | ✅ | ✅ | ✅ 100% |
| 1.3 | Classic Control (Continuous) | 1 | 🔄 | N/A | N/A | 🔄 | 🔄 0% |
| 2 | Box2D | 2 | ✅ | ✅ | ✅ | ✅ | ✅ 100% |
| 3 | MuJoCo | 11 | 🔄 | N/A | ⏸️ | ⏸️ | 🔄 25% |
| 4 | Atari | 6+ | ⏸️ | ⏸️ | N/A | N/A | ⏸️ 0% |

**Legend**: ✅ All envs solved | 🔄 In progress | ❌ Failing | ⏸️ Not started | N/A Not applicable

## Benchmark Algorithms

**Discrete** (Classic Control, Box2D, Atari):
- DQN, DDQN+PER
- A2C (GAE), A2C (n-step)
- PPO

**Continuous** (MuJoCo):
- PPO
- SAC (future)

*Other algorithms (REINFORCE, SARSA, SIL, etc.) are included for completeness in Phase 1 but not benchmarked beyond Classic Control.*

---

## Running Experiments

### Local

```bash
uv run slm-lab SPEC_FILE SPEC_NAME train
uv run slm-lab SPEC_FILE SPEC_NAME search
```

### Remote (dstack)

```bash
# Always source .env for HF upload credentials
source .env && uv run slm-lab run-remote --gpu SPEC_FILE SPEC_NAME train -n NAME
source .env && uv run slm-lab run-remote --gpu SPEC_FILE SPEC_NAME search -n NAME

# With variable substitution for templated specs
source .env && uv run slm-lab run-remote --gpu -s env=Hopper-v5 SPEC_FILE SPEC_NAME search -n NAME
```

### Result Tables

Tables below use columns that map to command arguments:
- **Spec File**: `SPEC_FILE` argument
- **Spec Name**: `SPEC_NAME` argument
- Use `train` mode for validation runs, `search` mode for hyperparameter tuning

---

## Hyperparameter Search

**When to use**: Algorithm fails to reach target on first run.

### Search Budget Guidelines

**Search space vs trial count**: Balance is critical for efficient exploration.
- **3 parameters** (e.g., gamma, lam, lr): Use 16 trials minimum
- **4-5 parameters**: Use 20-30 trials
- **>5 parameters**: Reduce search space first - too many dims won't explore well

**Keep search space small and tractable** - don't search obvious params that won't matter. Focus only on salient hyperparameters.

### Stage 1: ASHA Search

ASHA (Asynchronous Successive Halving) with early termination for wide exploration.

**Config**: `max_session=1`, `max_trial=16` (minimum for 3-param search), `search_scheduler` enabled

```json
{
  "meta": {
    "max_session": 1,
    "max_trial": 16,
    "search_resources": {"cpu": 1, "gpu": 0.125},
    "search_scheduler": {"grace_period": 1e5, "reduction_factor": 3}
  },
  "search": {
    "agent.algorithm.gamma__uniform": [0.98, 0.999],
    "agent.algorithm.lam__uniform": [0.9, 0.98],
    "agent.net.optim_spec.lr__loguniform": [1e-4, 1e-3]
  }
}
```

**IMPORTANT**: Always use scientific notation for large numbers (e.g., `1e5`, `3e6`, `50e6`).

### Stage 2: Multi-Session Refinement (Optional)

Narrow search with robust statistics. Skip if Stage 1 results are good.

**Config**: `max_session=4`, `max_trial=8`, **NO** `search_scheduler`

### Stage 3: Finalize

1. Update spec defaults with winning hyperparams (keep `search` block for future tuning)
2. For templated specs (`${env}`): create dedicated spec file
3. Commit spec changes
4. Run validation with `train` mode
5. Pull & verify: `uv run slm-lab pull SPEC_NAME`
6. Update Active Runs and env table in this doc, push to public HF if good

**Note**: ASHA requires `max_session=1`. Multi-session requires no scheduler. They are mutually exclusive.

---

## Spec Standardization

**Keep similar specs standardized** for consistency and reproducibility.

### MuJoCo PPO Specs

All 11 MuJoCo PPO specs use uniform config:
- **Network**: `[256, 256]` + tanh + orthogonal init
- **Optimizer**: Adam 3e-4 (single optimizer with `use_same_optim=true`)
- **Search**: 3 params (gamma, lam, lr) × 16 trials
- **Normalization**: `normalize_obs=true`, `normalize_reward=true`, `normalize_v_targets=true`

### Atari Specs

All Atari specs use uniform config:
- **Network**: ConvNet with shared actor-critic
- **Frame stacking**: 4 frames
- **Preprocessing**: Standard Atari wrappers

### Environment-Specific Requirements

**Always follow these requirements when creating/updating specs:**

| Env Category | num_envs | max_frame | log_frequency | grace_period |
|--------------|----------|-----------|---------------|--------------|
| Classic Control | 4 | 2e5-3e5 | 500 | 1e4 |
| Box2D (easy) | 8 | 3e5 | 1000 | 5e4 |
| Box2D (hard) | 16 | 3e6 | 1e4 | 2e5 |
| MuJoCo (easy) | 16 | 1e6 | 1e4 | 1e5 |
| MuJoCo (medium) | 16 | 3e6 | 1e4 | 2e5 |
| MuJoCo (hard) | 32 | 50e6 | 5e4 | 1e6 |
| Atari | 16 | 10e6 | 1e4 | 5e5 |

---

## Workflow Checklist

**When starting new runs:**
1. ☐ Update "Active Runs" section at top of this doc
2. ☐ Update env table with run status (🔄)
3. ☐ Verify spec file follows standardization guidelines
4. ☐ Commit spec changes if any

**When runs complete:**
1. ☐ Pull results: `source .env && uv run slm-lab pull SPEC_NAME`
2. ☐ Analyze `experiment_df.csv` for best hyperparams
3. ☐ Update spec file with best hyperparams as defaults
4. ☐ Run validation with `train` mode to confirm
5. ☐ Update env table with results (MA, FPS, status)
6. ☐ Move run to "Completed Runs" in Active Runs section
7. ☐ Commit all changes together

---

## Phase 1: Classic Control

### 1.1 CartPole-v1

[Environment docs](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

| Property | Value |
|----------|-------|
| Action | Discrete(2) - push left/right |
| State | Box(4) - position, velocity, angle, angular velocity |
| Target | **MA > 400** |
| num_envs | 4 |
| max_frame | 2e5 |
| log_frequency | 500 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ✅ | 499.7 | 315 | [slm_lab/spec/benchmark/ppo/ppo_cartpole.json](../slm_lab/spec/benchmark/ppo/ppo_cartpole.json) | `ppo_cartpole` |
| A2C | ✅ | 488.7 | 3.5k | [slm_lab/spec/benchmark/a2c/a2c_gae_cartpole.json](../slm_lab/spec/benchmark/a2c/a2c_gae_cartpole.json) | `a2c_gae_cartpole` |
| DQN | ✅ | 437.8 | 1k | [slm_lab/spec/benchmark/dqn/dqn_cartpole.json](../slm_lab/spec/benchmark/dqn/dqn_cartpole.json) | `dqn_boltzmann_cartpole` |
| DDQN+PER | ✅ | 430.4 | 8k | [slm_lab/spec/benchmark/dqn/dqn_cartpole.json](../slm_lab/spec/benchmark/dqn/dqn_cartpole.json) | `ddqn_per_boltzmann_cartpole` |
| SAC | ✅ | 431.1 | <100 | [slm_lab/spec/benchmark/sac/sac_cartpole.json](../slm_lab/spec/benchmark/sac/sac_cartpole.json) | `sac_cartpole` |
| PPOSIL | ✅ | 496.3 | 1.6k | [slm_lab/spec/benchmark/sil/ppo_sil_cartpole.json](../slm_lab/spec/benchmark/sil/ppo_sil_cartpole.json) | `ppo_sil_cartpole` |
| REINFORCE | ✅ | 427.2 | 14k | [slm_lab/spec/benchmark/reinforce/reinforce_cartpole.json](../slm_lab/spec/benchmark/reinforce/reinforce_cartpole.json) | `reinforce_cartpole` |
| SARSA | ✅ | 393.2 | 7k | [slm_lab/spec/benchmark/sarsa/sarsa_cartpole.json](../slm_lab/spec/benchmark/sarsa/sarsa_cartpole.json) | `sarsa_epsilon_greedy_cartpole` |

### 1.2 Acrobot-v1

[Environment docs](https://gymnasium.farama.org/environments/classic_control/acrobot/)

| Property | Value |
|----------|-------|
| Action | Discrete(3) - torque (-1, 0, +1) |
| State | Box(6) - link positions and angular velocities |
| Target | **MA > -100** |
| num_envs | 4 |
| max_frame | 3e5 |
| log_frequency | 500 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ✅ | -80.8 | 777 | [slm_lab/spec/benchmark/ppo/ppo_acrobot.json](../slm_lab/spec/benchmark/ppo/ppo_acrobot.json) | `ppo_acrobot` |
| DQN (Boltzmann) | ✅ | -96.2 | 600 | [slm_lab/spec/benchmark/dqn/dqn_acrobot.json](../slm_lab/spec/benchmark/dqn/dqn_acrobot.json) | `dqn_boltzmann_acrobot` |
| DDQN+PER | ✅ | -83.0 | 700 | [slm_lab/spec/benchmark/dqn/ddqn_per_acrobot.json](../slm_lab/spec/benchmark/dqn/ddqn_per_acrobot.json) | `ddqn_per_acrobot` |
| A2C | ✅ | -84.2 | 3.4k | [slm_lab/spec/benchmark/a2c/a2c_gae_acrobot.json](../slm_lab/spec/benchmark/a2c/a2c_gae_acrobot.json) | `a2c_gae_acrobot` |
| SAC | ✅ | -92 | 60 | [slm_lab/spec/benchmark/sac/sac_acrobot.json](../slm_lab/spec/benchmark/sac/sac_acrobot.json) | `sac_acrobot` |
| DQN (ε-greedy) | ✅ | -79.5 | 720 | [slm_lab/spec/benchmark/dqn/dqn_acrobot.json](../slm_lab/spec/benchmark/dqn/dqn_acrobot.json) | `dqn_epsilon_greedy_acrobot` |
| PPOSIL | ✅ | -83.1 | - | [slm_lab/spec/benchmark/sil/ppo_sil_acrobot.json](../slm_lab/spec/benchmark/sil/ppo_sil_acrobot.json) | `ppo_sil_acrobot` |

### 1.3 Pendulum-v1 (Continuous)

[Environment docs](https://gymnasium.farama.org/environments/classic_control/pendulum/)

| Property | Value |
|----------|-------|
| Action | Box(1) - torque [-2, 2] |
| State | Box(3) - cos(θ), sin(θ), angular velocity |
| Target | **MA > -200** |
| num_envs | 4 |
| max_frame | 3e5 |
| log_frequency | 500 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | 🔄 | -1214 | - | [slm_lab/spec/benchmark/ppo/ppo_pendulum.json](../slm_lab/spec/benchmark/ppo/ppo_pendulum.json) | `ppo_pendulum` |
| SAC | 🔄 | - | - | [slm_lab/spec/benchmark/sac/sac_pendulum.json](../slm_lab/spec/benchmark/sac/sac_pendulum.json) | `sac_pendulum` |

**Note**: Classic continuous control benchmark. Action bounds [-2, 2] require automatic RescaleAction wrapper (implemented in commit 56a6e69c). This is a simpler continuous environment for validating continuous action implementations before MuJoCo.

---

## Phase 2: Box2D

### 2.1 LunarLander-v3 (Discrete)

[Environment docs](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

| Property | Value |
|----------|-------|
| Action | Discrete(4) - no-op, fire left/main/right engine |
| State | Box(8) - position, velocity, angle, angular velocity, leg contact |
| Target | **MA > 200** |
| num_envs | 8 |
| max_frame | 3e5 |
| log_frequency | 1000 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| DDQN+PER | ✅ | 230.0 | 8.7k | [slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json](../slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json) | `ddqn_per_concat_lunar` |
| PPO | ✅ | 229.9 | 2.4k | [slm_lab/spec/benchmark/ppo/ppo_lunar.json](../slm_lab/spec/benchmark/ppo/ppo_lunar.json) | `ppo_lunar` |
| DQN | ✅ | 203.9 | 9.0k | [slm_lab/spec/benchmark/dqn/dqn_lunar.json](../slm_lab/spec/benchmark/dqn/dqn_lunar.json) | `dqn_concat_lunar` |
| A2C | ✅ | 304 | 3k | [slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json](../slm_lab/spec/benchmark/a2c/a2c_gae_lunar.json) | `a2c_gae_lunar` |
| SAC | 🔄 | - | - | [slm_lab/spec/benchmark/sac/sac_lunar.json](../slm_lab/spec/benchmark/sac/sac_lunar.json) | `sac_lunar` |

### 2.2 LunarLander-v3 (Continuous)

[Environment docs](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

| Property | Value |
|----------|-------|
| Action | Box(2) - main engine [-1,1], side engines [-1,1] |
| State | Box(8) - position, velocity, angle, angular velocity, leg contact |
| Target | **MA > 200** |
| num_envs | 8 |
| max_frame | 3e5 |
| log_frequency | 1000 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ✅ | 249.2 | 135 | [slm_lab/spec/benchmark/ppo/ppo_lunar.json](../slm_lab/spec/benchmark/ppo/ppo_lunar.json) | `ppo_lunar_continuous` |
| SAC | ✅ | 238.0 | 35 | [slm_lab/spec/benchmark/sac/sac_lunar.json](../slm_lab/spec/benchmark/sac/sac_lunar.json) | `sac_lunar_continuous` |

---

## Phase 3: MuJoCo

All MuJoCo environments use **continuous** action spaces.

**Template specs**: Use `-s env=EnvName-v5` for variable substitution. Successful envs get dedicated specs.

### 3.1 Hopper-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/hopper/)

| Property | Value |
|----------|-------|
| Action | Box(3) |
| State | Box(11) |
| Target | **MA > 3000** |
| num_envs | 16 |
| max_frame | 3e6 |
| log_frequency | 1e4 |
| grace_period | 2e5 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⚠️ 78% | 2517 | 1.5k | [slm_lab/spec/benchmark/ppo/ppo_hopper.json](../slm_lab/spec/benchmark/ppo/ppo_hopper.json) | `ppo_hopper` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=Hopper-v5` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=Hopper-v5` |

### 3.2 HalfCheetah-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/half_cheetah/)

| Property | Value |
|----------|-------|
| Action | Box(6) |
| State | Box(17) |
| Target | **MA > 5000** |
| num_envs | 16 |
| max_frame | 3e6 |
| log_frequency | 1e4 |
| grace_period | 2e5 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | 🔄 51% | 2615 | 1.5k | [slm_lab/spec/benchmark/ppo/ppo_halfcheetah.json](../slm_lab/spec/benchmark/ppo/ppo_halfcheetah.json) | `ppo_halfcheetah` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=HalfCheetah-v5` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=HalfCheetah-v5` |

### 3.3 Walker2d-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/walker2d/)

| Property | Value |
|----------|-------|
| Action | Box(6) |
| State | Box(17) |
| Target | **MA > 5000** |
| num_envs | 16 |
| max_frame | 3e6 |
| log_frequency | 1e4 |
| grace_period | 2e5 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⚠️ 45% | 1997 | 1.5k | [slm_lab/spec/benchmark/ppo/ppo_walker2d.json](../slm_lab/spec/benchmark/ppo/ppo_walker2d.json) | `ppo_walker2d` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=Walker2d-v5` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=Walker2d-v5` |

### 3.4 Ant-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/ant/)

| Property | Value |
|----------|-------|
| Action | Box(8) |
| State | Box(111) |
| Target | **MA > 5000** |
| num_envs | 16 |
| max_frame | 3e6 |
| log_frequency | 1e4 |
| grace_period | 2e5 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ❌ | -105 | 1.5k | [slm_lab/spec/benchmark/ppo/ppo_ant.json](../slm_lab/spec/benchmark/ppo/ppo_ant.json) | `ppo_ant` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=Ant-v5` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=Ant-v5` |

### 3.5 Swimmer-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/swimmer/)

| Property | Value |
|----------|-------|
| Action | Box(2) |
| State | Box(8) |
| Target | **MA > 130** |
| num_envs | 16 |
| max_frame | 1e6 |
| log_frequency | 1e4 |
| grace_period | 1e5 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | 🔄 | 43.3 | 1.5k | [slm_lab/spec/benchmark/ppo/ppo_swimmer.json](../slm_lab/spec/benchmark/ppo/ppo_swimmer.json) | `ppo_swimmer` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=Swimmer-v5` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=Swimmer-v5` |

### 3.6 Reacher-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/reacher/)

| Property | Value |
|----------|-------|
| Action | Box(2) |
| State | Box(11) |
| Target | **MA > -5** |
| num_envs | 16 |
| max_frame | 1e6 |
| log_frequency | 1e4 |
| grace_period | 1e5 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ✅ | -0.003 | 1.5k | [slm_lab/spec/benchmark/ppo/ppo_reacher.json](../slm_lab/spec/benchmark/ppo/ppo_reacher.json) | `ppo_reacher` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=Reacher-v5` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=Reacher-v5` |

### 3.7 Pusher-v5

[Environment docs](https://gymnasium.farama.org/environments/mujoco/pusher/)

| Property | Value |
|----------|-------|
| Action | Box(7) |
| State | Box(23) |
| Target | **MA > -20** |
| num_envs | 16 |
| max_frame | 1e6 |
| log_frequency | 1e4 |
| grace_period | 1e5 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ✅ | -0.75 | - | [slm_lab/spec/benchmark/ppo/ppo_pusher.json](../slm_lab/spec/benchmark/ppo/ppo_pusher.json) | `ppo_pusher` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=Pusher-v5` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=Pusher-v5` |

### 3.8 InvertedPendulum-v5 (PARKED)

[Environment docs](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/)

| Property | Value |
|----------|-------|
| Action | Box(1) |
| State | Box(4) |
| Target | **MA > 1000** |
| num_envs | 16 |
| max_frame | 1e6 |
| log_frequency | 1e4 |
| grace_period | 1e5 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⏸️ | 19.2 | 3k | [slm_lab/spec/benchmark/ppo/ppo_inverted_pendulum.json](../slm_lab/spec/benchmark/ppo/ppo_inverted_pendulum.json) | `ppo_inverted_pendulum` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=InvertedPendulum-v5` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=InvertedPendulum-v5` |

**PARKED**: Consistently failing despite many attempted fixes (log_std, entropy, architecture). Needs deeper investigation.

### 3.9 InvertedDoublePendulum-v5 (PARKED)

[Environment docs](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/)

| Property | Value |
|----------|-------|
| Action | Box(1) |
| State | Box(11) |
| Target | **MA > 9100** |
| num_envs | 16 |
| max_frame | 1e6 |
| log_frequency | 1e4 |
| grace_period | 1e5 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⏸️ | 73.6 | - | [slm_lab/spec/benchmark/ppo/ppo_inverted_double_pendulum.json](../slm_lab/spec/benchmark/ppo/ppo_inverted_double_pendulum.json) | `ppo_inverted_double_pendulum` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=InvertedDoublePendulum-v5` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=InvertedDoublePendulum-v5` |

**PARKED**: Consistently failing despite many attempted fixes (log_std, entropy, architecture). Needs deeper investigation.

### 3.10 Humanoid-v5 (PARKED)

[Environment docs](https://gymnasium.farama.org/environments/mujoco/humanoid/)

| Property | Value |
|----------|-------|
| Action | Box(17) |
| State | Box(376) |
| Target | **MA > 6000** |
| num_envs | 32 |
| max_frame | 50e6 |
| log_frequency | 5e4 |
| grace_period | 1e6 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⏸️ | - | - | [slm_lab/spec/benchmark/ppo/ppo_humanoid.json](../slm_lab/spec/benchmark/ppo/ppo_humanoid.json) | `ppo_humanoid` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_humanoid.json](../slm_lab/spec/benchmark/sac/sac_humanoid.json) | `sac_humanoid` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=Humanoid-v5` |

**PARKED**: Requires 50M frames - will hit 4h dstack limit (~20-25M frames typically). Deprioritized until medium envs solved.

### 3.11 HumanoidStandup-v5 (PARKED)

[Environment docs](https://gymnasium.farama.org/environments/mujoco/humanoid_standup/)

| Property | Value |
|----------|-------|
| Action | Box(17) |
| State | Box(376) |
| Target | **MA > 100000** |
| num_envs | 32 |
| max_frame | 50e6 |
| log_frequency | 5e4 |
| grace_period | 1e6 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⏸️ | - | - | [slm_lab/spec/benchmark/ppo/ppo_humanoid_standup.json](../slm_lab/spec/benchmark/ppo/ppo_humanoid_standup.json) | `ppo_humanoid_standup` |
| SAC | ⏸️ | - | - | [slm_lab/spec/benchmark/sac/sac_mujoco.json](../slm_lab/spec/benchmark/sac/sac_mujoco.json) | `sac_mujoco` `-s env=HumanoidStandup-v5` |
| A2C | ⏸️ | - | - | [slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json](../slm_lab/spec/benchmark/a2c/a2c_gae_mujoco.json) | `a2c_gae_mujoco` `-s env=HumanoidStandup-v5` |

**PARKED**: Very high reward scale environment. Requires 50M frames. Deprioritized until medium envs solved.

---

## Phase 4: Atari

All Atari environments use **Discrete** action spaces and **Box(210,160,3)** RGB image observations.

### 4.1 Pong-v5

[Environment docs](https://gymnasium.farama.org/environments/atari/pong/)

| Property | Value |
|----------|-------|
| Action | Discrete(6) |
| State | Box(210,160,3) RGB |
| Target | **MA > 18** |
| num_envs | 16 |
| max_frame | 10e6 |
| log_frequency | 1e4 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⏸️ | - | - | [slm_lab/spec/benchmark/ppo/ppo_pong.json](../slm_lab/spec/benchmark/ppo/ppo_pong.json) | `ppo_pong` |
| DQN | ⏸️ | - | - | [slm_lab/spec/benchmark/dqn/dqn_pong.json](../slm_lab/spec/benchmark/dqn/dqn_pong.json) | `dqn_pong` |

### 4.2 Qbert-v5

[Environment docs](https://gymnasium.farama.org/environments/atari/qbert/)

| Property | Value |
|----------|-------|
| Action | Discrete(6) |
| State | Box(210,160,3) RGB |
| Target | **MA > 15000** |
| num_envs | 16 |
| max_frame | 10e6 |
| log_frequency | 1e4 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⏸️ | - | - | [slm_lab/spec/benchmark/ppo/ppo_qbert.json](../slm_lab/spec/benchmark/ppo/ppo_qbert.json) | `ppo_qbert` |
| DQN | ⏸️ | - | - | [slm_lab/spec/benchmark/dqn/dqn_qbert.json](../slm_lab/spec/benchmark/dqn/dqn_qbert.json) | `dqn_qbert` |

### 4.3 Breakout-v5

[Environment docs](https://gymnasium.farama.org/environments/atari/breakout/)

| Property | Value |
|----------|-------|
| Action | Discrete(4) |
| State | Box(210,160,3) RGB |
| Target | **MA > 400** |
| num_envs | 16 |
| max_frame | 10e6 |
| log_frequency | 1e4 |

| Algorithm | Status | MA | FPS | Spec File | Spec Name |
|-----------|--------|-----|-----|-----------|-----------|
| PPO | ⏸️ | - | - | [slm_lab/spec/benchmark/ppo/ppo_atari.json](../slm_lab/spec/benchmark/ppo/ppo_atari.json) | `ppo_atari` `-s env=ALE/Breakout-v5` |
| DQN | ⏸️ | - | - | [slm_lab/spec/benchmark/dqn/dqn_atari.json](../slm_lab/spec/benchmark/dqn/dqn_atari.json) | `dqn_atari` `-s env=ALE/Breakout-v5` |

---

## Known Issues

**DQN Compute Inefficiency** ✅ RESOLVED
- Was 84x slower than A2C due to excessive gradient updates (10 updates/step vs standard 1)
- Fixed by adjusting `training_batch_iter` and `training_iter`
- Result: 3.7-15x speedup with equivalent learning

**SIL (Self-Imitation Learning)** ✅ RESOLVED
- Fixed venv-packed data handling in replay memory
- PPOSIL now achieves 124% of target on CartPole

---

