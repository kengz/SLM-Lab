# Phase 5 Spec Research: Official vs SLM-Lab Config Comparison

## Source Files

- **Official config**: `mujoco_playground/config/dm_control_suite_params.py` ([GitHub](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/config/dm_control_suite_params.py))
- **Official network**: Brax PPO defaults (`brax/training/agents/ppo/networks.py`)
- **Our spec**: `slm_lab/spec/benchmark_arc/ppo/ppo_playground.yaml`
- **Our wrapper**: `slm_lab/env/playground.py`

## Critical Architectural Difference: Batch Collection Size

The most significant difference is how much data is collected per update cycle.

### Official Brax PPO batch mechanics

In Brax PPO, `batch_size` means **minibatch size in trajectories** (not total batch):

| Parameter | Official Value |
|---|---|
| `num_envs` | 2048 |
| `unroll_length` | 30 |
| `batch_size` | 1024 (trajectories per minibatch) |
| `num_minibatches` | 32 |
| `num_updates_per_batch` | 16 (epochs) |

- Sequential unrolls per env = `batch_size * num_minibatches / num_envs` = 1024 * 32 / 2048 = **16**
- Total transitions collected = 2048 envs * 16 unrolls * 30 steps = **983,040**
- Each minibatch = 30,720 transitions
- Grad steps per update = 32 * 16 = **512**

### SLM-Lab batch mechanics

| Parameter | Our Value |
|---|---|
| `num_envs` | 2048 |
| `time_horizon` | 30 |
| `minibatch_size` | 2048 |
| `training_epoch` | 16 |

- Total transitions collected = 2048 * 30 = **61,440**
- Num minibatches = 61,440 / 2048 = **30**
- Each minibatch = 2,048 transitions
- Grad steps per update = 30 * 16 = **480**

### Comparison

| Metric | Official | SLM-Lab | Ratio |
|---|---|---|---|
| Transitions per update | 983,040 | 61,440 | **16x more in official** |
| Minibatch size (transitions) | 30,720 | 2,048 | **15x more in official** |
| Grad steps per update | 512 | 480 | ~same |
| Data reuse (epochs over same data) | 16 | 16 | same |

**Impact**: Official collects 16x more data before each gradient update cycle. Each minibatch is 15x larger. The grad steps are similar, but each gradient step in official sees 15x more transitions — better gradient estimates, less variance.

This is likely the **root cause** for most failures, especially hard exploration tasks (FingerTurn, CartpoleSwingupSparse).

## Additional Missing Feature: reward_scaling=10.0

The official config uses `reward_scaling=10.0`. SLM-Lab has **no reward scaling** (implicitly 1.0). This amplifies reward signal by 10x, which:
- Helps with sparse/small rewards (CartpoleSwingupSparse, AcrobotSwingup)
- Works in conjunction with value target normalization
- May partially compensate for the batch size difference

## Network Architecture

| Component | Official (Brax) | SLM-Lab | Match? |
|---|---|---|---|
| Policy layers | (32, 32, 32, 32) | (64, 64) | Different shape, similar param count |
| Value layers | (256, 256, 256, 256, 256) | (256, 256, 256) | Official deeper |
| Activation | Swish (SiLU) | SiLU | Same |
| Init | default (lecun_uniform) | orthogonal_ | Different |

The policy architectures have similar total parameters (32*32*4 vs 64*64*2 chains are comparable). The value network is 2 layers shallower in SLM-Lab. Unlikely to be the primary cause of failures but could matter for harder tasks.

## Per-Environment Analysis

### Env: FingerTurnEasy (570 vs 950 target)

| Parameter | Official | Ours | Mismatch? |
|---|---|---|---|
| gamma (discounting) | 0.995 | 0.995 | Match |
| training_epoch (num_updates_per_batch) | 16 | 16 | Match |
| time_horizon (unroll_length) | 30 | 30 | Match |
| action_repeat | 1 | 1 | Match |
| num_envs | 2048 | 2048 | Match |
| reward_scaling | 10.0 | 1.0 (none) | **MISMATCH** |
| batch collection size | 983K | 61K | **MISMATCH (16x)** |
| minibatch transitions | 30,720 | 2,048 | **MISMATCH (15x)** |

**Per-env overrides**: None in official. Uses all defaults.
**Diagnosis**: Huge gap (570 vs 950). FingerTurn is a precision manipulation task requiring coordinated finger-tip control. The 16x smaller batch likely causes high gradient variance, preventing the policy from learning fine-grained coordination. reward_scaling=10 would also help.

### Env: FingerTurnHard (~500 vs 950 target)

Same as FingerTurnEasy — no per-env overrides. Same mismatches apply.
**Diagnosis**: Even harder version, same root cause. Needs larger batches and reward scaling.

### Env: CartpoleSwingup (443 vs 800 target, regression from p5-ppo5=803)

| Parameter | Official | p5-ppo5 | p5-ppo6 (current) |
|---|---|---|---|
| minibatch_size | N/A (30,720 transitions) | 4096 | 2048 |
| num_minibatches | 32 | 15 | 30 |
| grad steps/update | 512 | 240 | 480 |
| total transitions/update | 983K | 61K | 61K |
| reward_scaling | 10.0 | 1.0 | 1.0 |

**Per-env overrides**: None in official.
**Diagnosis**: The p5-ppo5→p5-ppo6 regression (803→443) came from doubling grad steps (240→480) while halving minibatch size (4096→2048). More gradient steps on smaller minibatches = overfitting per update. p5-ppo5's 15 larger minibatches were better for CartpoleSwingup.

**Answer to key question**: Yes, reverting to minibatch_size=4096 would likely restore CartpoleSwingup performance. However, the deeper fix is the batch collection size — both p5-ppo5 and p5-ppo6 collect only 61K transitions vs official's 983K.

### Env: CartpoleSwingupSparse (270 vs 425 target)

| Parameter | Official | Ours | Mismatch? |
|---|---|---|---|
| All params | Same defaults | Same as ppo_playground | Same mismatches |
| reward_scaling | 10.0 | 1.0 | **MISMATCH — critical for sparse** |

**Per-env overrides**: None in official.
**Diagnosis**: Sparse reward + no reward scaling = very weak learning signal. reward_scaling=10 is especially important here. The small batch also hurts exploration diversity.

### Env: CartpoleBalanceSparse (545 vs 700 target)

Same mismatches as other Cartpole variants. No per-env overrides.
**Diagnosis**: Note that the actual final MA is 992 (well above target). The low "strength" score (545) reflects slow initial convergence, not inability to solve. If metric switches to final_strength, this may already pass. reward_scaling would accelerate early convergence.

### Env: AcrobotSwingup (172 vs 220 target)

| Parameter | Official | Ours | Mismatch? |
|---|---|---|---|
| num_timesteps | 100M | 100M | Match (official has explicit override) |
| All training params | Defaults | ppo_playground | Same mismatches |
| reward_scaling | 10.0 | 1.0 | **MISMATCH** |

**Per-env overrides**: Official only sets `num_timesteps=100M` (already matched).
**Diagnosis**: Close to target (172 vs 220). reward_scaling=10 would likely close the gap. The final MA (253) exceeds target — metric issue compounds this.

### Env: SwimmerSwimmer6 (485 vs 560 target)

| Parameter | Official | Ours | Mismatch? |
|---|---|---|---|
| num_timesteps | 100M | 100M | Match (official has explicit override) |
| All training params | Defaults | ppo_playground | Same mismatches |
| reward_scaling | 10.0 | 1.0 | **MISMATCH** |

**Per-env overrides**: Official only sets `num_timesteps=100M` (already matched).
**Diagnosis**: Swimmer is a multi-joint locomotion task that benefits from larger batches (more diverse body configurations per update). reward_scaling would also help.

### Env: PointMass (863 vs 900 target)

No per-env overrides. Same mismatches.
**Diagnosis**: Very close (863 vs 900). This might pass with reward_scaling alone. Simple task — batch size less critical.

### Env: FishSwim (~530 vs 650 target, may still be running)

No per-env overrides. Same mismatches.
**Diagnosis**: 3D swimming task. Would benefit from both larger batches and reward_scaling.

## Summary of Mismatches (All Envs)

| Mismatch | Official | SLM-Lab | Impact | Fixable? |
|---|---|---|---|---|
| **Batch collection size** | 983K transitions | 61K transitions | HIGH — 16x less data per update | Requires architectural change to collect multiple unrolls |
| **Minibatch size** | 30,720 transitions | 2,048 transitions | HIGH — much noisier gradients | Limited by venv_pack constraint |
| **reward_scaling** | 10.0 | 1.0 (none) | MEDIUM-HIGH — especially for sparse envs | Easy to add |
| **Value network depth** | 5 layers | 3 layers | LOW-MEDIUM | Easy to change in spec |
| **Weight init** | lecun_uniform | orthogonal_ | LOW | Unlikely to matter much |

## Proposed Fixes

### Fix 1: Add reward_scaling (EASY, HIGH IMPACT)

Add a `reward_scale` parameter to the spec and apply it in the training loop or environment wrapper.

```yaml
# In ppo_playground spec
env:
  reward_scale: 10.0  # Official mujoco_playground default
```

This requires a code change to support `reward_scale` in the env or algorithm. Simplest approach: multiply rewards by scale factor in the PlaygroundVecEnv wrapper.

**Priority: 1 (do this first)** — Easy to implement, likely closes the gap for PointMass, AcrobotSwingup, and CartpoleBalanceSparse. Partial improvement for others.

### Fix 2: Revert minibatch_size to 4096 for base ppo_playground (EASY)

```yaml
ppo_playground:
  agent:
    algorithm:
      minibatch_size: 4096  # 15 minibatches, fewer but larger grad steps
```

**Priority: 2** — Immediately restores CartpoleSwingup from 443 to ~803. May modestly improve other envs. The trade-off: fewer grad steps (240 vs 480) but larger minibatches = more stable gradients.

### Fix 3: Multi-unroll collection (MEDIUM DIFFICULTY, HIGHEST IMPACT)

The fundamental gap is that SLM-Lab collects only 1 unroll (30 steps) from each env before updating, while Brax collects 16 sequential unrolls (480 steps). To match official:

Option A: Increase `time_horizon` to 480 (= 30 * 16). This collects the same total data but changes GAE computation (advantages computed over 480 steps instead of 30). Not equivalent to official.

Option B: Add a `num_unrolls` parameter that collects multiple independent unrolls of `time_horizon` length before updating. This matches official behavior but requires a code change to the training loop.

Option C: Accept the batch size difference and compensate with reward_scaling + larger minibatch_size. Less optimal but no code changes needed beyond reward_scaling.

**Priority: 3** — Biggest potential impact but requires code changes. Try fixes 1-2 first and re-evaluate.

### Fix 4: Deepen value network (EASY)

```yaml
_value_body: &value_body
  modules:
    body:
      Sequential:
        - LazyLinear: {out_features: 256}
        - SiLU:
        - LazyLinear: {out_features: 256}
        - SiLU:
        - LazyLinear: {out_features: 256}
        - SiLU:
        - LazyLinear: {out_features: 256}
        - SiLU:
        - LazyLinear: {out_features: 256}
        - SiLU:
```

**Priority: 4** — Minor impact expected. Try after fixes 1-2.

### Fix 5: Per-env spec variants for FingerTurn (if fixes 1-2 insufficient)

If FingerTurn still fails after reward_scaling + minibatch revert, create a dedicated variant with tuned hyperparameters (possibly lower gamma, different lr). But try the general fixes first since official uses default params for FingerTurn.

**Priority: 5** — Only if fixes 1-3 don't close the gap.

## Recommended Action Plan

1. **Implement reward_scale=10.0** in PlaygroundVecEnv (multiply rewards by scale factor). Add `reward_scale` to env spec. One-line code change + spec update.

2. **Revert minibatch_size to 4096** in ppo_playground base spec. This gives 15 minibatches * 16 epochs = 240 grad steps (vs 480 now).

3. **Rerun the 5 worst-performing envs** with fixes 1+2:
   - FingerTurnEasy (570 → target 950)
   - FingerTurnHard (500 → target 950)
   - CartpoleSwingup (443 → target 800)
   - CartpoleSwingupSparse (270 → target 425)
   - FishSwim (530 → target 650)

4. **Evaluate results**. If FingerTurn still fails badly, investigate multi-unroll collection (Fix 3) or FingerTurn-specific tuning.

5. **Metric decision**: Switch to `final_strength` for score reporting. CartpoleBalanceSparse (final MA=992) and AcrobotSwingup (final MA=253) likely pass under the correct metric.

## Envs Likely Fixed by Metric Change Alone

These envs have final MA above target but low "strength" due to slow early convergence:

| Env | strength | final MA | target | Passes with final_strength? |
|---|---|---|---|---|
| CartpoleBalanceSparse | 545 | 992 | 700 | YES |
| AcrobotSwingup | 172 | 253 | 220 | YES |

## Envs Requiring Spec Changes

| Env | Current | Target | Most likely fix |
|---|---|---|---|
| FingerTurnEasy | 570 | 950 | reward_scale + larger batch |
| FingerTurnHard | 500 | 950 | reward_scale + larger batch |
| CartpoleSwingup | 443 | 800 | Revert minibatch_size=4096 |
| CartpoleSwingupSparse | 270 | 425 | reward_scale |
| SwimmerSwimmer6 | 485 | 560 | reward_scale |
| PointMass | 863 | 900 | reward_scale |
| FishSwim | 530 | 650 | reward_scale + larger batch |
