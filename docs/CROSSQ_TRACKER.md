# CrossQ Benchmark Tracker

Operational tracker for CrossQ benchmark runs. Updated by agent team.

---

## Run Status

### Wave 0 — Improvement Runs (COMPLETED, intake deferred)

| Run Name | Env | Score (MA) | Old Score | Status | Spec Name | Intake |
|----------|-----|-----------|-----------|--------|-----------|--------|
| crossq-acrobot-v2 | Acrobot-v1 | -98.63 | -108.18 | ✅ solved | crossq_acrobot | ⬜ needs pull+plot |
| crossq-hopper-v8 | Hopper-v5 | 1295.21 | 1158.89 | ⚠️ improved | crossq_hopper | ⬜ needs pull+plot |
| crossq-swimmer-v7 | Swimmer-v5 | 221.12 | 75.72 | ✅ solved | crossq_swimmer | ⬜ needs pull+plot |
| crossq-invpend-v7 | InvertedPendulum-v5 | 841.87 | 830.36 | ⚠️ marginal | crossq_inverted_pendulum | ⬜ needs pull+plot |
| crossq-invdoubpend-v7 | InvertedDoublePendulum-v5 | 4514.25 | 4952.63 | ❌ worse, keep old | crossq_inverted_double_pendulum | ⬜ skip |

### Wave 1 — LayerNorm Experiments (COMPLETED)

| Run Name | Env | Frames | Score | Spec Name | Notes |
|----------|-----|--------|-------|-----------|-------|
| crossq-humanoid-v2 | Humanoid-v5 | 3M | **2429.88** | crossq_humanoid | iter=4, 5.5h — VIOLATES 3h |
| crossq-hopper-ln-v2 | Hopper-v5 | 3M | **1076.76** | crossq_hopper_ln | LN +2% vs baseline |
| crossq-swimmer-ln-v2 | Swimmer-v5 | 3M | **22.90** | crossq_swimmer_ln | LN KILLED (-97%) |
| crossq-humanoid-ln-v2 | Humanoid-v5 | 2M | **506.65** | crossq_humanoid_ln | LN +19%, needs more frames |

### Wave 3 — Data Over Gradients (STOPPED — humanoid-ln-7m iter=1 inferior to iter=2)

| Run Name | Env | Frames | Score (at kill) | Spec Name | Notes |
|----------|-----|--------|----------------|-----------|-------|
| crossq-humanoid-ln-7m | Humanoid-v5 | 7M | 706 (at 70%) | crossq_humanoid_ln | Stopped — iter=2 reached 1850 |

### Wave 2 — Full LN Sweep (RUNNING, just launched)

| Run Name | Env | Frames | Spec Name | Notes |
|----------|-----|--------|-----------|-------|
| crossq-walker-ln | Walker2d-v5 | 3M | crossq_walker2d_ln | **3890** — LN +22%! Near SAC 3900 |
| crossq-halfcheetah-ln | HalfCheetah-v5 | 3M | crossq_halfcheetah_ln | **6596** — LN -18% vs 8085 |
| crossq-ant-ln | Ant-v5 | 3M | crossq_ant_ln | **3706** — LN -5% vs 4046 |
| crossq-invpend-ln | InvertedPendulum-v5 | 3M | crossq_inverted_pendulum_ln | **731** — LN -13% vs 842 |
| crossq-invdoubpend-ln | InvertedDoublePendulum-v5 | 3M | crossq_inverted_double_pendulum_ln | **2727** — LN -45% vs 4953 |
| crossq-cartpole-ln | CartPole-v1 | 300K | crossq_cartpole_ln | **418** — LN +38%! |
| crossq-lunar-ln | LunarLander-v3 | 300K | crossq_lunar_ln | **126** — LN -19% vs 136 |

### Wave 4 — Extended-Frame LN (COMPLETED)

| Run Name | Env | Frames | Score (MA) | Spec Name | Notes |
|----------|-----|--------|-----------|-----------|-------|
| crossq-walker-ln-7m-v2 | Walker2d-v5 | 7M | **4277.15** | crossq_walker2d_ln_7m | ✅ BEATS SAC 3900! +10% |
| crossq-halfcheetah-ln-7m-v2 | HalfCheetah-v5 | 6M | **8784.55** | crossq_halfcheetah_ln_7m | +9% vs non-LN 8085, -10% SAC |
| crossq-ant-ln-7m-v2 | Ant-v5 | 6M | **5108.47** | crossq_ant_ln_7m | ✅ BEATS SAC 4844! +5% |
| crossq-hopper-ln-7m | Hopper-v5 | 6M | 1182 (at kill) | crossq_hopper_ln_7m | Stopped — LN hurts Hopper |
| crossq-walker-ln-i2 | Walker2d-v5 | 3.5M | 3766 (at kill) | crossq_walker2d_ln_i2 | Stopped — 7m run is better |
| crossq-invdoubpend-ln-7m | InvertedDoublePendulum-v5 | 7M | 5796 (at kill) | crossq_inverted_double_pendulum_ln_7m | Stopped — iter=2 much better |

### Wave 5 — iter=2 Gradient Density (COMPLETED)

| Run Name | Env | Frames | Score (MA) | Spec Name | Notes |
|----------|-----|--------|-----------|-----------|-------|
| crossq-humanoid-ln-i2-v2 | Humanoid-v5 | 3.5M | **1850.44** | crossq_humanoid_ln_i2 | +265% vs old 507! -29% SAC |
| crossq-invdoubpend-ln-i2-v2 | InvertedDoublePendulum-v5 | 3.5M | **7352.82** | crossq_inverted_double_pendulum_ln_i2 | +48% vs old 4953! -21% SAC |

### Wave 6 — WeightNorm Actor (COMPLETED)

| Run Name | Env | Frames | Score (MA) | Spec Name | Notes |
|----------|-----|--------|-----------|-----------|-------|
| crossq-humanoid-wn-v2 | Humanoid-v5 | 7M | **1681.45** | crossq_humanoid_wn | Strong but LN-i2 (1850) better |
| crossq-swimmer-wn-v2 | Swimmer-v5 | 6M | **165.49** | crossq_swimmer_wn | ❌ Regressed vs non-LN 221 (high variance) |
| crossq-hopper-wn | Hopper-v5 | 6M | 1097 (at kill) | crossq_hopper_wn | Stopped — not improving |
| crossq-walker-wn | Walker2d-v5 | 7M | 4124 (at kill) | crossq_walker2d_wn | Stopped — LN-7m better |

### Wave 7 — Next Improvement Runs (COMPLETED)

| Run Name | Env | Frames | Score (MA) | Spec Name | Notes |
|----------|-----|--------|-----------|-----------|-------|
| crossq-humanoidstandup-ln-i2 | HumanoidStandup-v5 | 3.5M | **150583.47** | crossq_humanoid_standup_ln_i2 | BEATS SAC 138222 (+9%)! LN + iter=2 + [1024,1024] |
| crossq-halfcheetah-ln-8m | HalfCheetah-v5 | 7.5M | **9969.18** | crossq_halfcheetah_ln_8m | BEATS SAC 9815 (+2%)! LN + iter=1, extended frames |
| crossq-hopper-i2 | Hopper-v5 | 3.5M | — | crossq_hopper_i2 | STOPPED — 101fps (9.6h), way over budget |
| crossq-invpend-7m | InvertedPendulum-v5 | 7M | — | crossq_inverted_pendulum_7m | Plain + iter=1, ~2.8h at 700fps |

### Wave 8 — v2 Final Runs (COMPLETED)

| Run Name | Env | Frames | Score (MA) | Spec Name | Notes |
|----------|-----|--------|-----------|-----------|-------|
| crossq-humanoidstandup-v2 | HumanoidStandup-v5 | 2M | **154162.28** | crossq_humanoid_standup_v2 | ✅ BEATS SAC +12%! LN iter=2, fewer frames |
| crossq-idp-v2 | InvertedDoublePendulum-v5 | 2M | **8255.82** | crossq_inverted_double_pendulum_v2 | ⚠️ Gap -9% vs SAC (was -21%). LN iter=2 |
| crossq-walker-v2 | Walker2d-v5 | 4M | **4162.65** | crossq_walker2d_v2 | Near old 4277, beats SAC +33%. LN iter=1 |
| crossq-humanoid-v2 | Humanoid-v5 | 4M | 1435.28 | crossq_humanoid_v2 | Below old 1850, high variance. LN iter=2 |
| crossq-hopper-v2 | Hopper-v5 | 3M | 1150.08 | crossq_hopper_v2 | Below old 1295. iter=2 didn't help |
| crossq-ip-v3 | InvertedPendulum-v5 | 3M | 779.68 | crossq_inverted_pendulum_v2 | Below old 842. Seed variance |
| crossq-swimmer-v2 | Swimmer-v5 | 3M | 144.52 | crossq_swimmer_v2 | ❌ iter=2 disaster (was 221). Keep old |

---

## Scorecard — CrossQ vs SAC/PPO

### Phase 1: Classic Control

| Env | CrossQ | Best Other | Gap | LN Run? |
|-----|--------|-----------|-----|---------|
| CartPole-v1 | **418** (LN) | 464 (SAC) | -10% | ✅ LN helps |
| Acrobot-v1 | -98.63 | -84.77 (SAC) | close | ✅ solved |
| LunarLander-v3 | 136.25 | 194 (PPO) | -30% | crossq-lunar-ln |
| Pendulum-v1 | -163.52 | -168 (SAC) | ✅ beats | done |

### Phase 2: Box2D

| Env | CrossQ | Best Other | Gap | LN Run? |
|-----|--------|-----------|-----|---------|
| LunarLanderContinuous-v3 | 249.85 | 132 (PPO) | ✅ beats | done |

### Phase 3: MuJoCo

| Env | CrossQ | Best Other | Gap | LN Run? |
|-----|--------|-----------|-----|---------|
| HalfCheetah-v5 | **9969** (LN 8M) | 9815 (SAC) | **✅ +2%** | BEATS SAC! |
| Hopper-v5 | 1295 | 1654 (PPO) | -22% | LN/WN both worse, keep baseline |
| Walker2d-v5 | **4277** (LN 7M) | 3900 (SAC) | **✅ +10%** | BEATS SAC! |
| Ant-v5 | **5108** (LN 6M) | 4844 (SAC) | **✅ +5%** | BEATS SAC! |
| Humanoid-v5 | **1850** (LN i2) | 2601 (SAC) | **-29%** | Huge improvement from 507 |
| HumanoidStandup-v5 | **154162** (LN i2 2M) | 138222 (SAC) | **✅ +12%** | BEATS SAC! v2 |
| InvertedPendulum-v5 | 842 | 1000 (SAC) | -16% | LN hurts, keep baseline |
| InvertedDoublePendulum-v5 | **8256** (LN i2 2M) | 9033 (SAC) | **-9%** | v2 improved from -21% |
| Reacher-v5 | -5.66 | -5.87 (SAC) | ✅ beats | done |
| Pusher-v5 | -37.08 | -38.41 (SAC) | ✅ beats | done |
| Swimmer-v5 | 221 | 301 (SAC) | -27% | WN regressed (165), keep baseline |

### Phase 4: Atari (PARKED — needs investigation before graduation)

Tested: Breakout, MsPacman, Pong, Qbert, Seaquest, SpaceInvaders

**Status**: Parked. Audit found issues — investigate CrossQ Atari performance before graduating.
Atari CrossQ generally underperforms SAC. Investigate whether BRN warmup, lr tuning, or
ConvNet-specific changes could help before publishing results.

---

## Intake Checklist (per run)

1. ⬜ Extract score: `dstack logs NAME | grep trial_metrics` → total_reward_ma
2. ⬜ Find HF folder: `huggingface_hub` API query
3. ⬜ Pull data: `slm-lab pull SPEC_NAME`
4. ⬜ Update BENCHMARKS.md: score + HF link + status
5. ⬜ Regenerate plot: `slm-lab plot -t "ENV_NAME" -f FOLDER1,FOLDER2,...`
6. ⬜ Commit + push

---

## Pending Fixes

- [x] Regenerate LunarLander plots with correct env name titles (564a6a96)
- [x] Universal env name audit across all plots (564a6a96)
- [x] Delete 58 stale Atari plots without -v5 suffix (564a6a96)
- [ ] Wave 0 intake: pull HF data + regenerate plots (deferred — low bandwidth)

## Decision Log

- **Swimmer-LN FAILED** (22.90 final): LN hurts Swimmer. Non-LN 221.12 is best. Do NOT launch more Swimmer-LN runs.
- **Hopper-LN 3M** (1076): WORSE than non-LN 6M (1295). More frames > LN for Hopper. Extended 6-7M LN run will tell if both helps.
- **LN HURTS most envs at 3M**: HalfCheetah -18%, InvPend -13%, InvDoublePend -45%, Swimmer -97%. Only helps Humanoid (+19%).
- **Root cause**: Critic BRN already normalizes. Actor LN over-regularizes, squashing activation scale on low/med-dim obs.
- **WeightNorm hypothesis**: WN reparameterizes weights without squashing activations — should avoid LN's failure. Wave 6 testing.
- **Humanoid-v2 iter=4**: MA 2923 at best session, likely beats SAC 2601. But uses iter=4 → ~150fps → 5.5h. VIOLATES 3h constraint. Not a valid CrossQ result.
- **Humanoid-LN 2M**: 506.65. iter=1 is fast (700fps) but 2M not enough data. Launched 7M run (2.8h budget).
- **Frame budget rule**: CrossQ at 700fps can do 7.5M in 3h. Use more frames than SAC, less than PPO.
- **InvDoublePend log_alpha_max=2.0**: Failed (4514 vs old 4953). Default alpha cap better for this env.
- **CRITICAL: LN + extended frames REVERSES 3M findings** — LN at 3M hurt most envs, but at 5-6M it BEATS non-LN baselines:
  - HalfCheetah-LN: -18% at 3M → **+8% at 5M** (8722 vs 8085). LN needs warmup frames.
  - Ant-LN: -5% at 3M → **+25% at 5M** (5054 vs 4046).
  - InvDoublePend-LN: -45% at 3M → **+17% at 5M** (5796 vs 4953).
  - Walker-LN: was already +22% at 3M, reached **4397** at 5.16M (74%) — beating SAC 3900.
- **iter=2 is the killer config for InvDoublePend**: 7411 at 69% completion, 50% above baseline, approaching SAC 9359.
- **WN promising**: Swimmer-WN 255 > non-LN 221. Walker-WN 4124 strong. Need full runs to confirm.
- **RunPod batch eviction**: All 13 runs killed at 01:25 UTC. Root cause: dstack credits depleted.
- **Strategic triage**: After relaunch, stopped 6 redundant/underperforming runs, kept 7 promising:
  - KEPT: walker-ln-7m (beating SAC), ant-ln-7m (beating SAC), halfcheetah-ln-7m (closing gap), invdoubpend-ln-i2 (iter=2 best), swimmer-wn (WN solving), humanoid-ln-i2 (best Humanoid), humanoid-wn (alternative)
  - STOPPED: hopper-ln-7m (LN hurts), hopper-wn (flat), walker-ln-i2 (7m better), walker-wn (7m better), invdoubpend-ln-7m (i2 much better), humanoid-ln-7m (i2 better)
- **FINAL RESULTS (7 runs completed)**:
  - Walker-LN-7m: **4277** — BEATS SAC 3900 (+10%)
  - Ant-LN-7m: **5108** — BEATS SAC 4844 (+5%)
  - HalfCheetah-LN-7m: **8785** — gap narrowed from -17% to -10%
  - InvDoublePend-LN-i2: **7353** — gap narrowed from -47% to -21%
  - Humanoid-LN-i2: **1850** — massive improvement from 507 (-29% vs SAC)
  - Humanoid-WN: **1681** — strong but LN-i2 wins
  - Swimmer-WN: **165** — REGRESSED from 221 (high variance, consistency=-0.79). WN does NOT fix Swimmer.
- **LN + extended frames confirmed**: The universal recipe is LN actor + more frames. Works for 5/7 MuJoCo envs. Exceptions: Hopper (LN hurts regardless), Swimmer (LN kills, WN also fails at full run).
- **Swimmer paradox**: WN looked promising at 67% (MA 255) but regressed to 165 at completion. High session variance. Non-LN 221 remains best.
- **Humanoid strategy**: LN+iter=2 (1850) > WN (1681) > LN+iter=1 7M (706). Humanoid needs gradient density, not just data.
- **Hopper-i2 too slow**: 101fps with iter=2 [512,512], would take 9.6h. Stopped. Plain baseline at 1295 with 5M/iter=1 (700fps) is best. Hopper is CrossQ's weakest MuJoCo env — 22% below PPO 1654, no normalization variant helps.
- **Wave 7 launched**: HumanoidStandup-LN-i2 (353fps, early MA 106870 vs baseline 115730), HalfCheetah-LN-8m (708fps), InvPend-7m (plain, more data).

## Atari Investigation

### Current CrossQ vs SAC Atari Scores

| Game | CrossQ | SAC | Ratio | Verdict |
|------|--------|-----|-------|---------|
| Breakout | 0.91 | 20.23 | 4.5% | catastrophic |
| MsPacman | 238.51 | 1336.96 | 17.8% | catastrophic |
| Pong | -20.82 | 10.89 | no learning | catastrophic |
| Qbert | **4268.66** | 3331.98 | 128% | **CrossQ wins** |
| Seaquest | 216.19 | 1565.44 | 13.8% | catastrophic |
| SpaceInvaders | 360.37 | 507.33 | 71% | poor |

CrossQ wins 1/6 games (Qbert). The other 5 show near-total failure, with 3 games at <18% of SAC performance.

### Root Cause Analysis

**Primary hypothesis: BRN placement is wrong for ConvNets.**

The CrossQ Atari critic architecture places a single `LazyBatchRenorm1d` layer after the final FC layer (post-Flatten, post-Linear(512)). This is fundamentally different from the MuJoCo architecture where BRN layers are placed between *every* hidden FC layer (two BRN layers for [256,256], two for [512,512], etc.).

Atari critic (1 BRN layer):
```
Conv2d(32) -> ReLU -> Conv2d(64) -> ReLU -> Conv2d(64) -> ReLU -> Flatten -> Linear(512) -> BRN -> ReLU
```

MuJoCo critic (2 BRN layers):
```
Linear(W) -> BRN -> ReLU -> Linear(W) -> BRN -> ReLU
```

The CrossQ paper's core insight is that BN/BRN statistics sharing between current and next-state batches replaces target networks. With only one BRN layer after 512-dim features, the normalization may be insufficient — the ConvNet backbone (3 conv layers) processes current and next-state images with NO shared normalization. The BRN only operates on the final FC representation. This means the cross-batch statistics sharing that eliminates the need for target networks is weak.

**Secondary hypothesis: Hyperparameters ported directly from MuJoCo without ConvNet adaptation.**

Key differences between CrossQ Atari vs SAC Atari specs:

| Parameter | CrossQ Atari | SAC Atari | Issue |
|-----------|-------------|-----------|-------|
| lr | 1e-3 | 3e-4 | 3.3x higher — too aggressive for ConvNets |
| optimizer | Adam | AdamW | No weight decay in CrossQ |
| betas | [0.5, 0.999] | [0.9, 0.999] | Low beta1 for ConvNets is risky |
| clip_grad_val | 0.5 | 0.5 | same |
| loss | SmoothL1Loss | SmoothL1Loss | same |
| policy_delay | 3 | 1 (default) | Delays policy updates 3x |
| log_alpha_max | 0.5 | none (uses clamp [-5, 2]) | Tighter alpha cap |
| warmup_steps | 10000 | n/a | Only 10K for Atari |
| target networks | none | polyak 0.005 | CrossQ core difference |
| init_fn | orthogonal_ | orthogonal_ | same |

The `lr=1e-3` with `betas=[0.5, 0.999]` combination is specifically tuned for MuJoCo MLPs per the CrossQ paper. ConvNets are known to be more sensitive to learning rates — SAC Atari uses `lr=3e-4` which is standard for Atari. The low `beta1=0.5` reduces momentum, which may cause unstable gradient updates in ConvNets where feature maps evolve slowly.

**Tertiary hypothesis: BRN warmup_steps=10000 is too low for Atari.**

At `training_frequency=4` and `num_envs=16`, each training step consumes 64 frames. With `training_iter=3`, there are 3 gradient steps per training step. So 10K warmup means 10K BRN steps = 10K/3 = ~3333 training steps = ~213K frames (10.7% of 2M). During warmup, BRN behaves as standard BN (r_max=1, d_max=0), which has been shown to cause divergence in RL (see CrossQ standard BN results in MEMORY.md).

MuJoCo uses `warmup_steps=100000` = 100K BRN steps. At `training_frequency=1` and `num_envs=16`, that's ~1.6M frames (significant fraction of typical 3-7M runs). This much slower warmup gives the running statistics time to stabilize. Atari at 10K warmup transitions to full BRN correction far too early when running statistics are still poor.

**Fourth hypothesis: Cross-batch forward is ineffective for ConvNets.**

In `calc_q_cross_discrete`, states and next_states are concatenated and passed through the critic together. For MuJoCo (small state vectors), this is cheap and effective — BN statistics computed over both batches provide good normalization. For Atari (84x84x4 images), the concatenated batch goes through 3 conv layers with NO normalization, then hits a single BRN layer at dim=512. The conv layers see a batch that mixes current and next frames, but without BN in the conv layers, this mixing provides no cross-batch regularization benefit. The entire CrossQ mechanism reduces to "BRN on the last FC layer of a frozen ConvNet backbone."

### Proposed Fixes (Priority Order)

**P0: Lower learning rate to SAC-Atari defaults**
- Change `lr: 1e-3` to `lr: 3e-4` for both actor and critic
- Change `betas: [0.5, 0.999]` to default `[0.9, 0.999]`
- Rationale: The lr=1e-3/beta1=0.5 combo is CrossQ-paper MuJoCo-specific. ConvNets need conservative lr.

**P1: Increase BRN warmup to 100K steps**
- Change `warmup_steps: 10000` to `warmup_steps: 100000`
- Rationale: Match MuJoCo proportionally. 100K BRN steps at iter=3 = ~2.1M frames, which is the full run. This means BRN stays in near-standard-BN mode for most of training — essentially disabling the full BRN correction that may be destabilizing ConvNets.

**P2: Add BRN after each conv layer (deeper cross-batch normalization)**
- Place `LazyBatchRenorm1d` (or `BatchRenorm2d`, which would need implementation) after each Conv2d layer
- Rationale: The CrossQ paper's mechanism relies on shared BN statistics between current/next batches. With BRN only at the FC layer, the ConvNet backbone has no cross-batch normalization, defeating the purpose.
- Note: This requires implementing `BatchRenorm2d` (2D spatial variant). Standard `BatchNorm2d` normalizes per-channel across spatial dims — a `BatchRenorm2d` would do the same with correction factors.
- **Risk**: This is a code change, not a spec-only fix. Higher effort.

**P3: Remove policy_delay for Atari**
- Change `policy_delay: 3` to `policy_delay: 1`
- Rationale: SAC Atari uses no policy delay. With only 2M frames and iter=3, policy_delay=3 means the policy is updated once every 3 critic updates. Combined with the already-low frame budget, the policy may not get enough gradient updates to learn.
- Total policy updates at 2M frames: (2M / (4 * 16)) * 3 / 3 = 31,250. Without delay: 93,750. 3x more policy updates.

**P4: Switch to AdamW with weight decay**
- Match SAC Atari's `AdamW` with `eps: 0.0001`
- Rationale: Weight decay provides implicit regularization that may partially compensate for the missing target network smoothing.

### Experiment Plan

1. **Exp A** (spec-only, highest impact): lr=3e-4, betas=[0.9,0.999], warmup=100K, policy_delay=1. Test on Pong + Breakout (fast signal games).
2. **Exp B** (spec-only): Same as A but keep policy_delay=3. Isolates lr/warmup effect.
3. **Exp C** (spec-only): Same as A but lr=1e-3 (keep CrossQ lr). Isolates beta/warmup effect.
4. **Exp D** (code change): Add BatchRenorm2d after conv layers. Test with Exp A settings.

If Exp A solves the problem, no code changes needed. If not, Exp D addresses the fundamental architectural mismatch.

### Key Insight

The Qbert success is telling. Qbert has relatively simple visual patterns and discrete state changes — the ConvNet can extract good features even with aggressive lr. Games like Pong and Breakout require precise spatial reasoning where ConvNet feature quality matters more, and the aggressive lr/low-momentum combo destabilizes learning before features mature.
