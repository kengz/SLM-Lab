# Phase 5: Brax PPO vs SLM-Lab PPO — Comprehensive Comparison

Source: `google/brax` (latest `main`) and `google-deepmind/mujoco_playground` (latest `main`).
All values extracted from actual code, not documentation.

---

## 1. Batch Collection Mechanics

### Brax
The training loop in `brax/training/agents/ppo/train.py` (line 586–591) collects data via nested `jax.lax.scan`:

```python
(state, _), data = jax.lax.scan(
    f, (state, key_generate_unroll), (),
    length=batch_size * num_minibatches // num_envs,
)
```

Each inner call does `generate_unroll(env, state, policy, key, unroll_length)` — a `jax.lax.scan` of `unroll_length` sequential env steps. The outer scan repeats this `batch_size * num_minibatches // num_envs` times **sequentially**, rolling the env state forward continuously.

**DM Control default**: `num_envs=2048`, `batch_size=1024`, `num_minibatches=32`, `unroll_length=30`.
- Outer scan length = `1024 * 32 / 2048 = 16` sequential unrolls.
- Each unroll = 30 steps.
- Total data per training step = 16 * 2048 * 30 = **983,040 transitions** reshaped to `(32768, 30)`.
- Then `num_updates_per_batch=16` SGD passes, each splitting into 32 minibatches.
- **Effective gradient steps per collect**: 16 * 32 = 512.

### SLM-Lab
`time_horizon=30`, `num_envs=2048` → collects `30 * 2048 = 61,440` transitions.
`training_epoch=16`, `minibatch_size=4096` → 15 minibatches per epoch → 16 * 15 = 240 gradient steps.

### Difference
**Brax collects 16x more data per training step** by doing 16 sequential unrolls before updating. SLM-Lab does 1 unroll. This means Brax's advantages are computed over much longer trajectories (480 steps vs 30 steps), providing much better value bootstrap targets.

Brax also shuffles the entire 983K-transition dataset into minibatches, enabling better gradient estimates.

**Classification: CRITICAL**

**Fix**: Increase `time_horizon` or implement multi-unroll collection. The simplest fix: increase `time_horizon` from 30 to 480 (= 30 * 16). This gives the same data-per-update ratio. However, this would require more memory. Alternative: keep `time_horizon=30` but change `training_epoch` to 1 and let the loop collect multiple horizons before training — requires architectural changes.

**Simplest spec-only fix**: Set `time_horizon=480` (or even 256 as a compromise). This is safe because GAE with `lam=0.95` naturally discounts old data. Risk: memory usage increases 16x for the batch buffer.

---

## 2. Reward Scaling

### Brax
`reward_scaling` is applied **inside the loss function** (`losses.py` line 212):
```python
rewards = data.reward * reward_scaling
```
This scales rewards just before GAE computation. It does NOT modify the environment rewards.

**DM Control default**: `reward_scaling=10.0`
**Locomotion default**: `reward_scaling=1.0`
**Manipulation default**: `reward_scaling=1.0` (except PandaPickCubeCartesian: 0.1)

### SLM-Lab
`reward_scale` is applied in the **environment wrapper** (`playground.py` line 149):
```python
rewards = np.asarray(self._state.reward) * self._reward_scale
```

**Current spec**: `reward_scale: 10.0` (DM Control)

### Difference
Functionally equivalent — both multiply rewards by a constant before GAE. The location (env vs loss) shouldn't matter for PPO since rewards are only used in GAE computation.

**Classification: MINOR** — Already matching for DM Control.

---

## 3. Observation Normalization

### Brax
Uses Welford's online algorithm to track per-feature running mean/std. Applied via `running_statistics.normalize()`:
```python
data = (data - mean) / std
```
Mean-centered AND divided by std. Updated **every training step** before SGD (line 614).
`normalize_observations=True` for all environments.
`std_eps=0.0` (default, no epsilon in std).

### SLM-Lab
Uses gymnasium's `VectorNormalizeObservation` (CPU) or `TorchNormalizeObservation` (GPU), which also uses Welford's algorithm with mean-centering and std division.

**Current spec**: `normalize_obs: true`

### Difference
Both use mean-centered running normalization. Brax updates normalizer params inside the training loop (not during rollout), while SLM-Lab updates during rollout (gymnasium wrapper). This is a subtle timing difference but functionally equivalent.

Brax uses `std_eps=0.0` by default, while gymnasium uses `epsilon=1e-8`. Minor numerical difference.

**Classification: MINOR** — Already matching.

---

## 4. Value Function

### Brax
- **Loss**: Unclipped MSE by default (`losses.py` line 252–263):
  ```python
  v_error = vs - baseline
  v_loss = jnp.mean(v_error * v_error) * 0.5 * vf_coefficient
  ```
- **vf_coefficient**: 0.5 (default in `train.py`)
- **Value clipping**: Only if `clipping_epsilon_value` is set (default `None` = no clipping)
- **No value target normalization** — raw GAE targets
- **Separate policy and value networks** (always separate in Brax's architecture)
- Value network: 5 hidden layers of 256 (DM Control default) with `swish` activation
- **Bootstrap on timeout**: Optional, default `False`

### SLM-Lab
- **Loss**: MSE with `val_loss_coef=0.5`
- **Value clipping**: Optional via `clip_vloss` (default False)
- **Value target normalization**: Optional via `normalize_v_targets: true` using `ReturnNormalizer`
- **Architecture**: `[256, 256, 256]` with SiLU (3 layers vs Brax's 5)

### Difference
1. **Value network depth**: Brax uses **5 layers of 256** for DM Control, SLM-Lab uses **3 layers of 256**. This is a meaningful capacity difference for the value function, which needs to accurately estimate returns.

2. **Value target normalization**: SLM-Lab has `normalize_v_targets: true` with a `ReturnNormalizer`. Brax does NOT normalize value targets. This could cause issues if the normalizer is poorly calibrated.

3. **Value network architecture (Loco)**: Brax uses `[256, 256, 256, 256, 256]` for loco too.

**Classification: IMPORTANT**

**Fix**:
- Consider increasing value network to 5 layers: `[256, 256, 256, 256, 256]` to match Brax.
- Consider disabling `normalize_v_targets` since Brax doesn't use it and `reward_scaling=10.0` already provides good gradient magnitudes.
- Risk of regressing: the return normalizer may be helping envs with high reward variance. Test with and without.

---

## 5. Advantage Computation (GAE)

### Brax
`compute_gae` in `losses.py` (line 38–100):
- Standard GAE with `lambda_=0.95`, `discount=0.995` (DM Control)
- Computed over each unroll of `unroll_length` timesteps
- Uses `truncation` mask to handle episode boundaries within an unroll
- `normalize_advantage=True` (default): `advs = (advs - mean) / (std + 1e-8)` over the **entire batch**
- GAE is computed **inside the loss function**, once per SGD pass (recomputed each time with current value estimates? No — computed once with data from rollout, including stored baseline values)

### SLM-Lab
- GAE computed in `calc_gae_advs_v_targets` using `math_util.calc_gaes`
- Computed once before training epochs
- Advantage normalization: per-minibatch standardization in `calc_policy_loss`:
  ```python
  advs = math_util.standardize(advs)  # per minibatch
  ```

### Difference
1. **GAE horizon**: Brax computes GAE over 30-step unrolls. SLM-Lab also uses 30-step horizon. **Match**.
2. **Advantage normalization scope**: Brax normalizes over the **entire batch** (983K transitions). SLM-Lab normalizes **per minibatch** (4096 transitions). Per-minibatch normalization has more variance. However, both approaches are standard — SB3 also normalizes per-minibatch.
3. **Truncation handling**: Brax explicitly handles truncation with `truncation_mask` in GAE. SLM-Lab uses `terminateds` from the env wrapper, with truncation handled by gymnasium's auto-reset. These should be functionally equivalent.

**Classification: MINOR** — Approaches are different but both standard.

---

## 6. Learning Rate Schedule

### Brax
Default: `learning_rate_schedule=None` → **no schedule** (constant LR).
Optional: `ADAPTIVE_KL` schedule that adjusts LR based on KL divergence.
Base LR: `1e-3` (DM Control), `3e-4` (Locomotion).

### SLM-Lab
Uses `LinearToMin` scheduler:
```yaml
lr_scheduler_spec:
  name: LinearToMin
  frame: "${max_frame}"
  min_factor: 0.033
```
This linearly decays LR from `1e-3` to `1e-3 * 0.033 = 3.3e-5` over training.

### Difference
**Brax uses constant LR. SLM-Lab decays LR by 30x over training.** This is a significant difference. Linear LR decay can help convergence in the final phase but can also hurt by reducing the LR too early for long training runs.

**Classification: IMPORTANT**

**Fix**: Consider removing or weakening the LR decay for playground envs:
- Option A: Set `min_factor: 1.0` (effectively constant LR) to match Brax
- Option B: Use a much gentler decay, e.g. `min_factor: 0.1` (10x instead of 30x)
- Risk: Some envs may benefit from the decay. Test both.

---

## 7. Entropy Coefficient

### Brax
**Fixed** (no decay):
- DM Control: `entropy_cost=1e-2`
- Locomotion: `entropy_cost=1e-2` (some overrides to `5e-3`)
- Manipulation: varies, typically `1e-2` or `2e-2`

### SLM-Lab
**Fixed** (no_decay):
```yaml
entropy_coef_spec:
  name: no_decay
  start_val: 0.01
```

### Difference
**Match**: Both use fixed `0.01`.

**Classification: MINOR** — Already matching.

---

## 8. Gradient Clipping

### Brax
`max_grad_norm` via `optax.clip_by_global_norm()`:
- DM Control default: **None** (no clipping!)
- Locomotion default: `1.0`
- Vision PPO and some manipulation: `1.0`

### SLM-Lab
`clip_grad_val: 1.0` — always clips gradients by global norm.

### Difference
**Brax does NOT clip gradients for DM Control by default.** SLM-Lab always clips at 1.0.

Gradient clipping can be overly conservative, preventing the optimizer from taking large useful steps when gradients are naturally large (e.g., early training with `reward_scaling=10.0`).

**Classification: IMPORTANT** — Could explain slow convergence on DM Control envs.

**Fix**: Remove gradient clipping for DM Control playground spec:
```yaml
clip_grad_val: null  # match Brax DM Control default
```
Keep `clip_grad_val: 1.0` for locomotion spec. Risk: gradient explosions without clipping, but Brax demonstrates it works for DM Control.

---

## 9. Action Distribution

### Brax
Default: `NormalTanhDistribution` — samples from `Normal(loc, scale)` then applies `tanh` postprocessing.
- `param_size = 2 * action_size` (network outputs both mean and log_scale)
- Scale: `scale = (softplus(raw_scale) + 0.001) * 1.0` (min_std=0.001, var_scale=1)
- **State-dependent std**: The scale is output by the policy network (not a separate parameter)
- Uses `tanh` bijector with log-det-jacobian correction

### SLM-Lab
Default: `Normal(loc, scale)` without tanh.
- `log_std_init` creates a **state-independent** `nn.Parameter` for log_std
- Scale: `scale = clamp(log_std, -5, 0.5).exp()` → std range [0.0067, 1.648]
- **State-independent std** (when `log_std_init` is set)

### Difference
1. **Tanh squashing**: Brax applies `tanh` to bound actions to [-1, 1]. SLM-Lab does NOT. This is a fundamental architectural difference:
   - With tanh: actions are bounded, log-prob includes jacobian correction
   - Without tanh: actions can exceed env bounds, relying on env clipping

2. **State-dependent vs independent std**: Brax uses state-dependent std (network outputs it), SLM-Lab uses state-independent learnable parameter.

3. **Std parameterization**: Brax uses `softplus + 0.001` (min_std=0.001), SLM-Lab uses `clamp(log_std, -5, 0.5).exp()` with max std of 1.648.

4. **Max std cap**: SLM-Lab caps at exp(0.5)=1.648. Brax has no explicit cap (softplus can grow unbounded). However, Brax's `tanh` squashing means even large std doesn't produce out-of-range actions.

**Classification: IMPORTANT**

**Note**: For MuJoCo Playground where actions are already in [-1, 1] and the env wrapper has `PlaygroundVecEnv` with action space `Box(-1, 1)`, the `tanh` squashing may not be critical since the env naturally clips. But the log-prob correction matters for policy gradient quality.

**Fix**:
- The state-independent log_std is a reasonable simplification (CleanRL also uses it). Keep.
- The `max=0.5` clamp may be too restrictive. Consider increasing to `max=2.0` (CleanRL default) or removing the upper clamp entirely.
- Consider implementing tanh squashing as an option for playground envs.

---

## 10. Network Initialization

### Brax
Default: `lecun_uniform` for all layers (policy and value).
Activation: `swish` (= SiLU).
No special output layer initialization by default.

### SLM-Lab
Default: `orthogonal_` initialization.
Activation: SiLU (same as swish).

### Difference
- Brax uses `lecun_uniform`, SLM-Lab uses `orthogonal_`. Both are reasonable for swish/SiLU activations.
- `orthogonal_` tends to preserve gradient magnitudes across layers, which can be beneficial for deeper networks.

**Classification: MINOR** — Both are standard choices. `orthogonal_` may actually be slightly better for the 3-layer SLM-Lab network.

---

## 11. Network Architecture

### Brax (DM Control defaults)
- **Policy**: `(32, 32, 32, 32)` — 4 layers of 32, swish activation
- **Value**: `(256, 256, 256, 256, 256)` — 5 layers of 256, swish activation

### Brax (Locomotion defaults)
- **Policy**: `(128, 128, 128, 128)` — 4 layers of 128
- **Value**: `(256, 256, 256, 256, 256)` — 5 layers of 256

### SLM-Lab (ppo_playground)
- **Policy**: `(64, 64)` — 2 layers of 64, SiLU
- **Value**: `(256, 256, 256)` — 3 layers of 256, SiLU

### Difference
1. **Policy width**: SLM-Lab uses wider layers (64) but fewer (2 vs 4). Total params: ~similar for DM Control (4*32*32=4096 vs 2*64*64=8192). SLM-Lab's policy is actually larger per layer but shallower.

2. **Value depth**: 3 vs 5 layers. This is significant — the value function benefits from more depth to accurately represent complex return landscapes, especially for long-horizon tasks.

3. **DM Control policy**: Brax uses very small 32-wide networks. SLM-Lab's 64-wide may be slightly over-parameterized but shouldn't hurt.

**Classification: IMPORTANT** (mainly the value network depth)

**Fix**: Consider increasing value network to 5 layers to match Brax:
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

---

## 12. Clipping Epsilon

### Brax
Default: `clipping_epsilon=0.3` (in `train.py` line 206).
DM Control: not overridden → **0.3**.
Locomotion: some envs override to `0.2`.

### SLM-Lab
Default: `clip_eps=0.2` (in spec).

### Difference
Brax uses **0.3** while SLM-Lab uses **0.2**. This is notable — 0.3 allows larger policy updates per step, which can accelerate learning but risks instability. Given that Brax collects 16x more data per update (see #1), the larger clip epsilon is safe because the policy ratio variance is lower with more data.

**Classification: IMPORTANT** — Especially in combination with the batch size difference (#1).

**Fix**: Consider increasing to 0.3 for DM Control playground spec. However, this should only be done together with the batch size fix (#1), since larger clip epsilon with small batches risks instability.

---

## 13. Discount Factor

### Brax (DM Control)
Default: `discounting=0.995`
Overrides: BallInCup=0.95, FingerSpin=0.95

### Brax (Locomotion)
Default: `discounting=0.97`
Overrides: Go1Backflip=0.95

### SLM-Lab
DM Control: `gamma=0.995`
Locomotion: `gamma=0.97`
Overrides: FingerSpin=0.95

### Difference
**Match** for the main categories.

**Classification: MINOR** — Already matching.

---

## Summary: Priority-Ordered Fixes

### CRITICAL

| # | Issue | Brax Value | SLM-Lab Value | Fix |
|---|-------|-----------|--------------|-----|
| 1 | **Batch size (data per training step)** | 983K transitions (16 unrolls of 30) | 61K transitions (1 unroll of 30) | Increase `time_horizon` to 480, or implement multi-unroll collection |

### IMPORTANT

| # | Issue | Brax Value | SLM-Lab Value | Fix |
|---|-------|-----------|--------------|-----|
| 4 | **Value network depth** | 5 layers of 256 | 3 layers of 256 | Add 2 more hidden layers |
| 6 | **LR schedule** | Constant | Linear decay to 0.033x | Set `min_factor: 1.0` or weaken to 0.1 |
| 8 | **Gradient clipping (DM Control)** | None | 1.0 | Set `clip_grad_val: null` for DM Control |
| 9 | **Action std upper bound** | Softplus (unbounded) | exp(0.5)=1.65 | Increase max clamp from 0.5 to 2.0 |
| 11 | **Clipping epsilon** | 0.3 | 0.2 | Increase to 0.3 (only with larger batch) |

### MINOR (already matching or small effect)

| # | Issue | Status |
|---|-------|--------|
| 2 | Reward scaling | Match (10.0 for DM Control) |
| 3 | Obs normalization | Match (Welford running stats) |
| 5 | GAE computation | Match (lam=0.95, per-minibatch normalization) |
| 7 | Entropy coefficient | Match (0.01, fixed) |
| 10 | Network init | Minor difference (orthogonal vs lecun_uniform) |
| 13 | Discount factor | Match |

---

## Recommended Implementation Order

### Phase 1: Low-risk spec changes (test on CartpoleBalance/Swingup first)
1. Remove gradient clipping for DM Control: `clip_grad_val: null`
2. Weaken LR decay: `min_factor: 0.1` (or `1.0` for constant)
3. Increase log_std clamp from 0.5 to 2.0

### Phase 2: Architecture changes (test on several envs)
4. Increase value network to 5 layers of 256
5. Consider disabling `normalize_v_targets` since Brax doesn't use it

### Phase 3: Batch size alignment (largest expected impact, highest risk)
6. Increase `time_horizon` to 240 or 480 to match Brax's effective batch size
7. If time_horizon increase works, consider increasing `clipping_epsilon` to 0.3

### Risk Assessment
- **Safest changes**: #1 (no grad clip), #2 (weaker LR decay), #3 (wider std range)
- **Medium risk**: #4 (deeper value net — more compute, could slow training), #5 (removing normalization)
- **Highest risk/reward**: #6 (larger time_horizon — 16x more memory, biggest expected improvement)

### Envs Already Solved
Changes should be tested against already-solved envs (CartpoleBalance, CartpoleSwingup, etc.) to ensure no regression. The safest approach is to implement spec variants rather than modifying the default spec.

---

## Key Insight

The single largest difference is **data collection volume per training step**. Brax collects 16x more transitions before each update cycle. This provides:
1. Better advantage estimates (longer trajectory context)
2. More diverse minibatches (less overfitting per update)
3. Safety for larger clip epsilon and no gradient clipping

Without matching this, the other improvements will have diminished returns. The multi-unroll collection in Brax is fundamentally tied to its JAX/vectorized architecture — SLM-Lab's sequential PyTorch loop can approximate this by simply increasing `time_horizon`, at the cost of memory.

A practical compromise: increase `time_horizon` from 30 to 128 or 256 (4-8x, not full 16x) and adjust other hyperparameters accordingly.
