# Gymnasium Migration Guide

This document explains the principled approach to handling episode termination in SLM-Lab after migrating from OpenAI Gym to Gymnasium.

## The Three Signals

Gymnasium v1.0.0 changed episode termination from a single `done` signal to three signals:

```python
# Old (Gym):
observation, reward, done, info = env.step(action)

# New (Gymnasium):
observation, reward, terminated, truncated, info = env.step(action)
```

### Signal Definitions

1. **`terminated`**: True episode end (agent reached goal or failure state)
   - Example: CartPole falls over, goal reached in maze
   - Bootstrapping: **Do NOT bootstrap** - future returns should be zero

2. **`truncated`**: Time limit reached (episode cut off artificially)
   - Example: Max steps reached (500 for CartPole, 200 for MountainCar)
   - Bootstrapping: **DO bootstrap** - estimate V(s') for remaining value

3. **`done`**: Episode boundary for resets
   - Definition: `done = terminated OR truncated`
   - Usage: Trigger environment reset, end episode collection

## Implementation Principles

### 1. Memory Storage

All memory classes store all three signals:

```python
self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones', 'terminateds', 'truncateds']
```

**Rationale**: Algorithms need `terminated` for correct bootstrapping, `done` for episode boundaries.

### 2. Return/Advantage Calculations

Use `terminated` (NOT `done`) for zeroing future returns:

```python
# Correct - uses terminated
def calc_returns(rewards, terminateds, gamma):
    not_terminateds = 1 - terminateds
    for t in reversed(range(T)):
        rets[t] = rewards[t] + gamma * future_ret * not_terminateds[t]
    return rets

# Correct - uses terminated
def calc_gaes(rewards, terminateds, v_preds, gamma, lam):
    not_terminateds = 1 - terminateds
    deltas = rewards + gamma * v_preds[1:] * not_terminateds - v_preds[:-1]
    for t in reversed(range(T)):
        gaes[t] = deltas[t] + coef * not_terminateds[t] * future_gae
    return gaes
```

**Rationale**: When truncated, future returns exist but weren't observed. Bootstrap from V(s').

### 3. Q-Learning Targets

Use `terminated` (NOT `done`) for target calculation:

```python
# Correct - DQN/DDQN/SAC
q_targets = rewards + gamma * (1 - terminateds) * max_next_q_preds

# Incorrect - would zero out value on time limit
q_targets = rewards + gamma * (1 - dones) * max_next_q_preds
```

**Rationale**: Same as returns - time limits shouldn't zero out future value.

### 4. Episode Boundaries

Use `done` for episode resets and collection:

```python
# Control loop
done = np.logical_or(terminated, truncated)
if util.epi_done(done):
    state, info = env.reset()

# Episode collection (onpolicy)
if util.epi_done(done):
    # Save complete episode
    for k in self.data_keys:
        getattr(self, k).append(self.cur_epi_data[k])
```

**Rationale**: Need to reset environment whether terminated OR truncated.

### 5. Backward Compatibility

When parameters are missing, derive them from `done`:

```python
def update(self, state, action, reward, next_state, done, terminated=None, truncated=None):
    # Backward compatibility: derive missing values from done
    if terminated is None and truncated is None:
        terminated = done
        truncated = False
    elif terminated is None:
        terminated = np.logical_and(done, np.logical_not(truncated))
    elif truncated is None:
        truncated = np.logical_and(done, np.logical_not(terminated))
```

**Rationale**: Safe defaults for old code - assumes no time limits (common for simple envs).

## Common Mistakes

### ❌ Using `done` for bootstrapping

```python
# WRONG - zeros out value on time limits
q_targets = rewards + gamma * (1 - dones) * max_next_q_preds
```

This was correct in old Gym (no time limits), but breaks with Gymnasium truncation.

### ❌ Using `terminated` for episode resets

```python
# WRONG - won't reset on time limit
if util.epi_done(terminated):
    state, info = env.reset()
```

Environment needs reset on both terminated AND truncated.

### ❌ Storing only `done` and `terminated`

```python
# INCOMPLETE - can't reconstruct truncated
self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones', 'terminateds']
```

Need all three for complete information and debugging.

## Environment-Specific Behavior

### CartPole-v1 / Acrobot-v1 / LunarLander-v3
- No explicit time limits in Gymnasium versions
- `truncated` is always False (unless custom wrapper)
- Using `done` vs `terminated` has same effect
- Still use `terminated` for correctness with other envs

### MountainCar-v0
- Has 200-step time limit
- `truncated=True` when time limit hit without reaching goal
- MUST use `terminated` for bootstrapping or agent can't learn

### MuJoCo (HalfCheetah, Walker2d, etc.)
- Have 1000-step time limits
- Frequently hit time limits during training
- Using `done` instead of `terminated` significantly hurts learning

### Atari
- May have time limits depending on wrapper configuration
- Check `gymnasium.make_vec()` wrapper settings

## Testing Strategy

1. **Test on simple envs first** (CartPole, Acrobot)
   - No time limits, so `done==terminated`
   - Errors more likely from bugs than bootstrapping

2. **Test on time-limited envs** (MountainCar, MuJoCo)
   - Exposes bootstrapping correctness
   - Compare learning curves with/without correct `terminated` usage

3. **Check edge cases**
   - Episodes that end naturally
   - Episodes that hit time limit
   - Very short/long episodes

## References

- Gymnasium Migration Guide: https://gymnasium.farama.org/introduction/migration_guide/
- Gymnasium v1.0.0 Release: https://github.com/Farama-Foundation/Gymnasium/releases/tag/v1.0.0
