# GPU Tensor Path Design: Zero-Copy DLPack for MuJoCo Playground

**Status**: Approved design (Phase 2)
**Authors**: engineer-gpu-a, engineer-gpu-b
**Date**: 2026-03-05

## Summary

Add an optional GPU tensor path for MuJoCo Playground environments using DLPack zero-copy transfer between JAX and PyTorch. This eliminates unnecessary GPU-CPU-GPU roundtrips and JAX synchronization barriers on the act path, yielding an estimated 2-10% wall-time improvement with minimal code changes.

## Problem

MuJoCo Playground runs simulations on GPU via JAX/MJX. The current `PlaygroundVecEnv` converts JAX GPU arrays to numpy at the environment boundary:

```
JAX GPU tensor → np.asarray() [GPU→CPU sync + copy] → numpy
                                                        ↓
numpy → torch.from_numpy() → .to(device) [CPU→GPU copy] → PyTorch CUDA tensor
```

This forces two host-device transfers per step, plus a JAX synchronization barrier on `np.asarray()`. The sync barrier (~10-20us) blocks the Python thread until JAX computation completes, preventing pipelining of JAX simulation with PyTorch inference.

## Design

### Architecture: Single Class with `device` Parameter

`PlaygroundVecEnv` gets a `device` parameter. When set, `_to_output()` returns PyTorch CUDA tensors via DLPack instead of numpy. This is the **sole branching point** — no other method checks `device`.

```python
class PlaygroundVecEnv(gym.vector.VectorEnv):
    def __init__(self, env_name: str, num_envs: int, seed: int = 0, device: str | None = None):
        # ... existing init ...
        self._device = device
        if device is not None:
            import torch
            self._torch_device = torch.device(device)

    def _to_output(self, x: jax.Array):
        """Convert JAX array to output format. DLPack zero-copy when device is set."""
        if self._device is not None:
            return torch.from_dlpack(jax.dlpack.to_dlpack(x))
        return np.asarray(x).astype(np.float32)

    def _get_obs(self, state):
        obs = state.obs
        if isinstance(obs, dict):
            arrays = [obs[k] for k in sorted(obs.keys())]
            obs = jnp.concatenate(arrays, axis=-1)
        return self._to_output(obs)
```

**Rationale for single class over subclass**: A subclass that overrides `step()` duplicates the full step body (obs extraction, truncation handling, info extraction). When the parent changes, the subclass silently diverges. Overriding only `_to_output()` via a subclass is over-engineered for a single method. The `device` parameter with `_to_output()` as the sole branching point keeps all logic in one place with zero duplication risk.

### Activation via Spec

```json
{
  "env": {
    "name": "playground/HumanoidWalk",
    "num_envs": 16,
    "device": "cuda"
  }
}
```

In `slm_lab/env/__init__.py`, `_make_playground_env()` reads `env_spec.get("device")` and passes it to `PlaygroundVecEnv`:
- `device=None` (default): numpy output (backward compatible)
- `device="cuda"`: DLPack zero-copy PyTorch CUDA tensor output

### Data Flow with GPU Path

```
                          ACT PATH (zero-copy)
JAX GPU tensor ──DLPack──► PyTorch CUDA tensor ──► net.forward() ──► action
                                                                       │
                          MEMORY PATH (CPU copy)                       │
PyTorch CUDA tensor ──.cpu().numpy()──► numpy ──► replay buffer        │
                                                       │               │
                          TRAIN PATH (unchanged)       │               │
replay buffer ──to_torch_batch()──► .to(device) ──► training loop      │
                                                                       │
                          ACTION PATH (unchanged)                      │
action ──.cpu().numpy()──► numpy ──► env.step(actions) ◄───────────────┘
```

### Required Code Changes

#### 1. `PlaygroundVecEnv` in `slm_lab/env/playground.py` (~10 lines changed)

Add `device` parameter to `__init__`. Replace `_to_numpy()` with `_to_output()` that branches on `self._device`. Update `_get_obs()` and `step()` to call `_to_output()`.

#### 2. `_make_playground_env()` in `slm_lab/env/__init__.py` (~3 lines)

Read `device` from env spec and pass to `PlaygroundVecEnv`. Skip numpy-only wrappers when `device` is set.

#### 3. `guard_tensor()` in `slm_lab/agent/algorithm/policy_util.py` (~3 lines)

Add early return for torch.Tensor inputs before the numpy path:

```python
def guard_tensor(state, agent):
    if torch.is_tensor(state):
        if not agent.env.is_venv:
            state = state.unsqueeze(dim=0)
        return state
    # existing numpy path unchanged
    if not isinstance(state, np.ndarray):
        state = np.asarray(state)
    state = torch.from_numpy(np.ascontiguousarray(state))
    if not agent.env.is_venv:
        state = state.unsqueeze(dim=0)
    return state
```

Note: `calc_pdparam()` (line 80) already checks `if not torch.is_tensor(state)` and only converts from numpy when needed. The `.to(device)` call on line 82 is a no-op when the tensor is already on the correct device.

#### 4. `replay.add_experience()` in `slm_lab/agent/memory/replay.py` (~3 lines)

Guard for torch tensor inputs at the top of the method:

```python
if torch.is_tensor(state):
    state = state.cpu().numpy()
if torch.is_tensor(next_state):
    next_state = next_state.cpu().numpy()
```

#### 5. No changes needed

- **`to_action()`**: Already calls `action.cpu().numpy()` -- works for both numpy and CUDA tensor actions.
- **`to_torch_batch()`**: Operates on numpy from replay buffer -- unchanged.
- **Training loop**: Receives torch tensors from `to_torch_batch()` -- unchanged.
- **Action input to env**: `PlaygroundVecEnv.step()` receives numpy actions, converts to JAX via `jnp.array()` -- unchanged.

### Gymnasium Wrappers in GPU Mode

When `device` is set, gymnasium vector wrappers (`VectorNormalizeObservation`, `VectorClipObservation`, etc.) are **not applied** because they expect numpy arrays.

Normalization in GPU mode is handled by:
- **CrossQ**: BatchRenorm in critic networks handles input scaling internally
- **SAC**: Network-level normalization (`normalize=true` in net spec) or raw observations (MuJoCo obs are bounded)
- **PPO**: Network-level normalization or future torch-native wrapper

If benchmarks show that running-mean observation normalization is needed for GPU mode, a torch-native `VectorNormalizeObservation` equivalent can be added in a follow-up PR.

### Requirements

- **`XLA_PYTHON_CLIENT_PREALLOCATE=false`**: Must be set as an environment variable. JAX pre-allocates GPU memory by default, which can cause OOM when sharing GPU with PyTorch. This disables pre-allocation so both frameworks use on-demand allocation.
- **Same GPU**: JAX and PyTorch must use the same GPU device for DLPack zero-copy to work.

## Performance Analysis

### Speedup Source

The primary speedup comes from **eliminating JAX synchronization barriers**, not PCIe bandwidth savings:

- `np.asarray(jax_array)` forces a JAX `device_get()` which blocks the Python thread until the JAX computation completes (~10-20us per call)
- DLPack shares the underlying GPU buffer without synchronization -- PyTorch reads the data when it needs it (in the forward pass), which is already a GPU-GPU dependency
- This enables pipelining: JAX can begin the next simulation step while PyTorch runs inference on the current observation

### Estimated Wall-Time Improvement

| Algorithm | training_iter | Act Path Fraction | Estimated Speedup |
|-----------|--------------|-------------------|-------------------|
| CrossQ    | 1            | ~50% of step time | 5-10%             |
| SAC       | 4-8          | ~20% of step time | 2-5%              |
| PPO       | N/A (on-policy) | ~30%           | 2-5%              |

CrossQ benefits most because `training_iter=1` means the act path is a larger fraction of total wall time. Note: these estimates are conservative. The sync barrier elimination enables pipelining whose benefit is workload-dependent and hard to predict precisely without benchmarking.

### Memory overhead

Negligible. DLPack is zero-copy -- the torch tensor shares the JAX buffer. The only additional memory is the torch tensor metadata (~100 bytes per tensor).

## Decision Rationale

**Why not a separate `PlaygroundGPUEnv` subclass?**
Initially proposed, but rejected during design review. A subclass overriding only `_to_output()` is over-engineered for a single method. A subclass overriding `step()` duplicates the full step body and silently diverges when the parent changes. The `device` parameter with `_to_output()` as the sole branching point is simpler, has zero duplication risk, and keeps all logic in one place.

**Why not a full GPU pipeline (GPU memory, GPU wrappers)?**
The replay buffer stores 1M+ experiences. Keeping these as CUDA tensors would consume significant GPU memory with no benefit (random-access sampling doesn't benefit from GPU parallelism). The memory-to-train path via `to_torch_batch()` is called infrequently relative to the act path.

**Why not torch-native normalization wrappers?**
Writing and maintaining a parallel normalizer that exactly matches `VectorNormalizeObservation` (Welford algorithm, epsilon, clipping) creates a testing/maintenance burden and behavioral divergence risk. Network-level normalization (BatchRenorm, LayerNorm) is already in place for the primary use cases (CrossQ, SAC). Deferred per YAGNI — add only if benchmarks show it's needed.

## Future Work (Phase 3, Out of Scope)

**GPU-native replay buffer**: Store experiences as torch CUDA tensors in a pre-allocated GPU buffer instead of numpy. This would eliminate the remaining GPU-CPU-GPU roundtrip in the memory path (`_to_output()` → `.cpu().numpy()` → store → `to_torch_batch()` → `.to(device)`). Estimated additional speedup: 20-30%.

This requires significant changes to the memory module and is deferred to Phase 3.
