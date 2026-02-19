# SLM Lab Improvements Roadmap

SLM Lab's algorithms (PPO, SAC) are architecturally sound but use 2017-era defaults. This roadmap integrates material advances from the post-PPO RL landscape.

**Source**: [`notes/literature/ai/rl-landscape-2026.md`](../../notes/literature/ai/rl-landscape-2026.md)

**Hardware**: Mac (Apple Silicon) for dev, cloud GPU (A100/H100) for runs.

---

## Status

| Step | What | Status |
|:---:|------|--------|
| **1** | **GPU envs (MuJoCo Playground)** | **NEXT** |
| 2 | Normalization stack (layer norm, percentile) | DONE |
| 3 | CrossQ algorithm (batch norm critics) | DONE |
| 4 | Combine + full benchmark suite | TODO (after Step 1) |
| 5 | High-UTD SAC / RLPD | TODO |
| 6 | Pretrained vision encoders | TODO |

---

## NEXT: Step 1 — GPU Envs (MuJoCo Playground)

**Goal**: Remove env as the bottleneck. Run physics on GPU via [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground), keep training in PyTorch. Scale to 1000+ parallel envs for large-scale runs.

### The Stack

```
MuJoCo Playground  ← env definitions, registry, wrappers
       ↓
     Brax           ← EpisodeWrapper, AutoResetWrapper
       ↓
  MuJoCo MJX        ← JAX reimplementation of MuJoCo physics (GPU/TPU)
       ↓
   JAX / XLA        ← jit, vmap
```

### API Difference

Playground uses a **stateless functional API**, not Gymnasium OOP:

```python
# Gymnasium (today)                         # Playground
env = gym.make("HalfCheetah-v5")            env = registry.load("CheetahRun")
obs, info = env.reset()                     state = env.reset(rng)        # → State dataclass
obs, rew, term, trunc, info = env.step(a)   state = env.step(state, a)    # → new State
```

Key differences: functional (state passed explicitly), `jax.vmap` for batching (not `VectorEnv`), `jax.jit` for GPU compilation, single `done` flag (no term/trunc split), `observation_size`/`action_size` ints (no `gym.spaces`).

### Environment Catalog

**DM Control Suite (25 envs)** — standard RL benchmarks, but dm_control versions (different obs/reward/termination from Gymnasium MuJoCo):

| Playground | Nearest Gymnasium | Notes |
|-----------|-------------------|-------|
| `CheetahRun` | `HalfCheetah-v5` | Tolerance reward (target speed=10) |
| `HopperHop` / `HopperStand` | `Hopper-v5` | Different reward |
| `WalkerWalk` / `WalkerRun` | `Walker2d-v5` | dm_control version |
| `HumanoidWalk` / `HumanoidRun` | `Humanoid-v5` | CMU humanoid |
| `CartpoleSwingup` | `CartPole-v1` | Swing-up (harder) |
| `ReacherEasy/Hard`, `FingerSpin/Turn*`, `FishSwim`, `PendulumSwingup`, `SwimmerSwimmer6` | — | Various |

No Ant equivalent. Results NOT comparable across env suites.

**Locomotion (19 envs)** — real robots (Unitree Go1/G1/H1, Spot, etc.) with joystick control, gait tracking, recovery.

**Manipulation (10 envs)** — Aloha bimanual, Franka Panda, LEAP hand dexterity.

### Performance

Single-env MJX is ~10x slower than CPU MuJoCo. The win comes from massive parallelism:

| Hardware | Batch Size | Humanoid steps/sec |
|----------|-----------|-------------------|
| M3 Max (CPU) | ~128 | 650K |
| A100 (MJX) | 8,192 | 950K |

Training throughput on single A100: ~720K steps/sec (Cartpole PPO), ~91K steps/sec (Humanoid PPO). SAC 25-50x slower than PPO (off-policy overhead).

**Wall clock (1M frames)**: CPU ~80 min → GPU <5 min (PPO), ~30 min (SAC).

### Integration Design

Adapter at the env boundary. Algorithms unchanged.

```
Spec: env.backend = "playground", env.name = "CheetahRun", env.num_envs = 4096
  ↓
make_env() routes on backend
  ↓
PlaygroundVecEnv(VectorEnv)  ← jit+vmap internally, DLPack zero-copy at boundary
  ↓
VectorClockWrapper → Session.run_rl() (existing, unchanged)
```

Reference implementations: Playground's [`wrapper_torch.py`](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/wrapper_torch.py) (`RSLRLBraxWrapper`), [skrl](https://skrl.readthedocs.io/en/develop/api/envs/wrapping.html) Gymnasium-like wrapper.

### Changes

- `slm_lab/env/playground.py`: **New** — `PlaygroundVecEnv(VectorEnv)` adapter (JIT, vmap, DLPack, auto-reset, RNG management)
- `slm_lab/env/__init__.py`: `backend` routing in `make_env()`
- `pyproject.toml`: Optional `[playground]` dependency group (`mujoco-playground`, `jax[cuda12]`, `mujoco-mjx`, `brax`)
- Specs: New specs with `backend: playground`, Playground env names, `num_envs: 4096`

No changes to: algorithms, networks, memory, training loop, experiment control.

### Gotchas

1. **JIT startup**: First `reset()`/`step()` triggers XLA compilation (10-60s). One-time.
2. **Static shapes**: `num_envs` fixed at construction. Contacts padded to max possible.
3. **Ampere precision**: RTX 30/40 need `JAX_DEFAULT_MATMUL_PRECISION=highest` or training destabilizes.
4. **No Atari**: Playground is physics-only. Atari stays on CPU Gymnasium.

### Verify

PPO on CheetahRun — same reward as CPU baseline, 100x+ faster wall clock (4096 envs, A100).

### Migration Path

1. **Phase 1** (this step): Adapter + DM Control locomotion (CheetahRun, HopperHop, WalkerWalk, HumanoidWalk/Run)
2. **Phase 2**: Robotics envs (Unitree Go1/G1, Spot, Franka Panda, LEAP hand)
3. **Phase 3**: Isaac Lab (same adapter pattern, PhysX backend, sim-to-real)

---

## TODO: Step 4 — Combine + Full Benchmark Suite

**Goal**: Run PPO v2 and CrossQ+norm on MuJoCo envs. Record wall-clock and final reward (mean ± std, 4 seeds). This is the "before/after" comparison for the roadmap.

**Runs to dispatch** (via dstack, see `docs/BENCHMARKS.md`):

| Algorithm | Env | Spec | Frames |
|-----------|-----|------|--------|
| PPO v2 | HalfCheetah-v5 | `ppo_mujoco_v2_arc.yaml` | 1M |
| PPO v2 | Humanoid-v5 | `ppo_mujoco_v2_arc.yaml` | 2M |
| SAC v2 | HalfCheetah-v5 | `sac_mujoco_v2_arc.yaml` | 1M |
| SAC v2 | Humanoid-v5 | `sac_mujoco_v2_arc.yaml` | 2M |
| CrossQ | HalfCheetah-v5 | `crossq_mujoco_arc.yaml` | 1M |
| CrossQ | Humanoid-v5 | `crossq_mujoco_arc.yaml` | 2M |
| CrossQ | Hopper-v5 | `crossq_mujoco_arc.yaml` | 1M |
| CrossQ | Ant-v5 | `crossq_mujoco_arc.yaml` | 2M |

**Verify**: Both algorithms beat their v1/SAC baselines on at least 2/3 envs.

**Local testing results (200k frames, 4 sessions)**:
- PPO v2 (layer norm + percentile) beats baseline on Humanoid (272.67 vs 246.83, consistency 0.78 vs 0.70)
- Layer norm is the most reliable individual feature — helps on LunarLander (+56%) and Humanoid (+8%)
- CrossQ beats SAC on CartPole (383 vs 238), Humanoid (365 vs 356), with higher consistency
- CrossQ unstable on Ant (loss divergence) — may need tuning for high-dimensional action spaces

---

## Completed: Step 2 — Normalization Stack

**v2 = layer_norm + percentile normalization** (symlog dropped — harms model-free RL).

Changes:
- `net_util.py` / `mlp.py`: `layer_norm` and `batch_norm` params in `build_fc_model()` / `MLPNet`
- `actor_critic.py`: `PercentileNormalizer` (EMA-tracked 5th/95th percentile advantage normalization)
- `math_util.py`: `symlog` / `symexp` (retained but excluded from v2 defaults)
- `ppo.py` / `sac.py`: symlog + percentile normalization integration
- Specs: `ppo_mujoco_v2_arc.yaml`, `sac_mujoco_v2_arc.yaml`

## Completed: Step 3 — CrossQ

CrossQ: SAC variant with no target networks. Uses cross-batch normalization on concatenated (s,a) and (s',a') batches.

Changes:
- `crossq.py`: CrossQ algorithm inheriting from SAC
- `algorithm/__init__.py`: CrossQ import
- Spec: `crossq_mujoco_arc.yaml`

---

## TODO: Step 5 — High-UTD SAC / RLPD

**Goal**: `utd_ratio` alias for `training_iter`, demo buffer via `ReplayWithDemos` subclass.

Changes:
- `sac.py`: `utd_ratio` alias for `training_iter`
- `replay.py`: `ReplayWithDemos` subclass (50/50 symmetric sampling from demo and online data)
- Spec: `sac_mujoco_highutd_arc.yaml` (UTD=20 + layer norm critic)

**Verify**: High-UTD SAC on Hopper-v5 — converge in ~50% fewer env steps vs standard SAC.

## TODO: Step 6 — Pretrained Vision Encoders

**Goal**: DINOv2 encoder via torcharc, DrQ augmentation wrapper.

Changes:
- `pretrained.py`: `PretrainedEncoder` module (DINOv2, freeze/fine-tune, projection)
- `wrappers.py`: `RandomShiftWrapper` (DrQ-v2 ±4px shift augmentation)
- Spec: `ppo_vision_arc.yaml`

**Verify**: PPO with DINOv2 on DMControl pixel tasks (Walker Walk, Cartpole Swingup). Frozen vs fine-tuned comparison.

---

## Environment Plan (Future)

Three tiers of environment coverage:

| Tier | Platform | Purpose |
|------|----------|---------|
| **Broad/Basic** | [Gymnasium](https://gymnasium.farama.org/) | Standard RL benchmarks (CartPole, MuJoCo, Atari) |
| **Physics-rich** | [DeepMind MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) | GPU-accelerated locomotion, manipulation, dexterous tasks |
| **Sim-to-real** | [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) | GPU-accelerated, sim-to-real transfer, robot learning |

---

## Key Findings

- **Symlog harms model-free RL**: Tested across 6 envs — consistently hurts PPO and SAC. Designed for DreamerV3's world model, not direct value/Q-target compression.
- **Layer norm is the most reliable feature**: Helps on harder envs (LunarLander +56%, Humanoid +8%), neutral on simple envs.
- **CrossQ unstable on some envs**: Loss divergence on Ant and Hopper. Stable on CartPole, Humanoid. May need batch norm tuning for high-dimensional action spaces.
- **Features help more on harder envs**: Simple envs (CartPole, Acrobot) — baseline wins. Complex envs (Humanoid) — v2 and CrossQ win.

## Not in Scope

- World models (DreamerV3/RSSM) — Dasein Agent Phase 3.2c
- Plasticity loss mitigation (CReLU, periodic resets) — future work
- PPO+ principled fixes (ICLR 2025) — evaluate after base normalization stack
