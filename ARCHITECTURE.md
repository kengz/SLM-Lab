# SLM-Lab Architecture

Modular deep reinforcement learning framework in PyTorch. Spec-driven design: JSON specs fully define experiments with no code changes needed.

## Directory Structure

```
slm_lab/
  agent/                    # Agent: algorithm + network + memory
    __init__.py             # Agent class, MetricsTracker
    algorithm/              # RL algorithm implementations
      base.py               # Algorithm base class (act, update, sample)
      reinforce.py          # REINFORCE
      sarsa.py              # SARSA
      dqn.py                # DQN, DDQN
      actor_critic.py       # A2C
      ppo.py                # PPO
      sac.py                # SAC (continuous + discrete)
      crossq.py             # CrossQ (SAC without target networks)
      policy_util.py        # Action selection, exploration, distributions
    net/                    # Neural network architectures
      base.py               # Net base class
      mlp.py                # MLPNet — fully connected
      conv.py               # ConvNet — convolutional (Atari)
      recurrent.py          # RecurrentNet — LSTM
      torcharc_net.py       # TorchArc YAML-defined networks
      net_util.py           # Weight init, polyak update, gradient clipping
      batch_renorm.py       # Batch Renormalization (for CrossQ critics)
      weight_norm.py        # WeightNormLinear
    memory/                 # Experience storage
      base.py               # Memory base class
      replay.py             # Replay buffer (uniform sampling)
      prioritized.py        # Prioritized Experience Replay (SumTree)
      onpolicy.py           # OnPolicyBatchReplay (PPO, A2C)
  env/                      # Environment backends
    __init__.py             # make_env() — routing, wrappers, space detection
    wrappers.py             # ClockWrapper, Atari preprocessing, obs normalization
    playground.py           # PlaygroundVecEnv — JAX/MJX GPU-accelerated backend
  experiment/               # Training orchestration
    control.py              # Session, Trial, Experiment classes
    search.py               # ASHA hyperparameter search (Ray Tune)
    analysis.py             # Metrics analysis, plotting, trial aggregation
  lib/                      # Utilities
    util.py                 # General utilities (set_attr, random seed, CUDA)
    math_util.py            # Math helpers (discount, GAE, explained variance)
    ml_util.py              # ML helpers (to_torch_batch, SumTree)
    logger.py               # Loguru-based logging
    viz.py                  # Plotting (matplotlib)
    hf.py                   # HuggingFace upload/download
    perf.py                 # Performance optimizations (pinned memory, etc.)
    optimizer.py            # Custom optimizer utilities
    distribution.py         # Custom probability distributions
    env_var.py              # Environment variables (lab_mode, render)
    decorator.py            # @lab_api decorator
    profiler.py             # Performance profiler
    torch_profiler.py       # PyTorch profiler integration
  spec/                     # Experiment specifications
    spec_util.py            # Spec parsing, variable substitution, validation
    benchmark/              # Validated benchmark specs
      ppo/                  # PPO specs (classic, box2d, mujoco, atari)
      sac/                  # SAC specs
      crossq/               # CrossQ specs
      dqn/                  # DQN/DDQN specs
      a2c/                  # A2C specs
      playground/           # MuJoCo Playground specs (dm_control, locomotion, manipulation)
  cli/                      # CLI entry point
    __init__.py             # Typer CLI — run, run-remote, pull, list, plot
```

## Core Components

### Agent (`slm_lab/agent/__init__.py`)

The top-level RL agent. Holds references to:
- **Algorithm**: policy logic (act, update)
- **Memory**: experience storage
- **MetricsTracker**: training/eval statistics, checkpointing
- **Network**: via algorithm (algorithm owns the networks)

```python
agent = Agent(spec, mt=MetricsTracker(env, spec))
action = agent.act(state)       # Forward pass → action
agent.update(state, action, reward, next_state, done, terminated, truncated)
agent.save() / agent.load()     # Checkpoint management
```

### Algorithm (`slm_lab/agent/algorithm/`)

Each algorithm implements the core RL loop methods:

| Method | Purpose |
|--------|---------|
| `act(state)` | Select action given observation (with exploration) |
| `sample()` | Sample batch from memory |
| `update(...)` | Store transition, optionally run gradient updates |
| `calc_pdparam(state)` | Forward pass to get policy distribution parameters |
| `train()` | Execute gradient steps on sampled batch |

**Key parameters:**
- `training_frequency`: env steps between gradient updates (default: 1)
- `training_iter`: gradient steps per update (controls UTD ratio)
- `training_start_step`: random exploration before learning begins

### Network (`slm_lab/agent/net/`)

Neural networks used by algorithms. Three built-in architectures plus TorchArc YAML:

| Network | Use Case |
|---------|----------|
| `MLPNet` | Continuous control (MuJoCo, classic) |
| `ConvNet` | Image observations (Atari) |
| `RecurrentNet` | Sequential/partial observability |
| `TorchArcNet` | YAML-defined arbitrary architectures |

Networks handle: forward pass, loss computation, optimization, target network updates (polyak).

### Memory (`slm_lab/agent/memory/`)

| Memory | Algorithm | Behavior |
|--------|-----------|----------|
| `OnPolicyBatchReplay` | PPO, A2C | Stores one rollout, cleared after update |
| `Replay` | SAC, CrossQ, DQN | Circular buffer, uniform random sampling |
| `PrioritizedReplay` | DDQN+PER | SumTree priority sampling |

All memories store: `states, actions, rewards, next_states, dones, terminateds`.

## Training Loop

### Control Hierarchy

```
Experiment (search over hyperparameters)
  └── Trial (one hyperparameter configuration)
        └── Session (one training run with one seed)
```

### Session Loop (`control.py`)

```python
class Session:
    def run_rl(self):
        state, info = env.reset()
        while env.get() < env.max_frame:        # ClockWrapper tracks frames
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            agent.update(state, action, reward, next_state, done, terminated, truncated)
            self.try_ckpt(agent, env)            # Log/eval at intervals
            state = next_state                   # VecEnv auto-resets
```

**Multi-session**: `Trial.run_sessions()` runs `max_session` sessions. If `max_session > 1`, sessions run in parallel via `torch.multiprocessing`. Session 0 produces plots; all sessions contribute to trial-level statistics.

**Distributed**: `Trial.run_distributed_sessions()` shares global network parameters across sessions for A3C-style training.

### Checkpointing

At `log_frequency` intervals: save metrics, generate plots, save model checkpoint.
At `eval_frequency` intervals: run evaluation episodes (if `rigorous_eval` enabled).
Best model saved when `total_reward_ma` exceeds previous best.

## Environment Layer

### `make_env()` Routing

`slm_lab/env/__init__.py` routes by env name prefix:

```
env.name = "playground/CheetahRun"  → PlaygroundVecEnv (JAX/MJX)
env.name = "ALE/Pong-v5"           → Gymnasium + AtariVectorEnv
env.name = "Hopper-v5"             → Gymnasium + SyncVectorEnv/AsyncVectorEnv
env.name = "CartPole-v1"           → Gymnasium + SyncVectorEnv
```

### Wrapper Stack

All environments go through a common wrapper pipeline:

1. **Base env**: `gymnasium.make()` or `PlaygroundVecEnv`
2. **Action rescaling**: `RescaleAction` to [-1, 1] if needed
3. **Episode stats**: `RecordEpisodeStatistics` (+ `FullGameStatistics` for Atari life tracking)
4. **Normalization**: `NormalizeObservation`, `ClipObservation` (if `normalize_obs: true`)
5. **Reward processing**: `NormalizeReward`, `ClipReward` (Atari always clips to [-1, 1])
6. **Clock**: `ClockWrapper` wraps everything — tracks total frames for training loop termination

### Gymnasium Backend (default)

Standard path for Classic Control, Box2D, MuJoCo, Atari. Vectorization mode selected automatically:
- Classic Control, Box2D, or `num_envs < 8`: `SyncVectorEnv`
- ALE/Atari: `AtariVectorEnv` (native C++ vectorization, fastest) or sync for rendering
- Complex envs with `num_envs >= 8`: `AsyncVectorEnv`

### MuJoCo Playground Backend (`playground/`) — MJWarp Architecture

GPU-accelerated JAX environments from DeepMind. 54 environments across 3 categories:
- **DM Control Suite** (25): CheetahRun, HopperHop, WalkerWalk, HumanoidRun, CartpoleBalance, ...
- **Locomotion** (19): Go1JoystickFlatTerrain, SpotGetup, H1JoystickGaitTracking, ...
- **Manipulation** (10): PandaPickCube, AlohaHandOver, LeapCubeReorient, ...

#### MJWarp Backend

All playground environments use MJWarp (`impl='warp'`), hardcoded via `_config_overrides = {"impl": "warp"}` in `playground.py`. MJWarp uses NVIDIA Warp CUDA kernels for physics simulation, dispatched through JAX's XLA FFI (Foreign Function Interface).

**Critical: JAX is still required with MJWarp.** Warp-lang does NOT bypass JAX. JAX provides the tracing, compilation, and batching (`jax.vmap`) infrastructure; Warp provides the CUDA physics kernels called via XLA custom calls.

#### Installation

Playground dependencies are installed via `uv sync --group playground`, which pulls:
- `mujoco-playground` (environment definitions)
- `jax[cuda12]` (GPU dispatch layer)
- `warp-lang` (CUDA physics kernels)
- `brax` (wrapper utilities)

Configured in `pyproject.toml` as `playground[cuda] ; sys_platform != 'darwin'` — this installs `jax[cuda12]` + `warp-lang` together via the NVIDIA PyPI index. Do NOT manually `pip install jax[cuda12]` separately. On macOS, only CPU/numpy paths are available (no CUDA).

#### PlaygroundVecEnv Pipeline

`PlaygroundVecEnv` (`slm_lab/env/playground.py`) wraps the Playground API as `gymnasium.vector.VectorEnv`:

1. **Load**: `pg_registry.load(env_name, config_overrides={"impl": "warp"})` returns `MjxEnv`
2. **Wrap**: `wrap_for_brax_training(env)` applies three layers:
   - `VmapWrapper` — `jax.vmap` for batched parallel simulation across `num_envs`
   - `EpisodeWrapper` — step counting, sets `state.info["truncation"]` on time limit
   - `BraxAutoResetWrapper` — automatic reset on episode termination
3. **JIT**: `jax.jit(env.reset)` and `jax.jit(env.step)` compiled once at init
4. **State**: Brax `State` dataclass with `.obs`, `.reward`, `.done`, `.info["truncation"]`, `.metrics`

#### JAX-to-PyTorch Data Transfer

The `_to_output()` method handles the JAX→PyTorch boundary:

- **GPU path** (`device='cuda'`): DLPack zero-copy transfer via `torch.from_dlpack(jax_array)`. Both JAX and PyTorch share the same GPU memory — no data copy.
- **CPU path** (`device=None`): `np.asarray(jax_array)` materialization. Used on macOS or when no GPU is available.
- **Rewards/dones**: Always numpy (used for Python control flow and memory storage).

`XLA_PYTHON_CLIENT_PREALLOCATE=false` must be set when sharing GPU with PyTorch, preventing JAX from pre-allocating all GPU memory. Set automatically in `_make_playground_env()`.

#### Device Detection

Auto-detection in `make_env()`: `torch.cuda.is_available()` → `device='cuda'` (DLPack) or `None` (numpy). No manual device configuration needed.

#### Truncation Handling

Brax `EpisodeWrapper` sets `state.info["truncation"]` (1.0 = time limit, 0.0 = not truncated) as a dict entry, NOT a direct attribute. Accessed via `state.info.get("truncation")`. This distinguishes terminal states (agent failure) from truncation (time limit), which is critical for correct value bootstrapping.

#### Dict Observations

Some environments (locomotion, manipulation) return dict observations. `PlaygroundVecEnv._get_obs()` flattens these by sorting keys alphabetically and concatenating values along the last axis via `jnp.concatenate`.

#### GPU Performance

Confirmed on NVIDIA A5000: ~1737 fps during rollout, ~450 fps during training with gradient steps (PPO, 64 envs, CartpoleBalance).

#### dstack Cloud Configuration

`.dstack/run-gpu-train.yml` always installs playground dependencies and pre-clones mujoco_menagerie:

```yaml
commands:
  - uv sync --group playground
  - uv run python -c "from mujoco_playground._src.mjx_env import ensure_menagerie_exists; ensure_menagerie_exists()"
  - uv run slm-lab run ...
```

The `ensure_menagerie_exists()` call before training fixes a race condition where multiple sessions would simultaneously clone the menagerie repository. Without this pre-clone, only session 0 would succeed.

#### Wrapper Stack (Playground Path)

The playground wrapper pipeline in `_make_playground_env()`:

1. `PlaygroundVecEnv` — JAX/MJWarp batched simulation
2. `VectorRescaleAction` — rescale to [-1, 1] if needed
3. `VectorRecordEpisodeStatistics` — episode return/length tracking
4. `PlaygroundRenderWrapper` — MuJoCo rendering (dev mode only)
5. GPU mode: `TorchNormalizeObservation` (if `normalize_obs: true`)
6. CPU mode: `VectorNormalizeObservation`, `VectorClipObservation`, `VectorNormalizeReward`, `VectorClipReward`
7. `VectorClockWrapper` — frame counting for training loop termination

#### Key Files

| File | Purpose |
|------|---------|
| `slm_lab/env/playground.py` | `PlaygroundVecEnv` — JAX/MJWarp vectorized env |
| `slm_lab/env/__init__.py` | `_make_playground_env()` — routing and wrapper stack |
| `.dstack/run-gpu-train.yml` | Cloud GPU config with playground setup |
| `slm_lab/spec/benchmark_arc/` | Playground benchmark specs (PPO, SAC, CrossQ) |

### Future: Isaac Lab (planned)

NVIDIA GPU-accelerated environments via Isaac Sim. Separate optional install. Uses `ManagerBasedRLEnv` with gymnasium-compatible API. Would use a similar `isaac/` prefix routing pattern.

## Spec System

JSON specs fully define experiments — algorithm, network, memory, environment, and meta settings. No code changes needed to run different configurations.

```json
{
  "spec_name": {
    "agent": {
      "name": "SoftActorCritic",
      "algorithm": { "name": "SoftActorCritic", "gamma": 0.99, "training_iter": 4 },
      "memory": { "name": "Replay", "batch_size": 256, "max_size": 1000000 },
      "net": { "type": "MLPNet", "hid_layers": [256, 256], "optim_spec": { "lr": 3e-4 } }
    },
    "env": { "name": "playground/CheetahRun", "num_envs": 16, "max_frame": 2000000 },
    "meta": { "max_session": 4, "max_trial": 1, "log_frequency": 10000 }
  }
}
```

**Variable substitution**: `${var}` placeholders in specs, set via CLI `-s var=value`. Enables template specs for running the same config across multiple environments.

**Spec resolution** (`spec_util.py`):
1. Load JSON, select spec by name
2. Substitute `${var}` with CLI values (fail-fast on unresolved)
3. `make_env(spec)` creates environment from `spec["env"]`
4. `Agent(spec)` creates agent, algorithm, memory, network from `spec["agent"]`

**Search specs**: Add `"search"` key with parameter distributions for ASHA hyperparameter search:
```json
{
  "search": {
    "agent.algorithm.gamma__uniform": [0.993, 0.999],
    "agent.net.optim_spec.lr__loguniform": [1e-4, 1e-3]
  }
}
```
