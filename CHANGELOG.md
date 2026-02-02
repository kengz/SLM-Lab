# SLM-Lab v5.0.0

Modernization release for the current RL ecosystem. Updates SLM-Lab from OpenAI Gym to Gymnasium, adds correct handling of episode termination (the `terminated`/`truncated` fix), and migrates to modern Python tooling.

**TL;DR:** Install with `uv sync`, run with `slm-lab run`. Specs are simpler (no more `body` section or array wrappers). Environment names changed (`CartPole-v1`, `ALE/Pong-v5`, `Hopper-v5`). Code structure preserved for book readers.

> **Book readers:** For exact code from *Foundations of Deep Reinforcement Learning*, use `git checkout v4.1.1`

---

## Why This Release

SLM-Lab was created as an educational framework for deep reinforcement learning, accompanying *Foundations of Deep Reinforcement Learning*. The code prioritizes clarity and correctness—it should help you understand RL algorithms, not just run them.

Since v4, the RL ecosystem changed significantly:

- **OpenAI Gym is deprecated.** The Farama Foundation forked it as [Gymnasium](https://gymnasium.farama.org/), now the standard. Gym's `done` flag conflated two concepts: true termination (agent failed/succeeded) and time-limit truncation. Gymnasium fixes this with separate `terminated` and `truncated` signals—important for correct value estimation (see [below](#the-gymnasium-api-change)).

- **Roboschool is abandoned.** MuJoCo became free in 2022, so roboschool is no longer maintained. Gymnasium includes native MuJoCo bindings.

- **Python tooling modernized.** `conda` + `setup.py` → `uv` + `pyproject.toml`. Python 3.12+, PyTorch 2.8+. [uv](https://docs.astral.sh/uv/) emerged as a fast, reliable Python package manager—no more conda environment headaches.

- **Old dependencies don't build anymore.** The v4 dependency stack (old PyTorch, atari-py, mujoco-py, etc.) won't compile on modern hardware, especially ARM machines (Apple Silicon, AWS Graviton). Many deprecated packages simply don't run. A full rebuild was necessary.

This release updates SLM-Lab to work with modern dependencies while preserving the educational code structure. If you've read the book, the code should still be recognizable.

### Critical: Atari v5 Sticky Actions

**SLM-Lab uses Gymnasium ALE v5 defaults.** v5 default `repeat_action_probability=0.25` (sticky actions) randomly repeats agent actions to simulate console stochasticity, making evaluation harder but more realistic than v4 default 0.0 used by most benchmarks (CleanRL, SB3, RL Zoo). This follows [Machado et al. (2018)](https://arxiv.org/abs/1709.06009) research best practices. See [ALE version history](https://ale.farama.org/environments/#version-history-and-naming-schemes).

### Summary

| v4 | v5 |
|----|----|
| `conda activate lab && python run_lab.py` | `slm-lab run` |
| `CartPole-v0`, `PongNoFrameskip-v4` | `CartPole-v1`, `ALE/Pong-v5` |
| `RoboschoolHopper-v1` | `Hopper-v5` |
| `agent: [{...}]`, `env: [{...}]`, `body: {...}` | `agent: {...}`, `env: {...}` |
| `body.state_dim`, `body.memory` | `agent.state_dim`, `agent.memory` |

---

## Migration from v4

### 1. Install

```bash
uv sync
uv tool install --editable .
```

### 2. Update specs

Remove array brackets and `body` section:

```diff
 {
-  "agent": [{ "name": "PPO", ... }],
-  "env": [{ "name": "CartPole-v0", ... }],
-  "body": { "product": "outer", "num": 1 },
+  "agent": { "name": "PPO", ... },
+  "env": { "name": "CartPole-v1", ... },
   "meta": { ... }
 }
```

### 3. Update environment names

- Classic control: `v0`/`v1` → current version (`CartPole-v1`, `Pendulum-v1`, `LunarLander-v3`)
- Atari: `PongNoFrameskip-v4` → `ALE/Pong-v5`
- Roboschool → MuJoCo: see [Deprecations](#roboschool) for full mapping

### 4. Run

```bash
slm-lab run spec.json spec_name train
```

See `slm_lab/spec/benchmark/` for updated reference specs.

---

## The Gymnasium API Change

This matters for understanding the code, not just running it.

### The Problem

Gym's `done` flag was ambiguous—it meant "episode ended" but episodes end for two different reasons:

1. **Terminated:** True end state (CartPole fell, agent died, goal reached)
2. **Truncated:** Time limit hit (MuJoCo's 1000-step cap)

For value estimation, these need different treatment. Terminated means future returns are zero. Truncated means future returns exist but weren't observed—you should bootstrap from V(s').

### The Fix

Gymnasium separates the signals:

```python
# Gym
obs, reward, done, info = env.step(action)

# Gymnasium
obs, reward, terminated, truncated, info = env.step(action)
```

All SLM-Lab algorithms now use `terminated` for bootstrapping decisions:

```python
# Only zero out future returns on TRUE termination
q_targets = rewards + gamma * (1 - terminateds) * next_q_preds
```

This is why the code stores `terminateds` and `truncateds` separately in memory—algorithms need `terminated` for correct bootstrapping, `done` for episode boundaries.

This fix particularly matters for time-limited environments like MuJoCo (1000-step limit) where episodes frequently truncate during training. Using `done` instead of `terminated` there significantly hurts learning.

---

## Code Structure Changes

For book readers who want to trace through the code:

### Simplified Agent Design

The `Body` class was removed. Its responsibilities moved to more natural locations:

```python
# v4
state_dim = agent.body.state_dim
memory = agent.body.memory
env = agent.body.env

# v5
state_dim = agent.state_dim
memory = agent.memory
env = agent.env
```

Training metrics tracking is now in `MetricsTracker` (what `Body` was renamed to).

### Simplified Specs

Multi-agent configurations were rarely used. Specs are now flat:

```python
# v4: agent_spec = spec['agent'][0]
# v5: agent_spec = spec['agent']
```

### Architecture Preserved

The core design is unchanged:

```
Session → Agent → Algorithm → Network
              ↘ Memory
        → Env
```

---

## Algorithm Updates

**PPO:** New options for value target handling—`normalize_v_targets`, `symlog_transform` (from DreamerV3), `clip_vloss` (CleanRL-style).

**SAC:** Discrete action support uses exact expectation (Christodoulou 2019). Target entropy auto-calculated.

**Networks:** Optional `layer_norm` for MLP hidden layers. Custom optimizers (Lookahead, RAdam) removed—use native PyTorch `AdamW`.

All algorithms use `terminated` (not `done`) for correct bootstrapping.

---

## Benchmarks

All algorithms validated on Gymnasium. Full results in `docs/BENCHMARKS.md`.

| Category | REINFORCE | SARSA | DQN | DDQN+PER | A2C | PPO | SAC |
|----------|-----------|-------|-----|----------|-----|-----|-----|
| Classic Control | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Box2D | — | — | ✅ | ✅ | ⚠️ | ✅ | ✅ |
| MuJoCo (11 envs) | — | — | — | — | ⚠️ | ✅ All | ✅ All |
| Atari (54 games) | — | — | — | — | ✅ | ✅ | — |

**Atari benchmarks** use ALE v5 with sticky actions (`repeat_action_probability=0.25`). PPO tested with lambda variants (0.95, 0.85, 0.70) to optimize per-game performance. A2C uses GAE with lambda 0.95.

**Note on scores:** Gymnasium environment versions differ from old Gym—some are harder (CartPole-v1 has stricter termination than v0), some have different reward scales (MuJoCo v5 vs roboschool). Targets reference [CleanRL](https://docs.cleanrl.dev/) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) gymnasium benchmarks.

---

## New Features

**Hyperparameter search** now uses Ray Tune + Optuna + ASHA early stopping:

```bash
slm-lab run spec.json spec_name search    # Run search locally
```

Add `search_scheduler` to spec for ASHA early termination of poor trials. See `docs/BENCHMARKS.md` for search methodology.

---

## CLI Usage

The CLI uses [Typer](https://typer.tiangolo.com/). Use `--help` on any command for details:

```bash
slm-lab --help                           # List all commands
slm-lab run --help                       # Options for run command

# Installation
uv sync                                  # Install dependencies
uv tool install --editable .             # Install slm-lab command

# Basic usage
slm-lab run                              # PPO CartPole (default demo)
slm-lab run --render                     # With visualization
slm-lab run spec.json spec_name train    # Train from spec file
slm-lab run spec.json spec_name dev      # Dev mode (shorter run)
slm-lab run spec.json spec_name search   # Hyperparameter search

# Variable substitution (for template specs)
slm-lab run -s env=ALE/Breakout-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train

# Cloud training (dstack + HuggingFace)
slm-lab run-remote --gpu spec.json spec_name train   # Launch on cloud GPU
slm-lab list                                         # List experiments on HuggingFace
slm-lab pull spec_name                               # Download results locally

# Utilities
slm-lab run --stop-ray                   # Stop Ray processes
```

Modes: `dev` (quick test), `train` (full training), `search` (hyperparameter search), `enjoy` (evaluate saved model).

---

## Deprecations

### Multi-Agent / Multi-Environment

The v4 `body` spec section and array wrappers (`agent: [{...}]`) supported multi-agent and multi-environment configurations. These were rarely used and added complexity. v5 simplifies to single-agent single-env, which covers the vast majority of use cases and matches how most RL research is done.

### Unity ML-Agents and VizDoom

These integrations are removed from the core package. Both ecosystems have their own gymnasium-compatible wrappers now:
- Unity: [gymnasium-unity](https://gymnasium.farama.org/environments/third_party_environments/)
- VizDoom: [vizdoom gymnasium wrapper](https://gymnasium.farama.org/environments/third_party_environments/)

You can still use these environments with SLM-Lab by installing their wrappers and specifying the environment name in your spec.

### Roboschool

Roboschool is abandoned (MuJoCo became free in 2022). Use gymnasium's native MuJoCo environments instead:
- `RoboschoolHopper-v1` → `Hopper-v5`
- `RoboschoolHalfCheetah-v1` → `HalfCheetah-v5`
- `RoboschoolWalker2d-v1` → `Walker2d-v5`
- `RoboschoolAnt-v1` → `Ant-v5`
- `RoboschoolHumanoid-v1` → `Humanoid-v5`
