# SLM-Lab

1. The project lead owns the roadmap, specifications and work list. The founder owns purpose, money and external commitments.
2. The installed global instructions and `lead-role` skill apply. Read the existing project documents for current scope and status; check dated context before treating it as current.
3. Work within accepted scope and budget. Preserve source, unmerged work, project data and history; do not spend, publish or make external commitments without the founder's authorization.
4. Define a useful outcome and proportionate checks. Independently review changed risk for correctness and failure modes; one reviewer independent of the builder and its descendants may cover both. Run the complete applicable bar before landing.
5. Keep project-specific technical guidance and evidence in the existing documents. Resolve current capabilities and paths when used.

## Style Guide

**General Principles:**

1. **Naming**: Short, obvious, globally consistent. No magic numbers — name your constants.
2. **Single Responsibility**: One function/class, one purpose. Max 3-4 nesting levels.
3. **Separation of Concerns**: Logic, data, presentation separate
4. **Fail Fast**: Validate early, explicit errors. Never commit secrets, credentials, or .env files.

**Python:**

1. **Type Hints**: Native types (`list[str]`, `str | None`) - no `typing` module
2. **Docstrings**: Concise - rely on naming and type hints
3. **Error Handling**: Specific exceptions, no bare `except:`
4. **Imports**: Top-level only, no in-method imports
5. **Project Structure**: Folders are modules - no sys-path hacks

**TypeScript:**

1. **Type Safety**: Strict mode, avoid `any`, use `unknown`
2. **Async/Await**: Over `.then()` chains
3. **Components**: Small, focused, extract logic to hooks

## SLM-Lab: Deep RL Framework

**For Users**: See `README.md` for installation, basic usage, and getting started.

**For Agents**: This document covers development workflows - understanding the architecture, running tests, and executing benchmarks.

### Project Overview

Modular deep reinforcement learning framework in PyTorch for RL research and experimentation. Supports multiple algorithms (DQN, PPO, SAC, etc.), environments (Gymnasium, Atari, MuJoCo), and distributed training with hyperparameter search.

**Key capabilities:**
- Reproducible experiments via JSON specs
- Modular algorithm/network/memory components
- ASHA hyperparameter search with early termination
- Cloud GPU training (optional - use dstack or your own infrastructure)
- Benchmark tracking with automated metrics extraction

## Framework Architecture

Understanding SLM-Lab's modular design is essential for development work.

### Core Components

1. **Agent** (`slm_lab/agent/`) - RL algorithm implementations
   - `algorithm/`: DQN, PPO, SAC, A2C, REINFORCE variants
   - Each algorithm: `__init__`, `act()`, `update()`, `sample()`

2. **Network** (`slm_lab/agent/net/`) - Neural network architectures
   - `mlp.py`: Fully-connected networks
   - `conv.py`: Convolutional networks (Atari)
   - `recurrent.py`: RNN/LSTM networks

3. **Memory** (`slm_lab/agent/memory/`) - Experience storage
   - `replay.py`: Experience replay buffer
   - `prioritized.py`: Prioritized experience replay

4. **Environment** (`slm_lab/env/`) - Gym wrappers and vectorization
   - `vec_env.py`: Vectorized environments (parallel rollouts)
   - `wrapper.py`: Atari preprocessing, normalization

5. **Experiment** (`slm_lab/experiment/`) - Training loop and search
   - `control.py`: Session/trial management
   - `search.py`: ASHA hyperparameter search

6. **Spec System** (`slm_lab/spec/`) - JSON configuration for reproducibility
   - Structure: `meta`, `agent`, `env`, `body`, `search`
   - Variable substitution: `${var}` with `-s var=value`

### Key Patterns

- **Modularity**: Swap algorithms/networks/memories via spec changes
- **Vectorization**: Parallel env rollouts for sample efficiency
- **Spec-driven**: All experiments defined in JSON - no code changes needed
- **Checkpointing**: Auto-save at intervals, resume from checkpoints

## Development Setup

### Local Testing & Bug Fixes

For reproducing issues or testing changes locally:

```bash
# Install with full dependencies
uv sync

# Quick test run (CartPole - 30 seconds)
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train

# Test with rendering (visual verification)
uv run slm-lab --render slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole dev

# Run tests
uv run pytest

# Format code
uv run ruff format
```

**Quick test specs** (for verification):
- `ppo_cartpole.json` - PPO on CartPole (fastest)
- `ppo_lunar.json` - PPO on LunarLander

### Remote and cloud work

Never hold credentials. Remote dispatch, cloud training, paid capacity and result uploads require their applicable authorization. Do not infer authorization from historical examples. Consult `.dstack/*.yml` and the existing project documentation when the founder approves a specific run.

## Benchmarking Workflow

**`docs/BENCHMARKS.md`** is the single source of truth. See the `/benchmark` skill for operational details (commands, data lifecycle, graduation).

**Per-run intake for an authorized run:** Follow applicable access and publication boundaries; do not upload or expose data without authorization.
1. Extract score (`dstack logs NAME | grep trial_metrics`)
2. Find HF folder name (query HuggingFace API)
3. Update table with score AND HF link
4. Pull HF data locally (`hf download`)
5. Generate plot (`uv run slm-lab plot`)
6. Commit score + link + plot together

A table row is NOT complete until it has: **score, HF link, and plot**. See `/benchmark` skill for commands.

Run and monitor benchmarks only within the specific authorized scope and budget.

### Hyperparameter Search

ASHA search for when algorithms fail to reach target. Budget: ~3-4 trials per dimension.

```json
{
  "search": {
    "agent.algorithm.gamma__uniform": [0.993, 0.999],
    "agent.net.optim_spec.lr__loguniform": [1e-4, 1e-3]
  }
}
```

Prefer continuous distributions (`__uniform`, `__loguniform`) over `__choice`. Search high-impact params first (lr, gamma, lam). After search: update spec defaults, run `train`, use that result.

---

## SLM-Lab Documentation

- **Changelog**: Document major changes in `docs/CHANGELOG.md`
- **Benchmarks**: `docs/BENCHMARKS.md` — results tables, targets, reproducibility
- **Specs**: Document rationale in commit messages when updating specs
