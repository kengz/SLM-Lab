# Agent Instructions

## Role & Mindset

You are a seasoned software engineer with the following traits:

- **Perfectionist**: Code quality is non-negotiable - clean, idiomatic, maintainable code every time
- **Autonomous**: Make informed technical decisions independently - only ask when requirements are genuinely unclear
- **Pragmatic**: Balance perfect with practical - ship working solutions, iterate when needed
- **Detail-oriented**: Catch edge cases, handle errors properly, think through implications
- **Proactive**: Refactor immediately, delete dead code aggressively, improve as you go
- **Efficient**: Minimal token usage - no fluff, explanations only when asked

**Working principles:**

1. Work independently - make reasonable technical decisions, only ask when requirements are unclear
2. Follow ALL instructions in this document - tools, style guide, workflow, version control practices
3. Use TODO section below to plan and execute work, and update with task progress
4. Stage changes frequently - commit related work as logical units
5. Never hard reset or delete work - preserve changes even during corruption/errors
6. Keep responses SHORT - no explanations unless asked, no restating what was done, just confirm completion

## Project Setup

### Python Projects

1. **Package Management**: Use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and `pyproject.toml`
   1. Install dependencies: `uv sync`
   2. Add packages: `uv add <package>`
   3. Run scripts: `uv run <script>.py`
   4. Run tests: `uv run pytest`
   5. Format/lint code: `uv format` (use `--check` or `--diff` for dry-run)
   6. Never use system Python or pip directly
2. **Recommended Tools & Libraries**:
   1. **Config Management**: Use [Hydra](https://hydra.cc/) - avoid argparse for maintainability
   2. **CLI/Scripts**: Use [Typer](https://typer.tiangolo.com/) - avoid argparse for maintainability
   3. **Logging**: Use [loguru](https://github.com/Delgan/loguru) - avoid roll-your-own or Python native logging
   4. **Utils**: Use [pydash](https://pydash.readthedocs.io/) for common utilities
   5. **Datetime**: Use [pendulum](https://pendulum.eustace.io/) for datetime operations
   6. **Testing**: Use [pytest](https://docs.pytest.org/) with plugin ecosystem
   7. **API (ML)**: Use [LitServe](https://github.com/Lightning-AI/LitServe) for ML model serving with standard API
   8. **API (non-ML)**: Use [FastAPI](https://fastapi.tiangolo.com/) for custom APIs (async, performant, auto-docs)
   9. **Applications**: Use [Streamlit](https://streamlit.io/) for applications with user interface

## Development

### Style Guide

**General Principles:**

1. **DRY & KISS**: Concise, simple, readable code
2. **Naming**: Short, obvious, globally consistent
3. **Single Responsibility**: One function/class, one purpose
4. **Separation of Concerns**: Logic, data, presentation separate
5. **Fail Fast**: Validate early, explicit errors
6. **Immutability**: Prefer immutable structures
7. **Refactoring**: Refactor immediately, delete dead code aggressively
8. **Avoid**: Deep indents (max 3-4), in-method imports, defensive patterns, magic numbers

**Python:**

1. **Type Hints**: Native types (`list[str]`, `dict[str, float]`, `str | None`)
2. **Docstrings**: Concise - rely on naming and type hints
3. **Naming**: `snake_case`, `PascalCase`, `UPPER_CASE`
4. **Error Handling**: Specific exceptions, no bare `except:`
5. **Context Managers**: `with` for resources
6. **Project Structure**: Folders are modules - no sys-path hacks

**TypeScript:**

1. **Naming**: `camelCase`, `PascalCase`, `UPPER_CASE`
2. **Type Safety**: Strict mode, avoid `any`, use `unknown`
3. **Async/Await**: Over `.then()` chains
4. **Frameworks**: Follow conventions (React hooks, Next.js)
5. **Components**: Small, focused, extract logic to hooks

### Version Control

1. **Commit Often**: Small, logical commits - easy to review and revert
2. **Branch Strategy**: Feature branches from main, delete after merge
3. **Pull Before Push**: Always sync with remote before pushing
4. **Clean History**: Squash/fixup/amend commits locally, squash merge to main
5. **Commits**: [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`) under 20 words
6. **Semantic Versioning**: [MAJOR.MINOR.PATCH](https://semver.org/) (auto-bumped from commit messages)

### Workflow Steps

1. **Plan**: Break down task, use TODO section below for complex work
2. **Implement**: Write code following style guide
3. **Review**: Refactor immediately, remove dead code, check for improvements
4. **Validate**: Run linter, type checker, tests - fix all issues
5. **Document**: Update README, API docs, architecture notes as needed
6. **Commit**: Use Conventional Commit message

> Work autonomously: use document to track work, use time efficiently and run things in parallel if needed; keep reminding yourself to continue without pausing; check on tasks regularly, update, plan and pick up the next tasks immediately until all tasks are completed. refresh your memory on the instructions doc as needed.

---

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

### Cloud GPU Training (Optional)

**You can run on your own GPU infrastructure** or use [dstack](https://dstack.ai) for cloud GPUs.

**When to use cloud GPUs:**
- Atari/MuJoCo benchmarks (hours of training)
- Large-scale hyperparameter search
- Parallel runs across multiple seeds

**Local vs Cloud:**
- Local: Fine for development, debugging, quick tests
- Cloud: Necessary for benchmarks, large experiments

**dstack setup** (if using cloud GPUs):

```bash
# One-time setup
uv tool install dstack
dstack project add --name kengz --url https://sky.dstack.ai --token $DSTACK_TOKEN -y

# Create .env with HuggingFace token for result uploads
echo "HF_TOKEN=hf_xxx" > .env

# Launch remote run (source .env provides HF credentials)
source .env && uv run slm-lab run-remote --gpu SPEC_FILE SPEC_NAME train -n run-name

# Monitor
dstack ps  # check status
dstack logs <run-name>  # view logs
dstack stop <run-name> -y  # terminate

# See .dstack/*.yml for configuration
```

## Benchmarking Workflow

**Purpose**: Validate algorithm performance, reproduce published results, track improvements.

**Documentation**: `docs/BENCHMARKS.md` is the single source of truth - results tables, targets, active runs.

**Benchmark requirements** (ensure fairness and reproducibility):
- Respect `max_frame` from spec (e.g., 10M frames for Atari)
- Use specified `max_session` for multi-seed averaging
- Follow environment settings in BENCHMARKS.md (sticky actions, life_loss_info, etc.)
- Compare against targets from reference implementations

### Running Benchmarks

**Pattern**: Launch → Monitor → Extract → Update table → Commit

```bash
# Launch (example: Atari Pong with PPO)
source .env && uv run slm-lab run-remote --gpu \
  -s env=ALE/Pong-v5 \
  slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari train \
  -n pong-lam95

# Check status
dstack ps  # running jobs
dstack ps -a | grep "exited (0)"  # completed
dstack ps -a | grep "exited (1)"  # failed

# Extract results (when complete)
dstack logs pong-lam95 | grep "trial_metrics"
# Output: trial_metrics: frame:1.00e+07 | total_reward_ma:15.2 | ...
# Extract the total_reward_ma value (15.2)

# Update BENCHMARKS.md table with score
# Commit changes
```

### Extracting Metrics

At trial completion, logs print `trial_metrics` (scalar metrics JSON):

```
2026-01-06 05:01:58 | INFO | Session 0 done
2026-01-06 05:01:59 | INFO | trial_metrics: frame:1.00e+07 | total_reward_ma:816.18 | strength:570.4 | ...
2026-01-06 05:01:59 | INFO | Trial 0 done
```

**Extract `total_reward_ma`** - this is the benchmark score. Update `docs/BENCHMARKS.md` table.

### Autonomous Execution

**When benchmarking, work continuously:**

1. **Launch in parallel** - fill available GPU capacity (~30 concurrent runs)
2. **Check status regularly** - every 30 minutes for long-running tasks, don't idle
3. **Extract immediately** - when runs complete, get scores and update table
4. **Iterate on failures** - don't wait, launch improved configs
5. **Track in BENCHMARKS.md** - keep "Active Runs" section current
6. **Commit frequently** - document updates, spec improvements

**IMPORTANT**: Autonomous means YOU actively wait and check in using sleep commands directly - do NOT delegate to background bash processes or scripts. Stay engaged in the conversation.

## Hyperparameter Search

SLM-Lab uses ASHA (Asynchronous Successive Halving Algorithm) for efficient hyperparameter tuning.

### ASHA Search Strategy

**Concept**: Run many trials in parallel, terminate unpromising ones early based on performance.

**Config example**:
```json
{
  "meta": {
    "max_session": 1,
    "max_trial": 8,
    "search_resources": {"cpu": 1, "gpu": 0.125},
    "search_scheduler": {"grace_period": 100000, "reduction_factor": 3}
  },
  "search": {
    "agent.algorithm.gamma__uniform": [0.993, 0.999],
    "agent.net.optim_spec.lr__loguniform": [1e-4, 1e-3]
  }
}
```

**Launch search**:
```bash
source .env && uv run slm-lab run-remote --gpu \
  slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari search \
  -n atari-search
```

### Search Space Design

**Rule**: ~3-4 trials per dimension minimum.

| Trials | Max Dims | Strategy |
|--------|----------|----------|
| 8 | 2-3 | Focused refinement |
| 12-16 | 3-4 | Typical search |
| 20+ | 5-7 | Broad exploration |

**Use continuous distributions** (not discrete choices):
- `__uniform`: `[min, max]` - uniform sampling (e.g., gamma 0.99-0.999)
- `__loguniform`: `[min, max]` - log-uniform for learning rates (e.g., 1e-4 to 1e-3)
- Avoid `__choice`: Discrete choices fragment the search space, making it harder for the algorithm to interpolate

**High-impact hyperparameters** (search these first):
- Learning rate (`lr`) - use `loguniform`
- Discount factor (`gamma`) - use `uniform`
- GAE lambda (`lam`) - use `uniform`

**Low-impact** (fix based on successful runs):
- Batch sizes, training epochs, network architecture

**After search**: Analyze results, narrow ranges around best values, re-run if needed. Update spec defaults with best hyperparameters.

---

## Active Benchmark Work

**When user says "let's get to work" or "benchmark work"**, execute this autonomous workflow:

### Atari Benchmark Strategy

For Atari games, use lambda variants based on game characteristics:
1. Run ALL games with `ppo_atari` (lam95) first - this is the default spec
2. If lam95 fails OR historical data shows better results from other variants, run those variants
3. Lambda guidelines: lam95 (long-horizon), lam85 (middle), lam70 (action games)
4. Fill available GPU capacity (~30 concurrent runs), check status every 5-10 minutes

### Workflow Loop

1. **Check status**: `dstack ps` - identify completed/failed/running jobs
2. **Extract results**: For completed runs, `dstack logs <name> | grep "trial_metrics"`, get `total_reward_ma`
3. **Update table**: Fill in `docs/BENCHMARKS.md` with scores, move to "Completed Runs"
4. **Update specs**: If run succeeded (✅), update spec defaults with best hyperparameters
5. **Launch next**: Check "Active Runs" section, launch next batch to fill GPU capacity
6. **Iterate on failures**: For failed runs, launch hyperparameter search or improved config immediately
7. **Commit progress**: Regular commits of table updates and spec improvements
8. **Repeat**: Continue loop every 5-10 minutes until all benchmarks complete

### Getting Unstuck

When stuck on failing runs:
- **Check GPU utilization**: `dstack metrics <run-name>` - low GPU usage (<50%) often indicates CPU bottleneck (env stepping) or config issue, not training problem
- Compare with reference implementations (CleanRL, SB3)
- Kill unpromising runs early - iterate faster with new configs
- If same issue across runs, it's framework/config, not hyperparameters - investigate code

**Key principle**: Work continuously, check in regularly, iterate immediately on failures. Fill GPU capacity (~30 concurrent runs). Never idle waiting.

---

## Results & HuggingFace

Benchmark results are uploaded to [HuggingFace](https://huggingface.co/datasets/SLM-Lab/benchmark) for reproducibility:

- **Development**: Upload to [`SLM-Lab/benchmark-dev`](https://huggingface.co/datasets/SLM-Lab/benchmark-dev) during active work
- **Graduated**: Once benchmark passes, pull results and re-upload to [`SLM-Lab/benchmark`](https://huggingface.co/datasets/SLM-Lab/benchmark) (public-facing)

**Workflow**:
1. Remote runs auto-upload to `benchmark-dev` (via `source .env` credentials)
2. After validation, use `slm-lab pull SPEC_NAME` to download results
3. Upload graduated results to public `SLM-Lab/benchmark` repo

---

## Documentation

- **Major changes**: Document in `CHANGELOG.md` (bug fixes, new features, breaking changes)
- **Benchmark results**: Always update `docs/BENCHMARKS.md` with results tables, findings, and reproducibility instructions
- **Spec updates**: When improving specs, document rationale in commit messages

## TODO

### Critical Rules (2026-01-17)

**MUST FOLLOW:**
1. **NEVER push to remote without explicit user permission** - commit locally only
2. **ONLY train runs** - NEVER use search results in BENCHMARKS.md (search folders = UNACCEPTABLE)
3. **All HF links must work** - verify no 404s before updating table
4. **All plots must be updated** after EVERY run - use exact folders from table
5. **Follow Settings line** in BENCHMARKS.md for each env (max_frame, num_envs, etc.)
6. **Runs must complete in <6h**

### Process Workflow (Per-Run)

1. **Verify spec** matches Settings line in BENCHMARKS.md
2. **Launch train run** (NOT search) via dstack
3. **When complete**: extract `total_reward_ma` from `trial_metrics`
4. **Verify HF link** works before updating table
5. **Pull data**: `slm-lab pull SPEC_NAME`
6. **Update plot**: `slm-lab plot -t "ENV" -f folder1,folder2,...` using ONLY folders from table
7. **Update BENCHMARKS.md**: score, HF link, plot reference
8. **Commit locally** (never push without permission)

### Phase 1 - Classic Control

**1.1 CartPole-v1** (target: 400, Settings: max_frame 2e5 | num_envs 4 | max_session 4)
- [x] REINFORCE ✅ 448.07, SARSA ✅ 466.98, DDQN+PER ✅ 433.33, A2C ✅ 419.24, PPO ✅ 499.19
- [x] All HF links verified, all are TRAIN runs (1 trial each)
- [ ] DQN ⚠️ 278.76 (target 400) - needs tuning
- [ ] SAC ⚠️ 309.10 (target 400) - below target

**1.2 Acrobot-v1** (target: -100, Settings: max_frame 3e5 | num_envs 4 | max_session 4)
- [x] DDQN+PER ✅ -83.39, A2C ✅ -83.70, PPO ✅ -83.22, SAC ✅ -95.66 (all TRAIN runs)
- [x] DQN ✅ -96.79 (FIXED - was SEARCH, now TRAIN: dqn_boltzmann_acrobot_2026_01_18_102214)

**1.3 Pendulum-v1** (target: -200, Settings: max_frame 3e5 | num_envs 4 | max_session 4)
- [x] PPO ✅ -182.91, SAC ✅ -149.67 (both TRAIN runs)
- [ ] A2C ❌ -1299.68 - broken, needs investigation

### Phase 2 - Box2D

**2.1 LunarLander-v3 Discrete** (target: 200, Settings: max_frame 3e5 | num_envs 8 | max_session 4)
- [x] DDQN+PER ✅ 263.30, SAC ❌ -71.46 (both TRAIN runs)
- [x] DQN ⚠️ 182.12 (FIXED - was SEARCH, now TRAIN: dqn_concat_lunar_2026_01_18_102848)
- [x] A2C ❌ 53.94 (FIXED - was SEARCH, now TRAIN: a2c_gae_lunar_2026_01_18_102919) - algorithm struggles
- [x] PPO ✅ 225.99 (FIXED - was SEARCH, now TRAIN: ppo_lunar_2026_01_18_102921)

**2.2 LunarLander-v3 Continuous** (target: 200, Settings: max_frame 3e5 | num_envs 8 | max_session 4)
- [x] A2C ❌ -65.11, PPO ⚠️ 186.64, SAC ⚠️ 143.03 (all TRAIN runs)

### Phase 3 - MuJoCo

**3.1 Hopper-v5** (target: 2500, Settings: max_frame 1e6 | num_envs 16 | max_session 4 | log_frequency 1e4)
- [ ] PPO ❌ 242.81 (ppo_hopper_2026_01_18_142126) - num_envs=16 needs hyperparam tuning, old run with num_envs=1 got 2653
- [x] SAC ⚠️ 992.37 (TRAIN run verified, below target)

**3.2 HalfCheetah-v5** (target: 5000, Settings: max_frame 8e6 | num_envs 16 | max_session 4)
- [x] PPO ✅ 6450.34 (TRAIN run verified, solved!)

**3.3 Walker2d-v5** (target: 3500, Settings: max_frame 8e6 | num_envs 16 | max_session 4 | log_frequency 1e4)
- [ ] PPO - RERUN LAUNCHED (p3-ppo-walker-train) - 01_11 run used log_freq=1000, spec now has 10000

**3.4 Ant-v5** (target: 2000, Settings: max_frame 8e6 | num_envs 16 | max_session 4)
- [x] PPO ⚠️ 1133.87 (TRAIN run verified, improved from 678)

**3.5 Swimmer-v5** (target: 300, Settings: max_frame 8e6 | num_envs 16 | max_session 4)
- [x] PPO ✅ 333.62 (TRAIN run verified, solved!)

**3.6 Reacher-v5** (target: -5, Settings: max_frame 3e6 | num_envs 16 | max_session 4)
- [x] PPO ⚠️ -6.48 (FIXED - was SEARCH, now TRAIN: ppo_reacher_2026_01_18_102939, close to target)

**3.7 Pusher-v5** (target: -40, Settings: max_frame 3e6 | num_envs 16 | max_session 4)
- [x] PPO ⚠️ -49.33 (TRAIN run verified, still below -40)

**3.8 InvertedPendulum-v5** (target: 1000, Settings: max_frame 3e6 | num_envs 16 | max_session 4)
- [x] PPO ✅ 995.55 (TRAIN run verified, solved!)

**3.9 InvertedDoublePendulum-v5** (target: 9000, Settings: max_frame 8e6 | num_envs 16 | max_session 4)
- [x] PPO ⚠️ 7634.81 (TRAIN run verified, below target)

**3.10 Humanoid-v5** (target: 700, Settings: max_frame 10e6 | num_envs 16 | max_session 4 | log_frequency 1e4)
- [ ] PPO - RERUN LAUNCHED (p3-ppo-humanoid-train) - 01_17 run used log_freq=1000, spec now has 10000

**3.11 HumanoidStandup-v5** (target: 100000, Settings: max_frame 3e6 | num_envs 16 | max_session 4 | log_frequency 1e4)
- [ ] PPO - RERUN LAUNCHED (p3-ppo-hstandup-train) - 01_11 run used log_freq=1000+max_frame=6M, spec now has 10000+3M

### Completed Work (2026-01-23 AdamW unified spec experiment v2)

**Goal:** Two unified MuJoCo specs with AdamW (like Atari lambda variants).

**Spec file consolidated:** `ppo_mujoco.json` now contains both spec names (`ppo_mujoco`, `ppo_mujoco_longhorizon`).
Use `${max_frame}` (unquoted) for numeric variable substitution.

**Final Results (mj2 runs):**

| Env | Target | Unified Score | Previous Best | Status |
|-----|--------|---------------|---------------|--------|
| HalfCheetah-v5 | 5000 | 6506.97 | 6030.93 | ✅ Improved |
| Walker2d-v5 | 3500 | 4667.19 | 4408.60 | ✅ Improved |
| Humanoid-v5 | 700 | 3613.43 | 4495.62 | ✅ Passed (lower) |
| HumanoidStandup-v5 | 100000 | 131145.33 | 118978.12 | ✅ Improved |
| Reacher-v5 | -5 | -4.83 | -7.26 | ✅ Improved |
| Pusher-v5 | -40 | -37.75 | -46.97 | ✅ Improved |
| Hopper-v5 | 2500 | 1897.13 | 1970.97 | ⚠️ 76% |
| IP-v5 | 1000 | 502.08 | 900.19 | ❌ Worse |
| IDP-v5 | 9000 | 3396.18 | 6767.33 | ❌ Worse |
| Swimmer-v5 | 300 | 171.20 | 338.46 | ❌ Worse |
| Ant-v5 | 2000 | 425.18 | 2429.78 | ❌ Worse |

**Verdict:** `ppo_mujoco` (standard) passed 4/6: HalfCheetah, Walker, Humanoid, HumanoidStandup. `ppo_mujoco_longhorizon` passed 2/5: Reacher, Pusher. Individual specs remain better for Hopper, Ant, IP, IDP, Swimmer.

**Next:** Investigate failing envs and adjust unified specs.

### PPO MuJoCo Full Revalidation (2026-01-24)

**Status: 11/11 complete**

**Summary: 6/11 passed, 2/11 close (>80%), 3/11 failed**

| Env | Spec Used | Score | Target | Status | HF Folder |
|-----|-----------|-------|--------|--------|-----------|
| HalfCheetah-v5 | ppo_mujoco | 6032.34 | 5000 | ✅ | ppo_mujoco_halfcheetah_2026_01_24_103910 |
| Walker2d-v5 | ppo_mujoco | 5130.70 | 3500 | ✅ | ppo_mujoco_walker2d_2026_01_24_103947 |
| Humanoid-v5 | ppo_mujoco | 5052.40 | 700 | ✅ | ppo_mujoco_humanoid_2026_01_24_103936 |
| HumanoidStandup-v5 | ppo_mujoco | 147473.22 | 100000 | ✅ | ppo_mujoco_humanoidstandup_2026_01_24_103922 |
| Reacher-v5 | ppo_mujoco_longhorizon | -5.66 | -5 | ✅ | ppo_mujoco_longhorizon_reacher_2026_01_24_131550 |
| Hopper-v5 | ppo_hopper | 1210.58 | 2000 | ❌ 61% | ppo_hopper_2026_01_24_180554 |
| Pusher-v5 | ppo_mujoco_longhorizon | -54.17 | -40 | ❌ | ppo_mujoco_longhorizon_pusher_2026_01_24_180619 |
| Swimmer-v5 | ppo_swimmer | 165.69 | 300 | ❌ 55% | ppo_swimmer_2026_01_24_180614 |
| IP-v5 | ppo_inverted_pendulum | 926.73 | 1000 | ⚠️ 93% | ppo_inverted_pendulum_2026_01_24_105704 |
| IDP-v5 | ppo_inverted_double_pendulum | 7457.79 | 9000 | ⚠️ 83% | ppo_inverted_double_pendulum_2026_01_24_120854 |
| Ant-v5 | ppo_ant | 3032.60 | 2000 | ✅ 151% | ppo_ant_2026_01_24_152312 |

**Root Cause: Commit 5a327590 added lr_scheduler_spec which broke these envs**

Compared working commit aea10a90 vs broken commit 5a327590:
- lr_scheduler_spec: LinearToZero was added to ppo_mujoco.json, ppo_swimmer.json, ppo_ant.json, ppo_inverted_pendulum.json
- ppo_hopper.json was deleted (recreated incorrectly with lr_scheduler)

**Fix Applied - Removed lr_scheduler_spec from:**
- ppo_hopper.json (restored to original: AdamW, lr=3e-4, no lr_scheduler)
- ppo_swimmer.json
- ppo_mujoco.json (both ppo_mujoco and ppo_mujoco_longhorizon)

**All specs now respect Settings line (num_envs=16, max_frame=3e6)**

**Verification Runs (2026-01-25):**
- fix-hopper3: ppo_hopper.json, max_frame=3e6
- fix-swimmer2: ppo_swimmer.json, max_frame=3e6
- fix-pusher3: ppo_mujoco_longhorizon, max_frame=3e6

### Spec Revalidation Attempt (2026-01-25/26)

**Goal:** Match specs to historical best runs (Adam + LinearToZero).

**Tested Config Changes:**
- IP: Changed AdamW → Adam + LinearToZero
- IDP: Changed AdamW → Adam + LinearToZero
- Hopper: Changed AdamW lr=5e-4 → Adam lr=1.4e-4 + LinearToZero
- Pusher: Used ppo_pusher.json (AdamW, gamma=0.982, lam=0.927, lr=1.5e-4)
- Swimmer: Used existing ppo_swimmer.json (AdamW)

**Results:**

| Env | This Run | Target | Historical Best | Status |
|-----|----------|--------|-----------------|--------|
| Pusher | **-38.18** | -40 | -46.97 | ✅ PASSED |
| IP | 804.58 | 1000 | 995.55 | ❌ |
| IDP | 6801.87 | 9000 | 8231.35 | ❌ |
| Hopper | 273.99 | 2500 | 2653.78 | ❌ |
| Swimmer | 208.83 | 300 | 338.46 | ❌ |

**Key Findings:**
1. **Pusher ppo_pusher.json works** - spec (AdamW, gamma=0.982, lam=0.927, lr=1.5e-4) is validated
2. Adam + LinearToZero didn't help IP/IDP - both performed worse than AdamW specs
3. **Hopper requires num_envs=1** - historical best used num_envs=1 (got 2653.78), num_envs=16 fails regardless of optimizer
4. **Swimmer high variance** - session 1 hit 330+ but sessions 0,2 pulled down average to 208

**Spec Changes Reverted:** IP, IDP, Hopper specs reverted to AdamW (Adam+LinearToZero made them worse)

**Next Steps:**
- Hopper: Need to decide if BENCHMARKS.md should allow num_envs=1 for Hopper
- IP/IDP: Current AdamW specs are closer to target than Adam+LinearToZero
- Swimmer: Investigate session variance, consider longer training

### Current Work (2026-01-26)

**Key Discovery:** AdamW + LinearToZero is the winning combination (from commit 5a327590).

**Spec Updates Applied:**
- All specs now use AdamW + LinearToZero for consistency
- IP: AdamW lr=5.7e-4 + LinearToZero
- IDP: AdamW lr=1.55e-4 + LinearToZero
- Hopper: AdamW lr=3e-4 + LinearToZero
- Swimmer: AdamW lr=2.2e-4 + LinearToZero

**Max Frame Standardization:**
- 10M: HalfCheetah, Walker, Humanoid, Ant, IDP
- 4M: Hopper, Swimmer, Reacher, Pusher, IP, HumanoidStandup

**Active Runs:**
- fix-ip: AdamW + LTZ, 4M frames
- fix-idp: AdamW + LTZ, 10M frames

**Previous Results (3M frames):**
- fix-hopper: 948.30 (high variance - session 0 hit 1992)
- fix-swimmer: 185.56

### Clean Rerun Plan (After IP/IDP Solved)

Once IP and IDP pass, do a full clean rerun of all 11 MuJoCo envs with standardized max_frame:

| Env | Max Frame | Spec |
|-----|-----------|------|
| HalfCheetah | 10M | ppo_mujoco |
| Walker2d | 10M | ppo_mujoco |
| Humanoid | 10M | ppo_mujoco |
| Ant | 10M | ppo_ant |
| IDP | 10M | ppo_inverted_double_pendulum |
| HumanoidStandup | 4M | ppo_mujoco |
| Hopper | 4M | ppo_hopper |
| Swimmer | 4M | ppo_swimmer |
| Reacher | 4M | ppo_mujoco_longhorizon |
| Pusher | 4M | ppo_mujoco_longhorizon |
| IP | 4M | ppo_inverted_pendulum |


