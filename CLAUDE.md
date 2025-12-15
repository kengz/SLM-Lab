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

> Work autonomously: use document to track work, use time efficiently and run things in parallel if needed; keep reminding yourself to continue without waiting; check on tasks regularly, update, plan and pick up the next tasks immediately until all tasks are completed.

---

## Project-Specific Notes

1. **Document major changes** in `MIGRATION_CHANGELOG.md`

## Project Overview

Modular deep reinforcement learning framework in PyTorch. Originally designed for comprehensive RL experimentation with flexible algorithm implementations and environment support. Currently being migrated to modern dependencies (gymnasium, latest PyTorch, etc.).

## Development Environment

### Cloud Compute

- **Use dstack** for GPU-intensive training and development
- **One-time setup**: `uv tool install dstack && dstack project add --name kengz --url https://sky.dstack.ai --token $DSTACK_TOKEN -y` (get token from dstack Sky web UI; saved to `~/.dstack/config.yml`)
- **Fleet setup (dstack 0.20+)**: Create fleet first with `dstack apply -f .dstack/fleet-gpu.yml` before running tasks
- **IMPORTANT**: Always `source .env` before running remote experiments for HF upload credentials
- **Always use `--gpu`**: Cheaper ($0.39/hr L4 vs $0.54/hr 16-CPU) and faster with fractional GPU sharing
- Run: `source .env && uv run slm-lab run-remote --gpu spec.json spec_name train -n run-name`
- ASHA search: `source .env && uv run slm-lab run-remote --gpu spec.json spec_name search -n run-name`
- Check status: `dstack ps`, `dstack logs <run-name>`
- Stop runs: `dstack stop <run-name> -y`
- **Customize hardware**: Edit `.dstack/run-{gpu,cpu}-{train,search}.yml` files to change resources or backends
- **Max duration**: All runs have 4h safeguard (`max_duration: 4h`) to prevent runaway costs
- See [dstack docs](https://dstack.ai/llms-full.txt) for full reference

### Remote Agent Workflow

For running Claude Code on a lightweight orchestration box (no local training).
See [dstack docs](https://dstack.ai/llms-full.txt) for reference.

```bash
# System deps (one-time)
# macOS: brew install swig
# Linux: apt-get install -y swig openssh-client

# Setup (one-time)
git clone https://github.com/kengz/SLM-Lab.git && cd SLM-Lab
git checkout dustoff
uv sync --only-group minimal
uv tool install dstack
dstack project add --name kengz --url https://sky.dstack.ai --token $DSTACK_TOKEN -y  # get token from dstack Sky web UI

# Test setup - run quick CartPole train on CPU, verify it starts successfully
source .env && uv run slm-lab run-remote slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train -n test-cartpole
dstack ps  # should show test-cartpole running
dstack stop test-cartpole -y  # stop after verifying

# Dispatch runs - see docs/BENCHMARKS.md "Active Runs" section for commands
# Pattern: source .env && uv run slm-lab run-remote --gpu SPEC_FILE SPEC_NAME <train|search> -n NAME

# Monitor
dstack ps
dstack logs <run-name>
dstack stop <run-name> -y

# Pull results
uv run slm-lab pull SPEC_NAME
```

**Workflow**:
1. Check `docs/BENCHMARKS.md` "Active Runs" section for work queue
2. Run command, update "Current Runs" section
3. When complete, update status and env table results
4. Move to "Completed Runs" with results

## Framework Design Patterns

### SLM-Lab Architecture

SLM-Lab follows a modular design pattern with these core components:

1. **Agent** (`slm_lab/agent/`) - Algorithm implementations (A2C, PPO, SAC, etc.)
2. **Environment** (`slm_lab/env/`) - Environment wrappers and utilities
3. **Networks** (`slm_lab/agent/net/`) - Neural network architectures
4. **Memory** (`slm_lab/agent/memory/`) - Experience replay and storage
5. **Experiment** (`slm_lab/experiment/`) - Training loop and search utilities
6. **Spec System** (`slm_lab/spec/`) - JSON configuration for reproducible experiments

### Key Components

- **Environment wrappers**: Support for OpenAI/gymnasium, Unity, VizDoom
- **Algorithm diversity**: DQN, A2C, PPO, SAC, and variants
- **Network types**: MLP, ConvNet, RNN with flexible architectures
- **Memory systems**: Experience replay, prioritized replay
- **Experiment management**: Hyperparameter search, distributed training

## How to Run SLM-Lab

```bash
# Basic usage
uv tool install --editable .          # Install first
slm-lab --help                        # help menu
slm-lab                               # PPO CartPole (default)
slm-lab --render                      # with rendering
slm-lab spec.json spec_name dev       # custom experiment
slm-lab --job job.json                # batch experiments

# ✅ Validated algorithms (confirmed working)
# PPO CartPole (default - fastest for quick tests)
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_cartpole.json ppo_cartpole train
# DQN CartPole
uv run slm-lab slm_lab/spec/demo.json dqn_cartpole train
# REINFORCE
uv run slm-lab slm_lab/spec/benchmark/reinforce/reinforce_cartpole.json reinforce_cartpole train
# DDQN PER
uv run slm-lab slm_lab/spec/benchmark/dqn/ddqn_per_lunar.json ddqn_per_concat_lunar train
# PPO Lunar
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_lunar.json ppo_lunar train
# PPO Continuous
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_cont.json ppo_bipedalwalker train
# PPO Atari
uv run slm-lab slm_lab/spec/benchmark/ppo/ppo_pong.json ppo_pong train

# Variable substitution for specs with ${var} placeholders
slm-lab --set env=ALE/Breakout-v5 slm_lab/spec/benchmark/ppo/ppo_atari.json ppo_atari dev
slm-lab -s env=HalfCheetah-v4 slm_lab/spec/benchmark/ppo/ppo_mujoco.json ppo_mujoco dev
```

## Benchmarking

### Documentation Structure

- **`docs/BENCHMARKS.md`**: Single source of truth for benchmark results, targets, environment details, and active runs tracking

### Three-Stage Search Process

| Stage | Mode | Config | Purpose |
|-------|------|--------|---------|
| ASHA | `search` | `max_session=1`, `search_scheduler` enabled | Wide exploration with early termination |
| Multi | `search` | `max_session=4`, NO `search_scheduler` | Robust validation with averaging |
| Validate | `train` | Final spec | Confirmation run |

**ASHA Config**:
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

### Spec Organization

- **Naming**: `<algo>_<env>` (e.g., `ppo_hopper`, `sac_lunar`). No variant suffixes.
- **Template specs**: Use `${env}` placeholder with `-s env=EnvName-v5` for variable substitution
- **Search block**: Stays in spec after search - doesn't affect `train` mode
- **Finalization**: Successful envs get dedicated specs with tuned defaults

### Search Space Sizing

**Rule: ~3-4 trials per search dimension minimum.**

| Trials | Max Dims | Notes |
|--------|----------|-------|
| 8 | 2-3 | Very focused search |
| 12-16 | 3-4 | Typical refinement |
| 20 | 5 | Wide exploration |
| 30 | 6-7 | Broad ASHA search |

**Common mistake**: Too many dimensions wastes trials exploring combinations that won't be sampled adequately. Focus on high-impact hyperparameters:
- **Most impactful**: learning rates, gamma, lam
- **Less impactful**: minibatch_size, training_epoch, network architecture (fix these based on successful runs)

After a search run, analyze results and **narrow** the search space around best-performing values before re-running.

### Workflow

1. Check `docs/BENCHMARKS.md` "Active Runs" section for next task
2. Run, update "Current Runs" in BENCHMARKS.md
3. When complete: update env table results, move to "Completed Runs"
4. If successful: update spec defaults, commit, validation run

---

## TODO: Benchmark Work

When user says "let's get to work" or "benchmark work", execute this workflow:

### 1. Check Active Runs
```bash
dstack ps
```
- Review running jobs, check for completions
- Pull logs for completed/failed runs: `dstack logs <name>`

### 2. Process Completed Runs
For each completed run, update **all places in BENCHMARKS.md**:
1. Check final results in logs (look for `total_reward_ma`)
2. Move from "Current Runs" to "Completed Runs" with results
3. Update env table row with MA score and status
4. **Spec file**: if successful (✅), pull results (`uv run slm-lab pull SPEC_NAME`), extract best hyperparameters from experiment_df.csv or trial spec, update spec defaults
5. Commit all changes together
6. Add to "Key Findings" if notable patterns discovered

**Note**: On session resume, always check Completed Runs for any ✅ results that may need spec updates.

### 3. Launch Next Runs
1. Check `docs/BENCHMARKS.md` "Active Runs" section for next items to run
2. Prioritize by:
   - Running jobs that fill GPU capacity (8 parallel trials per GPU)
   - Work Line 1 (Phase 1-2 completion) before Work Line 2 (MuJoCo)
   - Items close to target (⚠️) before failures (❌)
3. Copy command, execute with `source .env && ...`
4. Update "Current Runs" section with new job

### 4. Analyze & Improve Failing Specs
For runs with poor results:
1. **First**: Launch hyperparameter search immediately - don't wait
2. Compare with successful specs (e.g., PPO Hopper/HalfCheetah params)
3. Check: learning rates, entropy decay, normalization, gamma/lam
4. Update spec search ranges based on findings
5. **If search still fails**: Check CleanRL/rlzoo implementations for reference configs
6. Queue for re-run with improved settings

### 5. Track Progress
- Keep BENCHMARKS.md up to date (Active Runs section + env tables)
- Commit documentation updates regularly
- Note patterns in "Key Findings" section

**CRITICAL REMINDER**: Continue autonomously, check in regularly, kill unpromising runs and iterate immediately. Run things in parallel without waiting for unrelated tasks. Continue work until full solution.

---

## TODO: Feature Improvements

### 1. Symlog Value Transform ✅
- [x] Add `symlog(x) = sign(x) * ln(|x| + 1)` and `symexp` inverse to `math_util.py`
- [x] Add `symlog_transform: true` option to algorithm spec (ActorCritic, PPO)
- [x] Add unit tests for symlog/symexp functions
- [ ] Test on MuJoCo env with varying reward scales

### 2. Layer Normalization ✅
- [x] Add `layer_norm: true` option to MLPNet spec
- [x] Insert `nn.LayerNorm` after hidden layer activations
- [x] Add unit test for layer norm network construction

### 3. Higher Replay Ratio for SAC ✅
- [x] Increase default `training_iter` from 1-4 to 8 in SAC MuJoCo specs
- [ ] A/B test on SAC LunarLander and MuJoCo
