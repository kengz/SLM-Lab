---
name: slm-lab-benchmark
description: Run SLM-Lab deep RL benchmarks, monitor dstack jobs, extract results, and update BENCHMARKS.md. Use when asked to run benchmarks, check run status, extract scores, update benchmark tables, or generate plots.
---

# SLM-Lab Benchmark Skill

## Critical Rules

1. **NEVER push to remote** without explicit user permission
2. **ONLY train runs** in BENCHMARKS.md — never search results
3. **Respect Settings line** for each env (max_frame, num_envs, etc.)
4. **Use `${max_frame}` variable** in specs — never hardcode
5. **Runs must complete in <6h** (dstack max_duration)
6. **Max 10 concurrent dstack runs** — launch in batches of 10, wait for capacity/completion before launching more. Never submit all runs at once; dstack capacity is limited and mass submissions cause "no offers" failures

## Frame Budget — MANDATORY CALCULATION (do this BEFORE every submission)

**dstack kills jobs at 6h with ZERO data** — no trial_metrics, no HF upload, nothing. A run killed at the wall = complete waste.

**Rule: max_frame = observed_fps × 5.5h × 3600** (5.5h, not 6h — leaves 30min margin)

**ALWAYS check FPS after 5-10 min of a new run before committing to the frame budget:**
```bash
dstack logs NAME --since 10m 2>&1 | grep "trial_metrics" | tail -3
# fps = frames_so_far / elapsed_seconds
```
If projected wall clock > 5.5h at observed fps → **stop immediately and relaunch with reduced max_frame**.

**Known fps at 64 envs (ppo_playground):**
| Env category | fps | Safe max_frame (5.5h) |
|---|---|---|
| CartpoleBalance, CheetahRun, WalkerWalk | ~450-1800 | 8M–10M |
| WalkerStand, HopperStand | ~270 | 5M |
| HumanoidStand | ~200 | 4M |
| HumanoidWalk | ~290 | 5M |
| Rough terrain loco (G1Rough, T1Rough, Go1Getup) | ~60-65 | 1M |
| BerkeleyHumanoidRough | ~36 | 700K |

**For unknown envs:** Submit with conservative 2M, check fps after 5 min, stop and relaunch with correct budget if needed.

**Phase 5 Playground spec selection:**
- DM Control (5.1): `ppo_playground` (1024 envs), `sac_playground` (256 envs), `crossq_playground` (16 envs)
- Locomotion (5.2) / Manipulation (5.3): `ppo_playground_loco` (512 envs), same SAC/CrossQ specs
- DM Control with NaN rewards: override with `-s normalize_obs=false`
- Run order: PPO first (fastest), then SAC, then CrossQ

## Per-Run Intake Checklist

**Every completed run MUST go through ALL of these steps. No exceptions. Do not skip any step.**

When a run completes (`dstack ps` shows `exited (0)`):

1. **Extract score**: `dstack logs NAME | grep "trial_metrics"` → get `total_reward_ma`
2. **Find HF folder name**: `dstack logs NAME 2>&1 | grep "Uploading data/"` → extract folder name from the upload log line
3. **Update table score** in BENCHMARKS.md
4. **Update table HF link**: `[FOLDER](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/FOLDER)`
5. **Pull HF data locally**: `source .env && huggingface-cli download SLM-Lab/benchmark-dev --local-dir data/benchmark-dev --repo-type dataset --include "data/FOLDER/*"`
6. **Generate plot** (MANDATORY — do NOT skip):
   ```bash
   uv run slm-lab plot -t "EnvName" -d data/benchmark-dev/data -f FOLDER1,FOLDER2,...
   ```
   CRITICAL RULES for plot generation:
   - Use ONLY the exact folder(s) from the HF Data column of the BENCHMARKS.md table — NEVER grep or ls to find folders
   - Multiple folders in data/benchmark-dev/data/ may exist for the same env (old failed runs + new good runs). Only use the canonical folder from the table.
   - Include ALL algorithms that have entries in the table for that env (e.g., both PPO and SAC folders if both have scores)
   - If the canonical folder is in local `data/` (not in `data/benchmark-dev/data/`), use `-d data` instead
   - `-d` sets the base data dir, `-f` takes folder names (NOT full paths)
7. **Display plot** (MANDATORY — call the Read tool on the image file, no exceptions):
   ```
   Read: docs/plots/EnvName_multi_trial_graph_mean_returns_ma_vs_frames.png
   ```
   This MUST happen in your agent turn — call Read, see the image, THEN send your completion message.
   Team-lead must also call Read to display it in the main conversation.
8. **Embed plot in BENCHMARKS.md** — for Phase 5 playground envs, ensure the plot is in the DM Control plot grid (search for the existing grid in the Phase 5 section). If the env is already in the grid, no action needed. If missing, add it.
9. **Commit** score + link + plot together

A row in BENCHMARKS.md is NOT complete until it has: score, HF link, and plot.

## Per-Run Graduation Checklist

**After intake, graduate each finalized run to public HF benchmark:**

1. **Upload folder to public HF**:
   ```bash
   source .env && huggingface-cli upload SLM-Lab/benchmark data/benchmark-dev/data/FOLDER data/FOLDER --repo-type dataset
   ```
2. **Update BENCHMARKS.md link**: Change `SLM-Lab/benchmark-dev` → `SLM-Lab/benchmark` for that entry
3. **Upload docs/ to public HF** (updated plots + BENCHMARKS.md):
   ```bash
   source .env && huggingface-cli upload SLM-Lab/benchmark docs docs --repo-type dataset
   source .env && huggingface-cli upload SLM-Lab/benchmark README.md README.md --repo-type dataset
   ```
4. **Commit** link update
5. **Push** to origin

## Launch

```bash
# Launch a run
source .env && uv run slm-lab run-remote --gpu \
  -s env=ALE/Pong-v5 SPEC_FILE SPEC_NAME train -n NAME

# Monitor
dstack ps                              # running jobs
dstack logs NAME | grep "trial_metrics" # extract score at completion

# Score = total_reward_ma from trial_metrics line
# trial_metrics: frame:1.00e+07 | total_reward_ma:816.18 | ...
```

## Data Lifecycle

```
Remote GPU run → auto-uploads to benchmark-dev (HF)
  ↓ Pull to local data/
  ↓ Generate plots (docs/plots/)
  ↓ Update BENCHMARKS.md (scores, links, plots)
  ↓ Graduate to public benchmark (HF)
  ↓ Update links: benchmark-dev → benchmark
  ↓ Upload docs/ to public benchmark (HF)
```

### Pull Data

```bash
# Pull full dataset (fast, single request — avoids rate limits)
source .env && hf download SLM-Lab/benchmark-dev \
  --local-dir data/benchmark-dev --repo-type dataset

# Or pull specific folder
source .env && hf download SLM-Lab/benchmark-dev \
  --local-dir data/benchmark-dev --repo-type dataset --include "data/FOLDER/*"

# KEEP this data — needed for plots AND graduation upload later
```

### Generate Plots

```bash
# Find folders for a game (check both local data/ and benchmark-dev)
ls data/ | grep -i pong
ls data/benchmark-dev/data/ | grep -i pong

# Generate comparison plot — use -d for base dir, -f for folder names only
# Use data/ as base (has info/ subfolder with trial_metrics)
uv run slm-lab plot -t "Pong-v5" -f ppo_pong_folder,sac_pong_folder,crossq_pong_folder
```

### Graduate to Public HF

When a run is finalized, graduate individually from `benchmark-dev` → `benchmark`:

```bash
# Upload individual folder
source .env && huggingface-cli upload SLM-Lab/benchmark \
  data/benchmark-dev/data/FOLDER data/FOLDER --repo-type dataset

# Update BENCHMARKS.md link for that entry: benchmark-dev → benchmark
# Then upload docs/ (includes updated plots + BENCHMARKS.md)
source .env && huggingface-cli upload SLM-Lab/benchmark docs docs --repo-type dataset
source .env && huggingface-cli upload SLM-Lab/benchmark README.md README.md --repo-type dataset
```

| Repo | Purpose |
|------|---------|
| `SLM-Lab/benchmark-dev` | Development — noisy, iterative |
| `SLM-Lab/benchmark` | Public — finalized, validated |

## Hyperparameter Search

Only when algorithm fails to reach target:

```bash
source .env && uv run slm-lab run-remote --gpu SPEC_FILE SPEC_NAME search -n NAME
```

Budget: ~3-4 trials per dimension. After search: update spec with best params, run `train`, use that result.

## Agent Team Workflow (MANDATORY for team lead)

**You are the team lead. Never work solo on benchmarks — always spawn an agent team.**

### Team Roles

**launcher** — Reads BENCHMARKS.md, identifies missing entries, launches up to 10 dstack runs. Checks FPS after ~5min and stops slow runs (>6h projected). Reports run names + envs to team lead.

**monitor** — Polls `dstack ps` every 5min (`sleep 300 && dstack ps`). Detects completions and failures. When runs complete, assigns intake tasks. When runs fail, reports to team lead immediately. Runs continuously until all runs are done.

**intake-A / intake-B / intake-C** — Each owns a batch of 3-4 completed runs. Executes the full intake checklist (score → HF folder → pull data → plot → BENCHMARKS.md update). Does NOT commit — team lead commits.

### Spawn Pattern

```
TeamCreate → TaskCreate (one per batch of runs) →
  Agent(launcher) + Agent(monitor) + Agent(intake-A) + Agent(intake-B) + ...
```

Spawn all agents in parallel. Intake agents start idle and pick up work as monitor assigns completed runs.

### Team Lead Responsibilities

1. **On spawn**: Brief each agent with full context (run names, env names, BENCHMARKS.md format, intake checklist)
2. **On intake completion**: Read each plot image (Read tool), verify BENCHMARKS.md edits, then commit
3. **On monitor report**: If runs fail, relaunch immediately; if fps too slow, stop + reduce frames
4. **Commit cadence**: Batch-commit after each intake wave (score + HF link + plot per commit)
5. **Shutdown team**: When all runs intaked and committed, send shutdown_request to all teammates

### Monitor Agent Instructions Template

```
You are monitor on team TEAM_NAME. Poll dstack ps every 5min.
Active runs: [LIST OF RUN NAMES]
When a run shows exited(0): send message to team-lead with run name and env name.
When a run shows exited(1) or failed: send message to team-lead immediately.
Use: while true; do dstack ps; sleep 300; done
Stop when team-lead sends shutdown_request.
```

### Intake Agent Instructions Template

```
You are intake-agent-X on team TEAM_NAME. Intake these completed runs: [LIST]
For each run, follow the full intake checklist in the benchmark skill.
Working dir: /Users/keng/projects/SLM-Lab
Do NOT commit — team lead commits.
After all runs done: send results summary to team-lead (scores, HF folders, any issues).
```

### Autonomous Execution

**Workflow loop** (team lead orchestrates, agents execute):
1. **launcher**: Identifies gaps in BENCHMARKS.md → launches up to 10 runs → reports to team lead
2. **monitor**: Watches for completions → notifies team lead → assigns intake work
3. **intake agents**: Execute full checklist per run → report to team lead
4. **team lead**: Reviews plots, commits, relaunches failures, spawns next batch

**Key principle**: Keep agents working in parallel. Never idle as team lead while GPU runs are active — spawn a monitor agent. Commit after each intake wave. Shut down team cleanly when done.

## Troubleshooting

- **Run interrupted**: Relaunch, increment name suffix (e.g., pong3 → pong4)
- **Low GPU usage** (<50%): CPU bottleneck or config issue
- **HF rate limit**: Download full dataset, not selective `--include` patterns
- **HF link 404**: Run didn't complete or upload failed — rerun
- **.env inline comments**: Break dstack env vars — put comments on separate lines
