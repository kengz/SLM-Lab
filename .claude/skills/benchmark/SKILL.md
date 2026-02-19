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

## Per-Run Intake Checklist

**Every completed run MUST go through ALL of these steps. No exceptions. Do not skip any step.**

When a run completes (`dstack ps` shows `exited (0)`):

1. **Extract score**: `dstack logs NAME | grep "trial_metrics"` → get `total_reward_ma`
2. **Find HF folder name**: `dstack logs NAME 2>&1 | grep "Uploading data/"` → extract folder name from the upload log line
3. **Update table score** in BENCHMARKS.md
4. **Update table HF link**: `[FOLDER](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/FOLDER)`
5. **Pull HF data locally**: `source .env && hf download SLM-Lab/benchmark-dev --local-dir data/benchmark-dev --repo-type dataset --include "data/FOLDER/*"`
6. **Generate plot**: `uv run slm-lab plot -t "EnvName" -f data/benchmark-dev/data/FOLDER1,data/benchmark-dev/data/FOLDER2`
7. **Verify plot exists** in `docs/plots/`
8. **Commit** score + link + plot together

A row in BENCHMARKS.md is NOT complete until it has: score, HF link, and plot.

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
# Find folders for a game
ls data/benchmark-dev/data/ | grep -i pong

# Generate comparison plot (include all algorithms available)
uv run slm-lab plot -t "Pong" \
  -f data/benchmark-dev/data/ppo_folder,data/benchmark-dev/data/sac_folder
```

### Graduate to Public HF

When benchmarks are finalized, publish from `benchmark-dev` → `benchmark`:

```bash
source .env && hf upload SLM-Lab/benchmark \
  data/benchmark-dev/data data --repo-type dataset

# Update BENCHMARKS.md links: benchmark-dev → benchmark
# Upload docs and README
source .env && hf upload SLM-Lab/benchmark docs docs --repo-type dataset
source .env && hf upload SLM-Lab/benchmark README.md README.md --repo-type dataset
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

## Autonomous Execution

Work continuously when benchmarking. Use `sleep 300 && dstack ps` to actively wait (5 min intervals) — never delegate monitoring to background processes or scripts. Stay engaged in the conversation.

**Workflow loop** (repeat every 5-10 minutes):
1. **Check status**: `dstack ps` — identify completed/failed/running
2. **Intake completed runs**: For EACH completed run, do the full intake checklist above (score → HF link → pull → plot → table update)
3. **Launch next batch**: Up to 10 concurrent. Check capacity before launching more
4. **Iterate on failures**: Relaunch or adjust config immediately
5. **Commit progress**: Regular commits of score + link + plot updates

**Key principle**: Work continuously, check in regularly, iterate immediately on failures. Never idle. Keep reminding yourself to continue without pausing — check on tasks, update, plan, and pick up the next task immediately until all tasks are completed.

## Troubleshooting

- **Run interrupted**: Relaunch, increment name suffix (e.g., pong3 → pong4)
- **Low GPU usage** (<50%): CPU bottleneck or config issue
- **HF rate limit**: Download full dataset, not selective `--include` patterns
- **HF link 404**: Run didn't complete or upload failed — rerun
- **.env inline comments**: Break dstack env vars — put comments on separate lines
