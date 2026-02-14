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

## Run → Score → Record

```bash
# Launch
source .env && uv run slm-lab run-remote --gpu \
  -s env=ALE/Pong-v5 SPEC_FILE SPEC_NAME train -n NAME

# Monitor
dstack ps                              # running jobs
dstack logs NAME | grep "trial_metrics" # extract score at completion

# Score = total_reward_ma from trial_metrics line
# trial_metrics: frame:1.00e+07 | total_reward_ma:816.18 | ...
```

## Data Lifecycle

Data flows through three stages: **pull → plot → graduate**. Keep local data until graduation is complete.

```
Remote GPU run → auto-uploads to benchmark-dev (HF)
                      ↓
               Pull to local data/
                      ↓
               Generate plots (docs/plots/)
                      ↓
               Update BENCHMARKS.md (scores, links)
                      ↓
               Graduate to public benchmark (HF)
                      ↓
               Update links: benchmark-dev → benchmark
                      ↓
               Upload docs/ to public benchmark (HF)
```

### Pull Data

```bash
# Pull full dataset (fast, single request — avoids rate limits)
source .env && uv run hf download SLM-Lab/benchmark-dev \
  --local-dir data/benchmark-dev --repo-type dataset

# KEEP this data — needed for plots AND graduation upload later
# Never rm -rf data/benchmark-dev/ until graduation is complete
```

### Generate Plots

```bash
# Find folders for a game (need a2c + ppo + sac for comparison)
ls data/benchmark-dev/data/ | grep -i pong

# Generate comparison plot
uv run slm-lab plot -t "Pong" \
  -f data/benchmark-dev/data/a2c_folder,data/benchmark-dev/data/ppo_folder,data/benchmark-dev/data/sac_folder
```

### Update BENCHMARKS.md

- Add score in results table
- Add HF link: `[FOLDER](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/FOLDER)`
- Status: ✅ Solved | ⚠️ Close (>80%) | ❌ Failed

### Graduate to Public HF

When benchmarks are finalized, publish from `benchmark-dev` → `benchmark`:

```bash
# Upload data (from local copy — already pulled above)
source .env && uv run hf upload SLM-Lab/benchmark \
  data/benchmark-dev/data data --repo-type dataset

# Update BENCHMARKS.md links: benchmark-dev → benchmark
# (find-replace in docs/BENCHMARKS.md)

# Upload docs (with updated links) and README
source .env && uv run hf upload SLM-Lab/benchmark docs docs --repo-type dataset
source .env && uv run hf upload SLM-Lab/benchmark README.md README.md --repo-type dataset
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

## Troubleshooting

- **Run interrupted**: Relaunch, increment name suffix (e.g., pong3 → pong4)
- **Low GPU usage** (<50%): CPU bottleneck or config issue
- **HF rate limit**: Download full dataset, not selective `--include` patterns
- **HF link 404**: Run didn't complete or upload failed — rerun
- **.env inline comments**: Break dstack env vars — put comments on separate lines
