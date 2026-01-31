---
name: slm-lab-benchmark
description: Run SLM-Lab deep RL benchmarks, monitor dstack jobs, extract results, and update BENCHMARKS.md. Use when asked to run benchmarks, check run status, extract scores, update benchmark tables, or generate plots.
---

# SLM-Lab Benchmark Workflow

## Critical Rules

1. **NEVER push to remote** without explicit user permission - commit locally only
2. **ONLY train runs** in BENCHMARKS.md - NEVER use search results (search folders = UNACCEPTABLE)
3. **Respect Settings line** for each env (max_frame, num_envs, etc.) - see [BENCHMARKS.md](docs/BENCHMARKS.md)
4. **Use `${max_frame}` variable** in specs - never hardcode max_frame values
5. **Verify HF links work** before updating table
6. **Runs must complete in <6h**

## Benchmark Contribution Workflow

### 1. Audit Spec Settings

**Before Running**: Ensure spec matches the **Settings** line in BENCHMARKS.md for each env.

Example Settings line: `max_frame 3e5 | num_envs 4 | max_session 4 | log_frequency 500`

**After Pulling**: Verify downloaded `spec.json` matches these rules before using data.

### 2. Run Benchmark & Commit Specs

```bash
# Remote (GPU) - auto-syncs to HuggingFace
source .env && slm-lab run-remote --gpu SPEC_FILE SPEC_NAME train -n NAME

# With variable substitution (MuJoCo/Atari)
source .env && slm-lab run-remote --gpu -s env=ENV -s max_frame=MAX_FRAME \
  SPEC_FILE SPEC_NAME train -n NAME

# Local (Classic Control only)
slm-lab run SPEC_FILE SPEC_NAME train
```

**Always commit the spec file** used for the run. Ensure BENCHMARKS.md has entry with correct SPEC_FILE and SPEC_NAME.

### 3. Monitor Status

**Monitor autonomously** - use sleep to check in periodically until completion:
```bash
sleep 900 && dstack ps                 # wait 15min then check
```

```bash
dstack ps                              # running jobs
dstack ps -a | head -20                # recent jobs (done/failed)
dstack logs NAME                       # view logs
dstack logs NAME | grep "trial_metrics" # extract final score
dstack stop NAME -y                    # terminate run
```

### 4. Record Scores & Plots

**Score**: At end of run, extract `total_reward_ma` from logs (`trial_metrics`):
```
trial_metrics: frame:1.00e+07 | total_reward_ma:816.18 | strength:570.4 | ...
```

**Link**: Add HuggingFace folder link to table:
- Format: `[FOLDER](https://huggingface.co/datasets/SLM-Lab/benchmark-dev/tree/main/data/FOLDER)`

**Pull Data Efficiently** (avoid rate limiting):
```bash
# DON'T use slm-lab pull for bulk downloads - will get rate limited
# Instead, pull specific folders directly from HF:

# Get folder name from run logs
dstack logs NAME | grep "SLM-Lab: Running"
# Output shows: data/FOLDER_NAME/...

# Pull only the folder you need
source .env && uv run hf download SLM-Lab/benchmark-dev \
  --include "data/FOLDER_NAME/*" --local-dir hf_data --repo-type dataset
```

**Plot**:
```bash
# Verify scores in trial_metrics.json match logs
# Ensure all runs share same max_frame

# Generate plot using ONLY folders from table
slm-lab plot -t "ENV_NAME" -f folder1,folder2,folder3
```

**Status legend**: ✅ Solved | ⚠️ Close (>80%) | ❌ Failed

### 5. Commit Changes

```bash
git add docs/BENCHMARKS.md slm_lab/spec/benchmark/...
git commit -m "docs: update ENV benchmark (SCORE)"
# NEVER push without explicit permission
```

## Publishing to Public Dataset

During development, runs upload to `SLM-Lab/benchmark-dev` (noisy, iterative). When benchmarks are finalized, publish clean results to the public `SLM-Lab/benchmark` dataset.

### Strategy: `hf_data/` IS the Manifest

The `hf_data/data/` directory defines what gets uploaded. Same process works for any subset (Phase 1-3, Atari, single env).

### Upload Workflow

HF dataset mirrors repo structure: `README.md`, `docs/`, `data/`

```bash
# 1. Clear hf_data/ before upload cycle
rm -rf hf_data/

# 2. Pull only folders you want from benchmark-dev
source .env && uv run hf download SLM-Lab/benchmark-dev \
  --include "data/ppo_cartpole_2026*/*" "data/sac_lunar*/*" \
  --local-dir hf_data --repo-type dataset

# 3. Upload README to public repo
source .env && uv run hf upload SLM-Lab/benchmark README.md README.md --repo-type dataset

# 4. Upload data to public repo
source .env && uv run hf upload SLM-Lab/benchmark hf_data/data data --repo-type dataset

# 5. Update BENCHMARKS.md links: benchmark-dev -> benchmark
#    (data now exists on public repo, so links will work)

# 6. Upload docs (with updated links)
source .env && uv run hf upload SLM-Lab/benchmark docs docs --repo-type dataset
```

### Two-Repo Strategy

| Repo | Purpose | Links in BENCHMARKS.md |
|------|---------|------------------------|
| `SLM-Lab/benchmark-dev` | Development iterations, noisy | During active work |
| `SLM-Lab/benchmark` | Public, finalized results | After publishing |

## Hyperparameter Search

Only when algorithm fails to reach target. Use search to find hyperparams, then run final `train` for benchmark.

```bash
source .env && slm-lab run-remote --gpu SPEC_FILE SPEC_NAME search -n NAME
```

**Search budget**: ~3-4 trials per dimension (8 trials = 2-3 dims, 16 = 3-4 dims).

**After search**: Update spec with best hyperparams, run `train` mode, use that result in BENCHMARKS.md.

## Troubleshooting

- **Run interrupted**: Expected with spot instances - relaunch with same command, increment run name (e.g. rv2 → rv3)
- **Low GPU usage** (<50%): CPU bottleneck or config issue, not training problem
- **Score below target**: Check hyperparams match spec, try search mode
- **HF link 404**: Run didn't complete or upload failed, rerun

For full details, see [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
