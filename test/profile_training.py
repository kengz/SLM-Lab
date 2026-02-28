"""Profile SLM-Lab training with PyTorch profiler.

Usage:
    uv run python test/profile_training.py              # PPO CartPole (default)
    uv run python test/profile_training.py --algo sac   # SAC CartPole
    uv run python test/profile_training.py --frames 10000 --algo ppo
"""

import argparse
import os
import sys

# Set env vars before any SLM-Lab imports
os.environ["lab_mode"] = "train"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["OPTIMIZE_PERF"] = "true"
os.environ["PROFILE"] = "false"
os.environ["RENDER"] = "false"
os.environ["LOG_EXTRA"] = "false"
os.environ["UPLOAD_HF"] = "false"

import torch
from torch.profiler import ProfilerActivity, profile

from slm_lab.experiment.control import Session
from slm_lab.spec import spec_util


ALGO_CONFIGS = {
    "ppo": {
        "spec_file": "benchmark/ppo/ppo_cartpole.json",
        "spec_name": "ppo_cartpole",
    },
    "sac": {
        "spec_file": "benchmark/sac/sac_cartpole.json",
        "spec_name": "sac_cartpole",
    },
}


def load_spec(algo: str, max_frame: int, num_envs: int) -> dict:
    """Load and configure spec for profiling."""
    config = ALGO_CONFIGS[algo]
    spec = spec_util.get(config["spec_file"], config["spec_name"])

    # Override for quick profiling
    spec["env"]["max_frame"] = max_frame
    spec["env"]["num_envs"] = num_envs
    spec["meta"]["max_session"] = 1
    spec["meta"]["log_frequency"] = max_frame + 1  # suppress checkpointing
    spec["meta"]["eval_frequency"] = max_frame + 1

    # Tick to set up directories and indices
    spec_util.tick(spec, "session")
    return spec


def run_profile(algo: str, max_frame: int, num_envs: int):
    """Run profiling for a given algorithm."""
    print(f"\n{'='*70}")
    print(f"Profiling {algo.upper()} on CartPole-v1")
    print(f"  max_frame={max_frame}, num_envs={num_envs}")
    print(f"{'='*70}\n")

    spec = load_spec(algo, max_frame, num_envs)
    session = Session(spec)

    # Profile only the RL loop (skip final analysis which needs checkpoint data)
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        session.run_rl()

    session.close()

    # Print tables sorted by total CPU time
    print(f"\n{'='*70}")
    print(f"[{algo.upper()}] Top 30 ops by cpu_time_total")
    print(f"{'='*70}")
    print(
        prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=30,
        )
    )

    print(f"\n{'='*70}")
    print(f"[{algo.upper()}] Top 30 ops by self_cpu_time_total")
    print(f"{'='*70}")
    print(
        prof.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=30,
        )
    )

    # Group by input shapes to find hot tensor ops
    print(f"\n{'='*70}")
    print(f"[{algo.upper()}] Top 20 ops grouped by input shape")
    print(f"{'='*70}")
    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total",
            row_limit=20,
        )
    )

    # Stack trace view for call hierarchy
    print(f"\n{'='*70}")
    print(f"[{algo.upper()}] Top 30 ops by self_cpu_time_total (with stack)")
    print(f"{'='*70}")
    print(
        prof.key_averages(group_by_stack_n=5).table(
            sort_by="self_cpu_time_total",
            row_limit=30,
        )
    )

    # Save Chrome trace
    trace_path = f"test/profile_trace_{algo}.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace saved to: {trace_path}")
    print(f"  Open chrome://tracing and load this file to visualize.\n")


def main():
    parser = argparse.ArgumentParser(description="Profile SLM-Lab training")
    parser.add_argument(
        "--algo",
        choices=list(ALGO_CONFIGS.keys()),
        default="ppo",
        help="Algorithm to profile (default: ppo)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=5000,
        help="Max frames for profiling (default: 5000)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel envs (default: 4)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Profile all algorithms",
    )
    args = parser.parse_args()

    if args.all:
        for algo in ALGO_CONFIGS:
            run_profile(algo, args.frames, args.num_envs)
    else:
        run_profile(args.algo, args.frames, args.num_envs)


if __name__ == "__main__":
    main()
