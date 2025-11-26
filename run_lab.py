"""
SLM-Lab CLI using typer.
"""

import json
import os
import subprocess
from glob import glob
from pathlib import Path

import torch.multiprocessing as mp
import typer
from huggingface_hub import snapshot_download

from slm_lab import EVAL_MODES, TRAIN_MODES
from slm_lab.experiment.control import Experiment, Session, Trial
from slm_lab.lib import env_var, logger, util
from slm_lab.lib import hf
from slm_lab.spec import spec_util

app = typer.Typer(help="Modular deep reinforcement learning framework")
logger = logger.get_logger(__name__)


def set_variables(spec, sets: list[str] | None):
    """Replace ${var} in spec with set values"""
    if not sets:
        return spec
    spec_str = json.dumps(spec)
    for item in sets:
        key, value = item.split("=", 1)
        spec_str = spec_str.replace(f"${{{key}}}", value)
    return json.loads(spec_str)


def get_spec(
    spec_file: str,
    spec_name: str,
    lab_mode: str,
    pre_: str | None,
    sets: list[str] | None = None,
):
    """Get spec using args processed from inputs"""
    if lab_mode in TRAIN_MODES:
        if pre_ is None:  # new train trial
            spec = spec_util.get(spec_file, spec_name)
        else:
            # for resuming with train@{predir}
            predir = pre_
            if predir == "latest":
                predir = sorted(glob(f"data/{spec_name}*/"))[-1]
            experiment_ts = util.get_experiment_ts(predir)
            logger.info(f"Resolved to train@{predir}")
            spec = spec_util.get(spec_file, spec_name, experiment_ts)
    elif lab_mode == "enjoy":
        # for enjoy@{session_spec_file}
        session_spec_file = pre_
        assert session_spec_file is not None, (
            "enjoy mode must specify a `enjoy@{session_spec_file}`"
        )
        spec = util.read(f"{session_spec_file}")
    else:
        raise ValueError(
            f"Unrecognizable lab_mode not of {TRAIN_MODES} or {EVAL_MODES}"
        )

    # Set variables if provided
    spec = set_variables(spec, sets)
    return spec


def run_spec(
    spec, lab_mode: str, spec_file: str = "", spec_name: str = "", keep_trials: int = 3
):
    """Run a spec in lab_mode"""
    os.environ["lab_mode"] = lab_mode
    spec = spec_util.override_spec(spec, lab_mode)

    if lab_mode in TRAIN_MODES:
        if lab_mode == "search":
            spec_util.tick(spec, "experiment")
            logger.info(
                f"SLM-Lab: Running {spec_file} {spec_name} {lab_mode} | output: {spec['meta']['prepath']}"
            )
            spec_util.save(spec)
            Experiment(spec, keep_trials=keep_trials).run()
        else:
            spec_util.tick(spec, "trial")
            max_session = spec["meta"]["max_session"]
            logger.info(
                f"SLM-Lab: Running {spec_file} {spec_name} {lab_mode} with {max_session} sessions | output: {spec['meta']['prepath']}"
            )
            spec_util.save(spec)
            Trial(spec).run()

        logger.info(f"Output: {spec['meta']['predir']}")
        # Upload after training completion
        hf.upload(spec)
    elif lab_mode in EVAL_MODES:
        Session(spec).run()
    else:
        raise ValueError(
            f"Unrecognizable lab_mode not of {TRAIN_MODES} or {EVAL_MODES}"
        )


def stop_ray_processes():
    """Stop all Ray processes and related Python processes"""
    # First stop Ray cluster with force
    subprocess.run(["uv", "run", "ray", "stop", "--force"])

    # Kill entire process group
    try:
        subprocess.run(["pkill", "-f", "slm-lab"], check=False)
        # Also kill lingering multiprocessing workers
        subprocess.run(["pkill", "-9", "-f", "multiprocessing"], check=False)
        logger.info("Stopped Ray cluster and killed SLM-Lab processes")
    except Exception as e:
        logger.warning(f"Failed to kill processes: {e}")


def run_experiment(
    spec_file: str,
    spec_name: str,
    lab_mode: str,
    sets: list[str] | None = None,
    keep_trials: int = 3,
):
    """Core experiment runner"""
    if "@" in lab_mode:  # process lab_mode@{predir/prename}
        lab_mode, pre_ = lab_mode.split("@")
    else:
        pre_ = None

    spec = get_spec(spec_file, spec_name, lab_mode, pre_, sets)
    run_spec(spec, lab_mode, spec_file, spec_name, keep_trials)


@app.command()
def run(
    spec_file: str = typer.Argument(
        "slm_lab/spec/benchmark/ppo/ppo_cartpole.json",
        help="JSON spec file path (or experiment dir for --upload-dir)",
    ),
    spec_name: str = typer.Argument("ppo_cartpole", help="Spec name within the file"),
    mode: str = typer.Argument(
        "dev",
        help="Execution mode: dev|train|search|enjoy. Note: search_scheduler and max_session>1 are mutually exclusive",
    ),
    # Flags ordered by relevance
    sets: list[str] = typer.Option(
        [],
        "--set",
        "-s",
        help="Set spec variables: KEY=VALUE (can be used multiple times)",
    ),
    render: bool = typer.Option(
        False, "--render", envvar="RENDER", help="Enable environment rendering"
    ),
    job: str | None = typer.Option(
        None, "--job", help="Run batch experiments from job file"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        envvar="LOG_LEVEL",
        help="Logging: DEBUG|INFO|WARNING|ERROR",
    ),
    optimize_perf: bool = typer.Option(
        True,
        "--optimize-perf/--no-optimize-perf",
        envvar="OPTIMIZE_PERF",
        help="Auto-optimize CPU threading and GPU settings (use --no-optimize-perf to disable)",
    ),
    cuda_offset: int = typer.Option(
        0, "--cuda-offset", envvar="CUDA_OFFSET", help="GPU device offset"
    ),
    profile: bool = typer.Option(
        False,
        "--profile",
        envvar="PROFILE",
        help="Enable non-invasive performance profiling",
    ),
    log_extra: bool = typer.Option(
        False,
        "--log-extra",
        envvar="LOG_EXTRA",
        help="Enable extra metrics logging (strength, stability, efficiency)",
    ),
    stop_ray: bool = typer.Option(
        False,
        "--stop-ray",
        help="Stop all Ray processes using Ray CLI",
    ),
    upload_hf: bool = typer.Option(
        False,
        "--upload-hf",
        envvar="UPLOAD_HF",
        help="Upload to HF_REPO after training completes",
    ),
    keep: int = typer.Option(
        3,
        "--keep",
        envvar="KEEP_TRIALS",
        help="Number of top trials to keep after search (default: 3). Use -1 to disable cleanup",
    ),
):
    """
    Run SLM-Lab experiments locally. Defaults to PPO CartPole in dev mode.

    Examples:
        slm-lab run                                                    # PPO CartPole
        slm-lab run --render                                           # With rendering
        slm-lab run spec.json spec_name train                          # Custom experiment
        slm-lab run --set env=ALE/Breakout-v5 spec.json atari dev      # With env override
        slm-lab run --job job/experiments.json                         # Batch experiments
        slm-lab run --stop-ray                                         # Stop Ray processes
    """
    # Handle --stop-ray flag first
    if stop_ray:
        stop_ray_processes()
        return

    # Set environment variables from CLI flags
    mode = env_var.set_from_cli(
        render,
        log_level,
        optimize_perf,
        cuda_offset,
        profile,
        log_extra,
        upload_hf,
        mode,
    )

    if job is not None:
        for spec_file, spec_and_mode in util.read(job).items():
            for spec_name, lab_mode in spec_and_mode.items():
                run_experiment(spec_file, spec_name, lab_mode, sets, keep)
    else:
        run_experiment(spec_file, spec_name, mode, sets, keep)


@app.command("run-remote")
def run_remote(
    spec_file: str = typer.Argument(..., help="JSON spec file path"),
    spec_name: str = typer.Argument(..., help="Spec name within the file"),
    mode: str = typer.Argument("train", help="Execution mode: train|search"),
    name: str = typer.Option(
        None, "--name", "-n", help="Run name (default: spec_name)"
    ),
    sets: list[str] = typer.Option(
        [], "--set", "-s", help="Set spec variables: KEY=VALUE"
    ),
    config: str = typer.Option(
        None, "--config", "-c", help="dstack config: gpu|cpu, auto-selects train/search variant"
    ),
):
    """
    Launch experiment on dstack with auto HF upload.

    Results upload to HF_REPO after training. Use `slm-lab pull` to sync locally.

    Config matrix (auto-selected based on mode):
      - cpu-train:  4-8 CPUs, 16-32GB   (~$0.13/hr) - MLP envs validation
      - cpu-search: 8-16 CPUs, 32-64GB  (~$0.20/hr) - MLP envs ASHA search
      - gpu-train:  L4 GPU              (~$0.39/hr) - Image envs validation
      - gpu-search: L4 GPU + 8-16 CPUs  (~$0.50/hr) - Image envs ASHA search

    Examples:
        slm-lab run-remote spec.json ppo_cartpole train           # GPU train
        slm-lab run-remote spec.json ppo_cartpole search          # GPU search
        slm-lab run-remote spec.json ppo_lunar train -c cpu       # CPU train (cheaper)
        slm-lab run-remote spec.json ppo_lunar search -c cpu      # CPU search (more parallelism)
    """
    run_name = name or spec_name.replace("_", "-")

    # Auto-select config file based on hardware type and mode
    # dev mode uses train config (single trial, lighter resources)
    hw = config or "gpu"
    config_mode = "train" if mode == "dev" else mode
    config_file = f".dstack/run-{hw}-{config_mode}.yml"

    cmd = ["uv", "run", "dstack", "apply", "-f", config_file, "-y", "--name", run_name]
    env = os.environ.copy()
    env["SPEC_FILE"] = spec_file
    env["SPEC_NAME"] = spec_name
    env["LAB_MODE"] = mode
    env["SPEC_VARS"] = " ".join(f"-s {item}" for item in sets) if sets else ""

    logger.info(f"Launching: {run_name} ({config_file})")
    logger.info(f"  {spec_file} / {spec_name} / {mode}")
    logger.info(f"  Pull: slm-lab pull {spec_name}")

    subprocess.run(cmd, env=env)


@app.command()
def pull(
    pattern: str = typer.Argument(..., help="Experiment pattern (e.g., ppo_cartpole)"),
    list_only: bool = typer.Option(
        False, "--list", "-l", help="List matching experiments without downloading"
    ),
):
    """
    Pull experiment results from HuggingFace to local data folder.

    Examples:
        slm-lab pull ppo_cartpole              # Pull matching experiments
        slm-lab pull ppo_cartpole_2025         # Pull specific experiment
        slm-lab pull --list ppo                # List without downloading
    """
    repo = hf.get_repo_id()
    api = hf.get_api()
    pattern = pattern.removeprefix("data/")

    try:
        files = api.list_repo_files(repo_id=repo, repo_type="dataset")

        experiments = set()
        for f in files:
            if f.startswith("data/"):
                parts = f.split("/")
                if len(parts) >= 2:
                    exp_folder = parts[1]
                    if pattern in exp_folder:
                        experiments.add(exp_folder)

        if not experiments:
            logger.warning(f"No experiments matching '{pattern}' found in {repo}")
            return

        experiments = sorted(experiments)

        if list_only:
            logger.info(f"Experiments matching '{pattern}' in {repo}:")
            for exp in experiments:
                logger.info(f"  {exp}")
            return

        logger.info(f"Found {len(experiments)} experiment(s) matching '{pattern}'")

        for exp in experiments:
            local_path = Path("data") / exp
            if local_path.exists():
                logger.info(f"  Skipping {exp} (already exists locally)")
                continue

            logger.info(f"  Downloading {exp}...")
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                local_dir=".",
                allow_patterns=[f"data/{exp}/*"],
                token=os.getenv("HF_TOKEN"),
            )
            logger.info(f"  ✅ Downloaded to {local_path}")

        logger.info("Pull complete!")

    except Exception as e:
        logger.error(f"Pull failed: {e}")
        raise typer.Exit(1)


@app.command()
def push(
    predir: str = typer.Argument(
        ..., help="Experiment directory (e.g., data/ppo_cartpole_2025_11_25_*)"
    ),
):
    """
    Push local experiment to HuggingFace.

    Examples:
        slm-lab push data/ppo_cartpole_2025_11_25_093345
    """
    from huggingface_hub import upload_folder, create_repo

    repo = hf.get_repo_id()
    predir = predir.rstrip("/")

    if not Path(predir).exists():
        logger.error(f"Directory not found: {predir}")
        raise typer.Exit(1)

    try:
        api = hf.get_api()
        api.whoami()
        create_repo(
            repo, repo_type="dataset", exist_ok=True, token=os.getenv("HF_TOKEN")
        )

        logger.info(f"Uploading {predir} to {repo}/{predir}...")
        commit_info = upload_folder(
            folder_path=predir,
            repo_id=repo,
            repo_type="dataset",
            path_in_repo=predir,
            token=os.getenv("HF_TOKEN"),
        )
        logger.info(f"✅ Upload complete: {commit_info.commit_url}")

    except Exception as e:
        logger.error(f"Push failed: {e}")
        raise typer.Exit(1)


@app.command("list")
def list_experiments(
    pattern: str = typer.Argument("", help="Optional pattern to filter experiments"),
):
    """
    List experiments available on HuggingFace.

    Examples:
        slm-lab list                    # List all experiments
        slm-lab list ppo                # List ppo experiments
    """
    repo = hf.get_repo_id()
    api = hf.get_api()

    try:
        files = api.list_repo_files(repo_id=repo, repo_type="dataset")

        experiments = set()
        for f in files:
            if f.startswith("data/"):
                parts = f.split("/")
                if len(parts) >= 2:
                    exp_folder = parts[1]
                    if not pattern or pattern in exp_folder:
                        experiments.add(exp_folder)

        if not experiments:
            logger.info(f"No experiments found in {repo}")
            return

        experiments = sorted(experiments)
        logger.info(f"Experiments in {repo}:")
        for exp in experiments:
            logger.info(f"  {exp}")
        logger.info(f"\nTotal: {len(experiments)} experiment(s)")

    except Exception as e:
        logger.error(f"List failed: {e}")
        raise typer.Exit(1)


def cli():
    """CLI entry point for uv tool installation"""
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    app()


if __name__ == "__main__":
    cli()
