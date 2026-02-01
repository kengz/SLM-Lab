"""SLM-Lab CLI using typer."""

import os
import subprocess
from glob import glob

import typer

from slm_lab.lib import env_var  # Lightweight, only imports os

app = typer.Typer(help="Modular deep reinforcement learning framework")


def _lazy_imports():
    """Lazy import heavy dependencies only when needed for training."""
    global EVAL_MODES, TRAIN_MODES, Experiment, Session, Trial, logger, util, hf, spec_util
    from slm_lab import EVAL_MODES, TRAIN_MODES
    from slm_lab.experiment.control import Experiment, Session, Trial
    from slm_lab.lib import logger as _logger, util, hf
    from slm_lab.spec import spec_util
    logger = _logger.get_logger(__name__)


# Placeholder for lazy-loaded logger
logger = None


def find_saved_spec(predir: str, spec_name: str) -> str | None:
    """Find saved spec file in predir, returns latest trial spec or None."""
    saved_specs = sorted(glob(f"{predir}/{spec_name}_t*_spec.json"))
    return saved_specs[-1] if saved_specs else None


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
            spec = spec_util.get(spec_file, spec_name, sets=sets)
        else:
            # for resuming with train@{predir}
            predir = pre_
            if predir == "latest":
                predir = sorted(glob(f"data/{spec_name}*/"))[-1]
            logger.info(f"Resolved to train@{predir}")
            # Use saved spec to preserve max_session from previous run
            saved_spec_path = find_saved_spec(predir, spec_name)
            if saved_spec_path:
                spec = util.read(saved_spec_path)
            else:
                experiment_ts = util.get_experiment_ts(predir)
                spec = spec_util.get(spec_file, spec_name, experiment_ts, sets=sets)
    elif lab_mode == "enjoy":
        # for enjoy@{spec_file}
        spec_file_path = pre_
        assert spec_file_path is not None, (
            "enjoy mode must specify `enjoy@{spec_file}`"
        )
        spec = util.read(spec_file_path)
    else:
        raise ValueError(
            f"Unrecognizable lab_mode not of {TRAIN_MODES} or {EVAL_MODES}"
        )
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
        hf.upload(spec)  # cleanup happens inside upload()
    elif lab_mode in EVAL_MODES:
        spec_util.tick(spec, "session")
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
        print("Stopped Ray cluster and killed SLM-Lab processes")
    except Exception as e:
        print(f"Warning: Failed to kill processes: {e}")


def run_experiment(
    spec_file: str,
    spec_name: str,
    lab_mode: str,
    sets: list[str] | None = None,
    keep_trials: int = 3,
):
    """Core experiment runner"""
    # Set multiprocessing start method BEFORE importing PyTorch (required for A3C CUDA compatibility)
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    _lazy_imports()  # Load heavy deps only when running experiments
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
        help="Execution mode: dev|train|search|enjoy. Note: search_scheduler auto-overrides max_session to 1",
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

    run_experiment(spec_file, spec_name, mode, sets, keep)


def cli():
    """CLI entry point for uv tool installation"""
    # Only set multiprocessing start method when needed for training
    # This allows lightweight commands (run-remote, pull, etc.) to work without torch
    app()


if __name__ == "__main__":
    cli()
