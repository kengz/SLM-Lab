"""
SLM-Lab CLI using typer.
"""

import json
import os
import subprocess
from glob import glob

import torch.multiprocessing as mp
import typer

from slm_lab import EVAL_MODES, TRAIN_MODES
from slm_lab.experiment.control import Experiment, Session, Trial
from slm_lab.lib import env_var, logger, util
from slm_lab.lib.hf import upload
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


def get_spec(spec_file: str, spec_name: str, lab_mode: str, pre_: str | None, sets: list[str] | None = None):
    """Get spec using args processed from inputs"""
    if lab_mode in TRAIN_MODES:
        if pre_ is None:  # new train trial
            spec = spec_util.get(spec_file, spec_name)
        else:
            # for resuming with train@{predir}
            predir = pre_
            if predir == "latest":
                predir = sorted(glob(f"data/{spec_name}*/"))[-1]
            _, _, _, _, experiment_ts = util.prepath_split(predir)
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


def run_spec(spec, lab_mode: str, spec_file: str = "", spec_name: str = ""):
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
            Experiment(spec).run()
        else:
            spec_util.tick(spec, "trial")
            max_session = spec["meta"]["max_session"]
            logger.info(
                f"SLM-Lab: Running {spec_file} {spec_name} {lab_mode} with {max_session} sessions | output: {spec['meta']['prepath']}"
            )
            spec_util.save(spec)
            Trial(spec).run()
        
        # Upload after training completion
        upload(spec)
    elif lab_mode in EVAL_MODES:
        Session(spec).run()
    else:
        raise ValueError(
            f"Unrecognizable lab_mode not of {TRAIN_MODES} or {EVAL_MODES}"
        )


def stop_ray_processes():
    """Stop all Ray processes and related Python processes"""
    # First stop Ray cluster
    subprocess.run(["uv", "run", "ray", "stop"])
    
    # Kill entire process group 
    try:
        subprocess.run(["pkill", "-f", "slm-lab"], check=False)
        logger.info("Stopped Ray cluster and killed SLM-Lab processes")
    except Exception as e:
        logger.warning(f"Failed to kill processes: {e}")


def run_experiment(spec_file: str, spec_name: str, lab_mode: str, sets: list[str] | None = None):
    """Core experiment runner"""
    if "@" in lab_mode:  # process lab_mode@{predir/prename}
        lab_mode, pre_ = lab_mode.split("@")
    else:
        pre_ = None

    spec = get_spec(spec_file, spec_name, lab_mode, pre_, sets)
    run_spec(spec, lab_mode, spec_file, spec_name)


def main(
    spec_file: str = typer.Argument(
        "slm_lab/spec/demo.json", help="JSON spec file path (or experiment dir for --upload-dir)"
    ),
    spec_name: str = typer.Argument("dqn_cartpole", help="Spec name within the file"),
    mode: str = typer.Argument("dev", help="Execution mode: dev|train|search|enjoy"),
    # Flags ordered by relevance
    sets: list[str] = typer.Option(
        [], "--set", "-s", help="Set spec variables: KEY=VALUE (can be used multiple times)"
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
    dstack: str | None = typer.Option(
        None,
        "--dstack",
        help="Run on dstack with given name (e.g., --dstack ppo-pong-16)",
    ),
    upload_hf: str | None = typer.Option(
        None,
        "--upload-hf",
        envvar="UPLOAD_HF",
        help="Upload to HF: auto after training (--upload-hf auto) or specify path for existing experiment",
    ),
):
    """
    Run SLM-Lab experiments. Defaults to CartPole demo in dev mode.

    Examples:
        slm-lab                                                        # CartPole demo (no rendering)
        slm-lab --render                                               # CartPole demo with rendering
        slm-lab spec.json spec_name dev                                # Custom experiment
        slm-lab spec.json spec_name train                              # Custom experiment
        slm-lab --set env=ALE/Breakout-v5 spec.json atari_spec dev     # Run with specific environment
        slm-lab -s env=HalfCheetah-v4 -s lr=0.001 spec.json spec_name dev  # Multiple overrides
        slm-lab slm_lab/spec/benchmark/ppo/ppo_pong.json ppo_pong train --dstack ppo-pong-16  # Run on dstack
        slm-lab --job job/experiments.json                             # Batch experiments
        slm-lab --stop-ray                                             # Stop all Ray processes (force stop search)
        slm-lab --upload-hf auto                                       # Auto-upload after training
        slm-lab --upload-hf data/dqn_cartpole_2025_09_21_173017        # Upload existing experiment to HF
    """
    # Handle --stop-ray flag first
    if stop_ray:
        stop_ray_processes()
        return
    
    # Handle --upload-hf for retroactive uploads (when it's a path)  
    if upload_hf and upload_hf != "auto":
        upload(upload_hf)
        return

    # Handle --dstack flag to run on cloud
    if dstack:
        cmd = ["uv", "run", "dstack", "apply", "-f", ".dstack/run.yml", "-y", "--name", dstack]
        env = os.environ.copy()
        env["SPEC_FILE"] = spec_file
        env["SPEC_NAME"] = spec_name
        env["LAB_MODE"] = mode
        env["SET_ARGS"] = " ".join(f"-s {item}" for item in sets) if sets else ""
        
        logger.info(f"Running on dstack: {' '.join(cmd)}")
        subprocess.run(cmd, env=env)
        return

    # Set environment variables from CLI flags
    mode = env_var.set_from_cli(
        render, log_level, optimize_perf, cuda_offset, profile, log_extra, upload_hf, mode
    )
    
    if job is not None:
        for spec_file, spec_and_mode in util.read(job).items():
            for spec_name, lab_mode in spec_and_mode.items():
                run_experiment(spec_file, spec_name, lab_mode, sets)
    else:
        run_experiment(spec_file, spec_name, mode, sets)


def cli():
    """CLI entry point for uv tool installation"""
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    typer.run(main)


if __name__ == "__main__":
    cli()
