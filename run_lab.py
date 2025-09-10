"""
SLM-Lab CLI using typer.
"""

import os
import subprocess
from glob import glob
from typing import Optional

import torch.multiprocessing as mp
import typer

from slm_lab import EVAL_MODES, TRAIN_MODES
from slm_lab.experiment.control import Experiment, Session, Trial
from slm_lab.lib import env_var, logger, util
from slm_lab.spec import spec_util

app = typer.Typer(help="Modular deep reinforcement learning framework")
logger = logger.get_logger(__name__)


def get_spec(spec_file: str, spec_name: str, lab_mode: str, pre_: Optional[str]):
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
    return spec


def run_spec(spec, lab_mode: str):
    """Run a spec in lab_mode"""
    os.environ["lab_mode"] = lab_mode
    spec = spec_util.override_spec(spec, lab_mode)

    if lab_mode in TRAIN_MODES:
        spec_util.save(spec)
        if lab_mode == "search":
            spec_util.tick(spec, "experiment")
            Experiment(spec).run()
        else:
            spec_util.tick(spec, "trial")
            Trial(spec).run()
    elif lab_mode in EVAL_MODES:
        Session(spec).run()
    else:
        raise ValueError(
            f"Unrecognizable lab_mode not of {TRAIN_MODES} or {EVAL_MODES}"
        )


def kill_ray_processes():
    """Kill all Ray processes (workaround for Ray signal handling bug)"""
    subprocess.run(["pkill", "-f", "ray"])
    logger.info("Killed Ray processes")


def run_experiment(spec_file: str, spec_name: str, lab_mode: str):
    """Core experiment runner"""
    if "@" in lab_mode:  # process lab_mode@{predir/prename}
        lab_mode, pre_ = lab_mode.split("@")
    else:
        pre_ = None

    spec = get_spec(spec_file, spec_name, lab_mode, pre_)
    run_spec(spec, lab_mode)


def main(
    spec_file: str = typer.Argument(
        "slm_lab/spec/demo.json", help="JSON spec file path"
    ),
    spec_name: str = typer.Argument("dqn_cartpole", help="Spec name within the file"),
    mode: str = typer.Argument("dev", help="Execution mode: dev|train|search|enjoy"),
    # Flags ordered by relevance
    render: bool = typer.Option(
        False, "--render", envvar="RENDER", help="Enable environment rendering"
    ),
    job: Optional[str] = typer.Option(
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
        help="Auto-optimize CPU threading, torch.compile, and GPU settings (use --no-optimize-perf to disable)",
    ),
    torch_compile: str = typer.Option(
        "auto",
        "--torch-compile",
        envvar="TORCH_COMPILE",
        help="auto|true|false: torch.compile smart detection (auto=Ampere+ only, true=force, may fail Apple Silicon)",
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
    kill_ray: bool = typer.Option(
        False,
        "--kill-ray",
        help="Kill all Ray processes (workaround for Ray signal handling bug)",
    ),
):
    """
    Run SLM-Lab experiments. Defaults to CartPole demo in dev mode.

    Examples:
        slm-lab                                    # CartPole demo (no rendering)
        slm-lab --render                           # CartPole demo with rendering
        slm-lab spec.json spec_name dev            # Custom experiment
        slm-lab spec.json spec_name train          # Custom experiment
        slm-lab --job job/experiments.json         # Batch experiments
        slm-lab --kill-ray                         # Kill all Ray processes (force stop search)
    """
    # Handle --kill-ray flag first
    if kill_ray:
        kill_ray_processes()
        return

    # Set environment variables from CLI flags
    mode = env_var.set_from_cli(
        render, log_level, optimize_perf, torch_compile, cuda_offset, profile, mode
    )

    if job is not None:
        for spec_file, spec_and_mode in util.read(job).items():
            for spec_name, lab_mode in spec_and_mode.items():
                run_experiment(spec_file, spec_name, lab_mode)
    else:
        run_experiment(spec_file, spec_name, mode)


def cli():
    """CLI entry point for uv tool installation"""
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    typer.run(main)


if __name__ == "__main__":
    cli()
