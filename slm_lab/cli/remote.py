"""Remote execution commands (dstack)."""

import os
import subprocess

import typer

from slm_lab.lib import logger

logger = logger.get_logger(__name__)


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
