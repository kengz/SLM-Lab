"""HuggingFace sync commands (pull, push, list)."""

import os
from pathlib import Path

import typer
from huggingface_hub import snapshot_download

from slm_lab.lib import hf, logger

logger = logger.get_logger(__name__)


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
            logger.info(f"  Downloaded to {local_path}")

        logger.info("Pull complete!")

    except Exception as e:
        logger.error(f"Pull failed: {e}")
        raise typer.Exit(1)


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
        logger.info(f"Upload complete: {commit_info.commit_url}")

    except Exception as e:
        logger.error(f"Push failed: {e}")
        raise typer.Exit(1)


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
