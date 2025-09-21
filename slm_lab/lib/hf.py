"""Hugging Face dataset upload for SLM-Lab experiment data."""

from pathlib import Path

import os

import typer
from huggingface_hub import HfApi, create_repo, upload_folder

from slm_lab.lib import env_var, logger, util

logger = logger.get_logger(__name__)

def get_repo_id():
    """Get HF dataset repo ID from env var or default."""
    return os.getenv("HF_DATASET_REPO", "SLM-Lab/benchmark")


def upload(spec_or_predir: dict | str):
    """Upload experiment to shared SLM-Lab benchmark dataset."""
    # Check if upload is enabled (only for auto mode during training)
    if isinstance(spec_or_predir, dict) and not env_var.upload_hf():
        return

    # Extract predir and experiment info
    prepath = (
        spec_or_predir["meta"]["prepath"]
        if isinstance(spec_or_predir, dict)
        else spec_or_predir
    )
    predir, _, _, spec_name, experiment_ts = util.prepath_split(prepath)

    if not Path(predir).exists():
        logger.error(f"Directory not found: {predir}")
        return

    repo_id = get_repo_id()
    
    try:
        HfApi().whoami()
        create_repo(repo_id, repo_type="dataset", exist_ok=True)

        folder_name = f"data/{spec_name}_{experiment_ts}"
        logger.info(f"Uploading {predir} to {repo_id}/{folder_name}...")
        
        commit_info = upload_folder(
            folder_path=predir, 
            repo_id=repo_id, 
            repo_type="dataset", 
            path_in_repo=folder_name
        )
        
        logger.info(f"✅ Upload complete: {commit_info.pr_url or commit_info.commit_url}")

    except Exception as e:
        logger.error(f"HF upload failed: {e}")


def retro_upload(
    predir: str = typer.Argument(
        ..., help="Experiment directory (e.g., data/ppo_cartpole_2025_09_12_102256)"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Upload existing SLM-Lab experiment to benchmark dataset."""
    if not Path(predir).exists():
        print(f"Directory not found: {predir}")
        raise typer.Exit(1)

    if not yes:
        size_mb = sum(f.stat().st_size for f in Path(predir).rglob("*") if f.is_file()) / 1024**2
        if not typer.confirm(f"Upload {predir} ({size_mb:.1f} MB) to {get_repo_id()}?"):
            return

    upload(predir)


if __name__ == "__main__":
    typer.run(retro_upload)
