"""Hugging Face dataset upload for SLM-Lab experiment data."""

import os
from pathlib import Path

import typer
from huggingface_hub import HfApi, create_repo, upload_folder

from slm_lab.lib import env_var, logger

logger = logger.get_logger(__name__)


def cleanup_models(predir: str) -> float:
    """Remove optim files not needed for enjoy mode before upload. Returns MB saved."""
    model_dir = Path(predir) / "model"
    if not model_dir.exists():
        return 0.0

    bytes_saved = 0
    for f in model_dir.glob("*_optim.pt"):
        bytes_saved += f.stat().st_size
        f.unlink()

    mb_saved = bytes_saved / (1024 * 1024)
    if mb_saved > 0:
        logger.info(f"Cleanup saved {mb_saved:.1f} MB")
    return mb_saved


def get_repo_id():
    """Get HF dataset repo ID from HF_REPO env var."""
    return os.getenv("HF_REPO", "SLM-Lab/benchmark-dev")


def get_api():
    """Get authenticated HfApi instance."""
    return HfApi(token=os.getenv("HF_TOKEN"))


def upload(spec_or_predir: dict | str):
    """Upload experiment to shared SLM-Lab benchmark dataset."""
    # Check if upload is enabled (only for auto/public mode during training)
    if isinstance(spec_or_predir, dict) and not env_var.upload_hf():
        return

    # Extract predir directly from spec or use the passed predir string
    if isinstance(spec_or_predir, dict):
        predir = spec_or_predir["meta"]["predir"]
    else:
        predir = spec_or_predir

    if not Path(predir).exists():
        logger.error(f"Directory not found: {predir}")
        return

    repo_id = get_repo_id()
    cleanup_models(predir)

    try:
        HfApi().whoami()
        create_repo(repo_id, repo_type="dataset", exist_ok=True)

        logger.info(f"Uploading {predir} to {repo_id}/{predir}...")
        commit_info = upload_folder(
            folder_path=predir,
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=predir,
        )
        logger.info(
            f"✅ Upload complete: {commit_info.pr_url or commit_info.commit_url}"
        )

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
        size_mb = (
            sum(f.stat().st_size for f in Path(predir).rglob("*") if f.is_file())
            / 1024**2
        )
        if not typer.confirm(f"Upload {predir} ({size_mb:.1f} MB) to {get_repo_id()}?"):
            return

    upload(predir)


if __name__ == "__main__":
    typer.run(retro_upload)
