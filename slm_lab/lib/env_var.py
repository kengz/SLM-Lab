"""Centralized environment variable configuration for CLI flags."""

import os


def set_from_cli(
    render, log_level, optimize_perf, cuda_offset, profile, log_extra, upload_hf, mode
):
    """Set environment variables from CLI flags."""
    # Force dev mode when profiling is enabled
    if profile and mode != "dev":
        mode = "dev"

    os.environ.update(
        {
            "RENDER": str(render).lower(),
            "LOG_LEVEL": log_level,
            "OPTIMIZE_PERF": str(optimize_perf).lower(),
            "CUDA_OFFSET": str(cuda_offset),
            "PROFILE": str(profile).lower(),
            "LOG_EXTRA": str(log_extra).lower(),
            "UPLOAD_HF": str(upload_hf).lower(),
        }
    )

    return mode  # Return potentially modified mode


def render():
    """Check if --render flag is enabled."""
    return os.getenv("RENDER", "false").lower() == "true"


def log_level():
    """Get current log level."""
    return os.getenv("LOG_LEVEL", "INFO")


def optimize_perf():
    """Check if --optimize-perf flag is enabled."""
    return os.getenv("OPTIMIZE_PERF", "true").lower() == "true"


def cuda_offset():
    """Get CUDA device offset."""
    return int(os.getenv("CUDA_OFFSET", "0"))


def profile():
    """Check if --profile flag is enabled."""
    return os.getenv("PROFILE", "false").lower() == "true"


def log_extra():
    """Check if extra metrics logging is enabled."""
    return os.getenv("LOG_EXTRA", "false").lower() == "true"


def upload_hf():
    """Check if Hugging Face upload is enabled."""
    return os.getenv("UPLOAD_HF", "false").lower() == "true"


def lab_mode():
    """Get current lab mode."""
    return os.getenv("lab_mode")
