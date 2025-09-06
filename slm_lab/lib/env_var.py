"""Centralized environment variable configuration for CLI flags."""

import os

from slm_lab.lib import logger


def set_from_cli(
    render, log_level, optimize_perf, torch_compile, cuda_offset, profile, mode
):
    """Set environment variables from CLI flags."""
    # Force dev mode when profiling is enabled
    if profile and mode != "dev":
        logger.info(f"Profiling enabled: forcing dev mode (was: {mode})")
        mode = "dev"

    os.environ.update(
        {
            "RENDER": str(render).lower(),
            "LOG_LEVEL": log_level,
            "OPTIMIZE_PERF": str(optimize_perf).lower(),
            "TORCH_COMPILE": torch_compile,
            "CUDA_OFFSET": str(cuda_offset),
            "PROFILE": str(profile).lower(),
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


def torch_compile():
    """Get --torch-compile flag value."""
    return os.getenv("TORCH_COMPILE", "auto").lower()


def cuda_offset():
    """Get CUDA device offset."""
    return int(os.getenv("CUDA_OFFSET", "0"))


def profile():
    """Check if --profile flag is enabled."""
    return os.getenv("PROFILE", "false").lower() == "true"


def lab_mode():
    """Get current lab mode."""
    return os.getenv("lab_mode")
