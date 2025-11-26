"""SLM-Lab CLI module."""

from slm_lab.cli.main import app, cli
from slm_lab.cli import remote, sync

# Register commands from submodules
app.command("run-remote")(remote.run_remote)
app.command()(sync.pull)
app.command()(sync.push)
app.command("list")(sync.list_experiments)

__all__ = ["app", "cli"]
