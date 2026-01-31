"""SLM-Lab CLI module."""

from slm_lab.cli.main import app, cli
from slm_lab.cli import plot, remote, sync

app.command("run-remote")(remote.run_remote)
app.command()(sync.pull)
app.command()(sync.push)
app.command("list")(sync.list_experiments)
app.command()(plot.plot)
app.command("plot-list")(plot.list_data)

__all__ = ["app", "cli"]
