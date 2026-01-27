"""SLM-Lab CLI module."""

from slm_lab.cli.main import app, cli
from slm_lab.cli import remote, sync

app.command("run-remote")(remote.run_remote)
app.command()(sync.pull)
app.command()(sync.push)
app.command("list")(sync.list_experiments)

# Plot commands require heavy deps (torch, cv2, plotly) - skip in minimal mode
try:
    from slm_lab.cli import plot
    app.command()(plot.plot)
    app.command("plot-list")(plot.list_data)
except ImportError:
    pass

__all__ = ["app", "cli"]
