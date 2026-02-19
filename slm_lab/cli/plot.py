"""Benchmark plotting commands for SLM-Lab CLI."""

import json
from pathlib import Path

import typer

from slm_lab.lib import logger, util, viz

logger = logger.get_logger(__name__)

# File patterns for trial metrics
TRIAL_METRICS_PATH = '*t0_trial_metrics.json'
SPEC_PATH = '*spec.json'

# Algorithm order for legend (fixed ordering)
ALGO_ORDER = ['REINFORCE', 'SARSA', 'DQN', 'DDQN+PER', 'A2C', 'PPO', 'SAC']
# Colors by algorithm lineage:
# - REINFORCE/SARSA: yellow/brown (classic methods)
# - DQN/DDQN+PER: teal/green tones (value-based)
# - A2C/PPO/SAC: blue/purple/red tones (actor-critic family)
ALGO_PALETTE = {
    'REINFORCE': 'hsl(45, 80%, 55%)',   # golden yellow
    'SARSA': 'hsl(30, 60%, 45%)',        # brown
    'DQN': 'hsl(175, 55%, 45%)',         # teal
    'DDQN+PER': 'hsl(145, 50%, 45%)',    # green
    'A2C': 'hsl(220, 65%, 55%)',         # blue
    'PPO': 'hsl(280, 55%, 55%)',         # purple
    'SAC': 'hsl(350, 65%, 55%)',         # red
}


def get_spec_data(folder_path: Path) -> dict | None:
    """Read spec file from experiment folder."""
    # Find spec file (usually named after the spec, e.g. ppo_cartpole_t0_spec.json)
    # or just spec.json in the folder root or info/ folder
    matches = list(folder_path.glob(SPEC_PATH))
    if not matches:
        matches = list((folder_path / 'info').glob(SPEC_PATH))

    if matches:
        return util.read(str(matches[0]))
    return None


def get_algo_name_from_spec(spec: dict) -> str:
    """Extract algorithm name from spec."""
    # Spec structure: {spec_name: {agent: ...}} OR {agent: ...} (resolved spec)
    try:
        if 'agent' in spec:
            agent_spec = spec['agent']
        else:
            # Get the first key (spec_name)
            spec_name = list(spec.keys())[0]
            agent_spec = spec[spec_name]['agent']

        # Handle list of agents (multi-agent) or single agent dict
        if isinstance(agent_spec, list):
            algo_name = agent_spec[0]['algorithm']['name']
        else:
            algo_name = agent_spec['algorithm']['name']

        # Standardize names
        name_map = {
            'VanillaDQN': 'DQN',
            'DoubleDQN': 'DDQN+PER', # Usually used with PER in benchmarks
            'PPO': 'PPO',
            'SAC': 'SAC',
            'A2C': 'A2C',
            'ActorCritic': 'A2C',
            'SoftActorCritic': 'SAC',
            'DQN': 'DQN',
            'Reinforce': 'REINFORCE',
            'REINFORCE': 'REINFORCE',
        }

        # specific check for DDQN/PER
        if algo_name == 'DoubleDQN':
             memory_name = ''
             if isinstance(agent_spec, list):
                 memory_name = str(agent_spec[0].get('memory', {}).get('name', ''))
             else:
                 memory_name = str(agent_spec.get('memory', {}).get('name', ''))

             if 'Prioritized' in memory_name:
                 return 'DDQN+PER'
             return 'DDQN'

        return name_map.get(algo_name, algo_name)
    except Exception as e:
        logger.warning(f"Could not extract algo name from spec: {e}")
        return "Unknown"


def get_env_name_from_spec(spec: dict) -> str:
    """Extract environment name from spec."""
    try:
        if 'env' in spec:
            return spec['env']['name']
        spec_name = list(spec.keys())[0]
        return spec[spec_name]['env']['name']
    except Exception:
        return None


def find_trial_metrics(folder_path: Path) -> str | None:
    """Find trial metrics file in a folder."""
    matches = list((folder_path / 'info').glob(TRIAL_METRICS_PATH))
    if matches:
        return str(matches[0])
    return None


def plot(
    folders: str = typer.Option(..., "--folders", "-f", help="Comma-separated data folder names (e.g., ppo_cartpole_2026_01_11,a2c_gae_cartpole_2026_01_11)"),
    title: str = typer.Option(None, "--title", "-t", help="Plot title. If omitted, extracted from spec env name."),
    data_folder: str = typer.Option("data", "--data-folder", "-d", help="Base data folder path"),
    output_folder: str = typer.Option("docs/plots", "--output", "-o", help="Output folder for plots"),
    showlegend: bool = typer.Option(True, "--legend/--no-legend", help="Show legend on plot"),
):
    """
    Plot benchmark comparison graphs from explicit folder paths.

    Specify exact experiment folders to compare. Title and legends are auto-detected from specs unless overridden.

    Examples:
        slm-lab plot -f ppo_cartpole_2026_01_11_100728,a2c_gae_cartpole_2026_01_11_100724

        slm-lab plot -t "Custom Title" -f folder1,folder2
    """
    data_path = Path(util.smart_path(data_folder))
    output_path = Path(util.smart_path(output_folder))

    folder_list = [f.strip() for f in folders.split(',')]

    trial_metrics_paths = []
    legends = []
    detected_title = None

    for folder_name in folder_list:
        folder_path = data_path / folder_name
        if not folder_path.exists():
            logger.error(f"Folder not found: {folder_path}")
            raise typer.Exit(1)

        metrics_path = find_trial_metrics(folder_path)
        if not metrics_path:
            logger.error(f"No trial_metrics found in: {folder_path}/info/")
            raise typer.Exit(1)

        trial_metrics_paths.append(metrics_path)

        # Get metadata from spec
        spec = get_spec_data(folder_path)
        if spec:
            algo_name = get_algo_name_from_spec(spec)
            env_name = get_env_name_from_spec(spec)
            if not detected_title and env_name:
                detected_title = env_name
        else:
            algo_name = folder_name.split('_')[0].upper() # Fallback

        legends.append(algo_name)
        logger.info(f"  {algo_name}: {metrics_path}")

    if len(trial_metrics_paths) < 1:
        logger.error("Need at least 1 folder to plot")
        raise typer.Exit(1)

    # Sort tracks by ALGO_ORDER to ensure consistent ordering and coloring
    combined = []
    for path, legend in zip(trial_metrics_paths, legends):
        try:
            order = ALGO_ORDER.index(legend)
        except ValueError:
            order = 999  # Put unknown at end
        combined.append((order, legend, path))

    combined.sort()

    legends = [x[1] for x in combined]
    trial_metrics_paths = [x[2] for x in combined]

    # Use detected title if not provided
    final_title = title if title else (detected_title if detected_title else "Benchmark")

    # Build palette (consistent colors for same algorithms)
    palette = [ALGO_PALETTE.get(legend, None) for legend in legends]
    # If any palette entry is None (unknown algo), fill with default palette
    default_palette = viz.get_palette(len(legends))
    for i, color in enumerate(palette):
        if color is None:
            palette[i] = default_palette[i % len(default_palette)]

    # Generate output filename from title (strips ALE/ prefix for Atari env names)
    filename_title = Path(final_title).name  # ALE/Pong-v5 → Pong-v5
    safe_title = filename_title.replace(' ', '_').replace('(', '').replace(')', '')
    graph_prepath = str(output_path / safe_title)

    viz.plot_multi_trial(
        trial_metrics_paths,
        legends,
        final_title,
        graph_prepath,
        ma=True,
        name_time_pairs=[('mean_returns', 'frames')],
        palette=palette,
        showlegend=showlegend,
    )

    output_file = f'{graph_prepath}_multi_trial_graph_mean_returns_ma_vs_frames.png'
    logger.info(f"Saved: {output_file}")


def list_data(
    data_folder: str = typer.Option("data", "--data-folder", "-d", help="Data folder path"),
):
    """
    List available experiments in data folder.

    Shows which experiments have trial_metrics files ready for plotting.
    """
    data_path = Path(util.smart_path(data_folder))
    if not data_path.exists():
        logger.error(f"Data folder not found: {data_folder}")
        raise typer.Exit(1)

    experiments = sorted([d.name for d in data_path.iterdir()
                         if d.is_dir() and not d.name.startswith('.')])

    if not experiments:
        logger.info(f"No experiments found in {data_folder}")
        return

    logger.info(f"\nExperiments in {data_folder}:")
    for exp in experiments:
        metrics_files = list((data_path / exp / 'info').glob('*trial_metrics.json'))
        status = "✓" if metrics_files else "○"
        logger.info(f"  {status} {exp}")

    logger.info(f"\n✓ = has trial_metrics (ready for plotting)")
    logger.info(f"○ = no trial_metrics")
