# [SLM Lab](https://www.amazon.com/dp/0135172381) <br> ![GitHub tag (latest SemVer)](https://img.shields.io/github/tag/kengz/slm-lab) ![CI](https://github.com/kengz/SLM-Lab/workflows/CI/badge.svg)

<p align="center">
  <i>Modular Deep Reinforcement Learning framework in PyTorch.</i>
  <br>
  <i>Companion library of the book <a href="https://www.amazon.com/dp/0135172381">Foundations of Deep Reinforcement Learning</a>.</i>
  <br>
  <a href="https://slm-lab.gitbook.io/slm-lab/">Documentation</a> · <a href="https://github.com/kengz/SLM-Lab/blob/master/docs/BENCHMARKS.md">Benchmark Results</a>
</p>

>**NOTE:** v5.0 updates to Gymnasium, `uv` tooling, and modern dependencies with ARM support - see [CHANGELOG.md](docs/CHANGELOG.md).
>
>Book readers: `git checkout v4.1.1` for *Foundations of Deep Reinforcement Learning* code.

|||||
|:---:|:---:|:---:|:---:|
| ![ppo beamrider](https://user-images.githubusercontent.com/8209263/63994698-689ecf00-caaa-11e9-991f-0a5e9c2f5804.gif) | ![ppo breakout](https://user-images.githubusercontent.com/8209263/63994695-650b4800-caaa-11e9-9982-2462738caa45.gif) | ![ppo kungfumaster](https://user-images.githubusercontent.com/8209263/63994690-60469400-caaa-11e9-9093-b1cd38cee5ae.gif) | ![ppo mspacman](https://user-images.githubusercontent.com/8209263/63994685-5cb30d00-caaa-11e9-8f35-78e29a7d60f5.gif) |
| BeamRider | Breakout | KungFuMaster | MsPacman |
| ![ppo pong](https://user-images.githubusercontent.com/8209263/63994680-59b81c80-caaa-11e9-9253-ed98370351cd.gif) | ![ppo qbert](https://user-images.githubusercontent.com/8209263/63994672-54f36880-caaa-11e9-9757-7780725b53af.gif) | ![ppo seaquest](https://user-images.githubusercontent.com/8209263/63994665-4dcc5a80-caaa-11e9-80bf-c21db818115b.gif) | ![ppo spaceinvaders](https://user-images.githubusercontent.com/8209263/63994624-15c51780-caaa-11e9-9c9a-854d3ce9066d.gif) |
| Pong | Qbert | Seaquest | Sp.Invaders |
| ![sac ant](https://user-images.githubusercontent.com/8209263/63994867-ff6b8b80-caaa-11e9-971e-2fac1cddcbac.gif) | ![sac halfcheetah](https://user-images.githubusercontent.com/8209263/63994869-01354f00-caab-11e9-8e11-3893d2c2419d.gif) | ![sac hopper](https://user-images.githubusercontent.com/8209263/63994871-0397a900-caab-11e9-9566-4ca23c54b2d4.gif) | ![sac humanoid](https://user-images.githubusercontent.com/8209263/63994883-0befe400-caab-11e9-9bcc-c30c885aad73.gif) |
| Ant | HalfCheetah | Hopper | Humanoid |
| ![sac doublependulum](https://user-images.githubusercontent.com/8209263/63994879-07c3c680-caab-11e9-974c-06cdd25bfd68.gif) | ![sac pendulum](https://user-images.githubusercontent.com/8209263/63994880-085c5d00-caab-11e9-850d-049401540e3b.gif) | ![sac reacher](https://user-images.githubusercontent.com/8209263/63994881-098d8a00-caab-11e9-8e19-a3b32d601b10.gif) | ![sac walker](https://user-images.githubusercontent.com/8209263/63994882-0abeb700-caab-11e9-9e19-b59dc5c43393.gif) |
| Inv.DoublePendulum | InvertedPendulum | Reacher | Walker |

SLM Lab is a software framework for **reinforcement learning** (RL) research and application in PyTorch. RL trains agents to make decisions by learning from trial and error—like teaching a robot to walk or an AI to play games.

## What SLM Lab Offers

| Feature | Description |
|---------|-------------|
| **Ready-to-use algorithms** | PPO, SAC, DQN, A2C, REINFORCE—validated on 70+ environments |
| **Easy configuration** | JSON spec files fully define experiments—no code changes needed |
| **Reproducibility** | Every run saves its spec + git SHA for exact reproduction |
| **Automatic analysis** | Training curves, metrics, and TensorBoard logging out of the box |
| **Cloud integration** | dstack for GPU training, HuggingFace for sharing results |

## Algorithms

| Algorithm | Type | Best For | Validated Environments |
|-----------|------|----------|------------------------|
| **REINFORCE** | On-policy | Learning/teaching | Classic |
| **SARSA** | On-policy | Tabular-like | Classic |
| **DQN/DDQN+PER** | Off-policy | Discrete actions | Classic, Box2D, Atari |
| **A2C** | On-policy | Fast iteration | Classic, Box2D, Atari |
| **PPO** | On-policy | General purpose | Classic, Box2D, MuJoCo (11), Atari (54) |
| **SAC** | Off-policy | Continuous control | Classic, Box2D, MuJoCo |

See [Benchmark Results](docs/BENCHMARKS.md) for detailed performance data.

## Environments

SLM Lab uses [Gymnasium](https://gymnasium.farama.org/) (the maintained fork of OpenAI Gym):

| Category | Examples | Difficulty | Docs |
|----------|----------|------------|------|
| **Classic Control** | CartPole, Pendulum, Acrobot | Easy | [Gymnasium Classic](https://gymnasium.farama.org/environments/classic_control/) |
| **Box2D** | LunarLander, BipedalWalker | Medium | [Gymnasium Box2D](https://gymnasium.farama.org/environments/box2d/) |
| **MuJoCo** | Hopper, HalfCheetah, Humanoid | Hard | [Gymnasium MuJoCo](https://gymnasium.farama.org/environments/mujoco/) |
| **Atari** | Breakout, MsPacman, and 54 more | Varied | [ALE](https://ale.farama.org/environments/) |

Any gymnasium-compatible environment works—just specify its name in the spec.

## Quick Start

```bash
# Install
uv sync
uv tool install --editable .

# Run demo (PPO CartPole)
slm-lab run                                    # PPO CartPole
slm-lab run --render                           # with visualization

# Run custom experiment
slm-lab run spec.json spec_name train          # local training
slm-lab run-remote spec.json spec_name train   # cloud training (dstack)

# Help (CLI uses Typer)
slm-lab --help                                 # list all commands
slm-lab run --help                             # options for run command

# Troubleshoot: if slm-lab not found, use uv run
uv run slm-lab run
```

## Cloud Training (dstack)

Run experiments on cloud GPUs with automatic result sync to HuggingFace.

```bash
# Setup
cp .env.example .env  # Add HF_TOKEN
uv tool install dstack  # Install dstack CLI
# Configure dstack server - see https://dstack.ai/docs/quickstart

# Run on cloud
slm-lab run-remote spec.json spec_name train           # CPU training (default)
slm-lab run-remote spec.json spec_name search          # CPU ASHA search (default)
slm-lab run-remote --gpu spec.json spec_name train     # GPU training (for image envs)

# Sync results
slm-lab pull spec_name    # Download from HuggingFace
slm-lab list              # List available experiments
```

Config options in `.dstack/`: `run-gpu-train.yml`, `run-gpu-search.yml`, `run-cpu-train.yml`, `run-cpu-search.yml`

### Minimal Install (Orchestration Only)

For a lightweight box that only dispatches dstack runs, syncs results, and generates plots (no local ML training):

```bash
uv sync --no-default-groups  # skip ML deps (torch, gymnasium, etc.)
uv tool install dstack
uv run --no-default-groups slm-lab run-remote spec.json spec_name train
uv run --no-default-groups slm-lab pull spec_name
uv run --no-default-groups slm-lab plot -f folder1,folder2
```

## Citation

If you use SLM Lab in your research, please cite:

```bibtex
@misc{kenggraesser2017slmlab,
    author = {Keng, Wah Loon and Graesser, Laura},
    title = {SLM Lab},
    year = {2017},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/kengz/SLM-Lab}},
}
```

## License

MIT
